//! Check batch ordering and concurrency, e.g. different states retain their own usage.
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use bdecide::{DecisionModel, Error, Metadata, Result, SystemOneRequest, SystemOneResponse};
use burn_std::stream::StreamId;
use indexmap::IndexMap;
use serde_json::json;

struct Probe {
    metadata: Metadata,
    entered: (Mutex<usize>, Condvar),
    streams: Mutex<Vec<StreamId>>,
}

impl DecisionModel for Probe {
    #[expect(
        clippy::panic,
        clippy::panic_in_result_fn,
        reason = "simulate a backend panic to verify batch recovery"
    )]
    fn system_one(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        if request.state == "panic" {
            panic!("simulated backend failure");
        }
        if request.state == "error" {
            return Err(Error::InvalidRequest("simulated invalid request".into()));
        }
        self.streams
            .lock()
            .map_err(|error| Error::Inference(error.to_string()))?
            .push(StreamId::current());
        let (lock, changed) = &self.entered;
        let mut entered = lock
            .lock()
            .map_err(|error| Error::Inference(error.to_string()))?;
        *entered += 1;
        changed.notify_all();
        // A sequential implementation times out instead of hanging this test forever.
        let (_entered, timeout) = changed
            .wait_timeout_while(entered, Duration::from_secs(5), |count| *count < 2)
            .map_err(|error| Error::Inference(error.to_string()))?;
        if timeout.timed_out() {
            return Err(Error::Inference("requests did not overlap".into()));
        }
        Ok(SystemOneResponse::new(
            request
                .state
                .as_str()
                .ok_or_else(|| Error::InvalidRequest("probe needs text state".into()))?,
            IndexMap::new(),
            self.metadata.clone(),
        ))
    }

    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
}

fn probe() -> Probe {
    Probe {
        metadata: Metadata::new("probe", "probe", "cpu"),
        entered: (Mutex::new(0), Condvar::new()),
        streams: Mutex::new(Vec::new()),
    }
}

#[test]
fn default_batch_overlaps_requests_on_distinct_burn_streams() {
    let model = probe();
    let requests =
        ["first", "second"].map(|state| SystemOneRequest::new(json!(state), IndexMap::new()));
    let responses = model.system_one_batch(&requests).expect("parallel batch");
    assert_eq!(
        responses
            .iter()
            .map(|response| response.model.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    let streams = model.streams.lock().expect("stream log");
    assert_eq!(streams.len(), 2);
    assert_ne!(streams.first(), streams.get(1));
    assert!(model.system_one_batch(&[]).expect("empty batch").is_empty());
}

#[test]
fn default_batch_returns_request_errors_and_contains_worker_panics() {
    for state in ["error", "panic"] {
        let request = SystemOneRequest::new(json!(state), IndexMap::new());
        let requests = [request.clone(), request];
        let error = probe()
            .system_one_batch(&requests)
            .expect_err("failed batch");
        match state {
            "error" => assert!(matches!(error, Error::InvalidRequest(_))),
            _ => assert!(matches!(error, Error::Inference(_))),
        }
    }
}

#[cfg(any(feature = "cpu", feature = "wgpu"))]
fn parity(fixture: &str, device: bdecide::Device) {
    use bdecide::hub::ModelSource;
    use bdecide::{AutoModel, LoadOptions};
    use camino::Utf8Path;
    use serde_json::Value;

    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let model = AutoModel::from_pretrained(
        LoadOptions::builder()
            .source(ModelSource::Local(root.clone()))
            .device(device)
            .build(),
    )
    .expect("fixture model");
    let mut requests: Vec<SystemOneRequest> = [
        json!({"state":"alpha", "questions":{"first":{"type":"noul","instructions":"cancel?"}}}),
        json!({"state":"beta alpha beta", "questions":{
            "first":{"type":"choice","instructions":"Select", "criteria":{"alpha":null,"beta":"beta"}},
            "second":{"type":"score","instructions":"Rate", "criteria":["low","high"]}
        }}),
        json!({"state":"beta", "questions":{"other":{"type":"noul","instructions":"continue?"}}}),
    ].into_iter().map(|value| serde_json::from_value(value).expect("valid request")).collect();
    if fixture == "tiny-laya" {
        requests.push(SystemOneRequest::new(
            json!("empty questions"),
            IndexMap::new(),
        ));
        let mut truncated = requests.first().expect("request").clone();
        truncated.state = json!("alpha ".repeat(100));
        truncated.options.max_len = Some(64);
        truncated.options.truncation = bdecide::Truncation::Truncate;
        requests.push(truncated);
    }
    if fixture.starts_with("tiny-d1-") {
        let reference: Value = serde_json::from_slice(
            &std::fs::read(root.join("reference.json")).expect("reference fixture"),
        )
        .expect("reference JSON");
        for case in reference
            .get("cases")
            .and_then(Value::as_array)
            .expect("reference cases")
        {
            let mut request: SystemOneRequest =
                serde_json::from_value(case.get("request").expect("reference request").clone())
                    .expect("valid request");
            if request.images.is_empty() && request.audio.is_empty() {
                continue;
            }
            // Keep fixture paths local to the checkpoint, e.g. image and WAV requests.
            for image in &mut request.images {
                if let bdecide::ImageInput::Path(path) = image {
                    *path = root.join(&*path);
                }
            }
            for audio in &mut request.audio {
                if let bdecide::AudioInput::Path(path) = audio {
                    *path = root.join(&*path);
                }
            }
            requests.push(request);
        }
    }
    let expected: Vec<Value> = requests
        .iter()
        .map(|request| {
            serde_json::to_value(model.system_one(request).expect("single inference"))
                .expect("response JSON")
        })
        .collect();
    for _ in 0..2 {
        let actual = model.system_one_batch(&requests).expect("batch inference");
        assert_eq!(actual.len(), requests.len());
        for (actual, expected) in actual.into_iter().zip(&expected) {
            compare(
                &serde_json::to_value(actual).expect("response JSON"),
                expected,
            );
        }
    }
    compare(
        &serde_json::to_value(
            model
                .system_one_batch(requests.get(..1).expect("first request"))
                .expect("single batch")
                .first()
                .expect("response"),
        )
        .expect("response JSON"),
        expected.first().expect("expected response"),
    );
    assert!(model.system_one_batch(&[]).expect("empty batch").is_empty());
    let mut invalid = requests.first().expect("request").clone();
    invalid.options.max_len = Some(1);
    assert!(matches!(
        model.system_one_batch(&[requests.first().expect("request").clone(), invalid]),
        Err(Error::InvalidRequest(_))
    ));
    model
        .system_one_batch(&requests)
        .expect("batch remains usable after error");
}

#[cfg(any(feature = "cpu", feature = "wgpu"))]
fn compare(actual: &serde_json::Value, expected: &serde_json::Value) {
    use serde_json::Value;
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len());
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>()
            );
            for (key, value) in expected {
                compare(actual.get(key).expect("response field"), value);
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                compare(actual, expected);
            }
        }
        (Value::Number(actual), Value::Number(expected)) => {
            assert!(
                (actual.as_f64().expect("number") - expected.as_f64().expect("number")).abs()
                    < 2e-4,
                "{actual} != {expected}"
            );
        }
        _ => assert_eq!(actual, expected),
    }
}

#[cfg(feature = "cpu")]
#[rstest::rstest]
#[case("tiny-laya")]
#[case("tiny-clef")]
#[case("tiny-vev-4b")]
#[case("tiny-wald")]
#[case("tiny-decider")]
#[case("tiny-von")]
#[case("tiny-d1-3b")]
#[case("tiny-d1-omni")]
fn cpu_batch_matches_single_requests(#[case] fixture: &str) {
    parity(fixture, bdecide::Device::flex());
}

#[cfg(feature = "wgpu")]
#[rstest::rstest]
#[case("tiny-laya")]
#[case("tiny-clef")]
#[case("tiny-vev-4b")]
#[case("tiny-wald")]
#[case("tiny-decider")]
#[case("tiny-von")]
#[case("tiny-d1-3b")]
#[case("tiny-d1-omni")]
#[ignore = "requires a wgpu adapter"]
fn wgpu_batch_matches_single_requests(#[case] fixture: &str) {
    parity(fixture, bdecide::Device::wgpu(Default::default()));
}
