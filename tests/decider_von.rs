//! Compare the public loaders and predictions with pinned upstream tiny models.
use std::fs;

use approx::abs_diff_eq;
use bdecide::hub::ModelSource;
use bdecide::{AutoModel, DecisionModel, Device, LoadOptions, Request};
#[cfg(feature = "cpu")]
use bdecide::{DeciderModel, Error, Truncation, VonModel};
#[cfg(feature = "cpu")]
use burn::tensor::Device as BurnDevice;
use camino::{Utf8Path, Utf8PathBuf};
use rstest::rstest;
use serde_json::Value;

fn fixture(name: &str) -> Utf8PathBuf {
    Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
fn reference(name: &str) -> Value {
    serde_json::from_slice(
        &fs::read(fixture(name).join("reference.json"))
            .expect("reference fixture and prediction must be valid"),
    )
    .expect("reference fixture and prediction must be valid")
}
fn load(name: &str, device: Device) -> AutoModel {
    AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(fixture(name)),
        device: Some(device),
    })
    .expect("reference fixture and prediction must be valid")
}
fn compare(actual: &Value, expected: &Value) {
    match expected {
        Value::Object(fields) => {
            for (key, value) in fields {
                // Upstream-only diagnostics do not belong to bdecide's common Answer schema,
                // e.g. Decider's per-level fit mass and Von's auxiliary noul_raw.
                if matches!(
                    key.as_str(),
                    "x_p_max" | "certainty" | "level_fit" | "fit_mass" | "noul_raw"
                ) {
                    continue;
                }
                if key == "score" && value.is_number() {
                    // Upstream rounds ordinal expectations to two decimals; bdecide retains precision.
                    assert!(abs_diff_eq!(
                        actual[key].as_f64().expect("score must be numeric"),
                        value.as_f64().expect("score reference must be numeric"),
                        epsilon = 0.0051
                    ));
                } else {
                    compare(&actual[key], value);
                }
            }
        }
        Value::Number(value) => assert!(
            abs_diff_eq!(
                actual.as_f64().expect("answer must be numeric"),
                value.as_f64().expect("answer reference must be numeric"),
                epsilon = 5.1e-4
            ),
            "{actual} != {expected}"
        ),
        Value::String(text) => {
            if let (Some(actual), Ok(expected)) =
                (actual.as_str(), serde_json::from_str::<Value>(text))
            {
                assert_eq!(
                    serde_json::from_str::<Value>(actual).expect("structured legends must be JSON"),
                    expected
                );
            } else {
                assert_eq!(actual, expected);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}
fn verify(name: &str, device: Device, case_limit: usize) {
    let model = load(name, device);
    assert_eq!(
        model.metadata().architecture,
        if name.starts_with("tiny-decider") {
            "decider"
        } else {
            "von"
        }
    );
    let reference = reference(name);
    for case in reference
        .get("cases")
        .expect("reference cases must exist")
        .as_array()
        .expect("reference fixture and prediction must be valid")
        .iter()
        .take(case_limit)
    {
        let request: Request = serde_json::from_value(case["request"].clone())
            .expect("reference fixture and prediction must be valid");
        let response = model
            .system_one(&request)
            .expect("reference fixture and prediction must be valid");
        compare(
            &serde_json::to_value(&response.answers)
                .expect("reference fixture and prediction must be valid"),
            &case["answers"],
        );
        assert_eq!(response.usage.output_tokens, 0);
        assert!(response.usage.input_tokens > 0);
    }
}
#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-decider")]
#[case("tiny-decider-list")]
#[case("tiny-von")]
#[case("tiny-von-joint")]
fn cpu_matches_upstream(#[case] name: &str) {
    verify(name, Device::flex(), usize::MAX);
}
#[cfg(feature = "wgpu")]
#[rstest]
#[case("tiny-decider")]
#[case("tiny-decider-list")]
#[case("tiny-von")]
#[case("tiny-von-joint")]
#[ignore = "requires a wgpu adapter"]
fn wgpu_matches_upstream(#[case] name: &str) {
    verify(name, Device::wgpu(Default::default()), 1);
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-decider")]
#[case("tiny-von")]
fn direct_loader_and_independent_questions(#[case] name: &str) {
    let source = ModelSource::Local(fixture(name));
    let direct: Box<dyn DecisionModel> = if name.starts_with("tiny-decider") {
        Box::new(
            DeciderModel::from_pretrained(&source, &BurnDevice::flex())
                .expect("reference fixture and prediction must be valid"),
        )
    } else {
        Box::new(
            VonModel::from_pretrained(&source, &BurnDevice::flex())
                .expect("reference fixture and prediction must be valid"),
        )
    };
    let auto = load(name, Device::flex());
    let mut request: Request =
        serde_json::from_value(reference(name)["cases"][0]["request"].clone())
            .expect("reference fixture and prediction must be valid");
    let expected = serde_json::to_value(
        auto.system_one(&request)
            .expect("reference fixture and prediction must be valid")
            .answers,
    )
    .expect("reference fixture and prediction must be valid");
    assert_eq!(
        expected,
        serde_json::to_value(direct.system_one(&request).unwrap().answers).unwrap()
    );
    let all = request.questions.clone();
    for (id, question) in all {
        request.questions.clear();
        request.questions.insert(id.clone(), question);
        let actual = serde_json::to_value(
            auto.system_one(&request)
                .expect("reference fixture and prediction must be valid")
                .answers,
        )
        .expect("reference fixture and prediction must be valid");
        assert_eq!(actual[&id], expected[&id]);
    }
}
#[cfg(feature = "cpu")]
#[test]
fn von_options_are_order_invariant() {
    use bdecide::Question;
    let model = load("tiny-von", Device::flex());
    let mut request: Request =
        serde_json::from_value(reference("tiny-von")["cases"][0]["request"].clone())
            .expect("reference fixture and prediction must be valid");
    request.questions.retain(|id, _| id == "choice");
    let before = serde_json::to_value(
        model
            .system_one(&request)
            .expect("reference fixture and prediction must be valid")
            .answers,
    )
    .expect("reference fixture and prediction must be valid");
    if let Some(Question::Choice { criteria, .. }) = request.questions.get_mut("choice") {
        criteria.reverse();
    }
    let after = serde_json::to_value(
        model
            .system_one(&request)
            .expect("reference fixture and prediction must be valid")
            .answers,
    )
    .expect("reference fixture and prediction must be valid");
    compare(&after, &before);
}

#[cfg(feature = "cpu")]
#[test]
fn decider_score_preserves_levels_with_small_choice_limit() {
    let directory = tempfile::tempdir().expect("temporary checkpoint must be writable");
    let root = Utf8Path::from_path(directory.path()).expect("checkpoint path must be UTF-8");
    for file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ] {
        fs::copy(fixture("tiny-decider-list").join(file), root.join(file))
            .expect("fixture artifact must exist");
    }
    let mut config: Value = serde_json::from_slice(
        &fs::read(fixture("tiny-decider-list").join("decider_config.json")).unwrap(),
    )
    .unwrap();
    // Limit Choice without dropping Score labels, e.g. three levels still need A/B/C.
    config["max_options"] = Value::from(2);
    fs::write(
        root.join("decider_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let model =
        DeciderModel::from_pretrained(&ModelSource::Local(root.into()), &BurnDevice::flex())
            .unwrap();
    let mut request: Request =
        serde_json::from_value(reference("tiny-decider-list")["cases"][0]["request"].clone())
            .unwrap();
    assert!(matches!(
        model.system_one(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.questions.retain(|id, _| id == "score");
    let response = model.system_one(&request).unwrap();
    let bdecide::Answer::Score {
        probabilities,
        legend,
        ..
    } = &response.answers["score"]
    else {
        panic!("Score question must return a Score answer");
    };
    assert_eq!(probabilities.len(), 3);
    assert_eq!(
        probabilities.keys().collect::<Vec<_>>(),
        legend.keys().collect::<Vec<_>>()
    );
    assert!(abs_diff_eq!(
        probabilities.values().sum::<f64>(),
        1.0,
        epsilon = 1e-6
    ));
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-decider")]
#[case("tiny-von")]
fn budgets_preserve_options_and_report_lost_state(#[case] name: &str) {
    let model = load(name, Device::flex());
    let mut request: Request = serde_json::from_value(serde_json::json!({"state":"alpha ".repeat(500), "questions":{"q":{"type":"choice","instructions":"Select", "criteria":{"alpha":null,"beta":null}}}})).expect("reference fixture and prediction must be valid");
    request.options.max_len = Some(100);
    assert!(matches!(
        model.system_one(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.options.truncation = Truncation::Truncate;
    let response = model
        .system_one(&request)
        .expect("reference fixture and prediction must be valid");
    assert_eq!(response.usage.input_tokens, 100);
    assert!(response.usage.truncated);
    assert!(response.usage.state_tokens_dropped > 0);
    assert_eq!(response.usage.truncated_questions, ["q"]);
    request.options.max_len = Some(4);
    assert!(matches!(
        model.system_one(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.options.max_len = Some(4097);
    assert!(matches!(
        model.system_one(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.options.max_len = None;
    request.options.head_max_len = Some(16);
    assert!(matches!(
        model.system_one(&request),
        Err(Error::InvalidRequest(_))
    ));
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::missing_head("missing")]
#[case::unexpected_head("unused")]
#[case::nonfinite_head("nan")]
#[case::wrong_shape("shape")]
fn von_rejects_corrupt_state_dict(#[case] kind: &str) {
    let directory = tempfile::tempdir().expect("reference fixture and prediction must be valid");
    let root = Utf8Path::from_path(directory.path())
        .expect("reference fixture and prediction must be valid");
    for file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "marker_calibration.json",
    ] {
        fs::copy(fixture("tiny-von").join(file), root.join(file))
            .expect("reference fixture and prediction must be valid");
    }
    fs::copy(
        fixture("tiny-von").join(format!("corrupt-{kind}.pt")),
        root.join("option_marker.pt"),
    )
    .expect("reference fixture and prediction must be valid");
    assert!(matches!(
        VonModel::from_pretrained(&ModelSource::Local(root.into()), &BurnDevice::flex()),
        Err(Error::Weights(_))
    ));
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-decider")]
#[case("tiny-von")]
#[tokio::test]
async fn hub_revisions_stay_pinned_and_load_offline(#[case] name: &str) {
    use bdecide::hub::{HubOptions, Token};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    let cache = tempfile::tempdir().expect("temporary cache must be writable");
    let sha = "1234567890123456789012345678901234567890";
    let mut files = vec!["config.json", "tokenizer.json", "tokenizer_config.json"];
    let mut absent = vec![("main", "rl_agent_config.json")];
    if name == "tiny-von" {
        files.extend(["marker_calibration.json", "option_marker.pt"]);
    } else {
        files.extend(["decider_config.json", "model.safetensors"]);
        absent.extend(
            [
                "joint_head_config.json",
                "vev.json",
                "serving.json",
                "model.safetensors.index.json",
            ]
            .into_iter()
            .map(|file| (sha, file)),
        );
    }
    for file in files {
        let revision = if file == "config.json" { "main" } else { sha };
        let response = ResponseTemplate::new(200)
            .insert_header("X-Repo-Commit", sha)
            .insert_header("ETag", format!("\"{}\"", file.replace('.', "-")))
            .set_body_bytes(
                fs::read(fixture(name).join(file)).expect("fixture artifact must exist"),
            );
        for verb in ["HEAD", "GET"] {
            Mock::given(method(verb))
                .and(path(format!(
                    "/test/renamed/resolve/{revision}/nested/{file}"
                )))
                .respond_with(response.clone())
                .expect(1)
                .mount(&server)
                .await;
        }
    }
    for (revision, file) in absent {
        Mock::given(method("HEAD"))
            .and(path(format!(
                "/test/renamed/resolve/{revision}/nested/{file}"
            )))
            .respond_with(ResponseTemplate::new(404).insert_header("X-Error-Code", "EntryNotFound"))
            .expect(1)
            .mount(&server)
            .await;
    }
    let mut options = HubOptions::new("test/renamed");
    options.endpoint = Some(server.uri());
    options.cache_dir = Some(
        Utf8Path::from_path(cache.path())
            .expect("cache path must be UTF-8")
            .into(),
    );
    options.subfolder = Some("nested".into());
    options.token = Token::Anonymous;
    let request: Request = serde_json::from_value(reference(name)["cases"][0]["request"].clone())
        .expect("reference request must be valid");
    let system_one = |options| {
        AutoModel::from_pretrained(LoadOptions {
            source: ModelSource::Hub(options),
            device: Some(Device::flex()),
        })
        .expect("pinned artifacts must load")
        .system_one(&request)
        .expect("reference request must predict")
    };
    let online =
        serde_json::to_value(system_one(options.clone())).expect("response must serialize");
    assert_eq!(online["metadata"]["commit_sha"], sha);
    options.local_files_only = true;
    assert_eq!(
        online,
        serde_json::to_value(system_one(options)).expect("offline response must serialize")
    );
    server.verify().await;
}
