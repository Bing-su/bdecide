#![cfg(feature = "wgpu")]
use bdecide::{AutoModel, DecisionModel, Device, LoadOptions, Request, hub::ModelSource};
use burn::{backend::Wgpu, tensor::Tensor};
use camino::Utf8Path;
use rstest::rstest;
use serde_json::Value;

#[cfg(target_os = "linux")]
#[rstest]
#[case::explicit("wgpu")]
#[case::automatic("auto")]
fn missing_adapter_keeps_jsonl_requests_recoverable(#[case] device: &str) {
    let directory = tempfile::tempdir().unwrap();
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
    let input = directory.path().join("requests.jsonl");
    let request = r#"{"state":"alpha","questions":{"q":{"type":"noul","instructions":"cancel?"}}}"#;
    std::fs::write(&input, format!("{request}\n{request}\n")).unwrap();
    let missing_driver = directory.path().join("missing-icd.json");
    // Linux AutoGraphicsApi uses Vulkan. Hide drivers in a child to test failure
    // before model allocation, including a second JSONL request on the same process.
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_bdecide"))
        .args([
            "predict",
            "--model",
            root.as_str(),
            "--device",
            device,
            "--jsonl",
            "--input",
        ])
        .arg(input)
        .env("VK_DRIVER_FILES", &missing_driver)
        .env("VK_ICD_FILENAMES", missing_driver)
        .env("XDG_RUNTIME_DIR", directory.path())
        .output()
        .unwrap();
    let fallback = device == "auto" && cfg!(feature = "cpu");
    assert_eq!(
        output.status.code(),
        Some(i32::from(!fallback)),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    for row in rows {
        if fallback {
            assert_eq!(row["metadata"]["device"], "cpu");
        } else {
            assert_eq!(row["error"]["kind"], "device");
        }
    }
}

// Opt in to verify GPU parity on a wgpu adapter, e.g. Mesa's software adapter:
// cargo test --features wgpu --test wgpu -- --ignored --nocapture
#[rstest]
#[case::explicit(Device::Wgpu)]
#[case::automatic(Device::Auto)]
#[ignore = "requires a wgpu adapter"]
fn wgpu_matches_independent_python_answers(#[case] device: Device) {
    // A host may use Burn first, e.g. create a GPU tensor before loading bdecide.
    // Every case warms up the runtime so test ordering cannot hide double registration.
    let _ = Tensor::<Wgpu<f32, i32>, 1>::zeros([1], &Default::default()).into_data();
    let fixtures = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(fixtures.join("tiny-laya")),
        device,
    })
    .unwrap();
    assert_eq!(model.metadata().device, "wgpu");
    // Multiple independent models must reuse Burn's registered wgpu runtime.
    let second = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(fixtures.join("tiny-laya")),
        device: Device::Wgpu,
    })
    .unwrap();
    assert_eq!(second.metadata().device, "wgpu");
    let reference: Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("reference.json")).unwrap()).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        let mut actual = serde_json::to_value(model.predict(&request).unwrap()).unwrap();
        let mut expected = case["response"]["answers"].clone();
        // Ignore JSON text formatting in legends, e.g. ["beta",null] and ["beta", null].
        for answers in [&mut actual["answers"], &mut expected] {
            for answer in answers.as_object_mut().unwrap().values_mut() {
                if let Some(Value::Object(legend)) = answer.get_mut("legend") {
                    for text in legend.values_mut() {
                        if let Ok(value) = serde_json::from_str::<Value>(text.as_str().unwrap()) {
                            *text = value;
                        }
                    }
                }
            }
        }
        let mut pairs = vec![(&actual["answers"], &expected)];
        while let Some((actual, expected)) = pairs.pop() {
            match expected {
                Value::Object(fields) => {
                    pairs.extend(fields.iter().map(|(key, value)| (&actual[key], value)))
                }
                Value::Number(number) => assert!(
                    (actual.as_f64().unwrap() - number.as_f64().unwrap()).abs() < 4e-4,
                    "{actual} != {expected}"
                ),
                _ => assert_eq!(actual, expected),
            }
        }
        assert_eq!(
            actual["usage"]["input_tokens"],
            case["response"]["usage"]["input_tokens"]
        );
        assert_eq!(actual["metadata"]["device"], "wgpu");
    }
}

#[rstest]
#[case("tiny-clef")]
#[case("tiny-clef-flash")]
#[ignore = "requires a wgpu adapter"]
fn clef_wgpu_matches_independent_python_answers(#[case] variant: &str) {
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(variant);
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(root.clone()),
        device: Device::Wgpu,
    })
    .unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let request = serde_json::from_value(case["request"].clone()).unwrap();
        let response = serde_json::to_value(model.predict(&request).unwrap()).unwrap();
        for (id, expected) in case["answers"].as_object().unwrap() {
            // The score legend is adapted to bdecide's string-valued legend contract.
            // Compare all remaining native Clef answer fields, e.g. probabilities.
            let mut pairs = vec![(&response["answers"][id], expected)];
            while let Some((actual, expected)) = pairs.pop() {
                match expected {
                    Value::Object(fields) => pairs.extend(
                        fields
                            .iter()
                            .filter(|(key, _)| *key != "legend")
                            .map(|(key, value)| (&actual[key], value)),
                    ),
                    Value::Number(number) => assert!(
                        (actual.as_f64().unwrap() - number.as_f64().unwrap()).abs() < 4e-4,
                        "{actual} != {expected}"
                    ),
                    _ => assert_eq!(actual, expected),
                }
            }
        }
        assert_eq!(response["metadata"]["device"], "wgpu");
        assert_eq!(
            response["usage"]["input_tokens"],
            case["input_ids"].as_array().unwrap().len()
        );
    }
}
