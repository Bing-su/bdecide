#![cfg(feature = "wgpu")]
use std::fs;
use std::process::Command;

use bdecide::hub::ModelSource;
use bdecide::{
    AutoModel,
    ClefProcessor,
    DecisionModel,
    Device,
    Error,
    LoadOptions,
    Question,
    Qwen3_5Config,
    Request,
};
use burn::tensor::wgpu::WgpuBackend;
use burn::tensor::{Device as BurnDevice, DeviceKind, Tensor};
use camino::Utf8Path;
use rstest::rstest;
use serde_json::Value;
use tempfile::tempdir;

#[rstest]
#[case::explicit("wgpu")]
#[case::automatic("auto")]
fn invalid_wgpu_configuration_keeps_jsonl_requests_recoverable(#[case] device: &str) {
    let directory = tempdir().unwrap();
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
    let input = directory.path().join("requests.jsonl");
    let request = r#"{"state":"alpha","questions":{"q":{"type":"noul","instructions":"cancel?"}}}"#;
    fs::write(&input, format!("{request}\n{request}\n")).unwrap();
    // Reject the selector before graphics APIs initialize, e.g. "invalid" cannot select hardware.
    // Keep both JSONL requests recoverable without accessing an adapter.
    let output = Command::new(env!("CARGO_BIN_EXE_bdecide"))
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
        .env("BURN_DEVICE", "wgpu")
        .env("CUBECL_WGPU_DEFAULT_DEVICE", "invalid")
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
            #[cfg(feature = "cpu")]
            assert_eq!(row["metadata"]["device"], format!("{:?}", Device::flex()));
        } else {
            assert_eq!(row["error"]["kind"], "device");
        }
    }
}

// Opt in to verify GPU parity on a wgpu adapter, e.g. Mesa's software adapter:
// cargo test --features wgpu --test wgpu -- --ignored --nocapture
#[rstest]
#[case::explicit(Some(Device::wgpu(Default::default())))]
#[case::automatic(None)]
#[ignore = "requires a wgpu adapter"]
fn wgpu_matches_independent_python_answers(#[case] device: Option<Device>) {
    let expected_device = format!("{:?}", device.clone().unwrap_or_default());
    // A host may use Burn first, e.g. create a GPU tensor before loading bdecide.
    // Every case warms up the runtime so test ordering cannot hide double registration.
    let _ = Tensor::<1>::zeros([1], &BurnDevice::wgpu(Default::default())).into_data();
    let fixtures = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(fixtures.join("tiny-laya")),
        device,
    })
    .unwrap();
    assert_eq!(model.metadata().device, expected_device);
    // Multiple independent models must reuse Burn's registered wgpu runtime.
    let second = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(fixtures.join("tiny-laya")),
        device: Some(Device::wgpu(Default::default())),
    })
    .unwrap();
    assert_eq!(
        second.metadata().device,
        format!("{:?}", Device::wgpu(Default::default()))
    );
    let reference: Value =
        serde_json::from_slice(&fs::read(fixtures.join("reference.json")).unwrap()).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        // Keep end-to-end Python parity for text prompts; native-token parity covers JSON.
        let text_only = request.state.is_string()
            && request.questions.values().all(|question| match question {
                Question::Choice { criteria, .. } | Question::Noul { criteria, .. } => criteria
                    .values()
                    .all(|value| value.is_string() || value.is_null()),
                Question::Score { criteria, .. } => criteria.iter().all(Value::is_string),
            });
        if !text_only {
            continue;
        }
        let mut actual = serde_json::to_value(model.system_one(&request).unwrap()).unwrap();
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
        assert_eq!(actual["metadata"]["device"], expected_device);
    }
}

#[test]
#[ignore = "requires a wgpu adapter"]
fn explicit_wgpu_device_preserves_adapter_selection() {
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
    // A working software adapter establishes that failure is from the supplied selector.
    let device = Device::wgpu_options()
        .device_kind(DeviceKind::Cpu)
        .graphics_api(WgpuBackend::Vulkan)
        .init()
        .unwrap();
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(root.clone()),
        device: Some(device.clone()),
    })
    .unwrap();
    assert_eq!(model.metadata().device, format!("{device:?}"));
    // Never replace an explicit device or fall back to CPU, e.g. an absent virtual GPU.
    assert!(matches!(
        AutoModel::from_pretrained(LoadOptions {
            source: ModelSource::Local(root),
            device: Some(Device::wgpu(DeviceKind::VirtualGpu(8191))),
        }),
        Err(Error::Device(_))
    ));
}

#[rstest]
#[case("tiny-clef")]
#[case("tiny-clef-flash")]
#[ignore = "requires a wgpu adapter"]
fn clef_wgpu_processes_text_and_json_requests(#[case] variant: &str) {
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(variant);
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(root.clone()),
        device: Some(Device::wgpu(Default::default())),
    })
    .unwrap();
    let reference: Value =
        serde_json::from_slice(&fs::read(root.join("reference.json")).unwrap()).unwrap();
    let config = Qwen3_5Config::from_pretrained(&root).unwrap();
    let processor = ClefProcessor::from_pretrained(&root, &config).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let request = serde_json::from_value(case["request"].clone()).unwrap();
        let encoded = processor.process(&request).unwrap();
        let response = serde_json::to_value(model.system_one(&request).unwrap()).unwrap();
        assert_eq!(
            response["answers"].as_object().unwrap().len(),
            request.questions.len()
        );
        // Compare native answers only for identical tokens, e.g. a null-only schema option.
        // Numeric WGPU parity for every upstream record is tested on reference tokens in Clef.
        if serde_json::to_value(&encoded.input_ids).unwrap() == case["input_ids"] {
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
        }
        assert_eq!(
            response["metadata"]["device"],
            format!("{:?}", Device::wgpu(Default::default()))
        );
        assert_eq!(response["usage"]["input_tokens"], encoded.input_ids.len());
    }
}
