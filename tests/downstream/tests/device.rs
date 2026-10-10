use bdecide::hub::ModelSource;
use bdecide::{AutoModel, DecisionModel, Device, LoadOptions, Request};

fn source() -> ModelSource {
    ModelSource::Local(format!("{}/../fixtures/tiny-laya", env!("CARGO_MANIFEST_DIR")).into())
}

fn infer(options: LoadOptions, expected_device: &Device) {
    let model =
        AutoModel::from_pretrained(options).expect("consumer backend must load the fixture");
    assert_eq!(model.metadata().device, format!("{expected_device:?}"));
    let request: Request = serde_json::from_str(
        r#"{"state":"alpha","questions":{"q":{"type":"noul","instructions":"cancel?"}}}"#,
    )
    .expect("request fixture must be valid");
    let response = model
        .system_one(&request)
        .expect("consumer backend must execute the request");
    assert_eq!(response.metadata.device, format!("{expected_device:?}"));
    assert_eq!(response.answers.len(), 1);
    assert!(response.usage.input_tokens > 0);
}

#[test]
fn external_burn_features_work_without_bdecide_backend_features() {
    // Exercise feature unification, e.g. Flex comes only from this consumer's Burn dependency.
    let options = LoadOptions::builder().source(source()).build();
    assert!(options.device.is_none());
    infer(options, &Device::default());
    infer(
        LoadOptions {
            source: source(),
            device: Some(Device::flex()),
        },
        &Device::flex(),
    );
}

#[test]
#[expect(
    deprecated,
    reason = "Exercise another executable backend without requiring GPU hardware"
)]
fn explicit_device_overrides_the_consumers_default_backend() {
    // NdArray must run even though the consumer's default is Flex, e.g. no silent replacement.
    let device = Device::ndarray();
    assert_ne!(device, Device::flex());
    infer(
        LoadOptions {
            source: source(),
            device: Some(device.clone()),
        },
        &device,
    );
}
