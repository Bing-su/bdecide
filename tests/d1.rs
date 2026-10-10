//! Compare public loading and multimodal decisions with pinned LiquidAI Python fixtures.
#[cfg(feature = "cpu")]
use bdecide::Error;
use bdecide::hub::ModelSource;
use bdecide::{Answer, AutoModel, DecisionModel, Device, LoadOptions, Request};
use burn::tensor::Device as BurnDevice;
use camino::{Utf8Path, Utf8PathBuf};
use rstest::rstest;
use serde_json::Value;

fn fixture(kind: &str) -> Utf8PathBuf {
    Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/tiny-d1-{kind}"))
}

fn request(root: &Utf8Path, value: Value) -> Request {
    let mut request: Request = serde_json::from_value(value).expect("valid reference fixture");
    // Resolve fixture-relative media explicitly, e.g. CLI paths remain relative to its cwd.
    for image in &mut request.images {
        if let bdecide::ImageInput::Path(path) = image {
            *path = root.join(&*path);
        }
    }
    if let Some(bdecide::AudioInput::Path(path)) = &mut request.audio {
        *path = root.join(&*path);
    }
    request
}

fn parity(device: Device, kind: &str, case_index: usize) {
    let root = fixture(kind);
    let model = AutoModel::from_pretrained(
        LoadOptions::builder()
            .source(ModelSource::Local(root.clone()))
            .device(device)
            .build(),
    )
    .expect("valid reference fixture");
    assert_eq!(
        model.metadata().architecture,
        if kind == "omni" { "d1_omni" } else { "d1" }
    );
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("reference.json")).expect("valid reference fixture"),
    )
    .expect("valid reference fixture");
    let case = reference
        .get("cases")
        .expect("reference cases")
        .as_array()
        .expect("valid reference fixture")
        .get(case_index)
        .expect("reference case");
    let response = model
        .system_one(&request(&root, case["request"].clone()))
        .expect("valid reference fixture");
    assert_eq!(response.usage.output_tokens, 0);
    assert!(response.usage.input_tokens > 0);
    for (id, answer) in response.answers {
        let expected = case
            .get("answers")
            .and_then(|answers| answers.get(&id))
            .expect("answer in reference");
        let check = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() < 3e-4,
                "{kind} case {case_index} {id}: {actual} != {expected}"
            );
        };
        match answer {
            Answer::Noul { noul, .. } => check(
                noul,
                expected["noul"].as_f64().expect("valid reference fixture"),
            ),
            Answer::Choice {
                choice,
                probabilities,
                ..
            } => {
                assert_eq!(
                    choice,
                    expected["choice"].as_str().expect("reference choice"),
                    "{kind} case {case_index} {id}: {probabilities:?}"
                );
                for (name, p) in probabilities {
                    check(
                        p,
                        expected["probabilities"]
                            .get(&name)
                            .and_then(Value::as_f64)
                            .expect("reference probability"),
                    );
                }
            }
            Answer::Score {
                score,
                probabilities,
                ..
            } => {
                check(
                    score,
                    expected["score"].as_f64().expect("valid reference fixture"),
                );
                for (name, p) in probabilities {
                    check(
                        p,
                        expected["probabilities"]
                            .get(&name)
                            .and_then(Value::as_f64)
                            .expect("reference probability"),
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::d1_text("3b", 0)]
#[case::d1_structured_state("3b", 1)]
#[case::d1_image("3b", 2)]
#[case::d1_multiple_images("3b", 3)]
#[case::d1_tiled_image("3b", 4)]
#[case::d1_noul_criteria("3b", 5)]
#[case::omni_text("omni", 0)]
#[case::omni_structured_state("omni", 1)]
#[case::omni_image("omni", 2)]
#[case::omni_multiple_images("omni", 3)]
#[case::omni_tiled_image("omni", 4)]
#[case::omni_noul_criteria("omni", 5)]
#[case::omni_audio("omni", 6)]
#[case::omni_pcm16("omni", 7)]
#[case::omni_state_truncation("omni", 8)]
#[case::omni_question_truncation("omni", 9)]
fn cpu_matches_liquidai_text_image_audio(#[case] kind: &str, #[case] case_index: usize) {
    parity(Device::Cpu, kind, case_index);
}

#[cfg(feature = "wgpu")]
#[rstest]
#[case::d1_text("3b", 0)]
#[case::d1_structured_state("3b", 1)]
#[case::d1_image("3b", 2)]
#[case::d1_multiple_images("3b", 3)]
#[case::d1_tiled_image("3b", 4)]
#[case::d1_noul_criteria("3b", 5)]
#[case::omni_text("omni", 0)]
#[case::omni_structured_state("omni", 1)]
#[case::omni_image("omni", 2)]
#[case::omni_multiple_images("omni", 3)]
#[case::omni_tiled_image("omni", 4)]
#[case::omni_noul_criteria("omni", 5)]
#[case::omni_audio("omni", 6)]
#[case::omni_pcm16("omni", 7)]
#[case::omni_state_truncation("omni", 8)]
#[case::omni_question_truncation("omni", 9)]
#[ignore = "requires a wgpu adapter"]
fn wgpu_matches_liquidai_text_image_audio(#[case] kind: &str, #[case] case_index: usize) {
    parity(Device::Wgpu, kind, case_index);
}

#[test]
#[cfg(feature = "cpu")]
fn rejects_unsupported_media_and_lost_tokens() {
    let mut q: Request = serde_json::from_value(serde_json::json!({
        "state": "alpha", "questions": {"q": {"type": "noul", "instructions": "cancel?"}},
        "audio": {"samples": [0.0], "sample_rate": 16000}
    }))
    .expect("valid reference fixture");
    let model = AutoModel::from_pretrained(
        LoadOptions::builder()
            .source(ModelSource::Local(fixture("3b")))
            .build(),
    )
    .expect("valid reference fixture");
    assert!(matches!(
        model.system_one(&q),
        Err(Error::InvalidRequest(_))
    ));
    q.images = vec![bdecide::ImageInput::Rgb {
        width: 1,
        height: 1,
        pixels: vec![0, 0, 0],
    }];
    assert!(matches!(q.validate(), Err(Error::InvalidRequest(_))));
    q.images.clear();
    q.audio = None;
    q.state = Value::String("alpha ".repeat(500));
    q.options.max_len = Some(240);
    let model = AutoModel::from_pretrained(
        LoadOptions::builder()
            .source(ModelSource::Local(fixture("omni")))
            .build(),
    )
    .expect("valid reference fixture");
    assert!(matches!(
        model.system_one(&q),
        Err(Error::InvalidRequest(_))
    ));
    q.options.truncation = bdecide::Truncation::Truncate;
    let response = model.system_one(&q).expect("valid reference fixture");
    assert!(response.usage.truncated);
    assert!(response.usage.state_tokens_dropped > 0);
    assert_eq!(response.usage.truncated_questions, ["q"]);
}

#[test]
#[cfg(feature = "cpu")]
fn d1_and_omni_have_separate_loaders() {
    let device = BurnDevice::flex();
    let d1 = ModelSource::Local(fixture("3b"));
    let omni = ModelSource::Local(fixture("omni"));
    assert_eq!(
        bdecide::D1Model::from_pretrained(&d1, &device)
            .expect("D1 fixture")
            .metadata()
            .architecture,
        "d1"
    );
    assert_eq!(
        bdecide::D1OmniModel::from_pretrained(&omni, &device)
            .expect("Omni fixture")
            .metadata()
            .architecture,
        "d1_omni"
    );
    assert!(matches!(
        bdecide::D1Model::from_pretrained(&omni, &device),
        Err(Error::InvalidCheckpoint(_) | Error::UnsupportedModel(_))
    ));
    assert!(matches!(
        bdecide::D1OmniModel::from_pretrained(&d1, &device),
        Err(Error::InvalidCheckpoint(_) | Error::UnsupportedModel(_))
    ));
}

#[derive(serde::Deserialize)]
struct BaseReference {
    tokens: Vec<u32>,
    selected: Vec<u32>,
    hidden: Vec<f32>,
    text_logits: Vec<f32>,
    image: String,
    image_tokens: Vec<u32>,
    image_features: Vec<f32>,
    image_logits: Vec<f32>,
}

fn base_parity(device: BurnDevice) {
    let root = fixture("3b");
    let reference: BaseReference = serde_json::from_str(
        &std::fs::read_to_string(root.join("base-reference.json")).expect("Transformers reference"),
    )
    .expect("reference schema");
    let model = bdecide::Lfm2VlForConditionalGeneration::from_pretrained(&root, &device)
        .expect("base checkpoint");
    let check = |actual: Vec<f32>, expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 3e-4,
                "Transformers index {i}: {actual} != {expected}"
            );
        }
    };
    let hidden = model
        .model
        .forward(&reference.tokens, None, &device)
        .expect("base hidden states");
    check(
        hidden.try_into_vec_as::<f32>().expect("hidden values"),
        &reference.hidden,
    );
    let logits = model
        .forward(&reference.tokens, None, &device)
        .expect("full vocabulary logits");
    check(
        logits.try_into_vec_as::<f32>().expect("logit values"),
        &reference.text_logits,
    );
    let selected: Vec<_> = reference
        .selected
        .iter()
        .map(|&id| {
            *reference
                .text_logits
                .get(
                    reference.text_logits.len() - model.model.config.text_config.vocab_size
                        + id as usize,
                )
                .expect("selected reference logit")
        })
        .collect();
    check(
        model
            .forward_selected(&reference.tokens, None, &reference.selected, &device)
            .expect("selected logits")
            .try_into_vec_as::<f32>()
            .expect("logit values"),
        &selected,
    );
    let features = model
        .model
        .get_image_features(
            &[bdecide::ImageInput::Path(root.join(&reference.image))],
            &device,
        )
        .expect("image features");
    check(
        features
            .clone()
            .try_into_vec_as::<f32>()
            .expect("feature values"),
        &reference.image_features,
    );
    check(
        model
            .forward_selected(
                &reference.image_tokens,
                Some(&features),
                &reference.selected,
                &device,
            )
            .expect("image logits")
            .try_into_vec_as::<f32>()
            .expect("logit values"),
        &reference.image_logits,
    );
    model
        .forward(&[], None, &device)
        .expect_err("reject empty tokens");
    model
        .forward(&reference.image_tokens, None, &device)
        .expect_err("require image features for placeholders");
    model
        .forward_selected(
            &reference.tokens,
            None,
            &[model.model.config.text_config.vocab_size as u32],
            &device,
        )
        .expect_err("reject out-of-vocabulary readout tokens");
}

#[test]
#[cfg(feature = "cpu")]
fn cpu_lfm2_vl_matches_transformers() {
    base_parity(BurnDevice::flex());
}

#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires a wgpu adapter"]
fn wgpu_lfm2_vl_matches_transformers() {
    base_parity(BurnDevice::wgpu(Default::default()));
}
