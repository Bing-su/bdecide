//! Check configurable image activations against Transformers with identical checkpoint weights.
use burn::tensor::Device;
use camino::Utf8Path;
use rstest::rstest;
use serde::Deserialize;

use crate::models::lfm2_vl::{Lfm2VlConfig, Lfm2VlForConditionalGeneration};
use crate::models::weights;
use crate::utils::activation::tests::assert_close;
use crate::utils::read_checkpoint_json;
use crate::{Error, ImageInput};

fn config() -> Lfm2VlConfig {
    Lfm2VlConfig::from_pretrained(&root()).expect("tiny vision-language config")
}

fn root() -> camino::Utf8PathBuf {
    Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-d1-3b")
}

#[derive(Deserialize)]
struct Reference {
    vision_act: String,
    projector_act: String,
    image_features: Vec<f32>,
}

fn matches_transformers(device: Device, vision_act: &str, projector_act: &str) {
    let root = root();
    let references: Vec<Reference> = read_checkpoint_json(&root.join("activation-reference.json"))
        .expect("activation references");
    let reference = references
        .into_iter()
        .find(|case| case.vision_act == vision_act && case.projector_act == projector_act)
        .expect("activation reference");
    let mut config = config();
    config.vision_config.hidden_act = vision_act.into();
    config.projector_hidden_act = projector_act.into();
    let mut model =
        Lfm2VlForConditionalGeneration::new(&config, &device).expect("configured model");
    let files = weights::checkpoint_files(&root, |_| true).expect("checkpoint files");
    // Loading weights must preserve both activation choices, e.g. a SiLU projector.
    weights::load(
        &mut model,
        &root,
        &files,
        &[],
        super::super::weights::weight_name,
    )
    .expect("checkpoint weights");
    let features = model
        .model
        .get_image_features(&[ImageInput::Path(root.join("rectangle.png"))], &device)
        .expect("image features");
    assert_close(
        features.into_data(),
        &reference.image_features,
        "image activations",
        3e-4,
    );
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::vision("relu", "gelu")]
#[case::projector("gelu_pytorch_tanh", "silu")]
#[case::both("gelu", "tanh")]
fn cpu_matches_transformers_activations(#[case] vision_act: &str, #[case] projector_act: &str) {
    matches_transformers(Device::flex(), vision_act, projector_act);
}

#[cfg(feature = "wgpu")]
#[rstest]
#[case::vision("relu", "gelu")]
#[case::projector("gelu_pytorch_tanh", "silu")]
#[case::both("gelu", "tanh")]
#[ignore = "requires a wgpu adapter"]
fn wgpu_matches_transformers_activations(#[case] vision_act: &str, #[case] projector_act: &str) {
    matches_transformers(Device::wgpu(Default::default()), vision_act, projector_act);
}

#[rstest]
fn rejects_unsupported_activations(
    #[values("prelu", "xielu", "unknown")] name: &str,
    #[values(true, false)] vision: bool,
) {
    let mut config = config();
    if vision {
        config.vision_config.hidden_act = name.into();
    } else {
        config.projector_hidden_act = name.into();
    }
    let error = config
        .validate()
        .expect_err("reject unsupported activation");
    assert!(matches!(error, Error::UnsupportedModel(_)));
    assert!(error.to_string().contains(name));
}
