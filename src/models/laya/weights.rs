//! Handle checkpoint precision and remaining PyTorch/Burn parameter format differences.
use camino::Utf8Path;

use super::modeling_laya::LayaDecisionModel;
use crate::Result;
use crate::models::weights;

/// Apply a Laya PyTorch checkpoint to a separately constructed Burn architecture.
///
/// Burn reads safetensors and converts floating point storage to FP32. Missing,
/// unexpected, or incorrectly shaped tensors fail instead of retaining random parameters.
/// For example: `load_laya(&mut architecture, Utf8Path::new("model.safetensors"))?`.
pub fn load_laya(model: &mut LayaDecisionModel, path: &Utf8Path) -> Result<()> {
    weights::load(
        model,
        Utf8Path::new(""),
        &[path.to_string()],
        &[],
        weights::identity_name,
    )
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use burn::tensor::Device as BurnDevice;

    use super::*;
    use crate::models::laya::LayaConfig;
    use crate::models::modernbert::ModernBertConfig;
    use crate::utils::read;
    #[test]
    fn rejects_wrong_shapes_without_modifying_the_existing_model() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
        let config: LayaConfig =
            serde_json::from_slice(&read(&root.join("rl_agent_config.json")).unwrap()).unwrap();
        let mut encoder: ModernBertConfig =
            serde_json::from_slice(&read(&root.join("encoder/config.json")).unwrap()).unwrap();
        encoder.hidden_size = 64;
        let mut model = LayaDecisionModel::init(&config, &encoder, &BurnDevice::flex()).unwrap();
        let before = model.type_emb.weight.id;
        let error = load_laya(&mut model, &root.join("model.safetensors")).unwrap_err();
        assert!(error.to_string().contains("Shape mismatch"));
        assert_eq!(model.type_emb.weight.id, before);
        assert_eq!(model.type_emb.weight.shape().dims::<2>(), [3, 64]);
    }
}
