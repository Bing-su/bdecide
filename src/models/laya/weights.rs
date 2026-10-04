//! Handle checkpoint precision and remaining PyTorch/Burn parameter format differences.
use super::modeling_laya::LayaDecisionModel;
use crate::{Error, Result};
use burn::{
    module::ParamId,
    store::{ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore, TensorSnapshot},
    tensor::backend::Backend,
};
use burn_std::DType;
use camino::Utf8Path;

/// Apply a Laya PyTorch checkpoint to a separately constructed Burn architecture.
///
/// Burn reads safetensors and converts floating point storage to FP32. Missing,
/// unexpected, or incorrectly shaped tensors fail instead of retaining random parameters.
/// For example: `load_laya(&mut architecture, Utf8Path::new("model.safetensors"))?`.
pub fn load_laya<B: Backend>(model: &mut LayaDecisionModel<B>, path: &Utf8Path) -> Result<()> {
    let mut store = SafetensorsStore::from_file(path)
        // Keep native Linear storage: e.g. self_attn.in_proj_weight becomes self_attn.qkv.weight.
        .with_key_remapping(
            r"^(head\.layers\.\d+\.self_attn)\.in_proj_weight$",
            "$1.qkv.weight",
        )
        .with_key_remapping(
            r"^(head\.layers\.\d+\.self_attn)\.in_proj_bias$",
            "$1.qkv.bias",
        )
        // Explicit norm aliases preserve strict unused checks, e.g. scorer.0.weight -> gamma.
        .with_key_remapping(r"^(.+(?:_norm|\.norm[12]?)|scorer\.0)\.weight$", "$1.gamma")
        .with_key_remapping(r"^(.+(?:_norm|\.norm[12]?)|scorer\.0)\.bias$", "$1.beta");
    let snapshots = store
        .get_all_snapshots()
        .map_err(|e| Error::Weights(e.to_string()))?;
    let mut converted = Vec::with_capacity(snapshots.len());
    for (name, snapshot) in snapshots {
        if !matches!(snapshot.dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::Weights(format!(
                "{name}: unsupported dtype {:?}",
                snapshot.dtype
            )));
        }
        let data = snapshot
            .to_data()
            .map_err(|e| Error::Weights(e.to_string()))?
            .convert::<f32>();
        if data
            .as_slice::<f32>()
            .map_err(|e| Error::Weights(e.to_string()))?
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(Error::Weights(format!("{name}: non-finite weights")));
        }
        converted.push(TensorSnapshot::from_data(
            data,
            name.split('.').map(str::to_owned).collect(),
            Vec::new(),
            ParamId::new(),
        ));
    }
    // Burn handles Linear transposition, including the fused head QKV projection.
    // Only replace the caller's parameters after the complete checkpoint passes.
    let mut candidate = model.clone();
    let applied = candidate.apply(converted, None, Some(Box::new(PyTorchToBurnAdapter)), false);
    if !applied.is_success() || !applied.missing.is_empty() || !applied.unused.is_empty() {
        return Err(Error::Weights(applied.to_string()));
    }
    *model = candidate;
    Ok(())
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::models::{laya::LayaConfig, modernbert::ModernBertConfig};
    use burn::backend::Flex;
    #[test]
    fn rejects_wrong_shapes_without_modifying_the_existing_model() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
        let config: LayaConfig = serde_json::from_slice(
            &crate::utils::read(&root.join("rl_agent_config.json")).unwrap(),
        )
        .unwrap();
        let mut encoder: ModernBertConfig =
            serde_json::from_slice(&crate::utils::read(&root.join("encoder/config.json")).unwrap())
                .unwrap();
        encoder.hidden_size = 64;
        let mut model =
            LayaDecisionModel::<Flex>::init(&config, &encoder, &Default::default()).unwrap();
        let before = model.type_emb.weight.id;
        let error = load_laya(&mut model, &root.join("model.safetensors")).unwrap_err();
        assert!(error.to_string().contains("Shape mismatch"));
        assert_eq!(model.type_emb.weight.id, before);
        assert_eq!(model.type_emb.weight.shape().dims::<2>(), [3, 64]);
    }
}
