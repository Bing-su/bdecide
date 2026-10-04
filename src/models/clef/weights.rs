use super::ClefDecisionModel;
use crate::{Error, Result, utils::read_checkpoint_json};
use burn::{
    module::ParamId,
    store::{ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore, TensorSnapshot},
    tensor::backend::Backend,
};
use burn_std::DType;
use camino::Utf8Path;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct WeightIndex {
    weight_map: BTreeMap<String, String>,
}

pub(crate) fn backbone_files(root: &Utf8Path) -> Result<Vec<String>> {
    if !root.join("model.safetensors.index.json").is_file() {
        return Ok(vec!["model.safetensors".into()]);
    }
    let index: WeightIndex = read_checkpoint_json(&root.join("model.safetensors.index.json"))?;
    let mut files = BTreeSet::new();
    for (name, file) in index.weight_map {
        // Do not let an external index escape the snapshot, e.g. ../../weights.
        if file.contains(['/', '\\']) || !file.ends_with(".safetensors") || file == ".safetensors" {
            return Err(Error::InvalidCheckpoint(format!(
                "invalid safetensors shard path: {file}"
            )));
        }
        if name == "lm_head.weight" || name.starts_with("model.language_model.") {
            files.insert(file);
        }
    }
    if files.is_empty() {
        return Err(Error::InvalidCheckpoint(
            "weight index contains no Clef backbone tensors".into(),
        ));
    }
    Ok(files.into_iter().collect())
}

pub(super) fn load_clef<B: Backend>(
    model: &mut ClefDecisionModel<B>,
    root: &Utf8Path,
) -> Result<()> {
    let mut files = backbone_files(root)?;
    files.push("joint_head.safetensors".into());
    let mut candidate = model.clone();
    let mut seen = BTreeSet::new();
    let mut required = BTreeSet::new();
    for file in files {
        let head = file == "joint_head.safetensors";
        let mut store = SafetensorsStore::from_file(root.join(&file));
        let snapshots = store
            .get_all_snapshots()
            .map_err(|e| Error::Weights(e.to_string()))?;
        let mut converted = Vec::new();
        for (source_name, snapshot) in snapshots {
            let name = if head {
                format!("head.{source_name}")
            } else if source_name == "lm_head.weight" {
                "output_embeddings.weight".into()
            } else if let Some(name) = source_name.strip_prefix("model.language_model.") {
                format!("language_model.{name}")
            } else if source_name.starts_with("model.visual.") {
                continue;
            } else {
                return Err(Error::Weights(format!(
                    "unexpected backbone tensor: {source_name}"
                )));
            };
            let name = remap(name);
            if !seen.insert(name.clone()) {
                return Err(Error::Weights(format!("duplicate tensor: {name}")));
            }
            if !matches!(snapshot.dtype, DType::F32 | DType::F16 | DType::BF16) {
                return Err(Error::Weights(format!(
                    "{name}: unsupported dtype {:?}",
                    snapshot.dtype
                )));
            }
            let mut data = snapshot
                .to_data()
                .map_err(|e| Error::Weights(e.to_string()))?
                .convert::<f32>();
            if data
                .as_slice::<f32>()
                .map_err(|e| Error::Weights(e.to_string()))?
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(Error::Weights(format!("{name}: non-finite weights")));
            }
            // Burn parameters use rank one for scalar logits, e.g. residual_gate [1].
            if data.shape.is_empty() {
                data.shape = burn::tensor::Shape::new([1]);
            }
            converted.push(TensorSnapshot::from_data(
                data,
                name.split('.').map(str::to_owned).collect(),
                Vec::new(),
                ParamId::new(),
            ));
        }
        // Apply one shard at a time so a 27B checkpoint need not hold every converted
        // shard in host memory. Verify coverage across all shards before replacing self.
        let applied = candidate.apply(converted, None, Some(Box::new(PyTorchToBurnAdapter)), false);
        if !applied.errors.is_empty() || !applied.unused.is_empty() {
            return Err(Error::Weights(applied.to_string()));
        }
        required.extend(applied.missing.into_iter().map(|(name, _)| name));
        required.extend(applied.applied);
    }
    let missing: Vec<_> = required.difference(&seen).collect();
    if !missing.is_empty() {
        return Err(Error::Weights(format!("missing Clef tensors: {missing:?}")));
    }
    *model = candidate;
    Ok(())
}

fn remap(name: String) -> String {
    // Keep idiomatic Rust naming while reading Transformers' parameter, e.g. A_log.
    if let Some(prefix) = name.strip_suffix(".A_log") {
        return format!("{prefix}.a_log");
    }
    if let Some(prefix) = name.strip_suffix(".in_proj_weight") {
        return format!("{prefix}.in_proj.weight");
    }
    if let Some(prefix) = name.strip_suffix(".in_proj_bias") {
        return format!("{prefix}.in_proj.bias");
    }
    if name.starts_with("head.") {
        if let Some(prefix) = name.strip_suffix(".weight").filter(|prefix| {
            prefix
                .rsplit('.')
                .next()
                .is_some_and(|part| part.contains("norm"))
        }) {
            return format!("{prefix}.gamma");
        }
        if let Some(prefix) = name.strip_suffix(".bias").filter(|prefix| {
            prefix
                .rsplit('.')
                .next()
                .is_some_and(|part| part.contains("norm"))
        }) {
            return format!("{prefix}.beta");
        }
    }
    name
}
