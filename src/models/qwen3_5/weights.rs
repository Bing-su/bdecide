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

#[derive(Clone, Copy)]
pub(crate) enum EmbeddingWeights {
    Untied,
    Input,
    Output,
}

impl EmbeddingWeights {
    pub(crate) fn is_tied(self) -> bool {
        !matches!(self, Self::Untied)
    }

    pub(crate) fn map_name(self, source: &str) -> Result<Option<String>> {
        let name = backbone_name(source)?;
        match (self, name.as_deref()) {
            (Self::Input, Some("lm_head.weight")) => Ok(None),
            (Self::Output, Some("lm_head.weight")) => Ok(Some("model.embed_tokens.weight".into())),
            _ => Ok(name),
        }
    }
}

pub(crate) fn embedding_weights(
    root: &Utf8Path,
    files: &[String],
    tied: bool,
) -> Result<EmbeddingWeights> {
    if !tied {
        return Ok(EmbeddingWeights::Untied);
    }
    let mut input = None;
    let mut output = None;
    for file in files {
        let mut store = SafetensorsStore::from_file(root.join(file));
        let snapshots = store
            .get_all_snapshots()
            .map_err(|error| Error::Weights(error.to_string()))?;
        for (name, snapshot) in snapshots {
            let slot = match name.as_str() {
                "model.language_model.embed_tokens.weight" | "model.embed_tokens.weight" => {
                    &mut input
                }
                "lm_head.weight" => &mut output,
                _ => continue,
            };
            if slot.replace(snapshot.clone()).is_some() {
                return Err(Error::Weights(format!(
                    "duplicate embedding tensor: {name}"
                )));
            }
        }
    }
    // Transformers ties symmetrically: either stored alias suffices. If both exist
    // with different values, retain independent weights rather than discarding one.
    // For example a .bin-to-safetensors export can include both tied names.
    match (input, output) {
        (Some(input), Some(output)) => {
            let input = snapshot_data("embed_tokens.weight", &input)?;
            let output = snapshot_data("lm_head.weight", &output)?;
            Ok(
                if input.shape == output.shape
                    && input
                        .as_slice::<f32>()
                        .map_err(|error| Error::Weights(error.to_string()))?
                        == output
                            .as_slice::<f32>()
                            .map_err(|error| Error::Weights(error.to_string()))?
                {
                    EmbeddingWeights::Input
                } else {
                    EmbeddingWeights::Untied
                },
            )
        }
        (Some(_), None) => Ok(EmbeddingWeights::Input),
        (None, Some(_)) => Ok(EmbeddingWeights::Output),
        (None, None) => Err(Error::Weights("missing tied embedding weights".into())),
    }
}

fn snapshot_data(name: &str, snapshot: &TensorSnapshot) -> Result<burn::tensor::TensorData> {
    if !matches!(snapshot.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::Weights(format!(
            "{name}: unsupported dtype {:?}",
            snapshot.dtype
        )));
    }
    let data = snapshot
        .to_data()
        .map_err(|error| Error::Weights(error.to_string()))?
        .convert::<f32>();
    if data
        .as_slice::<f32>()
        .map_err(|error| Error::Weights(error.to_string()))?
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(Error::Weights(format!("{name}: non-finite weights")));
    }
    Ok(data)
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
        if name == "lm_head.weight"
            || name.starts_with("model.") && !name.starts_with("model.visual.")
        {
            files.insert(file);
        }
    }
    if files.is_empty() {
        return Err(Error::InvalidCheckpoint(
            "weight index contains no Qwen3.5 backbone tensors".into(),
        ));
    }
    Ok(files.into_iter().collect())
}

pub(crate) fn load<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    root: &Utf8Path,
    files: &[String],
    tied_weights: &[(&str, &str)],
    map_name: impl Fn(&str, &str) -> Result<Option<String>>,
) -> Result<()> {
    let mut candidate = model.clone();
    let mut seen = BTreeSet::new();
    let mut required = BTreeSet::new();
    for file in files {
        let mut store = SafetensorsStore::from_file(root.join(file));
        let snapshots = store
            .get_all_snapshots()
            .map_err(|e| Error::Weights(e.to_string()))?;
        let mut converted = Vec::new();
        for (source_name, snapshot) in snapshots {
            let Some(name) = map_name(file, source_name)? else {
                continue;
            };
            if !seen.insert(name.clone()) {
                return Err(Error::Weights(format!("duplicate tensor: {name}")));
            }
            let mut data = snapshot_data(&name, snapshot)?;
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
    // Only a successfully loaded source satisfies a missing tied alias, e.g. lm_head -> embed_tokens.
    let missing: Vec<_> = required
        .difference(&seen)
        .filter(|name| {
            !tied_weights
                .iter()
                .any(|(target, source)| name.as_str() == *target && seen.contains(*source))
        })
        .collect();
    if !missing.is_empty() {
        return Err(Error::Weights(format!("missing tensors: {missing:?}")));
    }
    *model = candidate;
    Ok(())
}

pub(crate) fn backbone_name(source: &str) -> Result<Option<String>> {
    let name = if source == "lm_head.weight" {
        source.into()
    } else if source.starts_with("model.visual.") {
        // Decision requests use text; skip only the known vision tower, e.g. Vev's visual.blocks.
        return Ok(None);
    } else if let Some(name) = source
        .strip_prefix("model.language_model.")
        .or_else(|| source.strip_prefix("model."))
    {
        format!("model.{name}")
    } else {
        return Err(Error::Weights(format!(
            "unexpected backbone tensor: {source}"
        )));
    };
    Ok(Some(remap(name)))
}

pub(crate) fn remap(name: String) -> String {
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
