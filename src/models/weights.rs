//! Load PyTorch safetensors transactionally through Burn's snapshot and adapter APIs.
use crate::{Error, Result, utils::read_checkpoint_json};
use burn::{
    module::{ModuleVisitor, Param, ParamId},
    store::{
        ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore,
        TensorSnapshot,
    },
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use burn_std::DType;
use camino::Utf8Path;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct WeightIndex {
    weight_map: BTreeMap<String, String>,
}

pub(crate) fn checkpoint_files(
    root: &Utf8Path,
    include: impl Fn(&str) -> bool,
) -> Result<Vec<String>> {
    if !root.join("model.safetensors.index.json").is_file() {
        return Ok(vec!["model.safetensors".into()]);
    }
    let index: WeightIndex = read_checkpoint_json(&root.join("model.safetensors.index.json"))?;
    let mut files = BTreeSet::new();
    for (name, file) in index.weight_map {
        // Keep external shard paths inside the snapshot, e.g. reject ../../weights.
        if file.contains(['/', '\\']) || !file.ends_with(".safetensors") || file == ".safetensors" {
            return Err(Error::InvalidCheckpoint(format!(
                "invalid safetensors shard path: {file}"
            )));
        }
        if include(&name) {
            files.insert(file);
        }
    }
    if files.is_empty() {
        return Err(Error::InvalidCheckpoint(
            "weight index contains no matching tensors".into(),
        ));
    }
    Ok(files.into_iter().collect())
}

pub(crate) fn snapshot_data(name: &str, snapshot: &TensorSnapshot) -> Result<TensorData> {
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

pub(crate) fn identity_name(name: &str) -> Result<Option<String>> {
    Ok(Some(name.into()))
}

// Inspect paths without calling Param::val(), e.g. keep a 27B architecture lazy
// until its pretrained tensors replace the initializers. Burn's collect() initializes them.
#[derive(Default)]
struct CheckpointPaths {
    stack: Vec<(String, String)>,
    aliases: BTreeMap<String, String>,
    required: BTreeSet<String>,
}

impl CheckpointPaths {
    fn parameter(&mut self) {
        let name = self
            .stack
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(".");
        self.required.insert(name.clone());
        let Some((parameter, module_type)) = self.stack.last() else {
            return;
        };
        let prefix = name.rsplit_once('.').map_or("", |(prefix, _)| prefix);
        if let Some(alias) = PyTorchToBurnAdapter.get_alternative_param_name(parameter, module_type)
        {
            // Burn 0.21 reports norm aliases as unused; canonicalize using its
            // module metadata so strict checks still work, e.g. norm.weight -> gamma.
            let source = if prefix.is_empty() {
                alias
            } else {
                format!("{prefix}.{alias}")
            };
            self.aliases.insert(source, name.clone());
        }
        if module_type == "Struct:Linear" && (prefix == "in_proj" || prefix.ends_with(".in_proj")) {
            // PyTorch fuses attention parameters outside a Linear submodule,
            // e.g. self_attn.in_proj_weight; retain Burn's native transposition.
            self.aliases.insert(format!("{prefix}_{parameter}"), name);
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for CheckpointPaths {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.stack.push((name.into(), container_type.into()));
    }

    fn exit_module(&mut self, _: &str, _: &str) {
        self.stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, _: &Param<Tensor<B, D>>) {
        self.parameter();
    }

    fn visit_int<const D: usize>(&mut self, _: &Param<Tensor<B, D, Int>>) {
        self.parameter();
    }

    fn visit_bool<const D: usize>(&mut self, _: &Param<Tensor<B, D, Bool>>) {
        self.parameter();
    }
}

pub(crate) fn load<B: Backend, M: ModuleSnapshot<B>>(
    model: &mut M,
    root: &Utf8Path,
    files: &[String],
    tied_weights: &[(&str, &str)],
    map_name: impl Fn(&str) -> Result<Option<String>>,
) -> Result<()> {
    let mut paths = CheckpointPaths::default();
    model.visit(&mut paths);
    let mut candidate = model.clone();
    let mut seen = BTreeSet::new();
    for file in files {
        let mut store = SafetensorsStore::from_file(root.join(file));
        let snapshots = store
            .get_all_snapshots()
            .map_err(|error| Error::Weights(error.to_string()))?;
        let mut converted = Vec::with_capacity(snapshots.len());
        for (source, snapshot) in snapshots {
            let Some(name) = map_name(source)? else {
                continue;
            };
            let name = paths.aliases.get(&name).cloned().unwrap_or(name);
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
        // Convert and apply one shard at a time, e.g. avoid retaining all 27B shards.
        let applied = candidate.apply(converted, None, Some(Box::new(PyTorchToBurnAdapter)), false);
        if !applied.errors.is_empty() || !applied.unused.is_empty() {
            return Err(Error::Weights(applied.to_string()));
        }
    }
    // Only a loaded source satisfies a tied alias, e.g. lm_head -> embed_tokens.
    let missing: Vec<_> = paths
        .required
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
    // Commit only after every shard passes dtype, shape and coverage validation.
    *model = candidate;
    Ok(())
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::{
        backend::Flex,
        module::Module,
        nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
        store::BurnToPyTorchAdapter,
    };

    #[derive(Module, Debug)]
    struct Projections<B: Backend> {
        // Use an arbitrary norm name to prove aliases follow Burn's type, e.g. scale.weight.
        scale: LayerNorm<B>,
        in_proj: Linear<B>,
    }

    #[test]
    fn strict_loading_uses_burn_types_and_keeps_initializers_lazy() {
        let device = Default::default();
        let source = Projections::<Flex> {
            scale: LayerNormConfig::new(2).init(&device),
            in_proj: LinearConfig::new(2, 6).init(&device),
        };
        let directory = tempfile::tempdir().unwrap();
        let root = Utf8Path::from_path(directory.path()).unwrap();
        // Export real PyTorch names and layout, e.g. fused in_proj_weight [6, 2].
        source
            .save_into(
                &mut SafetensorsStore::from_file(root.join("model.safetensors"))
                    .with_to_adapter(BurnToPyTorchAdapter)
                    .with_key_remapping(r"^in_proj\.(weight|bias)$", "in_proj_$1"),
            )
            .unwrap();
        source
            .save_into(
                &mut SafetensorsStore::from_file(root.join("duplicate.safetensors"))
                    .with_full_path("scale.gamma"),
            )
            .unwrap();
        source
            .save_into(
                &mut SafetensorsStore::from_file(root.join("missing.safetensors"))
                    .with_full_path("scale.gamma")
                    .with_to_adapter(BurnToPyTorchAdapter),
            )
            .unwrap();
        source
            .save_into(
                &mut SafetensorsStore::from_file(root.join("unused.safetensors"))
                    .with_full_path("scale.gamma")
                    .with_key_remapping(r"^scale\.gamma$", "unknown.weight"),
            )
            .unwrap();
        let mut target = Projections::<Flex> {
            scale: LayerNormConfig::new(2).init(&device),
            in_proj: LinearConfig::new(2, 6).init(&device),
        };
        let before = target.in_proj.weight.id;
        for (files, message) in [
            (
                vec!["model.safetensors".into(), "duplicate.safetensors".into()],
                "duplicate tensor",
            ),
            (vec!["missing.safetensors".into()], "missing tensors"),
            (vec!["unused.safetensors".into()], "Unused"),
        ] {
            let error = load(&mut target, root, &files, &[], identity_name).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
            assert_eq!(target.in_proj.weight.id, before);
            assert!(!target.in_proj.weight.is_initialized());
            assert!(!target.scale.gamma.is_initialized());
        }
        load(
            &mut target,
            root,
            &["model.safetensors".into()],
            &[],
            identity_name,
        )
        .unwrap();
        assert_eq!(
            target.in_proj.weight.val().into_data(),
            source.in_proj.weight.val().into_data()
        );
        assert_eq!(
            target.scale.gamma.val().into_data(),
            source.scale.gamma.val().into_data()
        );
    }
}
