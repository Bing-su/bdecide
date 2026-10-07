//! Select Qwen text tensors and resolve Transformers' tied embedding aliases.
use burn::store::{ModuleStore, SafetensorsStore};
use camino::Utf8Path;

use super::Qwen3_5ForCausalLM;
use crate::models::weights::{self, snapshot_data};
use crate::{Error, Result};

pub(crate) fn load_causal_lm(model: &mut Qwen3_5ForCausalLM, root: &Utf8Path) -> Result<()> {
    let files = backbone_files(root)?;
    let embeddings = embedding_weights(root, &files, model.config.tie_word_embeddings)?;
    let aliases: &[(&str, &str)] = if embeddings.is_tied() {
        &[("lm_head.weight", "model.embed_tokens.weight")]
    } else {
        &[]
    };
    weights::load(model, root, &files, aliases, |name| {
        embeddings.map_name(name)
    })?;
    // Keep unequal stored aliases independent, e.g. a converted Transformers checkpoint.
    if embeddings.is_tied() {
        model.tie_weights();
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum EmbeddingWeights {
    Untied,
    Input,
    Output,
}

impl EmbeddingWeights {
    fn is_tied(self) -> bool {
        !matches!(self, Self::Untied)
    }

    fn map_name(self, source: &str) -> Result<Option<String>> {
        let name = backbone_name(source)?;
        match (self, name.as_deref()) {
            (Self::Input, Some("lm_head.weight")) => Ok(None),
            (Self::Output, Some("lm_head.weight")) => Ok(Some("model.embed_tokens.weight".into())),
            _ => Ok(name),
        }
    }
}

fn embedding_weights(root: &Utf8Path, files: &[String], tied: bool) -> Result<EmbeddingWeights> {
    if !tied {
        return Ok(EmbeddingWeights::Untied);
    }
    let mut input = None;
    let mut output = None;
    for file in files {
        let mut store = SafetensorsStore::from_file(root.join(file));
        let snapshots = store
            .get_all_tensors()
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
                if input.shape() == output.shape()
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

pub(crate) fn backbone_files(root: &Utf8Path) -> Result<Vec<String>> {
    weights::checkpoint_files(root, |name| {
        name == "lm_head.weight" || name.starts_with("model.") && !name.starts_with("model.visual.")
    })
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
    Ok(Some(name))
}
