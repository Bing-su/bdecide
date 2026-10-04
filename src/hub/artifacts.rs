//! Select each architecture's required files, e.g. Clef's indexed weight shards.
use super::{Artifacts, ModelSource, resolve};
use crate::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) enum Family {
    Laya,
    Clef,
}

/// Detect artifact layouts rather than repo IDs, e.g. a renamed Clef fine-tune.
pub(crate) fn resolve_auto(source: &ModelSource) -> Result<(Artifacts, Family)> {
    match resolve(source, &["rl_agent_config.json"]) {
        Ok(artifacts) => {
            resolve_more(
                source,
                &artifacts,
                &crate::models::laya::REQUIRED_ARTIFACTS[1..],
            )?;
            Ok((artifacts, Family::Laya))
        }
        Err(error) if artifact_absent(&error) => Ok((resolve_clef(source)?, Family::Clef)),
        Err(error) => Err(error),
    }
}

pub(crate) fn resolve_clef(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    let config: crate::models::qwen3_5::Qwen3_5Config =
        crate::utils::read_checkpoint_json(&artifacts.root.join("config.json"))?;
    config.validate()?;
    resolve_more(source, &artifacts, &["joint_head_config.json"])?;
    let head: crate::models::clef::ClefConfig =
        crate::utils::read_checkpoint_json(&artifacts.root.join("joint_head_config.json"))?;
    head.validate(&config)?;
    resolve_more(
        source,
        &artifacts,
        &[
            "joint_head.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )?;
    match resolve_more(source, &artifacts, &["model.safetensors.index.json"]) {
        Ok(()) => {}
        Err(error) if artifact_absent(&error) => {
            resolve_more(source, &artifacts, &["model.safetensors"])?;
            return Ok(artifacts);
        }
        Err(error) => return Err(error),
    }
    let files = crate::models::clef::weights::backbone_files(&artifacts.root)?;
    let names: Vec<&str> = files.iter().map(String::as_str).collect();
    resolve_more(source, &artifacts, &names)?;
    Ok(artifacts)
}

// Preserve the first config's commit and caller metadata across dynamic downloads,
// e.g. every shard referenced by an index must come from the same snapshot.
fn resolve_more(source: &ModelSource, artifacts: &Artifacts, required: &[&str]) -> Result<()> {
    let pinned = match source {
        ModelSource::Local(root) => ModelSource::Local(root.clone()),
        ModelSource::Hub(options) => {
            let mut options = options.clone();
            options.revision = artifacts.metadata.commit_sha.clone();
            ModelSource::Hub(options)
        }
    };
    resolve(&pinned, required)?;
    Ok(())
}

fn artifact_absent(error: &Error) -> bool {
    match error {
        Error::MissingArtifact(_) => true,
        Error::Hub(error) => match error.as_ref() {
            hf_hub::HFError::EntryNotFound { .. } | hf_hub::HFError::LocalEntryNotFound { .. } => {
                true
            }
            hf_hub::HFError::Http { context } => context.status.as_u16() == 404,
            _ => false,
        },
        _ => false,
    }
}
