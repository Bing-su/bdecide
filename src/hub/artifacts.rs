//! Select each architecture's required files, e.g. Clef's indexed weight shards.
use hf_hub::HFError;

use super::{Artifacts, ModelSource, resolve};
use crate::models::clef::ClefConfig;
use crate::models::laya::REQUIRED_ARTIFACTS;
use crate::models::qwen3_5::weights::backbone_files;
use crate::models::qwen3_5::{Qwen3_5Config, Qwen3_5TextConfig};
use crate::models::vev::VevConfig;
use crate::models::wald::WaldConfig;
use crate::utils::read_checkpoint_json;
use crate::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) enum Family {
    Laya,
    Clef,
    Vev,
    Wald,
}

/// Detect artifact layouts rather than repo IDs, e.g. a renamed Clef fine-tune.
pub(crate) fn resolve_auto(source: &ModelSource) -> Result<(Artifacts, Family)> {
    match resolve(source, &["rl_agent_config.json"]) {
        Ok(artifacts) => {
            resolve_more(source, &artifacts, &REQUIRED_ARTIFACTS[1..])?;
            Ok((artifacts, Family::Laya))
        }
        Err(error) if artifact_absent(&error) => {
            let artifacts = resolve(source, &["config.json"])?;
            Qwen3_5TextConfig::from_pretrained(&artifacts.root)?;
            let family = match resolve_more(source, &artifacts, &["joint_head_config.json"]) {
                Ok(()) => Family::Clef,
                Err(error) if artifact_absent(&error) => {
                    match resolve_more(source, &artifacts, &["vev.json"]) {
                        Ok(()) => Family::Vev,
                        Err(error) if artifact_absent(&error) => {
                            resolve_more(source, &artifacts, &["serving.json"])?;
                            Family::Wald
                        }
                        Err(error) => return Err(error),
                    }
                }
                Err(error) => return Err(error),
            };
            resolve_family(source, &artifacts, family)?;
            Ok((artifacts, family))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn resolve_clef(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    resolve_more(source, &artifacts, &["joint_head_config.json"])?;
    resolve_family(source, &artifacts, Family::Clef)?;
    Ok(artifacts)
}

pub(crate) fn resolve_vev(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    resolve_more(source, &artifacts, &["vev.json"])?;
    resolve_family(source, &artifacts, Family::Vev)?;
    Ok(artifacts)
}

pub(crate) fn resolve_wald(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    resolve_more(source, &artifacts, &["serving.json"])?;
    resolve_family(source, &artifacts, Family::Wald)?;
    Ok(artifacts)
}

fn resolve_family(source: &ModelSource, artifacts: &Artifacts, family: Family) -> Result<()> {
    Qwen3_5TextConfig::from_pretrained(&artifacts.root)?;
    match family {
        Family::Clef => {
            let config = Qwen3_5Config::from_pretrained(&artifacts.root)?;
            let head: ClefConfig =
                read_checkpoint_json(&artifacts.root.join("joint_head_config.json"))?;
            head.validate(&config)?;
            resolve_more(source, artifacts, &["joint_head.safetensors"])?;
        }
        Family::Vev => {
            let config: VevConfig = read_checkpoint_json(&artifacts.root.join("vev.json"))?;
            config.validate()?;
        }
        Family::Wald => {
            let config: WaldConfig = read_checkpoint_json(&artifacts.root.join("serving.json"))?;
            config.validate()?;
            resolve_more(source, artifacts, &["temperature.json"])?;
        }
        Family::Laya => {
            return Err(Error::UnsupportedModel(
                "Laya requires its fixed artifact layout".into(),
            ));
        }
    }
    resolve_more(
        source,
        artifacts,
        &["tokenizer.json", "tokenizer_config.json"],
    )?;
    match resolve_more(source, artifacts, &["model.safetensors.index.json"]) {
        Ok(()) => {}
        Err(error) if artifact_absent(&error) => {
            resolve_more(source, artifacts, &["model.safetensors"])?;
            return Ok(());
        }
        Err(error) => return Err(error),
    }
    let files = backbone_files(&artifacts.root)?;
    let names: Vec<&str> = files.iter().map(String::as_str).collect();
    resolve_more(source, artifacts, &names)
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
            HFError::EntryNotFound { .. } | HFError::LocalEntryNotFound { .. } => true,
            HFError::Http { context } => context.status.as_u16() == 404,
            _ => false,
        },
        _ => false,
    }
}
