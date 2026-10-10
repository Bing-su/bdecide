//! Select each architecture's required files, e.g. Clef's indexed weight shards.
use hf_hub::HFError;

use super::{Artifacts, ModelSource, resolve};
use crate::models::clef::ClefConfig;
use crate::models::d1::D1OmniConfig;
use crate::models::decider::DeciderConfig;
use crate::models::laya::REQUIRED_ARTIFACTS;
use crate::models::lfm2_vl::Lfm2VlConfig;
use crate::models::modernbert::ModernBertConfig;
use crate::models::qwen3_5::weights::backbone_files;
use crate::models::qwen3_5::{Qwen3_5Config, Qwen3_5TextConfig};
use crate::models::vev::VevConfig;
use crate::models::von::VonConfig;
use crate::models::wald::WaldConfig;
use crate::utils::read_checkpoint_json;
use crate::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) enum Family {
    Laya,
    Clef,
    Vev,
    Wald,
    Decider,
    Von,
    D1,
    D1Omni,
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
            let config: serde_json::Value =
                read_checkpoint_json(&artifacts.root.join("config.json"))?;
            let d1_family = match config.get("model_type").and_then(serde_json::Value::as_str) {
                Some("lfm2_vl") => Some(Family::D1),
                Some("d1_omni") => Some(Family::D1Omni),
                _ => None,
            };
            if let Some(family) = d1_family {
                resolve_family(source, &artifacts, family)?;
                return Ok((artifacts, family));
            }
            if config.get("model_type").and_then(serde_json::Value::as_str) == Some("modernbert") {
                resolve_more(source, &artifacts, &["marker_calibration.json"])?;
                resolve_family(source, &artifacts, Family::Von)?;
                return Ok((artifacts, Family::Von));
            }
            Qwen3_5TextConfig::from_pretrained(&artifacts.root)?;
            let family = match resolve_more(source, &artifacts, &["joint_head_config.json"]) {
                Ok(()) => Family::Clef,
                Err(error) if artifact_absent(&error) => {
                    match resolve_more(source, &artifacts, &["vev.json"]) {
                        Ok(()) => Family::Vev,
                        Err(error) if artifact_absent(&error) => {
                            match resolve_more(source, &artifacts, &["serving.json"]) {
                                Ok(()) => Family::Wald,
                                Err(error) if artifact_absent(&error) => {
                                    resolve_more(source, &artifacts, &["decider_config.json"])?;
                                    Family::Decider
                                }
                                Err(error) => return Err(error),
                            }
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

pub(crate) fn resolve_decider(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    resolve_more(source, &artifacts, &["decider_config.json"])?;
    resolve_family(source, &artifacts, Family::Decider)?;
    Ok(artifacts)
}

pub(crate) fn resolve_von(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    resolve_more(source, &artifacts, &["marker_calibration.json"])?;
    resolve_family(source, &artifacts, Family::Von)?;
    Ok(artifacts)
}

pub(crate) fn resolve_d1(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    resolve_family(source, &artifacts, Family::D1)?;
    Ok(artifacts)
}

pub(crate) fn resolve_d1_omni(source: &ModelSource) -> Result<Artifacts> {
    let artifacts = resolve(source, &["config.json"])?;
    resolve_family(source, &artifacts, Family::D1Omni)?;
    Ok(artifacts)
}

fn resolve_family(source: &ModelSource, artifacts: &Artifacts, family: Family) -> Result<()> {
    if matches!(family, Family::D1 | Family::D1Omni) {
        match family {
            Family::D1 => {
                Lfm2VlConfig::from_pretrained(&artifacts.root)?;
            }
            _ => {
                D1OmniConfig::from_pretrained(&artifacts.root)?;
            }
        }
        resolve_more(
            source,
            artifacts,
            &["tokenizer.json", "tokenizer_config.json"],
        )?;
        match resolve_more(source, artifacts, &["model.safetensors.index.json"]) {
            Ok(()) => {
                let files = crate::models::weights::checkpoint_files(&artifacts.root, |_| true)?;
                let names: Vec<_> = files.iter().map(String::as_str).collect();
                return resolve_more(source, artifacts, &names);
            }
            Err(error) if artifact_absent(&error) => {
                return resolve_more(source, artifacts, &["model.safetensors"]);
            }
            Err(error) => return Err(error),
        }
    }
    if matches!(family, Family::Von) {
        let encoder: ModernBertConfig = read_checkpoint_json(&artifacts.root.join("config.json"))?;
        encoder.validate()?;
        let calibration: VonConfig =
            read_checkpoint_json(&artifacts.root.join("marker_calibration.json"))?;
        calibration.validate()?;
        return resolve_more(
            source,
            artifacts,
            &[
                "option_marker.pt",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        );
    }
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
        Family::Decider => {
            let config: DeciderConfig =
                read_checkpoint_json(&artifacts.root.join("decider_config.json"))?;
            config.validate()?;
        }
        Family::Von => {
            return Err(Error::UnsupportedModel(
                "Von requires its Option-Marker artifact layout".into(),
            ));
        }
        Family::Laya => {
            return Err(Error::UnsupportedModel(
                "Laya requires its fixed artifact layout".into(),
            ));
        }
        Family::D1 | Family::D1Omni => {
            return Err(Error::UnsupportedModel(
                "d1 requires its LFM2 artifact layout".into(),
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
