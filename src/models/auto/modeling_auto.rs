//! Select a built-in model family while keeping backend initialization in one place.

use std::panic::{AssertUnwindSafe, catch_unwind};

use bon::{Builder, bon};
use burn::tensor::Device;
use camino::Utf8Path;

use crate::hub::{Artifacts, Family, HubOptions, ModelSource, resolve_auto};
use crate::models::clef::ClefModel;
use crate::models::d1::{D1Model, D1OmniModel};
use crate::models::decider::DeciderModel;
use crate::models::laya::LayaModel;
use crate::models::vev::VevModel;
use crate::models::von::VonModel;
use crate::models::wald::WaldModel;
use crate::{DecisionModel, Error, Metadata, Request, Response, Result};

#[derive(Debug, Clone, Builder)]
pub struct LoadOptions {
    pub source: ModelSource,
    /// Use this device unchanged, e.g. `Some(Device::cuda(0))`; `None` uses Burn's default.
    pub device: Option<Device>,
}

impl LoadOptions {
    /// Configure Hub loading, e.g. `LoadOptions::new("convaiinnovations/laya")`.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self::builder()
            .source(ModelSource::Hub(HubOptions::new(repo_id)))
            .build()
    }
}

/// Load a supported architecture from its artifacts, independent of repo names.
pub struct AutoModel {
    model: Box<dyn DecisionModel>,
}
#[bon]
impl AutoModel {
    /// Load and validate a checkpoint, e.g. `AutoModel::new(LoadOptions::new("repo/model"))?`.
    #[builder(start_fn = builder)]
    pub fn new(options: LoadOptions) -> Result<Self> {
        Self::from_pretrained(options)
    }

    /// Load pretrained weights once, e.g. `AutoModel::from_pretrained(options)?`.
    pub fn from_pretrained(options: LoadOptions) -> Result<Self> {
        let (artifacts, family) = resolve_auto(&options.source)?;
        if let Some(device) = options.device {
            return Self::load_on_device(&artifacts, family, &device);
        }
        // Honor downstream Burn features, e.g. CUDA enabled without any bdecide backend feature.
        let loaded = catch_unwind(Device::default)
            .map_err(|_panic| Error::Device("default device initialization failed".into()))
            .and_then(|device| Self::load_on_device(&artifacts, family, &device));
        #[cfg(feature = "cpu")]
        if matches!(loaded, Err(Error::Device(_))) {
            // Only backend failures permit fallback, e.g. malformed checkpoints still fail.
            return Self::load_on_device(&artifacts, family, &Device::flex());
        }
        loaded
    }

    fn load_on_device(artifacts: &Artifacts, family: Family, device: &Device) -> Result<Self> {
        // Probe the supplied runtime under the guard, e.g. an unavailable adapter stays recoverable.
        catch_unwind(AssertUnwindSafe(|| device.sync()))
            .map_err(|_panic| Error::Device("adapter initialization failed".into()))?
            .map_err(|error| Error::Device(error.to_string()))?;
        catch_unwind(AssertUnwindSafe(|| {
            Self::load(&artifacts.root, device, artifacts.metadata.clone(), family)
        }))
        .map_err(|_panic| Error::Device("device could not allocate this checkpoint".into()))?
    }

    fn load(root: &Utf8Path, device: &Device, metadata: Metadata, family: Family) -> Result<Self> {
        let model: Box<dyn DecisionModel> = match family {
            Family::Laya => Box::new(LayaModel::load(root, device, metadata)?),
            Family::Clef => Box::new(ClefModel::load(root, device, metadata)?),
            Family::Vev => Box::new(VevModel::load(root, device, metadata)?),
            Family::Wald => Box::new(WaldModel::load(root, device, metadata)?),
            Family::Decider => Box::new(DeciderModel::load(root, device, metadata)?),
            Family::Von => Box::new(VonModel::load(root, device, metadata)?),
            Family::D1 => Box::new(D1Model::load(root, device, metadata)?),
            Family::D1Omni => Box::new(D1OmniModel::load(root, device, metadata)?),
        };
        Ok(Self { model })
    }
}

impl DecisionModel for AutoModel {
    fn system_one(&self, request: &Request) -> Result<Response> {
        // Backend errors may be panics (e.g. device loss). Keep JSONL processing
        // recoverable instead of letting one backend failure terminate the CLI.
        catch_unwind(AssertUnwindSafe(|| self.model.system_one(request))).map_err(|_panic| {
            Error::Inference("Burn backend failed while executing the request".into())
        })?
    }

    fn metadata(&self) -> &Metadata {
        self.model.metadata()
    }
}
