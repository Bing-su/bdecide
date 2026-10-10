//! Select a built-in model family while keeping backend initialization in one place.

use std::panic::{AssertUnwindSafe, catch_unwind};

use bon::{Builder, bon};
use burn::tensor::Device as BurnDevice;
use camino::Utf8Path;

#[cfg(feature = "wgpu")]
use crate::hub::Artifacts;
use crate::hub::{Family, HubOptions, ModelSource, resolve_auto};
use crate::models::clef::ClefModel;
use crate::models::d1::{D1Model, D1OmniModel};
use crate::models::decider::DeciderModel;
use crate::models::laya::LayaModel;
use crate::models::vev::VevModel;
use crate::models::von::VonModel;
use crate::models::wald::WaldModel;
use crate::{DecisionModel, Error, Metadata, Request, Response, Result};

#[derive(Debug, Clone, Copy, Default)]
pub enum Device {
    #[default]
    Cpu,
    Wgpu,
    Auto,
}

#[derive(Debug, Clone, Builder)]
pub struct LoadOptions {
    pub source: ModelSource,
    #[builder(default)]
    pub device: Device,
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
        #[cfg(not(feature = "wgpu"))]
        if matches!(options.device, Device::Wgpu) {
            return Err(Error::Device("rebuild with --features wgpu".into()));
        }
        #[cfg(not(feature = "cpu"))]
        if matches!(options.device, Device::Cpu) {
            return Err(Error::Device(
                "CPU backend is disabled; enable cpu or select wgpu".into(),
            ));
        }
        let (artifacts, family) = resolve_auto(&options.source)?;
        #[cfg(feature = "wgpu")]
        if matches!(options.device, Device::Wgpu | Device::Auto)
            && let Some(model) = Self::load_wgpu(&artifacts, family, options.device)?
        {
            return Ok(model);
        }
        #[cfg(feature = "cpu")]
        {
            let mut metadata = artifacts.metadata;
            metadata.device = "cpu".into();
            Self::load(&artifacts.root, &BurnDevice::flex(), metadata, family)
        }
        #[cfg(not(feature = "cpu"))]
        {
            let _ = (artifacts, family);
            Err(Error::Device(
                "CPU backend is disabled; enable cpu or select wgpu".into(),
            ))
        }
    }

    // Return None only when Auto may fall back to CPU; an explicitly requested
    // wgpu device must report its failure, e.g. a missing adapter or allocation panic.
    #[cfg(feature = "wgpu")]
    fn load_wgpu(artifacts: &Artifacts, family: Family, requested: Device) -> Result<Option<Self>> {
        // Construct and probe inside the guard, e.g. a missing adapter must allow Auto fallback.
        let device = match catch_unwind(|| {
            let device = BurnDevice::wgpu(Default::default());
            device.sync().map(|()| device)
        }) {
            Ok(Ok(device)) => device,
            _ if matches!(requested, Device::Wgpu) => {
                return Err(Error::Device("wgpu adapter initialization failed".into()));
            }
            _ => return Ok(None),
        };
        let mut metadata = artifacts.metadata.clone();
        metadata.device = "wgpu".into();
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            Self::load(&artifacts.root, &device, metadata, family)
        }));
        match loaded {
            Ok(Ok(model)) => Ok(Some(model)),
            Ok(Err(error)) => Err(error),
            Err(_) if matches!(requested, Device::Wgpu) => Err(Error::Device(
                "wgpu could not allocate this checkpoint".into(),
            )),
            Err(_) => Ok(None),
        }
    }

    fn load(
        root: &Utf8Path,
        device: &BurnDevice,
        metadata: Metadata,
        family: Family,
    ) -> Result<Self> {
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
