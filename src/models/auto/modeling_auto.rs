//! Select a built-in model family while keeping backend initialization in one place.

use std::panic::{AssertUnwindSafe, catch_unwind};

use bon::{Builder, bon};
#[cfg(feature = "cpu")]
use burn::backend::Flex;
use burn::tensor::backend::Backend;
use camino::Utf8Path;

#[cfg(feature = "wgpu")]
use crate::hub::Artifacts;
use crate::{
    DecisionModel, Error, Metadata, Request, Response, Result,
    hub::{Family, HubOptions, ModelSource, resolve_auto},
    models::{clef::ClefModel, laya::LayaModel, vev::VevModel, wald::WaldModel},
};

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
            Self::load::<Flex>(&artifacts.root, &Default::default(), metadata, family)
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
        use burn::backend::{Wgpu, wgpu::WgpuDevice};

        let device = WgpuDevice::DefaultDevice;
        // Backend names load or reuse Burn's runtime, including a host's existing GPU.
        // Probe before allocating tensors so a missing adapter leaves no partial model.
        if catch_unwind(|| Wgpu::<f32, i32>::name(&device)).is_err() {
            return if matches!(requested, Device::Wgpu) {
                Err(Error::Device("wgpu adapter initialization failed".into()))
            } else {
                Ok(None)
            };
        }
        let mut metadata = artifacts.metadata.clone();
        metadata.device = "wgpu".into();
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            Self::load::<Wgpu<f32, i32>>(&artifacts.root, &device, metadata, family)
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

    fn load<B: Backend>(
        root: &Utf8Path,
        device: &B::Device,
        metadata: Metadata,
        family: Family,
    ) -> Result<Self> {
        let model: Box<dyn DecisionModel> = match family {
            Family::Laya => Box::new(LayaModel::<B>::load(root, device, metadata)?),
            Family::Clef => Box::new(ClefModel::<B>::load(root, device, metadata)?),
            Family::Vev => Box::new(VevModel::<B>::load(root, device, metadata)?),
            Family::Wald => Box::new(WaldModel::<B>::load(root, device, metadata)?),
        };
        Ok(Self { model })
    }
}
impl DecisionModel for AutoModel {
    fn predict(&self, request: &Request) -> Result<Response> {
        // Backend errors may be panics (e.g. device loss). Keep JSONL processing
        // recoverable instead of letting one backend failure terminate the CLI.
        catch_unwind(AssertUnwindSafe(|| self.model.predict(request))).map_err(|_panic| {
            Error::Inference("Burn backend failed while executing the request".into())
        })?
    }
    fn metadata(&self) -> &Metadata {
        self.model.metadata()
    }
}
