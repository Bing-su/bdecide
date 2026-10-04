//! Select a built-in model family while keeping backend initialization in one place.

use crate::{
    DecisionModel, Error, Metadata, Request, Response, Result,
    hub::{self, ModelSource},
    models::laya,
};

#[derive(Debug, Clone, Copy, Default)]
pub enum Device {
    #[default]
    Cpu,
    Wgpu,
    Auto,
}

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub source: ModelSource,
    pub device: Device,
}
impl LoadOptions {
    /// Configure Hub loading, e.g. `LoadOptions::new("convaiinnovations/laya")`.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            source: ModelSource::Hub(hub::HubOptions::new(repo_id)),
            device: Device::Cpu,
        }
    }
}

/// Load a supported architecture from its artifacts, independent of repo names.
pub struct AutoModel {
    model: Box<dyn DecisionModel>,
}
impl AutoModel {
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
        let artifacts = hub::resolve(&options.source, &laya::REQUIRED_ARTIFACTS)?;
        #[cfg(feature = "wgpu")]
        if matches!(options.device, Device::Wgpu | Device::Auto)
            && let Some(model) = Self::load_wgpu(&artifacts, options.device)?
        {
            return Ok(model);
        }
        #[cfg(feature = "cpu")]
        {
            let mut metadata = artifacts.metadata;
            metadata.device = "cpu".into();
            let model = laya::LayaModel::<burn::backend::Flex>::load(
                &artifacts.root,
                &Default::default(),
                metadata,
            )?;
            Ok(Self {
                model: Box::new(model),
            })
        }
        #[cfg(not(feature = "cpu"))]
        {
            let _ = artifacts;
            Err(Error::Device(
                "CPU backend is disabled; enable cpu or select wgpu".into(),
            ))
        }
    }

    // Return None only when Auto may fall back to CPU; an explicitly requested
    // wgpu device must report its failure, e.g. a missing adapter or allocation panic.
    #[cfg(feature = "wgpu")]
    fn load_wgpu(artifacts: &hub::Artifacts, requested: Device) -> Result<Option<Self>> {
        use burn::backend::{Wgpu, wgpu::WgpuDevice};
        use burn::tensor::backend::Backend;

        let device = WgpuDevice::DefaultDevice;
        // Backend names load or reuse Burn's runtime, including a host's existing GPU.
        // Probe before allocating tensors so a missing adapter leaves no partial model.
        if std::panic::catch_unwind(|| Wgpu::<f32, i32>::name(&device)).is_err() {
            return if matches!(requested, Device::Wgpu) {
                Err(Error::Device("wgpu adapter initialization failed".into()))
            } else {
                Ok(None)
            };
        }
        let mut metadata = artifacts.metadata.clone();
        metadata.device = "wgpu".into();
        let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            laya::LayaModel::<Wgpu<f32, i32>>::load(&artifacts.root, &device, metadata)
        }));
        match loaded {
            Ok(Ok(model)) => Ok(Some(Self {
                model: Box::new(model),
            })),
            Ok(Err(error)) => Err(error),
            Err(_) if matches!(requested, Device::Wgpu) => Err(Error::Device(
                "wgpu could not allocate this checkpoint".into(),
            )),
            Err(_) => Ok(None),
        }
    }
}
impl DecisionModel for AutoModel {
    fn predict(&self, request: &Request) -> Result<Response> {
        // Backend errors may be panics (e.g. device loss). Keep JSONL processing
        // recoverable instead of letting one backend failure terminate the CLI.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.model.predict(request)))
            .map_err(|_panic| {
                Error::Inference("Burn backend failed while executing the request".into())
            })?
    }
    fn metadata(&self) -> &Metadata {
        self.model.metadata()
    }
}
