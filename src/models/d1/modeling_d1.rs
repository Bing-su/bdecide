//! Wrap LFM2-VL with D1's decision prompts and verbalizers, e.g. Choice and Noul.
use burn::tensor::Device;
use camino::Utf8Path;
use indexmap::IndexMap;

use super::processing_d1::Processor;
use super::readout::{answer, values};
use crate::hub::{self, ModelSource};
use crate::models::lfm2_vl::{Lfm2VlForConditionalGeneration, vision};
use crate::{DecisionModel, Error, Metadata, Request, Response, Result, Usage};

/// D1-3B's decision API over LFM2-VL, e.g. text and image requests.
pub struct D1Model {
    network: Lfm2VlForConditionalGeneration,
    processor: Processor,
    device: Device,
    metadata: Metadata,
}

impl D1Model {
    /// Resolve D1-3B's artifacts, e.g. a local LFM2-VL checkpoint.
    pub fn from_pretrained(source: &ModelSource, device: &Device) -> Result<Self> {
        let artifacts = hub::resolve_d1(source)?;
        Self::load(&artifacts.root, device, artifacts.metadata)
    }

    pub(crate) fn load(root: &Utf8Path, device: &Device, mut metadata: Metadata) -> Result<Self> {
        let network = Lfm2VlForConditionalGeneration::from_pretrained(root, device)?;
        let processor = Processor::load_causal(root, &network.model.config)?;
        metadata.architecture = "d1".into();
        if metadata.device.is_empty() {
            metadata.device = format!("{device:?}");
        }
        Ok(Self {
            network,
            processor,
            device: device.clone(),
            metadata,
        })
    }
}

impl DecisionModel for D1Model {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn system_one(&self, request: &Request) -> Result<Response> {
        let limit = self.processor.validate(
            self.network
                .model
                .config
                .text_config
                .max_position_embeddings,
            request,
        )?;
        if request.audio.is_some() {
            return Err(Error::InvalidRequest("d1-3B does not accept audio".into()));
        }
        let images = vision::preprocess(&request.images, true)?;
        let media =
            (!images.is_empty()).then(|| self.network.model.encode_images(&images, &self.device));
        let mut usage = Usage::default();
        let mut answers = IndexMap::new();
        for (id, q) in &request.questions {
            let row = self
                .processor
                .causal(request, q, &images, limit, id, &mut usage)?;
            let ids: Vec<_> = row.groups.iter().flatten().copied().collect();
            let logits = values(self.network.forward_selected(
                &row.ids,
                media.as_ref(),
                &ids,
                &self.device,
            )?)?;
            let mut logits = logits.into_iter();
            let pooled = row
                .groups
                .iter()
                .map(|group| {
                    logits
                        .by_ref()
                        .take(group.len())
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect::<Vec<_>>();
            answers.insert(id.clone(), answer(q, &pooled, 0)?);
        }
        Ok(Response {
            model: self.metadata.model_id.clone(),
            answers,
            usage,
            metadata: self.metadata.clone(),
        })
    }
}
