//! Own multimodal embedding replacement and tied logits, e.g. D1 selects answer-token rows.
use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};
use burn_std::{IndexingUpdateOp, s};
use camino::Utf8Path;

use super::Lfm2VlConfig;
use super::vision::{self, ImageCrops, Lfm2VlMultiModalProjector, Siglip2VisionModel};
use crate::models::lfm2::Lfm2Model;
use crate::models::weights;
use crate::{Error, ImageInput, Result};

#[derive(Module, Debug)]
pub struct Lfm2VlModel {
    pub(crate) language_model: Lfm2Model,
    vision_tower: Siglip2VisionModel,
    multi_modal_projector: Lfm2VlMultiModalProjector,
    #[module(skip)]
    pub config: Lfm2VlConfig,
}

#[derive(Module, Debug)]
pub struct Lfm2VlForConditionalGeneration {
    pub model: Lfm2VlModel,
}

impl Lfm2VlModel {
    /// Initialize Transformers' model.language_model and vision modules.
    pub fn new(config: &Lfm2VlConfig, device: &Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            language_model: Lfm2Model::new(&config.text_config, device)?,
            vision_tower: Siglip2VisionModel::new(&config.vision_config, device)?,
            multi_modal_projector: Lfm2VlMultiModalProjector::new(
                config.vision_config.hidden_size,
                config.projector_hidden_size,
                config.text_config.hidden_size,
                Some(config.projector_hidden_act.parse()?),
                device,
            ),
            config: config.clone(),
        })
    }

    /// Encode images in crop order, e.g. several images form one contiguous feature tensor.
    pub fn get_image_features(&self, images: &[ImageInput], device: &Device) -> Result<Tensor<3>> {
        if images.is_empty() {
            return Err(Error::InvalidRequest(
                "image features require at least one image".into(),
            ));
        }
        Ok(self.encode_images(&vision::preprocess(images, true)?, device))
    }

    pub(crate) fn encode_images(&self, images: &[ImageCrops], device: &Device) -> Tensor<3> {
        vision::encode(
            &self.vision_tower,
            &self.multi_modal_projector,
            images,
            device,
        )
    }

    /// Replace image placeholders before the causal pass, e.g. one feature per <image> token.
    pub fn forward(
        &self,
        ids: &[u32],
        image_features: Option<&Tensor<3>>,
        device: &Device,
    ) -> Result<Tensor<3>> {
        if ids.is_empty()
            || ids.len() > self.config.text_config.max_position_embeddings
            || ids
                .iter()
                .any(|&id| id as usize >= self.config.text_config.vocab_size)
        {
            return Err(Error::InvalidRequest(
                "invalid LFM2-VL token IDs or sequence length".into(),
            ));
        }
        let positions: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.config.image_token_id).then_some(i as i64))
            .collect();
        if positions.len() != image_features.map_or(0, |features| features.dims()[1]) {
            return Err(Error::InvalidRequest(
                "image placeholders do not match media features".into(),
            ));
        }
        let mut h = self.language_model.embed(ids, device);
        if let Some(features) = image_features {
            if features.dims() != [1, positions.len(), self.config.text_config.hidden_size] {
                return Err(Error::InvalidRequest(
                    "invalid LFM2-VL image feature dimensions".into(),
                ));
            }
            let length = positions.len();
            let index = Tensor::<1, Int>::from_data(TensorData::new(positions, [length]), device);
            h = h.select_assign(1, index, features.clone(), IndexingUpdateOp::Assign);
        }
        Ok(self.language_model.forward(h))
    }
}

impl Lfm2VlForConditionalGeneration {
    /// Keep the tied readout implicit: released SafeTensors omit a separate lm_head.weight.
    pub fn new(config: &Lfm2VlConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            model: Lfm2VlModel::new(config, device)?,
        })
    }

    /// Load the base independently of D1, e.g. its published model.* tensor paths.
    pub fn from_pretrained(root: &Utf8Path, device: &Device) -> Result<Self> {
        let config = Lfm2VlConfig::from_pretrained(root)?;
        let mut model = Self::new(&config, device)?;
        let files = weights::checkpoint_files(root, |_| true)?;
        weights::load(&mut model, root, &files, &[], super::weights::weight_name)?;
        Ok(model)
    }

    /// Return full vocabulary logits, e.g. [1, length, vocab] as in Transformers.forward.
    pub fn forward(
        &self,
        ids: &[u32],
        image_features: Option<&Tensor<3>>,
        device: &Device,
    ) -> Result<Tensor<3>> {
        let h = self.model.forward(ids, image_features, device)?;
        Ok(h.matmul(
            self.model
                .language_model
                .embed_tokens
                .weight
                .val()
                .transpose()
                .unsqueeze_dim::<3>(0),
        ))
    }

    /// Read only selected vocabulary rows at the last position, e.g. yes/Yes/YES for D1.
    pub fn forward_selected(
        &self,
        ids: &[u32],
        image_features: Option<&Tensor<3>>,
        output_ids: &[u32],
        device: &Device,
    ) -> Result<Tensor<3>> {
        if output_ids.is_empty()
            || output_ids
                .iter()
                .any(|&id| id as usize >= self.model.config.text_config.vocab_size)
        {
            return Err(Error::InvalidRequest(
                "invalid LFM2-VL output token IDs".into(),
            ));
        }
        let h = self
            .model
            .forward(ids, image_features, device)?
            .slice(s![.., ids.len() - 1.., ..]);
        let index = Tensor::<1, Int>::from_data(
            TensorData::new(
                output_ids
                    .iter()
                    .map(|&id| i64::from(id))
                    .collect::<Vec<_>>(),
                [output_ids.len()],
            ),
            device,
        );
        let rows = self
            .model
            .language_model
            .embed_tokens
            .weight
            .val()
            .select(0, index);
        Ok(h.matmul(rows.transpose().unsqueeze_dim::<3>(0)))
    }
}
