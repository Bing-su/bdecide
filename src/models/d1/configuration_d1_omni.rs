//! Configure Omni's encoder, media modules and decision head, e.g. audio_text_length.
use std::collections::BTreeMap;

use camino::Utf8Path;
use serde::Deserialize;

use crate::models::lfm2::Lfm2Config;
use crate::models::lfm2_vl::Siglip2VisionConfig;
use crate::utils::read_checkpoint_json;
use crate::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct D1OmniConfig {
    pub model_type: String,
    pub text_config: Lfm2Config,
    pub vision_config: Siglip2VisionConfig,
    pub(crate) audio_config: AudioConfig,
    pub projector_hidden_size: usize,
    pub head_layers: usize,
    pub max_length: usize,
    pub image_text_length: usize,
    pub audio_text_length: usize,
    #[serde(default)]
    pub temperatures: BTreeMap<String, f64>,
    pub bos_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AudioConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub subsampling_conv_channels: usize,
    pub ff_expansion_factor: usize,
    pub n_heads: usize,
    pub conv_kernel_size: usize,
    pub residual_width: usize,
}

impl D1OmniConfig {
    /// Read Omni's config, e.g. d1_omni with a required audio_config.
    pub fn from_pretrained(root: &Utf8Path) -> Result<Self> {
        let config: Self = read_checkpoint_json(&root.join("config.json"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.text_config.validate()?;
        self.vision_config.validate()?;
        let t = &self.text_config;
        let a = &self.audio_config;
        if self.model_type != "d1_omni"
            || self.projector_hidden_size == 0
            || self.bos_token_id as usize >= t.vocab_size
            || t.conv_bias
            || a.feat_in != 128
            || [
                a.n_layers,
                a.d_model,
                a.subsampling_conv_channels,
                a.ff_expansion_factor,
                a.n_heads,
                a.residual_width,
            ]
            .contains(&0)
            || !a.d_model.is_multiple_of(a.n_heads)
            || !a.d_model.is_multiple_of(2)
            || a.conv_kernel_size == 0
            || a.conv_kernel_size.is_multiple_of(2)
            || !t.hidden_size.is_multiple_of(64)
            || self.head_layers == 0
            || self.max_length < 64
            || self.max_length > t.max_position_embeddings
            || self.image_text_length < 64
            || self.audio_text_length < 64
            || self
                .temperatures
                .values()
                .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(Error::UnsupportedModel(
                "invalid D1Omni encoder, head, context, calibration or FastConformer".into(),
            ));
        }
        Ok(())
    }
}
