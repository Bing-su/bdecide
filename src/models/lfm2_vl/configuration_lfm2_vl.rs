//! Preserve LFM2-VL's nested Transformers configuration, e.g. text_config.hidden_size.
use camino::Utf8Path;
use serde::{Deserialize, Serialize};

use crate::models::lfm2::Lfm2Config;
use crate::utils::activation::HiddenActivation;
use crate::utils::read_checkpoint_json;
use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm2VlConfig {
    pub model_type: String,
    pub text_config: Lfm2Config,
    pub vision_config: Siglip2VisionConfig,
    pub projector_hidden_size: usize,
    pub bos_token_id: u32,
    pub image_token_id: u32,
    pub tie_word_embeddings: bool,
    pub downsample_factor: usize,
    pub projector_bias: bool,
    pub projector_use_layernorm: bool,
    pub projector_hidden_act: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Siglip2VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_patches: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub hidden_act: String,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub vision_use_head: bool,
}

impl Lfm2VlConfig {
    /// Read a vision-language checkpoint, e.g. the config inherited by D1-3B.
    pub fn from_pretrained(root: &Utf8Path) -> Result<Self> {
        let config: Self = read_checkpoint_json(&root.join("config.json"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.text_config.validate()?;
        self.vision_config.validate()?;
        self.projector_hidden_act.parse::<HiddenActivation>()?;
        if self.model_type != "lfm2_vl"
            || self.projector_hidden_size == 0
            || self.downsample_factor != 2
            || !self.projector_bias
            || self.projector_use_layernorm
            || !self.tie_word_embeddings
            || self.bos_token_id as usize >= self.text_config.vocab_size
            || self.image_token_id as usize >= self.text_config.vocab_size
        {
            return Err(Error::UnsupportedModel(
                "LFM2-VL requires tied embeddings, valid token IDs and the released 2x2 projector"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl Siglip2VisionConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        self.hidden_act.parse::<HiddenActivation>()?;
        if [
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_patches,
        ]
        .contains(&0)
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.num_patches.isqrt().pow(2) != self.num_patches
            || self.patch_size != 16
            || self.num_channels != 3
            || self.vision_use_head
            || !self.layer_norm_eps.is_finite()
            || self.layer_norm_eps <= 0.0
        {
            return Err(Error::UnsupportedModel(
                "requires SigLIP2 NaFlex with 16px RGB patches".into(),
            ));
        }
        Ok(())
    }
}
