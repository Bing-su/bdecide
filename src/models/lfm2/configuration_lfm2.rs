//! Read LFM2 independently of decision heads, e.g. text_config inside LFM2-VL.
use camino::Utf8Path;
use serde::{Deserialize, Serialize};

use crate::utils::read_checkpoint_json;
use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub layer_types: Vec<String>,
    pub norm_eps: f64,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "yes")]
    pub block_auto_adjust_ff_dim: bool,
    pub block_ffn_dim_multiplier: f64,
    pub block_multiple_of: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_parameters: Option<RopeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f64,
    pub rope_type: String,
}

fn theta() -> f64 {
    1_000_000.0
}
fn yes() -> bool {
    true
}

impl Lfm2Config {
    /// Read a text checkpoint, e.g. config.json with LFM2 dimensions.
    pub fn from_pretrained(root: &Utf8Path) -> Result<Self> {
        let config: Self = read_checkpoint_json(&root.join("config.json"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if [
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.block_multiple_of,
            self.max_position_embeddings,
        ]
        .contains(&0)
            || self.num_hidden_layers == 0
            || self.layer_types.len() != self.num_hidden_layers
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !(self.hidden_size / self.num_attention_heads).is_multiple_of(2)
            || self
                .layer_types
                .iter()
                .any(|kind| !matches!(kind.as_str(), "conv" | "full_attention"))
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
            || !self.block_ffn_dim_multiplier.is_finite()
            || self.block_ffn_dim_multiplier <= 0.0
            || self.conv_l_cache != 3
            || !self.theta().is_finite()
            || self.theta() <= 0.0
            || self
                .rope_parameters
                .as_ref()
                .is_some_and(|rope| rope.rope_type != "default")
        {
            return Err(Error::UnsupportedModel(
                "invalid or unsupported LFM2 dimensions, convolution or RoPE".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map_or(self.rope_theta, |rope| rope.rope_theta)
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "validated positive dimensions follow Transformers' integer sizing rule"
    )]
    pub(crate) fn ffn_size(&self) -> usize {
        if self.block_auto_adjust_ff_dim {
            let hidden = (2 * self.intermediate_size / 3) as f64 * self.block_ffn_dim_multiplier;
            (hidden as usize).div_ceil(self.block_multiple_of) * self.block_multiple_of
        } else {
            self.intermediate_size
        }
    }
}
