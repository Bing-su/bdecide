use crate::{Error, Result, utils::activation::HiddenActivation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3_5Config {
    pub model_type: String,
    pub text_config: Qwen3_5TextConfig,
}

/// Read Transformers' nested text_config without depending on the vision tower.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Qwen3_5TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub full_attention_interval: usize,
    pub layer_types: Option<Vec<String>>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_parameters: serde_json::Value,
}

impl Default for Qwen3_5TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 248320,
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".into(),
            attention_bias: false,
            full_attention_interval: 4,
            layer_types: None,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            rope_parameters: serde_json::json!({"rope_type":"default", "rope_theta":10000000.0, "partial_rotary_factor":0.25}),
        }
    }
}

impl Qwen3_5Config {
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_5" {
            return Err(Error::UnsupportedModel(self.model_type.clone()));
        }
        self.text_config.validate()
    }
}
impl Qwen3_5TextConfig {
    pub(crate) fn rotary(&self) -> Result<(usize, f64)> {
        let rope = &self.rope_parameters;
        if rope
            .get("rope_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default")
            != "default"
        {
            return Err(Error::UnsupportedModel("Qwen3.5 scaled RoPE".into()));
        }
        let factor = rope
            .get("partial_rotary_factor")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.25);
        let theta = rope
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10000000.0);
        #[expect(
            clippy::cast_sign_loss,
            reason = "invalid factors are rejected below before dimensions are used"
        )]
        let dim = (self.head_dim as f64 * factor) as usize;
        if !factor.is_finite()
            || !(0.0..=1.0).contains(&factor)
            || dim == 0
            || !dim.is_multiple_of(2)
            || !theta.is_finite()
            || theta <= 0.0
        {
            return Err(Error::InvalidCheckpoint(
                "invalid Qwen3.5 rotary dimensions".into(),
            ));
        }
        Ok((dim, theta))
    }
    pub(crate) fn layer_types(&self) -> Vec<String> {
        self.layer_types.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|i| {
                    if (i + 1).is_multiple_of(self.full_attention_interval) {
                        "full_attention"
                    } else {
                        "linear_attention"
                    }
                    .into()
                })
                .collect()
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.hidden_act.parse::<HiddenActivation>()?;
        let dimensions = [
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.full_attention_interval,
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
        ];
        if dimensions.contains(&0)
            || dimensions.iter().any(|&n| n > i32::MAX as usize)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err(Error::InvalidCheckpoint(
                "invalid Qwen3.5 text configuration".into(),
            ));
        }
        // Validate projection arithmetic before creating lazy parameters, e.g. 2*K+V.
        let key = self
            .linear_key_head_dim
            .checked_mul(self.linear_num_key_heads);
        let value = self
            .linear_value_head_dim
            .checked_mul(self.linear_num_value_heads);
        if key
            .and_then(|k| k.checked_mul(2))
            .and_then(|k| value.and_then(|v| k.checked_add(v)))
            .is_none()
            || self
                .num_attention_heads
                .checked_mul(self.head_dim)
                .and_then(|n| n.checked_mul(2))
                .is_none()
            || self.hidden_size.checked_mul(self.vocab_size).is_none()
        {
            return Err(Error::InvalidCheckpoint(
                "Qwen3.5 dimensions overflow".into(),
            ));
        }
        let types = self.layer_types();
        if types.len() != self.num_hidden_layers
            || types
                .iter()
                .any(|t| t != "linear_attention" && t != "full_attention")
        {
            return Err(Error::InvalidCheckpoint(
                "invalid Qwen3.5 layer_types".into(),
            ));
        }
        self.rotary()?;
        Ok(())
    }
}
