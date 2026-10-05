use crate::{Error, Result, utils::activation::HiddenActivation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
pub struct Qwen3_5Config {
    #[builder(default = "qwen3_5", into)]
    pub model_type: String,
    #[serde(default)]
    #[builder(default)]
    pub tie_word_embeddings: bool,
    #[builder(default)]
    pub text_config: Qwen3_5TextConfig,
}

/// Read Transformers' nested text_config without depending on the vision tower.
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
#[serde(default)]
pub struct Qwen3_5TextConfig {
    #[builder(default = "qwen3_5_text", into)]
    pub model_type: String,
    #[builder(default = 248320)]
    pub vocab_size: usize,
    #[builder(default = 4096)]
    pub hidden_size: usize,
    #[builder(default = 12288)]
    pub intermediate_size: usize,
    #[builder(default = 32)]
    pub num_hidden_layers: usize,
    #[builder(default = 16)]
    pub num_attention_heads: usize,
    #[builder(default = 4)]
    pub num_key_value_heads: usize,
    #[builder(default = 256)]
    pub head_dim: usize,
    #[builder(default = 32768)]
    pub max_position_embeddings: usize,
    #[builder(default = 1e-6)]
    pub rms_norm_eps: f64,
    #[builder(default = "silu", into)]
    pub hidden_act: String,
    #[builder(default)]
    pub attention_bias: bool,
    #[builder(default)]
    pub tie_word_embeddings: bool,
    #[builder(default = 4)]
    pub full_attention_interval: usize,
    pub layer_types: Option<Vec<String>>,
    #[builder(default = 4)]
    pub linear_conv_kernel_dim: usize,
    #[builder(default = 128)]
    pub linear_key_head_dim: usize,
    #[builder(default = 128)]
    pub linear_value_head_dim: usize,
    #[builder(default = 16)]
    pub linear_num_key_heads: usize,
    #[builder(default = 32)]
    pub linear_num_value_heads: usize,
    #[builder(default = serde_json::json!({"rope_type":"default", "rope_theta":10000000.0, "partial_rotary_factor":0.25}))]
    pub rope_parameters: serde_json::Value,
}

impl Default for Qwen3_5TextConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3_5Config {
    /// Read the multimodal configuration, e.g. Vev's config with a nested text_config.
    pub fn from_pretrained(root: &camino::Utf8Path) -> Result<Self> {
        let path = root.join("config.json");
        let value: serde_json::Value = crate::utils::read_checkpoint_json(&path)?;
        let config: Self = match value.get("model_type").and_then(serde_json::Value::as_str) {
            Some("qwen3_5") => serde_json::from_value(value),
            other => {
                return Err(Error::UnsupportedModel(format!(
                    "Qwen3.5 config: {other:?}"
                )));
            }
        }
        .map_err(|error| Error::InvalidCheckpoint(format!("{path}: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Wrap text dimensions with the Qwen3.5 architecture tag, e.g. `Qwen3_5Config::new(text)`.
    pub fn new(text_config: Qwen3_5TextConfig) -> Self {
        Self::builder().text_config(text_config).build()
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_5" {
            return Err(Error::UnsupportedModel(self.model_type.clone()));
        }
        self.text_config.validate()
    }
}
impl Qwen3_5TextConfig {
    /// Extract the text configuration like Transformers, e.g. from Vev's nested config.
    pub fn from_pretrained(root: &camino::Utf8Path) -> Result<Self> {
        let path = root.join("config.json");
        let value: serde_json::Value = crate::utils::read_checkpoint_json(&path)?;
        let text = match value.get("model_type").and_then(serde_json::Value::as_str) {
            Some("qwen3_5") => value
                .get("text_config")
                .cloned()
                .ok_or_else(|| Error::InvalidCheckpoint(format!("{path}: missing text_config")))?,
            Some("qwen3_5_text") => value,
            other => {
                return Err(Error::UnsupportedModel(format!(
                    "Qwen3.5 text config: {other:?}"
                )));
            }
        };
        let config: Self = serde_json::from_value(text)
            .map_err(|error| Error::InvalidCheckpoint(format!("{path}: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Use Transformers' text defaults, e.g. `Qwen3_5TextConfig::new()`.
    pub fn new() -> Self {
        Self::builder().build()
    }

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
        if self.model_type != "qwen3_5_text" {
            return Err(Error::UnsupportedModel(self.model_type.clone()));
        }
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
