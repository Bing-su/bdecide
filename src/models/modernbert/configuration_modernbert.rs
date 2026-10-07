//! Read checkpoint configuration so model dimensions are validated before inference.
use bon::Builder;
use serde::Deserialize;
use serde_json::Value;

use crate::utils::activation::HiddenActivation;
use crate::{Error, Result};

/// Read ModernBERT dimensions from the checkpoint, e.g. encoder/config.json.
#[derive(Debug, Clone, Deserialize, Builder)]
#[builder(on(String, into))]
pub struct ModernBertConfig {
    #[builder(default = "modernbert")]
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "three")]
    #[builder(default = three())]
    pub global_attn_every_n_layers: usize,
    #[serde(default = "window")]
    #[builder(default = window())]
    pub local_attention: usize,
    #[serde(default = "epsilon")]
    #[builder(default = epsilon())]
    pub norm_eps: f64,
    #[serde(default)]
    #[builder(default)]
    pub norm_bias: bool,
    #[serde(default)]
    #[builder(default)]
    pub attention_bias: bool,
    #[serde(default)]
    #[builder(default)]
    pub mlp_bias: bool,
    #[serde(default = "activation")]
    #[builder(default = activation())]
    pub hidden_activation: String,
    #[serde(default)]
    #[builder(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    #[builder(default)]
    pub rope_parameters: Value,
    #[serde(default)]
    pub global_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
}

fn three() -> usize {
    3
}

fn window() -> usize {
    128
}

fn epsilon() -> f64 {
    1e-5
}

fn activation() -> String {
    "gelu".into()
}

impl ModernBertConfig {
    /// Supply encoder dimensions with standard attention defaults, e.g. `ModernBertConfig::new(8, 16, 32, 1, 2, 128)`.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        max_position_embeddings: usize,
    ) -> Self {
        Self::builder()
            .hidden_size(hidden_size)
            .intermediate_size(intermediate_size)
            .vocab_size(vocab_size)
            .num_hidden_layers(num_hidden_layers)
            .num_attention_heads(num_attention_heads)
            .max_position_embeddings(max_position_embeddings)
            .build()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.model_type != "modernbert" {
            return Err(Error::UnsupportedModel(format!(
                "ModernBERT encoder {} with activation {}",
                self.model_type, self.hidden_activation
            )));
        }
        self.hidden_activation.parse::<HiddenActivation>()?;
        let hidden_size = self.hidden_size;
        if hidden_size == 0
            || self.num_attention_heads == 0
            || !hidden_size.is_multiple_of(self.num_attention_heads)
            || !(hidden_size / self.num_attention_heads).is_multiple_of(2)
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.num_hidden_layers == 0
            || self.max_position_embeddings == 0
            || self.global_attn_every_n_layers == 0
            || self.local_attention < 2
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
            || hidden_size.checked_mul(4).is_none()
            || self.intermediate_size.checked_mul(2).is_none()
            || hidden_size.checked_mul(self.vocab_size).is_none()
        {
            return Err(Error::InvalidCheckpoint(
                "invalid ModernBERT dimensions, attention, or normalization".into(),
            ));
        }
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers {
            return Err(Error::InvalidCheckpoint(
                "layer_types does not match encoder layer count".into(),
            ));
        }
        for i in 0..self.num_hidden_layers {
            self.rope(i)?;
        }
        Ok(())
    }

    pub(crate) fn rope(&self, i: usize) -> Result<(f64, Option<usize>)> {
        let kind = self
            .layer_types
            .get(i)
            .map(String::as_str)
            .unwrap_or_else(|| {
                if i.is_multiple_of(self.global_attn_every_n_layers) {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            });
        let local = match kind {
            "full_attention" => None,
            "sliding_attention" => Some(self.local_attention / 2),
            _ => return Err(Error::UnsupportedModel(format!("attention type {kind}"))),
        };
        let params = self
            .rope_parameters
            .get(kind)
            .unwrap_or(&self.rope_parameters);
        if params
            .get("rope_type")
            .and_then(Value::as_str)
            .is_some_and(|s| s != "default")
        {
            return Err(Error::UnsupportedModel(
                "non-default rotary embeddings".into(),
            ));
        }
        let theta = params
            .get("rope_theta")
            .and_then(Value::as_f64)
            .or(if local.is_some() {
                self.local_rope_theta
            } else {
                self.global_rope_theta
            })
            .unwrap_or(if local.is_some() { 10000.0 } else { 160000.0 });
        if !theta.is_finite() || theta <= 0.0 {
            return Err(Error::InvalidCheckpoint("invalid rotary theta".into()));
        }
        Ok((theta, local))
    }
}
