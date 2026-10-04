//! Keep Laya head configuration separate from its ModernBERT encoder dimensions.
use crate::models::modernbert::ModernBertConfig;
use crate::{Error, Result};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

/// Describe the Laya decision head separately from its encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LayaConfig {
    pub encoder: String,
    #[serde(default = "two")]
    pub head_layers: usize,
    #[serde(default = "max_len")]
    pub max_len: usize,
    #[serde(default = "head_len")]
    pub head_max_len: usize,
    #[serde(default)]
    pub act_costs: IndexMap<String, f64>,
    #[serde(default = "temperatures")]
    pub temperature: Vec<Value>,
    #[serde(default)]
    pub temperature_by_options: IndexMap<String, Value>,
    #[serde(default)]
    pub binning_map: Value,
}

fn two() -> usize {
    2
}

fn max_len() -> usize {
    512
}

fn head_len() -> usize {
    192
}

fn temperatures() -> Vec<Value> {
    vec![Value::from(1.0); 3]
}

impl LayaConfig {
    pub(crate) fn validate(&self, encoder: &ModernBertConfig) -> Result<()> {
        // Laya chooses its own attention head count, e.g. hidden_size / 64.
        let head_count = (encoder.hidden_size / 64).max(1);
        if !encoder.hidden_size.is_multiple_of(head_count) {
            return Err(Error::InvalidCheckpoint(
                "decision head size is not divisible by its attention heads".into(),
            ));
        }
        // Avoid silently ignoring a calibration that would change confidence.
        if !self.binning_map.is_null()
            && self
                .binning_map
                .as_object()
                .is_none_or(|map| !map.is_empty())
        {
            return Err(Error::UnsupportedModel(
                "histogram binning calibration".into(),
            ));
        }
        if self.max_len < 4
            || self.max_len > encoder.max_position_embeddings
            || self.head_max_len < 16
            || self.head_max_len > encoder.max_position_embeddings
            || self.head_layers == 0
            || self.temperature.len() != 3
            || self.encoder.is_empty()
        {
            return Err(Error::InvalidCheckpoint(
                "invalid Laya context budgets or temperatures".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn temperature(&self, kind: usize, options: usize) -> f32 {
        let size = match options {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let name = match kind {
            0 => "choice",
            1 => "score",
            _ => "noul",
        };
        let value = self
            .temperature_by_options
            .get(&format!("{name}:{size}"))
            .or_else(|| self.temperature.get(kind));
        // Keep Python's runtime clamp, e.g. a stored 0.1 temperature becomes 0.5.
        let number = value
            .and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()))
            .filter(|v| v.is_finite())
            .unwrap_or(1.0);
        number.clamp(0.5, 5.0) as f32
    }
}
