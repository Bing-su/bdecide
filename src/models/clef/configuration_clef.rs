use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::{Error, Result, models::qwen3_5::Qwen3_5Config};

/// Match `joint_head_config.json`, e.g. width=1024 for both public releases.
#[derive(Debug, Clone, Deserialize, Serialize, Builder)]
pub struct ClefConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

impl ClefConfig {
    /// Specify the decision head dimensions, e.g. `ClefConfig::new(8, 8, 1, 1, 2, 16)`.
    pub fn new(
        hidden_size: usize,
        width: usize,
        routing_layers: usize,
        layers: usize,
        heads: usize,
        feedforward: usize,
    ) -> Self {
        Self {
            hidden_size,
            width,
            routing_layers,
            layers,
            heads,
            feedforward,
        }
    }

    pub fn validate(&self, backbone: &Qwen3_5Config) -> Result<()> {
        backbone.validate()?;
        if self.hidden_size != backbone.text_config.hidden_size
            || self.width == 0
            || self.heads == 0
            || !self.width.is_multiple_of(self.heads)
            || self.feedforward == 0
            || self.width.checked_mul(4).is_none()
            || self.width.checked_mul(self.feedforward).is_none()
        {
            return Err(Error::InvalidCheckpoint(
                "invalid Clef head dimensions".into(),
            ));
        }
        Ok(())
    }
}
