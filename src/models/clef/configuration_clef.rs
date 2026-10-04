use crate::{Error, Result, models::qwen3_5::Qwen3_5Config};
use serde::{Deserialize, Serialize};

/// Match `joint_head_config.json`, e.g. width=1024 for both public releases.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClefConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

impl ClefConfig {
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
