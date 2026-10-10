//! Read the checkpoint's temperature map and option-isolation mode.
use serde::Deserialize;

use crate::utils::decision::softmax;
use crate::{Error, Result};

#[derive(Deserialize)]
pub(crate) struct VonConfig {
    model_type: String,
    #[serde(default = "one")]
    pub(super) temperature: f64,
    #[serde(default)]
    pub(super) independent_options: bool,
    #[serde(default)]
    pub(super) digit_split: bool,
    calibration_map: Option<CalibrationMap>,
    pub(super) noul_zero_shot_prior: Option<NoulPrior>,
}
fn one() -> f64 {
    1.0
}
#[derive(Deserialize)]
pub(super) struct NoulPrior {
    pub(super) a: f64,
    pub(super) b: f64,
}
#[derive(Deserialize)]
struct CalibrationMap {
    #[serde(default)]
    bias: f64,
    #[serde(default)]
    entropy: f64,
    #[serde(default)]
    log_tokens: f64,
    #[serde(default)]
    n_options: f64,
    #[serde(default = "lo")]
    lo: f64,
    #[serde(default = "hi")]
    hi: f64,
}
fn lo() -> f64 {
    0.5
}
fn hi() -> f64 {
    12.0
}
impl VonConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.model_type != "option_marker" {
            return Err(Error::UnsupportedModel(
                "Von requires option_marker calibration".into(),
            ));
        }
        if !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.calibration_map.as_ref().is_some_and(|map| {
                [
                    map.bias,
                    map.entropy,
                    map.log_tokens,
                    map.n_options,
                    map.lo,
                    map.hi,
                ]
                .iter()
                .any(|v| !v.is_finite())
                    || map.lo <= 0.0
                    || map.lo > map.hi
            })
            || self
                .noul_zero_shot_prior
                .as_ref()
                .is_some_and(|prior| !prior.a.is_finite() || !prior.b.is_finite())
        {
            return Err(Error::InvalidCheckpoint("invalid Von calibration".into()));
        }
        Ok(())
    }

    pub(super) fn effective_temperature(&self, logits: &[f64], state_tokens: usize) -> Result<f64> {
        let Some(map) = &self.calibration_map else {
            return Ok(self.temperature);
        };
        let p = softmax(logits)?;
        let n = logits.len();
        let entropy = if n > 1 {
            -p.iter().map(|v| v * v.max(1e-12).ln()).sum::<f64>() / (n as f64).ln()
        } else {
            0.0
        };
        Ok((map.bias
            + map.entropy * entropy
            + map.log_tokens * (state_tokens.max(1) as f64).log10() / 4.0
            + map.n_options * n as f64 / 8.0)
            .clamp(map.lo, map.hi))
    }
}
