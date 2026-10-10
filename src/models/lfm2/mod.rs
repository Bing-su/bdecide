//! LFM2 configuration and backbone, e.g. LFM2-VL's language_model.
mod configuration_lfm2;
mod modeling_lfm2;

pub use configuration_lfm2::{Lfm2Config, RopeConfig};
pub use modeling_lfm2::Lfm2Model;
