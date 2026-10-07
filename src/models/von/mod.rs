//! Run Von's Option-Marker model with its calibrated posterior, e.g. wfzyx/von.
mod configuration_von;
mod modeling_von;
mod processing_von;

pub(crate) use configuration_von::VonConfig;
pub use modeling_von::VonModel;
