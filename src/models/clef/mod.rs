//! Clef and Clef-flash share a Qwen3.5 backbone and a joint schema head.
mod configuration_clef;
mod modeling_clef;
mod processing_clef;
pub(crate) mod weights;

pub use configuration_clef::ClefConfig;
pub use modeling_clef::{ClefDecisionModel, ClefModel};
pub use processing_clef::{ClefProcessor, EncodedQuestion, EncodedRecord};
