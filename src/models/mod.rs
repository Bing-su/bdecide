//! Expose model families and their automatic loader without owning implementations.
pub mod auto;
pub mod clef;
pub mod d1;
pub mod decider;
mod decision;
pub mod laya;
pub mod lfm2;
pub mod lfm2_vl;
pub mod modernbert;
pub mod qwen3_5;
pub mod vev;
pub mod von;
pub mod wald;
pub(crate) mod weights;

pub use auto::{AutoModel, Device, LoadOptions};
pub use decision::DecisionModel;
