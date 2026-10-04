//! Expose model families and their automatic loader without owning implementations.
pub mod auto;
pub mod clef;
pub mod laya;
pub mod modernbert;
pub mod qwen3_5;

pub use auto::{AutoModel, Device, LoadOptions};
