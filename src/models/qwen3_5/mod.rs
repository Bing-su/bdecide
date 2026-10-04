//! Mirror Transformers' Qwen3.5 text configuration and module names.
mod configuration_qwen3_5;
mod modeling_qwen3_5;

pub use configuration_qwen3_5::{Qwen3_5Config, Qwen3_5TextConfig};
pub use modeling_qwen3_5::Qwen3_5TextModel;
