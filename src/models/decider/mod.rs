//! Load merged dense Qwen3.5 Decider releases, e.g. Mapika/decider-2b.
mod configuration_decider;
mod modeling_decider;
mod processing_decider;

pub(crate) use configuration_decider::DeciderConfig;
pub use modeling_decider::DeciderModel;
