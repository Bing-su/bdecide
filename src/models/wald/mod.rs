//! Expose Wald's model while keeping configuration and prompt logic separate.
mod configuration_wald;
mod modeling_wald;
mod processing_wald;

pub(crate) use configuration_wald::WaldConfig;
pub use modeling_wald::WaldModel;
