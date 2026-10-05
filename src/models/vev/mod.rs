//! Expose Vev's model while keeping configuration and prompt logic separate.
mod configuration_vev;
mod modeling_vev;
mod processing_vev;

pub(crate) use configuration_vev::VevConfig;
pub use modeling_vev::VevModel;
