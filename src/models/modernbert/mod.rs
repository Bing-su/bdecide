//! Re-export ModernBERT configuration independently of its tensor implementation.
pub mod configuration_modernbert;
mod modeling_modernbert;

pub use configuration_modernbert::ModernBertConfig;
pub(crate) use modeling_modernbert::ModernBertModel;
