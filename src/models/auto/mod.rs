//! Re-export automatic loading so callers can use `models::auto::AutoModel`.
pub mod modeling_auto;

pub use modeling_auto::{AutoModel, Device, LoadOptions};
