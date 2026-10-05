//! Resolve pretrained sources while separating options, artifact selection, and I/O.
mod artifacts;
mod download;
mod environment;
mod options;

pub(crate) use artifacts::{Family, resolve_auto, resolve_clef, resolve_vev, resolve_wald};
pub(crate) use download::{Artifacts, resolve};
// Preserve named builder types too, e.g. callers can annotate HubOptionsBuilder.
pub use options::{HubOptions, HubOptionsBuilder, ModelSource, Token};
