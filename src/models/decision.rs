//! Define the shared prediction contract, e.g. repeated requests reuse loaded models.
use crate::{Metadata, Request, Response, Result};

/// Implement this interface to use an independently maintained decision model.
///
/// Implementations own preprocessing and output semantics; for example a Clef
/// implementation can accept the same typed questions with its own processor.
pub trait DecisionModel {
    fn predict(&self, request: &Request) -> Result<Response>;
    fn metadata(&self) -> &Metadata;
}
