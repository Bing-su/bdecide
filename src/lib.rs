//! Local typed decisions with reusable models and explicit loading effects.
//!
//! Parse a [`Request`] from JSON, load an [`AutoModel`] once, and reuse its
//! [`DecisionModel::predict`] method for successive requests.

#![recursion_limit = "256"]

mod error;
pub mod hub;
pub mod models;
mod request;
mod response;
mod utils;

pub use error::{Error, Result};
pub use models::{AutoModel, Device, LoadOptions};
pub use models::{
    clef::{ClefConfig, ClefDecisionModel, ClefModel, ClefProcessor},
    laya::{LayaConfig, LayaDecisionModel, LayaModel},
    modernbert::ModernBertConfig,
    qwen3_5::{
        CausalLMOutput, Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig, Qwen3_5TextModel,
    },
    vev::VevModel,
    wald::WaldModel,
};
pub use request::{NoulLabels, PredictOptions, Question, Request, Truncation};
pub use response::{Action, Answer, Metadata, Response, Usage};

/// Implement this interface to use an independently maintained decision model.
///
/// Implementations own preprocessing and output semantics; for example a Clef
/// implementation can accept the same typed questions with its own processor.
pub trait DecisionModel {
    fn predict(&self, request: &Request) -> Result<Response>;
    fn metadata(&self) -> &Metadata;
}
