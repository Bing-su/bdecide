//! Local typed decisions with reusable models and explicit loading effects.
//!
//! Parse a [`Request`] from JSON, load an [`AutoModel`] once, and reuse its
//! [`DecisionModel::system_one`] method for successive requests.

#![recursion_limit = "256"]

mod error;
pub mod hub;
mod media;
pub mod models;
mod request;
mod response;
mod utils;

pub use error::{Error, Result};
pub use media::{AudioInput, ImageInput};
pub use models::clef::{ClefConfig, ClefDecisionModel, ClefModel, ClefProcessor};
pub use models::d1::{D1Model, D1OmniConfig, D1OmniModel};
pub use models::decider::DeciderModel;
pub use models::laya::{LayaConfig, LayaDecisionModel, LayaModel};
pub use models::lfm2::{Lfm2Config, Lfm2Model};
pub use models::lfm2_vl::{Lfm2VlConfig, Lfm2VlForConditionalGeneration, Lfm2VlModel};
pub use models::modernbert::ModernBertConfig;
pub use models::qwen3_5::{
    CausalLMOutput,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
};
pub use models::vev::VevModel;
pub use models::von::VonModel;
pub use models::wald::WaldModel;
pub use models::{AutoModel, DecisionModel, Device, LoadOptions};
pub use request::{NoulLabels, PredictOptions, Question, Request, Truncation};
pub use response::{Action, Answer, Metadata, Response, Usage};
