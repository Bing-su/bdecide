//! Keep local System One types together, e.g. build requests and inspect ordered answers.
mod answer;
mod question;
mod request;
mod response;

pub use answer::{Action, SystemOneAnswer};
pub use question::{NoulLabels, SystemOneQuestion};
pub use request::{PredictOptions, SystemOneRequest, Truncation};
pub use response::{Metadata, SystemOneResponse, Usage};
