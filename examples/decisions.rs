//! Evaluate OpenAI-shaped questions locally, e.g. `cargo run --example decisions -- ./checkpoint`.
use std::env::args;
use std::error::Error;

use bdecide::hub::ModelSource;
use bdecide::{AutoModel, DecisionModel, DecisionsRequest, LoadOptions};
use camino::Utf8Path;

fn main() -> Result<(), Box<dyn Error>> {
    let repo = args()
        .nth(1)
        .unwrap_or_else(|| "convaiinnovations/laya-multilingual".into());
    // Local paths need an explicit source; LoadOptions::new interprets strings as Hub IDs.
    let options = if Utf8Path::new(&repo).is_dir() {
        LoadOptions::builder()
            .source(ModelSource::Local(repo.into()))
            .build()
    } else {
        LoadOptions::new(repo)
    };
    let model = AutoModel::from_pretrained(options)?;
    let request = DecisionsRequest::new(
        &model.metadata().model_id,
        "I want to cancel my subscription.",
    )
    .predicate("cancel", "Does the customer want to cancel?");
    let response = model.decisions(&request)?;
    if let Some(probability) = response.probability("cancel") {
        eprintln!("Cancellation probability: {probability}");
    }
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
