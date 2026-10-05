//! Load once and reuse the model for successive typed requests.
//! Example: cargo run --example predict -- convaiinnovations/laya-multilingual
use std::{env::args, error::Error};

use bdecide::{AutoModel, DecisionModel, LoadOptions, Request};

fn main() -> Result<(), Box<dyn Error>> {
    let repo = args()
        .nth(1)
        .unwrap_or_else(|| "convaiinnovations/laya-multilingual".into());
    let model = AutoModel::from_pretrained(LoadOptions::new(repo))?;
    let request: Request = serde_json::from_str(include_str!("request.json"))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&model.predict(&request)?)?
    );
    Ok(())
}
