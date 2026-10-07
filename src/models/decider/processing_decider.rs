//! Preserve Decider's option descriptions and long-array lookup annotations.
use serde_json::{Map, Value};

use crate::Result;
use crate::utils::render;

pub(super) fn state(value: &Value) -> Result<String> {
    render(&annotate(value))
}
fn annotate(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(i, item)| {
                    let item = annotate(item);
                    if items.len() < 8 {
                        return item;
                    }
                    let mut fields = Map::new();
                    fields.insert("_index".into(), Value::from(i));
                    // Caller-provided _index keeps upstream precedence, e.g. an annotated record.
                    if let Value::Object(object) = item {
                        fields.extend(object);
                    } else {
                        fields.insert("value".into(), item);
                    }
                    Value::Object(fields)
                })
                .collect(),
        ),
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, value)| (key.clone(), annotate(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}
pub(super) fn option(name: &str, description: Option<&Value>) -> Result<String> {
    match description {
        None | Some(Value::Null) => Ok(name.into()),
        Some(Value::String(text)) if text.is_empty() => Ok(name.into()),
        Some(value) => Ok(format!("{name}: {}", render(value)?)),
    }
}
pub(super) fn neutralize(text: String, enabled: bool) -> String {
    let key = text.trim().to_lowercase();
    if enabled
        && (key.starts_with("none of the above")
            || matches!(key.as_str(), "none" | "n/a" | "none of these"))
    {
        "not listed here".into()
    } else {
        text
    }
}
pub(super) fn strip_level(text: &str) -> Result<String> {
    // Strip only an integer prefix, e.g. "-2: severe" becomes "severe".
    let number = regex::Regex::new(r"^\s*-?\d+\s*:\s*")
        .map_err(|error| crate::Error::Tokenizer(error.to_string()))?;
    Ok(number.replace(text, "").into_owned())
}
