//! Share file and JSON handling without model-specific prompt conventions.

use crate::{Error, Result};
use camino::Utf8Path;
use serde::de::DeserializeOwned;
use serde_json::Value;

pub(crate) fn read(path: &Utf8Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| Error::Io {
        path: path.into(),
        source,
    })
}

pub(crate) fn read_checkpoint_json<T: DeserializeOwned>(path: &Utf8Path) -> Result<T> {
    // Keep syntax and schema failures with the model artifact, e.g. encoder/config.json.
    serde_json::from_slice(&read(path)?)
        .map_err(|source| Error::InvalidCheckpoint(format!("{path}: {source}")))
}

pub(crate) fn render(value: &Value) -> Result<String> {
    // Keep text verbatim and use standard JSON for structured data, e.g. {"a":1}.
    match value {
        Value::String(text) => Ok(text.clone()),
        _ => Ok(serde_json::to_string(value)?),
    }
}

pub(crate) fn number_text(number: &serde_json::Number) -> String {
    // Match Python's finite JSON/str floats, e.g. 1e-06 and -0.0, in model prompts.
    if !number.is_f64() {
        return number.to_string();
    }
    let value = number
        .as_f64()
        .expect("JSON floating number has an f64 representation");
    if value != 0.0 && (value.abs() < 1e-4 || value.abs() >= 1e16) {
        let text = format!("{value:e}");
        let (mantissa, exponent) = text
            .split_once('e')
            .expect("scientific formatting contains an exponent");
        let exponent: i32 = exponent
            .parse()
            .expect("scientific formatting produces an integer exponent");
        format!("{mantissa}e{exponent:+03}")
    } else {
        let text = value.to_string();
        if text.contains('.') {
            text
        } else {
            format!("{text}.0")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_text_and_compact_json_in_insertion_order() {
        assert_eq!(render(&Value::String("한글".into())).unwrap(), "한글");
        let value: Value =
            serde_json::from_str(r#"{"z":"한글","a":[true,null,1.0,0.000001]}"#).unwrap();
        assert_eq!(
            render(&value).unwrap(),
            r#"{"z":"한글","a":[true,null,1.0,1e-6]}"#,
        );
    }
}
