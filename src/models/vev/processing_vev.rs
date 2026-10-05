//! Preserve the published Vev prompts and state rendering, e.g. ordered choice labels.
use std::ops::Range;

use serde_json::Value;

use crate::{
    Error, Question, Result,
    utils::{render, sanitize},
};

const SYSTEM: &str = "You are a careful judge. Read the state, then answer the question about it. Reply with only the answer token, nothing else.";

pub(super) fn prompt(
    state: &str,
    question: &Question,
    labels: &[String],
) -> Result<(String, Range<usize>, Vec<String>)> {
    let mut text = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nState:\n");
    let start = text.len();
    text.push_str(state);
    let range = start..text.len();
    text.push_str(&format!(
        "\n\nQuestion: {}\n\n",
        sanitize(question.instructions().trim())
    ));
    let mut legend = Vec::new();
    match question {
        Question::Choice { criteria, .. } => {
            text.push_str("Options:\n");
            for ((key, value), label) in criteria.iter().zip(labels) {
                let description = desc(value)?;
                text.push_str(&format!("{label}. {}", sanitize(key)));
                if !description.is_empty() {
                    text.push_str(&format!(" - {}", sanitize(&description)));
                }
                text.push('\n');
            }
            text.push_str("\nAnswer with the letter of the single best option.");
        }
        Question::Score { criteria, .. } => {
            text.push_str(&format!(
                "Scale, from lowest (0) to highest ({}):\n",
                criteria.len() - 1
            ));
            for (i, value) in criteria.iter().enumerate() {
                let description = desc(value)?;
                text.push_str(&format!(
                    "{}\n",
                    format!("{i}. {}", sanitize(&description)).trim_end()
                ));
                legend.push(description);
            }
            text.push_str("\nAnswer with the level number only.");
        }
        Question::Noul { criteria, .. } => {
            text.push_str("Answer Yes or No.");
            for (key, label) in [("true", "Yes"), ("false", "No")] {
                if let Some(value) = criteria.get(key).filter(|value| !value.is_null()) {
                    text.push_str(&format!("\n{label} means: {}", sanitize(&desc(value)?)));
                }
            }
        }
    }
    text.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok((text, range, legend))
}

pub(super) fn render_state(value: &Value) -> Result<String> {
    if let Value::String(text) = value {
        return Ok(text.trim_end_matches('\n').into());
    }
    fn tree(value: &Value, depth: usize, out: &mut String) -> Result<()> {
        let pad = "  ".repeat(depth);
        let entries: Vec<_> = match value {
            Value::Object(fields) => fields
                .iter()
                .map(|(key, value)| (format!("{key}:"), value))
                .collect(),
            Value::Array(values) => values
                .iter()
                .enumerate()
                .map(|(i, value)| (format!("{}.", i + 1), value))
                .collect(),
            _ => {
                out.push_str(&format!("{pad}{}\n", desc(value)?));
                return Ok(());
            }
        };
        for (key, value) in entries {
            if key == "image:" && value.get("url").is_some_and(Value::is_string) {
                return Err(Error::InvalidRequest(
                    "Vev image inputs require a vision backend; this loader supports text states"
                        .into(),
                ));
            }
            out.push_str(&format!("{pad}{key}"));
            if value.as_object().is_some_and(|v| !v.is_empty())
                || value.as_array().is_some_and(|v| !v.is_empty())
            {
                out.push('\n');
                tree(value, depth + 1, out)?;
            } else {
                let text = if value.is_string() {
                    desc(value)?
                } else {
                    serde_json::to_string(value)?
                };
                if text.contains('\n') {
                    out.push('\n');
                    for line in text.split('\n') {
                        out.push_str(&format!("{}{}\n", "  ".repeat(depth + 1), line));
                    }
                } else {
                    out.push_str(&format!(" {text}\n"));
                }
            }
        }
        Ok(())
    }
    let mut out = String::new();
    tree(value, 0, &mut out)?;
    Ok(out.trim_end_matches('\n').into())
}

fn desc(value: &Value) -> Result<String> {
    // Vev uses an empty description for null, e.g. an option without explanatory text.
    match value {
        Value::Null => Ok(String::new()),
        _ => render(value),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn descriptions_and_state_leaves_preserve_values() {
        // Decode structured descriptions by value, e.g. nested Unicode and float exponents.
        let value = json!({"z": [true, null, 0.000001, -0.0], "a": "한글"});
        assert_eq!(
            serde_json::from_str::<Value>(&desc(&value).unwrap()).unwrap(),
            value
        );
        assert_eq!(desc(&Value::String("한글".into())).unwrap(), "한글");
        assert!(desc(&Value::Null).unwrap().is_empty());
        let state = json!({"float": 0.000001, "empty": [], "zero": -0.0});
        for (line, expected) in render_state(&state)
            .unwrap()
            .lines()
            .zip(state.as_object().unwrap().values())
        {
            let (_, text) = line.split_once(": ").unwrap();
            assert_eq!(serde_json::from_str::<Value>(text).unwrap(), *expected);
        }
    }
}
