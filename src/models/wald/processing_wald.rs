//! Preserve Wald's training prompts and renderer, e.g. repeated state spans.
use crate::{Question, utils::sanitize};
use serde_json::Value;
use std::ops::Range;

const REPEAT: &str = "\n\nRead the same context again before answering. This is a repeated copy, not additional events or independent evidence:\n";

pub(super) fn option_text(key: &str, value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => key.into(),
        Some(Value::String(text)) if text.is_empty() => key.into(),
        Some(value) => format!("{key}: {}", render(value, 0)),
    }
}

pub(super) fn prompt(
    state: &str,
    question: &Question,
    options: &[String],
    repeat: bool,
) -> (String, Vec<Range<usize>>) {
    let mut text = String::new();
    let mut ranges = Vec::new();
    if !state.trim().is_empty() {
        text.push_str("State:\n");
        let start = text.len();
        text.push_str(state);
        ranges.push(start..text.len());
        if repeat {
            text.push_str(REPEAT);
            let start = text.len();
            text.push_str(state);
            ranges.push(start..text.len());
        }
        text.push_str("\n\n");
    }
    text.push_str(&format!("Question: {}", sanitize(question.instructions())));
    let slug = |text: &str| {
        !text.is_empty()
            && text.len() <= 40
            && text
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            && text.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.-".contains(&byte)
            })
    };
    let keys: Vec<_> = if options.iter().all(|text| slug(text))
        && options
            .iter()
            .enumerate()
            .all(|(i, text)| !options.iter().take(i).any(|previous| previous == text))
    {
        options.to_vec()
    } else {
        (b'a'..=b'z')
            .take(options.len())
            .map(|letter| char::from(letter).to_string())
            .collect()
    };
    let positional = keys.iter().all(|key| {
        key.len() == 1
            && key
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_lowercase())
            || key.strip_prefix('o').is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            })
    });
    for ((option, key), letter) in options.iter().zip(&keys).zip(b'A'..=b'Z') {
        let option = if matches!(question, Question::Choice { .. }) && positional {
            option.strip_prefix(&format!("{key}: ")).unwrap_or(option)
        } else {
            option
        };
        text.push_str(&format!("\n({}) {}", char::from(letter), sanitize(option)));
    }
    text.push_str("\nAnswer: (");
    (text, ranges)
}

pub(super) fn render(value: &Value, depth: usize) -> String {
    // Preserve Wald's training renderer, e.g. arrays use '-' and booleans use Python's True/False.
    let pad = "  ".repeat(depth);
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::Number(number) => number.to_string(),
        Value::Array(values) => values
            .iter()
            .map(|value| format!("{pad}- {}", render(value, depth + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| {
                if value.is_array() || value.is_object() {
                    format!("{pad}{key}:\n{}", render(value, depth + 1))
                } else {
                    format!("{pad}{key}: {}", render(value, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Request, utils::read_checkpoint_json};
    use camino::Utf8Path;

    #[test]
    fn text_prompts_match_pinned_wald_renderer() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-wald");
        let reference: Value = read_checkpoint_json(&root.join("reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap().iter().take(5) {
            let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
            // Check training wording independently of scalar formatting, e.g. 1e-06 vs 1e-6.
            if !request.state.is_string() {
                continue;
            }
            let state = sanitize(&render(&request.state, 0));
            for (question, row) in request
                .questions
                .values()
                .zip(case["rows"].as_array().unwrap())
            {
                let text_only = match question {
                    Question::Choice { criteria, .. } | Question::Noul { criteria, .. } => criteria
                        .values()
                        .all(|value| value.is_string() || value.is_null()),
                    Question::Score { criteria, .. } => criteria.iter().all(Value::is_string),
                };
                if !text_only {
                    continue;
                }
                let options: Vec<_> = match question {
                    Question::Choice { criteria, .. } => criteria
                        .iter()
                        .map(|(key, value)| option_text(key, Some(value)))
                        .collect(),
                    Question::Score { criteria, .. } => {
                        criteria.iter().map(|value| render(value, 0)).collect()
                    }
                    Question::Noul { criteria, .. } => vec![
                        option_text("no", criteria.get("false")),
                        option_text("yes", criteria.get("true")),
                    ],
                };
                assert_eq!(prompt(&state, question, &options, true).0, row["prompt"]);
            }
        }
    }

    #[test]
    fn number_rendering_preserves_values() {
        // Scalar spelling is flexible; retain its value, e.g. tiny floats and signed zero.
        for number in [0.000001_f64, 1e20, 1.0, -0.0] {
            let text = render(&serde_json::json!(number), 0);
            assert_eq!(text.parse::<f64>().unwrap().to_bits(), number.to_bits());
        }
    }
}
