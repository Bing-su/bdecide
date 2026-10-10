//! Preserve Von's marker boundaries while neutralizing caller special-token literals.
use itertools::Itertools;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::utils::{render, token_ids};
use crate::{Error, Question, Result};

#[derive(Deserialize)]
pub(super) struct SpecialTokens {
    pub(super) mask_token: String,
    pub(super) sep_token: String,
}
pub(super) fn neutralize(text: &str, special: &SpecialTokens) -> String {
    let mut text = text.to_owned();
    for token in [&special.mask_token, &special.sep_token] {
        if let Some(first) = token.chars().next() {
            let rest = token.get(first.len_utf8()..).unwrap_or_default();
            text = text.replace(token, &format!("{first}\u{200d}{rest}"));
        }
    }
    text
}
pub(super) fn split_digits(text: &str) -> Result<String> {
    // Unicode decimal digits match Python's \d, e.g. Arabic ١٢ also splits;
    // numeric characters such as ² do not belong to a decimal-digit run.
    let digits = regex::Regex::new(r"\d+").map_err(|error| Error::Tokenizer(error.to_string()))?;
    Ok(digits
        .replace_all(text, |captures: &regex::Captures<'_>| {
            captures
                .get(0)
                .map_or_else(String::new, |run| run.as_str().chars().join(" "))
        })
        .into_owned())
}
pub(super) fn state(value: &Value) -> Result<String> {
    if let Value::Object(fields) = value {
        fields
            .iter()
            .map(|(key, value)| Ok(format!("{key}: {}", render(value)?)))
            .collect::<Result<Vec<_>>>()
            .map(|rows| rows.join("\n"))
    } else {
        render(value)
    }
}
pub(super) fn descriptions(question: &Question) -> Result<Vec<String>> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(key, value)| match value {
                Value::Null => Ok(key.trim().into()),
                Value::String(text) if text.is_empty() => Ok(key.trim().into()),
                _ => Ok(render(value)?.trim().into()),
            })
            .collect(),
        Question::Noul { criteria, .. } => [
            ("true", "Yes, condition holds true."),
            ("false", "No, condition is false."),
        ]
        .into_iter()
        .map(|(key, default)| match criteria.get(key) {
            None | Some(Value::Null) => Ok(default.into()),
            Some(Value::String(text)) if text.is_empty() => Ok(default.into()),
            Some(value) => render(value),
        })
        .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .map(|value| {
                if let Value::Object(fields) = value {
                    let what = fields
                        .get("what")
                        .map(render)
                        .transpose()?
                        .unwrap_or_default();
                    let examples = match fields.get("examples") {
                        None => Vec::new(),
                        Some(Value::Array(items)) => {
                            items.iter().map(render).collect::<Result<Vec<_>>>()?
                        }
                        Some(_) => {
                            return Err(Error::InvalidRequest(
                                "Von score examples must be an array".into(),
                            ));
                        }
                    };
                    Ok(if examples.is_empty() {
                        what.trim().into()
                    } else {
                        format!("{what} Examples: {}", examples.join(", "))
                            .trim()
                            .into()
                    })
                } else {
                    Ok(render(value)?.trim().into())
                }
            })
            .collect(),
    }
}

pub(super) struct Encoded {
    pub(super) ids: Vec<u32>,
    pub(super) markers: Vec<usize>,
    pub(super) state_positions: Vec<usize>,
}
pub(super) fn encode(
    tokenizer: &Tokenizer,
    special: &SpecialTokens,
    vocab: usize,
    state: &str,
    question: &str,
    options: &[String],
    digits: bool,
) -> Result<Encoded> {
    let clean = |text: &str| {
        let text = neutralize(text, special);
        if digits {
            split_digits(&text)
        } else {
            Ok(text)
        }
    };
    let question = clean(question)?;
    let state = clean(state)?;
    let mut text = format!("{question} ");
    // Offsets identify state tokens only, e.g. opt-in truncation cannot remove [MASK].
    let start = text.len();
    text.push_str(&state);
    let end = text.len();
    let trim_start = text.len() - text.trim_start().len();
    text = text.trim().to_owned();
    let mut packed = format!("{text} {} ", special.sep_token);
    for (i, option) in options.iter().enumerate() {
        if i > 0 {
            packed.push(' ');
        }
        packed.push_str(&format!("{} {}", special.mask_token, clean(option)?.trim()));
    }
    let encoded = tokenizer
        .encode(packed.as_str(), true)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let ids = encoded.get_ids().to_vec();
    if ids.iter().any(|id| *id as usize >= vocab) {
        return Err(Error::Tokenizer("token ID exceeds Von vocabulary".into()));
    }
    let mask = tokenizer
        .token_to_id(&special.mask_token)
        .ok_or_else(|| Error::InvalidCheckpoint("Von mask token is absent".into()))?;
    let markers: Vec<_> = ids
        .iter()
        .enumerate()
        .filter_map(|(i, id)| (*id == mask).then_some(i))
        .collect();
    if markers.len() != options.len() {
        return Err(Error::InvalidRequest(
            "Von input forged an option marker".into(),
        ));
    }
    let state_positions = encoded
        .get_offsets()
        .iter()
        .enumerate()
        .filter_map(|(i, (a, b))| {
            (a < b
                && start.saturating_sub(trim_start) <= *a
                && *b <= end.saturating_sub(trim_start))
            .then_some(i)
        })
        .collect();
    Ok(Encoded {
        ids,
        markers,
        state_positions,
    })
}
pub(super) fn count_state(tokenizer: &Tokenizer, text: &str, vocab: usize) -> Result<usize> {
    Ok(token_ids(tokenizer, text, vocab, "Von")?.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decimal_digit_runs_match_python() {
        assert_eq!(
            split_digits("2026 ١٢ ²³ 12.34").expect("digit pattern must compile"),
            "2 0 2 6 ١ ٢ ²³ 1 2.3 4"
        );
    }
}
