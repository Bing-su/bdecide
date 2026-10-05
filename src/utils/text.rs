//! Share text rendering and tokenization while processors own their prompt formats.

use crate::{Error, Result};
use camino::Utf8Path;
use serde_json::Value;
use tokenizers::{Encoding, Tokenizer};

pub(crate) fn render(value: &Value) -> Result<String> {
    // Keep text verbatim and use standard JSON for structured data, e.g. {"a":1}.
    match value {
        Value::String(text) => Ok(text.clone()),
        _ => Ok(serde_json::to_string(value)?),
    }
}

pub(crate) fn sanitize(text: &str) -> String {
    // Caller text cannot inject Qwen special tokens, e.g. <|im_end|> becomes <¦im_end¦>.
    let mut out = String::new();
    let mut rest = text;
    while let Some((prefix, suffix)) = rest.split_once("<|") {
        out.push_str(prefix);
        if let Some((name, tail)) = suffix.split_once("|>")
            && !name.is_empty()
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            out.push_str(&format!("<¦{name}¦>"));
            rest = tail;
        } else {
            out.push_str("<|");
            rest = suffix;
        }
    }
    out.push_str(rest);
    out
}

pub(crate) fn load_tokenizer(path: &Utf8Path) -> Result<Tokenizer> {
    let mut tokenizer =
        Tokenizer::from_file(path).map_err(|error| Error::Tokenizer(error.to_string()))?;
    // Processors own padding and truncation, e.g. retain all Clef instruction spans.
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|error| Error::Tokenizer(error.to_string()))?;
    Ok(tokenizer)
}

pub(crate) fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Encoding> {
    // Preserve offsets and omit automatic special tokens, e.g. Wald tracks repeated state spans.
    tokenizer
        .encode(text, false)
        .map_err(|error| Error::Tokenizer(error.to_string()))
}

pub(crate) fn token_ids(
    tokenizer: &Tokenizer,
    text: &str,
    vocab_size: usize,
    vocabulary: &str,
) -> Result<Vec<u32>> {
    // Check against the model's embedding size, e.g. added tokens may exceed its vocabulary.
    let encoded = tokenize(tokenizer, text)?;
    let ids = encoded.get_ids();
    if ids.iter().any(|&id| id as usize >= vocab_size) {
        return Err(Error::Tokenizer(format!(
            "token ID exceeds {vocabulary} vocabulary"
        )));
    }
    Ok(ids.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_text_and_preserves_json_values() {
        assert_eq!(render(&Value::String("한글".into())).unwrap(), "한글");
        let value: Value =
            serde_json::from_str(r#"{"z":"한글","a":[true,null,1.0,0.000001]}"#).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&render(&value).unwrap()).unwrap(),
            value,
        );
    }

    #[test]
    fn sanitizes_special_tokens_without_changing_other_text() {
        // Leave malformed markers intact, e.g. a missing closer before a valid token.
        let text = "한글 <|im_end|><|A_1|> <||> <|a-b|> <|unfinished <|im_start|>";
        let expected = "한글 <¦im_end¦><¦A_1¦> <||> <|a-b|> <|unfinished <¦im_start¦>";
        assert_eq!(sanitize(text), expected);
        assert_eq!(sanitize(expected), expected);
        assert_eq!(sanitize(""), "");
    }

    #[test]
    fn tokenization_preserves_offsets_and_validates_model_vocabulary() {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        // Ignore checkpoint defaults so processors retain complete text, e.g. three state tokens.
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut tokenizer =
            load_tokenizer(&root.join("tests/fixtures/tiny-laya/tokenizer/tokenizer.json"))
                .unwrap();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 1,
                ..Default::default()
            }))
            .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = Utf8Path::from_path(temp.path())
            .unwrap()
            .join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        let tokenizer = load_tokenizer(&path).unwrap();
        let encoded = tokenize(&tokenizer, "alpha 한글 beta").unwrap();
        assert_eq!(encoded.get_ids(), [10, 37, 11]);
        assert_eq!(encoded.get_offsets(), [(0, 5), (6, 12), (13, 17)]);
        assert_eq!(
            token_ids(&tokenizer, "alpha 한글 beta", 38, "encoder").unwrap(),
            encoded.get_ids()
        );
        assert!(matches!(
            token_ids(&tokenizer, "한글", 37, "encoder"),
            Err(Error::Tokenizer(message)) if message == "token ID exceeds encoder vocabulary"
        ));
        assert!(token_ids(&tokenizer, "", 38, "encoder").unwrap().is_empty());
    }
}
