//! Load and evaluate Vev's trained text decisions, e.g. CountingSheep/vev-4b.
use burn::tensor::Device as BurnDevice;
use camino::Utf8Path;
use indexmap::IndexMap;

use super::configuration_vev::VevConfig;
use super::processing_vev::{prompt, render_state};
use crate::hub::{self, ModelSource};
use crate::models::qwen3_5::readout::{Readout, answer, choice_confidence};
use crate::utils::{read_checkpoint_json, sanitize};
use crate::{DecisionModel, Error, Metadata, Question, Request, Response, Result, Usage};

/// Reuse the loaded text readout and label tokens, e.g. successive Vev requests.
pub struct VevModel {
    readout: Readout,
    labels: Vec<String>,
    label_ids: Vec<Vec<u32>>,
    metadata: Metadata,
}

impl VevModel {
    /// Load a merged release, e.g. `VevModel::from_pretrained(&source, &device)?`.
    pub fn from_pretrained(source: &ModelSource, device: &BurnDevice) -> Result<Self> {
        let artifacts = hub::resolve_vev(source)?;
        Self::load(&artifacts.root, device, artifacts.metadata)
    }

    pub(crate) fn load(
        root: &Utf8Path,
        device: &BurnDevice,
        mut metadata: Metadata,
    ) -> Result<Self> {
        let config: VevConfig = read_checkpoint_json(&root.join("vev.json"))?;
        config.validate()?;
        let mut readout = Readout::load(root, device)?;
        readout.max_positions = readout.max_positions.min(32768);
        let mut labels = Vec::new();
        let mut label_ids = Vec::new();
        let candidates =
            (b'A'..=b'Z')
                .map(|a| char::from(a).to_string())
                .chain((b'A'..=b'Z').flat_map(|a| {
                    (b'A'..=b'Z').map(move |b| format!("{}{}", char::from(a), char::from(b)))
                }));
        for label in candidates {
            let ids = readout.tokens(&label)?;
            if ids.len() == 1 && !label_ids.contains(&ids) {
                labels.push(label);
                label_ids.push(ids);
            }
            if labels.len() == 255 {
                break;
            }
        }
        if labels.len() != 255 {
            return Err(Error::InvalidCheckpoint(
                "Vev needs 255 distinct single-token choice labels".into(),
            ));
        }
        metadata.architecture = "vev".into();
        if metadata.device.is_empty() {
            metadata.device = format!("{device:?}");
        }
        Ok(Self {
            readout,
            labels,
            label_ids,
            metadata,
        })
    }
}

impl DecisionModel for VevModel {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn predict(&self, request: &Request) -> Result<Response> {
        self.readout.validate(request)?;
        for question in request.questions.values() {
            validate_question(question)?;
        }
        let state = sanitize(&render_state(&request.state)?);
        let mut usage = Usage {
            state_tokens: self.readout.tokens(&state)?.len(),
            ..Default::default()
        };
        let mut answers = IndexMap::new();
        for (id, question) in &request.questions {
            let (prompt, range, legend) = prompt(&state, question, &self.labels)?;
            let groups: Vec<Vec<u32>> = match question {
                Question::Choice { criteria, .. } => self
                    .label_ids
                    .iter()
                    .take(criteria.len())
                    .cloned()
                    .collect(),
                Question::Score { criteria, .. } => (0..criteria.len())
                    .map(|i| single(&self.readout, &i.to_string()))
                    .collect::<Result<_>>()?,
                Question::Noul { .. } => ["Yes", "No"]
                    .iter()
                    .map(|label| single(&self.readout, label))
                    .collect::<Result<_>>()?,
            };
            let probabilities =
                self.readout
                    .probabilities(&prompt, &[range], &groups, request, id, &mut usage)?;
            let confidence = choice_confidence(&probabilities);
            answers.insert(
                id.clone(),
                answer(question, probabilities, confidence, legend, 0)?,
            );
        }
        Ok(Response {
            model: self.metadata.model_id.clone(),
            answers,
            usage,
            metadata: self.metadata.clone(),
        })
    }
}

fn single(readout: &Readout, label: &str) -> Result<Vec<u32>> {
    let ids = readout.tokens(label)?;
    if ids.len() != 1 {
        return Err(Error::InvalidCheckpoint(format!(
            "Vev answer {label:?} must be a single token"
        )));
    }
    Ok(ids)
}

fn validate_question(question: &Question) -> Result<()> {
    let invalid = match question {
        Question::Choice { criteria, .. } => criteria.len() > 255,
        Question::Score { criteria, .. } => criteria.len() > 10,
        Question::Noul { .. } => false,
    };
    if invalid {
        return Err(Error::InvalidRequest(
            "Vev supports at most 255 choice options and 10 score levels".into(),
        ));
    }
    Ok(())
}
#[cfg(all(test, feature = "cpu"))]
mod tests {
    use serde_json::Value;

    use super::*;

    #[test]
    fn text_prompts_match_pinned_vev_renderer() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-vev-4b");
        let model = VevModel::load(
            &root,
            &BurnDevice::flex(),
            Metadata::new("fixture", "vev", "cpu"),
        )
        .unwrap();
        let reference: Value = read_checkpoint_json(&root.join("reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
            // Keep upstream wording checks for text; JSON rendering is covered by value checks.
            // For example, a structured criterion may now use compact separators.
            if !request.state.is_string() {
                continue;
            }
            let state = sanitize(&render_state(&request.state).unwrap());
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
                assert_eq!(
                    prompt(&state, question, &model.labels).unwrap().0,
                    row["prompt"]
                );
            }
        }
    }
}
