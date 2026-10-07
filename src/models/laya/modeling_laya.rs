//! Laya's processor, ModernBERT backbone, and decision head form one model family.

use std::iter::repeat_n;

use bon::bon;
use burn::module::{Module, Param};
use burn::nn::{
    Embedding,
    EmbeddingConfig,
    Gelu,
    Initializer,
    LayerNorm,
    LayerNormConfig,
    Linear,
    LinearConfig,
};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{Bool, Device as BurnDevice, Int, Tensor, TensorData};
use burn_std::s;
use camino::Utf8Path;
use indexmap::IndexMap;

use super::configuration_laya::LayaConfig;
use super::processing_laya::{Batch, LayaProcessor, kind};
use super::weights::load_laya;
use crate::hub::{ModelSource, resolve};
use crate::models::modernbert::{ModernBertConfig, ModernBertModel};
use crate::utils::attention::attend;
use crate::utils::{read_checkpoint_json, render};
use crate::{Action, Answer, DecisionModel, Error, Metadata, Question, Request, Response, Result};

// Keep the artifact layout with Laya so other architectures can supply their own.
pub(crate) const REQUIRED_ARTIFACTS: [&str; 5] = [
    "rl_agent_config.json",
    "encoder/config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

/// Own the loaded tensors and tokenizer so successive predictions reuse them.
pub struct LayaModel {
    architecture: LayaDecisionModel,
    processor: LayaProcessor,
    config: LayaConfig,
    metadata: Metadata,
    device: BurnDevice,
}
pub(crate) struct RawOutput {
    pub logits: Vec<Vec<f32>>,
    pub actions: Vec<f32>,
}

#[bon]
impl LayaModel {
    /// Load reusable pretrained tensors, e.g. `Self::new(root, device)?`.
    #[builder(start_fn = builder)]
    pub fn new(root: &Utf8Path, device: &BurnDevice) -> Result<Self> {
        Self::from_pretrained(root, device)
    }

    /// Load a local checkpoint onto a caller-selected Burn backend.
    ///
    /// For example use `LayaModel::from_pretrained(path, &burn::tensor::Device::flex())`.
    pub fn from_pretrained(root: &Utf8Path, device: &BurnDevice) -> Result<Self> {
        let artifacts = resolve(&ModelSource::Local(root.into()), &REQUIRED_ARTIFACTS)?;
        Self::load(root, device, artifacts.metadata)
    }

    pub(crate) fn load(
        root: &Utf8Path,
        device: &BurnDevice,
        mut metadata: Metadata,
    ) -> Result<Self> {
        metadata.architecture = "laya".into();
        if metadata.device.is_empty() {
            metadata.device = format!("{device:?}");
        }
        let config: LayaConfig = read_checkpoint_json(&root.join("rl_agent_config.json"))?;
        let encoder_config: ModernBertConfig =
            read_checkpoint_json(&root.join("encoder/config.json"))?;
        encoder_config.validate()?;
        config.validate(&encoder_config)?;
        let processor = LayaProcessor::load(root, &config, &encoder_config)?;
        let mut architecture = LayaDecisionModel::init(&config, &encoder_config, device)?;
        load_laya(&mut architecture, &root.join("model.safetensors"))?;
        Ok(Self {
            architecture,
            processor,
            config,
            metadata,
            device: device.clone(),
        })
    }

    pub(crate) fn forward(&self, batch: &Batch) -> Result<RawOutput> {
        let count = batch.rows.len();
        let length = batch.rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let options = batch
            .rows
            .iter()
            .map(|r| r.markers.len())
            .max()
            .unwrap_or(0);
        if count == 0 {
            return Ok(RawOutput {
                logits: Vec::new(),
                actions: Vec::new(),
            });
        }
        let mut ids = Vec::with_capacity(count * length);
        let mut padding = Vec::with_capacity(count * length);
        let mut types = Vec::with_capacity(count);
        for row in &batch.rows {
            ids.extend(row.ids.iter().map(|&id| i64::from(id)));
            ids.extend(repeat_n(
                i64::from(self.processor.pad),
                length - row.ids.len(),
            ));
            padding.extend(repeat_n(false, row.ids.len()));
            padding.extend(repeat_n(true, length - row.ids.len()));
            types.push(row.kind as i64);
        }
        let mask = Tensor::<4, Bool>::from_data(
            TensorData::new(padding, [count, 1, 1, length]),
            &self.device,
        );
        let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [count, length]), &self.device);
        let types = Tensor::<2, Int>::from_data(TensorData::new(types, [count, 1]), &self.device);
        let [_, hidden_size] = self.architecture.type_emb.weight.shape().dims();
        let mut markers = Vec::with_capacity(count * options * hidden_size);
        for row in &batch.rows {
            for index in 0..options {
                let marker = row.markers.get(index).copied().unwrap_or(0) as i64;
                markers.extend(repeat_n(marker, hidden_size));
            }
        }
        let markers = Tensor::<3, Int>::from_data(
            TensorData::new(markers, [count, options, hidden_size]),
            &self.device,
        );
        let (scores, pooled) = self.architecture.forward(ids, mask, types, markers);
        let scores = scores
            .try_into_vec_as::<f32>()
            .map_err(|e| Error::Inference(e.to_string()))?;
        let mut logits = Vec::with_capacity(count);
        let mut features = Vec::with_capacity(count * 4);
        for (row, values) in batch.rows.iter().zip(scores.chunks_exact(options)) {
            let row_logits: Vec<f32> = values.iter().take(row.markers.len()).copied().collect();
            features.extend(action_features(&row_logits)?);
            logits.push(row_logits);
        }
        let features =
            Tensor::<3>::from_data(TensorData::new(features, [count, 1, 4]), &self.device);
        let action_probabilities = self.architecture.action_probabilities(pooled, features);
        let action_count = action_probabilities.dims()[2];
        let action_values = action_probabilities
            .try_into_vec_as::<f32>()
            .map_err(|e| Error::Inference(e.to_string()))?;
        // Reject backend overflow instead of serializing a NaN confidence as null.
        if action_values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Inference("non-finite action probabilities".into()));
        }
        let actions = action_values
            .chunks_exact(action_count)
            .filter_map(|row| row.first().copied())
            .collect();
        Ok(RawOutput { logits, actions })
    }
}
impl DecisionModel for LayaModel {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn predict(&self, request: &Request) -> Result<Response> {
        let batch = self.processor.process(request)?;
        let raw = self.forward(&batch)?;
        let mut answers = IndexMap::new();
        for (((id, question), logits), action) in
            request.questions.iter().zip(raw.logits).zip(raw.actions)
        {
            let distribution = probabilities(
                &logits,
                self.config.temperature(kind(question).1, logits.len()),
            )?;
            answers.insert(id.clone(), decode_answer(question, &distribution, action)?);
        }
        Ok(Response {
            model: "laya-rl-agent".into(),
            answers,
            usage: batch.usage,
            metadata: self.metadata.clone(),
        })
    }
}

// Keep action-head features uncalibrated, e.g. temperature=1 even for a calibrated answer.
fn action_features(logits: &[f32]) -> Result<[f32; 4]> {
    let distribution = probabilities(logits, 1.0)?;
    let mut sorted = distribution.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let first = sorted.first().copied().unwrap_or(1.0);
    let second = sorted.get(1).copied().unwrap_or(0.0);
    let option_count = logits.len().max(2) as f32;
    let entropy = -distribution
        .iter()
        .map(|value| value * value.max(1e-9).ln())
        .sum::<f32>()
        / option_count.ln();
    Ok([first, first - second, entropy, option_count / 255.0])
}

// Decode calibrated probabilities separately from backend execution; e.g. score is
// the expected level index, while noul always reports the probability of true.
fn decode_answer(question: &Question, distribution: &[f32], action: f32) -> Result<Answer> {
    let action = Action {
        act_probability: round4(f64::from(action)),
    };
    let maximum = distribution.iter().copied().fold(0.0_f32, f32::max);
    let answer_confidence = round4(f64::from(maximum));
    let confidence = if distribution.len() < 2 {
        1.0
    } else {
        let entropy = -distribution
            .iter()
            .map(|value| value * value.clamp(1e-12, 1.0).ln())
            .sum::<f32>();
        round4(f64::from(
            (1.0_f32 - entropy / (distribution.len() as f32).ln()).clamp(0.0, 1.0),
        ))
    };
    let answer = match question {
        Question::Choice { criteria, .. } => {
            let selected = distribution
                .iter()
                .position(|value| *value >= maximum)
                .unwrap_or(0);
            let choice = criteria
                .get_index(selected)
                .ok_or_else(|| Error::Inference("missing choice index".into()))?
                .0
                .clone();
            let probabilities = criteria
                .keys()
                .cloned()
                .zip(distribution.iter().map(|value| round4(f64::from(*value))))
                .collect();
            Answer::Choice {
                choice,
                probabilities,
                confidence,
                answer_confidence,
                action,
            }
        }
        Question::Score { criteria, .. } => {
            let score = round4(
                distribution
                    .iter()
                    .enumerate()
                    .map(|(index, value)| index as f64 * f64::from(*value))
                    .sum(),
            );
            let legend = criteria
                .iter()
                .enumerate()
                .map(|(index, value)| Ok((index.to_string(), render(value)?)))
                .collect::<Result<_>>()?;
            let probabilities = distribution
                .iter()
                .enumerate()
                .map(|(index, value)| (index.to_string(), round4(f64::from(*value))))
                .collect();
            Answer::Score {
                score,
                legend,
                probabilities,
                confidence,
                answer_confidence,
                action,
            }
        }
        Question::Noul { .. } => {
            let true_probability = distribution
                .get(1)
                .copied()
                .ok_or_else(|| Error::Inference("missing true probability".into()))?;
            Answer::Noul {
                noul: round4(f64::from(true_probability)),
                confidence: round4(
                    f64::from(true_probability).max(1.0 - f64::from(true_probability)),
                ),
                answer_confidence,
                action,
            }
        }
    };
    Ok(answer)
}

fn probabilities(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err(Error::Inference("empty or non-finite logits".into()));
    }
    let scaled: Vec<f32> = logits.iter().map(|v| v / temperature).collect();
    let maximum = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = scaled.iter().map(|v| (v - maximum).exp()).collect();
    let sum: f32 = p.iter().sum();
    for value in &mut p {
        *value /= sum;
    }
    Ok(p)
}
fn round4(value: f64) -> f64 {
    (value * 10000.0).round_ties_even() / 10000.0
}

/// Initialize without a tokenizer or weight file, then apply [`super::weights::load_laya`].
///
/// For example: `LayaDecisionModel::init(&config, &encoder, &device)`.
#[derive(Module, Debug)]
pub struct LayaDecisionModel {
    pub(crate) encoder: ModernBertModel,
    head: LayaHead,
    pub(crate) type_emb: Embedding,
    // Tuples preserve Sequential indices, e.g. scorer.3.weight, including parameterless GELU.
    scorer: (LayerNorm, Linear, Gelu, Linear),
    act_head: (Linear, Gelu, Linear),
    // Keep the checkpoint buffer; runtime calibration is defined by config.
    pub(crate) temperature: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct LayaHead {
    layers: Vec<LayaHeadLayer>,
}

#[derive(Module, Debug)]
struct LayaHeadLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    self_attn: LayaSelfAttention,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
}

#[derive(Module, Debug)]
struct LayaSelfAttention {
    // Keep Burn's fused Linear so loading handles the PyTorch [3d, d] transpose once.
    in_proj: Linear,
    out_proj: Linear,
}

#[bon]
impl LayaDecisionModel {
    /// Initialize validated head and encoder dimensions, e.g. `Self::new(config, encoder, device)?`.
    #[builder(start_fn = builder)]
    pub fn new(
        config: &LayaConfig,
        encoder: &ModernBertConfig,
        device: &BurnDevice,
    ) -> Result<Self> {
        Self::init(config, encoder, device)
    }

    /// Compute option logits and the pooled state used by the action head.
    ///
    /// Shapes: IDs `[batch, length]`, boolean padding `[batch, 1, 1, length]`,
    /// types `[batch, 1]`, markers `[batch, options, hidden]`.
    /// Padding is `true` for blocked tokens, e.g. `[false, false, true]`.
    /// Returns logits `[batch, options, 1]` and pooled state `[batch, 1, hidden]`.
    pub fn forward(
        &self,
        ids: Tensor<2, Int>,
        padding: Tensor<4, Bool>,
        types: Tensor<2, Int>,
        markers: Tensor<3, Int>,
    ) -> (Tensor<3>, Tensor<3>) {
        let mut hidden = self.encoder.forward(ids, padding.clone()) + self.type_emb.forward(types);
        for layer in &self.head.layers {
            hidden = layer.forward(hidden, padding.clone());
        }
        let selected = hidden.clone().gather(1, markers);
        let (norm, input, activation, output) = &self.scorer;
        let scores = output.forward(activation.forward(input.forward(norm.forward(selected))));
        // Pool the first token without narrowing the batch or hidden channels.
        (scores, hidden.slice(s![.., 0..1, ..]))
    }

    /// Apply the action head to pooled state and four reference confidence features.
    /// For example, features have shape `[batch, 1, 4]`.
    pub fn action_probabilities(&self, pooled: Tensor<3>, features: Tensor<3>) -> Tensor<3> {
        let (input, activation, output) = &self.act_head;
        softmax(
            output
                .forward(activation.forward(input.forward(Tensor::cat(vec![pooled, features], 2)))),
            2,
        )
    }

    /// Validate dimensions before creating lazy Burn parameters.
    pub fn init(
        config: &LayaConfig,
        encoder: &ModernBertConfig,
        device: &BurnDevice,
    ) -> Result<Self> {
        encoder.validate()?;
        config.validate(encoder)?;
        let hidden_size = encoder.hidden_size;
        let linear = |input, output| LinearConfig::new(input, output).init(device);
        let norm = || LayerNormConfig::new(hidden_size).init(device);
        Ok(Self {
            encoder: ModernBertModel::init(encoder, device)?,
            head: LayaHead {
                layers: (0..config.head_layers)
                    .map(|_| LayaHeadLayer {
                        norm1: norm(),
                        norm2: norm(),
                        self_attn: LayaSelfAttention {
                            in_proj: linear(hidden_size, 3 * hidden_size),
                            out_proj: linear(hidden_size, hidden_size),
                        },
                        linear1: linear(hidden_size, 4 * hidden_size),
                        linear2: linear(4 * hidden_size, hidden_size),
                        heads: (hidden_size / 64).max(1),
                    })
                    .collect(),
            },
            type_emb: EmbeddingConfig::new(3, hidden_size).init(device),
            scorer: (
                norm(),
                linear(hidden_size, hidden_size),
                Gelu::new(),
                linear(hidden_size, 1),
            ),
            act_head: (
                linear(hidden_size + 4, 256),
                Gelu::new(),
                linear(256, config.act_costs.len() + 1),
            ),
            temperature: Initializer::Ones.init([3], device),
        })
    }
}

impl LayaHeadLayer {
    // Keep each residual update with its normalization and projection so layer
    // ordering is explicit, e.g. attention runs before the feed-forward network.
    fn forward(&self, hidden: Tensor<3>, padding: Tensor<4, Bool>) -> Tensor<3> {
        let normalized = self.norm1.forward(hidden.clone());
        let attention = attend(
            self.self_attn.in_proj.forward(normalized),
            self.heads,
            padding,
            None,
        );
        let hidden = hidden + self.self_attn.out_proj.forward(attention);

        // Match TransformerEncoderLayer's ReLU; the scorer and action head use GELU.
        let normalized = self.norm2.forward(hidden.clone());
        let feed_forward = self.linear2.forward(relu(self.linear1.forward(normalized)));
        hidden + feed_forward
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::Value;
    #[cfg(feature = "cpu")]
    use serde_json::json;

    use super::*;
    use crate::models::laya::processing_laya::Encoded;
    use crate::utils::read;

    fn verify_reference(device: BurnDevice, logit_epsilon: f64, answer_epsilon: f64) {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
        let model = LayaModel::from_pretrained(&root, &device).unwrap();
        let fixtures: Value =
            serde_json::from_slice(&read(&root.parent().unwrap().join("reference.json")).unwrap())
                .unwrap();
        for case in fixtures["cases"].as_array().unwrap() {
            let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
            let batch = model.processor.process(&request).unwrap();
            // Verify all native PyTorch sequences directly; JSON spacing can change our tokens.
            // For example, Python may encode adjacent punctuation differently in {"a": 1}.
            let reference_batch = Batch {
                rows: request
                    .questions
                    .values()
                    .zip(case["encoded"].as_array().unwrap())
                    .map(|(question, row)| Encoded {
                        ids: serde_json::from_value(row["ids"].clone()).unwrap(),
                        markers: serde_json::from_value(row["markers"].clone()).unwrap(),
                        kind: kind(question).1,
                    })
                    .collect(),
                usage: Default::default(),
            };
            assert_eq!(batch.rows.len(), reference_batch.rows.len());
            let same_tokens =
                batch
                    .rows
                    .iter()
                    .zip(&reference_batch.rows)
                    .all(|(actual, expected)| {
                        actual.ids == expected.ids && actual.markers == expected.markers
                    });
            let text_only = request.state.is_string()
                && request.questions.values().all(|question| match question {
                    Question::Choice { criteria, .. } | Question::Noul { criteria, .. } => criteria
                        .values()
                        .all(|value| value.is_string() || value.is_null()),
                    Question::Score { criteria, .. } => criteria.iter().all(Value::is_string),
                });
            if text_only {
                assert!(same_tokens);
            }
            let raw = model.forward(&reference_batch).unwrap();
            for (actual, expected) in raw.logits.iter().zip(case["logits"].as_array().unwrap()) {
                for (actual, expected) in actual.iter().zip(expected.as_array().unwrap()) {
                    assert!(
                        (f64::from(*actual) - expected.as_f64().unwrap()).abs() < logit_epsilon,
                        "logit {actual} != {expected}"
                    );
                }
            }
            for (actual, expected) in raw.actions.iter().zip(case["actions"].as_array().unwrap()) {
                assert!((f64::from(*actual) - expected.as_f64().unwrap()).abs() < logit_epsilon);
            }
            let mut actual = serde_json::to_value(model.predict(&request).unwrap()).unwrap();
            let mut expected = case["response"]["answers"].clone();
            // Compare JSON-valued legends by content, e.g. {"a":1} equals {"a": 1}.
            for answers in [&mut actual["answers"], &mut expected] {
                for answer in answers.as_object_mut().unwrap().values_mut() {
                    if let Some(Value::Object(legend)) = answer.get_mut("legend") {
                        for text in legend.values_mut() {
                            if let Ok(value) = serde_json::from_str::<Value>(text.as_str().unwrap())
                            {
                                *text = value;
                            }
                        }
                    }
                }
            }
            // Legends retain their values even when the prompts produce different predictions.
            for (id, answer) in expected.as_object().unwrap() {
                if let Some(legend) = answer.get("legend") {
                    assert_eq!(actual["answers"][id]["legend"], *legend);
                }
            }
            if same_tokens {
                compare(&actual["answers"], &expected, answer_epsilon);
            }
            assert_eq!(actual["usage"]["input_tokens"], batch.usage.input_tokens);
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_matches_pytorch_on_reference_tokens() {
        verify_reference(BurnDevice::flex(), 2e-5, 1.1e-4);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_pytorch_on_reference_tokens() {
        verify_reference(BurnDevice::wgpu(Default::default()), 4e-4, 4e-4);
    }

    fn compare(actual: &Value, expected: &Value, epsilon: f64) {
        match expected {
            Value::Object(fields) => {
                for (key, value) in fields {
                    compare(&actual[key], value, epsilon);
                }
            }
            Value::Number(value) => assert!(
                (actual.as_f64().expect("actual numeric answer")
                    - value.as_f64().expect("reference numeric answer"))
                .abs()
                    < epsilon,
                "{actual} != {expected}"
            ),
            _ => assert_eq!(actual, expected),
        }
    }
    proptest! {
        #[test]
        fn probabilities_form_a_distribution(logits in prop::collection::vec(-100_f32..100_f32,1..24), temperature in 0.5_f32..5_f32) {
            let p = probabilities(&logits,temperature).unwrap();
            prop_assert!(p.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
            prop_assert!((p.iter().sum::<f32>()-1.0).abs()<1e-5);
        }
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn default_truncation_refuses_loss_of_information() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
        let model = LayaModel::from_pretrained(&root, &BurnDevice::flex()).unwrap();
        let request: Request = serde_json::from_value(json!({"state":"alpha ".repeat(100),"questions":{"q":{"type":"noul","instructions":"cancel?"}}})).unwrap();
        assert!(matches!(
            model.predict(&request),
            Err(Error::InvalidRequest(_))
        ));
    }
}
