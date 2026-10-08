use std::ops::Range;

use bon::bon;
use burn::module::{Module, Param};
use burn::nn::{
    Dropout,
    DropoutConfig,
    Embedding,
    EmbeddingConfig,
    Gelu,
    Initializer,
    LayerNorm,
    LayerNormConfig,
    Linear,
    LinearConfig,
};
use burn::tensor::activation::{gelu, sigmoid, softmax};
use burn::tensor::{Device as BurnDevice, Int, Tensor, TensorData};
use burn_std::s;
use camino::Utf8Path;
use indexmap::IndexMap;

use super::weights::load_clef;
use super::{ClefConfig, ClefProcessor, EncodedRecord};
use crate::hub::{ModelSource, resolve_clef};
use crate::models::qwen3_5::{Qwen3_5Config, Qwen3_5ForCausalLM};
use crate::utils::attention::attention;
use crate::utils::{read_checkpoint_json, render};
use crate::{Action, Answer, DecisionModel, Error, Metadata, Question, Request, Response, Result};

/// Own Clef's pretrained backbone, head, and processor for repeated predictions.
pub struct ClefModel {
    architecture: ClefDecisionModel,
    processor: ClefProcessor,
    metadata: Metadata,
    device: BurnDevice,
}

/// Match the released Clef model while reusing the Qwen3.5 text backbone.
#[derive(Module, Debug)]
pub struct ClefDecisionModel {
    pub(super) backbone: Qwen3_5ForCausalLM,
    pub(super) head: JointSchemaHead,
}

#[derive(Module, Debug)]
pub(super) struct JointSchemaHead {
    hidden_norm: LayerNorm,
    memory_projection: Linear,
    question_projection: Linear,
    option_question_projection: Linear,
    global_projection: Linear,
    option_context_projection: Linear,
    option_lexical_projection: Linear,
    type_embedding: Embedding,
    evidence_layers: Vec<EvidenceRoutingLayer>,
    option_summary_norm: LayerNorm,
    layers: Vec<JointDecoderLayer>,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    residual_scorer: (Linear, Gelu, Dropout, Linear),
    prior_logit_scale: Param<Tensor<1>>,
    joint_logit_scale: Param<Tensor<1>>,
    residual_gate: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct EvidenceRoutingLayer {
    query_norm: LayerNorm,
    memory_norm: LayerNorm,
    attention: MultiheadAttention,
    feedforward_norm: LayerNorm,
    feedforward: (Linear, Gelu, Dropout, Linear, Dropout),
}
#[derive(Module, Debug)]
struct JointDecoderLayer {
    self_attn: MultiheadAttention,
    multihead_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}
#[derive(Module, Debug)]
struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    heads: usize,
}
impl MultiheadAttention {
    fn forward(&self, query: Tensor<3>, memory: Tensor<3>) -> Tensor<3> {
        let [batch, queries, width] = query.dims();
        let length = memory.dims()[1];
        let dim = width / self.heads;
        let query = self
            .in_proj
            .forward(query)
            .slice(s![.., .., ..width])
            .reshape([batch, queries, self.heads, dim])
            .swap_dims(1, 2);
        let memory = self.in_proj.forward(memory);
        let key = memory
            .clone()
            .slice(s![.., .., width..2 * width])
            .reshape([batch, length, self.heads, dim])
            .swap_dims(1, 2);
        let value = memory
            .slice(s![.., .., 2 * width..])
            .reshape([batch, length, self.heads, dim])
            .swap_dims(1, 2);
        self.out_proj.forward(
            attention(query, key, value, None, Default::default())
                .swap_dims(1, 2)
                .reshape([batch, queries, width]),
        )
    }
}
#[bon]
impl ClefDecisionModel {
    /// Initialize validated head and backbone dimensions, e.g. `Self::new(config, backbone, device)?`.
    #[builder(start_fn = builder)]
    pub fn new(config: &ClefConfig, backbone: &Qwen3_5Config, device: &BurnDevice) -> Result<Self> {
        Self::init(config, backbone, device)
    }

    pub fn init(
        config: &ClefConfig,
        backbone: &Qwen3_5Config,
        device: &BurnDevice,
    ) -> Result<Self> {
        let mut model = Self::init_for_loading(config, backbone, device)?;
        model.backbone.tie_weights();
        Ok(model)
    }

    // Defer tying until weights are loaded, e.g. skip random vocabulary allocation on load.
    fn init_for_loading(
        config: &ClefConfig,
        backbone: &Qwen3_5Config,
        device: &BurnDevice,
    ) -> Result<Self> {
        config.validate(backbone)?;
        let hidden = config.hidden_size;
        let width = config.width;
        let linear = |i, o, bias| LinearConfig::new(i, o).with_bias(bias).init(device);
        let norm = || LayerNormConfig::new(width).with_epsilon(1e-5).init(device);
        let attention = || MultiheadAttention {
            in_proj: linear(width, 3 * width, true),
            out_proj: linear(width, width, true),
            heads: config.heads,
        };
        let dropout = || DropoutConfig::new(0.0).init();
        let projection = || linear(hidden, width, false);
        // Clef's wrapper owns tying, e.g. root tie_word_embeddings may differ from text_config.
        let mut text_config = backbone.text_config.clone();
        text_config.tie_word_embeddings = backbone.tie_word_embeddings;
        Ok(Self {
            backbone: Qwen3_5ForCausalLM::init(&text_config, device)?,
            head: JointSchemaHead {
                hidden_norm: LayerNormConfig::new(hidden).with_epsilon(1e-5).init(device),
                memory_projection: projection(),
                question_projection: projection(),
                option_question_projection: projection(),
                global_projection: projection(),
                option_context_projection: projection(),
                option_lexical_projection: projection(),
                type_embedding: EmbeddingConfig::new(3, width).init(device),
                evidence_layers: (0..config.routing_layers)
                    .map(|_| EvidenceRoutingLayer {
                        query_norm: norm(),
                        memory_norm: norm(),
                        attention: attention(),
                        feedforward_norm: norm(),
                        feedforward: (
                            linear(width, config.feedforward, true),
                            Gelu::new(),
                            dropout(),
                            linear(config.feedforward, width, true),
                            dropout(),
                        ),
                    })
                    .collect(),
                option_summary_norm: norm(),
                layers: (0..config.layers)
                    .map(|_| JointDecoderLayer {
                        self_attn: attention(),
                        multihead_attn: attention(),
                        linear1: linear(width, config.feedforward, true),
                        linear2: linear(config.feedforward, width, true),
                        norm1: norm(),
                        norm2: norm(),
                        norm3: norm(),
                    })
                    .collect(),
                field_norm: norm(),
                option_norm: norm(),
                residual_scorer: (
                    linear(4 * width, width, true),
                    Gelu::new(),
                    dropout(),
                    linear(width, 1, true),
                ),
                prior_logit_scale: Initializer::Zeros.init([1], device),
                joint_logit_scale: Initializer::Zeros.init([1], device),
                residual_gate: Initializer::Zeros.init([1], device),
            },
        })
    }

    /// Return one logit tensor per question for a single unpadded record.
    ///
    /// Use the IDs and spans from `ClefProcessor::process`, e.g. each option span
    /// indexes the corresponding IDs in the supplied `[1, sequence_length]` tensor.
    pub fn forward(&self, ids: Tensor<2, Int>, record: &EncodedRecord) -> Vec<Tensor<3>> {
        let hidden = self
            .head
            .hidden_norm
            .forward(self.backbone.model.forward(ids.clone()));
        let lexical = self.backbone.embed_output(ids);
        self.head.forward(hidden, lexical, record)
    }
}

fn unit(input: Tensor<3>, eps: f64) -> Tensor<3> {
    input.clone() / input.square().sum_dim(2).sqrt().clamp_min(eps)
}
fn mean_span(values: &Tensor<3>, span: &Range<usize>) -> Tensor<3> {
    values
        .clone()
        .slice(s![.., span.start..span.end, ..])
        .mean_dim(1)
}

impl JointSchemaHead {
    fn forward(
        &self,
        hidden: Tensor<3>,
        lexical: Tensor<3>,
        record: &EncodedRecord,
    ) -> Vec<Tensor<3>> {
        let memory = self.memory_projection.forward(hidden.clone());
        let length = hidden.dims()[1];
        let global = hidden.clone().slice(s![.., length - 1..length, ..]);
        let question_vectors = Tensor::cat(
            record
                .questions
                .iter()
                .map(|q| mean_span(&hidden, &q.question_span))
                .collect(),
            1,
        );
        let mut lexical_options = Vec::new();
        let mut counts = Vec::new();
        let mut queries = Vec::new();
        for (index, question) in record.questions.iter().enumerate() {
            let context = Tensor::cat(
                question
                    .option_spans
                    .iter()
                    .map(|span| mean_span(&hidden, span))
                    .collect(),
                1,
            );
            let lexical = Tensor::cat(
                question
                    .option_spans
                    .iter()
                    .map(|span| mean_span(&lexical, span))
                    .collect(),
                1,
            );
            queries.push(
                self.option_context_projection.forward(context)
                    + self.option_lexical_projection.forward(lexical.clone())
                    + self
                        .option_question_projection
                        .forward(question_vectors.clone().slice(s![.., index..index + 1, ..])),
            );
            lexical_options.push(lexical);
            counts.push(question.option_spans.len());
        }
        let mut routed = Tensor::cat(queries, 1);
        for layer in &self.evidence_layers {
            let attention = layer.attention.forward(
                layer.query_norm.forward(routed.clone()),
                layer.memory_norm.forward(memory.clone()),
            );
            routed = routed + attention;
            let normalized = layer.feedforward_norm.forward(routed.clone());
            routed = routed
                + layer.feedforward.3.forward(
                    layer
                        .feedforward
                        .1
                        .forward(layer.feedforward.0.forward(normalized)),
                );
        }
        let base_fields = self.question_projection.forward(question_vectors.clone());
        let width = base_fields.dims()[2];
        let mut split = Vec::new();
        let mut summaries = Vec::new();
        let mut offset = 0;
        for (index, count) in counts.into_iter().enumerate() {
            let options = routed.clone().slice(s![.., offset..offset + count, ..]);
            offset += count;
            let field = base_fields.clone().slice(s![.., index..index + 1, ..]);
            let weights = softmax(
                (options.clone() * field).sum_dim(2) / (width as f64).sqrt(),
                1,
            );
            summaries.push((weights * options.clone()).sum_dim(1));
            split.push(options);
        }
        let types = Tensor::<2, Int>::from_data(
            TensorData::new(
                record
                    .questions
                    .iter()
                    .map(|q| q.question_type as i64)
                    .collect::<Vec<_>>(),
                [1, record.questions.len()],
            ),
            &hidden.device(),
        );
        let mut fields = base_fields
            + self.option_summary_norm.forward(Tensor::cat(summaries, 1))
            + self.global_projection.forward(global.clone())
            + self.type_embedding.forward(types);
        for layer in &self.layers {
            let normalized = layer.norm1.forward(fields.clone());
            fields = fields + layer.self_attn.forward(normalized.clone(), normalized);
            let normalized = layer.norm2.forward(fields.clone());
            fields = fields + layer.multihead_attn.forward(normalized, memory.clone());
            let normalized = layer.norm3.forward(fields.clone());
            fields = fields
                + layer
                    .linear2
                    .forward(gelu(layer.linear1.forward(normalized)));
        }
        let fields = self.field_norm.forward(fields);
        let prior_scale = self
            .prior_logit_scale
            .val()
            .clamp_max(100.0_f64.ln())
            .exp()
            .unsqueeze::<3>();
        let joint_scale = self
            .joint_logit_scale
            .val()
            .clamp_max(100.0_f64.ln())
            .exp()
            .unsqueeze::<3>();
        let gate = sigmoid(self.residual_gate.val()).unsqueeze::<3>();
        lexical_options
            .into_iter()
            .zip(split)
            .enumerate()
            .map(|(index, (lexical, routed))| {
                let anchor = unit(
                    question_vectors.clone().slice(s![.., index..index + 1, ..]) + global.clone(),
                    1e-12,
                );
                let prior = prior_scale.clone() * (unit(lexical, 1e-12) * anchor).sum_dim(2);
                let options = self.option_norm.forward(routed);
                let field = fields
                    .clone()
                    .slice(s![.., index..index + 1, ..])
                    .expand(options.dims());
                let cosine = (unit(field.clone(), 1e-8) * unit(options.clone(), 1e-8)).sum_dim(2);
                let features = Tensor::cat(
                    vec![
                        field.clone(),
                        options.clone(),
                        field.clone() * options.clone(),
                        (field - options).abs(),
                    ],
                    2,
                );
                let residual = self.residual_scorer.3.forward(
                    self.residual_scorer
                        .1
                        .forward(self.residual_scorer.0.forward(features)),
                );
                prior + gate.clone() * (joint_scale.clone() * cosine + residual)
            })
            .collect()
    }
}

#[bon]
impl ClefModel {
    /// Load reusable pretrained tensors, e.g. `Self::new(root, device)?`.
    #[builder(start_fn = builder)]
    pub fn new(root: &Utf8Path, device: &BurnDevice) -> Result<Self> {
        Self::from_pretrained(root, device)
    }

    /// Load either release's local artifacts, e.g. `ClefModel::from_pretrained`.
    pub fn from_pretrained(root: &Utf8Path, device: &BurnDevice) -> Result<Self> {
        let artifacts = resolve_clef(&ModelSource::Local(root.into()))?;
        Self::load(root, device, artifacts.metadata)
    }

    pub(crate) fn load(
        root: &Utf8Path,
        device: &BurnDevice,
        mut metadata: Metadata,
    ) -> Result<Self> {
        let backbone: Qwen3_5Config = read_checkpoint_json(&root.join("config.json"))?;
        let config: ClefConfig = read_checkpoint_json(&root.join("joint_head_config.json"))?;
        config.validate(&backbone)?;
        let processor = ClefProcessor::from_pretrained(root, &backbone)?;
        let mut architecture = ClefDecisionModel::init_for_loading(&config, &backbone, device)?;
        load_clef(&mut architecture, root)?;
        metadata.architecture = "clef".into();
        if metadata.device.is_empty() {
            metadata.device = format!("{device:?}");
        }
        Ok(Self {
            architecture,
            processor,
            metadata,
            device: device.clone(),
        })
    }

    fn forward(&self, record: &EncodedRecord) -> Result<Vec<Vec<f32>>> {
        let ids = Tensor::<2, Int>::from_data(
            TensorData::new(
                record
                    .input_ids
                    .iter()
                    .map(|&id| i64::from(id))
                    .collect::<Vec<_>>(),
                [1, record.input_ids.len()],
            ),
            &self.device,
        );
        self.architecture
            .forward(ids, record)
            .into_iter()
            .map(|logits| {
                logits
                    .try_into_vec_as::<f32>()
                    .map_err(|e| Error::Inference(e.to_string()))
            })
            .collect()
    }
}

impl DecisionModel for ClefModel {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn system_one(&self, request: &Request) -> Result<Response> {
        let record = self.processor.process(request)?;
        let logits = self.forward(&record)?;
        let mut answers = IndexMap::new();
        for (((id, question), encoded), logits) in
            request.questions.iter().zip(&record.questions).zip(logits)
        {
            let probabilities = probabilities(&logits)?;
            let distribution: IndexMap<_, _> = encoded
                .option_ids
                .iter()
                .cloned()
                .zip(probabilities)
                .collect();
            let confidence = distribution.values().copied().fold(0.0, f64::max);
            // Clef always answers; it has no Laya action/abstention head.
            // An act_probability of 1 preserves the repository's shared response contract.
            let action = Action {
                act_probability: 1.0,
            };
            let answer = match question {
                Question::Noul { .. } => Answer::Noul {
                    noul: round(
                        *distribution
                            .get("true")
                            .ok_or_else(|| Error::Inference("missing true probability".into()))?,
                    ),
                    confidence: round(confidence),
                    answer_confidence: round(confidence),
                    action,
                },
                Question::Choice { criteria, .. } => {
                    let mut choice = None;
                    let mut selected_probability = -1.0;
                    let mut ordered = IndexMap::new();
                    for key in criteria.keys() {
                        let probability = *distribution.get(key).ok_or_else(|| {
                            Error::Inference(format!("missing probability for {key}"))
                        })?;
                        if probability > selected_probability {
                            choice = Some(key.clone());
                            selected_probability = probability;
                        }
                        ordered.insert(key.clone(), round(probability));
                    }
                    Answer::Choice {
                        choice: choice.ok_or_else(|| Error::Inference("empty choice".into()))?,
                        confidence: round(selected_probability),
                        answer_confidence: round(confidence),
                        action,
                        probabilities: ordered,
                    }
                }
                Question::Score { criteria, .. } => Answer::Score {
                    score: round(
                        distribution
                            .values()
                            .enumerate()
                            .map(|(i, probability)| i as f64 * probability)
                            .sum(),
                    ),
                    confidence: round(confidence),
                    answer_confidence: round(confidence),
                    action,
                    legend: criteria
                        .iter()
                        .enumerate()
                        .map(|(i, value)| Ok((i.to_string(), render(value)?)))
                        .collect::<Result<_>>()?,
                    probabilities: distribution
                        .into_iter()
                        .map(|(key, value)| (key, round(value)))
                        .collect(),
                },
            };
            answers.insert(id.clone(), answer);
        }
        Ok(Response {
            model: self.metadata.model_id.clone(),
            answers,
            usage: record.usage,
            metadata: self.metadata.clone(),
        })
    }
}
fn probabilities(logits: &[f32]) -> Result<Vec<f64>> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::Inference("non-finite or empty Clef logits".into()));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let values: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = values.iter().sum();
    Ok(values
        .into_iter()
        .map(|value| f64::from(value / sum))
        .collect())
}
fn round(value: f64) -> f64 {
    (value * 10000.0).round_ties_even() / 10000.0
}

#[cfg(test)]
mod activation_tests {
    #[cfg(feature = "wgpu")]
    use rstest::rstest;

    use super::*;
    use crate::utils::activation::tests::{assert_close, reference};

    fn matches_python(device: BurnDevice, activation: Option<&str>) {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-clef");
        let config: ClefConfig =
            read_checkpoint_json(&root.join("joint_head_config.json")).unwrap();
        let mut backbone: Qwen3_5Config = read_checkpoint_json(&root.join("config.json")).unwrap();
        let mut reference = reference();
        // Isolate one GPU case without losing the complete CPU reference matrix, e.g. gelu.
        if let Some(name) = activation {
            reference.cases.retain(|case| case.name == name);
            assert_eq!(
                reference.cases.len(),
                1,
                "missing activation reference: {name}"
            );
        }
        let length = reference.input_ids.len();

        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(reference.input_ids, [1, length]), &device);
        for case in reference.cases {
            backbone.text_config.hidden_act.clone_from(&case.name);
            // Keep stage names in captured output to locate interrupted runs, e.g. checkpoint loading.
            eprintln!("[clef-parity] {}: initialize model", case.name);
            let mut model =
                ClefDecisionModel::init_for_loading(&config, &backbone, &device).unwrap();
            // Verify both MLP and Conv1D selection after strict sharded loading, e.g. relu.
            eprintln!("[clef-parity] {}: load checkpoint", case.name);
            load_clef(&mut model, &root).unwrap();
            eprintln!("[clef-parity] {}: forward and readback", case.name);
            let output = model.backbone.model.forward(ids.clone());
            let output = output.slice(s![.., length - 1..length, ..]).into_data();
            assert_close(output, &case.qwen3_5, &case.name, 2e-5);
            eprintln!("[clef-parity] {}: reference matched", case.name);
        }
        for name in ["prelu", "xielu", "unknown"] {
            backbone.text_config.hidden_act = name.into();
            let error = Qwen3_5ForCausalLM::init(&backbone.text_config, &device).unwrap_err();
            assert!(matches!(error, Error::UnsupportedModel(_)), "{error}");
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_matches_python_activation_options() {
        matches_python(BurnDevice::flex(), None);
    }

    #[cfg(feature = "wgpu")]
    #[rstest]
    #[case::gelu("gelu")]
    #[case::gelu_10("gelu_10")]
    #[case::gelu_fast("gelu_fast")]
    #[case::gelu_new("gelu_new")]
    #[case::gelu_python("gelu_python")]
    #[case::gelu_pytorch_tanh("gelu_pytorch_tanh")]
    #[case::gelu_python_tanh("gelu_python_tanh")]
    #[case::gelu_accurate("gelu_accurate")]
    #[case::hardswish("hardswish")]
    #[case::laplace("laplace")]
    #[case::leaky_relu("leaky_relu")]
    #[case::linear("linear")]
    #[case::mish("mish")]
    #[case::quick_gelu("quick_gelu")]
    #[case::relu("relu")]
    #[case::relu2("relu2")]
    #[case::relu6("relu6")]
    #[case::sigmoid("sigmoid")]
    #[case::silu("silu")]
    #[case::sqrtsoftplus("sqrtsoftplus")]
    #[case::swish("swish")]
    #[case::tanh("tanh")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_python_activation_options(#[case] activation: &str) {
        let device = BurnDevice::wgpu(Default::default());
        eprintln!("[clef-parity] {activation}: initialize adapter");
        eprintln!("[clef-parity] adapter: {:?}", device.identity());
        matches_python(device, Some(activation));
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::Value;
    #[cfg(feature = "cpu")]
    use serde_json::json;

    use super::*;
    #[cfg(feature = "cpu")]
    use crate::Truncation;
    use crate::models::clef::EncodedQuestion;

    fn verify_reference(device: BurnDevice, variant: &str, epsilon: f64) {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(variant);
        let model = ClefModel::from_pretrained(&root, &device).unwrap();
        let reference: Value = read_checkpoint_json(&root.join("reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
            let encoded = model.processor.process(&request).unwrap();
            // Compare inference on the same upstream tokens; JSON key order can change ours.
            // For example, description and option_id need not appear in Python's order.
            let span = |value: &Value| {
                let [start, end]: [usize; 2] = serde_json::from_value(value.clone()).unwrap();
                start..end
            };
            let reference_record = EncodedRecord::new(
                serde_json::from_value(case["input_ids"].clone()).unwrap(),
                case["questions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|question| {
                        EncodedQuestion::new(
                            question["question_id"].as_str().unwrap(),
                            question["question_type"].as_u64().unwrap() as usize,
                            span(&question["question_span"]),
                            question["option_spans"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(span)
                                .collect(),
                            serde_json::from_value(question["option_ids"].clone()).unwrap(),
                        )
                    })
                    .collect(),
            );
            for (question, expected) in encoded
                .questions
                .iter()
                .zip(case["questions"].as_array().unwrap())
            {
                assert_eq!(question.question_id, expected["question_id"]);
                assert_eq!(question.question_type, expected["question_type"]);
                assert_eq!(
                    serde_json::to_value(&question.option_ids).unwrap(),
                    expected["option_ids"]
                );
                assert!(!question.question_span.is_empty());
                assert!(question.question_span.end <= encoded.input_ids.len());
                assert_eq!(question.option_spans.len(), question.option_ids.len());
                assert!(
                    question
                        .option_spans
                        .iter()
                        .all(|span| !span.is_empty() && span.end <= encoded.input_ids.len())
                );
            }
            for (actual, expected) in model
                .forward(&reference_record)
                .unwrap()
                .iter()
                .zip(case["logits"].as_array().unwrap())
            {
                for (actual, expected) in actual.iter().zip(expected.as_array().unwrap()) {
                    assert!(
                        (f64::from(*actual) - expected.as_f64().unwrap()).abs() < epsilon,
                        "logit {actual} != {expected}"
                    );
                }
            }
            let response = serde_json::to_value(model.system_one(&request).unwrap()).unwrap();
            for (id, expected) in case["answers"].as_object().unwrap() {
                let actual = &response["answers"][id];
                for (key, expected) in expected.as_object().unwrap() {
                    if key == "legend" {
                        for (level, value) in expected.as_object().unwrap() {
                            if value.is_string() {
                                assert_eq!(actual[key][level], *value);
                            } else {
                                let text = actual[key][level].as_str().unwrap();
                                assert_eq!(serde_json::from_str::<Value>(text).unwrap(), *value);
                            }
                        }
                    } else if encoded.input_ids == reference_record.input_ids {
                        assert_eq!(actual[key], *expected, "{variant}: {id}.{key}");
                    }
                }
            }
            assert_eq!(response["usage"]["input_tokens"], encoded.input_ids.len());
            assert_eq!(response["usage"]["output_tokens"], 0);
        }
    }

    #[cfg(feature = "cpu")]
    #[rstest]
    #[case("tiny-clef")]
    #[case("tiny-clef-flash")]
    fn cpu_matches_transformers_on_reference_tokens(#[case] variant: &str) {
        verify_reference(BurnDevice::flex(), variant, 4e-5);
    }

    #[cfg(feature = "wgpu")]
    #[rstest]
    #[case("tiny-clef")]
    #[case("tiny-clef-flash")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_transformers_on_reference_tokens(#[case] variant: &str) {
        verify_reference(BurnDevice::wgpu(Default::default()), variant, 4e-4);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn refuses_state_loss_and_laya_only_options() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-clef-flash");
        let config = read_checkpoint_json(&root.join("config.json")).unwrap();
        let processor = ClefProcessor::from_pretrained(&root, &config).unwrap();
        let mut request: Request = serde_json::from_value(json!({"state":"alpha ".repeat(600), "questions":{"q":{"type":"noul","instructions":"cancel?"}}})).unwrap();
        assert!(matches!(
            processor.process(&request),
            Err(Error::InvalidRequest(_))
        ));
        request.options.truncation = Truncation::Truncate;
        let encoded = processor.process(&request).unwrap();
        assert!(encoded.usage.truncated);
        assert!(encoded.usage.state_tokens_dropped > 0);
        request.options.max_len = Some(4);
        assert!(matches!(
            processor.process(&request),
            Err(Error::InvalidRequest(_))
        ));
        request.options.max_len = None;
        request.options.head_max_len = Some(16);
        assert!(matches!(
            processor.process(&request),
            Err(Error::InvalidRequest(_))
        ));
    }
}
