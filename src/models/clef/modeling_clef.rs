use super::{ClefConfig, ClefProcessor, EncodedRecord, weights::load_clef};
use crate::{
    Action, Answer, DecisionModel, Error, Metadata, Question, Request, Response, Result,
    models::qwen3_5::{Qwen3_5Config, Qwen3_5TextModel},
    utils::{read_checkpoint_json, render},
};
use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    tensor::{
        Int, Tensor, TensorData,
        activation::{sigmoid, softmax},
        backend::Backend,
    },
};
use burn_std::s;
use camino::Utf8Path;
use indexmap::IndexMap;

/// Own Clef's pretrained backbone, head, and processor for repeated predictions.
pub struct ClefModel<B: Backend> {
    architecture: ClefDecisionModel<B>,
    processor: ClefProcessor,
    metadata: Metadata,
    device: B::Device,
}

/// Match the released Clef model while reusing the Qwen3.5 text backbone.
#[derive(Module, Debug)]
pub struct ClefDecisionModel<B: Backend> {
    language_model: Qwen3_5TextModel<B>,
    output_embeddings: Embedding<B>,
    head: JointSchemaHead<B>,
}

#[derive(Module, Debug)]
struct JointSchemaHead<B: Backend> {
    hidden_norm: LayerNorm<B>,
    memory_projection: Linear<B>,
    question_projection: Linear<B>,
    option_question_projection: Linear<B>,
    global_projection: Linear<B>,
    option_context_projection: Linear<B>,
    option_lexical_projection: Linear<B>,
    type_embedding: Embedding<B>,
    evidence_layers: Vec<EvidenceRoutingLayer<B>>,
    option_summary_norm: LayerNorm<B>,
    layers: Vec<JointDecoderLayer<B>>,
    field_norm: LayerNorm<B>,
    option_norm: LayerNorm<B>,
    residual_scorer: (Linear<B>, Gelu, Dropout, Linear<B>),
    prior_logit_scale: Param<Tensor<B, 1>>,
    joint_logit_scale: Param<Tensor<B, 1>>,
    residual_gate: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
struct EvidenceRoutingLayer<B: Backend> {
    query_norm: LayerNorm<B>,
    memory_norm: LayerNorm<B>,
    attention: MultiheadAttention<B>,
    feedforward_norm: LayerNorm<B>,
    feedforward: (Linear<B>, Gelu, Dropout, Linear<B>, Dropout),
}
#[derive(Module, Debug)]
struct JointDecoderLayer<B: Backend> {
    self_attn: MultiheadAttention<B>,
    multihead_attn: MultiheadAttention<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
}
#[derive(Module, Debug)]
struct MultiheadAttention<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    heads: usize,
}
impl<B: Backend> MultiheadAttention<B> {
    fn forward(&self, query: Tensor<B, 3>, memory: Tensor<B, 3>) -> Tensor<B, 3> {
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
            softmax(query.matmul(key.swap_dims(2, 3)) / (dim as f64).sqrt(), 3)
                .matmul(value)
                .swap_dims(1, 2)
                .reshape([batch, queries, width]),
        )
    }
}
impl<B: Backend> ClefDecisionModel<B> {
    pub fn init(config: &ClefConfig, backbone: &Qwen3_5Config, device: &B::Device) -> Result<Self> {
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
        Ok(Self {
            language_model: Qwen3_5TextModel::init(&backbone.text_config, device)?,
            // Clef's lexical prior uses lm_head weights, not the input embedding table.
            output_embeddings: EmbeddingConfig::new(backbone.text_config.vocab_size, hidden)
                .init(device),
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
    pub fn forward(&self, ids: Tensor<B, 2, Int>, record: &EncodedRecord) -> Vec<Tensor<B, 3>> {
        let hidden = self
            .head
            .hidden_norm
            .forward(self.language_model.forward(ids.clone()));
        let lexical = self.output_embeddings.forward(ids);
        self.head.forward(hidden, lexical, record)
    }
}

fn unit<B: Backend>(input: Tensor<B, 3>, eps: f64) -> Tensor<B, 3> {
    input.clone() / input.square().sum_dim(2).sqrt().clamp_min(eps)
}
fn mean_span<B: Backend>(values: &Tensor<B, 3>, span: &std::ops::Range<usize>) -> Tensor<B, 3> {
    values
        .clone()
        .slice(s![.., span.start..span.end, ..])
        .mean_dim(1)
}

impl<B: Backend> JointSchemaHead<B> {
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        lexical: Tensor<B, 3>,
        record: &EncodedRecord,
    ) -> Vec<Tensor<B, 3>> {
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
        let types = Tensor::<B, 2, Int>::from_data(
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
                + layer.linear2.forward(burn::tensor::activation::gelu(
                    layer.linear1.forward(normalized),
                ));
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

impl<B: Backend> ClefModel<B> {
    /// Load either release's local artifacts, e.g. `ClefModel::<Flex>::from_pretrained`.
    pub fn from_pretrained(root: &Utf8Path, device: &B::Device) -> Result<Self> {
        let artifacts = crate::hub::resolve_clef(&crate::hub::ModelSource::Local(root.into()))?;
        Self::load(root, device, artifacts.metadata)
    }
    pub(crate) fn load(
        root: &Utf8Path,
        device: &B::Device,
        mut metadata: Metadata,
    ) -> Result<Self> {
        let backbone: Qwen3_5Config = read_checkpoint_json(&root.join("config.json"))?;
        let config: ClefConfig = read_checkpoint_json(&root.join("joint_head_config.json"))?;
        config.validate(&backbone)?;
        let processor = ClefProcessor::from_pretrained(root, &backbone)?;
        let mut architecture = ClefDecisionModel::init(&config, &backbone, device)?;
        load_clef(&mut architecture, root)?;
        metadata.architecture = "clef".into();
        if metadata.device.is_empty() {
            metadata.device = B::name(device);
        }
        Ok(Self {
            architecture,
            processor,
            metadata,
            device: device.clone(),
        })
    }
    fn forward(&self, record: &EncodedRecord) -> Result<Vec<Vec<f32>>> {
        let ids = Tensor::<B, 2, Int>::from_data(
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
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| Error::Inference(e.to_string()))
            })
            .collect()
    }
}

impl<B: Backend> DecisionModel for ClefModel<B> {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    fn predict(&self, request: &Request) -> Result<Response> {
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
    use super::*;
    use crate::utils::activation::tests::{assert_close, reference};

    fn matches_python<B: Backend>() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-clef");
        let config: ClefConfig =
            read_checkpoint_json(&root.join("joint_head_config.json")).unwrap();
        let mut backbone: Qwen3_5Config = read_checkpoint_json(&root.join("config.json")).unwrap();
        let reference = reference();
        let length = reference.input_ids.len();
        let device = B::Device::default();
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(reference.input_ids, [1, length]),
            &device,
        );
        for case in reference.cases {
            backbone.text_config.hidden_act.clone_from(&case.name);
            let mut model = ClefDecisionModel::<B>::init(&config, &backbone, &device).unwrap();
            // Verify both MLP and Conv1D selection after strict sharded loading, e.g. relu.
            load_clef(&mut model, &root).unwrap();
            let output = model.language_model.forward(ids.clone());
            let output = output.slice(s![.., length - 1..length, ..]).into_data();
            assert_close(output, &case.qwen3_5, &case.name, 2e-5);
        }
        for name in ["prelu", "xielu", "unknown"] {
            backbone.text_config.hidden_act = name.into();
            let error = Qwen3_5TextModel::<B>::init(&backbone.text_config, &device).unwrap_err();
            assert!(matches!(error, Error::UnsupportedModel(_)), "{error}");
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_matches_python_activation_options() {
        matches_python::<burn::backend::Flex>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_python_activation_options() {
        matches_python::<burn::backend::Wgpu<f32, i32>>();
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rstest::rstest;
    use serde_json::Value;

    #[rstest]
    #[case("tiny-clef")]
    #[case("tiny-clef-flash")]
    fn released_processor_and_transformers_match_end_to_end(#[case] variant: &str) {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(variant);
        let model = ClefModel::<Flex>::from_pretrained(&root, &Default::default()).unwrap();
        let reference: Value = read_checkpoint_json(&root.join("reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
            let encoded = model.processor.process(&request).unwrap();
            assert_eq!(
                serde_json::to_value(&encoded.input_ids).unwrap(),
                case["input_ids"]
            );
            for (question, expected) in encoded
                .questions
                .iter()
                .zip(case["questions"].as_array().unwrap())
            {
                assert_eq!(question.question_id, expected["question_id"]);
                assert_eq!(question.question_type, expected["question_type"]);
                assert_eq!(
                    serde_json::json!([question.question_span.start, question.question_span.end]),
                    expected["question_span"]
                );
                assert_eq!(
                    serde_json::to_value(&question.option_ids).unwrap(),
                    expected["option_ids"]
                );
                assert_eq!(
                    serde_json::json!(
                        question
                            .option_spans
                            .iter()
                            .map(|span| [span.start, span.end])
                            .collect::<Vec<_>>()
                    ),
                    expected["option_spans"]
                );
            }
            for (actual, expected) in model
                .forward(&encoded)
                .unwrap()
                .iter()
                .zip(case["logits"].as_array().unwrap())
            {
                for (actual, expected) in actual.iter().zip(expected.as_array().unwrap()) {
                    assert!(
                        (f64::from(*actual) - expected.as_f64().unwrap()).abs() < 4e-5,
                        "logit {actual} != {expected}"
                    );
                }
            }
            let response = serde_json::to_value(model.predict(&request).unwrap()).unwrap();
            for (id, expected) in case["answers"].as_object().unwrap() {
                let actual = &response["answers"][id];
                for (key, expected) in expected.as_object().unwrap() {
                    if key == "legend" {
                        for (level, value) in expected.as_object().unwrap() {
                            assert_eq!(actual[key][level], render(value).unwrap());
                        }
                    } else {
                        assert_eq!(actual[key], *expected, "{variant}: {id}.{key}");
                    }
                }
            }
            assert_eq!(response["usage"]["input_tokens"], encoded.input_ids.len());
            assert_eq!(response["usage"]["output_tokens"], 0);
        }
    }

    #[test]
    fn refuses_state_loss_and_laya_only_options() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-clef-flash");
        let config = read_checkpoint_json(&root.join("config.json")).unwrap();
        let processor = ClefProcessor::from_pretrained(&root, &config).unwrap();
        let mut request: Request = serde_json::from_value(serde_json::json!({"state":"alpha ".repeat(600), "questions":{"q":{"type":"noul","instructions":"cancel?"}}})).unwrap();
        assert!(matches!(
            processor.process(&request),
            Err(Error::InvalidRequest(_))
        ));
        request.options.truncation = crate::Truncation::Truncate;
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
