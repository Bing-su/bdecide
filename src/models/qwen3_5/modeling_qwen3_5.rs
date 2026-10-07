#![expect(
    non_snake_case,
    reason = "Match Transformers A_log in modules and generated Burn records"
)]

use bon::bon;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::{sigmoid, silu, softplus};
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device as BurnDevice, Int, Tensor, TensorData};
use burn_std::s;
use camino::Utf8Path;

use super::Qwen3_5TextConfig;
use super::weights::{backbone_files, backbone_name, load_causal_lm};
use crate::models::weights;
use crate::utils::activation::HiddenActivation;
use crate::utils::attention::attention;
use crate::{Error, Result};

/// Qwen3.5's text backbone, matching `model.language_model` in Transformers.
#[derive(Module, Debug)]
pub struct Qwen3_5TextModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RMSNorm,
    #[module(skip)]
    pub config: Qwen3_5TextConfig,
}

/// Match Transformers' text-only causal LM, e.g. model.embed_tokens and lm_head.
#[derive(Module, Debug)]
pub struct Qwen3_5ForCausalLM {
    pub(crate) model: Qwen3_5TextModel,
    lm_head: Linear,
    #[module(skip)]
    pub config: Qwen3_5TextConfig,
}

/// Return vocabulary logits as a tensor, e.g. output.logits has [batch, sequence, vocab].
#[derive(Debug, Clone)]
pub struct CausalLMOutput {
    pub logits: Tensor<3>,
}

#[bon]
impl Qwen3_5ForCausalLM {
    /// Initialize a tied or untied readout, e.g. Wald-4B shares embed_tokens.
    #[builder(start_fn = builder)]
    pub fn new(config: &Qwen3_5TextConfig, device: &BurnDevice) -> Result<Self> {
        let mut model = Self::init(config, device)?;
        model.tie_weights();
        Ok(model)
    }

    pub(crate) fn init(config: &Qwen3_5TextConfig, device: &BurnDevice) -> Result<Self> {
        Ok(Self {
            model: Qwen3_5TextModel::new(config, device)?,
            lm_head: LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_bias(false)
                .init(device),
            config: config.clone(),
        })
    }

    /// Load strict text weights, e.g. a renamed Vev or Wald snapshot directory.
    pub fn from_pretrained(root: &Utf8Path, device: &BurnDevice) -> Result<Self> {
        let config = Qwen3_5TextConfig::from_pretrained(root)?;
        // Load before tying so large checkpoints do not allocate random embedding tables first.
        // Both stored aliases with unequal values stay independent, as in Transformers.from_pretrained.
        let mut model = Self::init(&config, device)?;
        load_causal_lm(&mut model, root)?;
        Ok(model)
    }

    /// Access the input table, e.g. get_input_embeddings().weight.
    pub fn get_input_embeddings(&self) -> &Embedding {
        self.model.get_input_embeddings()
    }

    /// Replace the input module, e.g. call tie_weights afterward to reattach lm_head.
    pub fn set_input_embeddings(&mut self, embeddings: Embedding) {
        self.model.set_input_embeddings(embeddings);
    }

    /// Access the vocabulary projection, e.g. get_output_embeddings().weight.
    pub fn get_output_embeddings(&self) -> &Linear {
        &self.lm_head
    }

    /// Replace the output projection, e.g. an independent pretrained lm_head.
    pub fn set_output_embeddings(&mut self, embeddings: Linear) {
        self.lm_head = embeddings;
    }

    /// Read output token vectors for Clef's lexical prior, e.g. `[batch, length, hidden]`.
    pub(crate) fn embed_output(&self, ids: Tensor<2, Int>) -> Tensor<3> {
        if self.lm_head.weight.id == self.model.embed_tokens.weight.id {
            return self.model.embed_tokens.forward(ids);
        }
        let [batch, length] = ids.dims();
        let weights = self.lm_head.weight.val();
        let hidden = weights.dims()[0];
        // Select tokens before transposing, e.g. never copy a whole 248K-token table.
        weights
            .select(1, ids.reshape([batch * length]))
            .transpose()
            .reshape([batch, length, hidden])
    }

    /// Share the same parameter ID, e.g. after replacing inputs and calling tie_weights().
    pub fn tie_weights(&mut self) {
        if self.config.tie_word_embeddings {
            // Burn stores Linear weights transposed relative to PyTorch; preserve the shared ID.
            self.lm_head.weight = self
                .model
                .embed_tokens
                .weight
                .clone()
                .map(Tensor::transpose);
        }
    }

    /// Return vocabulary logits; 0 keeps all positions, e.g. forward_builder().input_ids(ids).call().
    #[builder(start_fn = forward_builder, finish_fn = call)]
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        #[builder(default)] logits_to_keep: usize,
    ) -> Result<CausalLMOutput> {
        let [batch, length] = input_ids.dims();
        if batch == 0 || length == 0 {
            return Err(Error::InvalidRequest("empty causal LM input".into()));
        }
        let start = if logits_to_keep == 0 {
            0
        } else {
            length.saturating_sub(logits_to_keep)
        };
        let hidden = self
            .model
            .forward(input_ids)
            .slice(s![.., start..length, ..]);
        Ok(CausalLMOutput {
            logits: self.lm_head.forward(hidden),
        })
    }

    /// Evaluate one unpadded prompt, e.g. read only the A/B answer rows at its final token.
    pub(crate) fn forward_selected(
        &self,
        input_ids: &[u32],
        answer_ids: &[u32],
        device: &BurnDevice,
    ) -> Result<Vec<f32>> {
        let weights = self.lm_head.weight.val();
        let [hidden_size, vocab] = weights.dims();
        if input_ids.is_empty()
            || answer_ids.is_empty()
            || input_ids
                .iter()
                .chain(answer_ids)
                .any(|&id| id as usize >= vocab)
        {
            return Err(Error::InvalidRequest(
                "empty prompt/readout or token outside vocabulary".into(),
            ));
        }
        let ids = Tensor::from_data(
            TensorData::new(input_ids.to_vec(), [1, input_ids.len()]),
            device,
        );
        let hidden = self
            .model
            .forward(ids)
            .slice(s![0..1, input_ids.len() - 1..input_ids.len(), ..])
            .reshape([1, hidden_size]);
        let answers = Tensor::<1, Int>::from_data(
            TensorData::new(answer_ids.to_vec(), [answer_ids.len()]),
            device,
        );
        // Select tied rows before transposing so Flex never copies the full vocabulary table,
        // e.g. a Yes/No readout needs only two rows. Unequal stored aliases remain independent.
        let selected = if self.lm_head.weight.id == self.model.embed_tokens.weight.id {
            self.model
                .embed_tokens
                .weight
                .val()
                .select(0, answers)
                .transpose()
        } else {
            weights.select(1, answers)
        };
        hidden
            .matmul(selected)
            .try_into_vec_as::<f32>()
            .map_err(|error| Error::Inference(error.to_string()))
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tying_tests {
    use approx::abs_diff_eq;
    use burn::tensor::module::embedding;
    use camino::Utf8Path;

    use super::*;

    #[test]
    fn selected_readout_matches_forward_for_tied_and_replaced_output_weights() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-vev-4b");
        let device = BurnDevice::flex();
        let mut model = Qwen3_5ForCausalLM::from_pretrained(&root, &device).unwrap();
        let input_ids = [2u32, 3, 4];
        let answer_ids = [5u32, 0];
        for independent in [false, true] {
            if independent {
                // Keep the tied config but replace the output parameter, e.g. unequal stored aliases.
                model.lm_head.weight =
                    Param::from_tensor(Tensor::zeros(model.lm_head.weight.val().dims(), &device));
            }
            // Clef reads the same output table across batches, e.g. repeated token IDs.
            let ids =
                Tensor::from_data(TensorData::new(vec![2u32, 3, 2, 4, 2, 3], [2, 3]), &device);
            assert_eq!(
                model.embed_output(ids.clone()).into_data(),
                embedding(model.lm_head.weight.val().transpose(), ids).into_data()
            );
            let actual = model
                .forward_selected(&input_ids, &answer_ids, &device)
                .unwrap();
            let input = Tensor::from_data(TensorData::new(input_ids.to_vec(), [1, 3]), &device);
            let answers =
                Tensor::<1, Int>::from_data(TensorData::new(answer_ids.to_vec(), [2]), &device);
            let expected = model
                .forward(input, 1)
                .unwrap()
                .logits
                .select(2, answers)
                .try_into_vec_as::<f32>()
                .unwrap();
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!(abs_diff_eq!(actual, expected, epsilon = 1e-6));
            }
        }
    }

    #[test]
    fn tied_readout_shares_parameter_identity_and_retie_uses_replaced_inputs() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-vev-4b");
        let device = BurnDevice::flex();
        let mut model = Qwen3_5ForCausalLM::from_pretrained(&root, &device).unwrap();
        assert_eq!(
            model.get_input_embeddings().weight.id,
            model.get_output_embeddings().weight.id
        );
        // Replacing a parameter requires explicit retying, as Python's tie_weights does.
        let shape = model.model.embed_tokens.weight.val().dims();
        let mut embeddings = model.get_input_embeddings().clone();
        embeddings.weight = Param::from_tensor(Tensor::ones(shape, &device));
        model.set_input_embeddings(embeddings);
        model.tie_weights();
        assert_eq!(
            model.get_input_embeddings().weight.id,
            model.get_output_embeddings().weight.id
        );
        assert_eq!(
            model.lm_head.weight.val().into_data(),
            Tensor::<2>::ones([shape[1], shape[0]], &device).into_data()
        );
    }
}

#[derive(Module, Debug)]
struct Qwen3_5DecoderLayer {
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
    self_attn: Option<Qwen3_5Attention>,
    linear_attn: Option<Qwen3_5GatedDeltaNet>,
    mlp: Qwen3_5MLP,
}

#[derive(Module, Debug)]
struct Qwen3_5RMSNorm {
    weight: Param<Tensor<1>>,
    eps: f64,
    offset: bool,
}
impl Qwen3_5RMSNorm {
    fn new(dim: usize, eps: f64, offset: bool, device: &BurnDevice) -> Self {
        Self {
            weight: if offset {
                Initializer::Zeros
            } else {
                Initializer::Ones
            }
            .init([dim], device),
            eps,
            offset,
        }
    }

    fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        // Q/K and decoder norms store zero-centered weights; delta-net's gated norm does not.
        let weight = self.weight.val() + if self.offset { 1.0 } else { 0.0 };
        input.clone() / (input.square().mean_dim(D - 1) + self.eps).sqrt() * weight.unsqueeze()
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod norm_tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_normalizes_five_small_rows() {
        let device = BurnDevice::wgpu(Default::default());
        eprintln!("[qwen-norm] adapter: {:?}", device.identity());
        // Isolate Clef's first reduction without loading a model, e.g. five rows of width 48.
        let norm = Qwen3_5RMSNorm::new(48, 1e-6, true, &device);
        let output = norm
            .forward(Tensor::<3>::ones([1, 5, 48], &device))
            .try_into_vec_as::<f32>()
            .unwrap();
        assert_eq!(output.len(), 240);
        let expected = 1.0_f32 / (1.0_f32 + 1e-6).sqrt();
        for value in output {
            assert_relative_eq!(value, expected, epsilon = 2e-6);
        }
    }
}

#[derive(Module, Debug)]
struct Qwen3_5MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    // Keep the configured activation outside weight records, e.g. hidden_act="relu".
    #[module(skip)]
    act_fn: HiddenActivation,
}

#[derive(Module, Debug)]
struct Qwen3_5Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3_5RMSNorm,
    k_norm: Qwen3_5RMSNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
}

// Preserve external parameter spelling, e.g. linear_attn.A_log, in generated records.
#[derive(Module, Debug)]
struct Qwen3_5GatedDeltaNet {
    conv1d: Conv1d,
    dt_bias: Param<Tensor<1>>,
    A_log: Param<Tensor<1>>,
    norm: Qwen3_5RMSNorm,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    // Only Conv1D follows hidden_act; the final norm gate remains SiLU.
    #[module(skip)]
    activation: HiddenActivation,
}

#[bon]
impl Qwen3_5TextModel {
    /// Reuse the input parameter for tied readout, e.g. lm_head and embed_tokens share storage.
    pub fn get_input_embeddings(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Replace the token table, e.g. the embedding supplied by a causal LM wrapper.
    pub fn set_input_embeddings(&mut self, embeddings: Embedding) {
        self.embed_tokens = embeddings;
    }

    /// Initialize validated text dimensions, e.g. `Self::new(config, device)?`.
    #[builder(start_fn = builder)]
    pub fn new(config: &Qwen3_5TextConfig, device: &BurnDevice) -> Result<Self> {
        Self::init(config, device)
    }

    /// Load the text backbone, e.g. extract model.language_model from Vev's checkpoint.
    pub fn from_pretrained(root: &Utf8Path, device: &BurnDevice) -> Result<Self> {
        let config = Qwen3_5TextConfig::from_pretrained(root)?;
        let files = backbone_files(root)?;
        let mut model = Self::new(&config, device)?;
        weights::load(&mut model, root, &files, &[], |source| {
            backbone_name(source)
                .map(|name| name.and_then(|name| name.strip_prefix("model.").map(str::to_owned)))
        })?;
        Ok(model)
    }

    /// Initialize the Transformers module hierarchy, e.g. a tiny config for parity tests.
    pub fn init(config: &Qwen3_5TextConfig, device: &BurnDevice) -> Result<Self> {
        config.validate()?;
        let activation = config.hidden_act.parse::<HiddenActivation>()?;
        let linear = |i, o, bias| LinearConfig::new(i, o).with_bias(bias).init(device);
        let norm = |dim, offset| Qwen3_5RMSNorm::new(dim, config.rms_norm_eps, offset, device);
        let (rotary_dim, theta) = config.rotary()?;
        let hidden = config.hidden_size;
        let mut layers = Vec::new();
        for layer_type in config.layer_types() {
            let full = layer_type == "full_attention";
            let self_attn = full.then(|| Qwen3_5Attention {
                q_proj: linear(
                    hidden,
                    config.num_attention_heads * config.head_dim * 2,
                    config.attention_bias,
                ),
                k_proj: linear(
                    hidden,
                    config.num_key_value_heads * config.head_dim,
                    config.attention_bias,
                ),
                v_proj: linear(
                    hidden,
                    config.num_key_value_heads * config.head_dim,
                    config.attention_bias,
                ),
                o_proj: linear(
                    config.num_attention_heads * config.head_dim,
                    hidden,
                    config.attention_bias,
                ),
                q_norm: norm(config.head_dim, true),
                k_norm: norm(config.head_dim, true),
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                rotary_dim,
                theta,
            });
            let linear_attn = (!full).then(|| {
                let key = config.linear_key_head_dim * config.linear_num_key_heads;
                let value = config.linear_value_head_dim * config.linear_num_value_heads;
                let total = 2 * key + value;
                Qwen3_5GatedDeltaNet {
                    conv1d: Conv1dConfig::new(total, total, config.linear_conv_kernel_dim)
                        .with_groups(total)
                        .with_bias(false)
                        .with_padding(PaddingConfig1d::Explicit(
                            config.linear_conv_kernel_dim - 1,
                            0,
                        ))
                        .init(device),
                    dt_bias: Initializer::Ones.init([config.linear_num_value_heads], device),
                    A_log: Initializer::Zeros.init([config.linear_num_value_heads], device),
                    norm: norm(config.linear_value_head_dim, false),
                    in_proj_qkv: linear(hidden, total, false),
                    in_proj_z: linear(hidden, value, false),
                    in_proj_b: linear(hidden, config.linear_num_value_heads, false),
                    in_proj_a: linear(hidden, config.linear_num_value_heads, false),
                    out_proj: linear(value, hidden, false),
                    key_heads: config.linear_num_key_heads,
                    value_heads: config.linear_num_value_heads,
                    key_dim: config.linear_key_head_dim,
                    value_dim: config.linear_value_head_dim,
                    activation,
                }
            });
            layers.push(Qwen3_5DecoderLayer {
                input_layernorm: norm(hidden, true),
                post_attention_layernorm: norm(hidden, true),
                self_attn,
                linear_attn,
                mlp: Qwen3_5MLP {
                    gate_proj: linear(hidden, config.intermediate_size, false),
                    up_proj: linear(hidden, config.intermediate_size, false),
                    down_proj: linear(config.intermediate_size, hidden, false),
                    act_fn: activation,
                },
            });
        }
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, hidden).init(device),
            layers,
            norm: norm(hidden, true),
            config: config.clone(),
        })
    }

    /// Execute one unpadded record with `use_cache=False`, as Clef's reference does.
    pub fn forward(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        let mut hidden = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            let normalized = layer.input_layernorm.forward(hidden.clone());
            let attention = match (&layer.self_attn, &layer.linear_attn) {
                (Some(attention), _) => attention.forward(normalized),
                (_, Some(attention)) => attention.forward(normalized),
                _ => normalized,
            };
            hidden = hidden + attention;
            let normalized = layer.post_attention_layernorm.forward(hidden.clone());
            hidden = hidden
                + layer.mlp.down_proj.forward(
                    layer
                        .mlp
                        .act_fn
                        .forward(layer.mlp.gate_proj.forward(normalized.clone()))
                        * layer.mlp.up_proj.forward(normalized),
                );
        }
        self.norm.forward(hidden)
    }
}

impl Qwen3_5Attention {
    fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [batch, length, _] = hidden.dims();
        let projected = self.q_proj.forward(hidden.clone()).reshape([
            batch,
            length,
            self.heads,
            2 * self.head_dim,
        ]);
        let gate = projected
            .clone()
            .slice(s![.., .., .., self.head_dim..])
            .reshape([batch, length, self.heads * self.head_dim]);
        let query = self
            .q_norm
            .forward(projected.slice(s![.., .., .., ..self.head_dim]))
            .swap_dims(1, 2);
        let key = self
            .k_norm
            .forward(self.k_proj.forward(hidden.clone()).reshape([
                batch,
                length,
                self.kv_heads,
                self.head_dim,
            ]))
            .swap_dims(1, 2);
        let value =
            self.v_proj
                .forward(hidden)
                .reshape([batch, length, self.kv_heads, self.head_dim]);
        let query = self.rotate(query);
        let key = repeat_heads(self.rotate(key).swap_dims(1, 2), self.heads / self.kv_heads)
            .swap_dims(1, 2);
        let value = repeat_heads(value, self.heads / self.kv_heads).swap_dims(1, 2);
        // Use causal mode so Burn can avoid a dense triangle, e.g. during full-record inference.
        let output = attention(
            query,
            key,
            value,
            None,
            AttentionModuleOptions {
                is_causal: true,
                ..Default::default()
            },
        )
        .swap_dims(1, 2)
        .reshape([batch, length, self.heads * self.head_dim]);
        self.o_proj.forward(output * sigmoid(gate))
    }

    fn rotate(&self, input: Tensor<4>) -> Tensor<4> {
        let [_, _, length, _] = input.dims();
        let mut cosine = Vec::new();
        let mut sine = Vec::new();
        for pos in 0..length {
            for channel in 0..self.rotary_dim {
                let angle = pos as f64
                    / self.theta.powf(
                        (2 * (channel % (self.rotary_dim / 2))) as f64 / self.rotary_dim as f64,
                    );
                cosine.push(angle.cos() as f32);
                sine.push(angle.sin() as f32);
            }
        }
        let cosine = Tensor::<4>::from_data(
            TensorData::new(cosine, [1, 1, length, self.rotary_dim]),
            &input.device(),
        );
        let sine = Tensor::<4>::from_data(
            TensorData::new(sine, [1, 1, length, self.rotary_dim]),
            &input.device(),
        );
        let first = input.clone().slice(s![.., .., .., ..self.rotary_dim / 2]);
        let second = input
            .clone()
            .slice(s![.., .., .., self.rotary_dim / 2..self.rotary_dim]);
        let rotated = input.clone().slice(s![.., .., .., ..self.rotary_dim]) * cosine
            + Tensor::cat(vec![-second, first], 3) * sine;
        // Text has identical positions on all mRoPE axes; retain the non-rotary tail.
        if self.rotary_dim == self.head_dim {
            rotated
        } else {
            Tensor::cat(
                vec![rotated, input.slice(s![.., .., .., self.rotary_dim..])],
                3,
            )
        }
    }
}

fn repeat_heads(input: Tensor<4>, repeats: usize) -> Tensor<4> {
    let [batch, length, heads, dim] = input.dims();
    input
        .unsqueeze_dim::<5>(3)
        .expand([batch, length, heads, repeats, dim])
        .reshape([batch, length, heads * repeats, dim])
}

impl Qwen3_5GatedDeltaNet {
    fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [batch, length, _] = hidden.dims();
        let key_width = self.key_heads * self.key_dim;
        let value_width = self.value_heads * self.value_dim;
        let mixed = self
            .activation
            .forward(
                self.conv1d
                    .forward(self.in_proj_qkv.forward(hidden.clone()).swap_dims(1, 2))
                    .slice(s![.., .., ..length]),
            )
            .swap_dims(1, 2);
        let query = mixed.clone().slice(s![.., .., ..key_width]).reshape([
            batch,
            length,
            self.key_heads,
            self.key_dim,
        ]);
        let key = mixed
            .clone()
            .slice(s![.., .., key_width..2 * key_width])
            .reshape([batch, length, self.key_heads, self.key_dim]);
        let query = repeat_heads(l2norm(query), self.value_heads / self.key_heads)
            / (self.key_dim as f64).sqrt();
        let key = repeat_heads(l2norm(key), self.value_heads / self.key_heads);
        let value = mixed.slice(s![.., .., 2 * key_width..]).reshape([
            batch,
            length,
            self.value_heads,
            self.value_dim,
        ]);
        let beta = sigmoid(self.in_proj_b.forward(hidden.clone()));
        let decay = (-self.A_log.val().exp().unsqueeze::<3>()
            * softplus(
                self.in_proj_a.forward(hidden.clone()) + self.dt_bias.val().unsqueeze::<3>(),
                1.0,
            ))
        .exp();
        let mut state = Tensor::<4>::zeros(
            [batch, self.value_heads, self.key_dim, self.value_dim],
            &hidden.device(),
        );
        let mut outputs = Vec::with_capacity(length);
        // Use the exact recurrent delta rule, avoiding an approximation of linear attention.
        // No cache is retained between independent requests, e.g. ticket A and ticket B.
        for position in 0..length {
            let query = query
                .clone()
                .slice(s![.., position..position + 1, .., ..])
                .reshape([batch, self.value_heads, self.key_dim, 1]);
            let key = key
                .clone()
                .slice(s![.., position..position + 1, .., ..])
                .reshape([batch, self.value_heads, self.key_dim, 1]);
            let value = value
                .clone()
                .slice(s![.., position..position + 1, .., ..])
                .reshape([batch, self.value_heads, 1, self.value_dim]);
            let decay = decay
                .clone()
                .slice(s![.., position..position + 1, ..])
                .reshape([batch, self.value_heads, 1, 1]);
            let beta = beta
                .clone()
                .slice(s![.., position..position + 1, ..])
                .reshape([batch, self.value_heads, 1, 1]);
            state = state * decay;
            let delta = (value - (state.clone() * key.clone()).sum_dim(2)) * beta;
            state = state + key * delta;
            outputs.push((state.clone() * query).sum_dim(2).reshape([
                batch,
                1,
                self.value_heads,
                self.value_dim,
            ]));
        }
        let output = Tensor::cat(outputs, 1);
        let gate = self.in_proj_z.forward(hidden).reshape([
            batch,
            length,
            self.value_heads,
            self.value_dim,
        ]);
        self.out_proj
            .forward((self.norm.forward(output) * silu(gate)).reshape([batch, length, value_width]))
    }
}

fn l2norm(input: Tensor<4>) -> Tensor<4> {
    input.clone() / (input.square().sum_dim(3) + 1e-6).sqrt()
}
