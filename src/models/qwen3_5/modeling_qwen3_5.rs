use super::Qwen3_5TextConfig;
use crate::{
    Result,
    utils::{activation::HiddenActivation, attention::attention},
};
use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Int, Tensor, TensorData,
        activation::{sigmoid, silu, softplus},
        backend::Backend,
        ops::AttentionModuleOptions,
    },
};
use burn_std::s;

/// Qwen3.5's text backbone, matching `model.language_model` in Transformers.
#[derive(Module, Debug)]
pub struct Qwen3_5TextModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<Qwen3_5DecoderLayer<B>>,
    norm: Qwen3_5RMSNorm<B>,
}

#[derive(Module, Debug)]
struct Qwen3_5DecoderLayer<B: Backend> {
    input_layernorm: Qwen3_5RMSNorm<B>,
    post_attention_layernorm: Qwen3_5RMSNorm<B>,
    self_attn: Option<Qwen3_5Attention<B>>,
    linear_attn: Option<Qwen3_5GatedDeltaNet<B>>,
    mlp: Qwen3_5MLP<B>,
}

#[derive(Module, Debug)]
struct Qwen3_5RMSNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
    offset: bool,
}
impl<B: Backend> Qwen3_5RMSNorm<B> {
    fn new(dim: usize, eps: f64, offset: bool, device: &B::Device) -> Self {
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
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Q/K and decoder norms store zero-centered weights; delta-net's gated norm does not.
        let weight = self.weight.val() + if self.offset { 1.0 } else { 0.0 };
        input.clone() / (input.square().mean_dim(D - 1) + self.eps).sqrt() * weight.unsqueeze()
    }
}

#[derive(Module, Debug)]
struct Qwen3_5MLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
    // Keep the configured activation outside weight records, e.g. hidden_act="relu".
    #[module(skip)]
    act_fn: HiddenActivation,
}

#[derive(Module, Debug)]
struct Qwen3_5Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: Qwen3_5RMSNorm<B>,
    k_norm: Qwen3_5RMSNorm<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
}

#[derive(Module, Debug)]
struct Qwen3_5GatedDeltaNet<B: Backend> {
    conv1d: Conv1d<B>,
    dt_bias: Param<Tensor<B, 1>>,
    a_log: Param<Tensor<B, 1>>,
    norm: Qwen3_5RMSNorm<B>,
    in_proj_qkv: Linear<B>,
    in_proj_z: Linear<B>,
    in_proj_b: Linear<B>,
    in_proj_a: Linear<B>,
    out_proj: Linear<B>,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    // Only Conv1D follows hidden_act; the final norm gate remains SiLU.
    #[module(skip)]
    activation: HiddenActivation,
}

impl<B: Backend> Qwen3_5TextModel<B> {
    /// Initialize the Transformers module hierarchy, e.g. a tiny config for parity tests.
    pub fn init(config: &Qwen3_5TextConfig, device: &B::Device) -> Result<Self> {
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
                    a_log: Initializer::Zeros.init([config.linear_num_value_heads], device),
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
        })
    }

    /// Execute one unpadded record with `use_cache=False`, as Clef's reference does.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
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

impl<B: Backend> Qwen3_5Attention<B> {
    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
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
    fn rotate(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
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
        let cosine = Tensor::<B, 4>::from_data(
            TensorData::new(cosine, [1, 1, length, self.rotary_dim]),
            &input.device(),
        );
        let sine = Tensor::<B, 4>::from_data(
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

fn repeat_heads<B: Backend>(input: Tensor<B, 4>, repeats: usize) -> Tensor<B, 4> {
    let [batch, length, heads, dim] = input.dims();
    input
        .unsqueeze_dim::<5>(3)
        .expand([batch, length, heads, repeats, dim])
        .reshape([batch, length, heads * repeats, dim])
}

impl<B: Backend> Qwen3_5GatedDeltaNet<B> {
    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
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
        let decay = (-self.a_log.val().exp().unsqueeze::<3>()
            * softplus(
                self.in_proj_a.forward(hidden.clone()) + self.dt_bias.val().unsqueeze::<3>(),
                1.0,
            ))
        .exp();
        let mut state = Tensor::<B, 4>::zeros(
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

fn l2norm<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    input.clone() / (input.square().sum_dim(3) + 1e-6).sqrt()
}
