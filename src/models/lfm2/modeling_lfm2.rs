//! Share LFM2 blocks while preserving causal and media-prefix encoder semantics.
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Embedding,
    EmbeddingConfig,
    Linear,
    LinearConfig,
    PaddingConfig1d,
    RmsNorm,
    RmsNormConfig,
};
use burn::tensor::activation::silu;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use burn_std::s;

use super::Lfm2Config;
use crate::Result;
use crate::utils::attention::attention;

#[derive(Module, Debug)]
pub struct Lfm2Model {
    pub(crate) embed_tokens: Embedding,
    layers: Vec<Layer>,
    embedding_norm: RmsNorm,
    theta: f64,
    head_dim: usize,
    bidirectional: bool,
    #[module(skip)]
    pub config: Lfm2Config,
}

#[derive(Module, Debug)]
struct Layer {
    self_attn: Option<Attention>,
    conv: Option<ShortConv>,
    feed_forward: Mlp,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

#[derive(Module, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    heads: usize,
    kv_heads: usize,
}

#[derive(Module, Debug)]
struct ShortConv {
    conv: Conv1d,
    in_proj: Linear,
    out_proj: Linear,
}

#[derive(Module, Debug)]
struct Mlp {
    w1: Linear,
    w3: Linear,
    w2: Linear,
}

impl Lfm2Model {
    /// Initialize the causal backbone, e.g. LFM2-VL's language_model.
    pub fn new(c: &Lfm2Config, device: &Device) -> Result<Self> {
        c.validate()?;
        Ok(Self::init(c, false, device))
    }

    pub(crate) fn new_encoder(c: &Lfm2Config, device: &Device) -> Result<Self> {
        // D1Omni uses centered convolutions and always adjusts the feed-forward width.
        let mut c = c.clone();
        c.block_auto_adjust_ff_dim = true;
        c.validate()?;
        Ok(Self::init(&c, true, device))
    }

    fn init(c: &Lfm2Config, bidirectional: bool, device: &Device) -> Self {
        let d = c.hidden_size;
        let dim = d / c.num_attention_heads;
        let linear = |i, o, bias| LinearConfig::new(i, o).with_bias(bias).init(device);
        let norm = |width| {
            RmsNormConfig::new(width)
                .with_epsilon(c.norm_eps)
                .init(device)
        };
        Self {
            embed_tokens: EmbeddingConfig::new(c.vocab_size, d).init(device),
            layers: c
                .layer_types
                .iter()
                .map(|kind| Layer {
                    self_attn: (kind == "full_attention").then(|| Attention {
                        q_proj: linear(d, d, false),
                        k_proj: linear(d, c.num_key_value_heads * dim, false),
                        v_proj: linear(d, c.num_key_value_heads * dim, false),
                        out_proj: linear(d, d, false),
                        q_layernorm: norm(dim),
                        k_layernorm: norm(dim),
                        heads: c.num_attention_heads,
                        kv_heads: c.num_key_value_heads,
                    }),
                    conv: (kind == "conv").then(|| ShortConv {
                        conv: Conv1dConfig::new(d, d, c.conv_l_cache)
                            .with_groups(d)
                            .with_bias(c.conv_bias)
                            .with_padding(if bidirectional {
                                PaddingConfig1d::Explicit(1, 1)
                            } else {
                                PaddingConfig1d::Explicit(2, 0)
                            })
                            .init(device),
                        in_proj: linear(d, 3 * d, c.conv_bias),
                        out_proj: linear(d, d, c.conv_bias),
                    }),
                    feed_forward: Mlp {
                        w1: linear(d, c.ffn_size(), false),
                        w3: linear(d, c.ffn_size(), false),
                        w2: linear(c.ffn_size(), d, false),
                    },
                    operator_norm: norm(d),
                    ffn_norm: norm(d),
                })
                .collect(),
            embedding_norm: norm(d),
            theta: c.theta(),
            head_dim: dim,
            bidirectional,
            config: c.clone(),
        }
    }

    /// Embed token IDs, e.g. concatenate image embeddings before an encoder call.
    pub fn embed(&self, ids: &[u32], device: &Device) -> Tensor<3> {
        self.embed_tokens.forward(Tensor::<2, Int>::from_data(
            TensorData::new(
                ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
                [1, ids.len()],
            ),
            device,
        ))
    }

    /// Return causal hidden states from embeddings, e.g. [batch, length, hidden].
    pub fn forward(&self, h: Tensor<3>) -> Tensor<3> {
        self.forward_prefix(h, 0)
    }

    pub(crate) fn forward_prefix(&self, mut h: Tensor<3>, prefix: usize) -> Tensor<3> {
        let length = h.dims()[1];
        let mask = if self.bidirectional && prefix > 0 {
            let values: Vec<_> = (0..length)
                .flat_map(|q| (0..length).map(move |k| q < prefix && k >= prefix))
                .collect();
            Some(Tensor::<4, Bool>::from_data(
                TensorData::new(values, [1, 1, length, length]),
                &h.device(),
            ))
        } else {
            None
        };
        let (cos, sin) = rotary(length, self.head_dim, self.theta, &h.device());
        for layer in &self.layers {
            let x = layer.operator_norm.forward(h.clone());
            let x = if let Some(attn) = &layer.self_attn {
                attn.forward(
                    x,
                    cos.clone(),
                    sin.clone(),
                    mask.clone(),
                    !self.bidirectional,
                )
            } else if let Some(conv) = &layer.conv {
                conv.forward(x, self.bidirectional, prefix)
            } else {
                x
            };
            h = h + x;
            let x = layer.ffn_norm.forward(h.clone());
            h = h + layer.feed_forward.w2.forward(
                silu(layer.feed_forward.w1.forward(x.clone())) * layer.feed_forward.w3.forward(x),
            );
        }
        self.embedding_norm.forward(h)
    }
}

fn rotary(length: usize, dim: usize, theta: f64, device: &Device) -> (Tensor<4>, Tensor<4>) {
    let mut cos = Vec::with_capacity(length * dim);
    let mut sin = Vec::with_capacity(length * dim);
    for p in 0..length {
        for c in 0..dim {
            let phase = p as f32 / (theta as f32).powf((2 * (c % (dim / 2))) as f32 / dim as f32);
            cos.push(phase.cos());
            sin.push(phase.sin());
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [1, 1, length, dim]), device),
        Tensor::from_data(TensorData::new(sin, [1, 1, length, dim]), device),
    )
}

fn rotate(x: Tensor<4>, cos: Tensor<4>, sin: Tensor<4>) -> Tensor<4> {
    let half = x.dims()[3] / 2;
    x.clone() * cos
        + Tensor::cat(
            vec![
                -x.clone().slice(s![.., .., .., half..]),
                x.slice(s![.., .., .., ..half]),
            ],
            3,
        ) * sin
}

impl Attention {
    fn forward(
        &self,
        x: Tensor<3>,
        cos: Tensor<4>,
        sin: Tensor<4>,
        mask: Option<Tensor<4, Bool>>,
        causal: bool,
    ) -> Tensor<3> {
        let [b, l, d] = x.dims();
        let dim = d / self.heads;
        let q = self
            .q_layernorm
            .forward(
                self.q_proj
                    .forward(x.clone())
                    .reshape([b, l, self.heads, dim]),
            )
            .swap_dims(1, 2);
        let k = self
            .k_layernorm
            .forward(
                self.k_proj
                    .forward(x.clone())
                    .reshape([b, l, self.kv_heads, dim]),
            )
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, l, self.kv_heads, dim])
            .swap_dims(1, 2);
        let repeat = |x: Tensor<4>| {
            x.unsqueeze_dim::<5>(2)
                .expand([b, self.kv_heads, self.heads / self.kv_heads, l, dim])
                .reshape([b, self.heads, l, dim])
        };
        self.out_proj.forward(
            attention(
                rotate(q, cos.clone(), sin.clone()),
                repeat(rotate(k, cos, sin)),
                repeat(v),
                mask.map(|mask| mask.expand([b, self.heads, l, l])),
                burn::tensor::ops::AttentionModuleOptions {
                    is_causal: causal,
                    ..Default::default()
                },
            )
            .swap_dims(1, 2)
            .reshape([b, l, d]),
        )
    }
}

impl ShortConv {
    fn forward(&self, x: Tensor<3>, bidirectional: bool, prefix: usize) -> Tensor<3> {
        let [_, l, d] = x.dims();
        let projected = self.in_proj.forward(x);
        let gate = projected.clone().slice(s![.., .., d..2 * d]);
        let bx = (projected.clone().slice(s![.., .., ..d]) * projected.slice(s![.., .., 2 * d..]))
            .swap_dims(1, 2);
        let y = if bidirectional && prefix > 0 && prefix < l {
            // Zero-pad the media boundary; text retains its left media context, e.g. prefix=1.
            let media = self.conv.forward(bx.clone().slice(s![.., .., ..prefix]));
            let text = self
                .conv
                .forward(bx.slice(s![.., .., prefix - 1..]))
                .slice(s![.., .., 1..]);
            Tensor::cat(vec![media, text], 2)
        } else {
            self.conv.forward(bx)
        };
        self.out_proj.forward(gate * y.swap_dims(1, 2))
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
