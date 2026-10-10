//! Match the trained NeMo frontend and FastConformer, e.g. one prefix per 80 ms.
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{
    Initializer,
    LayerNorm,
    LayerNormConfig,
    Linear,
    LinearConfig,
    PaddingConfig1d,
    PaddingConfig2d,
    Relu,
};
use burn::signal::{StftOptions, hann_window, stft};
use burn::tensor::activation::{gelu, glu, relu, silu, softmax};
use burn::tensor::module::batch_norm;
use burn::tensor::ops::PadMode;
use burn::tensor::{Bool, Device, Tensor, TensorData};
use burn_std::s;

use super::configuration_d1_omni::AudioConfig;

#[derive(Module, Debug)]
pub(super) struct Audio {
    encoder: Conformer,
    adapter: Adapter,
    residual: Residual,
}
#[derive(Module, Debug)]
struct Conformer {
    pre_encode: Subsampling,
    layers: Vec<ConformerLayer>,
    width: usize,
}
#[derive(Module, Debug)]
struct Subsampling {
    conv: (Conv2d, Relu, Conv2d, Conv2d, Relu, Conv2d, Conv2d, Relu),
    out: Linear,
}
#[derive(Module, Debug)]
struct ConformerLayer {
    norm_feed_forward1: LayerNorm,
    feed_forward1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelativeAttention,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_feed_forward2: LayerNorm,
    feed_forward2: FeedForward,
    norm_out: LayerNorm,
}
#[derive(Module, Debug)]
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}
#[derive(Module, Debug)]
struct RelativeAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Param<Tensor<2>>,
    pos_bias_v: Param<Tensor<2>>,
    heads: usize,
}
#[derive(Module, Debug)]
struct ConvModule {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    batch_norm: InferenceBatchNorm,
    pointwise_conv2: Conv1d,
}
#[derive(Module, Debug)]
struct InferenceBatchNorm {
    // Keep pretrained statistics lazy and required, e.g. a missing running_var must fail loading.
    weight: Param<Tensor<1>>,
    bias: Param<Tensor<1>>,
    running_mean: Param<Tensor<1>>,
    running_var: Param<Tensor<1>>,
}
#[derive(Module, Debug)]
struct Adapter {
    norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
}
#[derive(Module, Debug)]
struct Residual {
    ln: LayerNorm,
    down: Linear,
    up: Linear,
}

impl Audio {
    pub(super) fn new(c: &AudioConfig, out: usize, device: &Device) -> Self {
        let d = c.d_model;
        let linear = |i, o| LinearConfig::new(i, o).init(device);
        let norm = || LayerNormConfig::new(d).with_epsilon(1e-5).init(device);
        let ff = || FeedForward {
            linear1: linear(d, d * c.ff_expansion_factor),
            linear2: linear(d * c.ff_expansion_factor, d),
        };
        let conv1 = |i, o, k, groups, pad| {
            Conv1dConfig::new(i, o, k)
                .with_groups(groups)
                .with_padding(PaddingConfig1d::Explicit(pad, pad))
                .init(device)
        };
        let conv2 = |i, o, k, groups, stride, pad| {
            Conv2dConfig::new([i, o], [k, k])
                .with_groups(groups)
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                .init(device)
        };
        let ch = c.subsampling_conv_channels;
        Self {
            encoder: Conformer {
                pre_encode: Subsampling {
                    conv: (
                        conv2(1, ch, 3, 1, 2, 1),
                        Relu::new(),
                        conv2(ch, ch, 3, ch, 2, 1),
                        conv2(ch, ch, 1, 1, 1, 0),
                        Relu::new(),
                        conv2(ch, ch, 3, ch, 2, 1),
                        conv2(ch, ch, 1, 1, 1, 0),
                        Relu::new(),
                    ),
                    out: linear(ch * c.feat_in.div_ceil(8), d),
                },
                layers: (0..c.n_layers)
                    .map(|_| ConformerLayer {
                        norm_feed_forward1: norm(),
                        feed_forward1: ff(),
                        norm_self_att: norm(),
                        self_attn: RelativeAttention {
                            linear_q: linear(d, d),
                            linear_k: linear(d, d),
                            linear_v: linear(d, d),
                            linear_out: linear(d, d),
                            linear_pos: LinearConfig::new(d, d).with_bias(false).init(device),
                            pos_bias_u: Initializer::Zeros.init([c.n_heads, d / c.n_heads], device),
                            pos_bias_v: Initializer::Zeros.init([c.n_heads, d / c.n_heads], device),
                            heads: c.n_heads,
                        },
                        norm_conv: norm(),
                        conv: ConvModule {
                            pointwise_conv1: conv1(d, 2 * d, 1, 1, 0),
                            depthwise_conv: conv1(
                                d,
                                d,
                                c.conv_kernel_size,
                                d,
                                (c.conv_kernel_size - 1) / 2,
                            ),
                            batch_norm: InferenceBatchNorm {
                                weight: Initializer::Ones.init([d], device),
                                bias: Initializer::Zeros.init([d], device),
                                running_mean: Initializer::Zeros.init([d], device),
                                running_var: Initializer::Ones.init([d], device),
                            },
                            pointwise_conv2: conv1(d, d, 1, 1, 0),
                        },
                        norm_feed_forward2: norm(),
                        feed_forward2: ff(),
                        norm_out: norm(),
                    })
                    .collect(),
                width: d,
            },
            adapter: Adapter {
                norm: norm(),
                linear_1: linear(d, out),
                linear_2: linear(out, out),
            },
            residual: Residual {
                ln: LayerNormConfig::new(out).with_epsilon(1e-5).init(device),
                down: linear(out, c.residual_width),
                up: linear(c.residual_width, out),
            },
        }
    }

    pub(super) fn forward(&self, samples: &[f32], device: &Device) -> Tensor<3> {
        let (mel, frames) = mel_features(samples, device);
        let (mut x, valid) = self.encoder.pre_encode.forward(mel, frames);
        let t = x.dims()[1];
        let positions: Vec<_> = (0..2 * t - 1)
            .flat_map(|i| {
                (0..self.encoder.width).map(move |j| {
                    let angle = (t as f32 - 1.0 - i as f32)
                        * ((2 * (j / 2)) as f32 * -(10000.0_f32.ln() / self.encoder.width as f32))
                            .exp();
                    if j % 2 == 0 { angle.sin() } else { angle.cos() }
                })
            })
            .collect();
        let pos = Tensor::<3>::from_data(
            TensorData::new(positions, [1, 2 * t - 1, self.encoder.width]),
            device,
        );
        for layer in &self.encoder.layers {
            let h = layer.norm_feed_forward1.forward(x.clone());
            x = x + layer.feed_forward1.forward(h) * 0.5;
            let h = layer.norm_self_att.forward(x.clone());
            x = x + layer.self_attn.forward(h, pos.clone(), valid);
            let h = layer.norm_conv.forward(x.clone());
            x = x + layer.conv.forward(h, valid);
            let h = layer.norm_feed_forward2.forward(x.clone());
            x = x + layer.feed_forward2.forward(h) * 0.5;
            x = layer.norm_out.forward(x);
        }
        let x = self.adapter.linear_2.forward(gelu(
            self.adapter
                .linear_1
                .forward(self.adapter.norm.forward(x.slice(s![.., ..valid, ..]))),
        ));
        x.clone()
            + self.residual.up.forward(gelu(
                self.residual.down.forward(self.residual.ln.forward(x)),
            ))
    }
}

fn time_mask(x: Tensor<4>, valid: usize) -> Tensor<4> {
    let t = x.dims()[2];
    let values: Vec<_> = (0..t)
        .map(|i| if i < valid { 1.0_f32 } else { 0.0 })
        .collect();
    let mask = Tensor::<4>::from_data(TensorData::new(values, [1, 1, t, 1]), &x.device());
    x * mask
}

impl Subsampling {
    fn forward(&self, mel: Tensor<3>, mut valid: usize) -> (Tensor<3>, usize) {
        let mut x = self
            .conv
            .0
            .forward(time_mask(mel.unsqueeze_dim::<4>(1), valid));
        valid = valid.div_ceil(2);
        x = relu(time_mask(x, valid));
        x = self.conv.2.forward(time_mask(x, valid));
        valid = valid.div_ceil(2);
        x = self.conv.3.forward(time_mask(x, valid));
        x = relu(time_mask(x, valid));
        x = self.conv.5.forward(time_mask(x, valid));
        valid = valid.div_ceil(2);
        x = self.conv.6.forward(time_mask(x, valid));
        x = relu(time_mask(x, valid));
        x = time_mask(x, valid);
        let [b, c, t, f] = x.dims();
        (
            self.out.forward(x.swap_dims(1, 2).reshape([b, t, c * f])),
            valid,
        )
    }
}

impl FeedForward {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.linear2.forward(silu(self.linear1.forward(x)))
    }
}

impl RelativeAttention {
    fn forward(&self, x: Tensor<3>, pos: Tensor<3>, valid: usize) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let dim = d / self.heads;
        let q = self
            .linear_q
            .forward(x.clone())
            .reshape([b, t, self.heads, dim]);
        let k = self
            .linear_k
            .forward(x.clone())
            .reshape([b, t, self.heads, dim])
            .swap_dims(1, 2);
        let v = self
            .linear_v
            .forward(x)
            .reshape([b, t, self.heads, dim])
            .swap_dims(1, 2);
        let p = self
            .linear_pos
            .forward(pos)
            .reshape([1, 2 * t - 1, self.heads, dim])
            .swap_dims(1, 2);
        let ac = (q.clone() + self.pos_bias_u.val().unsqueeze::<4>())
            .swap_dims(1, 2)
            .matmul(k.transpose());
        let bd = (q + self.pos_bias_v.val().unsqueeze::<4>())
            .swap_dims(1, 2)
            .matmul(p.transpose());
        let zero = Tensor::<4>::zeros([b, self.heads, t, 1], &bd.device());
        let bd = Tensor::cat(vec![zero, bd], 3)
            .reshape([b, self.heads, 2 * t, t])
            .slice(s![.., .., 1.., ..])
            .reshape([b, self.heads, t, 2 * t - 1])
            .slice(s![.., .., .., ..t]);
        let scores = (ac + bd) / (dim as f64).sqrt();
        let mask: Vec<_> = (0..t)
            .flat_map(|q| (0..t).map(move |k| q >= valid || k >= valid))
            .collect();
        let mask =
            Tensor::<4, Bool>::from_data(TensorData::new(mask, [1, 1, t, t]), &scores.device())
                .expand([b, self.heads, t, t]);
        let weights = softmax(scores.mask_fill(mask.clone(), -10000.0), 3).mask_fill(mask, 0.0);
        self.linear_out
            .forward(weights.matmul(v).swap_dims(1, 2).reshape([b, t, d]))
    }
}

impl ConvModule {
    fn forward(&self, x: Tensor<3>, valid: usize) -> Tensor<3> {
        let x = self.pointwise_conv1.forward(x.swap_dims(1, 2));
        let x = glu(x, 1);
        let t = x.dims()[2];
        let keep = Tensor::<3>::from_data(
            TensorData::new(
                (0..t)
                    .map(|i| if i < valid { 1.0_f32 } else { 0.0 })
                    .collect::<Vec<_>>(),
                [1, 1, t],
            ),
            &x.device(),
        );
        let x = self.depthwise_conv.forward(x * keep);
        let bn = &self.batch_norm;
        let x = batch_norm(
            x,
            bn.weight.val(),
            bn.bias.val(),
            bn.running_mean.val(),
            bn.running_var.val(),
            1e-5,
        );
        self.pointwise_conv2.forward(silu(x)).swap_dims(1, 2)
    }
}

pub(super) fn mel_features(samples: &[f32], device: &Device) -> (Tensor<3>, usize) {
    let frames = samples.len() / 160;
    let x = Tensor::<2>::from_data(
        TensorData::new(samples.to_vec(), [1, samples.len()]),
        device,
    );
    let emphasized = Tensor::cat(
        vec![
            x.clone().slice(s![.., ..1]),
            x.clone().slice(s![.., 1..]) - x.slice(s![.., ..samples.len() - 1]) * 0.97,
        ],
        1,
    );
    // Burn centers with reflection; the trained frontend uses zero padding instead.
    let spectrum = stft(
        emphasized.pad([(0, 0), (256, 256)], PadMode::Constant(0.0)),
        Some(hann_window(400, false, device)),
        StftOptions {
            n_fft: 512,
            hop_length: 160,
            win_length: Some(400),
            center: false,
            onesided: true,
        },
    );
    let power: Tensor<3> = spectrum.square().sum_dim(3).sqrt().square().squeeze_dim(3);
    let bank: Vec<_> = filterbank().into_iter().flatten().collect();
    let bank = Tensor::<2>::from_data(TensorData::new(bank, [128, 257]), device)
        .transpose()
        .unsqueeze_dim::<3>(0);
    let mel = (power.matmul(bank) + 2.0_f64.powi(-24)).log();
    let valid = mel.clone().slice(s![.., ..frames, ..]);
    // var_mean applies the sample-variance correction used by NeMo, e.g. divide by frames - 1.
    let (variance, mean) = valid.var_mean(1);
    let normalized = (mel - mean) / (variance.sqrt() + 1e-5);
    let mask = Tensor::<3, Bool>::from_data(
        TensorData::new(
            (0..=frames).map(|i| i >= frames).collect::<Vec<_>>(),
            [1, frames + 1, 1],
        ),
        device,
    );
    (
        normalized.mask_fill(mask.expand([1, frames + 1, 128]), 0.0),
        frames,
    )
}

fn filterbank() -> Vec<Vec<f32>> {
    let logstep = 6.4_f64.ln() / 27.0;
    let maximum = 15.0 + 8.0_f64.ln() / logstep;
    let edges: Vec<_> = (0..130)
        .map(|i| {
            let mel = maximum * i as f64 / 129.0;
            if mel >= 15.0 {
                1000.0 * (logstep * (mel - 15.0)).exp()
            } else {
                (200.0 / 3.0) * mel
            }
        })
        .collect();
    edges
        .windows(3)
        .map(|edge| {
            let [left, center, right] =
                <[f64; 3]>::try_from(edge).expect("windows(3) always has three edges");
            (0..257)
                .map(|i| {
                    let hz = i as f64 * 16000.0 / 512.0;
                    let weight = ((hz - left) / (center - left))
                        .min((right - hz) / (right - center))
                        .max(0.0) as f32;
                    weight * (2.0 / (right - left)) as f32
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontend_matches_python(device: Device) {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/tiny-d1-omni/mel.json"
        ))
        .expect("valid frontend reference");
        let samples: Vec<f32> = reference["samples"]
            .as_array()
            .expect("valid frontend reference")
            .iter()
            .map(|x| x.as_i64().expect("valid frontend reference") as f32 / 32768.0)
            .collect();
        let (mel, frames) = mel_features(&samples, &device);
        assert_eq!(
            frames,
            reference["frames"].as_u64().expect("reference frame count") as usize
        );
        let values = mel
            .try_into_vec_as::<f32>()
            .expect("valid frontend reference");
        let expected = reference["mel"]
            .as_array()
            .expect("valid frontend reference");
        assert_eq!(values.len(), expected.len());
        for (i, (&actual, expected)) in values.iter().zip(expected).enumerate() {
            let expected = expected.as_f64().expect("valid frontend reference") as f32;
            assert!(
                (actual - expected).abs() < 4e-4,
                "mel {i}: {actual} != {expected}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_frontend_matches_python() {
        frontend_matches_python(Device::flex());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_frontend_matches_python() {
        frontend_matches_python(Device::wgpu(Default::default()));
    }
}
