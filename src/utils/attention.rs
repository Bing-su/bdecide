//! Share multi-head attention across encoders and decision heads.
use burn::tensor::backend::Backend;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{BasicOps, Bool, Tensor, TensorData, module};
use burn_std::s;

// Centralize Burn's Flash input contract, e.g. transposed QKV and broadcast padding.
pub(crate) fn attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    options: AttentionModuleOptions,
) -> Tensor<B, 4> {
    module::attention(
        materialize(query),
        materialize(key),
        materialize(value),
        mask.map(materialize),
        None,
        options,
    )
}

fn materialize<B: Backend, K: BasicOps<B>>(input: Tensor<B, 4, K>) -> Tensor<B, 4, K> {
    // Copy into a dense buffer: Burn 0.21 Flash readers cannot use arbitrary strides,
    // e.g. swap_dims views or masks with a zero stride after expand.
    Tensor::empty(input.shape(), (&input.device(), input.dtype()))
        .slice_assign(s![.., .., .., ..], input)
}

pub(crate) fn attend<B: Backend>(
    qkv: Tensor<B, 3>,
    heads: usize,
    padding: Tensor<B, 4, Bool>,
    rope: Option<(f64, Option<usize>)>,
) -> Tensor<B, 3> {
    attend_with_positions(qkv, heads, padding, rope, None)
}

// Reuse attention with checkpoint-defined positions, e.g. Von restarts every option.
pub(crate) fn attend_with_positions<B: Backend>(
    qkv: Tensor<B, 3>,
    heads: usize,
    padding: Tensor<B, 4, Bool>,
    rope: Option<(f64, Option<usize>)>,
    positions: Option<&[usize]>,
) -> Tensor<B, 3> {
    let [batch, length, total] = qkv.dims();
    let hidden_size = total / 3;
    let head_width = hidden_size / heads;
    let device = qkv.device();
    let query = qkv
        .clone()
        .slice(s![.., .., 0..hidden_size])
        .reshape([batch, length, heads, head_width])
        .swap_dims(1, 2);
    let key = qkv
        .clone()
        .slice(s![.., .., hidden_size..2 * hidden_size])
        .reshape([batch, length, heads, head_width])
        .swap_dims(1, 2);
    let value = qkv
        .slice(s![.., .., 2 * hidden_size..3 * hidden_size])
        .reshape([batch, length, heads, head_width])
        .swap_dims(1, 2);
    let (query, key, mask) = if let Some((theta, window)) = rope {
        let mut cos = Vec::with_capacity(length * head_width);
        let mut sin = Vec::with_capacity(length * head_width);
        for position in 0..length {
            let position = positions
                .and_then(|ids| ids.get(position))
                .copied()
                .unwrap_or(position);
            for channel in 0..head_width {
                let exponent = (2 * (channel % (head_width / 2))) as f32 / head_width as f32;
                let phase = position as f32 * (1.0 / (theta as f32).powf(exponent));
                cos.push(phase.cos());
                sin.push(phase.sin());
            }
        }
        let cos =
            Tensor::<B, 4>::from_data(TensorData::new(cos, [1, 1, length, head_width]), &device);
        let sin =
            Tensor::<B, 4>::from_data(TensorData::new(sin, [1, 1, length, head_width]), &device);
        let apply_rotary = |hidden: Tensor<B, 4>| {
            // Split rotary channels while keeping every batch, head, and position.
            let first_half = hidden.clone().slice(s![.., .., .., 0..head_width / 2]);
            let second_half = hidden
                .clone()
                .slice(s![.., .., .., head_width / 2..head_width]);
            hidden * cos.clone() + Tensor::cat(vec![-second_half, first_half], 3) * sin.clone()
        };
        let mask = match window {
            Some(window) => {
                let local: Vec<bool> = (0..length)
                    .flat_map(|query_position| {
                        (0..length).map(move |key_position| {
                            let query = positions
                                .and_then(|ids| ids.get(query_position))
                                .copied()
                                .unwrap_or(query_position);
                            let key = positions
                                .and_then(|ids| ids.get(key_position))
                                .copied()
                                .unwrap_or(key_position);
                            query.abs_diff(key) > window
                        })
                    })
                    .collect();
                padding.bool_or(Tensor::<B, 4, Bool>::from_data(
                    TensorData::new(local, [1, 1, length, length]),
                    &device,
                ))
            }
            None => padding,
        };
        (apply_rotary(query), apply_rotary(key), mask)
    } else {
        (query, key, padding)
    };
    // Bool masks and default scaling allow Burn to select Flash Attention, e.g. for local RoPE.
    let mask = mask.expand([batch, heads, length, length]);
    attention(query, key, value, Some(mask), Default::default())
        .swap_dims(1, 2)
        .reshape([batch, length, hidden_size])
}

#[cfg(test)]
mod tests {
    use approx::abs_diff_eq;
    #[cfg(feature = "cpu")]
    use burn::backend::Flex;
    #[cfg(feature = "wgpu")]
    use burn::backend::Wgpu;

    use super::*;

    fn matches_masked_means<B: Backend>() {
        let device = B::Device::default();
        // Zero Q/K make the expected result an independent mean of visible values.
        // 513 tokens also exercise Flex's tiled path; late padded local rows see no keys.
        for length in [6_usize, 513] {
            let mut qkv = Vec::new();
            let mut padding = Vec::new();
            for batch in 0..2 {
                let valid = if batch == 0 { length } else { length / 2 };
                for position in 0..length {
                    qkv.extend([0.0_f32; 8]);
                    qkv.extend((0..4).map(|channel| {
                        (batch * 10 + channel * 2) as f32 + position as f32 / length as f32
                    }));
                    padding.push(position >= valid);
                }
            }
            let qkv = Tensor::<B, 3>::from_data(TensorData::new(qkv, [2, length, 12]), &device);
            let padding = Tensor::<B, 4, Bool>::from_data(
                TensorData::new(padding, [2, 1, 1, length]),
                &device,
            );
            for rope in [None, Some((10_000.0, None)), Some((10_000.0, Some(1)))] {
                let actual = attend(qkv.clone(), 2, padding.clone(), rope)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                for batch in 0..2 {
                    let valid = if batch == 0 { length } else { length / 2 };
                    for position in 0..length {
                        let visible: Vec<_> = (0..valid)
                            .filter(|&key| {
                                rope.and_then(|(_, window)| window)
                                    .is_none_or(|window| position.abs_diff(key) <= window)
                            })
                            .collect();
                        for channel in 0..4 {
                            let expected = if visible.is_empty() {
                                0.0
                            } else {
                                (batch * 10 + channel * 2) as f32
                                    + visible.iter().sum::<usize>() as f32
                                        / (visible.len() * length) as f32
                            };
                            let actual = actual[(batch * length + position) * 4 + channel];
                            assert!(
                                abs_diff_eq!(actual, expected, epsilon = 2e-5),
                                "length={length}, rope={rope:?}, batch={batch}, position={position}, \
                                 channel={channel}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_matches_masked_means() {
        matches_masked_means::<Flex>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_masked_means() {
        matches_masked_means::<Wgpu<f32, i32>>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_materializes_flash_inputs() {
        use burn::backend::wgpu::{CubeBackend, WgpuRuntime};

        type B = CubeBackend<WgpuRuntime, f32, i32, u32>;
        let device = Default::default();
        // Distinct batch/head values reveal misaddressing, e.g. a head reading its neighbor.
        let values: Vec<_> = (0..2 * 32 * 2 * 16).map(|i| i as f32).collect();
        let input =
            Tensor::<B, 4>::from_data(TensorData::new(values.clone(), [2, 32, 2, 16]), &device)
                .swap_dims(1, 2);
        let output = materialize(input);
        assert_eq!(
            output
                .clone()
                .into_primitive()
                .tensor()
                .meta
                .strides()
                .to_vec(),
            [1024, 512, 16, 1],
        );
        assert_eq!(
            output.swap_dims(1, 2).into_data().to_vec::<f32>().unwrap(),
            values,
        );

        // Head/query broadcasts must become physical rows, including a one-token sequence.
        for length in [1, 32] {
            let padding: Vec<_> = (0..2 * length).map(|i| i >= length + length / 2).collect();
            let mask = Tensor::<B, 4, Bool>::from_data(
                TensorData::new(padding.clone(), [2, 1, 1, length]),
                &device,
            )
            .expand([2, 2, length, length]);
            let output = materialize(mask);
            assert_eq!(
                output.clone().into_primitive().meta.strides().to_vec(),
                [2 * length * length, length * length, length, 1],
            );
            let actual: Vec<bool> = output.into_data().iter().collect();
            let expected: Vec<_> = padding
                .chunks_exact(length)
                .flat_map(|row| (0..2 * length).flat_map(move |_| row.iter().copied()))
                .collect();
            assert_eq!(actual, expected);
        }
    }
}
