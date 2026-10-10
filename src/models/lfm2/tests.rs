//! Check convolution boundary semantics independently, e.g. a one-token media prefix.
use burn::module::Param;
use burn::nn::Initializer;
use rstest::rstest;

use super::*;

fn short_conv_preserves_prefix_edges(
    device: Device,
    bidirectional: bool,
    prefix: usize,
    input: &[f32],
    expected: &[f32],
) {
    let mut conv = Conv1dConfig::new(1, 1, 3)
        .with_padding(if bidirectional {
            PaddingConfig1d::Explicit(1, 1)
        } else {
            PaddingConfig1d::Explicit(2, 0)
        })
        .init(&device);
    // Fixed taps and a nonzero bias expose missing boundary taps, e.g. last media token.
    conv.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(vec![2.0_f32, 3.0, 5.0], [1, 1, 3]),
        &device,
    ));
    conv.bias = Some(Param::from_tensor(Tensor::from_data([7.0_f32], &device)));
    let linear = |out| {
        LinearConfig::new(1, out)
            .with_bias(false)
            .with_initializer(Initializer::Ones)
            .init(&device)
    };
    let layer = ShortConv {
        conv,
        in_proj: linear(3),
        out_proj: linear(1),
    };
    let length = input.len();
    let x = Tensor::from_data(TensorData::new(input.to_vec(), [1, length, 1]), &device);
    let actual = layer
        .forward(x, bidirectional, prefix)
        .try_into_vec_as::<f32>()
        .expect("convolution values");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "bidirectional={bidirectional}, prefix={prefix}: {actual} != {expected}"
        );
    }
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::causal(false, 0, &[1., 2., 3., 4.], &[12., 60., 198., 488.])]
#[case::bidirectional(true, 0, &[1., 2., 3., 4.], &[30., 132., 366., 292.])]
#[case::one_token_prefix(true, 1, &[1., 2., 3., 4.], &[10., 132., 366., 292.])]
#[case::two_token_prefix(true, 2, &[1., 2., 3., 4.], &[30., 42., 366., 292.])]
#[case::full_prefix(true, 4, &[1., 2., 3., 4.], &[30., 132., 366., 292.])]
#[case::single_causal(false, 0, &[2.], &[54.])]
#[case::single_prefix(true, 1, &[2.], &[38.])]
fn cpu_short_conv_preserves_prefix_edges(
    #[case] bidirectional: bool,
    #[case] prefix: usize,
    #[case] input: &[f32],
    #[case] expected: &[f32],
) {
    short_conv_preserves_prefix_edges(Device::flex(), bidirectional, prefix, input, expected);
}

#[cfg(feature = "wgpu")]
#[rstest]
#[case::causal(false, 0, &[1., 2., 3., 4.], &[12., 60., 198., 488.])]
#[case::bidirectional(true, 0, &[1., 2., 3., 4.], &[30., 132., 366., 292.])]
#[case::one_token_prefix(true, 1, &[1., 2., 3., 4.], &[10., 132., 366., 292.])]
#[case::two_token_prefix(true, 2, &[1., 2., 3., 4.], &[30., 42., 366., 292.])]
#[case::full_prefix(true, 4, &[1., 2., 3., 4.], &[30., 132., 366., 292.])]
#[case::single_causal(false, 0, &[2.], &[54.])]
#[case::single_prefix(true, 1, &[2.], &[38.])]
#[ignore = "requires a wgpu adapter"]
fn wgpu_short_conv_preserves_prefix_edges(
    #[case] bidirectional: bool,
    #[case] prefix: usize,
    #[case] input: &[f32],
    #[case] expected: &[f32],
) {
    short_conv_preserves_prefix_edges(
        Device::wgpu(Default::default()),
        bidirectional,
        prefix,
        input,
        expected,
    );
}
