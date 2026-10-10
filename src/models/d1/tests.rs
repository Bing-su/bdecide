//! Check actual released parameter layouts without allocating their production tensors.
use std::collections::BTreeMap;

use burn::module::{Module, ModuleVisitor, Param};
use burn::store::{ModuleAdapter, PyTorchToBurnAdapter};
use burn::tensor::{Device, Tensor};
use burn_std::s;
use camino::Utf8Path;
use rstest::rstest;

use super::modeling_d1_omni::OmniModel;
use super::{D1OmniConfig, D1OmniModel};
use crate::hub::ModelSource;
use crate::models::lfm2_vl::{Lfm2VlConfig, Lfm2VlForConditionalGeneration};

#[derive(Default)]
struct Shapes {
    stack: Vec<(String, String)>,
    shapes: BTreeMap<String, Vec<usize>>,
}

impl ModuleVisitor for Shapes {
    fn enter_module(&mut self, name: &str, kind: &str) {
        self.stack.push((name.into(), kind.into()));
    }

    fn exit_module(&mut self, _: &str, _: &str) {
        self.stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        assert!(
            !param.is_initialized(),
            "release schema validation must keep weights lazy"
        );
        let mut names: Vec<_> = self.stack.iter().map(|(name, _)| name.clone()).collect();
        let (name, kind) = self.stack.last().expect("parameter module path");
        if let Some(alias) = PyTorchToBurnAdapter.get_alternative_param_name(name, kind) {
            *names.last_mut().expect("parameter name") = alias;
        }
        let mut shape: Vec<usize> = param.lazy_shape().iter().copied().collect();
        if kind == "Struct:Linear" {
            shape.reverse();
            if name == "bias" {
                shape.reverse();
            }
        }
        let mut name = names.join(".");
        if name.contains(".self_attn.in_proj.") {
            name = name.replace(".self_attn.in_proj.", ".self_attn.in_proj_");
        }
        self.shapes.insert(name, shape);
    }
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::d1(
    false,
    include_str!("../../../tests/fixtures/tiny-d1-3b/release-config.json"),
    include_str!("../../../tests/fixtures/tiny-d1-3b/release-shapes.json"),
)]
#[case::d1_omni(
    true,
    include_str!("../../../tests/fixtures/tiny-d1-omni/release-config.json"),
    include_str!("../../../tests/fixtures/tiny-d1-omni/release-shapes.json"),
)]
fn matches_full_release_parameter_shapes(
    #[case] omni: bool,
    #[case] config: &str,
    #[case] shapes: &str,
) {
    let device = Device::flex();
    let mut visitor = Shapes::default();
    if omni {
        let c: D1OmniConfig = serde_json::from_str(config).expect("release config");
        OmniModel::new(&c, &device)
            .expect("release config")
            .visit(&mut visitor);
    } else {
        let c: Lfm2VlConfig = serde_json::from_str(config).expect("release config");
        Lfm2VlForConditionalGeneration::new(&c, &device)
            .expect("release config")
            .visit(&mut visitor);
    }
    let mut expected: BTreeMap<String, Vec<usize>> =
        serde_json::from_str(shapes).expect("release shapes");
    expected.retain(|name, _| !name.ends_with("num_batches_tracked"));
    assert_eq!(visitor.shapes.len(), expected.len(), "release tensor count");
    for (name, shape) in expected {
        assert_eq!(
            visitor.shapes.get(&name),
            Some(&shape),
            "release tensor {name}"
        );
    }
}

#[test]
fn omni_media_prefix_never_reads_question_text() {
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-d1-omni");
    let device = Device::flex();
    let model = D1OmniModel::from_pretrained(&ModelSource::Local(root), &device)
        .expect("tiny Omni checkpoint");
    let model = model.network;
    let h = model.encoder.embed(&[3, 4, 5, 3, 4, 5], &device);
    let changed = Tensor::cat(
        vec![
            h.clone().slice(s![.., ..3, ..]),
            h.clone().slice(s![.., 3.., ..]) * 10.0,
        ],
        1,
    );
    let before = model
        .encoder
        .forward_prefix(h, 3)
        .slice(s![.., ..3, ..])
        .try_into_vec_as::<f32>()
        .expect("prefix values");
    let after = model
        .encoder
        .forward_prefix(changed, 3)
        .slice(s![.., ..3, ..])
        .try_into_vec_as::<f32>()
        .expect("prefix values");
    for (before, after) in before.into_iter().zip(after) {
        assert!((before - after).abs() < 1e-6);
    }
}
