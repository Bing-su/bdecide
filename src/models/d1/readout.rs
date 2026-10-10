//! Share typed decision output, e.g. each model supplies its own true-label index.
use burn::tensor::Tensor;

use crate::models::qwen3_5::readout::{self, softmax};
use crate::utils::render;
use crate::{Answer, Error, Question, Result};

pub(super) fn values(tensor: Tensor<3>) -> Result<Vec<f64>> {
    tensor
        .try_into_vec_as::<f32>()
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|error| Error::Inference(error.to_string()))
}

pub(super) fn answer(q: &Question, logits: &[f64], true_index: usize) -> Result<Answer> {
    let probabilities = softmax(logits)?;
    let confidence = probabilities.iter().copied().fold(0.0, f64::max);
    let legend = if let Question::Score { criteria, .. } = q {
        criteria.iter().map(render).collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    readout::answer(q, probabilities, confidence, legend, true_index)
}
