//! Share scalar decision decoding across model families, e.g. D1 and Qwen readouts.
use indexmap::IndexMap;

use crate::{Action, Answer, Error, Question, Result};

pub(crate) fn softmax(logits: &[f64]) -> Result<Vec<f64>> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::Inference("non-finite or empty answer logits".into()));
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let values: Vec<_> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = values.iter().sum();
    Ok(values.into_iter().map(|value| value / sum).collect())
}

pub(crate) fn answer(
    question: &Question,
    probabilities: Vec<f64>,
    confidence: f64,
    legend: Vec<String>,
    true_index: usize,
) -> Result<Answer> {
    let max = probabilities.iter().copied().fold(0.0, f64::max);
    let action = Action::new(1.0);
    Ok(match question {
        Question::Noul { .. } => Answer::Noul {
            noul: *probabilities
                .get(true_index)
                .ok_or_else(|| Error::Inference("missing boolean probability".into()))?,
            confidence,
            answer_confidence: max,
            action,
        },
        Question::Choice { criteria, .. } => {
            let best = argmax(&probabilities);
            Answer::Choice {
                choice: criteria
                    .get_index(best)
                    .ok_or_else(|| Error::Inference("missing choice probability".into()))?
                    .0
                    .clone(),
                probabilities: criteria.keys().cloned().zip(probabilities).collect(),
                confidence,
                answer_confidence: max,
                action,
            }
        }
        Question::Score { .. } => Answer::Score {
            score: probabilities
                .iter()
                .enumerate()
                .map(|(i, probability)| i as f64 * probability)
                .sum(),
            probabilities: probabilities
                .into_iter()
                .enumerate()
                .map(|(i, probability)| (i.to_string(), probability))
                .collect(),
            legend: legend
                .into_iter()
                .enumerate()
                .map(|(i, text)| (i.to_string(), text))
                .collect::<IndexMap<_, _>>(),
            confidence,
            answer_confidence: max,
            action,
        },
    })
}

pub(crate) fn argmax(probabilities: &[f64]) -> usize {
    // Keep insertion order on ties, e.g. equal A/B probabilities select A.
    probabilities
        .iter()
        .enumerate()
        .fold(
            (0, -1.0),
            |best, (i, &p)| if p > best.1 { (i, p) } else { best },
        )
        .0
}

pub(crate) fn choice_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() == 1 {
        return 1.0;
    }
    let prior = 1.0 / probabilities.len() as f64;
    (probabilities.iter().copied().fold(0.0, f64::max) - prior) / (1.0 - prior)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(vec![])]
    #[case(vec![f64::NAN])]
    #[case(vec![f64::INFINITY, 0.0])]
    #[case(vec![0.0, f64::NEG_INFINITY])]
    fn rejects_invalid_logits(#[case] logits: Vec<f64>) {
        assert!(matches!(softmax(&logits), Err(Error::Inference(_))));
    }

    #[test]
    fn stable_probabilities_preserve_ties_and_choice_confidence() {
        // Large equal logits stay finite and prefer the first choice, e.g. A over B.
        let probabilities = softmax(&[1000.0, 1000.0, 999.0]).unwrap();
        assert_relative_eq!(probabilities.iter().sum::<f64>(), 1.0, epsilon = 1e-15);
        assert_eq!(argmax(&probabilities), 0);
        assert_relative_eq!(choice_confidence(&[0.5, 0.5]), 0.0);
        assert_relative_eq!(choice_confidence(&[1.0]), 1.0);
    }
}
