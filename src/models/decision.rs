//! Define the shared System One contract, e.g. repeated requests reuse loaded models.
use std::panic::{AssertUnwindSafe, catch_unwind};

use burn_std::stream::Stream;
use itertools::Itertools;

use crate::{
    DecisionsRequest,
    DecisionsResponse,
    Error,
    Metadata,
    Result,
    SystemOneRequest,
    SystemOneResponse,
};

/// Implement this interface to use an independently maintained decision model.
///
/// Implementations own preprocessing and output semantics; for example a Clef
/// implementation can accept the same typed questions with its own processor.
pub trait DecisionModel {
    /// Evaluate OpenAI-shaped text questions with this loaded model.
    ///
    /// For example, `model.decisions(&request)?` maps predicate questions to Noul.
    /// `request.model` must equal `metadata().model_id`; no model is downloaded or switched.
    /// Confidence and token counts follow the local model, not OpenAI calibration.
    fn decisions(&self, request: &DecisionsRequest) -> Result<DecisionsResponse> {
        let converted = request.to_system_one()?;
        if request.model != self.metadata().model_id {
            return Err(Error::InvalidRequest(format!(
                "requested model {:?} does not match loaded model {:?}",
                request.model,
                self.metadata().model_id,
            )));
        }
        request.decode(self.system_one(&converted)?)
    }

    /// Preprocess a shared state, infer its typed answers, and decode the results.
    ///
    /// For example, `model.system_one(&request)?` evaluates Choice and Score questions together.
    fn system_one(&self, request: &SystemOneRequest) -> Result<SystemOneResponse>;

    /// Evaluate requests concurrently on Burn streams, preserving input order.
    ///
    /// For example, `model.system_one_batch(&[first, second])?` shares loaded weights.
    /// Empty input returns an empty vector; a failed request fails the batch.
    /// By default at most four workers run together, limiting concurrent activation memory.
    /// Speedup depends on backend scheduling and workload; this does not pad or
    /// combine requests into one tensor. Implementations may override it to do so.
    fn system_one_batch(&self, requests: &[SystemOneRequest]) -> Result<Vec<SystemOneResponse>>
    where
        Self: Sync,
    {
        if requests.len() <= 1 {
            return requests
                .iter()
                .map(|request| self.system_one(request))
                .try_collect();
        }
        // Bound activation memory but allow GPU streams even on a single-core host.
        let workers = std::thread::available_parallelism()
            .map_or(2, usize::from)
            .clamp(2, 4)
            .min(requests.len());
        std::thread::scope(|scope| {
            let handles = requests
                .chunks(requests.len().div_ceil(workers))
                .map(|chunk| {
                    let stream = Stream::new();
                    scope.spawn(move || {
                        catch_unwind(AssertUnwindSafe(|| {
                            stream.enter(|| {
                                chunk
                                    .iter()
                                    .map(|request| self.system_one(request))
                                    .collect::<Result<Vec<_>>>()
                            })
                        }))
                        .map_err(|_panic| {
                            Error::Inference("Burn backend failed while executing the batch".into())
                        })?
                    })
                })
                .collect_vec();
            // Join every worker before returning errors, e.g. no inference outlives the call.
            let results = handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_panic| Error::Inference("batch worker failed".into()))?
                })
                .collect_vec();
            results.into_iter().flatten_ok().try_collect()
        })
    }

    fn metadata(&self) -> &Metadata;
}
