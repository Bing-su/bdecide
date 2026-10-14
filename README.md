# bdecide

Uses Burn 0.22 and requires Rust 1.95 or newer. See the upstream
[release notes](https://github.com/tracel-ai/burn/releases/tag/v0.22.0) and
[migration guide](https://burn.dev/books/burn/migrating-to-0.22.html).

Models and tensors no longer take a backend type parameter. `bdecide::Device`
re-exports `burn::tensor::Device` and works with both direct loaders and `AutoModel`.
The CLI retains its `cpu`, `wgpu`, and `auto` selections.
Model metadata reports the Burn device's debug representation, e.g. `format!("{device:?}")`.

`DecisionModel::decisions(&DecisionsRequest)` accepts the text subset of the
[OpenAI Decisions request format](https://developers.openai.com/api/reference/resources/decisions/methods/create)
and returns `DecisionsResponse`. It uses the loaded local model's `system_one`
implementation. Run `cargo run --example decisions -- ./checkpoint` for a complete example.

```rust,ignore
use bdecide::{DecisionModel, DecisionsRequest};

// Use the loaded model's ID, e.g. a local checkpoint path rather than an OpenAI model name.
let request = DecisionsRequest::new(&model.metadata().model_id, "I was charged twice.")
    .predicate("refund", "Is a refund needed?")
    .choice("department", "Who should handle this complaint?", ["billing", "shipping"])
    .score("severity", "How severe is this issue?", ["low", "medium", "high"]);
let response = model.decisions(&request)?;
let department = response.choice_str("department");
let refund_probability = response.probability("refund");
let severity = response.score("severity");
```

| Helper | Purpose |
| --- | --- |
| `DecisionsRequest::new(model, input)` | Accept text directly or a `Vec<DecisionInputMessage>`; begin with no questions |
| `.predicate(name, instructions)`, `.choice(name, instructions, choices)`, `.score(name, instructions, levels)` | Append named questions in order; choice values can be strings or booleans, score levels can be labels |
| `.question(DecisionQuestion::predicate(instructions))` | Append an unnamed question; `DecisionQuestion::choice` and `::score` also support `.with_name(name)` |
| `DecisionChoice::new(value).with_description(text)`, `DecisionLevel::new(label).with_description(text)` | Supply descriptions instead of bare choice values or level labels |
| `DecisionInputMessage::user(content)`, `DecisionInputPart::text(text)` | Create text messages or ordered text parts |
| `response.answer(name)`, `response.answers_named(name)` | Borrow the first matching answer or iterate all matches |
| `response.probability(name)`, `.choice(name)`, `.choice_str(name)`, `.choice_bool(name)`, `.score(name)`, `.confidence(name)` | Read values without matching nested enums; `None` means absent or a different answer/value type |
| `response.choice_probability(name, value)`, `.score_probability(name, index)` | Read a specific typed choice or zero-based level probability |
| `answer.name()`, `.probability()`, `.choice()`, `.score()`, `.confidence()`, `.is_refusal()` | Inspect an individual answer, including unnamed answers from `response.answers` |

Constructors defer validation to `request.validate()` or `model.decisions(&request)`.
Single-answer lookups always use the first matching name, including a refusal;
they do not skip it in favor of a later matching type. Use `answers_named` for
duplicate names and `answer.is_refusal()` to distinguish refusal from other types.
String `"true"` and boolean `true` remain distinct during probability lookup.

| Decisions field | Local behavior |
| --- | --- |
| `model` | Must exactly match `model.metadata().model_id`; mismatch returns an error |
| `input` | Text or user messages; adjacent `input_text` parts join verbatim, messages become an ordered `state` conversation array |
| `predicate` | Default Noul prompt; returns the probability of true |
| `choice` | 2–255 distinct string/boolean values; JSON literal keys and full option objects preserve values and descriptions in prompts |
| `score` | Ordered label/description objects render as JSON text criteria; output keeps original labels and zero-based indices |
| `name`, `answers` | Optional names, including duplicates and empty strings, survive; answers follow question order |
| `confidence` | The local model's confidence, without OpenAI calibration guarantees |
| `usage` | Local input/output counts and their sum; cache and reasoning counts are zero because these local paths do not use them |
| `safety_identifier` | Accepts at most 128 characters; no local abuse tracking effect |
| Images | Explicit error before inference; use `system_one` for supported local image inputs |
| Refusal | Can deserialize hosted refusal answers; local failures remain errors, without action-based refusal thresholds |

Question and choice IDs used internally are independent of optional names.
All model-specific context and option limits still apply, and truncation remains
disabled. Existing System One requests and the `predict` CLI keep their formats.
This adapter supplies a Rust method and JSON types; it does not start an HTTP
server or call OpenAI.

```rust,ignore
use bdecide::{DecisionModel, SystemOneRequest};

// Keep local choice IDs and JSON criteria, e.g. route a complaint to billing.
let request = SystemOneRequest::builder().state("I was charged twice.").build()
    .noul("refund", "Is a refund needed?")
    .choice("department", "Route", [("billing", "Payments"), ("shipping", "Delivery")])
    .score("severity", "Rate", ["low", "medium", "high"]);
let response = model.system_one(&request)?;
let department = response.choice("department");
let refund_probability = response.noul("refund");
let severity = response.score("severity");
```

| System One helper | Purpose |
| --- | --- |
| `SystemOneRequest::builder().state(state).build()` | Accept text or JSON state; start with no questions and default options |
| `SystemOneRequest::new(state, questions)` | Keep the existing constructor with an ordered question map |
| `.noul(id, instructions)`, `.choice(id, instructions, criteria)`, `.score(id, instructions, criteria)` | Insert questions in order; choices take `(ID, JSON value)` pairs and scores take ordered JSON levels |
| `.question(id, question)`, `SystemOneQuestion::noul`, `SystemOneQuestion::choice`, `SystemOneQuestion::score` | Build questions separately; use enum fields for custom Noul criteria and labels |
| `response.answer(id)`, `.noul(id)`, `.choice(id)`, `.score(id)` | Borrow an answer or read its value; missing IDs or different answer types return `None` |
| `response.confidence(id)`, `.answer_confidence(id)`, `.action(id)`, `.act_probability(id)` | Read model confidence, answer confidence, and action fields for any answer type |
| `response.choice_probability(id, choice_id)`, `.score_probability(id, index)` | Look up an exact choice ID or zero-based score level key, independently of map order |
| `answer.noul()`, `.choice()`, `.score()`, `.confidence()`, `.answer_confidence()`, `.action()`, `.act_probability()`, `.choice_probability(choice_id)`, `.score_probability(index)` | Inspect an individual `SystemOneAnswer`; common fields are always present |

System One helpers defer validation to `request.validate()` or `model.system_one(&request)`.
Reusing a question or choice ID replaces its value while preserving its original position.
Serialization, media fields, and inference options keep their existing formats and defaults.

| Module | Files and responsibility |
| --- | --- |
| `src/system_one/` | `request.rs`: state, media, options, request validation; `question.rs`: questions, labels, and criteria validation; `response.rs`: response, metadata, and usage; `answer.rs`: answers and actions |
| `src/decisions/` | `request.rs`, `question.rs`, `response.rs`, `answer.rs`: the corresponding Decisions types and helpers; `input.rs`: messages and content; `adapter.rs`: validation and conversion to/from System One |

Each `mod.rs` declares private implementation files and re-exports the module's types.
Callers can import `bdecide::system_one::SystemOneRequest` or `bdecide::decisions::DecisionsRequest`;
existing crate-root imports such as `bdecide::SystemOneRequest` remain available.

`DecisionModel::system_one_batch(&[SystemOneRequest])` returns
`Result<Vec<SystemOneResponse>>` in input order and reuses the loaded weights.

| Batch behavior | Contract |
| --- | --- |
| Laya | Combine every request's questions into one padded Burn tensor batch and forward pass |
| Other built-in models | Run requests on explicit Burn streams with up to four workers; a single request reuses `system_one` |
| Empty input | Return an empty vector |
| Failed request | Return an error for the batch; wait for started workers before returning |
| Usage and answers | Preserve each request's accounting and question order |
| Custom implementations | Batch calls require `Sync`; existing single-request implementations remain valid |

```rust,ignore
use bdecide::DecisionModel;

// Reuse loaded weights across independent states, e.g. two customer records.
let responses = model.system_one_batch(&[first_request, second_request])?;
```

The default implementation uses `burn_std::stream::Stream::enter` on scoped
workers. Burn/CubeCL schedules these streams; CUDA/ROCm can overlap kernels,
while WGPU interleaves submissions. See [Burn's multi-stream explanation](https://tracel.ai/blog/release-0.19.0/).
Actual speedup depends on backend, request sizes and available memory.
Concurrent requests use more activation memory, and Laya pads to the longest
question in the supplied batch; callers should keep batches within their device budget.

| `LoadOptions.device`   | Behavior                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `Some(device)`         | Use the supplied Burn device without replacing it or falling back                   |
| `None`                 | Use Burn's default device; retry on CPU after backend failure when `cpu` is enabled |
| Omitted in the builder | Same as `None`: automatic selection                                                 |

```rust,ignore
use bdecide::{AutoModel, Device, LoadOptions};

// Share an explicit CPU device with other Burn models, e.g. in a mixed CPU/WGPU build.
let model = AutoModel::from_pretrained(
    LoadOptions::builder()
        .source(bdecide::hub::ModelSource::Local("checkpoint".into()))
        .device(Device::flex())
        .build(),
)?;
```

Library users can enable additional backends on their own Burn 0.22 dependency.
Cargo unifies those features with bdecide's Burn dependency; no matching bdecide
feature is required for an explicitly supplied device. For example:

```toml
[dependencies]
bdecide = { version = "0.1", default-features = false }
burn = { version = "0.22", default-features = false, features = ["cuda"] }
```

```rust,ignore
use bdecide::{AutoModel, Device, LoadOptions};

// Select the application's CUDA adapter, e.g. device 1 rather than Burn's default.
let model = AutoModel::from_pretrained(
    LoadOptions::builder()
        .source(bdecide::hub::ModelSource::Local("checkpoint".into()))
        .device(Device::cuda(1))
        .build(),
)?;
```

With `device: None`, Burn selects among its compiled backends and honors
`BURN_DEVICE`. This is a feature-based default, not a hardware availability probe.
Enable `cpu` on bdecide to retain CPU fallback after a default device failure.
The consumer regression package enables Burn's Flex and NdArray backends while
bdecide's `cpu` and `wgpu` features stay disabled:
`cargo test --locked --manifest-path tests/downstream/Cargo.toml`.

| Execution | Cargo features  | Direct loader device                             |
| --------- | --------------- | ------------------------------------------------ |
| CPU       | `cpu` (default) | `burn::tensor::Device::flex()`                   |
| WGPU      | `wgpu`          | `burn::tensor::Device::wgpu(Default::default())` |
| Both      | `cpu,wgpu`      | Select either device explicitly                  |

WGPU temporarily disables autotune and fusion while GPU crashes and fusion
storage-buffer binding limits are investigated.

```rust,ignore
use bdecide::{Device, Qwen3_5ForCausalLM};
use camino::Utf8Path;

// Select CPU explicitly even when the binary also enables WGPU.
let device = Device::flex();
let model = Qwen3_5ForCausalLM::from_pretrained(Utf8Path::new("checkpoint"), &device)?;
```

Existing Transformers SafeTensors and PyTorch `.pt` checkpoints keep their weight
names and layouts. Loading still rejects missing, unexpected, malformed, and
non-finite weights before replacing the model.

LiquidAI decision models run natively in Rust/Burn through `AutoModel`,
`D1Model::from_pretrained` or `D1OmniModel::from_pretrained`. They return Choice, Score and Noul probabilities
without generating output tokens.

| Model                                                                 | Inputs                         | Native components                                         |
| --------------------------------------------------------------------- | ------------------------------ | --------------------------------------------------------- |
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)               | Text/JSON and images           | Causal LFM2, SigLIP2 NaFlex, selected token readout       |
| [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | Text/JSON with images or audio | Bidirectional LFM2, decision head, SigLIP2, FastConformer |

| Module            | Types                                                           | Responsibility                                                         |
| ----------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `models::lfm2`    | `Lfm2Config`, `Lfm2Model`                                       | LFM2 blocks and causal hidden states                                   |
| `models::lfm2_vl` | `Lfm2VlConfig`, `Lfm2VlModel`, `Lfm2VlForConditionalGeneration` | Vision features, image token replacement and tied vocabulary logits    |
| `models::d1`      | `D1Model`                                                       | LFM2-VL decision prompts and selected token readout                    |
| `models::d1`      | `D1OmniConfig`, `D1OmniModel`                                   | Bidirectional media-prefix encoder, audio and calibrated decision head |

| Operation                                                   | Burn 0.22 API                                   | Model-specific behavior                                                           |
| ----------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------- |
| LFM2 RMS normalization                                      | `nn::RmsNorm`                                   | PyTorch `weight` loads through Burn's adapter                                     |
| LFM2 short convolution                                      | Grouped `nn::conv::Conv1d`                      | Causal or centered padding; media never reads question text                       |
| SigLIP2 and LFM2-VL projector activations                   | Shared `HiddenActivation` utility               | Follow `hidden_act` and `projector_hidden_act`, including Transformers' tanh GELU |
| Fixed activations in LFM2, Omni's projector, audio and head | Burn activation functions                       | Preserve the original SiLU, GELU and ReLU operations                              |
| Conformer gating and normalization                          | `activation::glu`, `tensor::module::batch_norm` | Frozen pretrained statistics remain required, lazy checkpoint parameters          |
| Audio normalization                                         | `Tensor::var_mean`                              | Sample variance over valid frames                                                 |
| Audio spectrum and window                                   | `signal::stft`, `signal::hann_window`           | NeMo's explicit zero padding and Slaney filterbank                                |

RoPE retains Transformers' split-half channel pairing and frequency calculation;
Burn's `RotaryEncoding` pairs adjacent channels and computes frequencies differently.
SigLIP2's position resize retains antialiasing, which Burn's `Interpolate2d` does
not expose. Qwen's zero-centered RMS weights and Conformer relative-position
attention also retain their model-specific implementations.

For example, save this as `request.json` and run
`bdecide predict --model LiquidAI/d1-3B --input request.json`:

```json
{
  "state": "Inspect this picture.",
  "images": ["picture.png"],
  "questions": {
    "damage": { "type": "noul", "instructions": "Is the item damaged?" }
  }
}
```

For Omni speech, replace `images` with `"audio": ["speech.wav"]` and use
`--model LiquidAI/d1-omni-600M`. Media paths resolve against the process working
directory; files are decoded locally. `state: null` permits media-only requests.

| Field    | Accepted values                                                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `images` | Ordered PNG/JPEG/WebP paths or `{"width":256,"height":256,"pixels":[...]}` with interleaved RGB bytes                                              |
| `audio`  | An ordered array of Symphonia-supported file paths (including WAV, FLAC, MP3 and Ogg), `{"samples":[0.0],"sample_rate":16000}` or `{"pcm16":[0],"sample_rate":16000}` |

Omitted or empty `audio` means no audio. D1Omni accepts at most one clip per request.
Audio must be 16 kHz mono; float samples must be finite and in `[-1,1]`.
The official frontend pads clips shorter than 0.5 s and cuts them at 30 s.
Images and audio cannot be combined in one request. Other model families reject
media input. Omni text calibration is applied only to text-only requests, as
in the released implementation.

Context and Omni instruction/option truncation remain opt-in through
`"options":{"truncation":"truncate"}` or `--truncate`; usage records
lost tokens. Each question is evaluated independently with shared media
embeddings, and `input_tokens` counts the complete sequences actually read.
The upstream packed-tree and tensor batch optimizations for D1 are not implemented;
`system_one_batch` uses independent Burn streams for these models.

The small fixtures use Transformers weight names and compare text, multiple/tiled
images, PCM/WAV audio, calibration and truncation against pinned LiquidAI Python
code. Independent LFM2 hidden states and LFM2-VL logits are also compared with
Transformers. Published tensor names and shapes are checked separately. Omni JSON states
use the same compact input text in both implementations.
Regenerate them with
`uv run python tests/generate_d1_reference.py`, then run `cargo test --test d1`.
These checks do not measure the full released checkpoints' performance.

```rust,ignore
use burn::tensor::{Device, DeviceKind, wgpu::WgpuBackend};

// Keep software validation off the physical GPU, e.g. with Mesa lavapipe installed.
let device = Device::wgpu_options()
    .device_kind(DeviceKind::Cpu)
    .graphics_api(WgpuBackend::Vulkan)
    .init()?;
```

Clef's GPU activation parity cases can be run individually. Each uses a five-token
input and reports adapter initialization, model initialization, checkpoint loading,
and forward/readback stages. On a host with unresolved shutdowns, leave inference
cases unexecuted; the smoke test does not establish model inference stability.

```sh
cargo test --locked --no-default-features --features wgpu --lib models::clef::modeling_clef::activation_tests::wgpu_matches_python_activation_options::case_01_gelu -- --ignored --exact --nocapture --test-threads=1
```

CI also runs every adapter-dependent inference test, including long Qwen sequences,
against the pinned Python reference fixtures. Reproduce the Mesa software checks with:

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
export XDG_RUNTIME_DIR=/tmp/bdecide-wgpu-runtime
export LIBGL_ALWAYS_SOFTWARE=1
export CUBECL_WGPU_DEFAULT_DEVICE=Cpu
mkdir -p "$XDG_RUNTIME_DIR"
cargo test --locked --no-default-features --features wgpu --all-targets -- --ignored --nocapture --test-threads=1
```
