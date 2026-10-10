# bdecide

Uses Burn 0.22 and requires Rust 1.95 or newer. See the upstream
[release notes](https://github.com/tracel-ai/burn/releases/tag/v0.22.0) and
[migration guide](https://burn.dev/books/burn/migrating-to-0.22.html).

Models and tensors no longer take a backend type parameter. Pass a Burn device
to direct model loaders; `AutoModel` and the CLI retain their `cpu`, `wgpu`, and
`auto` selections.

| Execution | Cargo features  | Direct loader device                             |
| --------- | --------------- | ------------------------------------------------ |
| CPU       | `cpu` (default) | `burn::tensor::Device::flex()`                   |
| WGPU      | `wgpu`          | `burn::tensor::Device::wgpu(Default::default())` |
| Both      | `cpu,wgpu`      | Select either device explicitly                  |

WGPU temporarily disables autotune and fusion while GPU crashes and fusion
storage-buffer binding limits are investigated.

```rust,ignore
use bdecide::Qwen3_5ForCausalLM;
use burn::tensor::Device;
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

For Omni speech, replace `images` with `"audio": "speech.wav"` and use
`--model LiquidAI/d1-omni-600M`. Media paths resolve against the process working
directory; files are decoded locally. `state: null` permits media-only requests.

| Field    | Accepted values                                                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `images` | Ordered PNG/JPEG/WebP paths or `{"width":256,"height":256,"pixels":[...]}` with interleaved RGB bytes                                              |
| `audio`  | A Symphonia-supported file path (including WAV, FLAC, MP3 and Ogg), `{"samples":[0.0],"sample_rate":16000}` or `{"pcm16":[0],"sample_rate":16000}` |

Audio must be 16 kHz mono; float samples must be finite and in `[-1,1]`.
The official frontend pads clips shorter than 0.5 s and cuts them at 30 s.
Images and audio cannot be combined in one request. Other model families reject
media input. Omni text calibration is applied only to text-only requests, as
in the released implementation.

Context and Omni instruction/option truncation remain opt-in through
`"options":{"truncation":"truncate"}` or `--truncate`; usage records
lost tokens. Each question is evaluated independently with shared media
embeddings, and `input_tokens` counts the complete sequences actually read.
The upstream packed-tree and batch optimizations are not implemented.

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
