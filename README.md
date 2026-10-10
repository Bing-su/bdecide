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
