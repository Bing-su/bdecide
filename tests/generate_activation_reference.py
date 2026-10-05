"""Generate activation and backbone references with Transformers 5.18.0.

For example: .venv/bin/python tests/generate_activation_reference.py
Reuse the existing tiny checkpoint weights so activation selection is the only change.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import torch
import transformers
from reference_utils import FIXTURES, write_json
from safetensors.torch import load_file
from transformers import (
    ModernBertConfig,
    ModernBertModel,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
)
from transformers.activations import ACT2FN
from transformers.models.qwen3_5 import modeling_qwen3_5


def generate_cases(
    root: Path,
    inputs: torch.Tensor,
    input_ids: torch.Tensor,
) -> list[dict]:
    """Compare every parameterless activation, e.g. last-token states cover both attention types."""
    modern_config = json.loads(
        (root / "tiny-laya/encoder/config.json").read_text(encoding="utf-8")
    )
    modern_weights = {
        key.removeprefix("encoder."): value
        for key, value in load_file(root / "tiny-laya/model.safetensors").items()
        if key.startswith("encoder.")
    }
    qwen_config = json.loads(
        (root / "tiny-clef/config.json").read_text(encoding="utf-8")
    )["text_config"]
    index = json.loads(
        (root / "tiny-clef/model.safetensors.index.json").read_text(encoding="utf-8")
    )
    qwen_weights = {
        key.removeprefix("model.language_model."): value
        for shard in sorted(set(index["weight_map"].values()))
        for key, value in load_file(root / "tiny-clef" / shard).items()
        if key.startswith("model.language_model.")
    }
    cases = []
    with torch.no_grad():
        for name in ACT2FN:
            # Learned activation tensors are outside our scope, e.g. PReLU's scalar weight.
            if name in {"prelu", "xielu"}:
                continue
            act = ACT2FN[name]
            assert not list(act.parameters())
            modern = ModernBertModel(
                ModernBertConfig.from_dict(
                    {
                        **modern_config,
                        "hidden_activation": name,
                        "_attn_implementation": "eager",
                    }
                )
            ).eval()
            modern.load_state_dict(modern_weights, strict=True)
            qwen = Qwen3_5TextModel(
                Qwen3_5TextConfig.from_dict(
                    {**qwen_config, "hidden_act": name, "_attn_implementation": "eager"}
                )
            ).eval()
            qwen.load_state_dict(qwen_weights, strict=True)
            # Last-token states cover all layers and both Qwen attention types without a large fixture.
            cases.append(
                {
                    "name": name,
                    "values": act(inputs).tolist(),
                    "modernbert": modern(
                        input_ids, attention_mask=torch.ones_like(input_ids)
                    )
                    .last_hidden_state[0, -1]
                    .tolist(),
                    "qwen3_5": qwen(input_ids, use_cache=False)
                    .last_hidden_state[0, -1]
                    .tolist(),
                }
            )

    return cases


def main(root: Path = FIXTURES) -> None:
    """Reuse tiny checkpoint weights for activation parity, e.g. only hidden_act varies."""
    assert transformers.__version__ == "5.18.0", (
        "Regenerate against the pinned Transformers version"
    )
    torch.set_num_threads(2)
    # Force official Torch references even if optional fused packages are installed;
    # e.g. causal-conv1d's SiLU-only kernel cannot generate the ReLU reference.
    for function in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        setattr(
            modeling_qwen3_5,
            function,
            inspect.unwrap(getattr(modeling_qwen3_5, function)),
        )
    inputs = torch.tensor(
        [
            -100,
            -50,
            -25,
            -20,
            -10,
            -7,
            -6,
            -3.0001,
            -3,
            -2.9999,
            -2,
            -1,
            -0.1,
            -0.00001,
            0,
            0.00001,
            0.1,
            1,
            2,
            2.9999,
            3,
            3.0001,
            5.9999,
            6,
            6.0001,
            10,
            20,
            20.0001,
            25,
            50,
            100,
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[2, 5, 7, 9, 11]])
    cases = generate_cases(root, inputs, input_ids)
    write_json(
        root / "activation-reference.json",
        {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "inputs": inputs.tolist(),
            "input_ids": input_ids[0].tolist(),
            "cases": cases,
        },
    )
    print(f"Generated {len(cases)} parameterless activation and backbone references")


if __name__ == "__main__":
    main()
