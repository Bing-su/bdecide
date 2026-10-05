"""Generate independent Clef fixtures with the released head and Transformers.

Run with transformers==5.18.0 and torch==2.14.1 (CPU):
    uv run python tests/generate_clef_reference.py
"""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

import torch
import transformers
from huggingface_hub import hf_hub_download
from reference_utils import (
    FIXTURES,
    FixtureRequest,
    Question,
    save_sharded_weights,
    write_json,
)
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

# Pin the released head and processor, e.g. regeneration cannot track moving main.
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"


def load_reference() -> ModuleType:
    """Load the released Clef implementation, e.g. use its independent span encoding."""
    source = hf_hub_download(
        "Cloudflare/clef-flash",
        "joint_schema_model.py",
        revision=REVISION,
        cache_dir=FIXTURES.parents[1] / ".cache" / "clef-reference",
    )
    spec = importlib.util.spec_from_file_location("clef_reference", source)
    if spec is None or spec.loader is None:
        message = f"Cannot load pinned Clef reference: {source}"
        raise ImportError(message)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)

    return reference


def create_tokenizer() -> PreTrainedTokenizerFast:
    """Keep deterministic word IDs and special tokens, e.g. <think> remains atomic."""
    vocabulary = [
        "[UNK]",
        "[PAD]",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "한글",
        "last",
        "z",
        "a",
        "other",
        "department",
        "urgency",
        "risk",
        "choice",
        "score",
        "noul",
        "true",
        "false",
        "Select",
        "Is",
        "this",
        "urgent",
        "cancel",
        "option_id",
        "description",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        ":",
        "?",
        ".",
        ",",
        "{",
        "}",
        "[",
        "]",
        '"',
        "FIELD",
        "ID",
        "TYPE",
        "INSTRUCTION",
        "ALLOWED",
        "OPTIONS",
        "OPTION",
        "END",
        "SCHEMA",
        "FIELDS",
        "system",
        "user",
        "assistant",
        "STATE",
        "JOINT",
        "DECISIONS",
        "The",
        "proposition",
        "is",
        "or",
        "the",
        "answer",
        "yes",
        "no",
        "a",
        "b",
        "e",
        "+",
        "-",
        "float",
        "nested",
        "zero",
    ]
    vocabulary = list(dict.fromkeys(vocabulary))
    tokenizer = Tokenizer(
        models.WordLevel(
            {word: i for i, word in enumerate(vocabulary)}, unk_token=vocabulary[0]
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<|im_start|>", "<|im_end|>", "<think>", "</think>"])
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token=vocabulary[1], unk_token=vocabulary[0]
    )


def reference_requests() -> list[FixtureRequest]:
    """Exercise ordered questions and truncation, e.g. nested Unicode state values."""
    questions: dict[str, Question] = {
        "department": {
            "type": "choice",
            "instructions": "Select",
            "criteria": {"z": "gamma", "a": "delta", "other": None},
        },
        "urgency": {
            "type": "score",
            "instructions": "Is this urgent?",
            "criteria": ["alpha", "beta", "gamma"],
        },
        "risk": {"type": "noul", "instructions": "cancel?"},
    }
    return [
        {"state": "alpha beta last", "questions": questions},
        {
            "state": {
                "z": "한글",
                "a": [True, {"z": 1, "a": 2}],
                "float": 0.000001,
                "zero": -0.0,
            },
            "questions": questions,
        },
        {
            "state": ["alpha " * 100, {"last": "cancel"}],
            "questions": questions,
            "options": {"truncation": "truncate", "max_len": 240},
        },
        {
            "state": "alpha",
            "questions": {
                "risk": {
                    "type": "noul",
                    "instructions": "cancel?",
                    "criteria": {"true": {"z": "beta", "a": "alpha"}, "false": None},
                }
            },
        },
        {
            "state": "alpha",
            "questions": {
                "only": {
                    "type": "choice",
                    "instructions": "Select",
                    "criteria": {"z": None},
                }
            },
        },
    ]


def generate_cases(
    reference: ModuleType,
    model: torch.nn.Module,
    fast: PreTrainedTokenizerFast,
    requests: list[FixtureRequest],
) -> list[dict]:
    """Run upstream span encoding and answer conversion, e.g. one logit row per question."""
    cases = []
    for request in requests:
        encoded = reference.encode_record(
            fast, request, max_length=request.get("options", {}).get("max_len", 512)
        )
        batch = reference.collate_records(
            [encoded], fast.pad_token_id, torch.device("cpu")
        )
        with torch.no_grad():
            logits = model(batch)[0]
        answers = {
            q.question_id: reference.systemone_answer(
                request["questions"][q.question_id],
                dict(zip(q.option_ids, row.softmax(-1).tolist())),
            )
            for q, row in zip(encoded.questions, logits)
        }
        cases.append(
            {
                "request": request,
                "input_ids": encoded.input_ids,
                "questions": [
                    {
                        "question_id": q.question_id,
                        "question_type": q.question_type,
                        "question_span": q.question_span,
                        "option_spans": q.option_spans,
                        "option_ids": q.option_ids,
                    }
                    for q in encoded.questions
                ],
                "logits": [row.tolist() for row in logits],
                "answers": answers,
            }
        )

    return cases


def generate_fixture(
    fixtures: Path,
    reference: ModuleType,
    fast: PreTrainedTokenizerFast,
    variant: str,
    hidden: int,
) -> None:
    """Generate one released layout, e.g. tiny-clef uses two backbone shards."""
    torch.manual_seed(732 + hidden)
    root = fixtures / variant
    root.mkdir(exist_ok=True)
    fast.save_pretrained(root)
    config = Qwen3_5TextConfig(
        vocab_size=len(fast),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=3,
        num_attention_heads=hidden // 8,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=hidden // 8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        max_position_embeddings=512,
        pad_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )
    config._attn_implementation = "eager"
    backbone = Qwen3_5ForCausalLM(config).eval()
    head_config = {
        "hidden_size": hidden,
        "width": 16,
        "routing_layers": 2,
        "layers": 2,
        "heads": 4,
        "feedforward": 32,
    }
    head = reference.JointSchemaHead(**head_config).eval()
    with torch.no_grad():
        head.prior_logit_scale.fill_(0.2)
        head.joint_logit_scale.fill_(0.4)
        head.residual_gate.fill_(-0.1)
    # Quantize to the release's storage precision before either runtime sees tensors.
    backbone.to(torch.bfloat16).float()
    head.to(torch.bfloat16).float()
    weights = {
        (
            "model.language_model." + key.removeprefix("model.")
            if key.startswith("model.")
            else key
        ): value.contiguous().bfloat16()
        for key, value in backbone.state_dict().items()
    }
    write_json(
        root / "config.json", {"model_type": "qwen3_5", "text_config": config.to_dict()}
    )
    write_json(root / "joint_head_config.json", head_config)
    save_file(
        {
            key: value.contiguous().bfloat16()
            for key, value in head.state_dict().items()
        },
        str(root / "joint_head.safetensors"),
    )
    if variant == "tiny-clef":
        # Split parameters to verify shard coverage, e.g. neither half can load alone.
        save_sharded_weights(root, weights)
    else:
        save_file(weights, str(root / "model.safetensors"))
    model = reference.ClefModel(backbone, head).eval()
    cases = generate_cases(reference, model, fast, reference_requests())
    write_json(
        root / "reference.json",
        {
            "reference": {
                "revision": REVISION,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            },
            "cases": cases,
        },
    )
    print(f"Generated {len(cases)} independent reference cases in {root}")


def main(fixtures: Path = FIXTURES) -> None:
    """Generate both Clef references on demand, e.g. imports never download upstream code."""
    torch.set_num_threads(2)
    reference = load_reference()
    fast = create_tokenizer()
    for variant, hidden in [
        ("tiny-clef-flash", 32),
        ("tiny-clef", 48),
    ]:
        generate_fixture(fixtures, reference, fast, variant, hidden)


if __name__ == "__main__":
    main()
