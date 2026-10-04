"""Generate independent Clef fixtures with the released head and Transformers.

Run with transformers==5.18.0 and torch==2.14.1 (CPU):
    uv run python tests/generate_clef_reference.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

# Pin the original head and processor so fixture regeneration cannot track moving main.
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
source = hf_hub_download(
    "Cloudflare/clef-flash",
    "joint_schema_model.py",
    revision=REVISION,
    cache_dir=Path(__file__).resolve().parents[1] / ".cache" / "clef-reference",
)
spec = importlib.util.spec_from_file_location("clef_reference", source)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
torch.set_num_threads(2)

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
fast = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer, pad_token=vocabulary[1], unk_token=vocabulary[0]
)

questions = {
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
requests = [
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

fixtures = Path(__file__).resolve().parent / "fixtures"
for variant, hidden, attention_heads, value_heads in [
    ("tiny-clef-flash", 32, 4, 4),
    ("tiny-clef", 48, 6, 6),
]:
    torch.manual_seed(732 + hidden)
    root = fixtures / variant
    root.mkdir(exist_ok=True)
    fast.save_pretrained(root)
    config = Qwen3_5TextConfig(
        vocab_size=len(fast),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=3,
        num_attention_heads=attention_heads,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=value_heads,
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
        attn_implementation="eager",
    )
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
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": config.to_dict()}, indent=2)
    )
    (root / "joint_head_config.json").write_text(json.dumps(head_config, indent=2))
    save_file(
        {
            key: value.contiguous().bfloat16()
            for key, value in head.state_dict().items()
        },
        str(root / "joint_head.safetensors"),
    )
    if variant == "tiny-clef":
        # Split parameters across shards to catch incomplete coverage and commit pinning.
        shards = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        index = {}
        for shard_index, shard in enumerate(shards):
            selected = {
                key: value
                for i, (key, value) in enumerate(weights.items())
                if i % 2 == shard_index
            }
            save_file(selected, str(root / shard))
            index.update(dict.fromkeys(selected, shard))
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index}, indent=2)
        )
    else:
        save_file(weights, str(root / "model.safetensors"))
    model = reference.ClefModel(backbone, head).eval()
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
    (root / "reference.json").write_text(
        json.dumps(
            {
                "reference": {
                    "revision": REVISION,
                    "transformers": transformers.__version__,
                    "torch": torch.__version__,
                },
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"Generated {len(cases)} independent reference cases in {root}")
