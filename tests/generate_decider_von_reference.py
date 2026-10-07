"""Generate tiny Decider/Von checkpoints against pinned upstream implementations.

Run: .venv/bin/python tests/generate_decider_von_reference.py
The Von reference disables the server's chains and forced Noul band, preserving
the model's calibrated posterior (VON_NOUL_DECISION=raw, VON_CHAINS_DIR=off).
"""

from __future__ import annotations

import importlib
import json
import string
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

import torch
import transformers
from reference_utils import FIXTURES, FixtureRequest, Question, write_json
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import (
    ModernBertConfig,
    ModernBertModel,
    PreTrainedTokenizerFast,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

DECIDER_REVISION = "45024082b7d4bb667bf9140b7a7c073d3b483f6b"
VON_REVISION = "7d0ff642a989510b2148c151e7109a3b9750fca5"
VON_WEIGHTS_REVISION = "498ceba33390b32cfefaab6422ec380318ba9b99"
MAX_OPTIONS = 255


def fetch(url: str) -> bytes:
    """Read fixed public sources only, e.g. a pinned model calibration file."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def load_sources(root: Path) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    """Use upstream prompt, scoring and formatting logic, e.g. isolated levels."""
    sources = [
        (
            "decider",
            "Mapika/decider",
            DECIDER_REVISION,
            "decider",
            ["prompt", "model", "systemone"],
        ),
        (
            "von",
            "wfzyx/von",
            VON_REVISION,
            "src/von",
            [
                "types",
                "device",
                "models/option_marker",
                "backends/base",
                "backends/option_marker_backend",
            ],
        ),
    ]
    for package, repo, revision, prefix, files in sources:
        for name in files:
            target = root / package / f"{name}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            directory = target.parent
            while directory != root:
                (directory / "__init__.py").touch()
                directory = directory.parent
            target.write_bytes(
                fetch(
                    f"https://raw.githubusercontent.com/{repo}/{revision}/{prefix}/{name}.py"
                )
            )
    sys.path.insert(0, str(root))
    return (
        importlib.import_module("decider.prompt"),
        importlib.import_module("decider.systemone"),
        importlib.import_module("von.models.option_marker"),
        importlib.import_module("von.backends.option_marker_backend"),
    )


def tokenizer(*, von: bool) -> PreTrainedTokenizerFast:
    """Keep all wide labels as single tokens, e.g. the 255th Decider choice."""
    specials = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    letters = list(string.ascii_uppercase)
    letters += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    tok = Tokenizer(models.BPE(unk_token=specials[0]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(
        [
            *letters,
            "Context Question Options Answer Proposed answer Does the proposed answer fit Select urgent alpha beta gamma delta low medium high Yes condition holds true No condition is false",
            " ".join(string.digits),
        ],
        trainer=trainers.BpeTrainer(
            vocab_size=4096,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    if von:
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token=specials[0],
        pad_token=specials[1],
        cls_token=specials[2],
        sep_token=specials[3],
        mask_token=specials[4],
    )


def requests() -> list[FixtureRequest]:
    """Exercise described/plain noul, ordinal objects, markers and question isolation."""
    questions: dict[str, Question] = {
        "choice": {
            "type": "choice",
            "instructions": "Select",
            "criteria": {"beta": "beta", "alpha": "alpha", "gamma": None},
        },
        "score": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["0: low", "medium", "high"],
        },
        "plain": {"type": "noul", "instructions": "Is this urgent?"},
        "described": {
            "type": "noul",
            "instructions": "Cancel?",
            "criteria": {"true": "cancel", "false": "continue"},
        },
    }
    result: list[FixtureRequest] = [
        {"state": "alpha 2026 urgent", "questions": questions}
    ]
    result += [
        {
            "state": "alpha",
            "questions": {
                "wide": {
                    "type": "choice",
                    "instructions": "Select",
                    "criteria": {str(i): "alpha" for i in range(n)},
                }
            },
        }
        for n in [11, 27, 255]
    ]
    result.append(
        {
            "state": "alpha [MASK] [SEP] beta",
            "questions": {
                "marker": {
                    "type": "choice",
                    "instructions": "Select [MASK]",
                    "criteria": {"alpha": "[SEP] alpha", "beta": "beta"},
                }
            },
        }
    )
    result.append(
        {
            "state": "alpha",
            "questions": {
                "object": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": [
                        {"what": "low", "examples": ["alpha", "beta"]},
                        {"what": "high"},
                    ],
                }
            },
        }
    )
    return result


def decider(
    root: Path,
    prompt: ModuleType,
    protocol: ModuleType,
    tok: PreTrainedTokenizerFast,
    *,
    isolated: bool,
) -> None:
    """Use Transformers for the dense readout and upstream for every prompt row."""
    root.mkdir(exist_ok=True)
    tok.save_pretrained(root)
    config = Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=4096,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )
    model = Qwen3_5ForCausalLM(config).float().eval()
    model.save_pretrained(root)
    release = {
        "temperature": 1.145,
        "temperature_by_type": {"choice": 1.164, "noul": 1.624, "score": 1.124},
        "neutralize_none": False,
        "layout": "plain",
        "max_options": 255,
        "max_state_tokens": 32768,
        "isolated_levels": isolated,
    }
    write_json(root / "decider_config.json", release)
    labels = prompt.label_table(tok)[1]
    cases = []
    for request in requests():
        rendered = {
            key: protocol.render_question(spec)
            for key, spec in request["questions"].items()
        }
        rows, index = protocol.plan_rows(rendered, isolated)
        types = protocol.row_types(rendered, index)
        probabilities = []
        reference_rows = []
        context = protocol.render_state(request["state"])
        for row, kind in zip(rows, types):
            example = SimpleNamespace(
                context=context,
                qs=[
                    SimpleNamespace(
                        text=row["question"], options=row["options"], gold=0
                    )
                ],
            )
            built = prompt.build(
                example,
                tok,
                SimpleNamespace(shuffle=lambda _x: None),
                max_options=255,
                max_ctx_tokens=32768,
            )
            ids = built["ids"]
            answer_ids = labels[: len(row["options"])]
            with torch.no_grad():
                logits = (
                    model(torch.tensor([ids]), use_cache=False)
                    .logits[0, -1, answer_ids]
                    .float()
                )
                temperature = release["temperature_by_type"][kind]
                probability = torch.softmax(logits / temperature, -1).tolist()
            probabilities.append(probability)
            reference_rows.append(
                {
                    "input_ids": ids,
                    "answer_ids": answer_ids,
                    "logits": logits.tolist(),
                    "question": row["question"],
                    "options": row["options"],
                }
            )
        cases.append(
            {
                "request": request,
                "answers": protocol.assemble(rendered, index, probabilities),
                "rows": reference_rows,
            }
        )
    write_json(
        root / "reference.json",
        {
            "upstream_revision": DECIDER_REVISION,
            "transformers": transformers.__version__,
            "cases": cases,
        },
    )


def corrupt_weights(root: Path, model: torch.nn.Module) -> None:
    """Write native files that expose strict loading failures, e.g. a missing scorer."""
    for kind in ["missing", "unused", "nan", "shape"]:
        weights = dict(model.state_dict())
        key = "scorer.out_proj.weight"
        if kind == "missing":
            del weights[key]
        elif kind == "unused":
            weights["unexpected.weight"] = weights[key].clone()
        elif kind == "nan":
            weights[key] = torch.full_like(weights[key], float("nan"))
        else:
            weights[key] = torch.zeros(1, 9)
        torch.save(weights, root / f"corrupt-{kind}.pt")


def von(
    root: Path,
    upstream: ModuleType,
    backend_module: ModuleType,
    tok: PreTrainedTokenizerFast,
    *,
    independent: bool,
) -> None:
    """Store the exact torch.save layout, including encoder/scorer, and score upstream."""
    root.mkdir(exist_ok=True)
    tok.save_pretrained(root)
    config = ModernBertConfig(
        vocab_size=len(tok),
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=4096,
        local_attention=8,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        pad_token_id=1,
        bos_token_id=2,
        cls_token_id=2,
        eos_token_id=3,
        sep_token_id=3,
        norm_bias=False,
    )
    config.save_pretrained(root)
    model = upstream.OptionMarkerModel.__new__(upstream.OptionMarkerModel)
    torch.nn.Module.__init__(model)
    model.encoder = ModernBertModel(config).float().eval()
    model.scorer = upstream.OptionMarkerScorer(hidden_size=16).float().eval()
    model.tokenizer = tok
    model.mask_token_id = tok.mask_token_id
    model.digit_split = not independent
    model.eval()
    torch.save(model.state_dict(), root / "option_marker.pt")
    if independent:
        corrupt_weights(root, model)
    calibration = json.loads(
        fetch(
            f"https://huggingface.co/wfzyx/von/raw/{VON_WEIGHTS_REVISION}/marker_calibration.json"
        )
    )
    calibration["digit_split"] = model.digit_split
    calibration["independent_options"] = independent
    write_json(root / "marker_calibration.json", calibration)
    backend = backend_module.OptionMarkerBackend.__new__(
        backend_module.OptionMarkerBackend
    )
    backend._model = model
    backend._lock = threading.Lock()
    backend._default_temp = calibration["temperature"]
    backend._calib_map = calibration["calibration_map"]
    backend._noul_prior = calibration["noul_zero_shot_prior"]
    backend._independent_options = independent
    backend._trunc_local = threading.local()
    backend._capture_local = threading.local()
    backend.device = torch.device("cpu")
    backend.chain_runner = None
    backend.max_state_tokens = 4096
    backend.on_overflow = "refuse"
    backend.noul_decision = "raw"
    cases = []
    for request in requests():
        criteria = next(iter(request["questions"].values())).get("criteria", {})
        if isinstance(criteria, dict) and len(criteria) == MAX_OPTIONS:
            continue
        answer = backend.evaluate(request["state"], request["questions"]).model_dump()
        cases.append({"request": request, "answers": answer["answers"]})
    write_json(
        root / "reference.json",
        {
            "upstream_revision": VON_REVISION,
            "weights_revision": VON_WEIGHTS_REVISION,
            "transformers": transformers.__version__,
            "cases": cases,
        },
    )


def main() -> None:
    """Seed tiny fixtures deterministically, e.g. isolated/plain modes of both families."""
    torch.set_num_threads(1)
    torch.manual_seed(827)
    with tempfile.TemporaryDirectory() as directory:
        prompt, protocol, marker, backend = load_sources(Path(directory))
        for isolated in [True, False]:
            decider(
                FIXTURES / ("tiny-decider" if isolated else "tiny-decider-list"),
                prompt,
                protocol,
                tokenizer(von=False),
                isolated=isolated,
            )
        for independent in [True, False]:
            von(
                FIXTURES / ("tiny-von" if independent else "tiny-von-joint"),
                marker,
                backend,
                tokenizer(von=True),
                independent=independent,
            )


if __name__ == "__main__":
    main()
