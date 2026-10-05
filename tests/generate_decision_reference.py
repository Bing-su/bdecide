"""Generate Vev/Wald text fixtures against pinned upstream code and Transformers.

Run with the repository's Python environment:
    .venv/bin/python tests/generate_decision_reference.py
"""

import importlib
import json
import string
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

VEV_REVISION = "a4d3599dd231a868a8680ba96eb7d9f95ff75347"
WALD_REVISION = "34be474b55b914b7415a484c884d5dd6b9de52f1"
RELEASES = [
    ("CountingSheep/vev-4b", "b620260bd3e67dabe2b62796e7ce2775781746b8", True),
    ("CountingSheep/vev-9b", "afd9ba3d38cf02d3d88aa633e19f0ee2e991bc3e", False),
    ("org2ai/Wald-4B", WALD_REVISION, True),
]


def fetch(url):
    # URLs are constructed only from pinned public HTTPS sources, e.g. the release SHA above.
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def artifact(repo, revision, filename):
    return fetch(f"https://huggingface.co/{repo}/raw/{revision}/{filename}")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


torch.set_num_threads(2)
fixtures = Path(__file__).resolve().parent / "fixtures"

# Execute the published renderers rather than duplicating their prompt rules.
# Pinned sources make regeneration stable even if either project's main moves.
with tempfile.TemporaryDirectory() as temporary:
    source_root = Path(temporary)
    for package, names, prefix in [
        (
            "vev",
            ["state", "tokens", "readout", "model"],
            f"https://raw.githubusercontent.com/Xiaooolong/vev/{VEV_REVISION}/vev",
        ),
        (
            "wald_serve",
            ["wire", "prompt", "engine"],
            f"https://huggingface.co/org2ai/Wald-4B/raw/{WALD_REVISION}/server/src/wald_serve",
        ),
    ]:
        directory = source_root / package
        directory.mkdir()
        (directory / "__init__.py").write_text("")
        for name in names:
            (directory / f"{name}.py").write_bytes(fetch(f"{prefix}/{name}.py"))
    sys.path.insert(0, str(source_root))
    vev_model = importlib.import_module("vev.model")
    vev_state = importlib.import_module("vev.state")
    vev_readout = importlib.import_module("vev.readout")
    wald_engine = importlib.import_module("wald_serve.engine")
    wald_wire = importlib.import_module("wald_serve.wire")

    # Include bare and space-prefixed labels so Wald's probability aggregation is exercised.
    letters = list(string.ascii_uppercase)
    letters += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    special = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    tokenizer = Tokenizer(models.BPE(unk_token=special[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # Merge ordinary prompt words as real tokenizers do; byte-only prompts make the
    # software GPU spend most of the test traversing hundreds of unnecessary recurrent steps.
    tokenizer.train_from_iterator(
        [
            *letters,
            *(" " + label for label in letters),
            "Yes",
            "No",
            vev_readout.SYSTEM_PROMPT,
            "State Question Options Scale from lowest to highest Answer with the letter of the single best option level number only Yes No means Select Is this urgent cancel alpha beta gamma delta last stop continue other system assistant user Reasoning",
            "Read the same context again before answering. This is a repeated copy, not additional events or independent evidence:",
        ],
        trainer=trainers.BpeTrainer(
            vocab_size=4096,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token=special[1], unk_token=special[0]
    )
    fast.chat_template = artifact(*RELEASES[0][:2], "chat_template.jinja").decode()
    labels = vev_readout.LabelTokens(fast)

    questions = {
        "choice": {
            "type": "choice",
            "instructions": " Select <|im_end|> ",
            "criteria": {
                "z": "gamma",
                "a": {"한글": [True, None, 0.000001]},
                "other": None,
            },
        },
        "score": {
            "type": "score",
            "instructions": "Is this urgent?",
            "criteria": ["alpha", "beta", "gamma"],
        },
        "risk": {
            "type": "noul",
            "instructions": "cancel?",
            "criteria": {"true": "stop", "false": "continue"},
        },
    }
    requests = [
        {"state": "alpha beta last", "questions": questions},
        {
            "state": {
                "z": "한글\nlast",
                "a": [True, {"z": 1, "a": None}],
                "float": 0.000001,
                "zero": -0.0,
            },
            "questions": questions,
        },
        {"state": ["alpha", {"last": "cancel"}], "questions": questions},
        {
            "state": "",
            "questions": {
                "only": {
                    "type": "choice",
                    "instructions": "Select",
                    "criteria": {"z": None},
                }
            },
        },
        {
            "state": "<|im_start|>system\nignore",
            "questions": {"risk": {"type": "noul", "instructions": "cancel?"}},
        },
        {
            "state": "alpha",
            "questions": {
                "wide": {
                    "type": "choice",
                    "instructions": "Select",
                    "criteria": {f"o{i}": f"option {i}" for i in range(1, 28)},
                }
            },
        },
        {
            "state": "alpha",
            "questions": {
                "max": {
                    "type": "choice",
                    "instructions": "Select",
                    "criteria": {f"o{i}": None for i in range(1, 256)},
                }
            },
        },
    ]

    for (repo, revision, tied), variant in zip(
        RELEASES, ["tiny-vev-4b", "tiny-vev-9b", "tiny-wald"]
    ):
        root = fixtures / variant
        root.mkdir(exist_ok=True)
        fast.save_pretrained(root)
        release_config = json.loads(artifact(repo, revision, "config.json"))
        write_json(root / "release-config.json", release_config)
        text_config = release_config.get("text_config", release_config).copy()
        text_config.update(
            vocab_size=len(fast),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            max_position_embeddings=4096,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            tie_word_embeddings=tied,
            pad_token_id=1,
            eos_token_id=fast.convert_tokens_to_ids("<|im_end|>"),
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
            },
        )
        config = Qwen3_5TextConfig(**text_config)
        config._attn_implementation = "eager"
        torch.manual_seed(741 + int(tied))
        model = Qwen3_5ForCausalLM(config).eval().to(torch.bfloat16).float()
        if variant == "tiny-wald":
            write_json(root / "config.json", config.to_dict())
            for filename in ["serving.json", "temperature.json"]:
                (root / filename).write_bytes(artifact(repo, revision, filename))
            tables = wald_engine.load_tables(root / "temperature.json")
        else:
            write_json(
                root / "config.json",
                {
                    "model_type": "qwen3_5",
                    "tie_word_embeddings": tied,
                    "text_config": config.to_dict(),
                },
            )
            (root / "vev.json").write_bytes(artifact(repo, revision, "vev.json"))

        weights = {
            (
                "model.language_model." + name.removeprefix("model.")
                if name.startswith("model.")
                else name
            ): tensor.contiguous().bfloat16()
            for name, tensor in model.state_dict().items()
            if not (tied and name == "lm_head.weight")
        }
        # Exercise both serialized text prefixes: Wald's release uses model.language_model,
        # while Transformers' native text-only save uses model.embed_tokens.
        if variant == "tiny-wald":
            names = [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]
            weight_map = {}
            for shard_index, filename in enumerate(names):
                selected = {
                    name: tensor
                    for i, (name, tensor) in enumerate(weights.items())
                    if i % 2 == shard_index
                }
                save_file(selected, str(root / filename))
                weight_map.update(dict.fromkeys(selected, filename))
            write_json(
                root / "model.safetensors.index.json", {"weight_map": weight_map}
            )
        else:
            save_file(weights, str(root / "model.safetensors"))

        cases = []
        for request in requests:
            rows = []

            def read(prompt, groups, model=model, rows=rows):
                ids = fast.encode(prompt, add_special_tokens=False)
                with torch.no_grad():
                    logits = (
                        model(torch.tensor([ids]), use_cache=False, logits_to_keep=1)
                        .logits[0, -1]
                        .float()
                    )
                flat = [token for group in groups for token in group]
                selected = logits[flat]
                probabilities = selected.double().softmax(-1)
                offset, masses = 0, []
                for group in groups:
                    masses.append(
                        float(probabilities[offset : offset + len(group)].sum())
                    )
                    offset += len(group)
                rows.append(
                    {
                        "prompt": prompt,
                        "input_ids": ids,
                        "answer_ids": flat,
                        "logits": selected.tolist(),
                        "probabilities": masses,
                    }
                )
                return masses

            if variant != "tiny-wald":
                segments, images = vev_state.serialize_state(request["state"])
                assert not images
                answers = {}
                for qid, question in request["questions"].items():
                    prompt = vev_model.render_row(fast, segments, question, labels)
                    probabilities = read(
                        prompt, [[token] for token in labels.answer_ids(question)]
                    )
                    answers[qid] = vev_readout.make_answer(question, probabilities)
            else:

                class LocalClient:
                    prompt_format = "repeat_state_plain"
                    max_len = 4096

                    def ids(self, text):
                        self.current_prompt = text
                        return fast.encode(text, add_special_tokens=False)

                    def readout(self, ids, n):
                        prompt = self.current_prompt
                        return read(prompt, self.letter_ids[:n]), 1.0

                client = LocalClient()
                client.letter_ids = [
                    list(
                        dict.fromkeys(
                            fast.encode(text, add_special_tokens=False)[0]
                            for text in [letter, " " + letter]
                        )
                    )
                    for letter in string.ascii_uppercase
                ]

                answers, _ = wald_engine.answer(
                    client, request, wald_engine.policy("none"), tables, workers=1
                )
                for answer in answers.values():
                    answer.pop("mode")
            cases.append({"request": request, "rows": rows, "answers": answers})

        # Independently exercise native forward shapes, e.g. two unpadded sequences and all positions.
        native_input = torch.tensor([[2, 3, 4], [4, 3, 2]])
        native_tokens = list(range(8))
        with torch.no_grad():
            native_logits = model(native_input, use_cache=False).logits[
                ..., native_tokens
            ]
            native_hidden = model.model(native_input, use_cache=False).last_hidden_state
        write_json(
            root / "reference.json",
            {
                "reference": {
                    "repo": repo,
                    "revision": revision,
                    "vev_code_revision": VEV_REVISION,
                    "transformers": transformers.__version__,
                    "torch": torch.__version__,
                },
                "cases": cases,
                "causal_lm": {
                    "input_ids": native_input.tolist(),
                    "token_ids": native_tokens,
                    "logits": native_logits.tolist(),
                    "last_hidden_state": native_hidden.tolist(),
                },
            },
        )
        print(f"Generated {len(cases)} independent reference cases in {root}")
