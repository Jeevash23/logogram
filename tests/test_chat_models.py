"""Models made for chat, run on plain text: detected when a model loads, recorded in the model's
info and in each run's manifest. Prompts and results are unchanged."""

from __future__ import annotations

import pytest

from conftest import _build_tiny_model
from logogram.backends.hub import _select_files
from logogram.backends.transformer_lens import TransformerLensBackend, boot_local, chat_format
from logogram.runner import run_spec

QWEN_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}"
    "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.mark.parametrize(
    ("model_id", "template", "ends", "instruct"),
    [
        # Names that say so.
        ("google/gemma-3-270m-it", None, ["<eos>"], True),
        ("Qwen/Qwen2.5-0.5B-Instruct", QWEN_TEMPLATE, ["<|im_end|>"], True),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None, ["</s>"], True),
        ("allenai/OLMo-2-0425-1B-SFT", None, ["<|endoftext|>"], True),
        # A template whose turns end where the model stops, without the name saying so.
        ("Qwen/Qwen3-0.6B", QWEN_TEMPLATE, ["<|endoftext|>", "<|im_end|>"], True),
        # Base models, some with a chat template they stop before.
        ("Qwen/Qwen2.5-0.5B", QWEN_TEMPLATE, ["<|endoftext|>"], False),
        ("Qwen/Qwen3-0.6B-Base", QWEN_TEMPLATE, ["<|endoftext|>"], False),
        ("openai-community/gpt2", None, ["<|endoftext|>"], False),
        ("google/gemma-3-1b-pt", None, ["<eos>"], False),
        ("EleutherAI/pythia-160m", None, ["<|endoftext|>"], False),
        ("local/model-italian", None, [], False),
    ],
)
def test_models_made_for_chat_are_told_apart(model_id, template, ends, instruct):
    chat = chat_format(model_id, template, ends)
    assert chat["instruct"] is instruct
    assert chat["template"] is (template is not None)
    assert (chat["reason"] is not None) is instruct


def test_the_reason_says_why():
    assert "(it)" in chat_format("google/gemma-3-270m-it", None, [])["reason"]
    reason = chat_format("Qwen/Qwen3-0.6B", QWEN_TEMPLATE, ["<|im_end|>"])["reason"]
    assert "<|im_end|>" in reason and "chat template" in reason


def _boot(folder, model_id):
    return TransformerLensBackend.from_bridge(
        boot_local(folder, device="cpu", dtype="float32"),
        model_id=model_id,
        revision="test",
        dtype="float32",
        process_weights=True,
    )


def test_a_model_is_checked_for_chat_when_it_loads(tmp_path, tiny_backend):
    from transformers import AutoTokenizer

    assert tiny_backend.info.extra["chat"] == {
        "instruct": False,
        "template": False,
        "reason": None,
    }
    folder = tmp_path / "chat"
    folder.mkdir()
    _build_tiny_model(folder)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    # Each turn ends with the model's end-of-sequence token.
    tokenizer.chat_template = "{% for m in messages %}{{ m.content }}<|endoftext|>{% endfor %}"
    tokenizer.save_pretrained(folder)
    chat = _boot(folder, "local/tiny").info.extra["chat"]
    assert chat["instruct"] and chat["template"]
    assert "<|endoftext|>" in chat["reason"]
    # Without a template, its name alone can say so.
    plain = tmp_path / "plain"
    plain.mkdir()
    _build_tiny_model(plain)
    named = _boot(plain, "local/tiny-it").info.extra["chat"]
    assert named["instruct"] and not named["template"] and "(it)" in named["reason"]


def test_a_run_records_it_and_its_prompts_are_unchanged(tiny_backend, project, spec_factory):
    spec = spec_factory(
        dataset={"path": "datasets/ioi.jsonl", "limit": 3},
        scope={
            "kind": "sites",
            "sites": [{"kind": "head", "layer": 1, "head": 0, "position": {"kind": "last"}}],
        },
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert outcome.manifest["model"]["chat"] == tiny_backend.info.extra["chat"]


def test_chat_templates_are_downloaded_with_the_tokenizer():
    files = _select_files(
        [
            ("config.json", 1),
            ("tokenizer_config.json", 1),
            ("chat_template.jinja", 1),
            ("additional_chat_templates/tool_use.jinja", 1),
            ("model.safetensors", 1),
        ]
    )
    assert [name for name, _ in files] == [
        "chat_template.jinja",
        "config.json",
        "tokenizer_config.json",
        "model.safetensors",
    ]
