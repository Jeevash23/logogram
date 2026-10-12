"""Chat templates: prompts wrapped in the model's own template, as the person's message or as the
start of the model's reply, tokenized whole; named positions move with the template's text, the
answer is read after the whole wrapped text, and what was used is recorded. Templates that can't
be used are refused with what to do, and no template at all reads exactly as before."""

from __future__ import annotations

import json
import threading

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from conftest import TINY_CHAT_TEMPLATE, _build_tiny_model, give_chat_template
from logogram.analysis import baseline_report, tokenize_pair
from logogram.backends.base import BackendError
from logogram.backends.transformer_lens import TransformerLensBackend, boot_local
from logogram.chat import chat_wrapper, template_sha256
from logogram.datasets import PromptRecord, load_dataset, write_dataset
from logogram.engine import run_experiment
from logogram.heads import HeadKindSettings, head_kind_scores
from logogram.inside import look_report
from logogram.ioi import generate_ioi
from logogram.prompts import prepare_prompts, tokenize_prompt
from logogram.report import methods_text
from logogram.results import largest_change
from logogram.runner import configure_determinism, run_spec
from logogram.spec import AssistantPrefix, UserTurn

USER = UserTurn(kind="user_turn", system=None)
REPLY = AssistantPrefix(kind="assistant_prefix", system=None, user="Hello")
TEMPLATE_SHA = template_sha256(TINY_CHAT_TEMPLATE)


@pytest.fixture(scope="session")
def chat_backend(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny-chat")
    _build_tiny_model(folder)
    give_chat_template(folder)
    configure_determinism()
    bridge = boot_local(folder, device="cpu", dtype="float32")
    return TransformerLensBackend.from_bridge(
        bridge, model_id="tiny-chat", revision="test", dtype="float32", process_weights=True
    )


def _spec(spec_factory, template, **overrides):
    chat = None if template is None else template.model_dump(mode="json")
    return spec_factory(
        model={"id": "tiny-chat", "revision": "test", "device": "cpu"},
        tokenization={"prepend_bos": template is None, "chat_template": chat},
        **overrides,
    )


def test_the_template_is_the_one_the_tokenizer_ships(chat_backend, tiny_backend):
    assert chat_backend.chat_template == TINY_CHAT_TEMPLATE
    assert chat_backend.info.extra["chat_template_sha256"] == TEMPLATE_SHA
    assert chat_backend.info.extra["chat"]["template"] is True
    assert tiny_backend.chat_template is None
    assert tiny_backend.info.extra["chat_template_sha256"] is None


def test_a_prompt_is_the_persons_message_and_the_reply_is_opened(chat_backend):
    chat = chat_wrapper(chat_backend, USER)
    assert chat is not None
    # The template writes the beginning-of-sequence token itself.
    assert (chat.before, chat.after) == ("<|endoftext|> Hello ", ". world")
    prompt = "When Mary and John went to the store, John gave a drink to"
    text, start = chat.wrap(chat_backend, prompt)
    assert text == f"<|endoftext|> Hello {prompt}. world" and start == len(chat.before)
    tokens = tokenize_prompt(chat_backend, prompt, prepend_bos=False, chat=chat)
    # The whole text, tokenized at once and without added special tokens.
    whole = chat_backend.tokenize(text, prepend_bos=False)
    assert tokens.ids == whole.ids and tokens.start == start
    assert tokens.ids[0] == chat_backend.bos_token_id
    assert tokens.tokens[1:3] == [" Hello", " When"] and tokens.tokens[-2:] == [".", " world"]


def test_a_prompt_can_be_the_start_of_the_reply(chat_backend):
    chat = chat_wrapper(chat_backend, REPLY)
    assert chat is not None
    # Rendered so that the text ends exactly with the prompt: the model goes on from it.
    assert chat.before == "<|endoftext|> Hello Hello. world " and chat.after == ""
    text, start = chat.wrap(chat_backend, "When Mary and John went")
    assert text.endswith("When Mary and John went") and text[start:] == "When Mary and John went"


def test_named_positions_move_with_the_template(chat_backend):
    records = generate_ioi(6, seed=0)
    plain = prepare_prompts(chat_backend, records, prepend_bos=False)
    wrapped = prepare_prompts(chat_backend, records, False, chat=chat_wrapper(chat_backend, USER))
    for a, b in zip(plain, wrapped, strict=True):
        # The template's text before the prompt is "<|endoftext|> Hello ": two tokens.
        assert b.length == a.length + 4  # and ". world" after it
        assert set(b.labels) == set(a.labels) and b.labels
        for label, position in a.labels.items():
            assert b.labels[label] == position + 2
            assert b.clean.tokens[b.labels[label]] == a.clean.tokens[position]
        # Clean and corrupt line up on the wrapped tokens.
        assert len(b.clean.ids) == len(b.corrupt.ids)


def test_answers_are_read_after_the_whole_wrapped_text(chat_backend):
    records = generate_ioi(4, seed=1)
    report = baseline_report(
        chat_backend, records, prepend_bos=False, batch_size=4, chat_template=USER
    )
    chat = chat_wrapper(chat_backend, USER)
    assert chat is not None
    for row, record in zip(report["prompts"], records, strict=True):
        text, _ = chat.wrap(chat_backend, record.clean)
        ids = chat_backend.tokenize(text, prepend_bos=False).ids
        # The reply's first token: predicted at the end of the template's opening of the reply.
        logits = chat_backend.final_logits(torch.tensor([ids]))[0].double()
        logprobs = torch.log_softmax(logits, -1)
        answer = chat_backend.single_token_id(record.answer)
        distractor = chat_backend.single_token_id(record.distractor)
        expected = float(logprobs[answer] - logprobs[distractor])
        assert row["clean_logit_diff"] == pytest.approx(expected, abs=1e-5)


def test_the_token_strip_marks_the_prompt_in_the_template(chat_backend):
    record = generate_ioi(1, seed=2)[0]
    strip = tokenize_pair(chat_backend, record, False, chat_template=USER)
    first, end = strip["prompt"]
    assert strip["clean"]["tokens"][:first] == ["<|endoftext|>", " Hello"]
    assert strip["clean"]["tokens"][end:] == [".", " world"]
    assert strip["aligned"] and strip["issues"] == []
    assert tokenize_pair(chat_backend, record, False)["prompt"] is None


def test_a_model_without_a_chat_template_is_refused(tiny_backend, project, spec_factory):
    with pytest.raises(BackendError, match=r"has no chat template.*Set the chat template to none"):
        chat_wrapper(tiny_backend, USER)
    spec = spec_factory(
        tokenization={"prepend_bos": False, "chat_template": {"kind": "user_turn", "system": None}}
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "failed" and "no chat template" in outcome.manifest["error"]


def test_a_template_that_refuses_a_system_message_says_so(chat_backend, monkeypatch):
    refusing = (
        "{% if messages[0].role == 'system' %}{{ raise_exception('System role not supported') }}"
        "{% endif %}" + TINY_CHAT_TEMPLATE
    )
    monkeypatch.setattr(chat_backend, "_chat_template", refusing)
    with pytest.raises(BackendError, match=r"refuses a system message.*Leave the system message"):
        chat_wrapper(chat_backend, UserTurn(kind="user_turn", system="Hello"))
    assert chat_wrapper(chat_backend, USER) is not None  # without one, it renders


def test_a_system_message_comes_first(chat_backend):
    chat = chat_wrapper(chat_backend, UserTurn(kind="user_turn", system="world"))
    assert chat is not None and chat.before == "<|endoftext|>world, Hello "


def test_a_template_that_changes_the_prompt_leaves_it_unusable(chat_backend, monkeypatch):
    trimming = TINY_CHAT_TEMPLATE.replace("{{ m.content }}.", "{{ m.content | trim }}.")
    monkeypatch.setattr(chat_backend, "_chat_template", trimming)
    record = PromptRecord(
        clean="When Mary and John went ",
        corrupt="When Mary and John went ",
        answer=" to",
        distractor=" and",
    )
    strip = tokenize_pair(chat_backend, record, False, chat_template=USER)
    kinds = {i["kind"] for i in strip["issues"]}
    assert "chat_template" in kinds
    message = next(i["message"] for i in strip["issues"] if i["kind"] == "chat_template")
    assert "removes the spaces" in message and "Remove those spaces" in message


def test_the_start_of_a_reply_ends_the_text(chat_backend, monkeypatch):
    # A template that adds to the reply after the prompt: the model wouldn't go on from it.
    trailing = TINY_CHAT_TEMPLATE.replace("world {{ m.content }}.", "world {{ m.content }} world.")
    monkeypatch.setattr(chat_backend, "_chat_template", trailing)
    chat = chat_wrapper(chat_backend, REPLY)
    assert chat is not None and chat.after == ""  # continue_final_message cuts after the prompt


def test_templates_are_rendered_without_a_clock(chat_backend, monkeypatch):
    dated = (
        "{%- if strftime_now is defined %}{%- set day = strftime_now('%d %b %Y') %}"
        "{%- else %}{%- set day = '26 Jul 2024' %}{%- endif %}" + TINY_CHAT_TEMPLATE
    ).replace("{{ bos_token }}", "{{ bos_token }}{{ day }}")
    monkeypatch.setattr(chat_backend, "_chat_template", dated)
    chat = chat_wrapper(chat_backend, USER)
    assert chat is not None and chat.before.startswith("<|endoftext|>26 Jul 2024")
    undated = TINY_CHAT_TEMPLATE.replace("{{ bos_token }}", "{{ strftime_now('%Y') }}")
    monkeypatch.setattr(chat_backend, "_chat_template", undated)
    with pytest.raises(BackendError, match="writes today's date"):
        chat_wrapper(chat_backend, USER)


def test_special_tokens_are_the_models_own(chat_backend, monkeypatch):
    # Booting gives a tokenizer without a beginning-of-sequence token a substitute; the template
    # reads the model's own, which is none.
    monkeypatch.setattr(chat_backend, "_chat_tokens", {"eos_token": "<|endoftext|>"})
    chat = chat_wrapper(chat_backend, USER)
    assert chat is not None and chat.before == " Hello "


def test_a_run_wraps_its_prompts_and_records_the_template(chat_backend, project, spec_factory):
    spec = _spec(
        spec_factory,
        USER,
        dataset={"path": "datasets/ioi.jsonl", "limit": 6},
        scope={"kind": "layer_position", "site": "resid_pre", "positions": "labels"},
    )
    outcome = run_spec(spec, project, backend=chat_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert outcome.manifest["chat_template"] == {"kind": "user_turn", "sha256": TEMPLATE_SHA}
    info = outcome.summary["chat_template"]
    assert (info["before"], info["after"], info["sha256"]) == (
        "<|endoftext|> Hello ",
        ". world",
        TEMPLATE_SHA,
    )
    assert "in the chat template, as the person's message" in outcome.summary["description"]
    text = methods_text(spec, outcome.summary, outcome.manifest, outcome.run_id)
    assert f"chat template (SHA-256 {TEMPLATE_SHA}) as the person's message" in text
    assert 'read as "<|endoftext|> Hello " + the prompt + ". world"' in text
    assert "tokenized whole without adding special tokens" in text
    saved = json.loads((outcome.folder / "spec.json").read_text(encoding="utf-8"))
    assert saved["tokenization"] == {
        "prepend_bos": False,
        "chat_template": {"kind": "user_turn", "system": None},
    }


def test_a_run_that_stopped_early_says_how_its_prompts_were_wrapped(
    chat_backend, project, spec_factory
):
    spec = _spec(
        spec_factory,
        USER,
        dataset={"path": "datasets/ioi.jsonl", "limit": 6},
        scope={"kind": "heads", "position": {"kind": "all"}},
    )
    cancel = threading.Event()

    def on_event(kind: str, data: dict) -> None:
        if kind == "layer":
            cancel.set()

    outcome = run_spec(spec, project, backend=chat_backend, on_event=on_event, cancel=cancel)
    assert outcome.status == "cancelled"
    assert outcome.summary is not None and outcome.summary["partial"]["unit"] == "layers"
    assert outcome.summary["chat_template"]["sha256"] == TEMPLATE_SHA
    assert outcome.manifest["chat_template"] == {"kind": "user_turn", "sha256": TEMPLATE_SHA}


def test_the_same_spec_with_a_template_gives_identical_results(chat_backend, project, spec_factory):
    spec = _spec(spec_factory, REPLY, dataset={"path": "datasets/ioi.jsonl", "limit": 5})
    a = run_spec(spec, project, backend=chat_backend)
    b = run_spec(spec, project, backend=chat_backend)
    assert a.status == b.status == "finished"
    table_a = pq.read_table(a.folder / "results.parquet")
    table_b = pq.read_table(b.folder / "results.parquet")
    assert largest_change(table_a, table_b) == 0.0


def test_no_template_reads_exactly_as_before(chat_backend, project, spec_factory):
    """chat_template: null tokenizes each prompt as it is, and runs bit for bit as a run whose
    prompts were tokenized directly."""
    spec = _spec(spec_factory, None, dataset={"path": "datasets/ioi.jsonl", "limit": 6})
    outcome = run_spec(spec, project, backend=chat_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert outcome.manifest.get("chat_template") is None
    assert outcome.summary.get("chat_template") is None
    records = load_dataset(project.root / "datasets" / "ioi.jsonl")[:6]
    prompts = prepare_prompts(chat_backend, records, True)
    for p, record in zip(prompts, records, strict=True):
        direct = chat_backend.tokenize(record.clean, True)
        assert (p.clean.ids, p.clean.offsets, p.clean.start) == (direct.ids, direct.offsets, 0)
    result = run_experiment(spec, chat_backend, prompts)
    table = pq.read_table(outcome.folder / "results.parquet")
    patched = np.asarray(table.column("patched_metric").to_numpy())
    assert np.array_equal(patched, result.patched.ravel())


def test_a_template_that_adds_nothing_changes_nothing(
    chat_backend, project, spec_factory, monkeypatch
):
    spec_plain = _spec(spec_factory, None, dataset={"path": "datasets/ioi.jsonl", "limit": 5})
    spec_plain = spec_plain.model_copy(
        update={"tokenization": spec_plain.tokenization.model_copy(update={"prepend_bos": False})}
    )
    plain = run_spec(spec_plain, project, backend=chat_backend)
    monkeypatch.setattr(chat_backend, "_chat_template", "{{ messages[-1]['content'] }}")
    wrapped = run_spec(
        _spec(spec_factory, USER, dataset={"path": "datasets/ioi.jsonl", "limit": 5}),
        project,
        backend=chat_backend,
    )
    assert plain.status == wrapped.status == "finished"
    a = pq.read_table(plain.folder / "results.parquet")
    b = pq.read_table(wrapped.folder / "results.parquet")
    assert largest_change(a, b) == 0.0


def test_your_text_and_the_head_finder_read_the_template_too(chat_backend):
    report = look_report(
        chat_backend,
        "When Mary and John went",
        prepend_bos=False,
        top_k=3,
        position=None,
        chat_template=USER,
    )
    first, end = report["text_tokens"]
    assert report["tokens"][:first] == ["<|endoftext|>", " Hello"]
    assert report["tokens"][end:] == [".", " world"]
    assert report["chat_template"] == {"kind": "user_turn", "system": None}
    settings = HeadKindSettings(
        seed=0,
        sequences=4,
        length=5,
        prepend_bos=False,
        batch_size=4,
        chat_template=USER.model_dump(),
    )
    scores = head_kind_scores(chat_backend, settings)
    assert scores["settings"]["chat_template"] == {"kind": "user_turn", "system": None}
    from logogram.heads import repeated_random_tokens

    tokens = repeated_random_tokens(chat_backend, settings)
    before = chat_backend.tokenize("<|endoftext|> Hello ", prepend_bos=False).ids
    after = chat_backend.tokenize(". world", prepend_bos=False).ids
    assert tokens.shape[1] == len(before) + 10 + len(after)
    assert tokens[0, : len(before)].tolist() == before
    assert tokens[0, -len(after) :].tolist() == after


def test_a_template_needs_prepend_bos_off_in_analyses_too():
    from pydantic import ValidationError

    from logogram.server.app import AnalysisRequest

    with pytest.raises(ValidationError, match="template writes the beginning"):
        AnalysisRequest(prepend_bos=True, batch_size=4, chat_template=USER)


def test_prompts_of_a_dataset_read_through_the_server(app_client, chat_backend, tmp_path):
    app, client, headers = app_client
    app.state.logogram.backend = chat_backend
    app.state.logogram.model_status = {"state": "ready"}
    client.post("/api/projects", json={"name": "Chat", "parent": str(tmp_path)}, headers=headers)
    write_dataset(app.state.logogram.project.dataset_file("ioi.jsonl"), generate_ioi(4, seed=0))
    template = json.dumps({"kind": "user_turn", "system": None})
    detail = client.get(
        "/api/dataset",
        params={"path": "datasets/ioi.jsonl", "prepend_bos": "false", "chat_template": template},
        headers=headers,
    ).json()
    plain = client.get(
        "/api/dataset", params={"path": "datasets/ioi.jsonl"}, headers=headers
    ).json()
    assert detail["issues"] == [] and detail["lengths"] != plain["lengths"]
    body = {
        "dataset": "datasets/ioi.jsonl",
        "index": 0,
        "prepend_bos": False,
        "chat_template": {"kind": "user_turn", "system": None},
        "batch_size": 4,
        "metric": {"kind": "logit_diff", "normalization": "dataset_gap"},
    }
    strip = client.post("/api/tokenize", json=body, headers=headers).json()
    assert strip["prompt"][0] == 2 and strip["clean"]["tokens"][0] == "<|endoftext|>"
    wrapped = client.post(
        "/api/chat/wrap",
        json={"chat_template": {"kind": "user_turn", "system": None}, "text": "When Mary"},
        headers=headers,
    ).json()
    assert wrapped["text"] == "<|endoftext|> Hello When Mary. world"
    assert wrapped["text"][wrapped["start"] : wrapped["end"]] == "When Mary"
    assert wrapped["sha256"] == TEMPLATE_SHA and wrapped["problem"] is None
    look = client.post(
        "/api/look",
        json={
            "text": "When Mary",
            "top_k": 3,
            "prepend_bos": False,
            "batch_size": 1,
            "chat_template": {"kind": "assistant_prefix", "system": None, "user": "Hello"},
        },
        headers=headers,
    ).json()
    assert look["tokens"][-2:] == [" When", " Mary"] and look["text_tokens"] == [5, 7]


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="chat-token", port=8765), serve_web=False)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer chat-token", "origin": "http://127.0.0.1:8765"}
    return app, client, headers


def test_continuing_your_text_reads_it_in_the_template(chat_backend):
    """Continuing the text goes on after the wrapped text, as the model's reply does."""
    from logogram.inside import continue_text

    plain = continue_text(chat_backend, "When Mary and John went", prepend_bos=False, tokens=2)
    wrapped = continue_text(
        chat_backend, "When Mary and John went", prepend_bos=False, tokens=2, chat_template=USER
    )
    assert wrapped["tokens"][:2] == ["<|endoftext|>", " Hello"]
    assert wrapped["tokens"][-2:] == [".", " world"]
    assert len(wrapped["tokens"]) == len(plain["tokens"]) + 4
