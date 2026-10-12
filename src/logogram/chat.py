"""Prompts as instruct models read them: wrapped in the model's own chat template.

A spec's ``tokenization.chat_template`` wraps every prompt, clean and corrupt alike, in the chat
template the model's tokenizer ships at the model's revision (never one of Logogram's), rendered
as text the way Hugging Face's ``apply_chat_template`` renders it:

* ``user_turn``: the prompt is the person's message (after the system message, if there is one),
  followed by the template's generation prompt, which opens the model's reply. The answer is the
  first token of the reply.
* ``assistant_prefix``: a fixed message from the person, then the prompt as the beginning of the
  model's reply, rendered as ``continue_final_message`` renders it: the text ends exactly with the
  prompt, and the model goes on with its reply.

The wrapped text is tokenized whole, without adding special tokens: the template writes the
beginning-of-sequence token where the model uses one. Every prompt is wrapped alike. The template's
text before and after a prompt is found once, by rendering a placeholder in the prompt's place;
each prompt's own rendering must then be exactly that text around the prompt, so a dataset's
character spans (its named positions) move by the length of the text before it. A template that
changes a prompt (trims its spaces, say) leaves that prompt unusable, with the reason, rather than
a guess at where its tokens are.

Templates are rendered without a clock: one that would write today's date uses its own fallback
date, and one that has none is refused, so a spec reads the same text on any day.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from logogram.backends.base import BackendError, ModelBackend
from logogram.spec import AssistantPrefix, UserTurn

# Stands in for the prompt to find the template's text around it: letters and digits only, so a
# template that trims or strips a message leaves it whole.
PLACEHOLDER = "Logogram0prompt0placeholder7f3a"


def template_sha256(template: str) -> str:
    """A chat template's SHA-256: of its source text, encoded as UTF-8."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


class TemplateFailed(Exception):
    """The template raised while rendering (its own raise_exception, or a variable it lacks)."""


@dataclass(frozen=True)
class ChatWrapper:
    """How one chat template wraps every prompt: the text before and after it."""

    kind: str
    system: str | None
    user: str | None
    before: str
    after: str
    # The SHA-256 of the template's source (backends.transformer_lens.template_sha256).
    sha256: str

    def wrap(self, backend: ModelBackend, prompt: str) -> tuple[str, int]:
        """The prompt in the template, and where it begins in that text. Raises
        :class:`WrapError` when the template doesn't keep the prompt as it is."""
        text = _render(backend, self.kind, self.system, self.user, prompt)
        if text != self.before + prompt + self.after:
            raise WrapError(self._mismatch(prompt, text), text)
        return text, len(self.before)

    def _mismatch(self, prompt: str, text: str) -> str:
        stripped = prompt.strip()
        if stripped != prompt and text == self.before + stripped + self.after:
            return (
                "the chat template removes the spaces at the start or end of this prompt, so the "
                "text it reads isn't the prompt as written. Remove those spaces from the prompt."
            )
        if self.kind == "assistant_prefix" and not text.endswith(prompt):
            return (
                "rendered as the start of the model's reply, the text doesn't end with this "
                "prompt, so the model wouldn't go on from it. Edit the prompt, or wrap it as the "
                "person's message."
            )
        return (
            "the chat template changes this prompt's text, so where its tokens are can't be "
            "told. Edit the prompt, or run without the chat template."
        )

    def to_dict(self) -> dict[str, Any]:
        """What a run's summary records, for the methods text to say."""
        return {
            "kind": self.kind,
            "system": self.system,
            "user": self.user,
            "sha256": self.sha256,
            "before": self.before,
            "after": self.after,
        }


class WrapError(ValueError):
    """A prompt the chat template doesn't keep as it is. ``text`` is what the template made."""

    def __init__(self, message: str, text: str) -> None:
        super().__init__(message)
        self.text = text


def chat_wrapper(
    backend: ModelBackend, template: UserTurn | AssistantPrefix | None
) -> ChatWrapper | None:
    """How ``template`` (a spec's tokenization.chat_template) wraps prompts for this model, or
    None for none. Refuses a model without a chat template, and a template that can't render the
    conversation (one that refuses a system message, say), saying what to do."""
    if template is None:
        return None
    name = backend.info.id
    source = backend.chat_template
    if source is None:
        raise BackendError(
            f"{name} has no chat template: its tokenizer doesn't ship one, so its prompts can "
            "only be read as plain text. Set the chat template to none (tokenization."
            "chat_template: null), or load a model made for chat, which ships one."
        )
    user = template.user if isinstance(template, AssistantPrefix) else None
    try:
        text = _render(backend, template.kind, template.system, user, PLACEHOLDER)
    except TemplateFailed as exc:
        if template.system is not None and _renders(backend, template.kind, user):
            raise BackendError(
                f"{name}'s chat template refuses a system message ({exc}). Leave the system "
                "message empty (tokenization.chat_template.system: null)."
            ) from exc
        raise BackendError(f"{name}'s chat template can't render this conversation: {exc}") from exc
    where = "the person's message" if template.kind == "user_turn" else "the start of the reply"
    if text.count(PLACEHOLDER) != 1:
        raise BackendError(
            f"{name}'s chat template doesn't write {where} once, as it was given, so where a "
            "prompt's tokens are can't be told. Run without the chat template."
        )
    before, after = text.split(PLACEHOLDER)
    if template.kind == "assistant_prefix" and after:
        raise BackendError(
            f"{name}'s chat template doesn't end the text with the start of the model's reply, "
            "so the model wouldn't go on from the prompt. Wrap prompts as the person's message "
            "instead."
        )
    return ChatWrapper(
        kind=template.kind,
        system=template.system,
        user=user,
        before=before,
        after=after,
        sha256=template_sha256(source),
    )


def _messages(
    kind: str, system: str | None, user: str | None, content: str
) -> list[dict[str, str]]:
    messages = [] if system is None else [{"role": "system", "content": system}]
    if kind == "user_turn":
        return [*messages, {"role": "user", "content": content}]
    return [
        *messages,
        {"role": "user", "content": user or ""},
        {"role": "assistant", "content": content},
    ]


def _render(
    backend: ModelBackend, kind: str, system: str | None, user: str | None, content: str
) -> str:
    import jinja2

    try:
        return backend.render_chat(
            _messages(kind, system, user, content),
            add_generation_prompt=kind == "user_turn",
            continue_final_message=kind == "assistant_prefix",
        )
    except BackendError:
        raise
    except jinja2.exceptions.UndefinedError as exc:
        if "strftime_now" in str(exc):
            raise BackendError(
                f"{backend.info.id}'s chat template writes today's date into every prompt, and "
                "has no date to use without a clock, so runs on different days would read "
                "different text. Run without the chat template."
            ) from exc
        raise TemplateFailed(str(exc)) from exc
    except (jinja2.exceptions.TemplateError, ValueError, TypeError) as exc:
        raise TemplateFailed(str(exc) or type(exc).__name__) from exc


def _renders(backend: ModelBackend, kind: str, user: str | None) -> bool:
    """Whether the template renders the conversation without a system message."""
    try:
        _render(backend, kind, None, user, PLACEHOLDER)
    except (TemplateFailed, BackendError):
        return False
    return True
