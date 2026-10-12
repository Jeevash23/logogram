# Run a chat model on its own template

Models made for chat, such as Qwen 2.5 0.5B Instruct or Gemma 3 270M IT, are trained on
conversations wrapped in their chat template: special tokens that mark who speaks, and where the
model's reply begins. Fed plain text, such a model is out of its element, and can behave
differently than in a chat. Logogram can read every prompt in the model's own template instead.

## When Logogram says so

When a model loads, Logogram checks whether it is made for chat: its name says so (`-it`,
`-Instruct`, `-Chat`, ...), or it stops generating at the token that ends each turn in its chat
template (as Qwen3 0.6B does). A chat template alone isn't enough, since some base models ship one
(Qwen 2.5 0.5B does). Until a template is chosen, the model dialog, the summary beside **Run** and
the baseline say that the prompts run as plain text, and each run's manifest records it.

## 1. Choose how prompts sit in the template

Load the model, then in **Experiment → Configure**, under **Data and model**, choose the **Chat
template**:

* **The person's message** (`user_turn`): each prompt is the person's message, after the system
  message if there is one, and the template then opens the model's reply. The answer is the first
  token of the reply, which in most templates has no leading space: write the answers that way
  (`"Mary"`, not `" Mary"`). Use it for questions.
* **The start of the reply** (`assistant_prefix`): a fixed message from the person (such as
  *Continue the sentence.*), then each prompt as the beginning of the model's reply, so the text
  ends exactly with the prompt and the model goes on from it. Answers keep their leading spaces.
  Use it for completions such as IOI's.

A **system message** is optional; left empty, there is none, and some templates then write a
default of their own. The form shows the first prompt exactly as the model will read it, with the
prompt underlined, and the token strip greys the template's tokens.

With Qwen 2.5's template and no system message, a prompt as the person's message reads:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
When Mary and John went to the store, John gave a drink to<|im_end|>
<|im_start|>assistant
```

## 2. Check the baseline

The baseline, the token strip, attention, layer predictions, the SAE views, the head finder and
**Your text** all wrap prompts the same way. **Check the baseline** in the template before running
anything: if the model doesn't prefer the answer there either, the task needs rewording for it.

## 3. Run

The wrapped text is tokenized whole, without adding special tokens: the template writes the
beginning-of-sequence token where the model uses one, so `prepend_bos` is `false`. The template's
text before and after a prompt is the same for every prompt, so named positions move by the length
of the text before it, clean and corrupt prompts still line up token by token, and the answer is
read after the whole wrapped text.

```json
"tokenization": {
  "prepend_bos": false,
  "chat_template": {"kind": "assistant_prefix", "system": null, "user": "Continue the sentence."}
}
```

The template is the one the tokenizer ships at the model's revision, rendered as Hugging Face's
`apply_chat_template` renders it, and each run's manifest records its SHA-256; the summary records
the template's text around each prompt, and the methods paragraph says both.

## What Logogram refuses, and why

* A template that changes a prompt, such as one that trims spaces at its start or end (Llama 3's
  does), leaves that prompt unusable, with the reason, rather than guessing where its tokens are.
* A model without a chat template, or a template that refuses a system message (Gemma's), is
  refused with what to do instead.
* Templates are rendered without a clock, so the same spec reads the same text on any day. A
  template that writes today's date uses its own fallback date (Llama 3.2's is 26 Jul 2024); one
  without a fallback is refused.

## From Python

```python
chat = lg.load_model("Qwen/Qwen2.5-0.5B-Instruct")
lg.look(chat, "The capital of France is", prepend_bos=False,
        chat_template={"kind": "user_turn", "system": None})
```

`lg.head_kinds` and `lg.feature_writes` take the same `chat_template`, and a spec's
`tokenization.chat_template` does the rest (see [the spec reference](../reference/spec.md#chat-template)).
