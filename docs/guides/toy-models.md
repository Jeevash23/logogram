# Check a method on a toy model

A method can only be trusted as far as it finds what is there. Logogram comes with two small
models whose circuits are known, because their weights were set by hand: learn the methods on a
model whose answer you know, or check that a method finds what was built. The
[README](../../README.md#toy-models) describes how the weights do it.

| Toy | What is planted | Task |
|---|---|---|
| `induction` | Two layers of four heads, attention only. L0 H2 is a previous-token head; L1 H1 an induction head, whose key reads L0 H2 | "… A B … F C … A" → " B"; the corrupt prompt swaps A and F, → " C" |
| `ioi` | Four layers of four heads, GPT-2 small's circuit with one head per step: L0 H2 previous-token, L1 H1 induction, L2 H3 S-inhibition, L3 H0 name mover | "Once Mary and John reached the store, John threw a drink to" → " Mary"; the flipped prompt → " John" |

## Write a toy's project

```bash
logogram toy                 # list the toys
logogram toy induction       # write its project, add its model, load it and check it
```

`logogram toy NAME [FOLDER]` writes a project (by default `toy-NAME` in the projects folder) and
ends by printing the commands that run one of its experiments or open it in the app. It holds:

- the model's files: a GPT-2 checkpoint in safetensors, which loads through the same code as any
  model;
- its datasets;
- its experiments, each with notes that say what it should find.

The model is added to this machine's local models as `toy-NAME`. Specs name it by the hash of its
files, which is the same on every machine. The command then loads the model and checks that it
solves every prompt and that each planted head attends where it was built to. Writing it again
updates the folder and keeps experiments you edited.

From Python, `lg.toy_model("induction")` does the same and returns the project, the loaded model,
its experiments' specs, the planted heads and the check. `toy.run("which-heads")` runs one of its
experiments.

## Run its experiments

In the app, open the project and load `toy-induction` (or `toy-ioi`) from the model dialog's local
models. The experiments are drafts in the history. Each one asks one question of the planted
circuit:

- which heads matter (`which-heads`);
- which of their inputs carries the difference (`which-input`);
- one composition by path patching (`previous-token-into-key`, `s-inhibition-into-query`,
  `induction-into-value`);
- whether the planted heads are the whole circuit (`is-the-circuit-all`);
- what gradients estimate (`gradient-estimates`, `integrated-gradients`);
- what each component writes directly (`direct-effects`);
- for the IOI toy, steering (`steer-the-slot`).

Run one, and read its result beside its notes.

## What each method should find

Effects are normalized: 1 restores (or breaks) all of the behavior, 0 none of it. Every head or
edge not named gives 0.

| Method | Induction toy | IOI toy |
|---|---|---|
| Head finder | L0 H2 previous-token (1.0), L1 H1 induction (1.0) | the same heads |
| Patching each head | L0 H2 and L1 H1, 1.0 each | L1 H1, L2 H3 and L3 H0, 1.0 each. L0 H2 gives 0, since it writes the same in both prompts |
| Patching queries, keys and values | L0 H2's value and L1 H1's key | L1 H1's query, L2 H3's value, L3 H0's query |
| Path patching | L0 H2 → L1 H1's key, 1.0 | L1 H1 → L2 H3's value, and L2 H3 → L3 H0's query, 1.0 each |
| Path patching each edge alone (the `edges` scope) | 1.0 for each planted edge along which the prompts differ (the embeddings → L0 H2's value, L0 H2 → L1 H1's key, L1 H1 → the logits), 0 for the others | 1.0 for the embeddings → L1 H1's query, L1 H1 → L2 H3's value, L2 H3 → L3 H0's query and L3 H0 → the logits, 0 for the others |
| Sets of sites | the planted heads alone: faithfulness 1.0 | the same |
| Direct logit attribution | L1 H1 writes all of the logit difference | L3 H0 writes all of it |
| Attribution patching, one gradient | L1 H1 1.0, exactly. L0 H2 gives 0, though patching gives 1.0 | L3 H0 1.0; the upstream heads stay near 0 |
| Integrated gradients | L0 H2: 0.47 with 8 steps, 0.92 with 16, 0.999 with 32 | the upstream heads stay near 0, even with 64 steps |
| Edge attribution, one gradient (EAP) | the edge L1 H1 → logits, 1.0 exactly; edges into queries and keys about 0 | the edge L3 H0 → logits, 1.0 exactly; edges into queries and keys about 0 |
| Edge attribution with 32 steps (EAP-IG) | L0 H2 → L1 H1's key, about 1.0 | |

Gradients miss what the planted heads do through their attention. Each planted head attends to
its target with all but 10⁻⁸ of its attention, so the gradient through its softmax is about
e⁻²⁰. A patch moves the attention all the way regardless.

This is the failure that [verifying estimates by patching](../concepts/attribution-patching.md)
and [by path patching](../concepts/edge-attribution.md) exists to catch. On these models you can
see it happen, and see integrated gradients recover it where the path between the prompts is
smooth enough.

The tests (`tests/test_toys.py`) check every number in this table on every change to Logogram.
