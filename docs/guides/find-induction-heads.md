# Find induction heads

An **induction head** completes patterns: having seen *A B* earlier, at a later *A* it attends to
the earlier *B*, and helps predict *B* again (Elhage et al., 2021; Olsson et al., 2022). Induction
heads are built from two simpler kinds: **previous-token heads**, which attend to the token just
before, and **duplicate-token heads**, which attend to the earlier copy of the current token. The
head finder in the Attention view scores every head of the loaded model as each kind.

## Score every head

1. Load the model (GPT-2 small, say) and open **Explore → Attention**.
2. Under **Find heads by what they attend to**, set the **Sequences**, the **Tokens in each copy**
   and the **Seed**, and press **Find heads**.
3. Choose the **Kind of head**: previous-token, duplicate-token or induction. The map shows every
   head's score in the ink scale, from 0 to 1, and the list beside it the strongest heads with
   their intervals.
4. Choose a head, on the map or in the list, to see its attention on your prompts above.

## What the scores mean

Each sequence is random tokens, repeated twice. A random sequence means nothing, so a head that
attends to the earlier copy of the current token, or to the token after it, does so by position
and identity alone. A head's score is the attention it pays to its kind's target, averaged over
the queries of the second copy (over every query after the first, for previous-token heads). The
score is kept per sequence, and reported as the mean over sequences with a seeded percentile
bootstrap interval over them.

The settings that change the scores are shown with them: the number of sequences, their length,
the seed, whether a beginning-of-sequence token comes first, and, for a model read in its chat
template, that each sequence sits where a prompt would. Changing a setting marks the scores as
stale until **Find again**.

## Read them with care

Attention says where a head reads, not what it does with it. A head can attend like an induction
head and matter little for a task; one that matters can attend diffusely. Check a head found here
with an intervention: in the IOI example, patching the induction heads 5.5 and 6.9 restores part of
the answer (a third and a tenth of the gap), since noticing that a name repeats is part of the task
(see [replicate IOI](replicate-ioi.md)). Pin two heads for **Head comparison** to see their
attention side by side on the same prompt.

## From Python

```python
import logogram as lg

model = lg.load_model("openai-community/gpt2")
kinds = lg.head_kinds(model, seed=0, sequences=32, length=32)
for head in kinds["top"]["induction"][:5]:
    print(f"L{head['layer']} H{head['head']}  {head['mean']:.2f} [{head['lo']:.2f}, {head['hi']:.2f}]")
```

`lg.head_kinds` returns, for each kind, every head's mean score with its interval, the
per-sequence values and the strongest heads. See [the Python API](../reference/python-api.md#lghead_kinds).

## References

* Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.* Transformer
  Circuits Thread.
* Olsson et al. (2022). *In-context Learning and Induction Heads.* Transformer Circuits Thread.
