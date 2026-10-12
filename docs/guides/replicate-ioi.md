# Replicate IOI on GPT-2

In *When Mary and John went to the store, John gave a drink to*, GPT-2 small says *Mary*. Wang et
al. (2022, *Interpretability in the Wild*) traced the heads that do it: induction and
duplicate-token heads notice that *John* repeats, S-inhibition heads pass that on, and name mover
heads copy the other name, *Mary*, to the output, while negative name movers push against it.
This guide reproduces the core of their evidence with the bundled IOI example.

## 1. Run the example

Open the IOI example (the first screen, **All projects**, or `lg.open_example("ioi-gpt2")` from
Python), and run its experiment: patch each head's output from the clean prompt into the corrupt
prompt, at every position, measured by the logit difference. [Getting started](../getting-started.md)
walks through it.

**Check the baseline first.** On the example's 32 prompts, GPT-2 small's clean logit difference
is about 2.9 (Wang et al. report 3.56 on their own prompts), and 29 of the 32 clean prompts
prefer the answer.

## 2. Read the heads against the paper

On GPT-2 small at revision `607a30d`, the strongest effects are:

| Head | Effect | Role in Wang et al. |
|---|---|---|
| 5.5 | 0.33 | induction head |
| 8.6 | 0.30 | S-inhibition head |
| 8.10 | 0.25 | S-inhibition head |
| 7.9 | 0.21 | S-inhibition head |
| 9.9 | 0.17 | name mover |
| 10.0 | 0.11 | name mover |
| 10.7 | −0.45 | negative name mover |
| 11.10 | −0.21 | negative name mover |

Each effect is a share of the clean–corrupt gap restored by that head alone, with its interval,
simultaneous band and q-value in the inspector. Patching measures a head's *total* effect: an
S-inhibition head restores a lot because the name movers downstream act on what it writes.

## 3. Find the name movers by what they write

The name movers are the heads that write the answer directly. Rerun with **direct logit
attribution** of the clean prompts at the last token (in the form, choose it as the method; see
[direct logit attribution](../concepts/direct-logit-attribution.md)):

```json
"experiment": {"kind": "direct_logit_attribution", "prompts": "clean"},
"scope": {"kind": "heads", "position": {"kind": "last"}}
```

The name movers 9.9, 9.6 and 10.0 have the largest direct effects, and 10.7 and 11.10 the most
negative: the measure Wang et al. found the name movers by.

## 4. Ask where each head acts

The example's prompts name their positions (`IO`, `S1`, `S2` and `end`). Patch the heads at one
position at a time, with a position of kind `label`:

```json
"scope": {"kind": "heads", "position": {"kind": "label", "label": "end"}}
```

In Wang et al.'s account, the name movers and S-inhibition heads act at the last token (`end`),
and the induction heads at the repeated name (`S2`); a sweep at each label shows whether GPT-2
small does the same on these prompts. A `layer_position` scope with `"positions": "labels"` asks
the same of the residual stream, layer by layer.

## 5. Ask how they connect

[Path patching](../concepts/path-patching.md) into the name movers' queries shows which heads
reach the name movers, and through which input:

```json
"experiment": {
  "kind": "path_patching",
  "direction": "clean_to_corrupt",
  "receivers": [
    {"kind": "head", "layer": 9, "head": 9, "input": "q"},
    {"kind": "head", "layer": 9, "head": 6, "input": "q"},
    {"kind": "head", "layer": 10, "head": 0, "input": "q"}
  ],
  "freeze_mlps": false
}
```

Wang et al. found the S-inhibition heads acting on the name movers this way, through their
queries; senders that matter here, and not through keys or values, support that reading.

## 6. Test the circuit

On the finished head sweep, **Test the top sites as a circuit** keeps its strongest 1, 2, 4, 8 …
heads alone, with every other head replaced, for a curve of faithfulness against the number of
heads kept (see [circuits](../concepts/circuits.md)). To test the paper's circuit itself, stage its
heads on the model map and choose **Test as a circuit**.

Wang et al. mean-ablated the heads outside their circuit over prompts with three unrelated names.
Generate such prompts with the IOI task's `abc` corruption (**Prompts → Generate a task**, or
`logogram generate ioi -n 32 --seed 0 -O corruption=abc --name ioi-abc`), and use mean ablation
over their corrupt prompts as the replacement. Keeping the paper's 26 heads this way keeps all of
the behavior (faithfulness 1.02), and taking 9.9 out costs 0.10 [0.06, 0.14].

## 7. Check it holds

**Check robustness** reruns the head sweep with one choice changed: the other direction (does the
answer *need* these heads?), resample ablation instead of patching, or more donors. A head whose
conclusion changes is flagged on the map. See [check a finding's
robustness](check-robustness.md).

## From Python

The [IOI notebook](../../notebooks/ioi-head-patching.ipynb) runs steps 1, 2 and 7 with
`import logogram as lg`.

## References

* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
