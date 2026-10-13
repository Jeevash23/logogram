# Path patching

[Activation patching](activation-patching.md) measures a component's whole effect: everything it
changes downstream. Path patching measures the part of that effect that travels to chosen
**receivers**: a later head's query, key or value input, or the logits read straight from the
final residual stream. It is how Wang et al. (2022) found that GPT-2 small's S-inhibition heads
act through the name mover heads' queries.

## What Logogram computes

For each **sender** (a head, attention output or MLP output of the sweep), in two steps:

1. The receiver prompt runs with the sender's activation taken from the source prompt, while
   every other attention head is held at its value in the receiver run. The change can then only
   travel through the residual stream, and through the MLPs, which recompute from it, unless
   `freeze_mlps` holds them at their receiver-run values too. What the receivers read in this run
   (each receiving head's query, key or value input, or the final residual stream for the logits)
   is recorded.
2. An unchanged receiver run gets only those recorded inputs patched in, at the receivers, and
   the [metric](metrics.md) is read there.

The result is the sender's effect through those receivers alone, normalized like patching: 1
means the path alone restores the clean behavior (in the clean → corrupt direction), 0 that
nothing travels along it.

* A sender in or after the last receiver's layer has no path to it, so a sweep keeps only the
  layers before the last receiver.
* A path starts at a component that writes into the residual stream, so queries, keys, values and
  neurons can't be senders.
* In models whose heads share keys and values (grouped-query attention, as in Llama and Qwen), a
  key or value receiver is a key/value head: its `head` counts the layer's key/value heads, and
  the path reaches every query head of the group through the key or value they share. The results
  write it `L5 KV1 k`. Tests on tiny Llama and Qwen 2 models check that a path from a head into a
  key/value head is what patching that one edge of the graph does.
* With [pairs of different lengths](activation-patching.md#pairs-of-different-lengths), the
  sender is patched at a label or the last token, from the source prompt's token there into the
  receiver's; the rest of both passes runs on the receiver prompt alone.
* From the last layer to the logits, a path is the whole effect: tests check that path patching
  there gives exactly what activation patching gives.

## Edges, one at a time

With an `edges` scope, each result is one edge of the model's graph: one sender's effect through
one receiver alone (a later head's query, key or value, a later MLP's input, or the logits), with
every other head and every MLP held, so the change reaches the receiver only directly. This is how
[edge attribution](edge-attribution.md) verifies its strongest estimates: **Verify top 20 by path
patching** writes such a spec from an edge attribution run. It needs `freeze_mlps` set to `true`,
and pairs of equal length.

## When to use it

Use path patching once activation patching has found the components that matter, to ask how they
connect: which earlier components a head reads from, and through which of its inputs. Frozen
heads make the measurement specific: the effect through the receivers, not through everything
the sender changes. **Check robustness** offers the same sweep with the MLPs held or released,
in the opposite direction, and the senders' whole effect by patching, so the path can be read
against the total.

## In a spec

```json
"experiment": {
  "kind": "path_patching",
  "direction": "clean_to_corrupt",
  "receivers": [
    {"kind": "head", "layer": 9, "head": 9, "input": "q"},
    {"kind": "head", "layer": 10, "head": 0, "input": "q"}
  ],
  "freeze_mlps": false
},
"scope": {"kind": "heads", "position": {"kind": "all"}}
```

`{"kind": "logits"}` is the receiver for the direct path to the output. See
[the spec reference](../reference/spec.md#experiment-path_patching).

## References

* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
* Goldowsky-Dill, MacLeod, Sato and Arora (2023). *Localizing Model Behavior with Path
  Patching.* arXiv.
