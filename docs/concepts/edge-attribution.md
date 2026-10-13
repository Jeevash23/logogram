# Edge attribution

[Attribution patching](attribution-patching.md) estimates what patching each component would do.
Edge attribution asks which **connections** between components carry the behavior: an edge is one
component's output as one later component reads it, such as a head's output read by a later
head's query. Logogram estimates every edge of the model's graph at once from gradients (EAP,
Syed, Rager and Conmy, 2023; with integrated gradients, EAP-IG, Hanna, Pezzelle and Belinkov,
2024), measures the strongest by [path patching](path-patching.md), and patches the strongest
together for a **faithfulness curve** beside edges chosen at random.

## The graph

The graph's **nodes** are the embeddings, every attention head (its output `z` through its slice
of the output projection), every MLP and the logits. Every node writes into the residual stream,
which every later node reads, and an **edge** runs from one node's output to one later node's
input:

* a head's query, key and value inputs, each its own edge from the embeddings and from every head
  and MLP of earlier layers;
* an MLP's input, from the embeddings, every earlier head and MLP, and, where a layer adds its
  attention before its MLP (GPT-2, Llama), the heads of its own layer; where both read the same
  residual stream (Pythia), not those;
* the logits, from every node.

GPT-2 small, 12 layers of 12 heads, has 157 senders, 445 receivers and 32,491 edges. A model of
the same size whose layers read attention and MLP from the same stream has 144 fewer.

**Shared keys and values.** With grouped-query attention (Llama, Qwen, Mistral), a layer has fewer
key/value heads than query heads, and each query head of a group reads its key/value head's key and
value. A key or value input then belongs to the key/value head: a layer has a query input for every
query head and a key and a value input for every key/value head, and an edge into one reaches
every query head of the group through what they share. The results write such a receiver `L5 KV1 k`
(key/value head 1's key in layer 5), and draw it at the middle of the query heads that read it.
Qwen 2.5 0.5B, 24 layers of 14 query heads sharing 2 key/value heads, has 361 senders, 457
receivers and 79,813 edges.

**Each receiver's own input.** In the model, a layer's heads all read one residual stream,
normalized once. For an edge to mean something on its own, each receiver reads its own copy of
the residual stream at its layer, before its normalization, and normalizes that copy itself: each
head for its query, key and value (each key/value head for its key and value, where heads share
them), as TransformerLens's split inputs do, each MLP for its input, and the final normalization
for the logits. The residual stream itself runs on unchanged; only what one receiver reads can
differ.

The `graph` scope chooses the receivers: `layers`, the layers whose heads and MLPs receive edges
(each from every earlier node), in order, or `null` for every layer; and `logits`, whether the
edges into the logits are included.

## What Logogram computes

Patching the edge from u to v means that v reads its own input plus u's output on the source
prompt minus u's output on the receiver prompt, while everything else reads what it read. To first
order, that changes the [metric](metrics.md) by

```text
estimate(u → v) = Σ over positions (u_source − u_receiver) · ∂metric/∂x_v
```

where u_source and u_receiver are u's outputs on the two prompts, x_v is v's own input, and the
gradient is taken at the receiver run. The sum runs over positions because u's output at a
position reaches v's input at that position only.

The gradient goes through v's normalization as the model computes it, with its scale recomputed
from x_v, because patching the edge would recompute it. Holding the scale, as
[direct logit attribution](direct-logit-attribution.md) does, would linearize another function.
With the scale recomputed, the gradient is orthogonal to x_v (lengthening the input leaves the
normalization's output as it was, up to its epsilon), so a sender that writes along the
receiver's own input is estimated to do little. Tests check the gradient against finite
differences of the metric on tiny models, and that each edge's estimate is the slope of patching
that edge.

One backward pass gives every receiver's gradient, and each edge's estimate follows from its
sender's difference between the prompts, so the whole graph costs a few passes per batch of
prompts. The estimate is normalized like patching: 1 would mean the edge alone restores the clean
behavior (in the clean → corrupt direction). The results label every value as an estimate.

Every edge keeps its per-prompt values and its [statistics](statistics.md), and the simultaneous
bands and q-values count every edge the run estimates. The summary lists the strongest 2,000;
`edges.parquet` in the run's folder holds them all, **Export** has them all as a CSV, and
`run.edges()` reads them in Python. The metric is read at the last token, so answers and
distractors must be single tokens or sets of them.

## One gradient or integrated gradients

With `"method": "gradient"` (EAP), the gradient is taken at the receiver run: one forward and
backward pass per batch estimates every edge. Like any first-order estimate, it misses saturation,
in attention patterns, normalizations and the final softmax, and can miss an edge or give it the
wrong sign.

With `"method": "integrated_gradients"` (EAP-IG), the gradient is averaged over `steps` runs whose
input embeddings lie evenly between the receiver prompt's and the source prompt's, at the
midpoints of equal intervals, as in
[attribution patching](attribution-patching.md#integrated-gradients); the senders' differences
stay those of the two prompts. It costs `steps` forward and backward passes instead of one. On
the IOI prompts with GPT-2 small, one gradient misses the embeddings' edge into the first MLP
(estimated −0.05, where path patching measures 0.99), which integrated gradients find (0.67).

## Verifying estimates by path patching

An estimate says which edges to look at; path patching measures them (Wang et al., 2022).
**Verify top 20 by path patching**, on an edge attribution run's results, measures the edges with
the largest estimated effects one at a time, in a run of its own that records which run it
verifies, and opens the two runs side by side. Each edge is path patched from its sender into its
receiver alone: the receiver prompt runs with the sender's output taken from the source prompt and
every other head and every MLP held at its value in the receiver run, so the change reaches the
receiver only directly, as the edge carries it; what the receiver reads there is then patched into
an unchanged receiver run, where the metric is read. Tests check that this separate code path
measures what patching the edge does.

The same is `logogram verify RUN --top 20` on the command line and `lg.verify_run(run, top=20)`
in Python, which measure at most 50 edges. Once the verification finishes, the edge attribution
run's results and the comparison of the two runs show **Estimates against path patching**, and
`lg.edge_verification(estimated, verified)` returns the same table:

* `rows`: each verified edge's estimate beside its measured effect, each with its interval, and
  its rank by each;
* `spearman`: Spearman's rank correlation between the estimates and the measurements;
* `same_sign`: how many of the `n` edges have an estimate and a measurement of the same sign.

A rank correlation near 1 says the estimates order the strongest edges as path patching does; an
edge whose sign differs is one the estimate gets wrong. On the 32 IOI prompts with GPT-2 small,
over the 50 strongest edges, the rank correlation is 0.89 for one gradient (47 of 50 with the same
sign) and 0.93 for integrated gradients (49 of 50). Verifying the strongest estimates can't find
an edge the estimate missed altogether: compare one gradient with integrated gradients (**Check
robustness** offers the other), and read the faithfulness curve.

## The faithfulness curve

With `curve`, the run then patches the k strongest edges together, for each k of `counts`, in
real forward passes. Edge patching is exact: the receiver prompt runs once, and each receiver reads
its own input plus, for every patched edge into it, the sender's output on the source prompt minus
its output in that pass. Patching every edge of the graph reproduces the source run, and patching
none the receiver run: tests check both, and a run over the whole graph reports how far its pass
with every edge patched is from the source run, with a warning when the edges don't add up to the
model in its precision. One edge patched alone is path patching of that edge.

`patch` says which edges carry the source prompt's values: with `"chosen"`, the k strongest do and
every other edge keeps the receiver run's own; with `"rest"`, every other edge does and the chosen
keep the receiver run's. **Faithfulness** is the share of the gap from no edge chosen (0) to every
edge chosen (1) that the chosen edges close, with a bootstrap interval (Hanna, Pezzelle and
Belinkov, 2024). Like the two directions of patching, the two choices ask different questions, so
there is no default:

| Direction | `patch` | The question | Faithfulness 1 is |
|---|---|---|---|
| clean → corrupt | `chosen` | Do the chosen edges alone bring the behavior back? | the clean run |
| clean → corrupt | `rest` | Does leaving only them corrupt keep the behavior away? | the corrupt run |
| corrupt → clean | `chosen` | Does corrupting only them take the behavior away? | the corrupt run |
| corrupt → clean | `rest` | Do the chosen edges alone keep the behavior? | the clean run |

**The random control** chooses the same numbers of edges at random: each of `draws` draws orders
every edge of the graph by a shuffle seeded with `seed` and takes the first k, as the strongest are
taken from the estimates. Each point of the control averages its draws prompt by prompt, and keeps
each draw's own faithfulness. A curve well above its control says the estimates put the edges that
matter first. Counts larger than the graph are left out, with a note.

In the IOI example on GPT-2 small, with every edge but the chosen carrying the corrupt prompts'
values, the 1,000 strongest edges (3% of the graph) keep 0.80 of the behavior [0.74, 0.86] and the
3,000 strongest 0.96, where as many edges chosen at random keep none of it. The edges are chosen
and measured on the same prompts.

## Which scope to use

* The `graph` scope, with edge attribution, estimates every edge into the chosen layers and the
  logits: a search over the whole graph.
* The `edges` scope, with path patching, measures up to 64 named edges one at a time, each its own
  result. `freeze_mlps` must be `true`, since an edge reaches its receiver directly, and
  `receivers` lists each edge's receiver once, in the order the edges first name them. **Verify**
  writes this spec; it can also be written by hand.
* A scope of components (`heads`, `layer_components`, `sites`), with path patching, measures each
  sender's effect through chosen receivers together, through the MLPs or with them held: see
  [path patching](path-patching.md).

## Cost

Per batch of prompts, a run captures the senders' outputs on both prompts (two forward passes),
then takes one forward and backward pass for one gradient, or `steps` of them. The curve adds, per
batch, one capture of the senders on the source prompts, one edge-patched forward pass for each
count, one more for each count and random draw, and one with every edge patched. Every receiver's
gradient is held at once, so a batch needs more memory than patching does.

Before a run, the experiment form says the graph's size on the loaded model and what the run
costs: the passes per batch and the memory a batch needs. A run is refused, with what to change,
when its graph has more than 100,000 edges (choose fewer layers), when its edges times its prompts
make more than 10 million per-prompt values (use fewer prompts with `dataset.limit`, or fewer
layers), or when a batch would need more than 80% of the memory free where it runs (lower
`execution.batch_size` to the size the message gives).

On the 32 IOI prompts with GPT-2 small, on two CPU threads, one gradient takes about a minute and
8 steps of integrated gradients three and a half; path patching the 50 strongest edges takes four
to six. A run that stops early keeps every edge's estimate over the prompts that finished; one
stopped during the curve keeps every edge and leaves the curve out.

## Limits

* **Models.** Edges need heads that each write into the residual stream and receivers that each
  read their own copy of it. The first time edge attribution or [ACDC](acdc.md) plans or runs on a
  model, Logogram measures that giving every receiver its own copy reproduces the model's output,
  that each copy is read before its normalization, and that changing one head's copy changes no
  other head's (for a shared key or value, no other key/value head's). It refuses models that
  normalize the attention output after combining the heads (Gemma 2, OLMo 2), whose heads don't
  each add into the residual stream, and models whose query heads don't divide evenly among their
  key/value heads. Tiny models of the GPT-2, Pythia, Llama and Qwen 2 families pass, and on each
  the tests check that every estimate is the slope of patching its edge, that one edge patched is
  path patching of that edge, and that patching every edge reproduces the source run.
* **Prompt pairs of equal length.** An edge carries its sender's output from one prompt into the
  other at every position, so edge attribution, ACDC and path patching of edges refuse
  [pairs of different lengths](activation-patching.md#pairs-of-different-lengths).
* **Answers of single tokens**, or sets of them: continuations of several tokens are refused.

## In the app

* In **Configure**, choose **Edge attribution**: the direction, **One gradient (EAP)** or
  **Integrated gradients (EAP-IG)**, the edges (**Edges into** every layer or chosen layers, and
  **Edges into the logits**) and the curve. Before the run, the summary says the graph's size, what
  the run costs, or why it would be refused.
* The results draw the strongest edges between the model's components, layers left to right, in
  the effect colors, beside the faithfulness curve and every listed edge ranked with its interval.
  The model map has no place for an edge.
* **Check robustness** offers the other estimator and the opposite direction. The HTML report
  draws the strongest edges, the curve and the verification (see
  [share a result](../guides/share-results.md)).

## When to use it

Use edge attribution to find how the components of a circuit connect: which earlier components a
head reads through its query, key or value, and what reaches the logits directly. Then verify the
strongest by path patching, read the curve against its control, and prefer integrated gradients
wherever one gradient and path patching disagree.

[ACDC](acdc.md) (Conmy et al., 2023), which Logogram also runs over the same graph, removes the
edges one at a time in real forward passes and keeps those the metric needs: a forward pass for
every edge, where edge attribution estimates them all at once, which is what Syed, Rager and Conmy
(2023) proposed it for. Use edge attribution to search a large graph, and ACDC where a circuit
whose every edge was measured is worth a pass per edge, on chosen layers of a larger model.

## In a spec

```json
"experiment": {
  "kind": "edge_attribution",
  "direction": "corrupt_to_clean",
  "method": "integrated_gradients",
  "steps": 8,
  "curve": {
    "counts": [10, 100, 1000, 3000, 10000],
    "patch": "rest",
    "control": {"seed": 0, "draws": 2}
  }
},
"scope": {"kind": "graph", "layers": null, "logits": true}
```

`steps` is `null` with `"method": "gradient"`, and `curve` is `null` to estimate only. Two edges
measured one at a time by path patching:

```json
"experiment": {
  "kind": "path_patching",
  "direction": "corrupt_to_clean",
  "receivers": [{"kind": "head", "layer": 9, "head": 9, "input": "q"}],
  "freeze_mlps": true
},
"scope": {"kind": "edges", "edges": [
  {"sender": {"kind": "head", "layer": 7, "head": 9},
   "receiver": {"kind": "head", "layer": 9, "head": 9, "input": "q"}},
  {"sender": {"kind": "head", "layer": 8, "head": 6},
   "receiver": {"kind": "head", "layer": 9, "head": 9, "input": "q"}}
]}
```

A sender is `{"kind": "embed"}`, a head (`layer` and `head`) or an MLP (`layer`); a receiver is a
head's `q`, `k` or `v`, an MLP's input (`{"kind": "mlp", "layer": 9}`) or `{"kind": "logits"}`.
Where heads share keys and values, a `k` or `v` receiver's `head` is the key/value head.
See the spec reference for [edge attribution](../reference/spec.md#experiment-edge_attribution),
the [`graph`](../reference/spec.md#scope-graph) and [`edges`](../reference/spec.md#scope-edges)
scopes and the [faithfulness curve](../reference/spec.md#faithfulness-curve).

## References

* Syed, Rager and Conmy (2023). *Attribution Patching Outperforms Automated Circuit Discovery.*
  arXiv (BlackboxNLP 2024).
* Hanna, Pezzelle and Belinkov (2024). *Have Faith in Faithfulness: Going Beyond Circuit Overlap
  When Finding Model Mechanisms.* COLM.
* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
* Conmy, Mavor-Parker, Lynch, Heimersheim and Garriga-Alonso (2023). *Towards Automated Circuit
  Discovery for Mechanistic Interpretability.* NeurIPS.
* Sundararajan, Taly and Yan (2017). *Axiomatic Attribution for Deep Networks.* ICML.
