# ACDC: automated circuit discovery

[Edge attribution](edge-attribution.md) estimates every edge of the model's graph at once, from
gradients. ACDC (Conmy, Mavor-Parker, Lynch, Heimersheim and Garriga-Alonso, 2023) measures them
instead: it removes the edges one at a time, in real forward passes, and keeps only those whose
removal changes the [metric](metrics.md) by the threshold or more. What is left is a circuit: a
subgraph of the model that does the task by itself, the rest of the model replaced by the source
prompt's values.

## What Logogram computes

The graph is edge attribution's: the embeddings, every attention head, every MLP and the logits,
with an edge from each node's output to each later node's input, each receiver reading its own copy
of the residual stream (see [the graph](edge-attribution.md#the-graph)). The `graph` scope chooses
the candidate edges, those into the heads and MLPs of its `layers` and into the logits; every other
edge of the model stays in the circuit throughout.

At first every edge is in the circuit, and the receiver prompt runs as it is. ACDC then tests the
candidate edges one at a time:

1. **Remove the edge.** The receiver prompt runs once more, edge-patched: every edge removed so
   far, and this one, carries its sender's output on the source prompt into its receiver, while
   every edge of the circuit carries what its sender outputs in that pass. A removed edge always
   carries the source run's value, whatever the circuit has become; an edge of the circuit
   carries its sender's value as the circuit computes it.
2. **Compare.** The change is the metric's mean over the prompts in that pass minus the
   circuit's just before.
3. **Decide.** When the change, counted as `rule` says, is below `threshold`, the edge stays
   removed and the circuit's metric is the new one. Otherwise the edge is put back.

With `"rule": "toward_source"`, the change counted is how far the metric moves toward the source
prompt's value: the change times the sign of the mean of (source − receiver) in the metric. A
removal that moves the metric the other way counts as negative, so that edge always stays
removed. This is the paper's rule: running the clean prompt, with the KL divergence from the clean
prompt's own prediction as the metric, an edge is removed when KL(model ‖ circuit without it) −
KL(model ‖ circuit) < τ. With `"rule": "either_way"`, the change counted is its size, whichever
way it goes (the absolute value, an option of ACDC's code): an edge that works against the
behavior is kept too.

**The order.** The edges are tested in the reverse of the graph's order: the receivers from the
logits back toward the embeddings (the logits; then each layer from the last: its MLP's input, then
its heads from the last, each head's value, key and query), and each receiver's edges from its
latest sender back to the embeddings. Where heads share keys and values, a layer's heads come as
its key/value heads from the last, each one's value and key, then the queries of its group's heads
from the last. A receiver's output reaches only receivers later in the graph, so each receiver is
tested after every receiver its output reaches: by the time an edge into a node is tested, whether
the node matters to the circuit is settled. A node whose outgoing edges were all removed no longer
matters, and its incoming edges change nothing when removed.

**The threshold** is in the metric's own units, compared with the change in its mean over the
prompts: nats for the KL divergence, logits for the logit difference. Read it against the gap
between the two runs: with a clean–corrupt gap of 3 logits, a threshold of 0.1 drops any edge
whose removal costs less than a thirtieth of it.

## What a run records

* **Every edge's change.** Each candidate edge keeps, for every prompt, the change its removal made
  when it was tested: the metric with it removed minus the circuit's just before. Its effect is
  that change normalized like patching (1 is a change as large as the whole gap from the receiver
  run to the source run), with its interval and the other [statistics](statistics.md).
* **The circuit:** the edges kept, `n_kept` of `n_edges`.
* **Its faithfulness:** the circuit's metric, per prompt, beside the receiver run's (every edge
  kept) and the source run's, normalized as [the faithfulness curve](edge-attribution.md#the-faithfulness-curve)
  normalizes with every edge but the chosen patched: 0 is the source run, 1 the receiver run, with
  a bootstrap interval from the run's resamples. Tests check that it is what the faithfulness
  curve gives for the same edges, bit for bit.

The summary lists the circuit's edges, then the strongest of the edges removed, 2,000 at most.
`edges.parquet` in the run's folder holds every edge, with `kept` and `step` (when it was tested,
from 0); **Export** has them all as a CSV, `run.edges()` reads them in Python, and `run.circuit()`
returns the circuit: its edges with their changes, the threshold and rule, and its faithfulness.

## Where it can mislead

* **The threshold.** Which edges stay depends on it, and not always gradually. Rerun with another
  threshold (**Check robustness** offers one) and compare the two circuits edge by edge
  ([compare two runs](../guides/compare-runs.md)): edges that stay at several thresholds are the
  ones to trust.
* **The order.** Each edge is judged against the circuit as it stood when it was tested. Where two
  senders carry the same thing to a receiver, the one tested first is removed, since the other
  still carries it, and the second is then needed and kept; tested the other way round, the other
  would stay. The circuit is one of the circuits that do the task, not the only one.
* **Small steps add up.** Every removal is compared with the circuit just before it, so many
  removals each below the threshold can move the metric far. Read the circuit's faithfulness, not
  only which edges it kept: a faithfulness well below 1 says the circuit lost part of the behavior
  on the way.
* **The rule.** With changes counted toward the source, edges that work against the behavior (as
  negative name movers do in IOI) are always removed: the circuit is what carries the behavior,
  not everything that affects it, and it can do more than the model. On the 32 IOI prompts with
  GPT-2 small, ACDC over the edges into the first layer and the logits (206 edges, corrupt →
  clean, a threshold of 0.05 logits) kept 19, among them the name movers' edges into the logits
  (L9 H9's, L9 H6's and L10 H0's). It removed the negative name movers' (L10 H7's, whose removal
  raised the logit difference by 2.6, and L11 H10's, by 1.2), so the circuit's logit difference
  is 7.9 where the model's is 2.9: a faithfulness of 1.86 [1.70, 2.02]. Count changes either way
  to keep such edges.
* **The prompts that chose it.** The circuit is chosen and its faithfulness measured on the same
  prompts, so the faithfulness is optimistic. Rerun ACDC on other prompts (a dataset generated
  with another seed) and compare the two circuits edge by edge: Logogram can't yet run a circuit
  found on one dataset as a fixed set of edges on another.
* **The direction.** `corrupt_to_clean` (ACDC's own) runs the clean prompts and asks which edges
  the behavior needs: everything else carries corrupt values. `clean_to_corrupt` runs the corrupt
  prompts and asks which edges must keep their corrupt values to keep the behavior away. They ask
  different questions, so there is no default.

## Cost

ACDC takes one forward pass per candidate edge and batch of prompts, beside the unpatched runs
(each side's, and the KL divergence's target first), one capture of the senders on the source
prompts and one pass with every edge kept: (E + 4) × batches passes in all, (E + 5) × batches with
the KL divergence. GPT-2 small's whole graph has 32,491 edges, so a run makes 32,491 passes for
each batch, where edge attribution takes a few; choose some layers to search part of the graph.
The senders' outputs on the source prompts are kept for every pass (in CPU memory on a GPU, when
they are large).

Before a run, the experiment form says how many passes it takes and how long they may take here:
the server times one edge-patched pass of the largest batch on the loaded model, with every edge
removed, and multiplies. On the 32 IOI prompts with GPT-2 small, on one CPU thread, a pass took
about 3 seconds: the 206 edges into the first layer and the logits took 11 minutes (the form had
said about 9), and the form puts the whole graph at about 50 hours. A run is refused, with what to
change, when its graph has more than 100,000 edges or its edges times its prompts make more than
10 million per-prompt values, as edge attribution's are.

A run that stops early, cancelled or failed, keeps nothing: until every edge has been tested, the
edges left aren't ACDC's circuit.

## Limits

ACDC has edge attribution's [limits](edge-attribution.md#limits): models whose heads each write
into the residual stream and whose receivers can read their own copies of it (measured the first
time it plans or runs on a model; models with grouped-query attention included, models that
normalize after combining their heads refused), prompt pairs of equal length, and answers of single
tokens or sets of them.

## On the toy models

On the [toy models](../guides/toy-models.md), whose circuits are known, ACDC keeps exactly the
planted edges along which the prompts differ, as path patching each edge alone finds them:

| Toy | Metric and threshold | Edges kept |
|---|---|---|
| `induction` | logit difference, 1 logit (the gap is 20); either direction | embeddings → L0 H2's value, L0 H2 → L1 H1's key, L1 H1 → the logits: 3 of 110 |
| `ioi` | KL divergence from the clean prompt's prediction, 0.1 nats (the gap is about 10) | embeddings → L1 H1's query, L1 H1 → L2 H3's value, L2 H3 → L3 H0's query, L3 H0 → the logits: 4 of 479 |

Each kept edge moved the metric all the way when it was tested (a change of 1.0), every other edge
not at all, and the circuit's faithfulness is 1.0. Where one gradient (EAP) misses the edges that
act through saturated attention, ACDC finds them, at the cost of a pass for every edge.

## In the app

* In **Configure**, choose **Circuit discovery (ACDC)**: the direction, the threshold and which
  change it is compared with, and the edges (**Edges into** every layer or chosen layers, and
  **Edges into the logits**). Before the run, the summary says the graph's size, the passes the run
  takes and how long they may take.
* The results draw the circuit between the model's components, layers left to right, each edge in
  the effect colors of the change its removal made; beside them, the edges kept, the circuit's
  faithfulness and the threshold, and every listed edge with whether it was kept. The inspector says
  how the edge chosen was tested.
* **Check robustness** offers another threshold, the other rule, the opposite direction, and edge
  attribution over the same graph, to compare its estimates with the edges ACDC kept. The HTML
  report draws the circuit.

## In a spec

```json
"experiment": {
  "kind": "acdc",
  "direction": "corrupt_to_clean",
  "threshold": 0.1,
  "rule": "toward_source"
},
"scope": {"kind": "graph", "layers": null, "logits": true},
"metric": {"kind": "kl", "target": "clean", "normalization": "dataset_gap"}
```

See the spec reference for [ACDC](../reference/spec.md#experiment-acdc) and the
[`graph`](../reference/spec.md#scope-graph) scope.

## References

* Conmy, Mavor-Parker, Lynch, Heimersheim and Garriga-Alonso (2023). *Towards Automated Circuit
  Discovery for Mechanistic Interpretability.* NeurIPS.
* Syed, Rager and Conmy (2023). *Attribution Patching Outperforms Automated Circuit Discovery.*
  arXiv (BlackboxNLP 2024).
* Hanna, Pezzelle and Belinkov (2024). *Have Faith in Faithfulness: Going Beyond Circuit Overlap
  When Finding Model Mechanisms.* COLM.
* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
