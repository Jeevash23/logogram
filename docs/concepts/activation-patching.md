# Activation patching

Activation patching asks where in a model a behavior lives by moving one piece of computation
from one input to another. Logogram runs one prompt of a pair, replaces the activation at one
site with its value from the other prompt, and measures how the model's answer changes. It is
also called causal tracing or interchange intervention.

## What Logogram computes

Each prompt pair has a **clean** prompt, where the model shows the behavior, and a **corrupt**
prompt that differs in what the behavior depends on. In the IOI example:

* clean: *When Mary and John went to the store, John gave a drink to* → *Mary*
* corrupt: *When Mary and John went to the store, Mary gave a drink to* → *John*

For each site and each pair, Logogram makes three forward passes' worth of measurements:

* the **receiver** run, the prompt the intervention happens in, unchanged;
* the **source** run, the other prompt, whose activation is copied;
* the **patched** run: the receiver prompt with the site's activation replaced by its value in
  the source run, everything else computed as usual from there on.

It reads the [metric](metrics.md), such as the logit difference between the answer and the
distractor, in each, and reports the site's **effect** on that prompt as

```text
effect = (patched − receiver) / (source − receiver)
```

where the denominator is the gap between the two prompts averaged over the dataset
(`dataset_gap`) or each prompt's own gap (`prompt_gap`), as the spec says (see
[normalization](metrics.md#normalization)). Every prompt's value is kept; the site's result is
their mean with a [bootstrap interval](statistics.md).

## Directions

The **direction** decides which prompt receives and which gives. It is a choice the spec states,
because the two ask different questions:

* **Clean → corrupt** (`clean_to_corrupt`, also called denoising) runs the corrupt prompt and
  patches in clean activations. *Is this site enough to restore the behavior?* An effect of 1
  means patching that site alone brings the metric all the way back to the clean prompt's.
* **Corrupt → clean** (`corrupt_to_clean`, noising) runs the clean prompt and patches in corrupt
  activations. *Does the behavior need this site?* An effect of 1 means corrupting that site
  alone moves the metric as far as the whole corrupt prompt does.

In both directions, 0 means the site changed nothing, a negative effect means it moved the metric
the other way, and an effect above 1 means it overshot the other prompt.

The two directions disagree in instructive ways (Heimersheim and Nanda, 2024). Where a behavior
needs several sites together (an AND), no single site restores much from corrupt to clean, but
corrupting any of them breaks it. Where several sites can each do the job (an OR, such as backup
heads that take over), corrupting one breaks little, while restoring any one restores a lot.
**Check robustness** offers the other direction for every patching run, so a finding can be read
both ways.

## Sites and positions

A spec names **abstract sites**, the same for every architecture:

| Site | What it is |
|---|---|
| `resid_pre` | The residual stream entering a layer |
| `resid_mid` | The residual stream between attention and the MLP (only in models whose layers add them one after the other) |
| `resid_post` | The residual stream leaving a layer |
| `attn_out` | A layer's attention output, the sum of its heads' writes |
| `mlp_out` | A layer's MLP output |
| `head` | One attention head's output `z`, before the output projection |
| `q` | One head's query |
| `k`, `v` | One key/value head's key or value |
| `neuron` | One MLP neuron's activation after the nonlinearity (for a gated MLP, the gate's activation times the up projection): the input to the MLP's output projection |
| `sae_feature` | One feature of an SAE (see [SAE features](sae-features.md)) |

Queries, keys, values and neurons are finer sites, the parts that make up a layer's attention and
MLP outputs: `{"kind": "q", "layer": 9, "head": 6, ...}` names one head's query, and
`{"kind": "neuron", "layer": 3, "neuron": 1201, ...}` one neuron. With grouped-query attention
(Llama 3, Qwen 2.5 and Gemma 2, among others), a model has fewer key and value heads than query
heads: a `k` or `v` site names a key/value head, which every query head of its group reads, so
patching it changes all of them. When a model loads, Logogram measures that these sites make up
the attention and MLP outputs (every query, key and value of a layer together reproduce its
attention output, one alone changes only the heads that read it, and every neuron together
reproduces the MLP output) and offers only those that hold: OLMo 2, which normalizes every head's
queries and keys together, offers values and neurons but not single queries or keys.

Activation patching, ablation, attribution patching and [sets of sites](circuits.md) take them.
[Direct logit attribution](direct-logit-attribution.md) splits off a neuron's term at the last
token and a value's at any position, and refuses queries and keys, which act on the logits only
through the attention's softmax. [Path patching](path-patching.md) and
[steering](steering.md) refuse all four: a path starts at a component that writes into the
residual stream, and steering adds to the residual stream itself. The model map has no cell for
them: the results show a sweep of queries, keys or values as a layer × head grid (a column per
key/value head for keys and values), and a sweep of neurons layer by layer, each layer's strongest
neurons with their intervals beside the distribution of all of them.

Each site has a **position**: `all` (every token at once), `last` (the token that predicts the
answer), a token `index` (negative counts from the end), or a named `label` from the dataset,
such as `S2` in IOI, which names a span of characters in the clean prompt and resolves to the last
token that overlaps it. Labels let templates of different lengths line up. Usually both prompts of
a pair have the same number of tokens, so that a position means the same in both; pairs of
different lengths are matched by their labels instead (below).

A **scope** is the set of sites one run measures, each on its own:

* `heads`: every head of every layer at one position, drawn as a layer × head grid;
* `head_inputs`: every head's query, or every key/value head's key or value (`input`), at one
  position, also a layer × head grid;
* `neurons`: every neuron of the MLPs of the chosen `layers`, at one position;
* `layer_position`: one stream site at every layer and every position (or every labelled
  position), a layer × position grid, as in causal tracing (Meng et al., 2022);
* `layer_components`: attention and MLP outputs of every layer at one position;
* `sites`: an explicit list, such as heads staged on the map;
* `site_sets`: [sets of sites intervened on together](circuits.md).

The `graph` and `edges` scopes hold connections between components rather than sites, for
[edge attribution](edge-attribution.md), [ACDC](acdc.md) and path patching of edges.

## Pairs of different lengths

A pair whose prompts tokenize to different lengths, such as a name of one token against a name of
three, can be used when the dataset labels its positions in both prompts: `positions` gives
character spans in the clean prompt, and `corrupt_positions` the same labels in the corrupt one.
A run then matches the two prompts label by label: patching at `S2` carries the activation at the
source prompt's `S2` token into the receiver prompt's own `S2` token, and `last` matches the two
prompts' last tokens. Without `corrupt_positions`, such a pair can't be used, and the Prompts
view says why.

Across different lengths, every position at once (`all`) and token indices have no meaning.
Methods that carry a value from one prompt into the other (activation patching, attribution
patching with one gradient, path patching, mean and resample ablation, and steering by the mean
difference) refuse them on such pairs, naming the pairs, and take the last token and labels.
Methods that read one prompt at a time take any position they otherwise take: zero ablation,
direct logit attribution and steering along an SAE feature. A few need pairs of equal length
altogether:

* integrated gradients, which run on inputs between the two prompts' embeddings, position by
  position;
* sets that keep their sites and patch the rest of the model, and mean or resample ablation of
  sets;
* the layer × position sweep of every token index (`"positions": "each"`), whose labelled form
  works;
* [edge attribution](edge-attribution.md), [ACDC](acdc.md) and path patching of edges, which
  carry each sender's output at every position.

In the app, the token strip lines the two prompts up by their labels and last tokens, each with
its own positions. With such pairs and a method that carries values between prompts, the
experiment form offers only the last token and labels, and says why; choosing such a method
replaces every position and token index with the last token (and the sweep of every token index
with the labelled one). A run that uses them says how they were matched. **Generate a
task** with **Full vocabulary** keeps the words the model splits into several tokens, and with
them pairs of different lengths, labelled in both prompts.

## When to use it

Activation patching is the reference measurement: it is exact for the intervention it describes,
and the estimates of [attribution patching](attribution-patching.md) are verified against it.
Use it to find which components carry a behavior, then narrow down with
[path patching](path-patching.md) (through which later components) and
[circuits](circuits.md) (whether a set of them is enough).

Its answers depend on the corrupt prompt. A corruption that changes one name tests the
computation that depends on that name; one that changes everything (or adds noise to the
embeddings) can make sites look important for reasons unrelated to the task. Zhang and Nanda
(2024) compare corruptions and metrics and recommend minimal pairs like these, with the logit
difference. Logogram also offers [ablation](ablation.md), which replaces an activation with a
baseline rather than with the other prompt's value.

## Exactness

Tests run every time the code changes check identities that patching must satisfy: patching every
clean activation of the residual stream at a layer gives an effect of exactly 1, patching corrupt
activations into the corrupt prompt gives exactly 0, and the same spec twice gives bit-identical
results. Patching every query, key and value of a layer at once gives what patching its attention
output gives, and every neuron of a layer what patching its MLP output gives. With pairs of
different lengths, patching the residual stream after the last layer at the last token
reproduces the source run exactly, however long each prompt is. Library hook names never appear
in a spec: the backend maps each abstract site to the model's own activation.

## In a spec

```json
"experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"},
"scope": {"kind": "heads", "position": {"kind": "all"}}
```

`{"kind": "head_inputs", "input": "q", "position": {"kind": "last"}}` sweeps every head's query
instead, and `{"kind": "neurons", "layers": [9, 10, 11], "position": {"kind": "last"}}` every
neuron of three layers. See
[the spec reference](../reference/spec.md#experiment-activation_patching) for every field.

## References

* Vig, Gehrmann, Belinkov, Qian, Nevo, Singer and Shieber (2020). *Investigating Gender Bias in
  Language Models Using Causal Mediation Analysis.* NeurIPS.
* Geiger, Lu, Icard and Potts (2021). *Causal Abstractions of Neural Networks.* NeurIPS.
* Meng, Bau, Andonian and Belinkov (2022). *Locating and Editing Factual Associations in GPT.*
  NeurIPS.
* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
* Zhang and Nanda (2024). *Towards Best Practices of Activation Patching in Language Models:
  Metrics and Methods.* ICLR.
* Heimersheim and Nanda (2024). *How to use and interpret activation patching.* arXiv.
