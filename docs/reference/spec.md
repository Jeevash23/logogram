<!-- Generated from src/logogram/spec.py by scripts/build_reference.py. Don't edit it by hand: run `uv run python scripts/build_reference.py`. -->

# The spec

Every experiment is a `spec.json`: the single description of an experiment, which the app, `logogram run` and the Python API all run the same way. Nothing that can change a number is left implicit. A spec states every such field, and a spec that leaves one out is refused with the field's name. This page lists every field, from the models in [`src/logogram/spec.py`](https://github.com/Jeevash23/logogram/blob/main/src/logogram/spec.py).

This is version 4. Version 1 specs (Logogram 0.1) still run: each choice one leaves out takes the value version 1 gave it, the run's first warning lists those values, and `logogram validate SPEC.json` shows them without running anything. Version 2 specs (Logogram 0.2 to 0.4) read with what version 3 added turned off, which is what they did: no chat template, a model from Hugging Face, and steering along the mean difference without continuations. Version 3 specs (Logogram 0.5 and 0.6) read as version 4 with what version 4 adds turned off: the SAE a spec names is a published one, and the layer predictions read the logit lens. Version 4 also adds methods and sources that a version 3 spec can't name.

Objects that come in several kinds (an experiment, a scope, a position, ...) say which with their `kind`, and each kind has fields of its own.

The bundled IOI example's spec:

```json
{
  "logogram_spec": 4,
  "name": "Which heads restore the answer?",
  "notes": "Run each corrupt prompt (the subject is replaced by the indirect object, so the model should prefer the other name) and patch in one head's output from the clean prompt at every position. A normalized effect of 1 means that head alone restores the clean behavior; negative values mean the head works against the answer.",
  "model": {
    "id": "openai-community/gpt2",
    "source": "hub",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "sha256": null,
    "dtype": "float32",
    "device": "auto",
    "process_weights": true
  },
  "dataset": {
    "path": "datasets/ioi.jsonl",
    "sha256": "2bd046d5d0190d42863dfa64e185f61228dd26a58458f7b6cf93140847478ac3",
    "limit": null
  },
  "tokenization": {
    "prepend_bos": true,
    "chat_template": null
  },
  "experiment": {
    "kind": "activation_patching",
    "direction": "clean_to_corrupt"
  },
  "scope": {
    "kind": "heads",
    "position": {
      "kind": "all"
    }
  },
  "metric": {
    "kind": "logit_diff",
    "normalization": "dataset_gap"
  },
  "statistics": {
    "bootstrap": 1000,
    "ci": 0.95,
    "seed": 0,
    "cluster": null
  },
  "execution": {
    "batch_size": 64
  },
  "predictions": null,
  "sae": null
}
```

## Spec

An experiment: what runs, on which model and prompts, and how its numbers are measured.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `logogram_spec` | integer | `4` | The spec's version. Version 1, 2 and 3 specs are read too, as version 4. |
| `name` | string | 1 to 200 characters | What the experiment is called, in the history and the results. |
| `notes` (optional, default `""`) | string | any | What you expect and why, saved with the spec and shown beside the results. |
| `model` | object | [Model](#model) | The model, exactly: its files, precision, device and weight processing. |
| `dataset` | object | [Dataset](#dataset) | The prompt pairs. |
| `tokenization` | object | [Tokenization](#tokenization) | How each prompt becomes tokens. |
| `experiment` | object | one of [`activation_patching`](#experiment-activation_patching), [`ablation`](#experiment-ablation), [`direct_logit_attribution`](#experiment-direct_logit_attribution), [`attribution_patching`](#experiment-attribution_patching), [`steering`](#experiment-steering), [`path_patching`](#experiment-path_patching) or [`edge_attribution`](#experiment-edge_attribution) ([Experiment](#experiment)) | The method, with every choice that changes its numbers. |
| `scope` | object | one of [`heads`](#scope-heads), [`layer_position`](#scope-layer_position), [`layer_components`](#scope-layer_components), [`sites`](#scope-sites), [`features`](#scope-features), [`site_sets`](#scope-site_sets), [`head_inputs`](#scope-head_inputs), [`neurons`](#scope-neurons), [`graph`](#scope-graph) or [`edges`](#scope-edges) ([Scope](#scope)) | The sites the method measures: one result per site. |
| `metric` | object | one of [`logit_diff`](#metric-logit_diff), [`logprob_diff`](#metric-logprob_diff), [`logprob`](#metric-logprob), [`prob`](#metric-prob), [`prob_diff`](#metric-prob_diff) or [`kl`](#metric-kl) ([Metric](#metric)) | What is read from the model's output, and how effects are normalized. |
| `statistics` | object | [Statistics](#statistics) | The intervals' bootstrap, confidence level and seed. |
| `execution` | object | [Execution](#execution) | How the work is batched. |
| `predictions` (optional, default `null`) | object or null | [Prediction settings](#prediction-settings), or `null` | An optional logit lens saved beside the results; it never changes an intervention. |
| `sae` (optional, default `null`) | object or null | [SAE](#sae), or `null` | The SAE or transcoder that feature sites, the features scope and steering along a feature refer to; `null` otherwise. |

## Model

In a spec at `model`.

The model, exactly: which files, in which precision, on which device.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `id` | string | 1 to 200 characters | The Hugging Face model id, or for a local model the name it was added under |
| `source` | string | `"hub"` or `"local"` | Where the weights come from: Hugging Face, or a folder on this machine that was added to Logogram's local models, found by the hash of its files |
| `revision` | string or null | any | Exact commit. `null` resolves the current main branch at run time. Always `null` for a local model. |
| `sha256` | string or null | any | For a local model, the hash of its files, which finds its folder on any machine it was added to. `null` for Hugging Face, where the revision pins the files. |
| `dtype` | string | `"float32"`, `"float16"` or `"bfloat16"` | The precision the model runs in. float16 and bfloat16 halve the memory, but round. |
| `device` | string | any | auto, cpu, cuda, cuda:N (one GPU of several) or mps |
| `process_weights` | boolean | `true` or `false` | Fold LayerNorm weights and center writing weights and the unembedding, as TransformerLens does by default. Logit differences are unchanged by this. |

## Dataset

In a spec at `dataset`.

The prompt pairs: a JSONL file in the project, one pair per line.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `path` | string | not empty | Project-relative path to a JSONL dataset |
| `sha256` | string or null | any | Expected hash of the file, if pinned |
| `limit` | integer or null | 1 or more, or `null` | Use only the first n prompts |

## Tokenization

In a spec at `tokenization`.

How each prompt becomes the tokens the model reads.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `prepend_bos` | boolean | `true` or `false` | Put the beginning-of-sequence token before each prompt. `false` with a chat template, which writes the beginning itself, and for tokenizers that have no such token. |
| `chat_template` | object or null | one of [`user_turn`](#chat-template-user_turn) or [`assistant_prefix`](#chat-template-assistant_prefix) ([Chat template](#chat-template)), or `null` | Wrap each prompt in the model's own chat template (the one its tokenizer ships at the model's revision), as instruct models are used; `null` feeds the raw text. |

## Experiment

In a spec at `experiment`.

One of these, chosen by `kind`: [`activation_patching`](#experiment-activation_patching), [`ablation`](#experiment-ablation), [`direct_logit_attribution`](#experiment-direct_logit_attribution), [`attribution_patching`](#experiment-attribution_patching), [`steering`](#experiment-steering), [`path_patching`](#experiment-path_patching) or [`edge_attribution`](#experiment-edge_attribution).

### Experiment: activation_patching

Copy activations from one prompt of each pair into the run on the other.

`clean_to_corrupt` runs the corrupt prompt and patches in clean activations
("does this restore the behavior?"). `corrupt_to_clean` runs the clean prompt and patches
in corrupt activations ("does this break it?").

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"activation_patching"` | Which experiment this is: always `activation_patching` here. |
| `direction` | string | `"clean_to_corrupt"` or `"corrupt_to_clean"` | `clean_to_corrupt` runs the corrupt prompt with clean activations (does this restore the behavior?); `corrupt_to_clean` runs the clean prompt with corrupt ones (does this break it?). |

### Experiment: ablation

Run the clean prompt and replace the activation with a baseline value.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"ablation"` | Which experiment this is: always `ablation` here. |
| `baseline` | object | one of [`zero`](#baseline-zero), [`mean`](#baseline-mean) or [`resample`](#baseline-resample) ([Baseline](#baseline)) | What replaces the activation. It changes the result, so there is no default. |

### Experiment: direct_logit_attribution

Split the logit difference of the chosen prompts into what each component writes directly.

At the last position, the residual stream is the sum of the embeddings and every attention and
MLP output. With the final normalization's scale held at its value in the run, the logit
difference is a sum of one term per component plus a constant from biases. A component's term
is its direct effect: it leaves out everything the component does through later components.
This is a decomposition of one forward pass, not an intervention.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"direct_logit_attribution"` | Which experiment this is: always `direct_logit_attribution` here. |
| `prompts` | string | `"clean"` or `"corrupt"` | Whose logit difference is split: the clean prompts' or the corrupt prompts'. |

### Experiment: attribution_patching

Estimate activation patching at every site from gradients (attribution patching).

For each site, the change patching would cause is estimated as (source activation - receiver
activation) · a gradient of the metric. With `method="gradient"` the gradient is taken at the
receiver run: one forward and backward pass per batch of prompts instead of one patched run
per site. It is a first-order estimate: it misses saturation (in attention, normalization and
the softmax) and can miss or even invert an effect.

With `method="integrated_gradients"` the gradient is averaged over `steps` runs whose input
embeddings lie evenly between the receiver's and the source's (midpoints of equal intervals),
as in EAP-IG. It follows the metric along the way from one prompt to the other, which corrects
much of the saturation a single gradient misses, at `steps` times the cost. Verify the
strongest sites with activation patching either way.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"attribution_patching"` | Which experiment this is: always `attribution_patching` here. |
| `direction` | string | `"clean_to_corrupt"` or `"corrupt_to_clean"` | `clean_to_corrupt` runs the corrupt prompt with clean activations (does this restore the behavior?); `corrupt_to_clean` runs the clean prompt with corrupt ones (does this break it?). |
| `method` | string | `"gradient"` or `"integrated_gradients"` | One gradient at the receiver run, or gradients averaged along the way between the two prompts' input embeddings (integrated gradients). |
| `steps` | integer or null | 2 to 64, or `null` | The points along the way, for integrated gradients; `null` for a single gradient. |

### Experiment: steering

Add a direction to the residual stream at several strengths, and measure what it does.

The direction is computed on a training split of the prompts and applied to the held-out
rest, so no prompt receives its own difference. The effect is normalized like patching: 1
means the steered prompt moves as far as switching from the receiver (`apply_to`) to the
other prompt. With `control`, a random direction of the same length, at the same strengths,
runs alongside. With `continuations`, the run also records what the model goes on to say.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"steering"` | Which experiment this is: always `steering` here. |
| `apply_to` | string | `"clean"` or `"corrupt"` | The prompts the direction is added to; it steers them toward the other prompts. |
| `direction` | object | one of [`mean_difference`](#steering-direction-mean_difference) or [`sae_feature`](#steering-direction-sae_feature) ([Steering direction](#steering-direction)) | What is added: the mean difference between the prompts of each training pair, or one SAE feature's decoder direction. |
| `coefficients` | list of numbers | 1 to 16 items | The strengths: 1 adds the whole mean difference (or the feature at its typical activation), negative values push the other way. |
| `train_fraction` | number | more than 0 and less than 1 | The share of pairs, drawn with seed, that compute the direction; the rest measure it. |
| `seed` | integer | 0 to 4,294,967,295 | The seed of the split into training and held-out pairs, and of the random control. |
| `control` | boolean | `true` or `false` | Also add a random direction of the same length, at the same strengths. |
| `continuations` | object or null | [Continuations](#continuations), or `null` | Also record what the model goes on to say; `null` records nothing. |

### Experiment: path_patching

Patch a component's effect along chosen paths only (path patching).

For each sender (a site of the scope), the receiver prompt runs with the sender's activation
from the source prompt and every other attention head held at its own value, so the change
travels only through the residual stream (and through the MLPs, unless `freeze_mlps`). The
receivers' inputs from that run are recorded, then patched into an otherwise unchanged
receiver run, and the metric is read there. The result is the sender's effect through those
receivers alone. Senders must come before at least one receiver.

With an `edges` scope, each edge is its own result: its sender's effect through its own
receiver alone, with every head and MLP held (`freeze_mlps`), so the change reaches the
receiver only directly, as the edge carries it. `receivers` then lists each edge's receiver.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"path_patching"` | Which experiment this is: always `path_patching` here. |
| `direction` | string | `"clean_to_corrupt"` or `"corrupt_to_clean"` | `clean_to_corrupt` runs the corrupt prompt with clean activations (does this restore the behavior?); `corrupt_to_clean` runs the clean prompt with corrupt ones (does this break it?). |
| `receivers` | list of objects | 1 to 64 items; each one of [`head`](#path-receiver-head), [`logits`](#path-receiver-logits) or [`mlp`](#path-receiver-mlp) ([Path receiver](#path-receiver)) | Where the paths end: later heads' queries, keys or values, or the logits. |
| `freeze_mlps` | boolean | `true` or `false` | Hold the MLPs at their own values too, so the change travels through the residual stream alone. |

### Experiment: edge_attribution

Estimate the effect of every edge of the model's graph from gradients (edge attribution
patching, EAP, as in Syed et al. 2023; with integrated gradients, EAP-IG, as in Hanna et al.
2024).

The estimate for the edge from u to v is (u's output on the source prompt − on the receiver
prompt) · the gradient of the metric with respect to v's own input, summed over positions.
v's own input is the copy of the residual stream that only v reads, before v's normalization;
the gradient goes through that normalization as the model computes it, its scale recomputed
from the input (see logogram.edges). With `method="integrated_gradients"` the gradient is
averaged over `steps` runs whose input embeddings lie evenly between the receiver's and the
source's, as in attribution patching. With `curve`, the strongest edges are then patched
together, edge by edge, for a faithfulness curve beside edges chosen at random.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"edge_attribution"` | Which experiment this is: always `edge_attribution` here. |
| `direction` | string | `"clean_to_corrupt"` or `"corrupt_to_clean"` | `clean_to_corrupt` runs the corrupt prompt with clean activations (does this restore the behavior?); `corrupt_to_clean` runs the clean prompt with corrupt ones (does this break it?). |
| `method` | string | `"gradient"` or `"integrated_gradients"` | One gradient at the receiver run, or gradients averaged along the way between the two prompts' input embeddings (integrated gradients). |
| `steps` | integer or null | 2 to 64, or `null` | The points along the way, for integrated gradients; `null` for a single gradient. |
| `curve` | object or null | [Faithfulness curve](#faithfulness-curve), or `null` | Patch the strongest edges together for a faithfulness curve; `null` to only estimate. |

## Scope

In a spec at `scope`.

One of these, chosen by `kind`: [`heads`](#scope-heads), [`layer_position`](#scope-layer_position), [`layer_components`](#scope-layer_components), [`sites`](#scope-sites), [`features`](#scope-features), [`site_sets`](#scope-site_sets), [`head_inputs`](#scope-head_inputs), [`neurons`](#scope-neurons), [`graph`](#scope-graph) or [`edges`](#scope-edges).

### Scope: heads

Every attention head in every layer, at one position: a layer x head grid.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"heads"` | Which scope this is: always `heads` here. |
| `position` | object | one of [`all`](#position-all), [`last`](#position-last), [`index`](#position-index) or [`label`](#position-label) ([Position](#position)) | The token or tokens of each prompt where every head is intervened on. |

### Scope: layer_position

One stream site at every layer and every position: a layer x position grid.

`positions="each"` uses every token index and needs prompts of equal length.
`positions="labels"` uses the dataset's named positions, which works across templates.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"layer_position"` | Which scope this is: always `layer_position` here. |
| `site` | string | `"resid_pre"`, `"resid_mid"`, `"resid_post"`, `"attn_out"` or `"mlp_out"` | The site measured at every layer and position. |
| `positions` | string | `"each"` or `"labels"` | `each`: every token index (prompts of one length); `labels`: the named positions. |

### Scope: layer_components

Attention and MLP outputs of every layer, at one position: a layer x component grid.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"layer_components"` | Which scope this is: always `layer_components` here. |
| `components` | list of strings | each `"resid_pre"`, `"resid_mid"`, `"resid_post"`, `"attn_out"` or `"mlp_out"` | The sites measured at every layer, each once, such as `attn_out` and `mlp_out`. |
| `position` | object | one of [`all`](#position-all), [`last`](#position-last), [`index`](#position-index) or [`label`](#position-label) ([Position](#position)) | The token or tokens of each prompt intervened on. |

### Scope: sites

An explicit list of sites, for example a single head chosen on the map.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"sites"` | Which scope this is: always `sites` here. |
| `sites` | list of objects | not empty; each [Site](#site) | The sites, each intervened on alone: one result per site. |

### Scope: features

Every feature of the spec's SAE, at one position (or summed over all of them).

Only attribution patching can sweep every feature: it estimates them all from one gradient
and keeps the `top` with the largest estimated effects (by magnitude) as the run's sites,
ready to verify by patching.

Choosing the strongest features and reporting them on the same prompts biases their effects
away from zero (the winner's curse). With `choose_on`, a seeded split of the prompts
chooses the features and only the other prompts report them. `None` chooses and reports on
every prompt.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"features"` | Which scope this is: always `features` here. |
| `position` | object | one of [`all`](#position-all), [`last`](#position-last), [`index`](#position-index) or [`label`](#position-label) ([Position](#position)) | The token of each prompt where the features are read, or all of them, summed. |
| `top` | integer | 1 to 500 | How many features, largest estimated effects first, the run keeps as its sites. |
| `choose_on` | number or null | more than 0 and less than 1, or `null` | The share of prompts that choose the features (the rest report them), or `null` for all. |
| `seed` | integer or null | 0 to 4,294,967,295, or `null` | The seed of the split into prompts that choose and prompts that report; `null` without it. |

### Scope: site_sets

Sets of sites, each intervened on at once: one result per set.

A circuit is evaluated with a few sets: the circuit alone (`complement`: everything else is
replaced), the circuit removed, and everything removed. Each set's effect is also reported as
a share of the set that replaces the whole universe, when the scope has one.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"site_sets"` | Which scope this is: always `site_sets` here. |
| `universe` | list of strings or null | each `"head"`, `"attn_out"`, `"mlp_out"`, `"q"`, `"k"`, `"v"` or `"neuron"`, or `null` | The components, at every layer and position, that make up the rest of the model for complement sets; `null` when no set is one. |
| `sets` | list of objects | 1 to 64 items; each [Site set](#site-set) | The sets, each intervened on at once: one result per set. |

### Scope: head_inputs

Every head's query, key or value in every layer, at one position: a layer x head grid.

Keys and values belong to key/value heads: with grouped-query attention a model has fewer of
them than query heads, and each is read by every query head of its group, so the grid of
`input="k"` or `"v"` has a column per key/value head.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"head_inputs"` | Which scope this is: always `head_inputs` here. |
| `input` | string | `"q"`, `"k"` or `"v"` | The input intervened on: each query head's query (q), or each key/value head's key (k) or value (v). |
| `position` | object | one of [`all`](#position-all), [`last`](#position-last), [`index`](#position-index) or [`label`](#position-label) ([Position](#position)) | The token or tokens of each prompt whose query, key or value is intervened on. |

### Scope: neurons

Every neuron of the MLPs of the chosen layers, at one position. A neuron's activation is
the MLP's hidden activation after its nonlinearity (for a gated MLP, the gate's activation
times the up projection): the input to the MLP's output projection.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"neurons"` | Which scope this is: always `neurons` here. |
| `layers` | list of integers | 1 to 512 items; each 0 or more | The layers whose MLP neurons are measured, each listed once. |
| `position` | object | one of [`all`](#position-all), [`last`](#position-last), [`index`](#position-index) or [`label`](#position-label) ([Position](#position)) | The token or tokens of each prompt where every neuron is intervened on. |

### Scope: graph

The edges of the model's graph, for edge attribution.

The graph's nodes are the embeddings, every attention head (its output, z through its slice of
the output projection), every MLP and the logits. An edge runs from a node's output to a later
node's input: a head's query, key or value input (each its own edge), an MLP's input, or the
logits. `layers` chooses the heads and MLPs whose inputs receive the edges (from every
earlier node), `null` for every layer; `logits` adds the edges into the logits.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"graph"` | Which scope this is: always `graph` here. |
| `layers` | list of integers or null | each 0 or more, or `null` | The layers whose heads and MLPs receive edges, in order; `null` for every layer. |
| `logits` | boolean | `true` or `false` | Include the edges into the logits. |

### Scope: edges

Edges measured one at a time, each its own result: path patching from the edge's sender
into its receiver alone, with every head and MLP held (see PathPatching).

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"edges"` | Which scope this is: always `edges` here. |
| `edges` | list of objects | 1 to 64 items; each [Edge](#edge) | The edges, each measured on its own, none repeated. |

## Metric

In a spec at `metric`.

One of these, chosen by `kind`: [`logit_diff`](#metric-logit_diff), [`logprob_diff`](#metric-logprob_diff), [`logprob`](#metric-logprob), [`prob`](#metric-prob), [`prob_diff`](#metric-prob_diff) or [`kl`](#metric-kl).

### Metric: logit_diff

logit(answer) - logit(distractor) at the last position. Answers and distractors are single
tokens (or sets of single tokens, read through the log of their summed probability).

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"logit_diff"` | Which metric this is: always `logit_diff` here. |
| `normalization` | string | `"dataset_gap"` or `"prompt_gap"` | What an effect is divided by: the gap between the two prompts' metric, averaged over the dataset (`dataset_gap`) or each prompt's own (`prompt_gap`). |

### Metric: logprob_diff

log P(answer) - log P(distractor): with answers of several tokens, each continuation's
log-probability is the sum over its tokens, each predicted from the prompt and the tokens
before it. For single tokens it equals the logit difference.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"logprob_diff"` | Which metric this is: always `logprob_diff` here. |
| `normalization` | string | `"dataset_gap"` or `"prompt_gap"` | What an effect is divided by: the gap between the two prompts' metric, averaged over the dataset (`dataset_gap`) or each prompt's own (`prompt_gap`). |

### Metric: logprob

log P(answer): the answer's log-probability (summed over its tokens).

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"logprob"` | Which metric this is: always `logprob` here. |
| `normalization` | string | `"dataset_gap"` or `"prompt_gap"` | What an effect is divided by: the gap between the two prompts' metric, averaged over the dataset (`dataset_gap`) or each prompt's own (`prompt_gap`). |

### Metric: prob

P(answer): the answer's probability (the product over its tokens; the sum over a set).

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"prob"` | Which metric this is: always `prob` here. |
| `normalization` | string | `"dataset_gap"` or `"prompt_gap"` | What an effect is divided by: the gap between the two prompts' metric, averaged over the dataset (`dataset_gap`) or each prompt's own (`prompt_gap`). |

### Metric: prob_diff

P(answer) - P(distractor), as in the greater-than task's probability difference.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"prob_diff"` | Which metric this is: always `prob_diff` here. |
| `normalization` | string | `"dataset_gap"` or `"prompt_gap"` | What an effect is divided by: the gap between the two prompts' metric, averaged over the dataset (`dataset_gap`) or each prompt's own (`prompt_gap`). |

### Metric: kl

KL(P_target || P): how far the next-token distribution at the last position is from the
`target` prompt's own. It reads the whole distribution, not only the answer.

Normalized like the others, it becomes the share of the divergence an intervention closes or
opens: with the target being the source prompt, 1 means the patched run predicts exactly what
the source prompt predicts.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"kl"` | Which metric this is: always `kl` here. |
| `target` | string | `"clean"` or `"corrupt"` | The prompt whose own next-token distribution the divergence is measured from. |
| `normalization` | string | `"dataset_gap"` or `"prompt_gap"` | What an effect is divided by: the gap between the two prompts' metric, averaged over the dataset (`dataset_gap`) or each prompt's own (`prompt_gap`). |

## Statistics

In a spec at `statistics`.

Percentile bootstrap over prompts. With `cluster`, the bootstrap resamples groups of
prompts that share a value of that `meta` field (for example `template`) instead of single
prompts, since prompts from one template aren't independent.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `bootstrap` | integer | 100 to 100,000 | How many bootstrap resamples the intervals are computed from. |
| `ci` | number | more than 0.5 and less than 1 | The confidence level of the intervals and simultaneous bands, such as 0.95. |
| `seed` | integer | 0 to 4,294,967,295 | Seeds the bootstrap, and Python, NumPy and PyTorch for the whole run. |
| `cluster` | string or null | any | A field of the prompts' meta whose values group them, or `null` to resample prompts. |

## Execution

In a spec at `execution`.

How the work is batched.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `batch_size` | integer | 1 to 4,096 | Rows per forward pass. Part of the spec because batch shape can change floating-point results in the last digits. |

## Prediction settings

In a spec at `predictions`.

An optional, explicitly configured diagnostic; it never changes an intervention.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `method` | string | `"final_norm_logit_lens"` or `"tuned_lens"` | Each layer's resid_post read through the final normalization and the unembedding (`final_norm_logit_lens`), or first through that layer's affine translator from a tuned lens trained in the project (`tuned_lens`; layers it has no translator for are read as the logit lens reads them). |
| `lens` | object or null | [Lens](#lens), or `null` | The tuned lens, for the tuned_lens method; `null` for the logit lens. |
| `prompt_index` | integer | 0 or more | The prompt pair's index in the dataset, from 0. |
| `which` | string | `"clean"` or `"corrupt"` | The pair's clean or corrupt prompt. |
| `position` | object | one of [`last`](#position-last) or [`index`](#position-index) ([Position](#position)) | The token whose next-token predictions are read. |
| `top_k` | integer | 1 to 20 | How many of the most likely tokens are listed at each layer. |

## SAE

In a spec at `sae`.

A sparse autoencoder (or transcoder): one published on Hugging Face, named by the
repository, the folder holding it (empty for the top level) and the exact commit, pinned when a
run starts; or one trained in the project, named by its folder in `saes/` and pinned by the
hash of its files.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `repo` | string | not empty | The Hugging Face repository, such as jbloom/GPT2-Small-SAEs-Reformatted; for an SAE trained in the project, its name. |
| `path` | string | any | The folder in the repository that holds the SAE, such as blocks.8.hook_resid_pre, empty for the top level; for an SAE trained in the project, saes/ and its name. |
| `revision` | string or null | any | The exact commit; `null` resolves the main branch when a run starts, and the run's spec pins it. Always `null` for an SAE trained in the project. |
| `source` | string | `"hub"` or `"project"` | Where the SAE comes from: Hugging Face, or the project's saes folder (trained there). |
| `sha256` | string or null | any | For an SAE trained in the project, the hash of its cfg.json and weights file (see logogram.sae_training); `null` uses the SAE as it is when a run starts, and the run's spec pins it. Always `null` for Hugging Face, where the revision pins the files. |

## Chat template

In a spec at `tokenization.chat_template`.

One of these, chosen by `kind`: [`user_turn`](#chat-template-user_turn) or [`assistant_prefix`](#chat-template-assistant_prefix).

### Chat template: user_turn

The prompt is the person's message. The template then opens the model's reply, so the
answer is the first token of the reply.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"user_turn"` | Which chat template this is: always `user_turn` here. |
| `system` | string or null | at most 4,000 characters, or `null` | A system message before the prompt, or `null` for none (some templates then write their own). |

### Chat template: assistant_prefix

The prompt is the beginning of the model's own reply to a fixed message from the person,
so the model goes on with it as its reply.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"assistant_prefix"` | Which chat template this is: always `assistant_prefix` here. |
| `system` | string or null | at most 4,000 characters, or `null` | A system message before the person's, or `null` for none (some templates then write their own). |
| `user` | string | 1 to 4,000 characters | The person's message that every prompt replies to. |

## Baseline

In a spec at `experiment.baseline` (experiment `ablation`).

One of these, chosen by `kind`: [`zero`](#baseline-zero), [`mean`](#baseline-mean) or [`resample`](#baseline-resample).

### Baseline: zero

Replace the activation with zeros.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"zero"` | Which baseline this is: always `zero` here. |

### Baseline: mean

Replace the activation with its mean over a stated reference set of prompts.

With an all-positions intervention the mean is taken per position, over reference prompts
of the same token length. With a single position it is taken at each reference prompt's own
resolved position.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"mean"` | Which baseline this is: always `mean` here. |
| `reference` | string | `"clean"` or `"corrupt"` | The prompts the mean is taken over: the dataset's clean or corrupt prompts. |

### Baseline: resample

Replace the activation with the same activation from other prompts (donors).

Each prompt draws `donors` donors from the pool without replacement, never itself, using
`seed`. The patched metric is averaged over donors. The same donors are used at every site.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"resample"` | Which baseline this is: always `resample` here. |
| `pool` | string | `"clean"` or `"corrupt"` | The prompts donors are drawn from: the dataset's clean or corrupt prompts. |
| `donors` | integer | 1 to 1,000 | How many donors each prompt draws. |
| `seed` | integer | 0 to 4,294,967,295 | The seed of the draw. |

## Steering direction

In a spec at `experiment.direction` (experiment `steering`).

One of these, chosen by `kind`: [`mean_difference`](#steering-direction-mean_difference) or [`sae_feature`](#steering-direction-sae_feature).

### Steering direction: mean_difference

At each steered site, the mean difference between the two prompts of each training pair,
from the receiver toward the other prompt. A strength of 1 adds the whole mean difference.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"mean_difference"` | Which steering direction this is: always `mean_difference` here. |

### Steering direction: sae_feature

The decoder direction of one feature of the spec's SAE, added where the SAE reads.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"sae_feature"` | Which steering direction this is: always `sae_feature` here. |
| `feature` | integer | 0 or more | The feature's index in the spec's SAE. |

## Continuations

In a spec at `experiment.continuations` (experiment `steering`).

Greedy continuations of the first held-out prompts, without steering and at each strength
(and with the control, when it runs): what the steered model says, beside what it scores.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `prompts` | integer | 1 to 8 | How many held-out prompts are continued, in the split's order. |
| `tokens` | integer | 1 to 64 | How many tokens each continuation adds, each the most likely one. |

## Path receiver

In a spec at `experiment.receivers[]` (experiment `path_patching`) and `scope.edges[].receiver` (scope `edges`).

One of these, chosen by `kind`: [`head`](#path-receiver-head), [`logits`](#path-receiver-logits) or [`mlp`](#path-receiver-mlp).

### Path receiver: head

A later attention head whose query, key or value input receives the path.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"head"` | Which path receiver this is: always `head` here. |
| `layer` | integer | 0 or more | The receiving head's layer, from 0. |
| `head` | integer | 0 or more | The receiving head's index in its layer. |
| `input` | string | `"q"`, `"k"` or `"v"` | The input the path reaches: the head's query, key or value. |

### Path receiver: logits

The residual stream at the end of the model, read directly by the unembedding.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"logits"` | Which path receiver this is: always `logits` here. |

### Path receiver: mlp

A later MLP's input: the residual stream as that MLP reads it, before its normalization.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"mlp"` | Which path receiver this is: always `mlp` here. |
| `layer` | integer | 0 or more | The receiving MLP's layer, from 0. |

## Faithfulness curve

In a spec at `experiment.curve` (experiment `edge_attribution`).

The k strongest edges as a circuit, for each k of `counts`.

`patch` says which edges carry the source prompt's values in the receiver run: with
`"chosen"`, the chosen edges do and every other edge keeps the receiver's own; with
`"rest"`, the chosen edges keep the receiver's and every other edge carries the source's.
Faithfulness is the share of the gap between the run with no edge chosen and the run with
every edge chosen that the chosen edges close: 0 with none, 1 with every edge of the graph.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `counts` | list of integers | 1 to 24 items; each 1 or more | The numbers of edges chosen, increasing: the curve's points. |
| `patch` | string | `"chosen"` or `"rest"` | Which edges carry the source prompt's values: the chosen ones, or every other one. |
| `control` | object | [Random edges](#random-edges) | The same counts of edges chosen at random, the curve to compare with. |

## Position

In a spec at `scope.position` (scope `heads`, `layer_components`, `features`, `head_inputs` or `neurons`), `predictions.position`, `scope.sites[].position` (scope `sites`) and `scope.sets[].sites[].position` (scope `site_sets`).

One of these, chosen by `kind`: [`all`](#position-all), [`last`](#position-last), [`index`](#position-index) or [`label`](#position-label).

### Position: all

Intervene on every token position at once.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"all"` | Which position this is: always `all` here. |

### Position: last

The final token of each prompt (the position that predicts the answer).

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"last"` | Which position this is: always `last` here. |

### Position: index

A fixed token index. Negative values count from the end (-1 is the last token).

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"index"` | Which position this is: always `index` here. |
| `index` | integer | any | The token's index in each prompt, from 0; negative counts from the end. |

### Position: label

A named position from the dataset, such as `S2` in IOI prompts.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"label"` | Which position this is: always `label` here. |
| `label` | string | not empty | A name in the prompts' `positions`: the last token that overlaps its span. |

## Site

In a spec at `scope.sites[]` (scope `sites`) and `scope.sets[].sites[]` (scope `site_sets`).

One place in the model, at a layer and a position of each prompt.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"resid_pre"`, `"resid_mid"`, `"resid_post"`, `"attn_out"`, `"mlp_out"`, `"head"`, `"sae_feature"`, `"q"`, `"k"`, `"v"` or `"neuron"` | A residual stream state, an attention or MLP output, one head's output (z), a head's query, key or value, one MLP neuron, or a feature of the spec's SAE. |
| `layer` | integer | 0 or more | The layer, from 0. |
| `head` (optional, default `null`) | integer or null | 0 or more, or `null` | The head's index in its layer: the head of a head site (its output z), the query head of a q site, the key/value head of a k or v site; `null` for any other site. |
| `feature` (optional, default `null`) | integer or null | 0 or more, or `null` | The feature's index in the spec's SAE, for an sae_feature site; `null` for any other. |
| `neuron` (optional, default `null`) | integer or null | 0 or more, or `null` | The neuron's index among its layer's MLP neurons, for a neuron site: its activation after the nonlinearity (for a gated MLP, the gate's activation times the up projection), the input to the MLP's output projection. Null for any other site. |
| `position` | object | one of [`all`](#position-all), [`last`](#position-last), [`index`](#position-index) or [`label`](#position-label) ([Position](#position)) | The token or tokens of each prompt intervened on. |

## Site set

In a spec at `scope.sets[]` (scope `site_sets`).

Sites intervened on together, in one forward pass.

With `complement`, the intervention covers every component of the scope's universe except
these sites: a set of heads then reads as "keep this circuit, replace the rest". A site at one
position keeps only that position of its component; the component's other positions are
replaced with the rest. A complement set with no sites replaces the whole universe.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `label` | string | 1 to 80 characters | The set's name in the results. |
| `sites` | list of objects | at most 4,096 items; each [Site](#site) | The sites intervened on together, or, with complement, the sites kept. |
| `complement` | boolean | `true` or `false` | Intervene on every component of the scope's universe except these sites. |

## Edge

In a spec at `scope.edges[]` (scope `edges`).

A sender's output read directly by one receiver's input, through the residual stream.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `sender` | object | [Edge sender](#edge-sender) | Where the edge starts. |
| `receiver` | object | one of [`head`](#path-receiver-head), [`logits`](#path-receiver-logits) or [`mlp`](#path-receiver-mlp) ([Path receiver](#path-receiver)) | Where the edge ends: a later head's query, key or value, a later MLP's input, or the logits. |

## Lens

In a spec at `predictions.lens`.

A tuned lens trained in the project (its folder in `lenses/`), pinned by the hash of its
translators' file, lens.safetensors, when a run starts.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `name` | string | any | The lens's name in the project's lenses folder. |
| `sha256` | string or null | any | The hash of its lens.safetensors; `null` uses the lens as it is when a run starts, and the run's spec pins it. |

## Random edges

In a spec at `experiment.curve.control` (experiment `edge_attribution`).

The control: edges chosen at random, `draws` times. Each draw orders every edge of the
graph by a seeded shuffle and takes the first k, as the strongest are taken from the
estimates.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `seed` | integer | 0 to 4,294,967,295 | The seed of the shuffles. |
| `draws` | integer | 1 to 16 | How many random orders are drawn: each point of the control averages them, prompt by prompt, and keeps each draw's own faithfulness. |

## Edge sender

In a spec at `scope.edges[].sender` (scope `edges`).

The node an edge starts from: the embeddings, an attention head's output, or an MLP's.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"embed"`, `"head"` or `"mlp"` |  |
| `layer` (optional, default `null`) | integer or null | 0 or more, or `null` | The head's or MLP's layer, from 0; `null` for the embeddings. |
| `head` (optional, default `null`) | integer or null | 0 or more, or `null` | The head's index in its layer; `null` for the embeddings and for an MLP. |
