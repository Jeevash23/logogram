<!-- Generated from src/logogram/sae_training.py and src/logogram/lenses.py by scripts/build_reference.py. Don't edit it by hand: run `uv run python scripts/build_reference.py`. -->

# Training specs

An SAE or a tuned lens trained in a project is trained from a training spec, which the app, `logogram sae train` and `logogram lens train`, and the Python API (`lg.train_sae`, `lg.train_lens`) all train the same way. As in an experiment spec, every choice that can change the weights is stated, with no default, and a spec that leaves one out is refused with the field's name. The spec as it ran (with the corpus's hash and the model's revision pinned) is saved beside what it trained, as training.json. This page lists every field, from the models in [`src/logogram/sae_training.py`](https://github.com/Jeevash23/logogram/blob/main/src/logogram/sae_training.py) and [`src/logogram/lenses.py`](https://github.com/Jeevash23/logogram/blob/main/src/logogram/lenses.py). What is computed is described in [Training an SAE](../concepts/sae-training.md) and [The tuned lens](../concepts/tuned-lens.md).

## SAE training spec

An SAE trained in the project: what it reads, from which texts, and every choice of its
training. Nothing that can change its weights has a default.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `logogram_sae_training` | integer | `1` | The training spec's version. |
| `name` | string | 1 to 80 characters | The SAE's name: it is saved as saes/NAME, never replacing another. |
| `notes` (optional, default `""`) | string | any | What it is for, kept in its training.json. |
| `model` | object | [Model](#model) | The model whose activations it is trained on, exactly. |
| `corpus` | object | [Corpus](#corpus) | The texts it is trained on, and held out from them, measured on. |
| `tokenization` | object | [Text tokenization](#text-tokenization) | How each text becomes tokens. |
| `site` | string | `"resid_pre"`, `"resid_post"`, `"attn_out"` or `"mlp_out"` | The activation it reads and writes: the residual stream before or after the layer, or the layer's attention or MLP output. |
| `layer` | integer | 0 or more | The layer, from 0. |
| `architecture` | object | [Top K](#top-k) | How it chooses the features that fire. |
| `d_sae` | integer | 1 to 2,097,152 | How many features it has. |
| `steps` | integer | 1 to 10,000,000 | How many steps of the optimizer it takes. |
| `batch_tokens` | integer | 1 to 1,048,576 | How many tokens' activations each step reads. |
| `lr` | number | more than 0 and at most 1 | Adam's learning rate, the same at every step. |
| `betas` | list of 2 numbers | each at least 0 and less than 1 | Adam's coefficients for its running averages of the gradient and of its square. |
| `eps` | number | more than 0 and at most 1 | Adam's term added to the denominator, for numerical stability. |
| `seed` | integer | 0 to 4,294,967,295 | Seeds the split into training and held-out texts, the order the texts are read in, the initial weights and the order of the tokens. |
| `held_out` | number | more than 0 and less than 1 | The share of the corpus's texts, drawn with the seed, held out from training and measured on. |
| `normalize` | string | `"mean_squared_norm"` or `"none"` | `mean_squared_norm`: train on the activations times one factor, measured on the training tokens, that makes their mean squared norm d_in (it is folded into the saved weights and recorded); none: train on the activations as they are. |
| `auxk` | object or null | [Aux K](#aux-k), or `null` | The auxiliary loss for dead latents; `null` trains without it. |
| `log_every` | integer | 1 or more | How often (in steps) the loss curve records the mean of the steps since its last point. |
| `model_batch` | integer | 1 to 1,024 | Texts per forward pass of the model. Part of the spec because batch shape can change floating-point results in the last digits. |

## Lens training spec

A tuned lens trained in the project: on which model and texts, which layers, and every
choice of its training. Nothing that can change its translators has a default.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `logogram_lens_training` | integer | `1` | The training spec's version. |
| `name` | string | 1 to 80 characters | The lens's name: it is saved as lenses/NAME, never replacing another. |
| `notes` (optional, default `""`) | string | any | What it is for, kept in its training.json. |
| `model` | object | [Model](#model) | The model it reads, exactly. |
| `corpus` | object | [Corpus](#corpus) | The texts it is trained on, and held out from them, measured on. |
| `tokenization` | object | [Text tokenization](#text-tokenization) | How each text becomes tokens. |
| `layers` | list of integers or null | not empty; each 0 or more, or `null` | The layers that get a translator, in order; `null` for every layer but the last (whose readout is the model's own prediction). |
| `steps` | integer | 1 to 10,000,000 | How many steps of the optimizer it takes. |
| `batch_tokens` | integer | 1 to 1,048,576 | How many tokens each step reads. |
| `lr` | number | more than 0 and at most 1 | The optimizer's learning rate, the same at every step. |
| `betas` | list of 2 numbers | each at least 0 and less than 1 | Adam's coefficients for its running averages of the gradient and of its square. |
| `eps` | number | more than 0 and at most 1 | Adam's term added to the denominator, for numerical stability. |
| `weight_decay` | number or null | more than 0 and at most 1, or `null` | Decoupled weight decay (AdamW) on each translator's change from the identity, pulling it towards the logit lens; `null` for none. |
| `seed` | integer | 0 to 4,294,967,295 | Seeds the split into training and held-out texts and the order the texts are read in. |
| `held_out` | number | more than 0 and less than 1 | The share of the corpus's texts, drawn with the seed, held out from training and measured on. |
| `log_every` | integer | 1 or more | How often (in steps) the loss curve records the mean of the steps since its last point. |
| `model_batch` | integer | 1 to 1,024 | Texts per forward pass of the model. Part of the spec because batch shape can change floating-point results in the last digits. |

## Model

In a training spec at `model`.

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

## Corpus

In a training spec at `corpus`.

The texts trained on: a corpus in the project's corpora folder.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `name` | string | 1 to 80 characters | The corpus's name, as corpora/NAME.jsonl. |
| `sha256` | string or null | any | The hash of its file; `null` uses the corpus as it is when training starts, and the training.json written beside the result pins it. |

## Text tokenization

In a training spec at `tokenization`.

How each text becomes the tokens the model reads.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `prepend_bos` | boolean | `true` or `false` | Put the beginning-of-sequence token before each text. Its own position, the same in every text and unlike any other, is never trained on or measured. |
| `max_tokens` | integer | 2 to 4,096 | The most tokens read of each text (the beginning-of-sequence token included); the rest of a longer text is left out. |

## Top K

In a training spec at `architecture`.

A TopK SAE (Gao et al. 2024): each token keeps its `k` largest pre-activations, each
through a ReLU, and every other feature is zero.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `kind` | string | `"topk"` | The architecture: TopK is the one Logogram trains. |
| `k` | integer | 1 or more | How many features each token keeps. |

## Aux K

In a training spec at `auxk`.

The auxiliary loss for dead latents of Gao et al. (2024): the `k` dead latents with the
largest pre-activations reconstruct what the live ones missed, so they get gradient again.

| Field | Type | Allowed values | Description |
|---|---|---|---|
| `k` | integer | 1 or more | How many dead latents reconstruct the error (at most as many as are dead). |
| `coefficient` | number | more than 0 and at most 10 | What the auxiliary loss is multiplied by before it is added (Gao et al. use 1/32). |
| `dead_after` | integer | 1 or more | A latent is dead once it hasn't fired on this many training tokens. |
