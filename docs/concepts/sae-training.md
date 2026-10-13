# Training an SAE

A published SAE fits the model and the data it was trained on. When none fits yours (a fine-tune,
a model nobody has published SAEs for, a site or a domain of text the published ones don't
cover), Logogram trains one in the project, on a corpus of the project's texts, and uses it
everywhere a published SAE is used: reading features, patching them, estimating every feature,
steering along one. [SAE features and transcoders](sae-features.md) explains what an SAE is and
how its features are used; this page says exactly how one is trained.

## What is trained

A **TopK sparse autoencoder** (Gao et al., 2024) on one site of one layer: the residual stream
entering or leaving the layer (`resid_pre`, `resid_post`), or the layer's attention or MLP output
(`attn_out`, `mlp_out`). For an activation `x` it computes

```text
f = TopK(ReLU((x − b_dec) · W_enc + b_enc))      the k largest; every other feature is 0
x̂ = f · W_dec + b_dec                             the reconstruction
```

with `d_sae` features. TopK sets the number of features that fire per token (L0) directly, so
there is no sparsity penalty to tune, and it reads exactly as published TopK SAEs do.

## How it is trained

1. **The texts.** The corpus is read, refused if it has changed since its hash was pinned. A
   seeded share of its texts (`held_out`) is held out: never trained on, only measured on. The
   rest are read in a seeded order. Each text is tokenized (after the beginning-of-sequence token,
   with `prepend_bos`) and cut to `max_tokens` tokens.
2. **The activations.** The model reads the training texts, `model_batch` texts per forward pass,
   until it has the site's activation at `steps × batch_tokens` tokens, or at every training token
   when there are fewer; they are then read several times over (the estimate says how many). The
   beginning-of-sequence token's own position is skipped: its activation is the same in every
   text and unlike any other. Every activation is held in memory, in float32.
3. **Normalization.** With `normalize` `mean_squared_norm`, every activation is multiplied by one
   factor, measured on the training tokens, that makes their mean squared norm `d_in` (as SAELens's
   `expected_average_only_in` does), so learning rates carry over between models and sites.
4. **Initialization.** The decoder's rows are random directions of unit length and the encoder
   is the decoder's transpose (Gao et al.); the encoder's bias is zero, and the decoder's bias is
   the activations' mean (Bricken et al. and Gao et al. take their geometric median, which
   outliers move less).
5. **Steps.** `steps` steps of Adam (`lr`, `betas`, `eps`, the same learning rate at every step),
   each on `batch_tokens` tokens drawn in a seeded order, a new order each time every token has
   been read. Each step minimizes the batch's **fraction of variance unexplained**,
   `Σ‖x − x̂‖² / Σ‖x − x̄‖²` (x̄ the batch's mean). With `auxk`, it adds `coefficient` times the
   **AuxK** loss of Gao et al.: AuxK's own `k` dead latents (those that haven't fired on
   `dead_after` training tokens) with the largest pre-activations reconstruct what the live ones
   missed,
   `Σ‖e − ê‖² / Σ‖x − x̄‖²`, with `e = x − x̂` and `ê` their decoder rows weighted by their
   activations, without a bias. Dead latents get gradient again that way. Before each step the
   gradient's component along each decoder row is projected out, and after it each row is made
   unit length again, so a feature's size lives in its activation, not in its row (Gao et al.).
6. **Saving.** The normalization factor is folded into the weights (the encoder multiplied by it,
   the decoder and its bias divided by it), so the saved SAE reads activations as the model
   computes them, and the factor is recorded.

The loss curve records, every `log_every` steps, the mean of the steps since its last point: the
fraction of variance unexplained, the features that fire per token, the AuxK loss and the share of
latents that are dead.

## What is measured

On the held-out texts, with the SAE exactly as saved (read back as any SAELens SAE is):

* **Variance explained**: one minus the fraction of variance unexplained, `1 − Σ‖x − x̂‖² /
  Σ‖x − x̄‖²`, over every held-out token.
* **L0**: how many features fire per token, on average (at most `k`).
* **Dead share**: the share of features that fire on no held-out token.
* **Loss recovered**: the model's next-token loss over the held-out texts, measured three ways at
  every token the SAE reads that has a next token: as the model runs (clean), with the SAE's
  reconstruction in place of the activation at every token it reads (spliced), and with the
  activation set to zero there. The loss recovered is `(zero − spliced) / (zero − clean)`: 1 when
  splicing the SAE in costs nothing, 0 when it is no better than removing the site.

## Every choice

A training spec (`logogram_sae_training: 1`) states every choice that can change the weights, with
no default; the [training specs](../reference/training.md#sae-training-spec) list each field.

```json
{
  "logogram_sae_training": 1,
  "name": "gpt2-resid_post-6",
  "model": {"id": "openai-community/gpt2", "source": "hub", "revision": null, "sha256": null,
            "dtype": "float32", "device": "auto", "process_weights": true},
  "corpus": {"name": "openwebtext-10k", "sha256": null},
  "tokenization": {"prepend_bos": true, "max_tokens": 128},
  "site": "resid_post",
  "layer": 6,
  "architecture": {"kind": "topk", "k": 32},
  "d_sae": 6144,
  "steps": 2000,
  "batch_tokens": 1024,
  "lr": 0.00033,
  "betas": [0.9, 0.999],
  "eps": 6.25e-10,
  "seed": 0,
  "held_out": 0.1,
  "normalize": "mean_squared_norm",
  "auxk": {"k": 512, "coefficient": 0.03125, "dead_after": 102400},
  "log_every": 20,
  "model_batch": 16
}
```

The app starts from these values for a model (eight times as many features as the residual stream
is wide, 32 per token, Gao et al.'s learning rate scaled by the width, their AuxK), shows every one
and lets each be changed. `model_batch` is part of the spec because the shape of a batch can change
floating-point results in the last digits, as in an experiment spec.

## The files

The SAE is the folder `saes/NAME` in the project, written whole or not at all (a cancelled or
failed training writes nothing), never over another:

* `cfg.json` and `sae_weights.safetensors`: the SAE as SAELens 6 writes a TopK SAE
  (`architecture` `topk`, its `k`, `apply_b_dec_to_input`, `normalize_activations` `none`, the hook
  it reads), float32. Logogram reads it back with the same reader as published SAEs, and SAELens
  can read it too.
* `training.json`: the spec as it ran, with the corpus's hash and the model's revision pinned.
* `metrics.json`: the loss curve, the held-out measurements, the normalization factor, the texts
  and tokens it read, and where it ran (the versions of the libraries and the device), as a run's
  manifest records them.

A spec names it with `source` `project`, its name as `repo`, its folder as `path`, no revision,
and `sha256`, the hash of its two files (the SHA-256 of the hex SHA-256 of `cfg.json` followed by
that of `sae_weights.safetensors`). A run pins the hash when it starts, and refuses an SAE whose
files have changed since:

```json
"sae": {"repo": "gpt2-resid_post-6", "path": "saes/gpt2-resid_post-6", "revision": null,
        "source": "project", "sha256": null}
```

## Reproducibility

The same spec on the same machine gives bit-identical weights. The split comes from the raw PCG64
stream, as runs split prompts, so it is the same on any machine; the order of the texts and tokens
and the initial weights come from seeded generators on the CPU, whatever device the model is on;
the batches' composition is fixed by the spec; the SAE trains in float32; and deterministic
algorithms stay on throughout (a training during which they were turned off is refused). Another
machine, device or version of PyTorch can round differently in the last digits.

## Where it can mislead

* **A good reconstruction isn't an interpretable feature.** Variance explained and loss recovered
  say the SAE describes the activation; they don't say its features mean anything. Read them on
  prompts before naming them, and verify the ones that matter by patching.
* **The corpus is the SAE's world.** Features are learned from the texts given. An SAE trained on
  a few hundred short texts learns their vocabulary, not the model's; one trained on prompts from
  one task fits that task and little else. Measure the fit on the prompts you study (Features,
  **Measure the fit**).
* **A small run under-trains.** Published SAEs read hundreds of millions of tokens. A run of a
  million tokens on a laptop gives coarser features and more dead ones; the curve shows whether it
  was still improving, and the dead share how many features it wasted.
* **The model as it was loaded.** An SAE trained with TransformerLens's weight processing reads
  the processed (centered) residual stream. Loaded on the model with processing the other way, or
  on another model of the same width, it is offered with a warning, and its fit says how much that
  matters.
* **Normalization is folded in.** The decoder's rows have unit length in the normalized space, not
  in the model's: in the saved SAE they are one over the factor long. Strength 1 of steering along
  a feature is its typical activation either way.

## References

* Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike and Wu (2024). *Scaling and
  evaluating sparse autoencoders.* arXiv.
* Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary
  Learning.* Transformer Circuits Thread.
* Cunningham, Ewart, Riggs, Huben and Sharkey (2023). *Sparse Autoencoders Find Highly
  Interpretable Features in Language Models.* ICLR 2024.
* Bloom, Tigges, Duong and Chanin (2024). *SAELens.* Software.
