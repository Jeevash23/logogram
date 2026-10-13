# SAE features and transcoders

A component's activation mixes many things the model represents at once. A **sparse autoencoder**
(SAE) rewrites one of the model's activations as a few active **features** out of thousands, each
a direction in the model, plus an **error** it misses (Bricken et al., 2023; Cunningham et al.,
2023). Features are often easier to read than neurons or heads, and Logogram treats them as sites
like any other: they can be read on prompts, patched, estimated and steered along.

## Finding a published SAE

People name published SAEs by their SAELens release and id, such as `gpt2-small-res-jb` and
`blocks.8.hook_resid_pre`. Logogram ships a catalogue of SAELens's registry: the 13,857 SAEs in
the 170 releases of SAELens 6.54.0, each with the model it was trained on, its Hugging Face
repository and folder, the hook it reads, its width, the architecture its configuration states,
and whether Logogram reads it exactly. The catalogue is part of the package and is never fetched:
`scripts/sae_catalogue.py` builds it from a sae-lens wheel, and SAELens's MIT notice travels with
it. Nothing downloads until an SAE is loaded.

* In **Explore → Features**, **Published SAEs** lists those trained on the loaded model, the ones
  Logogram reads exactly first. Search by release, hook, site, layer (`layer 8`) or width (`16k`),
  choose one to see where it is published, what it reads, its licence and how it was checked, and
  press **Load this SAE**. **Or name a repository and folder** loads any other.
* On the command line, `logogram saes MODEL` lists a model's SAEs by its Hugging Face id, with
  words to search for after it (`logogram saes openai-community/gpt2 layer 8 resid_pre`),
  `--exact` for those read exactly and `--json` for every field.
* In Python, `lg.list_saes(model, "layer 8")` returns them as a table, and
  `lg.load_sae(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", model=model)`
  loads one; `lg.load_sae(repo, path, model=model)` loads one by its repository and folder.

Each SAE's `reads` says what Logogram makes of it:

* `exact`: Logogram's own readers, run on its configuration, read it exactly. `verified`, when
  it was checked on its files at one commit, says how: its weights file's header, every
  parameter's name and shape as Logogram reads them (`"read": "header"`), or its files read whole
  (`"weights"`), with its `fit` when it was also run on its model: the share of the activations'
  variance it explains and how many features fire per token, over plain text.
* `refused`: Logogram doesn't read it, and `reason` says why. Most are the first Gemma Scope's
  NumPy archives, PyTorch pickles (which Logogram never opens), formats Logogram doesn't read yet
  (Llama Scope's, circuit-tracer's transcoders) and cross-layer transcoders. The app doesn't offer
  to load them.
* `unknown`: its configuration couldn't be read when the catalogue was made, so Logogram checks it
  when it loads.

Logogram reads 9,035 of the 13,857 exactly: GPT-2 small's SAELens SAEs (Joseph Bloom's, the
feature-splitting ones and Apollo Research's), OpenAI's TopK SAEs on it but for those between
attention and MLP, Pythia-70m's, Gemma 2B's, the matryoshka SAEs for Gemma 2 and Gemma 3 1B, and
all of Gemma Scope 2's SAEs and transcoders. 424 SAEs across every release it reads were checked
by their weights file's header, and 16, on GPT-2 small, Pythia-70m and Gemma 3 270M, by their
weights and their fit to the model. An SAE that SAELens's registry reads with settings its own
configuration doesn't state (Mistral 7B's, which SAELens rescales) is refused before anything
downloads, also when it is named by its repository and folder: its files alone don't say what it
computes. However an SAE is chosen, a spec records its repository, folder and revision, with
`source` `hub`, never its release and id.

## Loading an SAE

Whether chosen from the catalogue or named by its repository and folder, an SAE downloads once,
at the repository's current commit unless a revision is given, and Logogram reads:

* the SAELens format, the EleutherAI (sparsify) format, and Gemma Scope 2 (`config.json` and
  `params.safetensors`, with the JumpReLU activation of Rajamanoharan et al., 2024), including
  TopK SAEs (Gao et al., 2024);
* from safetensors files only. The first Gemma Scope's NumPy archives (Lieberum et al., 2024) are
  refused, as is anything whose input scaling, architecture, settings or tensors Logogram can't
  reproduce exactly, such as sparsify's group-max activation.

An SAE made for another model's sizes, or for a site the model doesn't have, is refused with the
reason. Its revision is pinned when a run starts, like a model's.

**Measure the fit** before trusting any feature. An SAE only fits activations like those it was
trained on, so Logogram measures, on your own prompts: the fraction of the activations' variance
it explains, how many features fire per token, and the logit difference with the SAE's
reconstruction spliced in place of the activation. A poor fit, for example with
TransformerLens's weight processing set the other way from training, shows before anything else.
The fit is the last check: a file can be trained on something its configuration doesn't record.

## SAEs trained in the project

When no published SAE fits the model, the site or the text, Logogram trains a TopK SAE in the
project, on a corpus of its texts, deterministically: **Train an SAE…** in Features, `logogram
sae train` or `lg.train_sae` (see [training an SAE](sae-training.md) and the
[guide](../guides/train-an-sae.md)). It is saved as `saes/NAME` in SAELens's format, read by the
same reader as published SAEs, and listed under **Trained in this project**. A spec names it with
`source` `project`, its name and folder, and `sha256`, the hash of its files, which a run pins when
it starts. It is read from the project's files, never downloaded, and one trained on another model
loads with a warning.

## Transcoders

A **transcoder** reads an MLP's input and writes its output (Dunefsky, Chlenski and Nanda, 2024):
a feature's value comes from the MLP's input, and patching it changes the MLP's output along the
feature's decoder direction. **Skip transcoders** also add their input times a matrix (Paulo,
Shabalin and Belrose, 2025): EleutherAI's add `x @ W_skip.T`, as sparsify computes it, and Gemma
Scope 2's affine ones add `x @ affine_skip_connection`. Patching a feature leaves the input as it
is, so the skip connection doesn't change; attribution patching reports the part of a site's
estimate that flows through it beside the features' part. sparsify files written before
February 2025 don't record whether they are transcoders, so Logogram reads it from their
training run's `config.json` at the top of the repository.

SAEs on attention heads' outputs (SAELens `hook_z`, Gemma Scope 2's attention SAEs) read the
output `z` of every head of the layer, side by side as the output projection reads it; patching
one of their features changes every head's output. SAEs with a skip connection of their own,
cross-layer transcoders and crosscoders are refused for now.

## Reading features

The Features view lists the features that fire on each token of a prompt and, for one feature,
its activation along the prompt and the prompts where it fires most, all from your own prompts.
**Your text** does the same for any text, and `lg.look(model, text, sae=sae)` from Python. SAEs
aren't trained on the beginning-of-sequence token, whose activations are unlike any other's, so
listings start after it. Descriptions of features live on sites like Neuronpedia, which Logogram
doesn't contact.

**What a feature writes** lists the tokens it promotes and suppresses directly: one unit of the
feature adds its decoder direction to the activation the SAE writes (times the input's spread,
for SAEs that standardize their inputs; through each head's slice of the output projection, for
SAEs on heads' outputs). That direction is added to the residual stream entering the final
normalization at one token, skipping every later layer, and read through the final normalization,
with its scale held at its value there, and the unembedding, before any soft cap on the logits.
The values are each token's logit change per unit of the feature, relative to the mean change
over the vocabulary. It is refused where it isn't defined: when the residual stream couldn't be
verified to be a sum of its components, when the final normalization and unembedding don't
reproduce the model's output, and for SAEs on heads' outputs in models that normalize the
attention output after combining its heads (Gemma 2 and 3, OLMo 2).

## Patching features

A feature is a site of kind `sae_feature`, with its layer and index:

```json
{"kind": "sae_feature", "layer": 8, "feature": 1234, "position": {"kind": "last"}}
```

**Patching a feature** encodes the receiver's activation, changes only that feature to its value
in the source prompt (or to zero, for zero ablation), and keeps the SAE's error: the activation
moves along the feature's decoder direction, and nothing else changes. This is how Marks et al.
(2024) patch features.

**Attribution patching** can sweep every feature at once (the `features` scope), and keep the
`top` strongest as the run's sites, then [verify them by patching](attribution-patching.md). The
results also say how much of the whole site's estimated effect the features account for; the
rest is the SAE's error. Choosing the strongest of thousands of features and reporting them on the
same prompts biases their effects away from zero (the winner's curse), so the scope can choose
them on a seeded share of the prompts and report them only on the rest:

```json
"scope": {"kind": "features", "position": {"kind": "last"}, "top": 50, "choose_on": 0.5, "seed": 0},
"sae": {"repo": "jbloom/GPT2-Small-SAEs-Reformatted", "path": "blocks.8.hook_resid_pre",
        "revision": null, "source": "hub", "sha256": null}
```

**Steering** can add a feature's decoder direction instead of a mean difference: see
[steering along an SAE feature](steering.md#steering-along-an-sae-feature).

## When to use them

Use features to name what a component carries, once patching has found the component: which
features at that site matter for the answer, and what each writes. Check the fit first, read the
error's share of the effect, and verify estimates by patching.

## References

* Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary
  Learning.* Transformer Circuits Thread.
* Cunningham, Ewart, Riggs, Huben and Sharkey (2023). *Sparse Autoencoders Find Highly
  Interpretable Features in Language Models.* ICLR 2024.
* Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike and Wu (2024). *Scaling and
  evaluating sparse autoencoders.* arXiv.
* Rajamanoharan, Lieberum, Sonnerat, Conmy, Varma, Kramár and Nanda (2024). *Jumping Ahead:
  Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders.* arXiv.
* Lieberum et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma
  2.* arXiv.
* Dunefsky, Chlenski and Nanda (2024). *Transcoders Find Interpretable LLM Feature Circuits.*
  NeurIPS.
* Paulo, Shabalin and Belrose (2025). *Transcoders Beat Sparse Autoencoders for
  Interpretability.* arXiv.
* Marks, Rager, Michaud, Belinkov, Bau and Mueller (2024). *Sparse Feature Circuits: Discovering
  and Editing Interpretable Causal Graphs in Language Models.* ICLR 2025.
