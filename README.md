# Logogram

A local workbench for causal experiments inside language models.

In *Arrival*, an unfamiliar language is decoded by running small, controlled experiments and
watching what changes. Activation patching works the same way: replace one activation with its
value from a different input, and measure what happens to the output. Logogram makes those
experiments visual, fast and rigorous. Load GPT-2, set up a clean/corrupted prompt pair, sweep
every attention head, watch the results paint onto a map of the model, then click a head and see
the evidence behind its number.

* **A research instrument, not a demo.** Every number is traceable to the method that produced
  it: the intervention, baseline, direction, position, metric, normalization, n and confidence
  interval are always shown.
* **Local-first.** No account, no cloud, no telemetry. Everything runs on your machine.
* **Plain files.** Experiments are folders you can commit, share and rerun.
* **One spec, two front ends.** The app and `logogram run` execute the same `spec.json`, and
  produce identical results on the same machine.

## Install

Logogram needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install logogram
```

To run the code in a clone of this repository instead, install it with `uv tool install .`; the
commands in the table below work the same way with `.` in place of `logogram`. The web app is
prebuilt inside the package, so you never need Node. What changed in each version is in
[CHANGELOG.md](CHANGELOG.md).

PyTorch is chosen per machine:

| Machine | Install | Notes |
|---|---|---|
| Linux with an NVIDIA GPU | `uv tool install logogram` | The default Linux wheels include CUDA. |
| Windows with an NVIDIA GPU | `uv tool install --torch-backend=auto logogram` | Picks the CUDA build that matches your driver. |
| Apple Silicon | `uv tool install logogram` | Runs on the CPU unless you choose MPS (see Models). Use a native arm64 Python, not one under Rosetta. |
| CPU only | `uv tool install --torch-backend=cpu logogram` | Smaller download. GPT-2 small runs well on a CPU. |

Then check the setup:

```bash
logogram doctor
```

It reports the operating system, CPU, memory, GPU, video memory, compute backend and recommended
precision, and gives the exact command to fix common problems, such as an NVIDIA GPU with a
CPU-only PyTorch build.

## Quickstart

```bash
logogram
```

This starts a local server on 127.0.0.1, prints its address and opens your browser.

1. The first time, a system check shows what Logogram found. Choose **Open the example project**.
   Logogram copies it to `Documents/Logogram/ioi-example`, an ordinary project folder.
2. The example contains 32 indirect object identification (IOI) prompts and a ready-made
   experiment: patch each head's output from the clean prompt into the corrupt prompt. Press
   **Run**. GPT-2 small (about 0.5 GB) downloads once from Hugging Face; the sweep then takes
   about a minute on a CPU, a few seconds on a GPU.
3. Results paint onto the model map layer by layer. Click the strongest heads (L9 H9, L10 H7,
   L8 H6 …) to see, in the inspector, the exact intervention, the metric, the confidence
   interval, how many prompts flipped, the distribution of per-prompt effects and the prompts
   with the strongest and weakest effects.

   Each run also gets a **logogram**, a circular ink glyph written by its results: layers run
   clockwise from the top, and the ink swells outward (cobalt) where a layer's strongest effect
   is positive and inward (ochre) where it is negative. A run with a few strong layers reads as
   a heavy, lopsided ring; a null result as a thin, even one. Logograms appear in the history,
   on the results page, and beside the model atlas, where each layer number opens that layer's
   strongest component.
4. Press **A** on a head to see its attention pattern, or right-click any cell for **Patch here**,
   **Ablate here** and **Compare across runs**.
5. Press **Check robustness** to rerun the sweep with a different baseline, direction or donor
   count, or, for a run in float16 or bfloat16, in float32. Logogram reports the rank
   correlation, the overlap of the top components and the components whose conclusion changed,
   and flags them on the map.

Keyboard: arrow keys move across the map, **Ctrl/⌘ K** opens the command palette (type `L9H9` to
jump to a head), **1–7** switch views, **[** and **]** step through prompts, **P**, **B**, **A**
and **C** patch, ablate, show attention or compare the selected component, and **Ctrl/⌘ Enter**
runs the experiment form.

Use **History** to reopen experiments or compare two runs. On a narrow window, **Inspector**
opens a panel that closes with Escape.
Attention heatmaps support arrow keys, Home and End, with a spoken value readout and an optional
values table. Baseline and attention show whether they use the selected run or the experiment
form, including BOS, prompt limit and batch size. Inspecting a saved run requires its model,
revision, dtype, device and weight processing; **Load a model → Use saved experiment model
settings** fills those choices in.

Finished results offer a per-prompt CSV, a PNG heatmap with a symmetric legend, and a copyable or
downloadable methods description. The CSV retains every site/prompt row; text fields that could
be interpreted as spreadsheet formulas are prefixed with an apostrophe.

## Explore, experiment, evidence

The workbench has three workspaces:

* **Explore** opens the model atlas, a layer cutaway, attention patterns, head comparison,
  and layer predictions. The atlas uses the model's real dimensions. Double-click a component
  to open its layer, or choose **Layer explorer** and use the layer rail. The cutaway exposes
  residual inputs and outputs, individual head outputs, the combined attention output, and the
  MLP, connected the way the model was measured to add them when it loaded (one after the
  other, or both from the same input); GPT-2 also shows its layer norms.
* **Experiment** contains prompts, baseline checks, the intervention form, and the complete
  spec. Select a token, choose components, and press **Add to experiment**. The staging tray
  keeps exact sites and positions. **Configure experiment** preserves the source run's
  methodological settings for review. Each staged site is a separate intervention in a sweep;
  the tray does not perform a simultaneous circuit intervention.
* **Evidence** contains the full result heatmap, run comparisons, and research notes. Selecting
  an effect opens its per-prompt evidence and recorded method in the inspector.

Pin up to two heads for **Head comparison**. Both attention maps use the same prompt variant,
analysis context, and 0–1 ink scale. Selecting a query token in either map or the token ribbon
links both readouts. Attention weights describe where a head reads; causal effect and its
confidence interval come from a separate intervention and remain labelled as such.

**Save selection** keeps named sites, exact positions, model settings, source run, and written
observations in `research.json`. Notes can be reopened, edited, or deleted from Evidence.
Revision checks reject conflicting edits from another tab, keeping the unsaved text available
so it can be saved as a separate note. New run summaries retain the model's dimensions and
validated block layout, so saved runs can be explored without loading weights.

### Layer predictions

**Layer predictions** implements a final-norm logit lens: project each layer's `resid_post` at
the chosen token through the loaded model's final normalization and vocabulary projection,
including biases and the logit soft-capping some models apply, then apply softmax. The normalization is recomputed at every
layer, following the distinction explained in the [TransformerLens lens documentation](https://transformerlensorg.github.io/TransformerLens/content/backward_lens.html).
Intermediate projections are descriptive diagnostics, not causal measurements or calibrated
early predictions. No trained lens is fitted. The diagnostic is offered only for models whose
final normalization and projection reproduce their own output when they load (see Models).

Choose clean or corrupt input, a prompt, a last-token or indexed position, and 1–20 top tokens,
then press **Measure predictions**. BOS handling, dataset limit and hash, model revision,
weight processing, dtype, and original length-group batch composition follow the active
analysis context. The trajectory and per-layer table show answer and distractor probabilities,
logit difference, and top projected tokens. At an earlier position, the answer and distractor
are still the dataset's reference tokens, not necessarily the expected next token there.

Measuring adds these settings to the experiment form. The optional spec field is:

```json
"predictions": {
  "method": "final_norm_logit_lens",
  "prompt_index": 0,
  "which": "clean",
  "position": {"kind": "last"},
  "top_k": 5
}
```

Save and run that spec from the app or CLI to write `predictions.json` beside the intervention
results. The shared runner calls the same diagnostic implementation as the preview. Saved
reports can be reopened without model weights. Removing the optional diagnostic in Experiment
leaves the intervention settings intact.

## Command line

```text
logogram                 start the app and open the browser
logogram open PATH       start with a project open
logogram run SPEC.json   run an experiment headlessly and print a summary
logogram doctor          environment and hardware report, with fixes
logogram check-updates   ask PyPI whether a newer Logogram is out
logogram serve --dev     API only, for working on the web app
logogram --version
```

The app listens on port 8765, or the next free one; `--port N` picks another (`--port 0` takes
any free port) and `--no-browser` skips opening the browser.

`logogram run` writes a new experiment folder next to the original. When the spec came from a
finished run, it also checks the new results against the old ones:

```text
Identical to 20261007-203446-which-heads-restore-the-answer: all 4,608 per-prompt values match exactly.
```

## Prompts

A dataset is a JSONL file in the project's `datasets/` folder, one prompt pair per line:

```json
{"clean": "When Mary and John went to the store, John gave a drink to",
 "corrupt": "When Mary and John went to the store, Mary gave a drink to",
 "answer": " Mary", "distractor": " John",
 "positions": {"IO": [5, 9], "S1": [14, 18], "S2": [38, 42], "end": [56, 58]}}
```

* `clean` and `corrupt` must tokenize to the same length, so positions line up.
* `answer` and `distractor` must each be a single token (usually with a leading space).
* `positions` is optional: named character spans in the clean prompt. A label refers to the last
  token that overlaps its span. Named positions let you patch at, say, `S2` across templates of
  different lengths.

The app shows both prompts token by token, highlights where they differ, and refuses to run on
prompts that can't be used, saying which and why. The built-in IOI generator writes ABBA and BABA
prompts from several templates with generic names, with a chosen size and seed, and two
corruptions: `flip` (the repeated name becomes the other name) and `abc` (three unrelated names).

## Experiments

**Activation patching** copies an activation from one prompt of each pair into the run on the
other. *Clean → corrupt* runs the corrupt prompt and patches in clean activations: does this
restore the behavior? *Corrupt → clean* runs the clean prompt and patches in corrupt
activations: does this break it?

**Ablation** runs the clean prompt and replaces the activation with a baseline, which you must
choose:

* *zero*: zeros;
* *mean*: the mean over a stated reference set (the dataset's clean or corrupt prompts), per
  position when every position is replaced;
* *resample*: the same activation in other prompts. Each prompt draws `donors` donors from the
  pool without replacement, never itself, with a seed; the result is averaged over donors, and
  the same donors are used at every site.

**Attribution patching** estimates activation patching at every site at once. The change patching
would cause is estimated to first order, as (source activation − receiver activation) · the
gradient of the logit difference at the receiver run: one forward pass on the source prompts and
one forward and backward pass on the receivers cover a whole sweep, so every head at every
position takes seconds even on larger models. Values are labelled as estimates. A first-order
estimate misses saturation (in attention, normalization and the final softmax) and can miss or
even invert an effect, so **Verify top 10 by patching** on the results page patches the sites
with the largest estimated effects for real and opens the comparison: the rank correlation, and
any estimate whose sign patching confidently reverses. **Check robustness** can also patch the
whole sweep. Do that for residual stream sweeps: an estimate is least reliable where patching
replaces a whole token's representation. In the IOI example on GPT-2 small, patching the residual
stream at the changed name in the first layer restores the whole answer while its estimate is
slightly negative, so verifying the top estimates would never reach it. Head outputs track
patching closely.

**Direct logit attribution** splits the logit difference of the clean or the corrupt prompts
(your choice; there is no default) into what each head, attention output and MLP output writes
directly into the residual stream at the last token. The final normalization is read with its
scale held at its value in the run, so the logit difference is exactly the sum of one term per
component plus the embeddings and biases; the results page shows that split. A component's direct
effect leaves out what it does through later components, which patching measures: **Check
robustness** offers the same components patched, so direct and total effects can be compared
side by side. Nothing is replaced, so there are no sign flips or patched probabilities.
Logogram refuses it where it isn't defined: for residual stream states (no component writes
them), at positions other than the last token, for single heads in models that normalize the
attention output after combining heads (Gemma 2, OLMo 2), and for models that soft-cap their
logits.

**Path patching** measures a component's effect through chosen receivers only: a later head's
query, key or value, or the logits read directly from the final residual stream. For each sender
(a head, attention output or MLP output of the sweep), the receiver prompt runs with the sender's
activation from the source prompt while every other attention head is held at its own value, so the
change travels only through the residual stream (and the MLPs, unless you hold them too). What the
receivers read in that run is recorded and patched into an unchanged receiver run, where the metric
is read. Senders in or after the last receiver's layer have no path, so a sweep keeps only the
layers before it. In models that share keys and values across heads (grouped-query attention),
only queries can be receivers. Tests check that from the last layer to the logits a path is the
whole effect, exactly as patching measures it. **Check robustness** offers holding or releasing
the MLPs, the opposite direction, and the senders' whole effect by patching.

**Steering** adds a direction to the residual stream and measures what it does. At each steered
site (one residual stream state per layer, or chosen ones, at one token of each prompt) the
direction is the mean difference between the two prompts of each pair, from the prompts you steer
toward the other ones. It is computed on a seeded training split of the pairs and measured only on
the held-out rest, so no prompt receives its own difference. Each strength multiplies the whole
mean difference: 1 adds all of it, negative values push the other way. The effect is normalized
like patching, so 1 means the steered prompts moved as far as switching to the other prompt. A
random direction of the same length, at the same strengths, runs alongside as a control, and the
results show both: a layer × strength map with the control columns muted, and, for a selected site,
its effect at every strength with intervals. **Check robustness** offers another split of the pairs
or the other direction. A mean difference steers only when the pairs differ the same way. In IOI
prompts that mix the ABBA and BABA orders, the difference at the last token flips with the order,
so the mean cancels and steering does no more than the control; generate prompts of one order to
steer. When no site and strength does more than the control, the results say so.

**SAE features.** A sparse autoencoder (SAE) rewrites one of the model's activations as a few active
features out of thousands, each a direction in the model, plus an error it misses. In
**Explore → Features**, load a published SAE from Hugging Face (Logogram suggests ones made for the
suggested models, or give any repository), then **Measure the fit** on your prompts: the fraction
of the activations' variance the SAE explains, how many features fire per token, and the logit
difference with the SAE's reconstruction spliced in. SAEs only fit activations like those they were
trained on, so a poor fit (for example with TransformerLens's weight processing switched the other
way) is shown before anything else. The view then lists the features that fire on each token of a
prompt and, for one feature, its activation along the prompt and the prompts where it fires most,
all computed from your own prompts. Feature descriptions live on sites like Neuronpedia, which
Logogram doesn't contact.

Features are sites like any other (`sae_feature`, with a feature index). **Patching a feature**
encodes the receiver's activation and changes only that feature, to its value in the source prompt
(or to zero, for zero ablation): the activation moves along the feature's decoder direction, and the
SAE's error is kept. **Attribution patching** can sweep **every SAE feature** at once and keep the
strongest, then verify them by patching; the results also show how much of the whole site's
estimated effect the features account for (the rest is the SAE's error). Logogram reads the
SAELens and EleutherAI formats from safetensors files only; Gemma Scope's NumPy archives are
refused, as are SAEs whose input scaling or architecture it can't reproduce exactly.

**Sites** are abstract: `resid_pre`, `resid_mid`, `resid_post`, `attn_out`, `mlp_out` and `head`
(an attention head's output `z`), each at a layer, head and position. Positions are `all`,
`last`, a token `index` or a named `label`. **Sweeps** cover every head at one position
(layer × head), one stream site at every layer and position (layer × position), or attention
and MLP outputs per layer.

**The metric** is the logit difference at the last token, answer minus distractor. The
**normalized effect** of each prompt is (patched − receiver) ÷ gap, where the gap is
(source − receiver) for patching and (corrupt − clean) for ablation. By default the gap is the
dataset's mean, so the mean effect is the usual normalized metric; you can choose each prompt's
own gap instead. 0 is no change; 1 is a change as large as switching to the other prompt.

**Statistics** for every site: n, mean, standard deviation, a percentile bootstrap confidence
interval over prompts (seeded; the same resamples for every site), the number of prompts whose
logit difference changed sign, and the number whose effect has the opposite sign to the mean.
Per-prompt values are always kept.

## The spec

Every experiment is a `spec.json`. Nothing that can change a number is left implicit.

```json
{
  "logogram_spec": 1,
  "name": "Which heads restore the answer?",
  "notes": "",
  "model": {
    "id": "openai-community/gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "dtype": "float32",
    "device": "auto",
    "process_weights": true
  },
  "dataset": {"path": "datasets/ioi.jsonl", "sha256": "2bd046d5…", "limit": null},
  "tokenization": {"prepend_bos": true},
  "experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"},
  "scope": {"kind": "heads", "position": {"kind": "all"}},
  "metric": {"kind": "logit_diff", "normalization": "dataset_gap"},
  "statistics": {"bootstrap": 1000, "ci": 0.95, "seed": 0},
  "execution": {"batch_size": 64}
}
```

| Field | Values |
|---|---|
| `model.revision` | A commit. `null` resolves the current main branch at run time; the saved spec pins what ran. |
| `model.process_weights` | Fold LayerNorm and center weights, as TransformerLens does by default. Logit differences don't change. Value biases are folded into the attention output's bias, so zero ablation of head outputs and heads' direct effects do change (by up to several logits in Qwen 2.5, whose value biases are large); a layer's attention output doesn't. |
| `dataset.sha256` | If set, the run refuses a dataset file that has changed. |
| `experiment` | `{"kind": "activation_patching", "direction": "clean_to_corrupt" \| "corrupt_to_clean"}`, `{"kind": "ablation", "baseline": …}` with `{"kind": "zero"}`, `{"kind": "mean", "reference": "clean" \| "corrupt"}` or `{"kind": "resample", "pool": "clean" \| "corrupt", "donors": 10, "seed": 0}`, `{"kind": "attribution_patching", "direction": …}` (same directions as patching), `{"kind": "direct_logit_attribution", "prompts": "clean" \| "corrupt"}`, `{"kind": "path_patching", "direction": …, "receivers": [{"kind": "head", "layer": 9, "head": 9, "input": "q" \| "k" \| "v"}, {"kind": "logits"}], "freeze_mlps": false}`, or `{"kind": "steering", "apply_to": "clean" \| "corrupt", "coefficients": [-1, 1, 2], "train_fraction": 0.5, "seed": 0, "control": true}` with a scope of one residual component per layer (`layer_components`) or residual `sites`, at one token |
| `scope` | `{"kind": "heads", "position": …}`, `{"kind": "layer_position", "site": "resid_pre", "positions": "each" \| "labels"}`, `{"kind": "layer_components", "components": ["attn_out", "mlp_out"], "position": …}` `{"kind": "sites", "sites": [{"kind": "head", "layer": 9, "head": 9, "position": {"kind": "label", "label": "end"}}]}` (feature sites are `{"kind": "sae_feature", "layer": 8, "feature": 1234, "position": …}`), or, for attribution patching, `{"kind": "features", "position": …, "top": 50}` |
| `sae` | The SAE that feature sites belong to: `{"repo": "…", "path": "blocks.8.hook_resid_pre", "revision": "…"}`. The revision is pinned when a run starts. |
| `metric.normalization` | `dataset_gap` or `prompt_gap` |
| `execution.batch_size` | Recorded because batch shape can change floating-point results in the last digits. |

## Projects

```text
my-project/
  project.json
  research.json                      saved selections and research notes (created on first save)
  datasets/*.jsonl
  datasets/snapshots/<sha256>.jsonl   immutable input bytes, shared by runs with identical data
  experiments/<id>/spec.json          the experiment, with revision and dataset hash pinned
  experiments/<id>/results.parquet    one row per site and prompt
  experiments/<id>/summary.json       per-site statistics and the layout of the results
  experiments/<id>/predictions.json   optional per-layer prediction diagnostic
  experiments/<id>/manifest.json      versions, device, dtype, model revision, dataset hash, timing
  .gitignore
```

`results.parquet` has everything needed to recompute the statistics: site, layer, head,
position, prompt, the patched logit difference and answer probability, and the receiver and
reference logit differences. Paths inside a project are relative, so folders can be moved,
committed and shared.

Each new run snapshots its input dataset before loading the model and pins that snapshot in the
executed spec. Editing the original dataset leaves existing runs reproducible. A modified
snapshot is refused by its hash; it is never silently overwritten. Keep snapshots alongside
experiments when sharing a project. Older runs still use the dataset path in their original spec.

Browser tabs follow the server's open project. Switching projects clears the other tabs' forms,
prompts, cached analyses and results; stale requests are rejected. A project cannot be switched
while a job runs. The status line shows lost connections, and interrupted runs are marked failed
when their project is reopened, with their saved spec available to rerun.

**Reproducibility.** Runs seed Python, NumPy and PyTorch from `statistics.seed`, require deterministic
PyTorch algorithms (unsupported operations fail rather than merely warn), and disable
TF32, and record Logogram, Python, PyTorch, TransformerLens and transformers versions, the device
and GPU model, the dtype, the model id and revision, whether weights were processed, the dataset
hash and the wall time. The same spec on the same machine gives bit-identical results. Results
on different devices or library versions can differ; compare their per-prompt values before
treating them as equivalent.

## Privacy

* Everything runs locally. There is no account, no telemetry and no analytics. The app loads no
  fonts or scripts from the internet; its fonts are bundled.
* Logogram goes online in two cases only. When you load a model, it asks Hugging Face to resolve
  its revision, read its size for the memory estimate and download it once (your own Hugging
  Face login is used for gated models; Hugging Face telemetry is turned off); loading an SAE
  downloads it the same way. And if you allow
  it, it asks pypi.org for the newest Logogram version number once a day (see Updating). Nothing
  about you, your machine or your work is sent.
* The server listens on 127.0.0.1 only. Every request needs the random session token from the
  address printed in the terminal (exchanged for a same-site cookie when the page opens). The
  browser is opened with a single-use code instead, because other users of a shared machine can
  read a program's command line. The server checks the Host and Origin headers, sends no CORS
  headers, and refuses cross-site requests.
* Projects are often shared, so Logogram treats their contents with care: it writes only into
  folders that are really inside the project (never through a symlink that leads out of it), and
  a damaged or unexpected file shows up as an error on that file rather than breaking the project.
* Weights are loaded only from safetensors files, which cannot run code. Logogram never unpickles
  files.
* Settings and the list of recent projects are kept in your user configuration folder, not in
  projects.

## Updating

Logogram tells you when a new version is out, in the way you choose:

* **Never online by itself.** Each version knows its release date. A few months after it, the
  app and the terminal suggest checking for a newer one.
* **Check now.** Press it in the update panel (or the System check), or run
  `logogram check-updates`. Logogram asks pypi.org for the newest version number.
* **Daily, if you allow it.** The System check asks once whether Logogram may look once a day.
  You can change your mind in the update panel at any time.

When a newer version is known, a small **Logogram 0.x.y** button appears in the top bar with what's
new and the exact update command for how you installed it, and starting `logogram` prints it in
the terminal. Updating never touches your projects, results or settings:

| Installed with | Update with |
|---|---|
| `uv tool install logogram` | `uv tool upgrade logogram` |
| `uv tool install .` from a copy of this repository | `git pull`, then `uv tool install --reinstall .` |
| `uv sync` (development) | `git pull`, then `uv sync` |

A check is one HTTPS request for Logogram's public package information on pypi.org. It carries
Logogram's version number and nothing else.

## Models

TransformerLens 4 loads well over a hundred architectures, and Logogram works with the
decoder-only language models among them: GPT-2, Pythia, Llama, Mistral, SmolLM, Qwen, Gemma,
OLMo, Phi and more. The model dialog lists starting points, and any other Hugging Face id can be
typed in. Before downloading, Logogram checks that TransformerLens supports the architecture and
estimates the memory needed (weights, activations and a margin) against what is free. Processing
the weights needs more for a moment while the model loads: TransformerLens works on float32
copies, three to four times the float32 size of the weights. The estimate includes them, so a
model that fits only without processing (Qwen 2.5 0.5B on a 6 GB GPU) says so before it loads.

Use float32 whenever the model fits. float16 and bfloat16 halve the memory and run two to three
times faster on a GPU, but they round. In the IOI example on GPT-2 small and Qwen 2.5, float16
moved effects by at most 0.003 and bfloat16 by up to 0.03: the strongest sites stayed in place,
but effects smaller than about 0.01 changed order. On Pythia-70m, bfloat16 doubled the clean
logit difference and reordered the heads, and float16 overflows. Check results that matter
against float32: **Check robustness** reruns a 16-bit run in float32 and compares the two.

On Apple Silicon, **Automatic** runs models on the CPU. TransformerLens reports that Apple's MPS
can give silently wrong results, and Logogram hasn't been checked on it yet, so MPS is used only
when you choose it under **Device**. It is faster; check results that matter on the CPU.

Rather than trusting a list, Logogram checks every model when it loads, on a short fixed input:

* TransformerLens's version of the model must predict what the original model predicts. If weight
  processing changes the predictions, loading with processed weights is refused. So is a dtype in
  which the model overflows (Pythia in float16).
* It measures how each layer adds attention and the MLP to the residual stream: one after the
  other (sequential), or both from the same input (parallel, as in Pythia, GPT-J and Phi). Only
  sequential layers have a residual stream between attention and MLP (`resid_mid`), and the layer
  explorer draws what was measured.
* Layer predictions are offered when the final normalization and unembedding, with any logit
  soft-capping, reproduce the model's output.

GPT-2 small is the model the bundled example was made for. Every method has been run end to end
with real weights on GPT-2 small (with a SAELens and an OpenAI SAE), Pythia-70m (with an
EleutherAI SAE) and Qwen 2.5 0.5B, and the test suite runs the sanity checks on tiny random models
of the Llama, Pythia, Qwen 2, Gemma 2 and OLMo 2 families without downloading anything. The
example's names are single tokens for GPT-2 and Qwen but not all for Pythia; for another model,
generate IOI prompts in the app, which keeps only names that are single tokens for it. A small
model may not do the task at all (Pythia-70m prefers the repeated name), and its runs then say so.
Pythia publishes checkpoints from throughout training as revisions (`step1000` to `step143000`),
so an experiment can be rerun at several points of training.

Some tokenizers, such as Qwen's, have no beginning-of-sequence token. Logogram then runs prompts
without one, and the spec records it. Answers and distractors must still be single tokens.

## Development

```bash
uv sync                       # Python environment with dev tools
uv run pytest                 # tests use a tiny random model and never use the network
uv run ruff check src tests scripts
python3 scripts/check_privacy.py
```

The web app lives in `web/` (React, TypeScript, Vite). For live reloading, run the API and the
dev server side by side:

```bash
uv run logogram serve --dev
```

```bash
cd web && npm install && npm run dev
```

Open the development address printed by `logogram serve --dev`. Before committing changes to the
web app, build it into the package with `npm run build`; CI checks that the committed build
matches the sources.

Web regression checks:

```bash
cd web
npm test                      # context, cache and keyboard navigation checks
npx playwright install chromium
npm run test:browser           # temporary local project and tiny model; no Hub access
```

The browser fixture never opens the user's projects or cached models. CI runs these workflows
on Linux, and the Python suite plus an installed-wheel smoke check on Linux, macOS and Windows.
The wheel check covers the CLI, bundled example and fonts, authenticated API and served web
assets. Source distributions include the web sources and lockfiles so the bundle can be rebuilt.

Releasing: raise `__version__` and set `__released__` to the release date in
`src/logogram/__init__.py` (the date drives the "this version is getting old" hint), commit, and
push a tag with the same version, such as `v0.1.0`. The Release workflow runs every CI check on
that commit, builds the wheel and source distribution, and uploads them to PyPI through Trusted
Publishing, so no token is stored anywhere. PyPI never accepts the same version twice.

Before tagging, manually check a first GPT-2 download, cancel and retry it, then run
`scripts/validate_real_weights.py` on each supported compute backend. It runs every method on a
real model in a temporary project and checks exact identities, agreement between methods and
bit-identical reruns; `--compare-device` reruns the head sweep on a second device, and `--sae`
adds SAE features. CPU and CUDA pass it. Apple's MPS still needs it run on a Mac:

```bash
uv run python scripts/validate_real_weights.py --device mps --compare-device cpu
```

Pythia and Qwen 2.5 have also been run end to end with their real weights; other families are
checked when they load and in the tiny-model tests.

Model access goes through `logogram.backends.base.ModelBackend`. TransformerLens is the only
backend today; remote execution and other libraries can be added behind the same interface.
Not built yet, with room left for them: attribution graphs with transcoders, Gemma Scope's NumPy
SAE files, steering with SAE features, remote compute, a native desktop wrapper, plugins and
assistants.

## License

MIT. See [LICENSE](LICENSE).
