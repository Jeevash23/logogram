# Getting started

This page takes you from installing Logogram to reading your first result: which of GPT-2
small's attention heads carry the answer in the indirect object identification (IOI) task. It
takes about ten minutes, most of it the first download of GPT-2 small. The app can walk you
through the same steps: open the example from the first screen with **Walk me through it**, or
choose **Help → Walk me through the example** at any time. New to interpretability? **Start
here**, under the tagline on the first screens and the first page of **Help → Guide**, explains in
five minutes the ideas this page relies on: clean and corrupt prompts, patching a head, what its
effect means, and the parts of a model.

## Install

Logogram needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/):

```bash
uv tool install logogram
logogram doctor
```

`logogram doctor` reports the operating system, the CPU, the memory, the GPU and the compute
backend PyTorch will use, and gives the exact command that fixes common problems, such as an
NVIDIA GPU with a PyTorch built for the CPU only. The [README](../README.md#install) lists the
install command for each kind of machine (CUDA on Windows, Apple Silicon, CPU only).

## Open the example

```bash
logogram
```

This starts Logogram on 127.0.0.1, prints its address and opens your browser. Everything runs
on your computer: there is no account, no telemetry, and the app goes online only to download
models from Hugging Face when you ask.

The first time, a system check shows what Logogram found, and the bundled examples: experiments
on GPT-2 small whose results can be read against the papers that found them. Choose **Open**
beside the *IOI example* (indirect object identification). Logogram copies the example into
`Documents/Logogram/ioi-example`, an ordinary project folder, and opens it. A copy made before
opens as you left it; it is never replaced.

The example holds 32 prompt pairs and one experiment, ready to run:

* **The prompts.** In each pair, the clean prompt is *When Mary and John went to the store, John
  gave a drink to*, where GPT-2 small answers *Mary*. The corrupt prompt repeats the other name
  instead (*…, Mary gave a drink to*), so the model should prefer *John*.
* **The experiment.** For each attention head, run the corrupt prompt, copy that head's output
  in from the clean prompt at every position, and measure how much of the clean answer comes
  back. This is [activation patching](concepts/activation-patching.md).

## Run it

The workbench has three workspaces across the top: **Explore** (green) looks inside the model,
**Experiment** (orange) sets up an experiment, and **Evidence** (pink) reads results. Until your
first run finishes, a button beside the tabs names the next of four steps.

1. **Experiment → Prompts** shows each pair token by token, with the tokens that differ marked.
   Logogram checks that both prompts of a pair have the same number of tokens, so that each
   position means the same thing in both (or, for
   [pairs of different lengths](concepts/activation-patching.md#pairs-of-different-lengths), that
   their positions are labelled in both), and says which pairs it can't use, and why.
2. **Experiment → Baseline** runs the model on the clean and corrupt prompts without changing
   anything. Press **Check the baseline**: the clean prompts should prefer the answer and the
   corrupt ones the other name. The gap between them is what every effect is measured against,
   so if the model doesn't show the behavior, nothing that follows means much. The baseline
   needs the model loaded: choose **Load a model** beside the tabs (GPT-2 small is the first
   suggestion), or press **Run** first, which loads the model the experiment names.
3. **Experiment → Configure** is the experiment form. Every choice that can change a number is
   there: the method, the direction, the sites to sweep, the metric and its normalization, and
   the statistics. Nothing is filled in silently. Press **Run** (or Ctrl/⌘ Enter). GPT-2 small
   (about 0.5 GB) downloads once from Hugging Face; the sweep then takes about a minute on a
   CPU, a few seconds on a GPU.

## Read the map

Results paint onto the map layer by layer as they arrive. **Evidence → Results** shows the run as
a grid: one row per layer, from layer 0 at the top, and one column per head. **Explore → Model**
shows the same results on the model map, beside each layer's residual stream, attention output
and MLP.

* **Color is the effect**, on a diverging scale symmetric around zero, with its legend beside
  the map: cobalt for positive effects, amber for negative ones, and neutral for none.
* **The effect** is normalized: 1 means patching that head alone restored the clean answer
  completely, 0 that it changed nothing, and a negative value that it pushed further toward
  the other name. See [metrics and normalization](concepts/metrics.md).
* Choose **Show values** to print each cell's mean, and **±1** to fix the scale at one gap
  either way, so runs can be compared by eye.

In the IOI example, a handful of heads stand out. The induction head 5.5 and the S-inhibition
heads 8.6, 8.10 and 7.9 restore the most, a fifth to a third of the gap each, then the name
movers 9.9 and 10.0, while the negative name movers 10.7 and 11.10 work against the answer: the
roles Wang et al. (2022) found for them. Logogram writes a head as `L9 H9` (layer 9, head 9).

## Open a site

Choose a cell, with the mouse or the arrow keys, to open it in the **inspector**, at the right
(on a narrow window, **Inspector** opens it as a panel). It shows everything behind the number:

* the exact intervention and metric, as the spec states them;
* the mean effect with its confidence interval, its simultaneous band and q-value (see
  [statistics](concepts/statistics.md));
* how many prompts' preference for the answer flipped, and how many went against the mean;
* the distribution of the per-prompt effects, and the prompts with the strongest and weakest
  effects, each a click away in the token strip.

Right-click a cell, or press Shift F10 on the model map, for **Patch here**, **Ablate here** and
**Compare across runs**; press **A** on a head for its attention pattern.

## The logogram

Each run gets a logogram: a circular ink glyph written by its results. Layers run clockwise from
the top, and the ink swells outward where a layer's strongest effect is positive and inward
where it is negative, in the effect colors. A run with a few strong layers reads as a heavy,
lopsided ring; a null result as a thin, even one. Logograms appear in the history, on the
results page, and beside the model map, where each layer number opens that layer's strongest
component. Glyphs made from a name alone, such as a project's, mark identity only.

## Check that it holds

A finding should survive changes that shouldn't matter. On the results, **Check robustness**
reruns the sweep with one choice changed: a different baseline, the other direction, more
donors, or float32 for a run in float16 or bfloat16. Logogram then reports the rank correlation
between the two runs, the overlap of their strongest sites, and the sites whose conclusion
changed, flagged on the map. See [check a finding's robustness](guides/check-robustness.md).

## Keyboard

**Ctrl/⌘ K** opens the command palette (type `L9H9` to jump to a head), **1** to **6** go to the
open workspace's tabs, **[** and **]** step through prompts, **P**, **B**, **A** and **C** patch,
ablate, show attention or compare the selected component, and **?** lists every shortcut. The
single-key shortcuts can be turned off in that list, for speech input or keys pressed by
accident.

## Next

* [The workbench](workbench.md) describes every view.
* [Replicate IOI on GPT-2](guides/replicate-ioi.md) goes further with the same example: the
  other direction, mean ablation over unrelated names, and a circuit.
* The [concepts](README.md#concepts) explain each method exactly, and the
  [notebooks](../notebooks/README.md) run the same experiments from Python.
