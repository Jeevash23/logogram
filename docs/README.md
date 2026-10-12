# Logogram's documentation

Logogram is a local workbench for causal experiments inside language models: activation
patching, ablation, attribution patching, path patching, direct logit attribution, steering and
SAE features, with the statistics to read them by. These pages teach the app and the methods,
from a first run to the exact definition of every number.

## Start here

* [Getting started](getting-started.md): install Logogram, open the IOI example, run it, and read
  the map, the inspector and the logogram.
* [The workbench](workbench.md): what each workspace and view is for.
* [Models](models.md): which models load, memory, precision, weight processing, devices and GPUs.

## Concepts

What Logogram computes, exactly, and when to use each method. Each page cites the papers the
method comes from.

| Page | What it covers |
|---|---|
| [Activation patching](concepts/activation-patching.md) | Copying an activation from one prompt into the run on the other; denoising and noising; what an effect of 1 means; sites and positions |
| [Ablation](concepts/ablation.md) | Replacing an activation with zeros, a mean or other prompts' values, and why the baseline has no default |
| [Metrics and normalization](concepts/metrics.md) | The logit difference and the other metrics, answers of several tokens, the normalized effect |
| [Statistics](concepts/statistics.md) | Bootstrap intervals, clusters, simultaneous bands, q-values, sign flips, paired comparisons |
| [Attribution patching](concepts/attribution-patching.md) | Estimating every site from gradients, integrated gradients, and verifying estimates by patching |
| [Path patching](concepts/path-patching.md) | A component's effect through chosen receivers only |
| [Direct logit attribution](concepts/direct-logit-attribution.md) | Splitting the logit difference into what each component writes |
| [Steering](concepts/steering.md) | Adding a direction to the residual stream, with held-out prompts and a random control |
| [SAE features and transcoders](concepts/sae-features.md) | Sparse autoencoders, transcoders, patching features, what a feature writes |
| [Circuits](concepts/circuits.md) | Sets of sites intervened on at once: faithfulness, completeness, minimality, interactions |
| [The logit lens and the model checks](concepts/logit-lens-and-model-checks.md) | Reading each layer's prediction, and what Logogram measures about a model when it loads |
| [Runs and projects](concepts/runs-and-projects.md) | What a run keeps, a run that stops early, reproducibility, the files of a project |

## How-to guides

| Guide | |
|---|---|
| [Replicate IOI on GPT-2](guides/replicate-ioi.md) | Find the heads that carry the indirect object, and read them against the paper |
| [Find induction heads](guides/find-induction-heads.md) | Score every head with the head finder in the Attention view |
| [Work with an SAE](guides/work-with-an-sae.md) | Load a published SAE, check its fit, read and patch its features, steer along one |
| [Check a finding's robustness](guides/check-robustness.md) | Rerun with one choice changed, and read what moved |
| [Compare two runs](guides/compare-runs.md) | Paired differences, rank correlation and changed conclusions |
| [Sweep seeds, datasets or checkpoints](guides/sweeps.md) | Run one spec across several values, and read which sites hold |
| [Queue runs and run them headless](guides/queue.md) | The project's queue, `logogram queue` and overnight runs |
| [Run a chat model on its own template](guides/chat-models.md) | Wrap prompts in the model's chat template |
| [Study your own fine-tune](guides/local-models.md) | Add a model folder on this machine and run specs on it |
| [Share a result](guides/share-results.md) | The HTML report, figures, CSV and the methods paragraph |
| [Use a remote GPU over SSH](guides/remote-gpu.md) | Run Logogram on a rented GPU and use it from your browser |

## Reference

Generated from the code (`uv run python scripts/build_reference.py` writes them again):

* [The spec](reference/spec.md): every field of `spec.json`, with its type, allowed values and
  description.
* [The command line](reference/cli.md): every command and option.
* [The Python API](reference/python-api.md): every public function and class of
  `import logogram as lg`.

The [notebooks](../notebooks/README.md) use the Python API on GPT-2 small, and the
[README](../README.md) has the installation table, troubleshooting and privacy notes.
