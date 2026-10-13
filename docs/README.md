# Logogram's documentation

Logogram is a local workbench for causal experiments inside language models: activation
patching, ablation, attribution patching, path patching, edge attribution, direct logit
attribution, steering and SAE features, with the statistics to read them by. These pages teach
the app and the methods, from a first run to the exact definition of every number.

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
| [Choosing a method](concepts/choosing-a-method.md) | Which method answers which question (where, how it flows, what a site represents, what steering does), what each costs and where it misleads |
| [Activation patching](concepts/activation-patching.md) | Copying an activation from one prompt into the run on the other; denoising and noising; what an effect of 1 means; sites (queries, keys, values and neurons too) and positions; pairs of different lengths |
| [Ablation](concepts/ablation.md) | Replacing an activation with zeros, a mean or other prompts' values, and why the baseline has no default |
| [Metrics and normalization](concepts/metrics.md) | The logit difference and the other metrics, answers of several tokens, the normalized effect |
| [Statistics](concepts/statistics.md) | Bootstrap intervals, clusters, simultaneous bands, q-values, sign flips, paired comparisons |
| [Attribution patching](concepts/attribution-patching.md) | Estimating every site from gradients (neurons and SAE features included), integrated gradients, and verifying estimates by patching |
| [Path patching](concepts/path-patching.md) | A component's effect through chosen receivers only, and single edges one at a time |
| [Edge attribution](concepts/edge-attribution.md) | Estimating every edge of the model's graph from gradients (EAP and EAP-IG), verifying the strongest by path patching, and the faithfulness curve |
| [Direct logit attribution](concepts/direct-logit-attribution.md) | Splitting the logit difference into what each component, neuron or value writes |
| [Steering](concepts/steering.md) | Adding a direction to the residual stream, with held-out prompts and a random control |
| [Linear probes](concepts/probes.md) | What can be read from a site: held-out accuracy, a shuffled-label control and selectivity |
| [Directions](concepts/directions.md) | Directions and subspaces saved in the project: patching along them, removing them, steering along them and reading them out |
| [Distributed alignment search](concepts/das.md) | Learning the subspace that carries an effect, against the whole site and a random subspace |
| [SAE features and transcoders](concepts/sae-features.md) | Finding a published SAE by name, sparse autoencoders, transcoders, patching features, what a feature writes |
| [Circuits](concepts/circuits.md) | Sets of sites intervened on at once: faithfulness, completeness, minimality, interactions |
| [The logit lens and the model checks](concepts/logit-lens-and-model-checks.md) | Reading each layer's prediction, and what Logogram measures about a model when it loads |
| [Runs and projects](concepts/runs-and-projects.md) | What a run keeps, a run that stops early, reproducibility, the files of a project |

## How-to guides

| Guide | |
|---|---|
| [Replicate IOI on GPT-2](guides/replicate-ioi.md) | Find the heads that carry the indirect object, and read them against the paper |
| [Find induction heads](guides/find-induction-heads.md) | Score every head with the head finder in the Attention view |
| [Check a method on a toy model](guides/toy-models.md) | Two models whose circuits are known, and what each method should find on them |
| [Work with an SAE](guides/work-with-an-sae.md) | Find a published SAE by name and load it, check its fit, read and patch its features, steer along one |
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
