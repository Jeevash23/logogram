# Changelog

What changed in each version of Logogram.

## 0.2.0 (unreleased)

Specs that state every choice, more metrics, circuits, five more tasks, statistics that hold across
many sites, a Python API and more commands, Gemma Scope 2 and transcoders, and integrated
gradients, all checked by patching.

### Changed

- **Specs are version 2.** Every choice that can change a number must be stated: the model's
  revision, dtype, device and weight processing, the dataset's hash and limit, tokenization, the
  metric, the statistics (bootstrap, interval, seed, clustering) and the batch size, and every
  field of an ablation's baseline and a sweep's positions. A version 1 spec still runs: where it
  leaves a choice out, it takes version 1's value, the run's first warning lists each one, and
  `logogram validate` shows the list without running.
- Spec paths must lie inside the project, and an SAE's revision must be a commit, tag or branch.
- Seeds must be whole numbers from 0 to 2³² − 1; a negative or huge seed used to crash a run.
- Sign flips count prompts whose preference for the answer, log P(answer) − log P(distractor),
  changes sign (for single tokens that is the logit difference, as before).
- Attribution patching no longer reports patched logit differences, probabilities and sign flips
  computed from first-order estimates; like direct logit attribution, it leaves them empty.
- The count of prompts with an effect of the opposite sign ignores rounding noise, so a site with
  no effect no longer reads as half opposite.
- Steering compares each strength with its random control prompt by prompt, from the same
  resamples, and doing significantly *less* than the control no longer counts as beating it.

### Added

- **Metrics:** the log-probability difference, the answer's log-probability, its probability,
  the probability difference and the KL divergence from the clean or corrupt prompt's next-token
  distribution, beside the logit difference. Every run also keeps each prompt's answer
  probability and preference.
- **Answer sets and continuations:** an answer or distractor can be a list of single tokens, read
  as one answer (every later year in greater-than), or text of several tokens, read token by
  token. Patching at every position covers the continuation too.
- **Sets of sites:** intervene on several sites at once, or on everything except a set ("keep
  the circuit, replace the rest"), and read each set's faithfulness (how much of the behavior the
  kept sites carry alone), the share of removing a set (how complete a circuit is), the
  faithfulness each site adds (minimality) and the interaction of two sites, all with intervals.
- **Tasks:** seeded datasets for greater-than, docstring, gendered pronouns, subject–verb
  agreement and capital cities, beside IOI, each with templates of one token length, named
  positions and the metric that reads them, from the app, `logogram generate` or Python.
- **Integrated gradients** for attribution patching (as in EAP-IG): the gradient is averaged over
  steps between the two prompts' embeddings, which corrects much of what one gradient misses.
  Its estimates are verified by patching like any others.
- **Statistics across many sites:** simultaneous bands that hold for every site of a sweep at
  once, Benjamini–Hochberg q-values, a bootstrap that resamples clusters of prompts (such as
  templates, with a warning when there are too few), and paired differences between two runs on
  the same prompts.
- **The every-feature sweep can choose its strongest features on a seeded share of the prompts
  and report them on the rest**, so their effects aren't biased by the choice.
- **Gemma Scope 2** SAEs and transcoders (`config.json` and `params.safetensors`), EleutherAI
  transcoders, and SAEs on attention heads' outputs. Transcoder features are read from an MLP's
  input and patched in its output.
- **A Python API for notebooks:** `logogram.run`, `load_run`, `list_runs`, `compare_runs`,
  `verify_run`, `check_robustness`, `generate` and `write_dataset`, with tables as pandas or
  Arrow and runs that show their heatmap in a notebook. Importing `logogram` loads no PyTorch.
- **Commands:** `list`, `show`, `compare`, `diff`, `verify`, `robustness`, `export`, `validate`,
  `tasks` and `generate`.
- **The web app** contains errors to the part of the page that failed, with a way to reload it
  and copy the details; undoes and redoes changes to the experiment form (Ctrl/⌘ Z), says when an
  action replaces the form and offers to undo it, and keeps an unsaved form for each project; asks
  before cancelling a run; and lists every shortcut on **?**. Radio groups take one tab stop and
  arrow keys, form fields have visible edges in both themes, and the status line announces a
  run's milestones rather than every tick.

### Fixed

- **Path patching with head receivers and the logits receiver measured only the logits path.**
  The logits receiver now adds its change to what the head receivers changed, so the two together
  measure both paths (checked against each alone).
- A metric or gap that isn't a finite number, as a 16-bit overflow gives, stops the run with the
  reason instead of finishing with empty effects; patched values that aren't finite are reported.
- A failed run's saved error no longer contains this machine's paths (the home folder, the
  Hugging Face cache or the project folder).
- Running out of memory during a run says to lower `execution.batch_size`, and frees the memory.
- Memory: gradient methods no longer keep what weight gradients would need; attribution patching
  computes each layer's terms once per batch instead of once per site; on a GPU, a layer's
  captured activations move to CPU memory when they would take more than a quarter of what is
  free; an SAE's fit is measured a chunk of tokens at a time; and the memory estimate counts what
  gradient methods and path patching hold.
- A deeply nested or very large file in a shared project no longer stops the project or its
  history from opening; it shows up as an error on that run or dataset.
- The server checks which project a tab shows on the SAE routes too, a tab whose session has
  expired stops reconnecting, unloading a model waits for work in progress, and cancelling a run
  no longer waits behind queued analyses.
- The token strip shows a run's own prompts again when you return to its results, instead of
  the experiment form's.
- Heatmaps and the model map draw hover and selection on a layer of their own, and find cells
  through a map, so large sweeps no longer redraw everything on every mouse move.

## 0.1.2 (2026-10-08)

Fixes and checks that came out of validating every method on real weights.

### Changed

- On Apple Silicon, **Automatic** runs models on the CPU. TransformerLens reports that Apple's MPS
  can give silently wrong results, and Logogram hasn't been checked on it yet, so MPS runs only when
  you choose it; the system check and the model dialog say why.

### Added

- **Check robustness** reruns a float16 or bfloat16 run in float32 and compares the two.
- When processing a model's weights is what doesn't fit, the model dialog offers to turn it off.
- When a generated IOI dataset has names the loaded model splits into several tokens, one click
  makes a new one with names it doesn't.
- Steering results say when the direction does no more than its random control at any site and
  strength.
- Attribution patching of residual stream sites warns that it can miss effects there, and that
  **Check robustness** patches every site.
- `scripts/validate_real_weights.py` runs every method on a real model and checks identities,
  agreement and bit-identical reruns, on one device or two. It is how a release, or Apple's MPS, is
  checked.

### Fixed

- The baseline says that the clean prompts prefer the distractor, instead of preferring the answer
  by a negative amount.

### Development

- Tests run on Python 3.14 too, and the Linux runners are pinned to Ubuntu 24.04.
- The test client uses httpx2, as Starlette recommends.

## 0.1.1 (2026-10-08)

Every method has now run end to end on real weights: GPT-2 small with a SAELens and an OpenAI
SAE, Pythia-70m with an EleutherAI SAE, and Qwen 2.5 0.5B. This release fixes what that turned up.

### Fixed

- Runs work on Apple Silicon. Every method converted tensors to float64 on the model's device,
  which Apple's Metal (MPS) doesn't have, so no run got past its baseline on a Mac. This still
  needs a check on a Mac.
- A model whose own forward pass overflows in the chosen dtype (Pythia in float16) is refused, with
  the dtype to use. Its load checks used to read as passed.
- When processing a model's weights doesn't fit in memory, loading says so and frees the memory,
  instead of failing with a raw CUDA error and keeping it.
- The memory estimate counts weight processing, which needs three to four times the float32 size
  of the weights while the model loads. Qwen 2.5 0.5B on a 6 GB GPU now reads "won't fit" with
  processing and "fits" without it.
- Models without a beginning-of-sequence token, such as Qwen, run without one, as documented.
  TransformerLens 4 had given them its end-of-text token instead.
- Models and SAEs that Logogram downloaded open offline without a revision, and offline memory
  estimates use the model's real size.
- `logogram run` reports an identical rerun of direct logit attribution or attribution patching as
  identical.
- Direct logit attribution in float16 and bfloat16 no longer warns about drift that is only
  rounding.

### Added

- Runs warn when the model prefers the distractor on the clean prompts, so it doesn't show the
  behavior they test (Pythia-70m on IOI).
- The model dialog shows the memory that weight processing needs. Qwen 2.5 0.5B is marked as
  tested.
- Project links on PyPI, and this changelog, which the update notice links to.

### Documentation

- What has been validated with real weights, and where attribution patching is least reliable:
  residual stream sites at the token that differs between the prompts.
- Steering needs prompt pairs that differ the same way; IOI prompts that mix the ABBA and BABA
  orders cancel out.
- What float16 and bfloat16 do to results, and how weight processing changes heads' direct effects.

## 0.1.0 (2026-10-08)

First public release.

- Activation patching, ablation (zero, mean or resample), direct logit attribution, attribution
  patching verified by patching, steering with a random control, path patching, and SAE features,
  all described by one spec that the app and `logogram run` execute the same way.
- Every model is checked when it loads: that TransformerLens reproduces its predictions, how its
  layers add into the residual stream, and whether the logit lens reproduces its output.
- Per-prompt values with seeded bootstrap intervals, and bit-identical reruns on the same machine.
- The workbench: Explore, Experiment and Evidence, logograms written by results, robustness checks
  and run comparisons.
- Local-first: no account and no telemetry, safetensors weights only, and an update check that
  runs only when you allow it.
