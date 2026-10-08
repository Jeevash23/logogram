# Changelog

What changed in each version of Logogram.

## 0.1.2 (unreleased)

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
