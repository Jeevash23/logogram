# AGENTS.md

Guidance for anyone, human or AI, changing this repository. Read it before editing.

Logogram is a local-first workbench for causal experiments in language models (activation
patching, ablation, robustness checks). It is a research instrument: correctness and
traceability come before features.

## Privacy rules (this repository is public)

* No tracked file may identify the person or machine it was made on: no absolute paths, home
  directories, usernames, hostnames, email addresses, hardware names, tokens or keys. This
  applies to code, comments, tests, examples, configs, fixtures, docs and screenshots.
* Use `platformdirs` for runtime locations (settings, recent projects) and project-relative paths
  in anything written into a project (specs, manifests, summaries).
* Don't commit run outputs from your machine: manifests record the device and GPU model. The
  bundled example ships a spec and a dataset only.
* No telemetry, analytics, crash reporting, update checks, CDNs or remote fonts. Bundle assets
  locally. The only network access is Hugging Face downloads that the user starts, using their
  own login.
* The server binds 127.0.0.1, requires the session token on every HTTP and WebSocket request,
  checks Host, Origin and Fetch Metadata, and sends no CORS headers. Keep it that way. The token
  never goes on a command line (the browser gets a single-use launch code).
* Project folders come from other people. Write into them only through `Project.writable` /
  `prepare_run_dir` / `dataset_file` and the `fileio` helpers (which never follow a symlink at the
  file), and read files only after checking they are regular files inside the project.
* Never unpickle untrusted files. Load weights from safetensors only.
* Run `python3 scripts/check_privacy.py` before committing. CI runs it on every push.

## Science rules

* The experiment spec (`src/logogram/spec.py`) is the single source of truth. The app and
  `logogram run` execute the same spec through `logogram.runner.run_spec`; never add a code path
  that only one of them uses.
* Every choice that can change a number lives in the spec and is shown in the UI: direction,
  baseline, site, position, metric, normalization, seeds, batch size, dtype, model revision.
  Nothing methodological is silently defaulted. Ablation has no default baseline.
* Specs use abstract sites (`resid_pre`, `resid_mid`, `resid_post`, `attn_out`, `mlp_out`,
  `head`). Library hook names appear only inside a backend (`src/logogram/backends/`).
* Keep per-prompt values. Statistics come from them: n, mean, SD, a seeded percentile bootstrap
  CI, sign flips and opposite-sign counts.
* Determinism: the same spec on the same machine must give bit-identical results. Seed
  everything, keep batch composition fixed, and don't introduce nondeterministic kernels.
* The sanity tests in `tests/test_sanity.py` must keep passing: patching all clean activations
  at a layer gives effect 1; patching corrupt into corrupt gives 0; the same spec twice gives
  identical results. Tests use a tiny local model and must never touch the network.

## Design rules

* Calm, exact and quietly beautiful: fog grey, ink and stillness. The model map is the one bold
  element; everything around it stays restrained.
* Color is for data only. Chrome stays neutral (tokens in `web/src/styles/tokens.css`). Effects
  use the diverging cobalt–neutral–ochre scale, symmetric around zero, always with a legend.
  Attention uses the single-hue ink ramp. Selection is an outline ring, not a new color.
* One typeface, Instrument Sans, with tabular figures for numbers. Monospace appears only in the
  spec view. Sentence case everywhere; no all-caps labels.
* Motion only explains state changes, such as cells filling as results stream in or panels
  opening.
* Copy uses plain verbs that say exactly what happens. Errors say what went wrong and how to fix
  it.
* Avoid the generic AI-app look: no decorative gradients, glassmorphism, glow, sparkle icons,
  emoji, chat panels, grids of identical rounded cards, purple-blue palettes, or 3D brains and
  node galaxies.
* Draw the model map and heatmaps on canvas. Use Radix primitives for menus, dialogs, tooltips
  and popovers, Zustand for state, d3-scale and d3-color for scales, CSS variables and CSS
  modules for styling. No component kits or chart libraries.
* Every view must work with the keyboard and in both light and dark themes.

## Layout of the code

```text
src/logogram/
  spec.py              experiment spec (pydantic)
  datasets.py, ioi.py  JSONL datasets and the IOI generator
  prompts.py           tokenization, alignment checks, length groups
  sites.py             scope expansion and result layouts
  engine.py            the sweep: captures, patches, streams per layer, cancels
  stats.py             bootstrap and summary statistics
  results.py           summary.json and results.parquet
  schema.py            schemas for run files, history listings and live events
  fileio.py            atomic writes that never follow symlinks
  runner.py            run a spec end to end (used by the app and the CLI)
  compare.py           robustness checks and run comparisons
  analysis.py          token strip, baseline check, attention patterns
  backends/            ModelBackend interface, TransformerLens backend, Hugging Face access
  server/              FastAPI app, security, response models, job and event handling
  system.py            hardware report, doctor, memory estimates
  cli.py               typer command line
  web_dist/            built web app (generated by `npm run build` in web/, committed)
  examples/ioi-gpt2/   the bundled example project
web/                   React + TypeScript + Vite sources
scripts/check_privacy.py
tests/
```

## Commands

```bash
uv sync
uv run pytest
uv run ruff check src tests scripts && uv run ruff format src tests scripts
python3 scripts/check_privacy.py
cd web && npm install && npm run build   # rebuild src/logogram/web_dist after web changes
uv run logogram serve --dev              # API for `npm run dev` in web/
```
