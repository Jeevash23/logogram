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
  bundled examples ship a spec and a dataset each.
* No telemetry, analytics, crash reporting, CDNs or remote fonts. Bundle assets locally. Network
  access is limited to Hugging Face downloads the user starts (with their own login) and the
  version check in `src/logogram/updates.py`: one GET of pypi.org's JSON for `logogram`, carrying
  only the version in its User-Agent, run when the user asks or once a day if they opted in
  (`update_check` in settings, off until they say yes). Keep it opt-in and keep it that small.
* The server binds 127.0.0.1, requires the session token on every HTTP and WebSocket request,
  checks Host, Origin and Fetch Metadata, and sends no CORS headers. Keep it that way. The token
  never goes on a command line (the browser gets a single-use launch code).
* Project folders come from other people. Write into them only through `Project.writable` /
  `prepare_run_dir` / `dataset_file` and the `fileio` helpers (which never follow a symlink at the
  file), and read files only after checking they are regular files inside the project.
* Never unpickle untrusted files. Load weights, SAEs included, from safetensors only.
* Run `python3 scripts/check_privacy.py` before committing. CI runs it on every push.

## Science rules

* The experiment spec (`src/logogram/spec.py`) is the single source of truth. The app,
  `logogram run` and the Python API execute the same spec through `logogram.runner.run_spec`;
  never add a code path that only one of them uses.
* Every choice that can change a number lives in the spec and is shown in the UI: direction,
  baseline, site, position, metric, normalization, seeds, clustering, batch size, dtype, model
  revision. Nothing methodological is silently defaulted: a spec of the current version
  (`SPEC_VERSION` in `spec.py`) states every such field, and a new one is required. Specs of
  earlier versions still run: each version's upgrade function (`upgrade_v1`, `upgrade_v2`,
  `upgrade_v3`) fills in the fields the next version added, with the values that reproduce
  earlier behavior (`upgrade_v1` lists those that were a choice). Ablation has no default
  baseline.
* Specs use abstract sites (`resid_pre`, `resid_mid`, `resid_post`, `attn_out`, `mlp_out`,
  `head`). Library hook names appear only inside a backend (`src/logogram/backends/`).
* Keep per-prompt values. Statistics come from them: n, mean, SD, a seeded percentile bootstrap
  CI, sign flips and opposite-sign counts.
* Determinism: the same spec on the same machine must give bit-identical results. Seed
  everything, keep batch composition fixed, and don't introduce nondeterministic kernels.
* The sanity tests in `tests/test_sanity.py` must keep passing: patching all clean activations
  at a layer gives effect 1; patching corrupt into corrupt gives 0; the same spec twice gives
  identical results. `tests/test_architectures.py` runs them on tiny models of other families.
  Tests use tiny local models and must never touch the network.
* Don't trust a model to be what its architecture name says. `check_model` in the backend
  measures, when a model loads, that TransformerLens reproduces its predictions, how its layers
  add into the residual stream, and whether the logit lens reproduces its output; features that
  rely on a property must check that it was measured.

## Design rules

* White and black, with color that means something. The page is pure white (or pure black); regions
  are separated by space and hairlines, never grey washes or drop shadows (only menus, dialogs and
  tooltips float). Navigation runs across the top; there is no sidebar.
* Color has jobs (tokens in `web/src/styles/tokens.css`):
  * each workspace has one: green Explore (`--explore`), orange Experiment (`--experiment`), pink
    Evidence (`--evidence`). The open workspace's pill, its tab underline, its progress and its
    radio marks use it (`--ws` holds the current one);
  * green (`--live`) means a loaded model;
  * effects use the diverging cobalt–neutral–amber scale, symmetric around zero, always with a
    legend, and nothing else uses those hues. Attention uses the single-hue ink ramp.
  * Primary buttons are solid ink (white in the dark theme), not colored.
* Selection is circled in ink (`inkRing` in `web/src/lib/canvas.ts`), not a new color.
* Logograms (`web/src/lib/logogram.ts`) are data, not decoration. A run's glyph is written by its
  results: layers clockwise from the top, the ink swelling outward where a layer's strongest
  effect is positive and inward where it is negative, bleeding in the effect colors. Glyphs from
  a seed alone mark identity only: the brand (`BRAND_SEED`), projects, empty states. The welcome
  screens' color bloom behind the brand glyph is the one decorative use of color.
* One typeface, Instrument Sans (weight and width axes): condensed widths for titles and big
  figures (`--display-stretch`, `.figure`), tabular figures for numbers. Its word space is
  narrow, so body text widens it (`--word-space`). Monospace appears only in the spec view.
  Sentence case everywhere; no all-caps labels.
* Motion only explains state changes, such as cells filling as results stream in, a glyph
  appearing, or panels opening. Respect reduced motion.
* Copy uses plain verbs that say exactly what happens. Errors say what went wrong and how to fix
  it. Titles say what a view found or is for; they don't repeat the tab's name.
* Avoid the generic AI-app look and the enterprise look: no sidebars of icons, grey dashboards,
  decorative gradients, glassmorphism, glow, sparkle icons, emoji, chat panels, grids of
  identical rounded cards, purple-blue palettes, or 3D brains and node galaxies.
* Draw the model map and heatmaps on canvas. Use Radix primitives for menus, dialogs, tooltips
  and popovers, Zustand for state, d3-scale and d3-color for scales, CSS variables and CSS
  modules for styling. No component kits or chart libraries.
* Every view must work with the keyboard and in both the white and black themes, and down to an
  800-pixel window.

## Layout of the code

```text
src/logogram/
  spec.py              experiment spec (pydantic); earlier versions' specs are upgraded here
  datasets.py, ioi.py  JSONL datasets and the IOI generator
  tasks.py             seeded datasets for the other tasks (greater-than, docstring, ...)
  prompts.py           tokenization, alignment checks, length groups, seeded splits
  chat.py              prompts wrapped in the model's own chat template
  metrics.py           the metrics: answers, answer sets and continuations, scored per prompt
  sites.py             scope expansion and result layouts
  engine.py            the sweep: captures, patches, streams per layer, cancels; routes methods
  circuits.py          sets of sites intervened on at once, faithfulness and minimality
  direct.py            direct logit attribution
  atp.py               attribution patching (first-order estimates of patching)
  verify.py            the spec that verifies an estimate by patching its strongest sites
  steering.py          steering with held-out prompts and a random control
  paths.py             path patching
  edges.py             edge attribution (EAP, EAP-IG), edge patching and the faithfulness curve
  features.py          SAE features: patching and attribution patching
  sae.py               SAE encode, decode and fit
  sae_training.py      TopK SAEs trained in the project from a training spec (saes/NAME)
  lenses.py            tuned lenses trained in the project from a training spec (lenses/NAME)
  training.py          what both share: a corpus's texts split and ordered by seed, folders
                       written whole or not at all
  stats.py             bootstrap (over prompts or clusters), bands, q-values, paired differences
  results.py           summary.json and results.parquet
  runs.py              the per-prompt evidence behind one site's number
  report.py            a run as words, and as one self-contained HTML page
  exports.py           CSV exports of a saved run
  schema.py            schemas for run files, history listings and live events
  fileio.py            atomic writes that never follow symlinks
  project.py           project folders: where runs, datasets and drafts live, read safely
  research.py          research notes and named selections
  corpus_import.py     corpora from files, Hugging Face datasets' files and a dataset's prompts
  top_activations.py   where a neuron, an SAE feature, a direction or a head is most active in a corpus
  representations.py   activations at a site over prompts or texts: principal components, mean
                       differences, directions saved from them, exported activations
  runner.py            run a spec end to end (used by the app, the CLI and the Python API)
  partial.py           what a run that stops early keeps of what finished, and what it leaves out
  owner.py             which process runs a run, so a run going elsewhere isn't taken for interrupted
  runqueue.py          the project's run queue (.logogram/queue.json) and the one process running it
  history.py           what people say about runs (titles, stars, the archive), and deleting runs
  robustness.py        the spec that reruns a run with one choice changed
  sweeps.py            one spec run across several seeds, datasets or revisions, and its table
  compare.py           run comparisons, paired prompt by prompt where they can be
  api.py               the Python API for notebooks (imported lazily by logogram/__init__.py)
  analysis.py          token strip, baseline check, attention patterns
  inside.py            Your text: any text's predictions, logit lens, attention and features
  heads.py             the head finder: previous-token, duplicate-token and induction heads
  backends/            ModelBackend interface, TransformerLens backend, Hugging Face access,
                       published SAE formats, the catalogue of published SAEs (SAELens's
                       registry, built by scripts/sae_catalogue.py)
  local_models.py      models from folders on this machine, found by the hash of their files
  server/              FastAPI app, security, response models, job and event handling
  system.py            hardware report, doctor, memory estimates, the environment (logogram env)
  support.py           the problem report: versions and the last error, no names or paths
  updates.py           opt-in version check against PyPI, update commands
  cli.py               typer command line
  web_dist/            built web app (generated by `npm run build` in web/, committed)
  gallery.py           the bundled examples: what each asks, and the result to expect
  toys.py              toy models whose circuits are known (weights set by hand), their datasets,
                       and experiments that say what each method should find
  examples/            the example projects (IOI, greater-than, factual recall, gendered pronoun
                       on GPT-2 small): a spec and a dataset each
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
