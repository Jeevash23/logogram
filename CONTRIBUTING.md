# Contributing to Logogram

Thank you for helping. Logogram is a research instrument: a result is worth only as much as the
ability to check how it was made, so correctness and traceability come before features.

[AGENTS.md](AGENTS.md) holds the rules every change follows, for people and AI assistants alike.
Read it before you change anything; this page summarizes it and says how to work. To report a
problem or suggest an idea, open an issue. A security problem goes in a private report instead
(see [SECURITY.md](SECURITY.md)).

## Setting up

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and Node 24 for the web app.

```bash
git clone https://github.com/Jeevash23/logogram.git
cd logogram
uv sync                       # the Python environment, with the dev tools
cd web && npm ci              # the web app's dependencies, as locked
```

To work on the web app with live reloading, run the API and the dev server side by side, then
open the development address that `logogram serve --dev` prints:

```bash
uv run logogram serve --dev
```

```bash
cd web && npm run dev
```

## The checks

CI runs these on every push. Run them before you open a pull request:

```bash
uv run pytest -q              # tiny local models; the tests never use the network
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
npx --yes pyright@1.1.408 --pythonpath .venv/bin/python
python3 scripts/check_privacy.py
```

And for the web app, in `web/`:

```bash
npx tsc --noEmit -p .
npm test                      # the logic tests
npm run build                 # rebuilds src/logogram/web_dist
npx playwright install chromium
npm run test:browser          # the app in a browser, with an axe accessibility scan
LOGOGRAM_COLOR_SCHEME=dark npm run test:browser
```

`uv run ruff format src tests scripts` fixes the formatting. A change to the web app includes the
rebuilt `src/logogram/web_dist`: CI checks that the committed build matches the sources. If
another checkout of Logogram is running its browser checks at the same time, give this one
another port with `LOGOGRAM_BROWSER_PORT=8883`.

## The rules, in short

**Privacy.** This repository is public. No tracked file may identify the person or machine it
was made on: no absolute paths, home folders, user or host names, e-mail addresses, hardware
names, tokens or keys, in code, comments, tests, fixtures, docs or screenshots. Don't commit run
outputs from your machine (manifests record its device). There is no telemetry, analytics, crash
reporting, CDN or remote font, and the only network access is Hugging Face downloads the user
starts and the opt-in version check in `src/logogram/updates.py`. The server binds 127.0.0.1,
requires the session token on every request, checks Host, Origin and Fetch Metadata, and sends
no CORS headers; the token never goes on a command line. Projects come from other people: write
into them only through `Project.writable`, `prepare_run_dir`, `dataset_file` and the `fileio`
helpers, read only regular files inside them, never unpickle anything, and load weights from
safetensors only.

**Science.** The experiment spec (`src/logogram/spec.py`) is the single source of truth: the app,
`logogram run` and the Python API all run it through `logogram.runner.run_spec`. Every choice
that can change a number lives in the spec, is shown in the app and is never silently defaulted.
Sites are abstract (`resid_pre`, `attn_out`, `head` and so on); library hook names appear only
inside `src/logogram/backends/`. Per-prompt values are kept, and the statistics come from them.
The same spec on the same machine gives bit-identical results. The sanity tests in
`tests/test_sanity.py` must keep passing, and a model is checked when it loads rather than
trusted to be what its architecture's name says.

**Design.** White and black, with color that means something: one color for each workspace,
green for a loaded model, and the cobalt-to-amber effect scale for effects only. Regions are
separated by space and hairlines. One typeface, sentence case, plain verbs that say what
happens, and errors that say how to fix them. The model map and heatmaps are drawn on canvas;
menus, dialogs, tooltips and popovers are Radix primitives; state lives in Zustand; there are no
component kits or chart libraries. Every view works with the keyboard, in both themes, down to an
800-pixel window.

## Adding a method

A method is a kind of experiment in the spec, and a new one goes through the same places as the
existing ones (attribution patching, steering and path patching are good examples to follow):

1. **The spec.** Add a model for it to `src/logogram/spec.py` with a new `kind`, stating every
   choice that can change its numbers, with no default for any of them. Describe it in words in
   `describe_intervention`, which the history, the run's summary and the command line show.
2. **The computation.** Write it in a module of its own (as `atp.py`, `steering.py` and
   `paths.py` are) that returns an `EngineResult` with a value for every site and prompt, and
   route the new kind to it in `engine.run_experiment`. Reach the model only through the
   `ModelBackend` interface (`backends/base.py`) and abstract sites.
3. **Results.** `runner.run_spec` turns the per-prompt values into statistics and writes the run's
   files. Extend `schema.py` and `results.py` only for what the method adds to `summary.json`.
4. **The app and the command line.** Add the method to the experiment form
   (`web/src/lib/formState.ts`, `web/src/lib/buildSpec.ts`, `web/src/views/ExperimentView.tsx`),
   its methods text (`web/src/lib/export.ts`), the choices **Check robustness** offers for it
   (`web/src/components/RobustnessDialog.tsx`) and, if its summary has more to say, the command
   line's (`_print_summary` in `src/logogram/cli.py`).
5. **Tests.** On the tiny models in `tests/conftest.py`, check an identity the method must
   satisfy (as the sanity tests check that patching every clean activation gives an effect of 1),
   that a rerun is bit-identical, and that it works on the other model families in
   `tests/test_architectures.py`.
6. **Documentation.** Say in the README what the method measures, where it can mislead, and its
   fields in the spec table.

A new field on an existing method is required in version 2 specs. For version 1 specs,
`upgrade_v1` in `spec.py` fills in the value that reproduces what they did before.

## Pull requests

Keep a pull request to one change, with tests for what it adds, and say in its description what
changed and how you checked it. Commit messages say what the change does, in plain words.
