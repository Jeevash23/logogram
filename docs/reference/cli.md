<!-- Generated from src/logogram/cli.py by scripts/build_reference.py. Don't edit it by hand: run `uv run python scripts/build_reference.py`. -->

# The command line

Logogram: a local workbench for causal experiments inside language models. Every command and option below is read from [`src/logogram/cli.py`](https://github.com/Jeevash23/logogram/blob/main/src/logogram/cli.py).

Commands that read runs use the project in the current folder or above it, or `--project PATH`; a run is its id (its folder's name in `experiments/`) or its folder. `logogram COMMAND --help` prints the same as this page, for one command.

| Command | What it does |
|---|---|
| [`logogram`](#logogram) | Start Logogram and open it in your browser. |
| [`logogram archive`](#logogram-archive) | Archive a run: it stays in the project, but out of `logogram list` and the app's history until they are asked to show archived runs. |
| [`logogram check-updates`](#logogram-check-updates) | Ask PyPI whether a newer Logogram is out (sends nothing about you or your work). |
| [`logogram compare`](#logogram-compare) | Compare two finished runs: rank correlation, top sites, conclusions that changed, and paired differences where they measured the same prompts. |
| [`logogram delete`](#logogram-delete) | Delete a finished, failed or cancelled run, or a draft: its folder and everything in it. |
| [`logogram diff`](#logogram-diff) | List every choice that differs between two specs (name and notes aside). |
| [`logogram doctor`](#logogram-doctor) | Report the environment and hardware, with fixes for common problems. |
| [`logogram env`](#logogram-env) | Print what results depend on beyond the spec, as JSON: Logogram's, Python's and the libraries' versions, the operating system's family and the devices PyTorch can use. |
| [`logogram export`](#logogram-export) | Write a finished run's values as CSV: one row per site and prompt, or per site. |
| [`logogram generate`](#logogram-generate) | Write a seeded dataset of prompt pairs for a task into the project's datasets folder. |
| [`logogram list`](#logogram-list) | List the project's runs and drafts, newest first, by their titles (★ starred). |
| [`logogram methods`](#logogram-methods) | Print a run's methods paragraph: every choice that can change a number, in sentences. |
| [`logogram models`](#logogram-models) | Model folders on this machine: add one under a name, list them, remove one. |
| [`logogram models add`](#logogram-models-add) | Hash a model folder's files and add it under NAME. |
| [`logogram models list`](#logogram-models-list) | List the model folders added on this machine. |
| [`logogram models remove`](#logogram-models-remove) | Forget a local model. |
| [`logogram open`](#logogram-open) | Start Logogram with a project open. |
| [`logogram queue`](#logogram-queue) | The project's run queue, kept in the project: add specs to it, list it, clear it, and run it to the end without the app (overnight on a remote GPU, say). |
| [`logogram queue add`](#logogram-queue-add) | Add a spec to the end of the project's queue. |
| [`logogram queue clear`](#logogram-queue-clear) | Take every run out of the queue. |
| [`logogram queue list`](#logogram-queue-list) | List the queued runs, first in line first, and say whether the queue is paused or who runs it. |
| [`logogram queue run`](#logogram-queue-run) | Run the project's queue to the end, one run at a time, headless: overnight on a remote GPU, say. |
| [`logogram rename`](#logogram-rename) | Give a run a title, shown instead of its spec's name. |
| [`logogram report`](#logogram-report) | Write a finished run as one HTML page anyone can open: its map, strongest sites, every site's numbers, the methods and where it came from. |
| [`logogram robustness`](#logogram-robustness) | Rerun with one methodological choice changed, then compare the two runs. |
| [`logogram run`](#logogram-run) | Run an experiment headlessly and print a summary. |
| [`logogram saes`](#logogram-saes) | List the published SAEs trained on MODEL, from SAELens's registry as the catalogue that ships with Logogram records it (nothing is fetched), and whether Logogram reads each exactly or why not. |
| [`logogram serve`](#logogram-serve) | Run the server (use --dev with `npm run dev` when working on the web app). |
| [`logogram show`](#logogram-show) | Print a finished run's method, baseline and strongest sites, or why a run failed, with the traceback it kept. |
| [`logogram star`](#logogram-star) | Star a run: ★ in `logogram list`, and found first in the app's history. |
| [`logogram sweep`](#logogram-sweep) | Run one spec across several values of one choice (seeds, datasets or revisions), one run after another, and show which of its strongest sites keep their sign and size across them. |
| [`logogram tasks`](#logogram-tasks) | List the tasks Logogram generates prompts for, with their options and metric. |
| [`logogram toy`](#logogram-toy) | Write a model whose circuit is known, with its datasets and experiments that say what each method should find, add it to the local models, and check that it does what it was built to. |
| [`logogram validate`](#logogram-validate) | Check a spec without running it, and say what it describes and what it left out. |
| [`logogram verify`](#logogram-verify) | Patch, for real, the sites an attribution patching run estimated to matter most (or path patch, one at a time, the edges an edge attribution run estimated), and compare the measurement with the estimate. |

## logogram

Start Logogram and open it in your browser.

```text
logogram [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--version` | nothing | off | Show the version. |
| `--port` | integer, 0 to 65,535 | `8765` | Port to listen on (127.0.0.1 only; 0 picks any free port). |
| `--no-browser` | nothing | off | Don't open a browser. |
| `--verbose` / `-v` | nothing | off | Log what the server does, with the traceback of each error. |
| `--log-file` | file | — | Also append the log to this file. |
| `--forwarded-port` | integer, 1 to 65,535; repeat it for more | — | A port on the browser's computer that forwards to this server, such as an SSH tunnel's local end (ssh -L 9000:127.0.0.1:8765 HOST): addresses on it are accepted too. Repeat it for several, or list them in LOGOGRAM_FORWARDED_PORTS. |

## logogram archive

Archive a run: it stays in the project, but out of `logogram list` and the app's history until they are asked to show archived runs. Nothing is deleted.

```text
logogram archive [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A run id, or its folder. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--restore` | nothing | off | Bring it back from the archive instead. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram check-updates

Ask PyPI whether a newer Logogram is out (sends nothing about you or your work).

```text
logogram check-updates
```

## logogram compare

Compare two finished runs: rank correlation, top sites, conclusions that changed, and paired differences where they measured the same prompts. A run that stopped early is compared on what it kept.

```text
logogram compare [OPTIONS] RUN_A RUN_B
```

| Argument | Takes | Description |
|---|---|---|
| `RUN_A` | text | The first run. |
| `RUN_B` | text | The second run. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram delete

Delete a finished, failed or cancelled run, or a draft: its folder and everything in it. There is no undo; `logogram archive` keeps a run out of the way instead.

```text
logogram delete [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A run id, or its folder. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--yes` / `-y` | nothing | off | Delete without asking first. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram diff

List every choice that differs between two specs (name and notes aside).

```text
logogram diff [OPTIONS] A B
```

| Argument | Takes | Description |
|---|---|---|
| `A` | text | A run id, its folder, or a spec file. |
| `B` | text | Another run id, folder or spec file. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram doctor

Report the environment and hardware, with fixes for common problems.

```text
logogram doctor [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--report` | nothing | off | Print only a short report to paste into an issue: versions, the operating system's family and the device, without names, paths or hardware models. |
| `--hardware` | nothing | off | With --report, add the processor, GPUs, memory and OS version. |

## logogram env

Print what results depend on beyond the spec, as JSON: Logogram's, Python's and the libraries' versions, the operating system's family and the devices PyTorch can use. The same spec, on the same machine with the same versions, gives bit-identical results; other versions or devices may differ in the last digits. Nothing names you or this machine: no user or host names, no paths, and no hardware models unless --hardware is given.

```text
logogram env [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--hardware` | nothing | off | Also name the processor, the GPUs and the memory. |
| `--requirements` | nothing | off | Print pinned requirements that install these versions instead. |

## logogram export

Write a finished run's values as CSV: one row per site and prompt, or per site.

```text
logogram export [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A finished run. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--output` / `-o` | path | — | Where to write the CSV (default: stdout). |
| `--sites` | nothing | off | One row per site instead of per site and prompt (for an edge attribution or ACDC run, every edge of its graph). |
| `--prompt-text` | nothing | off | Add each prompt's text to its rows. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram generate

Write a seeded dataset of prompt pairs for a task into the project's datasets folder.

```text
logogram generate [OPTIONS] TASK
```

| Argument | Takes | Description |
|---|---|---|
| `TASK` | text | A task, as `logogram tasks` lists them. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `-n` | integer | required | How many prompt pairs. |
| `--seed` | integer | required | The generator's seed: the same seed, the same file. |
| `--name` | text | — | The dataset's file name (default: the task's id). |
| `--option` / `-O` | text; repeat it for more | — | A task option as NAME=VALUE (lists comma-separated). |
| `--tokenizer` | text | — | A model already in the Hugging Face cache whose tokenizer the prompts are made for (words it splits are left out). Nothing is downloaded. |
| `--overwrite` / `--no-overwrite` | nothing | off | Replace a dataset of the same name. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram list

List the project's runs and drafts, newest first, by their titles (★ starred). Archived runs are left out unless --archived is given.

```text
logogram list [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |
| `--archived` | nothing | off | List archived runs too. |

## logogram methods

Print a run's methods paragraph: every choice that can change a number, in sentences.

```text
logogram methods [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A run. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram models

Model folders on this machine: add one under a name, list them, remove one. Specs name a local model by its name and the hash of its files, never by its folder.

Its commands: [`logogram models add`](#logogram-models-add), [`logogram models list`](#logogram-models-list) and [`logogram models remove`](#logogram-models-remove).

## logogram models add

Hash a model folder's files and add it under NAME.

```text
logogram models add [OPTIONS] PATH
```

| Argument | Takes | Description |
|---|---|---|
| `PATH` | path | The model's folder: config.json, tokenizer files, safetensors. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--name` | text | required | What specs call it, such as my-fine-tune. |

## logogram models list

List the model folders added on this machine.

```text
logogram models list
```

## logogram models remove

Forget a local model. Its folder and files are left as they are.

```text
logogram models remove NAME
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The name it was added under. |

## logogram open

Start Logogram with a project open.

```text
logogram open [OPTIONS] PATH
```

| Argument | Takes | Description |
|---|---|---|
| `PATH` | path | A project folder. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--port` | integer, 0 to 65,535 | `8765` | Port to listen on (127.0.0.1 only; 0 picks any free port). |
| `--no-browser` | nothing | off | Don't open a browser. |
| `--verbose` / `-v` | nothing | off | Log what the server does, with the traceback of each error. |
| `--log-file` | file | — | Also append the log to this file. |
| `--forwarded-port` | integer, 1 to 65,535; repeat it for more | — | A port on the browser's computer that forwards to this server, such as an SSH tunnel's local end (ssh -L 9000:127.0.0.1:8765 HOST): addresses on it are accepted too. Repeat it for several, or list them in LOGOGRAM_FORWARDED_PORTS. |

## logogram queue

The project's run queue, kept in the project: add specs to it, list it, clear it, and run it to the end without the app (overnight on a remote GPU, say). The app shows the same queue.

Its commands: [`logogram queue add`](#logogram-queue-add), [`logogram queue clear`](#logogram-queue-clear), [`logogram queue list`](#logogram-queue-list) and [`logogram queue run`](#logogram-queue-run).

## logogram queue add

Add a spec to the end of the project's queue. Nothing runs yet: `logogram queue run` runs the queue, or start it in the app.

```text
logogram queue add [OPTIONS] SPEC.json
```

| Argument | Takes | Description |
|---|---|---|
| `SPEC.json` | path | An experiment spec. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the spec). |

## logogram queue clear

Take every run out of the queue. A run in progress goes on; nothing starts after it.

```text
logogram queue clear [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--yes` / `-y` | nothing | off | Clear it without asking first. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram queue list

List the queued runs, first in line first, and say whether the queue is paused or who runs it.

```text
logogram queue list [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram queue run

Run the project's queue to the end, one run at a time, headless: overnight on a remote GPU, say. Each run's progress is printed, and runs added meanwhile (here or in the app) run too. A run that fails doesn't stop the queue. Ctrl+C stops the run in progress (it keeps what finished) and leaves the rest queued. The app can't run the queue at the same time.

```text
logogram queue run [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |
| `--verbose` / `-v` | nothing | off | If a run fails, print its traceback. |

## logogram rename

Give a run a title, shown instead of its spec's name. The spec never changes: the title is kept beside it, in annotations.json.

```text
logogram rename [OPTIONS] RUN TITLE
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A run id, or its folder. |
| `TITLE` | text | The title to show instead of the spec's name ("" clears it). |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram report

Write a finished run as one HTML page anyone can open: its map, strongest sites, every site's numbers, the methods and where it came from.

```text
logogram report [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A finished run. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--output` / `-o` | path | — | Where to write the page (default: RUN.html here). |
| `--hardware` | nothing | off | Name the CPU or GPU it ran on (left out by default). |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram robustness

Rerun with one methodological choice changed, then compare the two runs.

```text
logogram robustness [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A finished run. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--experiment` | text | — | The experiment to run instead, as JSON: {"kind": "ablation", ...}. |
| `--dtype` | text | — | The dtype to rerun in instead (float32, float16, bfloat16). |
| `--top` | integer | `10` | How many sites to list. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram run

Run an experiment headlessly and print a summary.

```text
logogram run [OPTIONS] SPEC.json
```

| Argument | Takes | Description |
|---|---|---|
| `SPEC.json` | path | An experiment spec. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the spec). |
| `--top` | integer | `10` | How many sites to list. |
| `--verbose` / `-v` | nothing | off | If the run fails, print its traceback. |

## logogram saes

List the published SAEs trained on MODEL, from SAELens's registry as the catalogue that ships with Logogram records it (nothing is fetched), and whether Logogram reads each exactly or why not.

```text
logogram saes [OPTIONS] MODEL [SEARCH...]
```

| Argument | Takes | Description |
|---|---|---|
| `MODEL` | text | The model's Hugging Face id, such as openai-community/gpt2. |
| `SEARCH` | text | Words to search for: a release, a hook, layer 8, a width such as 16k. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--exact` | nothing | off | List only the SAEs Logogram reads exactly. |
| `--json` | nothing | off | Print them as JSON. |

## logogram serve

Run the server (use --dev with `npm run dev` when working on the web app).

```text
logogram serve [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--port` | integer, 0 to 65,535 | `8765` | Port to listen on (127.0.0.1 only; 0 picks any free port). |
| `--dev` | nothing | off | API only, for the Vite dev server in web/. |
| `--no-browser` | nothing | off | Don't open a browser. |
| `--project` | path | — | Open this project. |
| `--verbose` / `-v` | nothing | off | Log what the server does, with the traceback of each error. |
| `--log-file` | file | — | Also append the log to this file. |
| `--forwarded-port` | integer, 1 to 65,535; repeat it for more | — | A port on the browser's computer that forwards to this server, such as an SSH tunnel's local end (ssh -L 9000:127.0.0.1:8765 HOST): addresses on it are accepted too. Repeat it for several, or list them in LOGOGRAM_FORWARDED_PORTS. |

## logogram show

Print a finished run's method, baseline and strongest sites, or why a run failed, with the traceback it kept. A run that stopped early also prints what it kept of what finished.

```text
logogram show [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A run id, or its folder. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |
| `--top` | integer | `10` | How many sites to list. |

## logogram star

Star a run: ★ in `logogram list`, and found first in the app's history.

```text
logogram star [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | A run id, or its folder. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--remove` | nothing | off | Remove the star instead. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram sweep

Run one spec across several values of one choice (seeds, datasets or revisions), one run after another, and show which of its strongest sites keep their sign and size across them.

```text
logogram sweep [OPTIONS] SPEC_OR_RUN
```

| Argument | Takes | Description |
|---|---|---|
| `SPEC_OR_RUN` | text | A spec file, or a run (its id or folder). A finished run stands in for its own value, which isn't run again. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--field` | text | — | The field to vary, dotted as in the spec: statistics.seed, experiment.baseline.seed (resample ablation's donors), experiment.seed (steering's split) or scope.seed (the features' split). Give its values with --values. |
| `--values` | text | — | The field's values, in order, separated by commas, such as 0,1,2,3. |
| `--datasets` | text | — | Sweep the dataset instead: dataset files of the project, in order, separated by commas (a bare name is looked for in datasets/). |
| `--revisions` | text | — | Sweep the model's revision instead: commits, tags or branches of its Hugging Face repository, in order, separated by commas, such as step1000,step2000. |
| `--top` | integer, 0 to 500 | `10` | How many of the strongest sites to show (0: all). |
| `--choose` | text | `first` | Choose the strongest sites on the first value (first) or on all (all). |
| `--csv` | path | — | Also write the table, one row per site and value, as CSV. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |
| `--verbose` / `-v` | nothing | off | If a run fails, print its traceback. |

## logogram tasks

List the tasks Logogram generates prompts for, with their options and metric.

```text
logogram tasks
```

## logogram toy

Write a model whose circuit is known, with its datasets and experiments that say what each method should find, add it to the local models, and check that it does what it was built to.

```text
logogram toy [NAME] [FOLDER]
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | induction or ioi. Leave it out to list the toys. |
| `FOLDER` | path | Where its project goes (default: toy-NAME in the projects folder). |

## logogram validate

Check a spec without running it, and say what it describes and what it left out.

```text
logogram validate SPEC.json
```

| Argument | Takes | Description |
|---|---|---|
| `SPEC.json` | path | An experiment spec. |

## logogram verify

Patch, for real, the sites an attribution patching run estimated to matter most (or path patch, one at a time, the edges an edge attribution run estimated), and compare the measurement with the estimate.

```text
logogram verify [OPTIONS] RUN
```

| Argument | Takes | Description |
|---|---|---|
| `RUN` | text | An attribution or edge attribution run. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--top` | integer | `10` | How many of the strongest estimates to patch (of an edge attribution run's, at most 50). |
| `--project` | path | — | Project folder (default: the one containing the current folder). |
