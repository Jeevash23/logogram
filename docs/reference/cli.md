<!-- Generated from src/logogram/cli.py by scripts/build_reference.py. Don't edit it by hand: run `uv run python scripts/build_reference.py`. -->

# The command line

Logogram: a local workbench for causal experiments inside language models. Every command and option below is read from [`src/logogram/cli.py`](https://github.com/Jeevash23/logogram/blob/main/src/logogram/cli.py).

Commands that read runs use the project in the current folder or above it, or `--project PATH`; a run is its id (its folder's name in `experiments/`) or its folder. `logogram COMMAND --help` prints the same as this page, for one command.

| Command | What it does |
|---|---|
| [`logogram`](#logogram) | Start Logogram and open it in your browser. |
| [`logogram activations`](#logogram-activations) | Write the activations at sites and a position, one row per prompt or text, to activations/NAME.safetensors (one float32 tensor per site, [rows, width]), with activations/NAME.json saying what they are: the model, the sites, the position, the dataset or corpus and its hash, and each row's index, side and class. |
| [`logogram archive`](#logogram-archive) | Archive a run: it stays in the project, but out of `logogram list` and the app's history until they are asked to show archived runs. |
| [`logogram check-updates`](#logogram-check-updates) | Ask PyPI whether a newer Logogram is out (sends nothing about you or your work). |
| [`logogram compare`](#logogram-compare) | Compare two finished runs: rank correlation, top sites, conclusions that changed, and paired differences where they measured the same prompts. |
| [`logogram corpora`](#logogram-corpora) | The project's corpora: plain texts in its corpora/ folder, for the analyses that read text rather than prompt pairs. |
| [`logogram corpora delete`](#logogram-corpora-delete) | Delete a corpus: its texts and their description. |
| [`logogram corpora import`](#logogram-corpora-import) | Import texts as a corpus: from a file, from one file of a Hugging Face dataset (downloaded once, at the exact commit its revision names), or from a project dataset's prompts. |
| [`logogram corpora rename`](#logogram-corpora-rename) | Give a corpus another name. |
| [`logogram datasets`](#logogram-datasets) | The project's datasets: list them, rename one, delete one. |
| [`logogram datasets delete`](#logogram-datasets-delete) | Delete a dataset file from the project, for good. |
| [`logogram datasets rename`](#logogram-datasets-rename) | Give a dataset file another name. |
| [`logogram delete`](#logogram-delete) | Delete a finished, failed or cancelled run, or a draft: its folder and everything in it. |
| [`logogram diff`](#logogram-diff) | List every choice that differs between two specs (name and notes aside). |
| [`logogram directions`](#logogram-directions) | The directions saved in the project (its directions/ folder): probes' weights, learned subspaces, principal components and mean differences. |
| [`logogram directions delete`](#logogram-directions-delete) | Delete a direction. |
| [`logogram directions rename`](#logogram-directions-rename) | Give a direction another name. |
| [`logogram directions save`](#logogram-directions-save) | Save what a probe or a learned subspace run found at one of its sites as a direction: a probe's weights (with two classes, toward the second class from the first), or the learned subspace. |
| [`logogram doctor`](#logogram-doctor) | Report the environment and hardware, with fixes for common problems. |
| [`logogram env`](#logogram-env) | Print what results depend on beyond the spec, as JSON: Logogram's, Python's and the libraries' versions, the operating system's family and the devices PyTorch can use. |
| [`logogram export`](#logogram-export) | Write a finished run's values as CSV: one row per site and prompt, or per site. |
| [`logogram generate`](#logogram-generate) | Write a seeded dataset of prompt pairs for a task into the project's datasets folder. |
| [`logogram lens`](#logogram-lens) | Tuned lenses trained in the project: train one, deterministically, from a training spec or from options that state every choice, or list them. |
| [`logogram lens list`](#logogram-lens-list) | List the tuned lenses trained in the project, with what their training measured. |
| [`logogram lens train`](#logogram-lens-train) | Train a tuned lens (an affine translator per layer, read through the model's final normalization and unembedding) on a corpus of the project, and save it as lenses/NAME. |
| [`logogram list`](#logogram-list) | List the project's runs and drafts, newest first, by their titles (★ starred). |
| [`logogram methods`](#logogram-methods) | Print a run's methods paragraph: every choice that can change a number, in sentences. |
| [`logogram models`](#logogram-models) | Model folders on this machine: add one under a name, list them, remove one. |
| [`logogram models add`](#logogram-models-add) | Hash a model folder's files and add it under NAME. |
| [`logogram models list`](#logogram-models-list) | List the model folders added on this machine. |
| [`logogram models remove`](#logogram-models-remove) | Forget a local model. |
| [`logogram new`](#logogram-new) | Create a project, as the app's Create project does: a folder named after NAME, with project.json, datasets/ and experiments/. |
| [`logogram open`](#logogram-open) | Start Logogram with a project open. |
| [`logogram project`](#logogram-project) | The project: give it another name, or delete it. |
| [`logogram project delete`](#logogram-project-delete) | Delete the project's folder and everything in it, for good: its runs, drafts, datasets and queue. |
| [`logogram project rename`](#logogram-project-rename) | Give the project another name, in its project.json. |
| [`logogram projects`](#logogram-projects) | The projects opened lately, as the app lists them: list them, or take one out of the list. |
| [`logogram projects forget`](#logogram-projects-forget) | Take a project out of the recent list. |
| [`logogram queue`](#logogram-queue) | The project's run queue, kept in the project: add specs to it, list it, clear it, and run it to the end without the app (overnight on a remote GPU, say). |
| [`logogram queue add`](#logogram-queue-add) | Add a spec to the end of the project's queue. |
| [`logogram queue clear`](#logogram-queue-clear) | Take every run out of the queue. |
| [`logogram queue list`](#logogram-queue-list) | List the queued runs, first in line first, and say whether the queue is paused or who runs it. |
| [`logogram queue run`](#logogram-queue-run) | Run the project's queue to the end, one run at a time, headless: overnight on a remote GPU, say. |
| [`logogram rename`](#logogram-rename) | Give a run a title, shown instead of its spec's name. |
| [`logogram report`](#logogram-report) | Write a finished run as one HTML page anyone can open: its map, strongest sites, every site's numbers, the methods and where it came from. |
| [`logogram robustness`](#logogram-robustness) | Rerun with one methodological choice changed, then compare the two runs. |
| [`logogram run`](#logogram-run) | Run an experiment headlessly and print a summary. |
| [`logogram sae`](#logogram-sae) | SAEs trained in the project: train one, deterministically, from a training spec or from options that state every choice, or list them. |
| [`logogram sae list`](#logogram-sae-list) | List the SAEs trained in the project, with what their training measured. |
| [`logogram sae train`](#logogram-sae-train) | Train a TopK SAE on one site of one layer, on a corpus of the project, and save it as saes/NAME (in SAELens's format, with training.json and metrics.json). |
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

## logogram activations

Write the activations at sites and a position, one row per prompt or text, to activations/NAME.safetensors (one float32 tensor per site, [rows, width]), with activations/NAME.json saying what they are: the model, the sites, the position, the dataset or corpus and its hash, and each row's index, side and class.

```text
logogram activations [OPTIONS] DATASET_OR_CORPUS
```

| Argument | Takes | Description |
|---|---|---|
| `DATASET_OR_CORPUS` | text | A dataset's path in the project (datasets/ioi.jsonl), or a corpus's name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--site` | text; repeat it for more | required | KIND:LAYER, such as resid_post:6; repeat it for more sites. |
| `-o` / `--output` | text | required | The export's name: activations/NAME.safetensors. |
| `--model` | text | required | The model: a Hugging Face id, or a local model's name. |
| `--position` | text | `last` | last, mean, a token index (negative counts from the end) or label:NAME. |
| `--revision` | text | — | The model's commit, branch or tag. |
| `--dtype` | text | `float32` | float32, float16 or bfloat16. |
| `--device` | text | `auto` | auto, cpu, cuda, cuda:N or mps. |
| `--process-weights` / `--no-process-weights` | nothing | on | Fold LayerNorm weights as TransformerLens does by default. |
| `--bos` / `--no-bos` | nothing | on | Begin each text with the beginning-of-sequence token. |
| `--batch-size` | integer | `16` | Prompts or texts in a batch. |
| `--sides` | text | `both` | A dataset's clean, corrupt or both prompts (each clean first). |
| `--mark` | text | — | A dataset's rows' class, recorded with each: side, or meta:FIELD. |
| `--limit` | integer | — | Read the first N prompt pairs or texts only. |
| `--max-tokens` | integer | `128` | A corpus's texts are cut to their first N tokens. |
| `--overwrite` / `--no-overwrite` | nothing | off | Replace an export of the same name. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

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

## logogram corpora

The project's corpora: plain texts in its corpora/ folder, for the analyses that read text rather than prompt pairs. Without a command, list them.

Its commands: [`logogram corpora delete`](#logogram-corpora-delete), [`logogram corpora import`](#logogram-corpora-import) and [`logogram corpora rename`](#logogram-corpora-rename).

```text
logogram corpora [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram corpora delete

Delete a corpus: its texts and their description. There is no undo.

```text
logogram corpora delete [OPTIONS] NAME
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The corpus's name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--yes` / `-y` | nothing | off | Delete without asking first. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram corpora import

Import texts as a corpus: from a file, from one file of a Hugging Face dataset (downloaded once, at the exact commit its revision names), or from a project dataset's prompts. Rows without text are left out, and counted.

```text
logogram corpora import [OPTIONS] NAME [FILE]
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The corpus's name in the project. |
| `FILE` | path | A file on this machine: Parquet, JSON Lines, JSON, CSV, TSV or text. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--hub` | text | — | A Hugging Face dataset, such as stanfordnlp/imdb. |
| `--file` | text | — | With --hub: the file in the dataset, such as data/train.parquet. |
| `--revision` | text | — | With --hub: a commit, branch or tag (default: main), pinned as a commit. |
| `--dataset` | text | — | A project dataset's prompts, such as datasets/ioi.jsonl. |
| `--sides` | text | `both` | With --dataset: clean, corrupt or both (each pair's clean prompt first). |
| `--label` | text | — | With --dataset: label texts by side, or by a meta field as meta:FIELD. |
| `--format` | text | — | lines, paragraphs, jsonl, json, csv, tsv or parquet (default: from the file's name). |
| `--split` | text | `lines` | A text file's texts: one per line (lines) or by blank lines (paragraphs). |
| `--text-field` | text | `text` | The field or column that holds each text. |
| `--label-field` | text | — | The field or column that holds each text's label. |
| `--limit` | integer | — | Keep the first N texts (with --dataset, the first N pairs). |
| `--overwrite` / `--no-overwrite` | nothing | off | Replace a corpus of the same name. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram corpora rename

Give a corpus another name.

```text
logogram corpora rename [OPTIONS] NAME NEW
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The corpus's name. |
| `NEW` | text | Its new name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram datasets

The project's datasets: list them, rename one, delete one. Runs keep their own copy of the prompts they ran on, so neither changes a run.

Its commands: [`logogram datasets delete`](#logogram-datasets-delete) and [`logogram datasets rename`](#logogram-datasets-rename).

```text
logogram datasets [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram datasets delete

Delete a dataset file from the project, for good. Runs that ran on it keep their own copy of its prompts in datasets/snapshots/. Refused while a saved draft or a queued run names it.

```text
logogram datasets delete [OPTIONS] NAME
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The dataset's file name, such as ioi.jsonl. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--yes` / `-y` | nothing | off | Delete it without asking first. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram datasets rename

Give a dataset file another name. Its prompts, and so its hash, stay the same; runs that ran on it keep their own copy. Refused while a saved draft or a queued run names it.

```text
logogram datasets rename [OPTIONS] NAME NEW
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The dataset's file name, such as ioi.jsonl. |
| `NEW` | text | Its new file name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram delete

Delete a finished, failed or cancelled run, or a draft: its folder and everything in it. A run in progress, or a draft waiting in the queue, can't be deleted. There is no undo; `logogram archive` keeps a run out of the way instead.

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

## logogram directions

The directions saved in the project (its directions/ folder): probes' weights, learned subspaces, principal components and mean differences. Without a command, list them.

Its commands: [`logogram directions delete`](#logogram-directions-delete), [`logogram directions rename`](#logogram-directions-rename) and [`logogram directions save`](#logogram-directions-save).

```text
logogram directions [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram directions delete

Delete a direction. Runs that used it keep their own copy.

```text
logogram directions delete [OPTIONS] NAME
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The direction's name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--yes` / `-y` | nothing | off | Delete without asking first. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram directions rename

Give a direction another name. Runs that used it keep their own copy.

```text
logogram directions rename [OPTIONS] NAME NEW
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The direction's name. |
| `NEW` | text | Its new name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram directions save

Save what a probe or a learned subspace run found at one of its sites as a direction: a probe's weights (with two classes, toward the second class from the first), or the learned subspace. Steering, directional ablation, subspace patching and readouts can then use it.

```text
logogram directions save [OPTIONS] RUN_ID SITE NAME
```

| Argument | Takes | Description |
|---|---|---|
| `RUN_ID` | text | A probe or learned subspace run's id. |
| `SITE` | integer | The site's index, as logogram show lists the run's sites. |
| `NAME` | text | The direction's name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--row` | integer | — | One vector of a learned subspace, or a probe of more than two classes: the class whose weights are saved. |
| `--replace` | nothing | off | Replace a direction of the same name. |
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

Write a seeded dataset of prompt pairs for a task into the project's datasets folder. --overwrite replaces a dataset of the same name, unless a saved draft or a queued run names it (they would fail, or run on other prompts); finished runs keep their own copy.

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

## logogram lens

Tuned lenses trained in the project: train one, deterministically, from a training spec or from options that state every choice, or list them. Each is saved as lenses/NAME; the layer predictions of a spec read it by name and the hash of its file.

Its commands: [`logogram lens list`](#logogram-lens-list) and [`logogram lens train`](#logogram-lens-train).

## logogram lens list

List the tuned lenses trained in the project, with what their training measured.

```text
logogram lens list [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram lens train

Train a tuned lens (an affine translator per layer, read through the model's final normalization and unembedding) on a corpus of the project, and save it as lenses/NAME. The same spec on the same machine gives the same translators, bit for bit.

```text
logogram lens train [OPTIONS] [SPEC.json]
```

| Argument | Takes | Description |
|---|---|---|
| `SPEC.json` | path | A training spec. Leave it out to state every choice with the options instead. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--name` | text | — | Its name: it is saved as lenses/NAME. |
| `--model` | text | — | A Hugging Face id, or the name of a local model. |
| `--revision` | text | — | The model's commit, tag or branch, or none (the current main). |
| `--dtype` | text | — | float32, float16 or bfloat16. |
| `--device` | text | — | auto, cpu, cuda, cuda:N or mps. |
| `--process-weights` / `--no-process-weights` | nothing | — | Load the model with TransformerLens's weight processing, or without. |
| `--corpus` | text | — | The corpus's name, in corpora/. |
| `--layers` | text | — | The layers that get a translator, such as 0,1,2, or all-but-last. |
| `--steps` | integer | — | Steps of the optimizer. |
| `--batch-tokens` | integer | — | Tokens per step. |
| `--lr` | number | — | The learning rate. |
| `--betas` | text | — | Adam's betas, such as 0.9,0.999. |
| `--eps` | number | — | Adam's epsilon, such as 1e-8. |
| `--weight-decay` | text | — | AdamW's weight decay, such as 0.001, or none. |
| `--seed` | integer | — | The seed of everything random. |
| `--held-out` | number | — | The share of texts held out and measured on. |
| `--log-every` | integer | — | Steps per point of the loss curve. |
| `--prepend-bos` / `--no-prepend-bos` | nothing | — | Begin each text with BOS, or not. |
| `--max-tokens` | integer | — | The most tokens read of a text. |
| `--model-batch` | integer | — | Texts per forward pass. |
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

## logogram new

Create a project, as the app's Create project does: a folder named after NAME, with project.json, datasets/ and experiments/. Prints its path.

```text
logogram new [OPTIONS] NAME
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The project's name; its folder is named after it. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--parent` | folder | — | The folder to make it in (default: the projects folder the app uses, Documents/Logogram). |

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

## logogram project

The project: give it another name, or delete it.

Its commands: [`logogram project delete`](#logogram-project-delete) and [`logogram project rename`](#logogram-project-rename).

## logogram project delete

Delete the project's folder and everything in it, for good: its runs, drafts, datasets and queue. Asks for the project's name to be typed first, unless --yes. Refused while a run is in progress in it or its queue is being run, and when the folder holds anything Logogram doesn't make there (it may be your own folder), listing it.

```text
logogram project delete [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--yes` / `-y` | nothing | off | Delete it without asking first. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram project rename

Give the project another name, in its project.json. Its folder keeps its name.

```text
logogram project rename [OPTIONS] NAME
```

| Argument | Takes | Description |
|---|---|---|
| `NAME` | text | The project's new name. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram projects

The projects opened lately, as the app lists them: list them, or take one out of the list.

Its commands: [`logogram projects forget`](#logogram-projects-forget).

## logogram projects forget

Take a project out of the recent list. Its folder is left as it is.

```text
logogram projects forget PATH
```

| Argument | Takes | Description |
|---|---|---|
| `PATH` | text | The project's folder, as the list shows it. |

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

## logogram sae

SAEs trained in the project: train one, deterministically, from a training spec or from options that state every choice, or list them. Each is saved as saes/NAME in SAELens's format, and specs name it by that folder and the hash of its files.

Its commands: [`logogram sae list`](#logogram-sae-list) and [`logogram sae train`](#logogram-sae-train).

## logogram sae list

List the SAEs trained in the project, with what their training measured.

```text
logogram sae list [OPTIONS]
```

| Option | Takes | Default | Description |
|---|---|---|---|
| `--project` | path | — | Project folder (default: the one containing the current folder). |

## logogram sae train

Train a TopK SAE on one site of one layer, on a corpus of the project, and save it as saes/NAME (in SAELens's format, with training.json and metrics.json). The same spec on the same machine gives the same weights, bit for bit.

```text
logogram sae train [OPTIONS] [SPEC.json]
```

| Argument | Takes | Description |
|---|---|---|
| `SPEC.json` | path | A training spec. Leave it out to state every choice with the options instead. |

| Option | Takes | Default | Description |
|---|---|---|---|
| `--name` | text | — | Its name: it is saved as saes/NAME. |
| `--model` | text | — | A Hugging Face id, or the name of a local model. |
| `--revision` | text | — | The model's commit, tag or branch, or none (the current main). |
| `--dtype` | text | — | float32, float16 or bfloat16. |
| `--device` | text | — | auto, cpu, cuda, cuda:N or mps. |
| `--process-weights` / `--no-process-weights` | nothing | — | Load the model with TransformerLens's weight processing, or without. |
| `--corpus` | text | — | The corpus's name, in corpora/. |
| `--site` | text | — | SITE:LAYER, such as resid_post:6 (resid_pre, resid_post, attn_out, mlp_out). |
| `--d-sae` | integer | — | How many features. |
| `--k` | integer | — | How many features each token keeps. |
| `--steps` | integer | — | Steps of the optimizer. |
| `--batch-tokens` | integer | — | Tokens per step. |
| `--lr` | number | — | Adam's learning rate. |
| `--betas` | text | — | Adam's betas, such as 0.9,0.999. |
| `--eps` | number | — | Adam's epsilon, such as 1e-8. |
| `--seed` | integer | — | The seed of everything random. |
| `--held-out` | number | — | The share of texts held out and measured on. |
| `--normalize` | text | — | mean_squared_norm (scaled to a mean squared norm of the width) or none. |
| `--auxk` | text | — | The AuxK loss as K,COEFFICIENT,DEAD_AFTER (tokens), or none. |
| `--log-every` | integer | — | Steps per point of the loss curve. |
| `--prepend-bos` / `--no-prepend-bos` | nothing | — | Begin each text with BOS, or not. |
| `--max-tokens` | integer | — | The most tokens read of a text. |
| `--model-batch` | integer | — | Texts per forward pass. |
| `--project` | path | — | Project folder (default: the one containing the current folder). |

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
