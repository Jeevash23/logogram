<!-- Generated from src/logogram/api.py by scripts/build_reference.py. Don't edit it by hand: run `uv run python scripts/build_reference.py`. -->

# The Python API

Logogram from Python: run specs, read runs and their per-prompt values, compare and check them.

Everything here goes through the same code as the app and `logogram run` (the runner, the
robustness and verification specs, the comparison), so a run made from a notebook is the same run
the app would make, and it lands in the project like any other:

```python
import logogram as lg

project = lg.open_project("~/Documents/Logogram/ioi-example")
model = lg.load_model("openai-community/gpt2")   # or a local model, by the name it was added under
spec = lg.load_spec(project.root / "experiments/ioi-head-patching/spec.json")
run = lg.run(spec, project, model=model)
run.sites()          # one row per site: mean effect, interval, q-value, ...
run.per_prompt()     # one row per site and prompt: everything the statistics came from
run                  # in a notebook: the method, the strongest sites and a heatmap
run.rename("Name movers").star()   # as the history shows it; the spec never changes
lg.delete_run(project, old_run)    # its folder and everything in it, for good
seeds = lg.sweep(run, field="statistics.seed", values=[0, 1, 2, 3])  # one run per value
seeds.table()        # the strongest sites at every value: do they keep their sign and size?
lg.list_saes(model, "layer 8")    # published SAEs for the model, and what Logogram reads of them
sae = lg.load_sae(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", model=model)
toy = lg.toy_model("induction")   # a model whose circuit is known, with experiments to run
toy.run("which-heads").sites()    # the planted heads at 1, every other head at 0
lg.import_corpus(project, "reviews", "reviews.csv", text_field="review", label_field="stars")
lg.top_examples(model, project, "reviews", {"kind": "neuron", "layer": 6, "neuron": 1234})
pcs = lg.principal_components(model, project, "datasets/ioi.jsonl", "resid_post:6", mark="side")
lg.save_direction(project, "pc1", pcs["components"][0], model=model, site="resid_post:6",
                  method="pca")
acts = lg.activations(model, project, "reviews", ["resid_post:6"], "mean")  # NumPy arrays
```

Tables come back as pandas DataFrames when pandas is installed, and as pyarrow Tables otherwise.

Every public name of `import logogram as lg`, from the docstrings in [`src/logogram/api.py`](https://github.com/Jeevash23/logogram/blob/main/src/logogram/api.py). Importing `logogram` loads no PyTorch; each function loads what it needs on first use.

## Functions

| Function | What it does |
|---|---|
| [`lg.activations`](#lgactivations) | The activations at `sites` and `position`, one row per prompt or text, as NumPy arrays `[rows, width]` in float64 by site (`"resid_post:6"`); nothing is written. |
| [`lg.add_local_model`](#lgadd_local_model) | Add a model folder on this machine (config.json, the tokenizer's files and safetensors weights) under `name`: its files are hashed (`on_progress(bytes done, bytes in all, file)`), and specs then name it by `name` and that hash, never by its folder. |
| [`lg.check_robustness`](#lgcheck_robustness) | Rerun with one choice changed (the experiment, or the dtype), as Check robustness does. |
| [`lg.compare_runs`](#lgcompare_runs) | Compare two finished runs of one project: rank correlation, top-k overlap, conclusions that changed, and paired differences where they measured the same prompts. |
| [`lg.continue_text`](#lgcontinue_text) | What `model` writes after `text`: `tokens` new tokens (1 to 64), each the most likely one (ties to the lowest token id), as the app's Your text view continues it (`plain`). |
| [`lg.create_project`](#lgcreate_project) | Create a project folder called after `name` inside `parent`. |
| [`lg.delete_run`](#lgdelete_run) | Delete a finished, failed or cancelled run, or a draft (a [`Run`](#lgrun-1) or a run id): its folder and everything in it, for good. |
| [`lg.edge_verification`](#lgedge_verification) | An edge attribution run's estimates beside their measurements by path patching (the run [`verify_run()`](#lgverify_run) made from it), edge by edge, ranked both ways, with Spearman's rank correlation of the two and how many have the same sign. |
| [`lg.export_activations`](#lgexport_activations) | Write the activations [`activations()`](#lgactivations) reads to `activations/NAME.safetensors` in the project (one float32 tensor per site, `[rows, width]`, named `"kind:layer"`) and what they are to `activations/NAME.json`: the model, the sites, the position, the dataset or corpus read and its hash, each row's index, side and class (`mark`: `"side"` or `"meta:FIELD"` for a dataset; a corpus's labels), and how texts were tokenized. |
| [`lg.feature_writes`](#lgfeature_writes) | The tokens a feature promotes and suppresses directly, as the app's Features view shows them: one unit of the feature's decoder direction, added to the residual stream entering the final normalization at one token of `prompt`, read through that normalization with its scale held at its value there and through the unembedding, skipping every later layer. |
| [`lg.generate`](#lggenerate) | Prompt pairs for a task (see `logogram.tasks.TASKS`). |
| [`lg.head_kinds`](#lghead_kinds) | Score every head of `model` (from [`load_model()`](#lgload_model)) as a previous-token, duplicate-token and induction head, on `sequences` seeded random sequences of `length` tokens repeated twice. |
| [`lg.import_corpus`](#lgimport_corpus) | Import texts as the corpus `name`, from one of three places: * `path`, a file on this machine: Parquet, JSON Lines, a JSON array, CSV, TSV or text (`format` says which when its name doesn't; a text file is split into `"lines"` or `"paragraphs"`, as `split` says). |
| [`lg.list_corpora`](#lglist_corpora) | The project's corpora (plain texts in its `corpora/` folder), by name: how many texts each has, its labels, the hash of its file and where its texts came from; `error` says why one can't be read. |
| [`lg.list_directions`](#lglist_directions) | The directions saved in the project (its `directions/` folder), by name: the model and site each was found on, how many dimensions it has, how it was made (`made_by`: the method, the run and in words) and the hash of its tensor file; `error` says why one can't be read. |
| [`lg.list_local_models`](#lglist_local_models) | The models added from folders on this machine: name, folder, hash and size. |
| [`lg.list_runs`](#lglist_runs) | Every run and draft in the project, newest first, as the history lists them: archived runs too (`archived`), and runs whose spec can't be read (`damaged`, with `error` saying why). |
| [`lg.list_saes`](#lglist_saes) | The published SAEs trained on a model (a loaded model, or its Hugging Face id), from the catalogue of SAELens's registry that ships with Logogram (nothing is fetched): one row per SAE, those Logogram reads exactly first, with its release and id, repository and folder, the hook it reads, its site, layer and width, its architecture, whether Logogram reads it exactly (`reads`: exact, refused or unknown) and if not why (`reason`), and how it was checked on its files if it was (`verified`). |
| [`lg.load_model`](#lgload_model) | Download (once) and load a model, checked as the app checks it. |
| [`lg.load_run`](#lgload_run) | A run of `project` by its id (the name of its experiments/ folder). |
| [`lg.load_sae`](#lgload_sae) | Download (once) and read a published SAE or transcoder at an exact revision, onto `model`'s device: by its Hugging Face repository and folder, or by its SAELens release and id (`release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre"`), found in the catalogue of SAELens's registry that ships with Logogram (see [`list_saes()`](#lglist_saes)). |
| [`lg.load_spec`](#lgload_spec) | A spec from a dict, a JSON string or a spec.json file. |
| [`lg.load_sweep`](#lgload_sweep) | A sweep of `project` by its id, as its runs record it: its runs in the order of its values (a value whose run is gone, or never ran, has none). |
| [`lg.look`](#lglook) | Read any text with `model` (from [`load_model()`](#lgload_model)), without making a dataset: its tokens, the `top_k` most likely next tokens after each, and at `position` (the last token by default) what each layer would predict, as the app's Your text view shows. |
| [`lg.mean_difference`](#lgmean_difference) | The mean activation at `site` and `position` over the rows of class `positive` minus the mean over those of class `negative` (read as [`activations()`](#lgactivations) reads them; a dataset's rows take their class from `mark`, `"side"` or `"meta:FIELD"`, a corpus's from their labels), as a NumPy vector. |
| [`lg.open_example`](#lgopen_example) | Copy a bundled example into a folder of its own inside `parent` (by default the projects folder the app uses, Documents/Logogram) and open it, as the app's examples do: a copy made before is opened as it is, never replaced. |
| [`lg.open_project`](#lgopen_project) | Open a project folder (one with a project.json). |
| [`lg.principal_components`](#lgprincipal_components) | The principal components of the activations at `site` and `position` (read as [`activations()`](#lgactivations) reads them), as the app's Representations view shows them: the rows centered on their mean and decomposed in float64, each component signed so its largest-magnitude loading is positive. |
| [`lg.read_corpus`](#lgread_corpus) | A corpus of the project: its `texts`, its `records` (each text with its label, id and meta), `labels()` (the distinct labels, as text), the `sha256` of its file and `meta` (where its texts came from). |
| [`lg.read_direction`](#lgread_direction) | A saved direction: its `vectors` (`[k, width]`, orthonormal rows, float32) and `meta` (the model and site it was found on, how it was made, its hash). |
| [`lg.remove_local_model`](#lgremove_local_model) | Forget a local model. |
| [`lg.run`](#lgrun) | Run a spec and write its experiment folder into `project`, as the app does. |
| [`lg.save_direction`](#lgsave_direction) | Save `vectors` (one vector `[width]`, or `[k, width]` for a subspace; a NumPy array, a tensor or a list) as the direction `name`, made orthonormal, with the length of each vector before kept. |
| [`lg.sweep`](#lgsweep) | Run a run's spec (or a spec, into `project`) across `values` of one `field`, one run after another, as `logogram sweep` does: a seed the spec states (`statistics.seed`, `experiment.baseline.seed`, `experiment.seed` or `scope.seed`), the dataset (`dataset.path`: dataset files of the project) or the model's revision (`model.revision`: commits, tags or branches, each run pinning its commit). |
| [`lg.top_examples`](#lgtop_examples) | Where a unit is most active in the corpus `corpus`, as the app's Corpus view shows it. |
| [`lg.toy_model`](#lgtoy_model) | A model whose circuit is known, to learn the methods on and to check them against: `"induction"` (a previous-token head and an induction head) or `"ioi"` (GPT-2 small's indirect-object circuit, one head per step), with weights set by hand (see `logogram.toys`). |
| [`lg.verify_run`](#lgverify_run) | Patch, for real, the `top` sites an attribution patching run estimated to matter most; for an edge attribution run, path patch its `top` strongest edges one at a time, at most 50 (compare the two with [`edge_verification()`](#lgedge_verification)). |
| [`lg.write_corpus`](#lgwrite_corpus) | Write `texts` as the corpus `name`, each with the label at the same place in `labels` (a string, a whole number or true/false) when they are given; empty texts are left out. |
| [`lg.write_dataset`](#lgwrite_dataset) | Write prompt pairs into the project's datasets/ folder; returns its path in the project. |

### lg.activations

```python
lg.activations(
    model: Any,
    project: Project,
    source: Any,
    sites: str | Sequence[str],
    position: Any = 'last',
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = 'both',
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
) -> dict[str, Any]
```

The activations at `sites` and `position`, one row per prompt or text, as NumPy
arrays `[rows, width]` in float64 by site (`"resid_post:6"`); nothing is written.

`source` is a project dataset's path (`"datasets/ioi.jsonl"`: its `sides`,
`"clean"`, `"corrupt"` or `"both"`, each pair's clean prompt then its corrupt one) or a
corpus's name (each text cut to its first `max_tokens` tokens); `limit` reads the first
pairs or texts only. `sites` are `"KIND:LAYER"` (`resid_pre`, `resid_mid`,
`resid_post`, `attn_out` or `mlp_out`). `position` is `"last"`, a token index
(negative counts from the end), `"label:NAME"` (a labelled position of the prompts) or
`"mean"` (the mean over the tokens after the beginning-of-sequence token). The same rows
always run in the same batches, so they give the same numbers. Write them to the project
with [`export_activations()`](#lgexport_activations).

### lg.add_local_model

```python
lg.add_local_model(
    path: str | Path,
    name: str,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]
```

Add a model folder on this machine (config.json, the tokenizer's files and safetensors
weights) under `name`: its files are hashed (`on_progress(bytes done, bytes in all, file)`), and specs then name it by `name` and that hash, never by its folder. Load it with
`load_model(name)`. See `logogram.local_models` for how the hash is defined.

### lg.check_robustness

```python
lg.check_robustness(
    original: Run,
    *,
    experiment: dict[str, Any] | None = None,
    dtype: str | None = None,
    model: Any = None,
) -> Run
```

Rerun with one choice changed (the experiment, or the dtype), as Check robustness does.

### lg.compare_runs

```python
lg.compare_runs(a: Run, b: Run) -> dict[str, Any]
```

Compare two finished runs of one project: rank correlation, top-k overlap, conclusions
that changed, and paired differences where they measured the same prompts.

### lg.continue_text

```python
lg.continue_text(
    model: Any,
    text: str,
    *,
    tokens: int,
    prepend_bos: bool = True,
    sae: Any = None,
    feature: int | None = None,
    activation: float | None = None,
    start: int | None = None,
) -> dict[str, Any]
```

What `model` writes after `text`: `tokens` new tokens (1 to 64), each the most likely
one (ties to the lowest token id), as the app's Your text view continues it (`plain`). With
`sae` (from [`load_sae()`](#lgload_sae)) and `feature`, also a continuation with that feature added
where the SAE writes (`steered`): `activation` units of it, in the feature's own units (its
decoder row times `activation`), at every position from token `start` on (the first real
token by default), the new tokens included. `steer` says what was added, with the feature's
largest activation on the text for reference.

### lg.create_project

```python
lg.create_project(parent: str | Path, name: str) -> Project
```

Create a project folder called after `name` inside `parent`.

### lg.delete_run

```python
lg.delete_run(project: Project, run: Run | str) -> None
```

Delete a finished, failed or cancelled run, or a draft (a [`Run`](#lgrun-1) or a run id): its
folder and everything in it, for good. Archive a run to keep it out of the way instead. A run
in progress can't be deleted. Runs derived from it stay, and keep its id in their manifest.

### lg.edge_verification

```python
lg.edge_verification(estimated: Run, verified: Run) -> dict[str, Any]
```

An edge attribution run's estimates beside their measurements by path patching (the run
[`verify_run()`](#lgverify_run) made from it), edge by edge, ranked both ways, with Spearman's rank
correlation of the two and how many have the same sign.

### lg.export_activations

```python
lg.export_activations(
    model: Any,
    project: Project,
    name: str,
    source: Any,
    sites: str | Sequence[str],
    position: Any = 'last',
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = 'both',
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
    overwrite: bool = False,
) -> dict[str, Any]
```

Write the activations [`activations()`](#lgactivations) reads to `activations/NAME.safetensors` in
the project (one float32 tensor per site, `[rows, width]`, named `"kind:layer"`) and what
they are to `activations/NAME.json`: the model, the sites, the position, the dataset or
corpus read and its hash, each row's index, side and class (`mark`: `"side"` or
`"meta:FIELD"` for a dataset; a corpus's labels), and how texts were tokenized. Returns where
they were written.

### lg.feature_writes

```python
lg.feature_writes(
    model: Any,
    sae: Any,
    feature: int,
    prompt: str,
    *,
    prepend_bos: bool,
    position: int | None = None,
    top: int = 10,
    chat_template: dict[str, Any] | None = None,
) -> dict[str, Any]
```

The tokens a feature promotes and suppresses directly, as the app's Features view shows
them: one unit of the feature's decoder direction, added to the residual stream entering the
final normalization at one token of `prompt`, read through that normalization with its
scale held at its value there and through the unembedding, skipping every later layer. Each
token's logit change per unit of the feature, relative to the mean change over the
vocabulary. `position` is the token (negative counts from the end); by default, where the
feature fires most in the prompt. `chat_template` wraps the prompt in the model's own
template, as a spec's tokenization.chat_template does (with `prepend_bos` false). See
`logogram.features.feature_writes()`.

### lg.generate

```python
lg.generate(
    task: str,
    n: int,
    seed: int,
    options: dict[str, Any] | None = None,
    *,
    model: Any = None,
) -> list[Any]
```

Prompt pairs for a task (see `logogram.tasks.TASKS`). With `model`, only words that are
single tokens for it are used, and clean and corrupt prompts have the same length in it.

### lg.head_kinds

```python
lg.head_kinds(
    model: Any,
    *,
    seed: int = 0,
    sequences: int = 32,
    length: int = 32,
    prepend_bos: bool = True,
    batch_size: int = 16,
    bootstrap: int = 1000,
    ci: float = 0.95,
    chat_template: dict[str, Any] | None = None,
) -> dict[str, Any]
```

Score every head of `model` (from [`load_model()`](#lgload_model)) as a previous-token,
duplicate-token and induction head, on `sequences` seeded random sequences of `length`
tokens repeated twice. Returns, per kind, each head's mean score with its bootstrap interval
over sequences, the per-sequence values, and the strongest heads; the app shows the same in
the Attention view. `chat_template` (as a spec writes it, with `prepend_bos` false) puts
each sequence where a prompt goes in the model's own chat template.

### lg.import_corpus

```python
lg.import_corpus(
    project: Project,
    name: str,
    path: str | Path | None = None,
    *,
    repo: str | None = None,
    file: str | None = None,
    revision: str | None = None,
    dataset: str | None = None,
    sides: str = 'both',
    label: str | None = None,
    format: str | None = None,
    split: str = 'lines',
    text_field: str = 'text',
    label_field: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> Any
```

Import texts as the corpus `name`, from one of three places:

* `path`, a file on this machine: Parquet, JSON Lines, a JSON array, CSV, TSV or text
  (`format` says which when its name doesn't; a text file is split into `"lines"` or
  `"paragraphs"`, as `split` says). The corpus records the file's name and hash, never
  its folder.
* `repo` and `file`, a file of a Hugging Face dataset, downloaded once at the exact
  commit `revision` resolves to (the main branch when None), which the corpus records.
* `dataset`, a project dataset's prompt pairs (`"datasets/ioi.jsonl"`): their
  `sides` (`"clean"`, `"corrupt"` or `"both"`), labelled by `label`: `"side"`,
  or `"meta:FIELD"` for a field of their meta.

`text_field` and `label_field` name the field or column that holds each text and its
label (no label when None); rows without text are left out. `limit` keeps the first texts
(the first pairs, for a dataset). Returns the corpus (as [`read_corpus()`](#lgread_corpus) does).

### lg.list_corpora

```python
lg.list_corpora(project: Project) -> list[dict[str, Any]]
```

The project's corpora (plain texts in its `corpora/` folder), by name: how many texts
each has, its labels, the hash of its file and where its texts came from; `error` says why
one can't be read. Import one with [`import_corpus()`](#lgimport_corpus) or write one with
[`write_corpus()`](#lgwrite_corpus).

### lg.list_directions

```python
lg.list_directions(project: Project) -> list[dict[str, Any]]
```

The directions saved in the project (its `directions/` folder), by name: the model and
site each was found on, how many dimensions it has, how it was made (`made_by`: the method,
the run and in words) and the hash of its tensor file; `error` says why one can't be
read.

### lg.list_local_models

```python
lg.list_local_models() -> list[dict[str, Any]]
```

The models added from folders on this machine: name, folder, hash and size.

### lg.list_runs

```python
lg.list_runs(project: Project) -> list[dict[str, Any]]
```

Every run and draft in the project, newest first, as the history lists them: archived runs
too (`archived`), and runs whose spec can't be read (`damaged`, with `error` saying why).

### lg.list_saes

```python
lg.list_saes(model: Any, search: str = '') -> Any
```

The published SAEs trained on a model (a loaded model, or its Hugging Face id), from the
catalogue of SAELens's registry that ships with Logogram (nothing is fetched): one row per SAE,
those Logogram reads exactly first, with its release and id, repository and folder, the hook
it reads, its site, layer and width, its architecture, whether Logogram reads it exactly
(`reads`: exact, refused or unknown) and if not why (`reason`), and how it was checked on
its files if it was (`verified`). `search` keeps the ones it matches, as the app's search
does: `"layer 8"`, `"16k"`, `"resid_post"`, a release's name. Load one with
[`load_sae()`](#lgload_sae) (`release=`, `sae_id=`).

### lg.load_model

```python
lg.load_model(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: str = 'float32',
    device: str = 'auto',
    process_weights: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    source: str | None = None,
) -> Any
```

Download (once) and load a model, checked as the app checks it. Pass it to [`run()`](#lgrun) to
reuse it across runs; every choice it was loaded with is recorded in each run's spec.

`model_id` is a Hugging Face id, or the name a local model was added under (see
[`add_local_model()`](#lgadd_local_model)): `source` says which ("hub" or "local"); by default a name this
machine's local models have (it has no slash) is that local model. `device` is "auto",
"cpu", "cuda" (the default GPU), "cuda:N" (one GPU of several) or "mps".

### lg.load_run

```python
lg.load_run(project: Project, run_id: str, *, model: Any = None) -> Run
```

A run of `project` by its id (the name of its experiments/ folder).

### lg.load_sae

```python
lg.load_sae(
    repo: str | None = None,
    path: str = '',
    *,
    model: Any,
    revision: str | None = None,
    release: str | None = None,
    sae_id: str | None = None,
) -> Any
```

Download (once) and read a published SAE or transcoder at an exact revision, onto
`model`'s device: by its Hugging Face repository and folder, or by its SAELens release and
id (`release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre"`), found in the
catalogue of SAELens's registry that ships with Logogram (see [`list_saes()`](#lglist_saes)). Either way
the SAE is its repository, folder and revision, which is what a spec records; without a
revision, the repository's current commit is pinned. It is refused if Logogram can't read it
exactly, was made for another model's sizes or reads a site the model doesn't have. Pass it to
[`run()`](#lgrun) with a spec that names it, or explore its features with [`feature_writes()`](#lgfeature_writes).

### lg.load_spec

```python
lg.load_spec(data: dict[str, Any] | str | Path) -> Spec
```

A spec from a dict, a JSON string or a spec.json file. Version 1 specs are read too; the
run they make lists the values they left out.

### lg.load_sweep

```python
lg.load_sweep(project: Project, sweep_id: str, *, model: Any = None) -> Sweep
```

A sweep of `project` by its id, as its runs record it: its runs in the order of its values
(a value whose run is gone, or never ran, has none).

### lg.look

```python
lg.look(
    model: Any,
    text: str,
    *,
    prepend_bos: bool = True,
    top_k: int = 5,
    position: int | None = None,
    chat_template: dict[str, Any] | None = None,
    sae: Any = None,
) -> dict[str, Any]
```

Read any text with `model` (from [`load_model()`](#lgload_model)), without making a dataset: its
tokens, the `top_k` most likely next tokens after each, and at `position` (the last token
by default) what each layer would predict, as the app's Your text view shows. With
`chat_template` (as a spec writes it, such as `{"kind": "user_turn", "system": None}`,
with `prepend_bos` false), the text is wrapped in the model's own chat template. With
`sae` (from [`load_sae()`](#lgload_sae)), `features` also lists the `top_k` features that fire most
on each token, with their activations, and `active` how many fire there (SAEs aren't
trained on the beginning-of-sequence token, so read from `first_real_token` on).

### lg.mean_difference

```python
lg.mean_difference(
    model: Any,
    project: Project,
    source: Any,
    site: Any,
    position: Any,
    positive: str,
    negative: str,
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = 'both',
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
) -> Any
```

The mean activation at `site` and `position` over the rows of class `positive`
minus the mean over those of class `negative` (read as [`activations()`](#lgactivations) reads them; a
dataset's rows take their class from `mark`, `"side"` or `"meta:FIELD"`, a corpus's from
their labels), as a NumPy vector. Save it as a direction with [`save_direction()`](#lgsave_direction)
(`method="mean_difference"`).

### lg.open_example

```python
lg.open_example(example: str = 'ioi-gpt2', parent: str | Path | None = None) -> Project
```

Copy a bundled example into a folder of its own inside `parent` (by default the projects
folder the app uses, Documents/Logogram) and open it, as the app's examples do: a copy made
before is opened as it is, never replaced. The examples are `ioi-gpt2`,
`greater-than-gpt2`, `factual-recall-gpt2` and `gendered-pronoun-gpt2`: a spec and a
dataset each, made for GPT-2 small.

### lg.open_project

```python
lg.open_project(path: str | Path) -> Project
```

Open a project folder (one with a project.json).

### lg.principal_components

```python
lg.principal_components(
    model: Any,
    project: Project,
    source: Any,
    site: Any,
    position: Any = 'last',
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = 'both',
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
) -> dict[str, Any]
```

The principal components of the activations at `site` and `position` (read as
[`activations()`](#lgactivations) reads them), as the app's Representations view shows them: the rows
centered on their mean and decomposed in float64, each component signed so its
largest-magnitude loading is positive. Returns `components` (`[k, width]`, at most ten),
`explained` (each one's share of the variance), `coords` (`[rows, k]`), `mean`,
`rows` (each row's index, side, class and text) and `settings`. Save a component as a
direction with [`save_direction()`](#lgsave_direction).

### lg.read_corpus

```python
lg.read_corpus(project: Project, name: str, sha256: str | None = None) -> Any
```

A corpus of the project: its `texts`, its `records` (each text with its label, id and
meta), `labels()` (the distinct labels, as text), the `sha256` of its file and `meta`
(where its texts came from). With `sha256`, it is refused unless its file still has that
hash.

### lg.read_direction

```python
lg.read_direction(project: Project, name: str) -> Any
```

A saved direction: its `vectors` (`[k, width]`, orthonormal rows, float32) and `meta`
(the model and site it was found on, how it was made, its hash).

### lg.remove_local_model

```python
lg.remove_local_model(name: str) -> None
```

Forget a local model. Its folder and files are left as they are.

### lg.run

```python
lg.run(
    experiment: Spec | dict[str, Any],
    project: Project,
    *,
    model: Any = None,
    sae: Any = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> Run
```

Run a spec and write its experiment folder into `project`, as the app does. `model`
is reused when it matches the spec (otherwise the spec's model is loaded).

### lg.save_direction

```python
lg.save_direction(
    project: Project,
    name: str,
    vectors: Any,
    *,
    model: Any,
    site: Any,
    method: str = 'import',
    detail: str = '',
    notes: str = '',
    overwrite: bool = False,
) -> Any
```

Save `vectors` (one vector `[width]`, or `[k, width]` for a subspace; a NumPy array,
a tensor or a list) as the direction `name`, made orthonormal, with the length of each
vector before kept. `model` is the loaded model it was found on (from [`load_model()`](#lgload_model)),
`site` where (`"resid_post:6"`, or None), `method` how (`"pca"`,
`"mean_difference"`, `"probe"`, `"das"`, `"sae_feature"` or `"import"`) and
`detail` in words. Steering, ablation along it and the analyses that read directions use
it by name. An existing direction of that name is replaced only with `overwrite`.

### lg.sweep

```python
lg.sweep(
    experiment: Run | Spec | dict[str, Any] | str | Path,
    project: Project | None = None,
    *,
    field: str,
    values: Sequence[Any],
    model: Any = None,
    sae: Any = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> Sweep
```

Run a run's spec (or a spec, into `project`) across `values` of one `field`, one run
after another, as `logogram sweep` does: a seed the spec states (`statistics.seed`,
`experiment.baseline.seed`, `experiment.seed` or `scope.seed`), the dataset
(`dataset.path`: dataset files of the project) or the model's revision (`model.revision`:
commits, tags or branches, each run pinning its commit). A finished run stands in for its own
value, which isn't run again. Returns the runs, one per value in the order given; a run that
doesn't finish is kept with its status, and the next value runs.

### lg.top_examples

```python
lg.top_examples(
    model: Any,
    project: Project,
    corpus: str,
    unit: dict[str, Any],
    *,
    sae: Any = None,
    texts: int | None = None,
    max_tokens: int = 128,
    prepend_bos: bool = True,
    batch_size: int = 16,
    top_k: int = 20,
    context: int = 8,
) -> dict[str, Any]
```

Where a unit is most active in the corpus `corpus`, as the app's Corpus view shows it.
`unit` is `{"kind": "neuron", "layer": 3, "neuron": 12}`, `{"kind": "sae_feature", "feature": 1234}` (a feature of `sae`, from [`load_sae()`](#lgload_sae)), `{"kind": "direction", "name": "sides"}` (a saved direction, read at its site, or at `"site": {"kind": "resid_post", "layer": 6}`) or `{"kind": "head", "layer": 9, "head": 9}` (the length of
what the head writes into the residual stream).

The first `texts` texts (every text when None) are read, each cut to its first
`max_tokens` tokens, after the beginning-of-sequence token with `prepend_bos` (that token
is never counted). Returns the `top_k` tokens where the unit is largest (`top`), each with
`context` tokens on either side and the unit's value at each, for a signed unit (a direction
of one dimension) the `top_k` most negative (`bottom`), a `histogram` of its values, the
share of tokens where it is above zero (`active_share`), and the `settings` that made
them. Ties go to the earlier text, then the earlier token.

### lg.toy_model

```python
lg.toy_model(
    name: str,
    folder: str | Path | None = None,
    *,
    device: str = 'cpu',
    dtype: str = 'float32',
) -> ToyModel
```

A model whose circuit is known, to learn the methods on and to check them against:
`"induction"` (a previous-token head and an induction head) or `"ioi"` (GPT-2 small's
indirect-object circuit, one head per step), with weights set by hand (see
`logogram.toys`). Writes its project into `folder` (by default `toy-NAME` in the
projects folder): the model's files in `model/`, added to this machine's local models as
`toy-NAME` so specs name it by the hash of its files, its datasets, and experiments whose
notes say what each method should find. Then loads it, checked as any model is, and checks
that it does its task and that its planted heads attend where they were built to; a toy that
doesn't is refused. Run its experiments with `toy.run("which-heads")`.

### lg.verify_run

```python
lg.verify_run(estimated: Run, top: int = 10, *, model: Any = None) -> Run
```

Patch, for real, the `top` sites an attribution patching run estimated to matter most;
for an edge attribution run, path patch its `top` strongest edges one at a time, at most 50
(compare the two with [`edge_verification()`](#lgedge_verification)).

### lg.write_corpus

```python
lg.write_corpus(
    project: Project,
    name: str,
    texts: Sequence[str],
    labels: Sequence[Any] | None = None,
    *,
    overwrite: bool = False,
) -> Any
```

Write `texts` as the corpus `name`, each with the label at the same place in
`labels` (a string, a whole number or true/false) when they are given; empty texts are left
out. Returns the corpus (as [`read_corpus()`](#lgread_corpus) does).

### lg.write_dataset

```python
lg.write_dataset(
    project: Project,
    name: str,
    records: list[Any],
    *,
    overwrite: bool = False,
) -> str
```

Write prompt pairs into the project's datasets/ folder; returns its path in the project.

## Classes

### lg.Run

A run in a project: its spec, summary and manifest as written, and its per-prompt values.
`model` is the loaded model it was run with, when there is one, for the runs made from
it ([`verify_run()`](#lgverify_run), [`check_robustness()`](#lgcheck_robustness), [`sweep()`](#lgsweep)).

Its attributes: `project` (Project), `id` (str), `spec` (Spec), `summary` (dict[str, Any] | None), `manifest` (dict[str, Any]) and `model` (Any).

#### Run.folder

```python
run.folder  # Path
```

The run's folder: experiments/ and its id, in the project.

#### Run.status

```python
run.status  # str
```

`finished`, `failed`, `cancelled`, `running`, or `draft` for a spec that
hasn't run. A run whose process stopped before it finished reads as `failed`.

#### Run.traceback

```python
run.traceback  # str | None
```

What a failed run kept of the error that stopped it (this machine's paths scrubbed).

#### Run.partial

```python
run.partial  # dict[str, Any] | None
```

For a run that was cancelled or failed mid-sweep and kept what finished: how far it
got (`note`, such as "Stopped after 7 of 12 layers: these are the layers that
finished.") and what it doesn't keep (`left_out`). Its sites, statistics and per-prompt
values are those of what finished, as a finished run's are. None for a finished run.

#### Run.title

```python
run.title  # str
```

What the run is called: the title it was given (see `rename()`), or its spec's
name.

#### Run.starred

```python
run.starred  # bool
```

Whether the run is starred (see `star()`).

#### Run.archived

```python
run.archived  # bool
```

Whether the run is archived (see `archive()`).

#### Run.rename

```python
run.rename(title: str | None) -> Run
```

Give the run a title, shown instead of its spec's name (None or "" clears it). The spec
never changes: the title is kept beside it, in annotations.json.

#### Run.star

```python
run.star(starred: bool = True) -> Run
```

Star the run, so the history finds it first; `star(False)` removes the star.

#### Run.archive

```python
run.archive(archived: bool = True) -> Run
```

Keep the run out of the history until it is asked to show archived runs; nothing is
deleted. `archive(False)` brings it back.

#### Run.sites

```python
run.sites() -> Any
```

One row per site: what it is, its mean effect and interval, the simultaneous band and
q-value, sign flips and the metric patched. An edge attribution run's summary lists its
strongest edges only: `edges()` has every one.

#### Run.methods

```python
run.methods() -> str
```

The methods paragraph: every choice that can change a number, in sentences, as the app
copies it.

#### Run.report

```python
run.report(path: str | Path | None = None, *, hardware: bool = False) -> str
```

The run as one self-contained HTML page anyone can open: its map, strongest sites,
every site's numbers, the methods and where it came from. Written to `path` when given;
hardware model names are left out unless `hardware` is true.

#### Run.continuations

```python
run.continuations() -> dict[str, Any] | None
```

What a steering run's model went on to say (its continuations.json): for each of the
first held-out prompts, the continuation without steering and one per steered site and
strength (and random control). None for a run that recorded none.

#### Run.results

```python
run.results() -> Any
```

results.parquet as a pyarrow Table: one row per site and prompt.

#### Run.per_prompt

```python
run.per_prompt() -> Any
```

The per-prompt values every statistic came from, as a DataFrame (or a pyarrow Table).

#### Run.site

```python
run.site(index: int) -> dict[str, Any]
```

One site's evidence: each prompt's effect and values, the strongest and weakest.

#### Run.strongest

```python
run.strongest(k: int = 10) -> list[dict[str, Any]]
```

The `k` sites with the largest mean effects, by magnitude, as summary.json holds
them: each site's label, effect with its interval, and the rest of its statistics.

#### Run.edges

```python
run.edges() -> Any
```

An edge attribution run's every edge, one row each (its edges.parquet): the sender and
receiver, the mean estimated effect and its interval, the simultaneous band and q-value.
The summary lists only the strongest.

#### Run.curve

```python
run.curve() -> dict[str, Any] | None
```

An edge attribution run's faithfulness curve: for each k, the faithfulness of the k
strongest edges and of k edges chosen at random, with intervals; None without one.

### lg.Sweep

A sweep's runs, one per value in the order given (see [`sweep()`](#lgsweep)): a list of
[`Run`](#lgrun-1), with the sweep's `id`, `field` and `values`.

#### Sweep.summary

```python
sweep.summary(choose: str = 'first', top: int = 10) -> dict[str, Any]
```

The sweep as the app's sweep view shows it: for the `top` strongest sites (0: all),
chosen on the `"first"` value or on `"all"` values, each site's effect and interval
at every value, the spread of its means across values, and whether it keeps its sign and
size (`verdict`: holds, size, sign or unknown).

#### Sweep.table

```python
sweep.table(choose: str = 'first', top: int = 10) -> Any
```

`summary()` as a table, one row per site and value (as the app exports it). A
value without results has no numbers, never zeros, and says why in `status`.

## Errors

### lg.api.LogogramError

A run that didn't finish, or a request that can't be met; the message says why.
