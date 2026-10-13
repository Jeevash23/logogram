# Runs and projects

A project is an ordinary folder you can commit, share and rerun. Every run in it is a folder of
plain files: the spec that says what ran, the per-prompt values every statistic came from, and a
manifest of where and how it ran.

## A project's files

```text
my-project/
  project.json
  research.json                      saved selections and research notes (created on first save)
  datasets/*.jsonl                   prompt pairs, one per line
  datasets/snapshots/<sha256>.jsonl  the exact bytes each run read, shared by runs of the same data
  experiments/<id>/spec.json         the experiment, with its model revision and dataset hash pinned
  experiments/<id>/results.parquet   one row per site and prompt
  experiments/<id>/summary.json      each site's statistics, and the layout of the results
  experiments/<id>/manifest.json     versions, device, dtype, model revision, dataset hash, timing
  experiments/<id>/annotations.json  a title, a star or the archive mark, when given
  experiments/<id>/predictions.json  the optional layer predictions (logit lens or tuned lens)
  experiments/<id>/continuations.json  what a steered model went on to say, when asked
  experiments/<id>/traceback.txt     a failed run's traceback, without this machine's paths
  saes/<name>/                       an SAE trained in the project, in SAELens's format, with its
                                     training spec and what it measured
  lenses/<name>/                     a tuned lens trained in the project, with its training spec
                                     and what it measured
  .logogram/queue.json               the project's run queue (git ignores it)
```

Paths inside a project are relative, so a project can be moved, committed and opened on another
machine. A run snapshots its dataset before the model loads and pins that snapshot in the spec it
saves, so editing the original dataset leaves every run reproducible; a snapshot that changed is
refused by its hash. A spec never changes after it runs: a run's title, star and archive mark
live in `annotations.json` beside it.

## Reproducibility

The same spec on the same machine gives bit-identical results. Runs seed Python, NumPy and
PyTorch from `statistics.seed`, require PyTorch's deterministic algorithms (an operation without
one fails rather than warns), turn off TF32, and keep each batch's composition fixed, because
batch shape can change floating-point results in the last digits: that is why
`execution.batch_size` is part of the spec. `logogram run` on a finished run's spec checks it:

```text
Identical to 20261007-203446-which-heads-restore-the-answer: all 4,608 per-prompt values match exactly.
```

On another device (the CPU, CUDA, MPS or another GPU) or with other library versions, results can
differ in the last digits, because the same sums run in another order. Each manifest records the
versions of Logogram, Python, PyTorch, TransformerLens, Transformers, NumPy and pyarrow, the device
and its model; the Results view says when they differ from this machine's, and **Compare runs**
says when two runs differ in them. `logogram env` prints this machine's, and `logogram env
--requirements` pins them.

## A run that stops early

A sweep streams its results as it goes, and a run that is cancelled (in the app, with Ctrl+C on
the command line, or by interrupting a notebook's kernel), or that fails partway, keeps what
finished:

* a sweep that goes layer by layer (patching, ablation, path patching), steered site by steered
  site, or set by set keeps the layers, sites or sets every prompt has run at: "Stopped after 7 of
  12 layers: these are the layers that finished.";
* attribution patching and direct logit attribution, which measure every site in one pass per
  batch of prompts, keep every site over the prompts that finished;
* SAE feature sweeps report nothing before they end, and keep nothing.

A partial run opens like a finished one, with nothing drawn for the rest, and its status stays
cancelled or failed; the history marks it partial. Its kept sites have the numbers the whole run
would give them: the same per-prompt values, resamples and normalization. What depends on every
site is left out, and the summary says so: simultaneous bands and q-values (which correct for
every site of the sweep together), and comparisons with sets that didn't finish. A steered site is
kept with its random control. A partial run can be compared and exported; checking its robustness
or verifying it needs a finished run.

## Runs elsewhere

A run started by `logogram run`, `logogram queue run` or a notebook lists as running in another
process until it ends, and the app notices when it does. `owner.json`, written while a run runs,
holds the process's id and start time and a random token, nothing about you or the machine, so
that a run whose process stopped (closed, killed or crashed) is told apart from one still going:
it is marked failed, with its spec ready to run again.

## Stability

Each release reads the run folders earlier releases wrote: their specs, manifests, summaries and
per-prompt results list, open, compare and export as before, and the tests keep runs written by
0.1.0 and 0.2.0 to hold every release to this. Version 1 and 2 specs still run (see
[the spec reference](../reference/spec.md)).
