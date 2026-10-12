# Sweep seeds, datasets or checkpoints

A run's intervals say how much its result would move with other prompts like its own. They leave
out everything else that could change it: the seed of a resample ablation or of a split, which
prompts a generator happened to write, or the point in training a checkpoint came from. A
**sweep** runs one spec once for each of several values of one choice, and shows which of its
strongest sites keep their sign and size across them.

## What can be swept

* **A seed the spec states**: `statistics.seed` (the bootstrap's), `experiment.baseline.seed`
  (resample ablation's donors), `experiment.seed` (steering's split) or `scope.seed` (an SAE
  features sweep's split). A spec without that seed has nothing to sweep there.
* **The dataset** (`dataset.path`): several dataset files of the project, such as the same task
  generated with several seeds. A dataset's hash follows its path, so each run pins its own file.
* **The model's revision** (`model.revision`): commits, tags or branches on Hugging Face, such as
  Pythia's training checkpoints `step1000` to `step143000`. Each run pins the commit its revision
  points to when it starts.

Each run is an ordinary spec that differs from the original in that one field, and its manifest
records the sweep: its id, the field, the run's value and its place among the values. A finished
run stands in for its own value, which isn't run again.

## In the app

1. On a finished run's results, press **Sweep…**.
2. Choose what to vary. For datasets, choose files of the project, or let Logogram generate them
   with the task's generator, one per seed. For revisions, type them, or list the repository's
   branches and tags from Hugging Face when you ask.
3. The runs wait in the [queue](queue.md) and run one after another. **Evidence → Sweeps** shows
   the sweep as its runs finish.

## Reading a sweep

The Sweeps view shows the strongest sites, chosen on the first value or on all of them (it says
which), and for each site:

* its effect and interval at every value, drawn in the order given (revisions read as a
  trajectory over training);
* the **spread** of its means across values, beside a run's own interval: when the spread is wider
  than the interval, something the interval leaves out moves the result;
* what it does across the values, in words:

| Verdict | When |
|---|---|
| Keeps its sign and size | No two of its intervals exclude each other. The view also says whether its interval excludes zero at every value, some or none |
| Changes size | Two of its intervals exclude each other |
| Flips sign | It is above zero at one value and below zero at another, with both intervals excluding zero |
| Unknown | It was measured at fewer than two values |

A value whose run failed, was cancelled or never ran is **missing**, never zero, and the view says
why. Choose a point to open that value's run at the site; the arrow keys move between sites and
values. The table exports as CSV, one row per site and value.

**Choosing the sites** is where the [winner's curse](../concepts/statistics.md#the-winners-curse)
enters: sites chosen as the strongest on a value look stronger there than they are. Choosing on
the first value lets the others show the chosen sites without that bias; choosing on all values
shows every site strong anywhere, and is biased everywhere.

## From the command line

```bash
logogram sweep RUN --field statistics.seed --values 0,1,2,3           # the bootstrap's seed
logogram sweep RUN --field experiment.baseline.seed --values 0,1,2,3  # resample ablation's donors
logogram sweep SPEC.json --datasets ioi-seed1.jsonl,ioi-seed2.jsonl,ioi-seed3.jsonl
logogram sweep RUN --revisions step1000,step8000,step64000,step143000  # Pythia's checkpoints
```

It runs the values in turn, then prints which of the strongest sites keep their sign and size.
`--choose all` chooses the sites on every value, `--top N` shows more of them, and `--csv FILE`
writes every shown site at every value.

## From Python

```python
seeds = lg.sweep(run, field="statistics.seed", values=[0, 1, 2, 3])
seeds.summary()   # what the Sweeps view shows: each site at every value, and its verdict
seeds.table()     # one row per site and value
again = lg.load_sweep(project, seeds.id)
```

## Two examples

**Is the IOI circuit a property of these prompts?** Generate the IOI task with seeds 1, 2 and 3
(**Prompts → Generate a task**), and sweep the example's run over the three datasets. Heads that
keep their sign and size carry the behavior on any prompts the generator writes; one that
changes size depends on the sample.

**When does a circuit form?** Run a spec on Pythia-160m and sweep `model.revision` over
`step1000`, `step8000`, `step32000` and `step143000`. Generate the prompts with the model loaded,
so every name is one token for Pythia's tokenizer, and check the baseline at an early checkpoint:
a model that doesn't do the task yet has a gap too small to normalize by.
