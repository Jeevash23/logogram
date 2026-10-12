# Logogram

A local workbench for causal experiments inside language models.

In *Arrival*, an unfamiliar language is decoded by running small, controlled experiments and
watching what changes. Activation patching works the same way: replace one activation with its
value from a different input, and measure what happens to the output. Logogram makes those
experiments visual, fast and rigorous. Load GPT-2, set up a clean/corrupted prompt pair, sweep
every attention head, watch the results paint onto a map of the model, then click a head and see
the evidence behind its number.

* **A research instrument, not a demo.** Every number is traceable to the method that produced
  it: the intervention, baseline, direction, position, metric, normalization, n and confidence
  interval are always shown.
* **Local-first.** No account, no cloud, no telemetry. Everything runs on your machine.
* **Plain files.** Experiments are folders you can commit, share and rerun.
* **One spec, two front ends.** The app and `logogram run` execute the same `spec.json`, and
  produce identical results on the same machine.

The [documentation](docs/README.md) teaches the app and the methods, from a first run to the exact
definition of every number.

## Install

Logogram needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install logogram
```

To run the code in a clone of this repository instead, install it with `uv tool install .`; the
commands in the table below work the same way with `.` in place of `logogram`. The web app is
prebuilt inside the package, so you never need Node. What changed in each version is in
[CHANGELOG.md](CHANGELOG.md).

PyTorch is chosen per machine:

| Machine | Install | Notes |
|---|---|---|
| Linux with an NVIDIA GPU | `uv tool install logogram` | The default Linux wheels include CUDA. |
| Windows with an NVIDIA GPU | `uv tool install --torch-backend=auto logogram` | Picks the CUDA build that matches your driver. |
| Apple Silicon | `uv tool install logogram` | Runs on the CPU unless you choose MPS (see Models). Use a native arm64 Python, not one under Rosetta. |
| CPU only | `uv tool install --torch-backend=cpu logogram` | Smaller download. GPT-2 small runs well on a CPU. |

Then check the setup:

```bash
logogram doctor
```

It reports the operating system, CPU, memory, GPU, video memory, compute backend and recommended
precision, and gives the exact command to fix common problems, such as an NVIDIA GPU with a
CPU-only PyTorch build.

## Quickstart

```bash
logogram
```

This starts a local server on 127.0.0.1, prints its address and opens your browser.

1. The first time, a system check shows what Logogram found, and the [examples](#examples). Open
   the IOI example. Logogram copies it to `Documents/Logogram/ioi-example`, an ordinary project
   folder.
2. The example contains 32 indirect object identification (IOI) prompts and a ready-made
   experiment: patch each head's output from the clean prompt into the corrupt prompt. Press
   **Run**. GPT-2 small (about 0.5 GB) downloads once from Hugging Face; the sweep then takes
   about a minute on a CPU, a few seconds on a GPU.
3. Results paint onto the model map layer by layer. Click the strongest heads (L9 H9, L10 H7,
   L8 H6 …) to see, in the inspector, the exact intervention, the metric, the confidence
   interval, how many prompts flipped, the distribution of per-prompt effects and the prompts
   with the strongest and weakest effects.

   Each run also gets a **logogram**, a circular ink glyph written by its results: layers run
   clockwise from the top, and the ink swells outward (cobalt) where a layer's strongest effect
   is positive and inward (ochre) where it is negative. A run with a few strong layers reads as
   a heavy, lopsided ring; a null result as a thin, even one. Logograms appear in the history,
   on the results page, and beside the model atlas, where each layer number opens that layer's
   strongest component.
4. Press **A** on a head to see its attention pattern, or right-click any cell for **Patch here**,
   **Ablate here** and **Compare across runs**.
5. Press **Check robustness** to rerun the sweep with a different baseline, direction or donor
   count, or, for a run in float16 or bfloat16, in float32. Logogram reports the rank
   correlation, the overlap of the top components and the components whose conclusion changed,
   and flags them on the map.

[Getting started](docs/getting-started.md) takes these steps in more detail, and
[the workbench](docs/workbench.md) describes every view: the keyboard (**?** lists every shortcut),
**History**, the queue, comparisons, sweeps, runs that stop early and what they keep, and the HTML
report, figures and CSV a finished run exports. **Help**, in the top bar, opens a short guide, a
walkthrough of the IOI example at its own controls, the keyboard shortcuts, the documentation on
GitHub, a problem report to paste into an issue (nothing is sent: you copy it), the system check,
and About.

## Documentation

The [documentation](docs/README.md) teaches the app and the methods:

* [Getting started](docs/getting-started.md): install, open the IOI example, run it, and read the
  map, the inspector and the logogram. [The workbench](docs/workbench.md) says what each view is
  for, and [Models](docs/models.md) which models load and how.
* [Concepts](docs/README.md#concepts): what each method computes, exactly, and the papers it
  comes from: activation patching, ablation, metrics and normalization, statistics, attribution
  patching, path patching, direct logit attribution, steering, SAE features, circuits, and the
  logit lens with the checks each model passes when it loads.
* [How-to guides](docs/README.md#how-to-guides): replicate IOI, find induction heads, work with
  an SAE, check a finding's robustness, compare runs, sweep seeds, datasets or checkpoints, queue
  runs and run them headless, run a chat model on its template, study your own fine-tune, share a
  result, and use a remote GPU.
* [Reference](docs/README.md#reference), generated from the code: every field of
  [the spec](docs/reference/spec.md), every command and option of
  [the command line](docs/reference/cli.md), and every function of
  [the Python API](docs/reference/python-api.md).
* [Notebooks](notebooks/README.md), which open in Colab: IOI head patching with its statistics,
  an attribution-patching estimate checked by patching, and an SAE's features on GPT-2 small.

## Examples

Four example projects come with Logogram. Each is a spec and a dataset made by Logogram's own task
generators for GPT-2 small, runs on a CPU in about a minute, and asks a question a paper answered
by a causal experiment, so its result can be read against the paper. The first screens and **All
projects** list them with the question, the model, the method and the result to expect. Opening
one copies it into its own folder in `Documents/Logogram`; a copy made before opens as you left
it, never replaced. Each spec's notes say what was published and what Logogram measured on GPT-2
small (revision `607a30d`), and the results show them under the run's title.

| Example | Experiment | Published | Measured on GPT-2 small |
|---|---|---|---|
| Indirect object identification: which heads carry the name? | Patch each head's output clean → corrupt at every position; logit difference | Name movers 9.9 and 10.0, S-inhibition heads 7.3, 7.9, 8.6 and 8.10, induction heads 5.5 and 6.9; negative name movers 10.7 and 11.10 (Wang et al. 2022) | 5.5 0.33, 8.6 0.30, 8.10 0.25, 7.9 0.21, 9.9 0.17, 10.0 0.11; 10.7 −0.45, 11.10 −0.21 |
| Greater-than: which layers predict a year after 1732? | Patch each layer's attention and MLP output at the last token from the prompt starting in 01 (corrupt → clean); probability difference | MLPs 8 to 11 compute it, 9 and 10 most, after heads in layers 5 to 9 bring the start year (Hanna et al. 2023) | MLP 9 0.36, MLP 10 0.34, MLP 8 0.20, attention 9 0.16, MLP 11 0.09; layers 0 to 4 nothing |
| Factual recall: where does France become Paris? | Patch the residual stream before each layer at the country's last token and at the last token, clean → corrupt; logit difference | An early site at the subject's last token, a late site at the last token (Meng et al. 2022) | The country restores 0.93 to 0.95 through layer 6 and 0.82 at layer 9; the last token takes over in layers 10 and 11 (0.48, 0.78). Unlike the paper's MLP windows, single middle-layer MLPs restore 0.04 or less each |
| Gendered pronouns: which heads write “he”? | Direct logit attribution of every head at the last token | Heads 10.9 and 9.7 (Mathwin et al. 2023, in Conmy et al. 2023) | 10.9 0.35 and 9.7 0.22 of the logit difference, the next head 0.10 |

Once a week, the Golden workflow runs every example on GPT-2 small's real weights and checks these
findings (see Development).

## Explore, experiment, evidence

The workbench has three workspaces, described view by view in [the workbench](docs/workbench.md):

* **Explore** looks inside the model: the model map with the model's real dimensions, a layer's
  diagram connected the way the model was measured to add attention and the MLP, attention
  patterns and the [head finder](docs/guides/find-induction-heads.md), head comparison, layer
  predictions (a [final-norm logit lens](docs/concepts/logit-lens-and-model-checks.md), offered for
  models where it reproduces their output), an [SAE's features](docs/concepts/sae-features.md), and
  **Your text**, which reads any text and continues it, with and without an SAE feature added.
* **Experiment** sets up what runs: the prompt pairs, the baseline check, the experiment form,
  which shows every choice that can change a number, and the complete spec. Choose components on
  the map and **Add to experiment** to measure them one by one, or **Test as a circuit** to
  intervene on them together.
* **Evidence** reads what happened: each run's results with their statistics, the inspector's
  per-prompt evidence for any site, [comparisons](docs/guides/compare-runs.md) between runs,
  [sweeps](docs/guides/sweeps.md) of one spec across seeds, datasets or checkpoints, and research
  notes.

## Command line

```text
logogram                      start the app and open the browser
logogram open PATH            start with a project open
logogram run SPEC.json        run an experiment headlessly and print a summary
logogram validate SPEC.json   check a spec without running it, and say what it left out
logogram list                 the project's runs and drafts, newest first (--archived for all)
logogram rename RUN TITLE     show a title instead of the spec's name ("" clears it)
logogram star RUN             star a run (--remove takes the star off)
logogram archive RUN          keep a run out of the list (--restore brings it back)
logogram delete RUN           delete a run's folder for good, after asking (--yes doesn't ask)
logogram show RUN             a finished run's method, baseline and strongest sites, or why it failed
logogram compare RUN RUN      how two runs agree, with paired differences where they can be paired
logogram diff A B             every choice that differs between two runs or spec files
logogram verify RUN           patch the strongest sites of an attribution patching run (path-patch
                              an edge attribution run's strongest edges), and compare
logogram robustness RUN       rerun with one choice changed (--experiment JSON, or --dtype)
logogram sweep SPEC_OR_RUN    the same spec across seeds, datasets or revisions, one run each (--csv FILE)
logogram export RUN           a run's per-prompt values as CSV (-o FILE; --sites, --prompt-text)
logogram report RUN           a run as one HTML page anyone can open (-o FILE, --hardware)
logogram methods RUN          a run's methods paragraph
logogram queue add SPEC.json  add a spec to the project's queue (the app shows the same queue)
logogram queue list           the queued runs, and whether the queue is paused or who runs it
logogram queue clear          take every run out of the queue, after asking (--yes doesn't ask)
logogram queue run            run the queue to the end, one run at a time, without the app
logogram tasks                the tasks Logogram writes prompts for, with their options
logogram saes MODEL [WORDS]   the published SAEs for a model, and whether Logogram reads each exactly
logogram generate TASK        a seeded dataset for a task (-n N --seed S, --option NAME=VALUE)
logogram models add PATH --name NAME   add a model folder on this machine (its files are hashed)
logogram models list          the model folders added here, with their hashes
logogram models remove NAME   forget a model folder (its files stay)
logogram doctor               environment and hardware report, with fixes (each GPU by its cuda:N)
logogram env                  the versions and devices results depend on, as JSON (--requirements pins them)
logogram doctor --report      a short report to paste into an issue (--hardware adds the hardware)
logogram check-updates        ask PyPI whether a newer Logogram is out
logogram serve --dev          API only, for working on the web app
logogram --version
```

The app listens on port 8765, or the next free one; `--port N` picks another (`--port 0` takes
any free port) and `--no-browser` skips opening the browser. `--forwarded-port N` accepts the
address of a port that forwards to the app, such as an SSH tunnel's (see
[Running on a remote GPU](#running-on-a-remote-gpu)). Only errors reach the terminal; `--verbose`
also logs what the server does, with the traceback of each failed run, and `--log-file PATH`
appends the same to a file. Commands that read runs use the project in the current folder or
above it, or `--project PATH`; a run is its id or its folder.

A failed run keeps its traceback in `traceback.txt`, without this machine's paths or names:
`logogram show RUN` prints it, `logogram run --verbose` prints it as the run fails, and the app
shows it under **Show details**. Ctrl+C during `logogram run` records the run as cancelled, and
keeps what finished; `logogram show RUN` prints a partial run with what it kept and what it
doesn't, and `logogram list` marks it.

`logogram queue run` works through the project's queue, overnight on a remote GPU say: each run's
progress, then how it ended. A run that fails doesn't stop the queue, one that can't start (its
dataset is gone) leaves it, and runs added meanwhile, here or in the app, run too. Ctrl+C stops
the run in progress (it keeps what finished) and leaves the rest queued. While it runs the queue,
the app shows the queue as running in another process and won't start it too; while the app runs
it, `logogram queue run` says so and stops.

`logogram run` writes a new experiment folder next to the original. When the spec came from a
finished run, it also checks the new results against the old ones:

```text
Identical to 20261007-203446-which-heads-restore-the-answer: all 4,608 per-prompt values match exactly.
```

`logogram generate` makes prompts for a model's tokenizer when you name one already in the Hugging
Face cache (`--tokenizer openai-community/gpt2`); it never downloads one. Without it, words are
assumed to be single tokens, so generate the dataset again in the app once the model is loaded,
which keeps only words that are single tokens for it.

`logogram sweep` runs one spec once for each of several values of one choice, one run after
another, then prints which of its strongest sites keep their sign and size across the values:

```text
logogram sweep RUN --field statistics.seed --values 0,1,2,3           # the bootstrap's seed
logogram sweep RUN --field experiment.baseline.seed --values 0,1,2,3  # resample ablation's donors
logogram sweep SPEC.json --datasets ioi-seed1.jsonl,ioi-seed2.jsonl,ioi-seed3.jsonl
logogram sweep RUN --revisions step1000,step8000,step64000,step143000  # Pythia's checkpoints
```

Each run is an ordinary spec that differs from the original in that one field (a dataset's hash
follows its path), and its manifest records the sweep. A finished run stands in for its own value,
which isn't run again. A revision runs as the commit it points to when its run starts, and the
run's spec pins that commit. The seeds a sweep can vary are the ones a spec states:
`statistics.seed`, `experiment.baseline.seed` (resample ablation), `experiment.seed` (steering's
split) and `scope.seed` (the features' split). Sites are chosen on the first value unless
`--choose all` chooses them on all values; `--top N` shows more of them, and `--csv FILE` writes
each shown site at every value.

## Python

The same runs from a notebook or a script. They go through the same code as the app and
`logogram run`, and land in the project like any other run. Importing `logogram` loads no
PyTorch; functions load what they need.

```python
import logogram as lg

project = lg.open_project("~/Documents/Logogram/ioi-example")
model = lg.load_model("openai-community/gpt2")   # load once, pass it to every run
spec = lg.load_spec(project.root / "experiments/ioi-head-patching/spec.json")
run = lg.run(spec, project, model=model)

run                  # in a notebook: the method, a heatmap and the strongest sites
run.sites()          # one row per site: mean effect, interval, simultaneous band, q-value, …
run.per_prompt()     # one row per site and prompt: everything the statistics came from
run.methods()        # the methods paragraph, in sentences
run.report("run.html")   # the run as one HTML page anyone can open
run.strongest(5)

estimate = lg.run(lg.load_spec({**spec.model_dump(), "experiment": {
    "kind": "attribution_patching", "direction": "clean_to_corrupt",
    "method": "integrated_gradients", "steps": 8}}), project, model=model)
checked = lg.verify_run(estimate, top=10)     # patch the strongest estimates for real
lg.compare_runs(estimate, checked)            # rank correlation, paired differences
edges = lg.run(lg.load_spec({**spec.model_dump(), "scope": {"kind": "graph", "layers": None,
    "logits": True}, "experiment": {"kind": "edge_attribution", "direction": "corrupt_to_clean",
    "method": "gradient", "steps": None, "curve": {"counts": [10, 100, 1000], "patch": "rest",
    "control": {"seed": 0, "draws": 1}}}}), project, model=model)
edges.edges()        # every edge: its estimate, interval, simultaneous band and q-value
edges.curve()        # the faithfulness of the k strongest edges, and of k chosen at random
lg.edge_verification(edges, lg.verify_run(edges, top=20))   # each edge against path patching
lg.check_robustness(run, experiment={"kind": "ablation", "baseline": {"kind": "zero"}})
seeds = lg.sweep(run, field="statistics.seed", values=[0, 1, 2, 3])  # one run per value
seeds.table()        # the strongest sites at every value, and whether they keep sign and size

records = lg.generate("greater_than", 200, seed=0, model=model)
lg.write_dataset(project, "greater-than", records)

sae = lg.load_sae("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre", model=model)
lg.list_saes(model, "layer 8")   # SAELens's registry for this model, and what Logogram reads of each
sae = lg.load_sae(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", model=model)
lg.feature_writes(model, sae, 1234, "When Mary and John went to the store, John gave a drink to",
                  prepend_bos=True)   # the tokens the feature promotes and suppresses directly

lg.add_local_model("~/models/my-fine-tune", "my-fine-tune")   # hash a model folder, once
tuned = lg.load_model("my-fine-tune", device="cuda:1")         # a local model, on the second GPU
chat = lg.load_model("Qwen/Qwen2.5-0.5B-Instruct")
lg.look(chat, "The capital of France is", prepend_bos=False,
        chat_template={"kind": "user_turn", "system": None})   # read in its own chat template

lg.continue_text(model, "When Mary and John went to the store,", tokens=20,
                 sae=sae, feature=1234, activation=8.0)   # greedy, without and with the feature

steered = lg.run(lg.load_spec({**spec.model_dump(), "experiment": {
    "kind": "steering", "apply_to": "corrupt", "direction": {"kind": "sae_feature", "feature": 1234},
    "coefficients": [1, 4], "train_fraction": 0.5, "seed": 0, "control": True,
    "continuations": {"prompts": 3, "tokens": 20}},
    "scope": {"kind": "sites", "sites": [{"kind": "resid_pre", "layer": 8, "position": {"kind": "last"}}]},
    "sae": {"repo": "jbloom/GPT2-Small-SAEs-Reformatted", "path": "blocks.8.hook_resid_pre",
            "revision": None}}), project, model=model, sae=sae)
steered.continuations()   # what the steered model said, prompt by prompt and strength by strength
```

Tables come back as pandas DataFrames when pandas is installed, and as pyarrow Tables otherwise.
`lg.list_runs(project)` and `lg.load_run(project, run_id)` read runs made anywhere. A run that
fails raises `lg.api.LogogramError` with the reason, and `lg.load_run(project, run_id).traceback`
reads the traceback it kept. Interrupting the kernel records the run as cancelled, keeping what
finished: `run.partial` says how far a run that stopped early got and what it doesn't keep.
`run.rename("Name movers")`, `run.star()` and `run.archive()` change what the history shows
(`run.title`, `run.starred`, `run.archived`), and `lg.delete_run(project, run)` deletes a run's
folder for good. `lg.sweep` returns a sweep's runs, one per value; a run that fails keeps its
status there and the next value runs. `sweep.summary()` holds what the app's sweep view shows,
and `lg.load_sweep(project, sweep.id)` reads a sweep back.

## Running on a remote GPU

Logogram can run on a machine you reach over SSH, such as a rented GPU, while you use it in the
browser on your own computer. Everything still runs on the remote machine, which still listens on
127.0.0.1 only: SSH carries the page to you.

1. **Install it on the remote machine** and check that PyTorch sees the GPU:

   ```bash
   uv tool install logogram
   logogram doctor
   ```

   The default Linux wheels include CUDA. If the GPU isn't used, `logogram doctor` says why and
   gives the command that fixes it.

2. **Start it there without a browser.** Run it inside `tmux` or `screen`, so that it keeps
   running when the SSH connection drops:

   ```bash
   logogram --no-browser
   ```

   With a project open, that is `logogram open PATH --no-browser`. Logogram prints its address,
   with a new session token:

   ```text
   Logogram 0.3.0 is running at:

     http://127.0.0.1:8765/?token=…
   ```

   The port is 8765, or the next free one if something else has it: read it in that line.

3. **Forward the port** from your own computer, in a terminal of its own, with the port from the
   address (`gpu-box` is the remote machine, as you name it to `ssh`):

   ```bash
   ssh -N -L 8765:127.0.0.1:8765 gpu-box
   ```

4. **Open the printed address** in the browser on your own computer. Copy it from the terminal
   rather than putting the token in a command. It works until Logogram stops; each start makes a
   new token.

If port 8765 is taken on your own computer (by a Logogram of its own, say), forward another
local port, and start the remote Logogram with that port allowed:

```bash
ssh -N -L 9000:127.0.0.1:8765 gpu-box
```

```bash
logogram --no-browser --forwarded-port 9000
```

It then prints the address on port 9000 too. `logogram open` and `logogram serve` take the same
option; it can be repeated, or given as `LOGOGRAM_FORWARDED_PORTS=9000,9001`. Without it, the
page says that Logogram didn't answer (status 421) and which option to add. Only the loopback
names 127.0.0.1, localhost and [::1] are accepted, and the session token is needed as always: a
tunnel or service that puts Logogram on another address is refused by design.

**Where things live.** Everything is on the remote machine:

* Projects are in its documents folder, `~/Documents/Logogram` on Linux (the example is copied to
  `~/Documents/Logogram/ioi-example`). The app's folder browser shows the remote machine's
  folders.
* Models download into its Hugging Face cache, `~/.cache/huggingface/hub` unless `HF_HOME` or
  `HF_HUB_CACHE` says otherwise; point `HF_HOME` at a larger disk if the home folder is small
  (`logogram doctor` warns when the cache's disk is nearly full). For gated models, run
  `hf auth login` on the remote machine.
* Settings and the list of recent projects are in `~/.config/logogram`.

**Overnight runs.** Without the app, queue specs with `logogram queue add SPEC.json` and run them
in `tmux` or `screen` with `logogram queue run`, which reuses the loaded model from one run to the
next. The queue is the project's, so the app shows it too, and specs queued in the app run as well.

**Copying runs back.** A project is a folder: copy it with `rsync` or `scp`, from your own
computer.

```bash
rsync -a gpu-box:Documents/Logogram/my-project/ my-project/
```

Each run is a folder in `experiments/`, and the prompts it ran on are in `datasets/` (with the
exact bytes in `datasets/snapshots/`), so copy the whole project, or both of those folders. Inside
the copy, `logogram list` and `logogram show RUN` read its runs, and `logogram open my-project`
shows them in the app. The manifests record the remote device and GPU: a rerun on another device
can differ in the last digits, so compare per-prompt values before treating the two as the same.

## Prompts

A dataset is a JSONL file in the project's `datasets/` folder, one prompt pair per line:

```json
{"clean": "When Mary and John went to the store, John gave a drink to",
 "corrupt": "When Mary and John went to the store, Mary gave a drink to",
 "answer": " Mary", "distractor": " John",
 "positions": {"IO": [5, 9], "S1": [14, 18], "S2": [38, 42], "end": [56, 58]}}
```

* `clean` and `corrupt` usually tokenize to the same length, so positions line up. Prompts of
  different lengths need `corrupt_positions` (below).
* `answer` and `distractor` are usually single tokens (with a leading space). Either can also be a
  list of single tokens, read as one answer whose probability is their sum (in greater-than, every
  year after the start year), or text of several tokens, a continuation whose probability is the
  product of each token's, each predicted from the prompt and the tokens before it. Continuations
  need a metric other than the logit difference (see Experiments).
* `positions` is optional: named character spans in the clean prompt. A label refers to the last
  token that overlaps its span. Named positions let you patch at, say, `S2` across templates of
  different lengths.
* `corrupt_positions` gives the same labels as character spans in the corrupt prompt, as in
  `{"IO": [5, 9], "S1": [14, 18], "S2": [38, 49], "end": [63, 65]}` for a corrupt prompt that
  names "Bartholomew" where the clean one names "John". With it, the two prompts may tokenize to different lengths: a run
  matches them label by label, carrying the source prompt's token at a label (or its last token)
  into the receiver's token there. Every position and token indices have no meaning across
  different lengths, so a method that carries values between the prompts (patching, attribution
  patching with one gradient, path patching, mean and resample ablation, steering by the mean
  difference) refuses them on such pairs, naming the pairs; zero ablation, direct logit
  attribution and steering along an SAE feature read one prompt at a time and take any position
  they otherwise take. Integrated gradients, sets that keep their sites and patch the rest, and
  mean or resample ablation of sets need pairs of equal length. Pairs of equal length may give
  `corrupt_positions` too.
* `meta` is optional: anything about the prompt, such as `{"template": "went"}`. The bootstrap can
  resample prompts by one of its fields (see Statistics).

The app shows both prompts token by token, highlights where they differ, and refuses to run on
prompts that can't be used, saying which and why. Prompts of different lengths are lined up by
their labels, each with its own positions, and the prompts table shows both prompts' labels.

**Editing prompts.** The Prompts view searches a dataset's pairs (both prompts, the answers and
`meta`), shows them a hundred at a time, and edits or removes pairs, including answer sets and
positions. A dataset file never changes, since runs record its hash: edits stay in the tab, marked
as unsaved, until they are saved as a new dataset, checked as an import is.

**Tasks.** Logogram writes seeded datasets for six classic tasks, from the app's Prompts view, the
command line (`logogram tasks`, `logogram generate`) or Python (`lg.generate`). Each fills
templates whose clean and corrupt prompts have the same token length, names its positions, and
records each prompt's template in `meta`:

| Task | What the model should say | Corrupt prompt | Metric |
|---|---|---|---|
| `ioi` | the indirect object in "When Mary and John went to the store, John gave a drink to" (Wang et al. 2022) | `flip` (the repeated name becomes the other) or `abc` (three unrelated names) | logit difference |
| `greater_than` | a year after the start year in "The war lasted from the year 1732 to the year 17" (Hanna et al. 2023) | the start year becomes …01 | probability difference |
| `docstring` | the next argument of a function in its docstring (Heimersheim and Janiak 2023) | the definition's arguments renamed | log-probability difference |
| `gendered_pronoun` | " he" or " she" after a first name (Mathwin et al. 2023) | a name of the other gender | logit difference |
| `subject_verb_agreement` | " is" or " are" after a noun of the other number (Linzen et al. 2016) | the subject in the other number | logit difference |
| `factual_recall` | a country's capital | another country | logit difference (log-probability difference for capitals of several tokens) |

With a model loaded, the app keeps only words that are single tokens for it. Each task says how
many distinct prompts its templates allow; a request for more is refused. The `full_vocabulary`
option keeps every word, also those the model splits, and the pairs whose prompts then differ in
length, labelling both prompts' positions and recording `"vocabulary": "full"` in each prompt's
`meta` (IOI's names of several tokens are then read with the log-probability difference). GPT-2
small does all of them but the docstring task, where it prefers the right argument in about half
the prompts.

## Experiments

**Activation patching** copies an activation from one prompt of each pair into the run on the
other. *Clean → corrupt* runs the corrupt prompt and patches in clean activations: does this
restore the behavior? *Corrupt → clean* runs the clean prompt and patches in corrupt
activations: does this break it?

**Ablation** runs the clean prompt and replaces the activation with a baseline, which you must
choose:

* *zero*: zeros;
* *mean*: the mean over a stated reference set (the dataset's clean or corrupt prompts), per
  position when every position is replaced;
* *resample*: the same activation in other prompts. Each prompt draws `donors` donors from the
  pool without replacement, never itself, with a seed; the result is averaged over donors, and
  the same donors are used at every site.

**Attribution patching** estimates activation patching at every site at once. The change patching
would cause is estimated to first order, as (source activation − receiver activation) · the
gradient of the logit difference at the receiver run: one forward pass on the source prompts and
one forward and backward pass on the receivers cover a whole sweep, so every head at every
position takes seconds even on larger models. Values are labelled as estimates. A first-order
estimate misses saturation (in attention, normalization and the final softmax) and can miss or
even invert an effect, so **Verify top 10 by patching** on the results page patches the sites
with the largest estimated effects for real and opens the comparison: the rank correlation, and
any estimate whose sign patching confidently reverses. **Check robustness** can also patch the
whole sweep. Do that for residual stream sweeps: an estimate is least reliable where patching
replaces a whole token's representation. In the IOI example on GPT-2 small, patching the residual
stream at the changed name in the first layer restores the whole answer while its estimate is
slightly negative, so verifying the top estimates would never reach it. Head outputs track
patching closely.

*Integrated gradients* (as in EAP-IG, Hanna et al. 2024) correct much of what one gradient misses:
the gradient is averaged over `steps` runs whose input embeddings lie evenly between the receiver
prompt's and the source prompt's, so the estimate follows the metric along the way from one prompt
to the other instead of only at its start. It costs `steps` forward and backward passes instead of
one; 8 to 16 steps are usual. Its estimates are verified by patching the same way. On the IOI
prompts with GPT-2 small, 16 steps estimate the residual stream at S2 in layer 0 at 1.00, where
patching measures 1.00 and one gradient −0.13; over every residual stream site at the named
positions, the estimates correlate 0.99 with patching (one gradient: 0.56). For head outputs the
two agree about equally well with patching (0.98 and 0.99).

**Direct logit attribution** splits the logit difference of the clean or the corrupt prompts
(your choice; there is no default) into what each head, attention output and MLP output writes
directly into the residual stream at the last token. The final normalization is read with its
scale held at its value in the run, so the logit difference is exactly the sum of one term per
component plus the embeddings and biases; the results page shows that split. A component's direct
effect leaves out what it does through later components, which patching measures: **Check
robustness** offers the same components patched, so direct and total effects can be compared
side by side. Nothing is replaced, so there are no sign flips or patched probabilities.
Logogram refuses it where it isn't defined: for residual stream states (no component writes
them), at positions other than the last token, for single heads in models that normalize the
attention output after combining heads (Gemma 2, OLMo 2), and for models that soft-cap their
logits.

**Path patching** measures a component's effect through chosen receivers only: a later head's
query, key or value, or the logits read directly from the final residual stream. For each sender
(a head, attention output or MLP output of the sweep), the receiver prompt runs with the sender's
activation from the source prompt while every other attention head is held at its own value, so the
change travels only through the residual stream (and the MLPs, unless you hold them too). What the
receivers read in that run is recorded and patched into an unchanged receiver run, where the metric
is read. Senders in or after the last receiver's layer have no path, so a sweep keeps only the
layers before it. In models that share keys and values across heads (grouped-query attention),
only queries can be receivers. Tests check that from the last layer to the logits a path is the
whole effect, exactly as patching measures it. **Check robustness** offers holding or releasing
the MLPs, the opposite direction, and the senders' whole effect by patching.

**Edge attribution** estimates which edges of the model's graph carry the behavior (EAP, Syed et
al. 2023; with integrated gradients, EAP-IG, Hanna et al. 2024). The graph's nodes are the
embeddings, every attention head's output, every MLP and the logits. An edge runs from a node's
output to a later node's input through the residual stream: a head's query, key and value are an
edge each, an MLP's input one and the logits one, so GPT-2 small has 32,491 edges. An edge's
estimate is (the sender's output on the source prompt − on the receiver prompt) · the gradient of
the metric with respect to the receiver's own copy of the residual stream, summed over positions.
The gradient goes through the receiver's normalization with its scale recomputed from its input,
as patching the edge would recompute it (holding the scale, as direct logit attribution does, would
linearize another function). Tests check it against finite differences on tiny models, and that
each estimate is the slope of patching its edge. Every edge keeps its per-prompt values and
statistics: the summary lists the strongest 2,000, and edges.parquet in the run's folder holds
them all (Export → every edge as CSV). Before a run, the form says the graph's size and what the
run costs; a graph of more than 100,000 edges, more than 10 million per-prompt values or a batch
that won't fit in memory is refused with what to change. **Verify top 20 by path patching**
measures the strongest edges one at a time (each sender path-patched into its receiver alone,
every other head and MLP held) and shows each edge's estimate beside its measurement, with their
rank correlation. On the 32 IOI prompts with GPT-2 small, on two CPU threads, one gradient takes
about a minute and 8 steps of integrated gradients three and a half; path patching the 50
strongest edges takes four to six. Their rank correlation with the estimates is 0.89 (47 of 50
with the same sign), and 0.93 with integrated gradients (49 of 50). One gradient misses the
embeddings' edge into the first MLP (estimated −0.05, path patching 0.99), which integrated
gradients find (0.67).

The **faithfulness curve** keeps the k strongest edges as a circuit, for each k you choose: the
receiver prompts run with only those edges carrying the source prompt's values (`"patch":
"chosen"`), or every edge but them (`"rest"`), beside edges chosen at random (seeded draws).
Faithfulness is the share of the gap from no edge chosen (0) to every edge chosen (1) that the
chosen edges close. Edge patching is exact: each receiver reads its own input plus, for every
patched edge into it, the sender's output on the source prompt minus its output in that run, so
patching every edge reproduces the source run and patching none the receiver run (tests check
both, and a run says how far its every-edge pass is from the source run). Each k costs one patched
forward pass per batch, and one more per random draw. In the IOI example, with every edge but the
chosen carrying the corrupt prompts' values, the 1,000 strongest edges (3% of the graph) keep 0.80
of the behavior [0.74, 0.86] and the 3,000 strongest 0.96, where as many edges chosen at random
keep none of it.

Edge attribution needs heads that each write into the residual stream and receivers that can each
read their own copy of it. Logogram measures both when a run starts, and refuses models that
normalize after combining heads (Gemma 2, OLMo 2) or whose heads share keys and values
(grouped-query attention, as in Llama and Qwen), which it doesn't build edges for yet.

**Steering** adds a direction to the residual stream and measures what it does. At each steered
site (one residual stream state per layer, or chosen ones, at one token of each prompt) the
direction is the mean difference between the two prompts of each pair, from the prompts you steer
toward the other ones. It is computed on a seeded training split of the pairs and measured only on
the held-out rest, so no prompt receives its own difference. Each strength multiplies the whole
mean difference: 1 adds all of it, negative values push the other way. The effect is normalized
like patching, so 1 means the steered prompts moved as far as switching to the other prompt. A
random direction of the same length, at the same strengths, runs alongside as a control, and the
results show both: a layer × strength map with the control columns muted, and, for a selected site,
its effect at every strength with intervals. **Check robustness** offers another split of the pairs
or the other direction. A mean difference steers only when the pairs differ the same way. In IOI
prompts that mix the ABBA and BABA orders, the difference at the last token flips with the order,
so the mean cancels and steering does no more than the control; generate prompts of one order to
steer. When no site and strength does more than the control, the results say so.

**Steering along an SAE feature** adds one feature's decoder direction, W_dec[f], instead of the
mean difference (**Direction** in the form, or **Steer along this feature** in a feature's panel in
Explore → Features). It is added where the SAE's output is read into the model, and the scope names
that site alone, at one token of each prompt:

* an SAE on a residual stream site adds it to that site of its layer;
* an SAE on an attention output adds it to that layer's attention output, which is the same as
  adding it to the residual stream between attention and MLP (after the layer, in models whose
  attention and MLP read the same stream);
* an SAE on an MLP output, and a transcoder (which writes its MLP's output), add it to that MLP's
  output, the same as adding it to the residual stream after the layer;
* an SAE on attention heads' outputs adds it to every head's output `z`, before the output
  projection carries it into the attention output (the scope names that attention output). Where
  the heads add up to the attention output, this is the same as adding what the output projection
  makes of it there; in models that normalize the attention output after combining the heads
  (Gemma 2 and 3, OLMo 2) it is still exactly what the feature writes, but no fixed direction in
  the residual stream.

Tests measure each of these equivalences on tiny models, through TransformerLens's own hooks.
W_dec[f] times a is what the feature writes when its activation is a, so strength 1 is the
feature's **typical activation**: the mean of its positive activations at the steered token of the
training pairs' clean and corrupt prompts. A run records it (with how often the feature fired
there), and refuses a feature that never fires there. SAEs that standardize their inputs scale what
a feature writes by the input's spread, so there each held-out prompt's vector is also multiplied
by the standard deviation of its own activation at that token. The random control is a random
direction with the length of the steering vector at strength 1. A scope that doesn't name the SAE's
site, a feature the SAE doesn't have and an SAE made for another model are refused with the reason.

**What the steered model says**: with `continuations`, a steering run also continues the first
held-out prompts (in the split's order, up to 8, by up to 64 tokens): without steering, at each
strength and along the control. Decoding is greedy, the most likely token each time with ties to the
lowest token id, so the same spec gives the same continuations. Each forward pass adds the vector
at the steered site and token of the prompt, as when it was measured; the tokens the model
generates aren't steered themselves. They are saved as `continuations.json` beside the results,
shown under the results map (each prompt, then one line of tokens per strength, the control's lines
muted), and included in the HTML report.

**SAE features.** A sparse autoencoder (SAE) rewrites one of the model's activations as a few active
features out of thousands, each a direction in the model, plus an error it misses. In
**Explore → Features**, load a published SAE: one of those SAELens's registry lists for the loaded
model, found by release, hook, layer or width and marked with whether Logogram reads it exactly
(see *Published SAEs by name* below), or any Hugging Face repository and folder. Then **Measure the
fit** on your prompts: the fraction
of the activations' variance the SAE explains, how many features fire per token, and the logit
difference with the SAE's reconstruction spliced in. SAEs only fit activations like those they were
trained on, so a poor fit (for example with TransformerLens's weight processing switched the other
way) is shown before anything else. The view then lists the features that fire on each token of a
prompt and, for one feature, its activation along the prompt and the prompts where it fires most,
all computed from your own prompts. Feature descriptions live on sites like Neuronpedia, which
Logogram doesn't contact.

**What a feature writes** lists the tokens it promotes and suppresses directly, in the feature's
panel and in the inspector for a feature site of a run. One unit of the feature adds its decoder
direction to the activation the SAE writes (times the input's spread at that token, for SAEs that
standardize their inputs; through each head's slice of the output projection, for SAEs on
attention heads' outputs). That direction is added to the residual stream entering the final
normalization at one token, skipping every later layer, and read through the final normalization,
with its scale held at its value at that token as direct logit attribution holds it, and the
unembedding, before any soft cap on the logits. The values are each token's logit change per unit
of the feature, relative to the mean change over the vocabulary (which also makes them the same
with or without weight processing). The scale is one positive number per token, so the tokens rank
the same at every token and only the sizes change. The token is the one chosen in the prompt,
where the feature fires most, or, in the inspector, the site's position. It is refused where it
isn't defined: when the residual stream couldn't be verified to be a sum of its components, when
the final normalization and unembedding don't reproduce the model's output, and for SAEs on heads'
outputs in models that normalize the attention output after combining its heads (Gemma 2 and 3,
OLMo 2), where patching those features is still exact.

Features are sites like any other (`sae_feature`, with a feature index). **Patching a feature**
encodes the receiver's activation and changes only that feature, to its value in the source prompt
(or to zero, for zero ablation): the activation moves along the feature's decoder direction, and the
SAE's error is kept. **Attribution patching** can sweep **every SAE feature** at once and keep the
strongest, then verify them by patching; the results also show how much of the whole site's
estimated effect the features account for (the rest is the SAE's error). Choosing the strongest of
thousands of features and reporting them on the same prompts biases their effects away from zero
(the winner's curse), so the sweep can choose them on a seeded share of the prompts and report
them only on the rest (`choose_on` and `seed` in the scope).

Logogram reads the SAELens and EleutherAI (sparsify) formats, and Gemma Scope 2 (`config.json`
and `params.safetensors`, JumpReLU), from safetensors files only; the first Gemma Scope's NumPy
archives are refused, as are SAEs whose input scaling, architecture, settings or tensors it can't
reproduce exactly (such as sparsify's group-max activation, or settings from forks of sparsify).
**Transcoders** (from Gemma Scope 2 or EleutherAI) read an MLP's input and write its output: a
feature's value comes from the MLP's input, and patching it changes the MLP's output along the
feature's decoder direction. **Skip transcoders** also add their input times a matrix:
EleutherAI's add `x @ W_skip.T`, as sparsify computes it, and Gemma Scope 2's affine ones add
`x @ affine_skip_connection`: their matrices are square, but on Gemma 3 270M the affine transcoders
of layers 5, 12 and 15 explain 99%, 85% and 79% of their MLP outputs' variance read this way, and
37%, 9% and less than nothing transposed, so this is the way round the files mean (SAELens 6.54.0
adds the transpose). The input doesn't change when a
feature is patched, so neither does the skip connection, and attribution patching reports the
part of the site's estimate that flows through it beside the features' part. sparsify files
written before February 2025 don't record whether they are transcoders, so Logogram reads it from
their training run's `config.json` at the top of the repository (they also subtract `b_dec` from
the input, as sparsify did then). SAEs trained on attention heads' outputs (SAELens `hook_z`,
Gemma Scope 2's attention SAEs) read the output `z` of every head of the layer, side by side as
the output projection reads it; patching a feature changes every head's output, which reaches
the residual stream through the output projection. SAEs with a skip connection (which rebuild an
activation partly from itself), cross-layer transcoders and crosscoders are refused for now. The
fit measured on your prompts remains the last check, since a file can be trained on something its
configuration doesn't record: EleutherAI's SmolLM2 skip transcoders were trained on normalized
inputs and outputs, which only their model card says, and they fit poorly as published.

**Published SAEs by name.** People find SAEs by their SAELens release and id
(`gpt2-small-res-jb`, `blocks.8.hook_resid_pre`); Logogram ships a catalogue of the 13,857 SAEs in
the 170 releases of SAELens 6.54.0's registry, with the model each was trained on, its repository
and folder, the hook it reads, its width, the architecture its configuration states, and whether
Logogram reads it exactly or why not. Logogram reads 9,035 of them exactly: GPT-2 small's SAELens
SAEs (Joseph Bloom's, the feature-splitting ones and Apollo Research's), OpenAI's TopK SAEs on it
but for those between attention and MLP, Pythia-70m's, Gemma 2B's, the matryoshka SAEs for Gemma
2 and Gemma 3 1B, and all of Gemma Scope 2's SAEs and transcoders. Most of the others are the first
Gemma Scope's NumPy archives, PyTorch pickles (which Logogram never opens), formats it doesn't read
yet (Llama Scope's, circuit-tracer's transcoders) and cross-layer transcoders. The verdicts come
from Logogram's own readers run on each SAE's configuration (and, for SAELens's format, its weights
file's header), and they were checked on the real files, which the catalogue records: 424 SAEs
across every release Logogram reads by their weights file's header, and 16 on GPT-2 small,
Pythia-70m and Gemma 3 270M by their weights and their fit to the model. An SAE that SAELens's
registry reads with settings its own configuration
doesn't state (Mistral 7B's, which SAELens rescales) is refused before anything downloads, also
when named by its folder. The catalogue is part of the package and is never fetched;
`scripts/sae_catalogue.py` builds it from a sae-lens wheel, and SAELens's MIT notice travels with
it. `logogram saes MODEL` lists a model's SAEs, and `lg.load_sae(release=..., sae_id=...)` loads
one. However an SAE is chosen, specs record its repository, folder and revision, pinned when it
loads.

**Sites** are abstract: `resid_pre`, `resid_mid`, `resid_post`, `attn_out`, `mlp_out` and `head`
(an attention head's output `z`), each at a layer, head and position. Positions are `all`,
`last`, a token `index` or a named `label`. **Sweeps** cover every head at one position
(layer × head), one stream site at every layer and position (layer × position), or attention
and MLP outputs per layer.

**Queries, keys, values and neurons** are finer sites: `q` (one head's query), `k` and `v` (one
key/value head's key or value) and `neuron` (one MLP neuron's activation after its nonlinearity,
for a gated MLP the gate's activation times the up projection: the input to the MLP's output
projection), such as `{"kind": "q", "layer": 9, "head": 6, "position": …}` or
`{"kind": "neuron", "layer": 3, "neuron": 1201, "position": …}`. With grouped-query attention
(Llama 3, Qwen 2.5 and Gemma 2, among others) a model has fewer key and value heads than query
heads: a key or value site names a key/value head, which every query head of its group reads, so
patching it changes all of them. The scopes `head_inputs` and `neurons` sweep every head's query,
key or value, and every neuron of chosen layers; the app shows the first as a layer × head map,
and the second layer by layer, each layer's strongest neurons with their intervals beside the
distribution of all of them. When a model loads, Logogram measures that these sites make up its
attention and MLP outputs (all queries, keys and values together reproduce the attention output,
one alone changes only the heads that read it, all neurons together reproduce the MLP output)
and offers only those that hold: OLMo 2 normalizes every head's queries and keys together, so it
offers values and neurons but not single queries or keys. Patching, ablation, attribution
patching and sets of sites take them. Direct logit attribution reads a neuron at the last token
and a value at any position (through the heads that read it, their attention patterns as in the
run). It refuses queries and keys, which act only through the softmax, and neurons of models
that normalize the MLP's output (Gemma 2, OLMo 2), where no neuron writes a term of its own.

**Sets of sites** test a circuit. A `site_sets` scope lists sets of sites, each intervened on at
once, in one forward pass: one result per set. A set marked `complement` intervenes on every
component of the scope's `universe` (every head, or every attention and MLP output, at every
position) except its sites, so "keep this circuit, replace the rest" is one set; a site at one
position keeps only that position. Replacements come from the other prompt (patching), donors
(resample ablation), the mean over prompts of the same length (mean ablation) or zeros. When the
scope has a set that replaces everything (a `complement` set with no sites), every set is also
read against it, from the same resamples:

* **faithfulness**, for a set that keeps sites: the part of the behavior those sites carry alone,
  1 when keeping only them loses nothing and 0 when it does no better than replacing everything;
* **share**, for a set that removes sites: a share near 1 says removing them does what removing
  everything does (the circuit is complete);
* for a keeping set one site short of another, the faithfulness that site adds (minimality, one
  site at a time);
* for a set of two sites that are also run alone, the effect of both beyond the sum of the two
  (their interaction).

With GPT-2 small, keeping the 26 heads of Wang et al.'s IOI circuit (at every position) and
mean-ablating every other head over ABC prompts keeps all of the behavior (faithfulness 1.02);
taking 9.9 out of it costs 0.10 [0.06, 0.14], and ablating 9.9 and 9.6 together breaks less than
the two do alone, −0.14 [−0.18, −0.11], as the backup name movers the paper describes would.

**The metric** reads the answer at the end of each prompt, in one of six ways: the logit
difference (answer minus distractor, at the last token), the log-probability difference, the
answer's log-probability, its probability, the probability difference (as in greater-than), or the
KL divergence of the next-token distribution from the clean or the corrupt prompt's own, which
reads the whole distribution rather than the answer. The logit difference needs answers of one
token (or sets of them); the others also read continuations. Every run also keeps each prompt's
answer probability and its preference, log P(answer) − log P(distractor) (the logit difference
for single tokens), from which sign flips are counted. Direct logit attribution splits the logit
difference only. The **normalized effect** of each prompt is (patched − receiver) ÷ gap in the
metric, where the gap is (source − receiver) for patching and (corrupt − clean) for ablation. By
default the gap is the dataset's mean, so the mean effect is the usual normalized metric; you can
choose each prompt's own gap instead. 0 is no change; 1 is a change as large as switching to the
other prompt.

**Statistics** for every site: n, mean, standard deviation, a percentile bootstrap confidence
interval over prompts (seeded; the same resamples for every site), the number of prompts whose
preference for the answer changed sign, and the number whose effect has the opposite sign to the
mean (rounding noise aside). A sweep tests many sites, and some intervals miss zero by chance, so
every sweep also reports **simultaneous bands**, which hold for all its sites together at the
chosen confidence (max-rank percentile method), and **q-values** (Benjamini–Hochberg, from
bootstrap p-values). Prompts made from one template aren't independent: `statistics.cluster` names
a field of the prompts' `meta`, such as `template`, and the bootstrap then resamples whole
clusters, which widens intervals honestly; with fewer than 10 clusters the run warns that its
intervals are rough. Two runs on the same prompts are compared prompt by prompt, with an interval
for each site's difference from the same resamples, and steering compares each strength with its
control the same way. Per-prompt values are always kept.

## The spec

Every experiment is a `spec.json`. Nothing that can change a number is left implicit: a spec
must state every such choice, and a spec that leaves one out is refused with the field's name.

```json
{
  "logogram_spec": 3,
  "name": "Which heads restore the answer?",
  "notes": "",
  "model": {
    "id": "openai-community/gpt2",
    "source": "hub",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "sha256": null,
    "dtype": "float32",
    "device": "auto",
    "process_weights": true
  },
  "dataset": {"path": "datasets/ioi.jsonl", "sha256": "2bd046d5…", "limit": null},
  "tokenization": {"prepend_bos": true, "chat_template": null},
  "experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"},
  "scope": {"kind": "heads", "position": {"kind": "all"}},
  "metric": {"kind": "logit_diff", "normalization": "dataset_gap"},
  "statistics": {"bootstrap": 1000, "ci": 0.95, "seed": 0, "cluster": null},
  "execution": {"batch_size": 64}
}
```

Version 1 specs (from Logogram 0.1) still run. Where one leaves a choice out, it takes the value
version 1 gave it, the run's first warning lists each such value, and the spec saved with the run
states them all. `logogram validate SPEC.json` shows the list without running anything. Version 2
specs (Logogram 0.2 to 0.4) read as version 3 with what version 3 added turned off, which is what
they did: no chat template, a model from Hugging Face, and steering along the mean difference
without continuations.

| Field | Values |
|---|---|
| `model.source` | `"hub"` for Hugging Face, or `"local"` for a folder on this machine added to Logogram's local models. A local model's `id` is the name it was added under, and its folder is found by `model.sha256`, the hash of its files, so the spec never holds a path. |
| `model.revision` | A commit. `null` resolves the current main branch at run time; the saved spec pins what ran. Always `null` for a local model. |
| `model.device` | `auto`, `cpu`, `cuda` (the default GPU), `cuda:N` (one GPU of several, such as `cuda:1`; see [Choosing a GPU](#choosing-a-gpu)) or `mps`. |
| `model.sha256` | For a local model, the hash of its files (see [Local models](#local-models)); `null` for Hugging Face, where the revision pins the files. |
| `tokenization.chat_template` | `null` feeds each prompt as raw text. `{"kind": "user_turn", "system": null}` makes each prompt the person's message in the model's own chat template, so the answer is the first token of the reply; `{"kind": "assistant_prefix", "system": null, "user": "…"}` makes it the start of the model's reply to a fixed message. `system` is an optional system message. The template is the one the tokenizer ships at the model's revision (see [Chat templates](#chat-templates)). With a template, `prepend_bos` is `false`: the template writes the beginning of the text. |
| `model.process_weights` | Fold LayerNorm and center weights, as TransformerLens does by default. Logit differences don't change. Value biases are folded into the attention output's bias, so zero ablation of head outputs and heads' direct effects do change (by up to several logits in Qwen 2.5, whose value biases are large); a layer's attention output doesn't. |
| `dataset.sha256` | If set, the run refuses a dataset file that has changed. |
| `experiment` | `{"kind": "activation_patching", "direction": "clean_to_corrupt" \| "corrupt_to_clean"}`, `{"kind": "ablation", "baseline": …}` with `{"kind": "zero"}`, `{"kind": "mean", "reference": "clean" \| "corrupt"}` or `{"kind": "resample", "pool": "clean" \| "corrupt", "donors": 10, "seed": 0}`, `{"kind": "attribution_patching", "direction": …, "method": "gradient", "steps": null}` (same directions as patching; or `"method": "integrated_gradients", "steps": 8`), `{"kind": "direct_logit_attribution", "prompts": "clean" \| "corrupt"}`, `{"kind": "path_patching", "direction": …, "receivers": [{"kind": "head", "layer": 9, "head": 9, "input": "q" \| "k" \| "v"}, {"kind": "mlp", "layer": 3}, {"kind": "logits"}], "freeze_mlps": false}`, `{"kind": "edge_attribution", "direction": …, "method": "gradient", "steps": null, "curve": {"counts": [10, 100, 1000], "patch": "chosen" \| "rest", "control": {"seed": 0, "draws": 1}}}` (`curve` `null` for none) with a `graph` scope, or `{"kind": "steering", "apply_to": "clean" \| "corrupt", "direction": {"kind": "mean_difference"}, "coefficients": [-1, 1, 2], "train_fraction": 0.5, "seed": 0, "control": true, "continuations": null}` with a scope of one residual component per layer (`layer_components`) or residual `sites`, at one token. The direction can instead be one SAE feature's decoder direction, `{"kind": "sae_feature", "feature": 1234}` (with the spec's `sae`, and `sites` naming where the SAE writes), and `continuations` can be `{"prompts": 3, "tokens": 20}` to record what the steered model goes on to say |
| `scope` | `{"kind": "heads", "position": …}`, `{"kind": "layer_position", "site": "resid_pre", "positions": "each" \| "labels"}`, `{"kind": "layer_components", "components": ["attn_out", "mlp_out"], "position": …}` `{"kind": "sites", "sites": [{"kind": "head", "layer": 9, "head": 9, "position": {"kind": "label", "label": "end"}}]}` (feature sites are `{"kind": "sae_feature", "layer": 8, "feature": 1234, "position": …}`), `{"kind": "head_inputs", "input": "q" \| "k" \| "v", "position": …}`, `{"kind": "neurons", "layers": [3, 4], "position": …}`, for attribution patching, `{"kind": "features", "position": …, "top": 50, "choose_on": 0.5, "seed": 0}` (`choose_on` and `seed` both `null` to choose and report on every prompt), `{"kind": "site_sets", "universe": ["head"], "sets": [{"label": "circuit", "sites": […], "complement": true}]}`, for edge attribution `{"kind": "graph", "layers": null, "logits": true}` (`layers` the heads and MLPs that receive edges, `null` for every layer), or, for path patching edges one at a time, `{"kind": "edges", "edges": [{"sender": {"kind": "head", "layer": 5, "head": 5}, "receiver": {"kind": "head", "layer": 9, "head": 6, "input": "v"}}]}` (senders `{"kind": "embed"}`, heads or `{"kind": "mlp", "layer": 0}`; `freeze_mlps` true, and the experiment's receivers those of the edges) |
| `sae` | The SAE that feature sites belong to: `{"repo": "…", "path": "blocks.8.hook_resid_pre", "revision": "…"}`. The revision is pinned when a run starts. |
| `metric` | `{"kind": "logit_diff" \| "logprob_diff" \| "logprob" \| "prob" \| "prob_diff", "normalization": …}` or `{"kind": "kl", "target": "clean" \| "corrupt", "normalization": …}` |
| `metric.normalization` | `dataset_gap` or `prompt_gap` |
| `statistics.cluster` | `null` to resample prompts, or a field of the prompts' `meta` (such as `"template"`) to resample clusters of prompts |
| `execution.batch_size` | Recorded because batch shape can change floating-point results in the last digits. |

## Projects

```text
my-project/
  project.json
  research.json                      saved selections and research notes (created on first save)
  datasets/*.jsonl
  datasets/snapshots/<sha256>.jsonl   immutable input bytes, shared by runs with identical data
  experiments/<id>/spec.json          the experiment, with revision and dataset hash pinned
  experiments/<id>/results.parquet    one row per site and prompt
  experiments/<id>/summary.json       per-site statistics and the layout of the results
  experiments/<id>/predictions.json   optional per-layer prediction diagnostic
  experiments/<id>/manifest.json      versions, device, dtype, model revision, dataset hash, timing
  experiments/<id>/traceback.txt      a failed run's traceback, without this machine's paths
  experiments/<id>/owner.json         while a run runs: its process id and start time
  .gitignore
```

`results.parquet` has everything needed to recompute the statistics: site, layer, head,
position, prompt, the patched, receiver and reference values of the metric, the answer
probability, and the preference log P(answer) − log P(distractor) (the logit difference for
single tokens). Paths inside a project are relative, so folders can be moved,
committed and shared.

Each new run snapshots its input dataset before loading the model and pins that snapshot in the
executed spec. Editing the original dataset leaves existing runs reproducible. A modified
snapshot is refused by its hash; it is never silently overwritten. Keep snapshots alongside
experiments when sharing a project. Older runs still use the dataset path in their original spec.

Browser tabs follow the server's open project. Switching projects clears the other tabs' forms,
prompts, cached analyses and results; stale requests are rejected. A project cannot be switched
while a job runs. The status line shows lost connections. A run started elsewhere, by
`logogram run` or from Python, lists as running in another process until it ends; the app
notices when it does. A run whose process stopped before it finished (closed, killed or
crashed) is marked failed, with its saved spec available to rerun. `owner.json` tells the two
apart: the process id and its start time, and a random token, nothing about you or the machine.

**Reproducibility.** Runs seed Python, NumPy and PyTorch from `statistics.seed`, require deterministic
PyTorch algorithms (unsupported operations fail rather than merely warn), and disable
TF32, and record Logogram, Python, PyTorch, TransformerLens and transformers versions, the device
and GPU model, the dtype, the model id and revision, whether weights were processed, the dataset
hash and the wall time. The same spec on the same machine gives bit-identical results. Results
on different devices or library versions can differ; compare their per-prompt values before
treating them as equivalent.

## Stability

**What stays readable.** Each release reads the run folders earlier releases wrote: their specs,
manifests, summaries and per-prompt results list, open, compare and export as before. What a
later release added and an older run didn't measure (corrections for the number of sites, say)
is left out; runs from 0.1, which measured the logit difference only, read it as their metric.
Version 1 and version 2 specs both run: a version 1 spec takes version 1's value for each choice
it left out, `logogram validate` lists them, and its runs say so. The tests keep run folders
written by 0.1.0 and 0.2.0 to hold every release to this.

**What is bit-identical.** The same spec, on the same machine, with the same versions of
Logogram, PyTorch, TransformerLens, Transformers and NumPy, gives the same per-prompt values to
the last bit. `logogram run` on a finished run's spec checks this.

**What may differ.** On another device (the CPU, CUDA or MPS, or another GPU) or with other
library versions, results may differ in the last digits, because the same sums run in another
order. Each manifest records the versions and the device; the Results view says when they
differ from this machine's, and Compare says when two runs differ in them. `logogram env`
prints this machine's, and `logogram env --requirements` pins its versions.

## Privacy

* Everything runs locally. There is no account, no telemetry and no analytics. The app loads no
  fonts or scripts from the internet; its fonts are bundled.
* Logogram goes online in two cases only. When you load a model, it asks Hugging Face to resolve
  its revision, read its size for the memory estimate and download it once (your own Hugging
  Face login is used for gated models; Hugging Face telemetry is turned off); loading an SAE
  downloads it the same way. And if you allow
  it, it asks pypi.org for the newest Logogram version number once a day (see Updating). Nothing
  about you, your machine or your work is sent.
* The server listens on 127.0.0.1 only. Every request needs the random session token from the
  address printed in the terminal (exchanged for a same-site cookie when the page opens). The
  browser is opened with a single-use code instead, because other users of a shared machine can
  read a program's command line. The server checks the Host and Origin headers (loopback names
  on its own port, and on the forwarded ports you name with `--forwarded-port`), sends no CORS
  headers, and refuses cross-site requests.
* A problem report (**Help → Report a problem**, or `logogram doctor --report`) is built on your
  machine and sent nowhere. It never names users, hosts or paths (they are taken out of the error
  it quotes), and names hardware models only if you tick **Include hardware details**.
  To report a security problem, see [SECURITY.md](SECURITY.md).
* Projects are often shared, so Logogram treats their contents with care: it writes only into
  folders that are really inside the project (never through a symlink that leads out of it), and
  a damaged, oversized or unexpected file shows up as an error on that file rather than breaking
  the project. A failed run's error and traceback are saved without this machine's paths, your
  account name or the machine's name. `logogram env` names hardware only with `--hardware`.
* Weights are loaded only from safetensors files, which cannot run code. Logogram never unpickles
  files.
* Settings and the list of recent projects are kept in your user configuration folder, not in
  projects.

## Updating

Logogram tells you when a new version is out, in the way you choose:

* **Never online by itself.** Each version knows its release date. A few months after it, the
  app and the terminal suggest checking for a newer one.
* **Check now.** Press it in the update panel (or the System check), or run
  `logogram check-updates`. Logogram asks pypi.org for the newest version number.
* **Daily, if you allow it.** The System check asks once whether Logogram may look once a day.
  You can change your mind in the update panel at any time.

When a newer version is known, a small **Logogram 0.x.y** button appears in the top bar with what's
new and the exact update command for how you installed it, and starting `logogram` prints it in
the terminal. Updating never touches your projects, results or settings:

| Installed with | Update with |
|---|---|
| `uv tool install logogram` | `uv tool upgrade logogram` |
| `uv tool install .` from a copy of this repository | `git pull`, then `uv tool install --reinstall .` |
| `uv sync` (development) | `git pull`, then `uv sync` |

A check is one HTTPS request for Logogram's public package information on pypi.org. It carries
Logogram's version number and nothing else.

## Models

TransformerLens 4 loads well over a hundred architectures, and Logogram works with the
decoder-only language models among them: GPT-2, Pythia, Llama, Mistral, SmolLM, Qwen, Gemma,
OLMo, Phi and more. The model dialog lists starting points, and any other Hugging Face id can be
typed in. Before downloading, Logogram checks that TransformerLens supports the architecture and
estimates the memory needed against what is free, the moment of loading included, when processing
the weights needs three to four times their float32 size. [Models](docs/models.md) says which
models have been checked, how much memory they need, and what weight processing changes.

Use float32 whenever the model fits. float16 and bfloat16 halve the memory and run two to three
times faster on a GPU, but they round: in the IOI example, effects smaller than about 0.01 changed
order, and on Pythia-70m bfloat16 reordered the heads. Check results that matter against float32:
**Check robustness** reruns a 16-bit run in float32 and compares the two. On Apple Silicon,
**Automatic** runs models on the CPU, since TransformerLens reports that Apple's MPS can give
silently wrong results; MPS is used only when you choose it under **Device**.

Rather than trusting a list, Logogram checks every model when it loads, on a short fixed input:
TransformerLens's version of the model must predict what the original model predicts (processed
weights that change the predictions are refused, and so is a dtype in which the model overflows),
how each layer adds attention and
the MLP to the residual stream is measured (one after the other, or both from the same input, as
in Pythia), and layer predictions are offered only where the final normalization and unembedding
reproduce the model's output. [The model checks](docs/concepts/logit-lens-and-model-checks.md)
say what each check measures and what relies on it.

**Models made for chat** (instruction-tuned ones, such as Qwen 2.5 0.5B Instruct) are trained on
conversations wrapped in their chat template, but Logogram feeds every prompt as plain text unless
the spec chooses the template, so such a model can behave differently here than in a chat. When a
model loads, Logogram checks whether it is made for chat (its name says so, or it stops generating
at the token that ends each turn of its template), and the model dialog, the summary beside **Run**
and the baseline say so until its chat template is chosen.

### Chat templates

A model that ships a chat template can read the prompts as a chat: choose **Chat template** in the
experiment form (under Data and model), or set the spec's `tokenization.chat_template`. Every
prompt, clean and corrupt alike, is wrapped in the model's own template, the one its tokenizer
ships at the model's revision, either as **the person's message** (`user_turn`), so the answer is
the first token of the reply, or as **the start of the reply** (`assistant_prefix`) to a fixed
message. [Run a chat model on its own template](docs/guides/chat-models.md) says how the wrapped
text is tokenized, how answers and named positions are read, and which templates are refused and
why; each run's manifest records the template's SHA-256.

### Local models

A model on this machine that isn't on Hugging Face, such as a fine-tune, can be added from its
folder: config.json, the tokenizer's files, and the weights as safetensors (a folder whose weights
are only pickles is refused, since opening a pickle can run code). Add it under a name with **Add a
model folder** in the model dialog, `logogram models add PATH --name NAME`, or
`lg.add_local_model(path, name)`. Its files are hashed once; a spec then names the model by that
name and hash (`"source": "local"`, `"sha256"`), never by its folder, and a spec from someone else
finds the same files on this machine by their hash. [Study your own fine-tune](docs/guides/local-models.md)
defines the hash exactly and says what happens when a folder's files change.

### Choosing a GPU

On a machine with several GPUs, the model dialog offers each one, by name and free memory, and the
memory estimate and its verdict use that GPU's free memory. A spec names one as `cuda:N`;
`cuda` is the default GPU (normally GPU 0), so a model loaded on it serves a spec that asks for
either. A spec naming a GPU this machine doesn't have is refused with the GPUs it has.
`logogram doctor` lists them, and each run's manifest records the GPU it ran on: its index and its
name.

## Troubleshooting

The messages people see most often, as Logogram shows them, and what to do. If yours isn't here,
**Help → Report a problem** (or `logogram doctor --report`) builds a report to paste into a
[new issue](https://github.com/Jeevash23/logogram/issues/new/choose); it sends nothing.

**The page**

* *Logogram needs its session token.* (status 401): the page was opened without the session
  token, or with an old one. Open the address printed in the terminal where Logogram runs, the one
  ending in `?token=…`. Each start makes a new token, and the address the browser was first
  opened with works only once.
* *This page's session has ended. Open the link printed in the terminal where Logogram is
  running.* Logogram was restarted, so the tab's token is no longer valid: open the new address.
* *Can't reach the Logogram server. Check that `logogram` is still running in your terminal.*
  The server stopped, or the SSH tunnel to it closed. Start it, or the tunnel, again.
* *Logogram didn't answer this request: it was sent to 127.0.0.1:9000, and this Logogram listens
  on port 8765.* (status 421): the browser reached Logogram through a port it wasn't told about,
  usually an SSH tunnel's local end. Start Logogram with `--forwarded-port 9000` (see
  [Running on a remote GPU](#running-on-a-remote-gpu)). An address with another name, such as
  the machine's own, is refused on any port: open it at 127.0.0.1 instead. Logogram answers only
  to 127.0.0.1, localhost and [::1], so that other websites can't reach it.

**A model that won't load**

* *… is gated. Accept its license on huggingface.co, then log in on this machine with
  `hf auth login` and try again.* Do both on the machine where Logogram runs.
* *No model called … was found on Hugging Face.* Check the id, which looks like `owner/name`.
  Private models need `hf auth login` first.
* *Couldn't reach Hugging Face, and … isn't in your local cache.* A model downloads once, the first
  time it loads: connect to the internet for that.
* *This model only publishes pickle weights (.bin).* Logogram loads safetensors weights only,
  because pickle files can run code when opened. Choose a model that publishes
  `model.safetensors`.
* *TransformerLens … can't load … models, so Logogram can't either.* The model dialog says so
  before downloading anything: choose a model of a supported family, such as GPT-2, Llama, Qwen,
  Gemma, Pythia or OLMo.
* *Processing the weights of … changed its predictions …* Load it with weight processing off:
  untick **Process weights** in the model dialog, or set `"process_weights": false` in the spec.
* *In float16, …'s own forward pass gives values that aren't finite numbers (it overflows), so its
  predictions can't be checked. Load it in bfloat16 or float32, which have a wider range than
  float16.* Pythia is one such model.
* *The spec asks for CUDA, but PyTorch can't see a CUDA GPU here.* `logogram doctor` names the
  cause, usually a CPU-only PyTorch, and the command that fixes it. Or set `model.device` to
  `auto` or `cpu` in the spec.

**A model or a run that doesn't fit.** The model dialog estimates, before downloading, the memory
a model needs against what is free, and the experiment form's summary says whether a batch fits.

* *The model doesn't fit in GPU memory while its weights are processed …* Processing works on
  float32 copies of the weights, several times their size, for a moment: load it with weight
  processing off, or choose a 16-bit dtype, a smaller model or another device.
* *The model doesn't fit in GPU memory. Choose a smaller model, a 16-bit dtype or another device.*
* *The run ran out of GPU memory at a batch of 64 rows. Lower execution.batch_size in the spec …*
  Attribution patching, direct logit attribution and path patching hold every layer's activations
  at once, so they need smaller batches than patching does.

**Prompts and answers.** The app shows both prompts of each pair token by token, and says which
pairs can't be used, and why:

* *the answer ' Mary Mary' is several tokens (' Mary' + ' Mary'). The logit difference reads one
  token …* Use a single-token answer, usually with a leading space (`" Mary"`, not `"Mary"`), or a
  log-probability or probability metric, which read answers of several tokens. Prompts generated
  in the app with the model loaded use only words that are single tokens for it.
* *clean has 15 tokens and corrupt has 16, so positions can't be aligned.* Change what differs
  between the two prompts to words with the same number of tokens, or give the pair
  `corrupt_positions`: its labels in the corrupt prompt, which lets it be patched label by label.
* *position 'S2' (characters 38–42) doesn't cover any token.* The span's character offsets don't
  reach a token of the clean prompt: correct them.
* *This model's tokenizer has no beginning-of-sequence token. Set tokenization.prepend_bos to
  false in the spec.* The app does this by itself when such a model loads; a spec written by hand
  must say so.

**Apple Silicon (MPS)**

* **Automatic** runs on the CPU: TransformerLens reports that Apple's MPS can give silently wrong
  results, and Logogram hasn't been checked on it, so MPS is used only when you choose it under
  **Device**. Check results that matter by rerunning on the CPU and comparing the two runs.
* *The spec asks for Apple's MPS, which isn't available here. Set model.device to auto or cpu.*
  The spec was made on a Mac: change its `model.device`.
* `logogram doctor` says *Python is running under Rosetta*: an Intel Python can't use the Apple
  GPU. Install a native arm64 Python and reinstall Logogram with it, as it says.
* `logogram doctor` says *Apple GPU (MPS) unavailable*: MPS needs macOS 12.3 or later.

**Windows**

* *dataset.path must be a path inside the project, written with forward slashes and without '..',
  such as datasets/ioi.jsonl*: specs travel between machines, so their paths are relative to the
  project and use forward slashes, on Windows too.
* A path with spaces needs quotes, in PowerShell and the Command Prompt alike:
  `logogram open "D:\Research projects\ioi"`. A project's path may start with `~` for your home
  folder: `logogram open ~/Documents/Logogram/ioi-example`.
* `logogram doctor` says *Your NVIDIA GPU isn't being used*: this PyTorch is CPU-only. Reinstall
  with the command it gives, which picks the CUDA build that matches your driver.
* A warning that the Hugging Face cache uses symlinks, but your machine doesn't support them, is
  harmless: downloads take more disk space. Windows' Developer Mode lets the cache use symlinks.

**A run that failed.** The results page says **This run failed**, with the reason, and
**Edit and rerun** puts its spec back in the form; **Show details** shows the traceback. The same
reason is saved in the run's folder, in `experiments/<id>/manifest.json` under `"error"`, and the
traceback in `traceback.txt`, both without this machine's paths or names; `logogram show RUN`
prints them, and `logogram list` shows the run as failed. *The server stopped before this run
finished. Rerun its saved spec.* means Logogram stopped while the run was going (an SSH session
that closed without `tmux`, for example), and *The process running this run stopped before it
finished.* means the same of a `logogram run` or a notebook: the spec is saved, so run it again.
Warnings from Logogram and its libraries appear in the terminal where it runs; `--verbose` shows
what the server does, and `--log-file PATH` keeps it.

## Development

[CONTRIBUTING.md](CONTRIBUTING.md) lists the checks to run before a pull request, the rules in
[AGENTS.md](AGENTS.md) in short, and how to add a method.

```bash
uv sync                       # Python environment with dev tools
uv run pytest                 # tests use a tiny random model and never use the network
uv run ruff check src tests scripts
python3 scripts/check_privacy.py
```

The web app lives in `web/` (React, TypeScript, Vite). For live reloading, run the API and the
dev server side by side:

```bash
uv run logogram serve --dev
```

```bash
cd web && npm install && npm run dev
```

Open the development address printed by `logogram serve --dev`. Before committing changes to the
web app, build it into the package with `npm run build`; CI checks that the committed build
matches the sources.

Web regression checks:

```bash
cd web
npm test                      # context, cache and keyboard navigation checks
npx playwright install chromium
npm run test:browser           # temporary local project and tiny model; no Hub access
LOGOGRAM_COLOR_SCHEME=dark npm run test:browser   # the same, in the dark theme
```

The browser checks include an axe scan of each workspace's main views, which fails on serious or
critical accessibility violations.

The browser fixture never opens the user's projects or cached models. CI runs these workflows
on Linux, and the Python suite plus an installed-wheel smoke check on Linux, macOS and Windows.
The wheel check covers the CLI, bundled example and fonts, authenticated API and served web
assets. Source distributions include the web sources and lockfiles so the bundle can be rebuilt.
CI also type-checks `src/` with pyright, lints with ruff's security, async and blind-except
rules as well, checks that `uv.lock` matches `pyproject.toml`, keeps test coverage at 85% or more, runs
the tests with every dependency at the lowest version `pyproject.toml` allows, and audits the
locked Python and web dependencies for known vulnerabilities; its actions are pinned to commits,
and Dependabot and CodeQL watch them. Once a week, the Golden workflow runs every method on GPT-2
small's real weights and checks the bundled examples against what was published about GPT-2
small: the indirect-object circuit, greater-than, factual recall and gendered pronouns
(`scripts/validate_real_weights.py --golden`; `--golden-only` runs just those checks in a few
minutes).

Releasing: raise `__version__` and set `__released__` to the release date in
`src/logogram/__init__.py` (the date drives the "this version is getting old" hint), commit, and
push a tag with the same version, such as `v0.1.0`. The Release workflow runs every CI check on
that commit, builds the wheel and source distribution, and uploads them to PyPI through Trusted
Publishing, so no token is stored anywhere. PyPI never accepts the same version twice.

Before tagging, manually check a first GPT-2 download, cancel and retry it, then run
`scripts/validate_real_weights.py` on each supported compute backend. It runs every method on a
real model in a temporary project and checks exact identities, agreement between methods and
bit-identical reruns; `--compare-device` reruns the head sweep on a second device, and `--sae`
adds SAE features. CPU and CUDA pass it. Apple's MPS still needs it run on a Mac:

```bash
uv run python scripts/validate_real_weights.py --device mps --compare-device cpu
```

Pythia and Qwen 2.5 have also been run end to end with their real weights; other families are
checked when they load and in the tiny-model tests.

The catalogue of published SAEs (`src/logogram/backends/sae_catalogue.json`) is rebuilt for a new
sae-lens release with `scripts/sae_catalogue.py build --sae-lens WHEEL_OR_SITE_PACKAGES --cache
DIR`, which reads each SAE's configuration from Hugging Face into `DIR`, and checked with
`scripts/sae_catalogue.py verify --cache DIR --text FILE`, which reads a sample's weights files and
runs the small models' SAEs on their models, recording each check in the catalogue.

Model access goes through `logogram.backends.base.ModelBackend`. TransformerLens is the only
backend today; remote execution and other libraries can be added behind the same interface.
Not built yet, with room left for them: chat templates for models made for chat, attribution
graphs over transcoders, cross-layer transcoders and crosscoders, the first Gemma Scope's NumPy
SAE files, remote compute, a native desktop wrapper, plugins and assistants.

## License

MIT. See [LICENSE](LICENSE). To cite Logogram, see [CITATION.cff](CITATION.cff), which GitHub's
**Cite this repository** reads.
