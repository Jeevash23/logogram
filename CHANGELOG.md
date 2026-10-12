# Changelog

What changed in each version of Logogram.

## 0.4.0 (2026-10-12)

Runs you can keep track of and trust across machines, a way to work on a remote GPU, help in the
app, examples with the results they should give, and SAE features read more exactly.

### Added

- **Three more examples, each read against its paper**, on GPT-2 small: greater-than (Hanna et
  al. 2023; MLPs 8 to 11 carry it, as published), factual recall traced with another country in
  place of noise (Meng et al. 2022; the early site at the subject and the late site at the last
  token, though one MLP at a time doesn't reproduce the MLP claim, as its notes say), and the
  gendered pronoun (Mathwin et al. 2023; heads 10.9 and 9.7 write it). The welcome screens list
  every example with its question, method, the result to expect and the paper; each opens as a
  copy of its own, never overwritten. A weekly check runs all of them on GPT-2 small's real
  weights.
- **A run's notes beside its result**, folded to three lines, and in its methods paragraph.
- **Managing runs.** Each run can have a title, shown instead of its spec's name, a star, and a
  place in the archive. These are kept beside the spec, in `annotations.json`, so the spec never
  changes. The history can be searched by title, name, id, method, model and dataset, and filtered
  to starred or archived runs. Rename (F2), star, archive and delete from each row's menu. Deleting
  asks first and removes the run's folder for good (archive keeps a run out of sight instead); runs
  derived from it stay. The same is in the command line (`logogram rename`, `star`, `archive`,
  `delete`; `logogram list --archived`) and the Python API (`run.rename()`, `run.star()`,
  `run.archive()`, `lg.delete_run()`). The command palette searches every run.
- **Runs in other processes.** A run started with `logogram run` or from a notebook is no longer
  marked failed when the app opens its project: an owner record in its folder says which process
  runs it, and the history shows it running and fills in its results when it ends. A run whose
  process stopped is marked failed with that reason.
- **A failed run's details.** A failed run keeps its traceback in `traceback.txt`, without this
  machine's paths or names; the app shows it under **Show details**, with a copy button, and
  `logogram show` and `Run.traceback` give it too. Ctrl+C during `logogram run`, or a notebook's
  interrupt, records the run as cancelled. `--verbose` logs what the server does, and
  `--log-file PATH` keeps the log.
- **Where a run ran.** `logogram env` prints the versions and devices results depend on
  (`--requirements` pins them). The Results view says when a run ran with other versions of
  Logogram, PyTorch, TransformerLens, Transformers or NumPy, or on another kind of device, and
  Compare lists how two runs' environments differ.
- **A remote GPU over SSH.** `--forwarded-port N` (or `LOGOGRAM_FORWARDED_PORTS`) lets the browser
  reach Logogram through a tunnel's local port; the server still listens on 127.0.0.1 only and still
  answers only to loopback names. A request to an unknown port says exactly which option to add.
  The README has a guide.
- **A Help menu**: keyboard shortcuts, guides, **Report a problem** (a report to paste into an
  issue, with versions, the loaded model and the last error, scrubbed of paths, names and tokens;
  also `logogram doctor --report`), the system check and About.
- **What a feature writes**: the tokens an SAE or transcoder feature promotes and suppresses
  directly, through the final normalization (its scale held, as direct logit attribution holds it)
  and the unembedding, in the Features view and the Inspector (`POST /api/sae/writes`,
  `lg.feature_writes`).
- **Transcoders with a skip connection** (EleutherAI's and Gemma Scope 2's affine ones) are read
  exactly; attribution patching reports what flows through the skip connection. **SAEs on attention
  heads' outputs** (`hook_z`) are read, from SAELens and Gemma Scope 2.
- **Models made for chat** are recognized when they load (by their name, or by stopping at their
  chat template's end-of-turn token). The model dialog, the summary beside **Run** and the baseline
  say that their prompts run as plain text, and each run's manifest records it.
- SECURITY.md, CONTRIBUTING.md, CITATION.cff and issue templates.

### Changed

- History opens failed and cancelled runs on Results, where the reason and **Edit and rerun** are.
- A run whose spec can't be read is listed as damaged, with what is wrong, instead of vanishing.
- `logogram list` hides archived runs unless `--archived` is given.
- Typer 0.16 or later is required: earlier versions can't show `--help` with Click 8.2 or later.

### Fixed

- The welcome screens had no main landmark, and the tagline's green "Explore" had too little
  contrast on white; its words are now ink, underlined in their workspace's color.
- Opening a project marked runs going on in another process as failed.
- A workspace chosen while a project was opening was undone a moment later.
- Patching a feature of an SAE that standardizes its inputs (OpenAI's GPT-2 SAEs) failed.
- Features of SAEs on attention heads' outputs weren't laid out by head in the Features view.
- EleutherAI transcoders in the current format had `b_dec` subtracted from their input; files that
  use settings sparsify's releases don't define were read as TopK instead of being refused.

## 0.3.0 (2026-10-11)

The interface, made easier to work in: the result where you look first, links and Back, a run
queue, a summary beside Run, a prompts table you can search and edit, the thresholds behind each
verdict, and nothing that is only explained on hover.

### Added

- **A run queue.** While a job runs, **Run** reads **Queue**: the experiment waits its turn, listed
  under **Queued** in History, and starts when the jobs ahead of it end, whether they finish, fail
  or are cancelled. A queued run can be removed until it starts. The queue belongs to the open
  project and lives in the server's memory (`POST`, `GET` and `DELETE /api/queue`).
- **Links, Back and Forward.** The address keeps your place: the view, and the run it shows or the
  two it compares. Back and Forward move between views, a reload comes back to the same place, and
  **All projects** leads back to the open project as you left it. The address never carries the
  session token.
- **A summary beside the experiment form**, in view while the form scrolls: what will run and how
  much work it is, whether a batch of that size fits in the memory free with the loaded model
  (`POST /api/models/run-memory`), whether the baseline was checked for these prompts and what it
  found, and, when the form can't run yet, what is missing, right above **Run**.
- **The prompts table** can be searched (both prompts, answers and metadata), shows 100 rows a
  page, and is one tab stop whose arrow keys, Page Up and Down, Home and End choose a prompt.
  Prompts can be edited or removed, and the edits are saved as a new dataset
  (`POST /api/datasets/derive`), so the dataset a run recorded never changes.
- **Compare: a site in both runs.** Choosing a cell, or a row of the table, circles it on both
  maps and shows each run's effect and interval, B − A (paired prompt by prompt when the runs
  share prompts), its flags, and a way into each run at that site.
- **Single-key shortcuts can be turned off**, in the shortcuts list or the command palette, for
  speech input or keys pressed by accident. Shortcuts with Ctrl or ⌘, the arrow keys and Esc keep
  working.

### Changed

- **The result first:** the Results heatmap starts on the first screen, right under the run's
  figures; the corrections for testing many sites follow it, with their explanation on request.
- **The token strip** remembers being hidden or shown, starts hidden on a short window, and stays
  off views that show tokens themselves. Its tokens are one tab stop with arrow keys between them.
- **One name for every view**, shared by the tabs, the command palette and the keys, and the
  number keys go to the open workspace's tabs (**1** to **5**) instead of seven views across
  workspaces.
- **The thresholds behind verdicts are stated where the verdict is:** the behavior is present
  above a 0.5 gap, a robustness check holds above a 0.8 rank correlation, and an SAE fits poorly
  under 60% of the variance explained.
- **Nothing is explained only on hover.** Explanations show on keyboard focus too; a disabled
  button with a reason stays focusable so the reason can be read; choices that can't be made say
  why under them; and what matters (the memory estimate's rows, the flag mark on result maps and
  the forest plot) is written into the page.
- **Color keeps its job:** focus rings, selection and progress outside a workspace are ink;
  intervals are drawn in a neutral band instead of the evidence pink; callouts and boxed regions
  are set off by hairlines instead of grey.
- The app follows the computer's light or dark setting until a theme is chosen, so a dark-mode
  user no longer sees it switch to white after it loads.
- View titles say what the view shows instead of repeating the tab's name.
- The exported methods paragraph is written in sentences, without raw JSON or true/false values,
  and exported figures have a white background.
- The baseline's dot plot is one tab stop: the arrow keys go from prompt to prompt and read the
  chosen prompt's values. Every progress bar has a name, and smooth scrolling respects reduced
  motion.

## 0.2.0 (2026-10-11)

Specs that state every choice, more metrics, circuits, five more tasks, statistics that hold across
many sites, a Python API and more commands, Gemma Scope 2 and transcoders, and integrated
gradients, all checked by patching.

### Changed

- **Specs are version 2.** Every choice that can change a number must be stated: the model's
  revision, dtype, device and weight processing, the dataset's hash and limit, tokenization, the
  metric, the statistics (bootstrap, interval, seed, clustering) and the batch size, and every
  field of an ablation's baseline and a sweep's positions. A version 1 spec still runs: where it
  leaves a choice out, it takes version 1's value, the run's first warning lists each one, and
  `logogram validate` shows the list without running.
- Spec paths must lie inside the project, and an SAE's revision must be a commit, tag or branch.
- Seeds must be whole numbers from 0 to 2³² − 1; a negative or huge seed used to crash a run.
- Sign flips count prompts whose preference for the answer, log P(answer) − log P(distractor),
  changes sign (for single tokens that is the logit difference, as before).
- Attribution patching no longer reports patched logit differences, probabilities and sign flips
  computed from first-order estimates; like direct logit attribution, it leaves them empty.
- The count of prompts with an effect of the opposite sign ignores rounding noise, so a site with
  no effect no longer reads as half opposite.
- Steering compares each strength with its random control prompt by prompt, from the same
  resamples, and doing significantly *less* than the control no longer counts as beating it.

### Added

- **Metrics:** the log-probability difference, the answer's log-probability, its probability,
  the probability difference and the KL divergence from the clean or corrupt prompt's next-token
  distribution, beside the logit difference. Every run also keeps each prompt's answer
  probability and preference.
- **Answer sets and continuations:** an answer or distractor can be a list of single tokens, read
  as one answer (every later year in greater-than), or text of several tokens, read token by
  token. Patching at every position covers the continuation too.
- **Sets of sites:** intervene on several sites at once, or on everything except a set ("keep
  the circuit, replace the rest"), and read each set's faithfulness (how much of the behavior the
  kept sites carry alone), the share of removing a set (how complete a circuit is), the
  faithfulness each site adds (minimality) and the interaction of two sites, all with intervals.
- **Tasks:** seeded datasets for greater-than, docstring, gendered pronouns, subject–verb
  agreement and capital cities, beside IOI, each with templates of one token length, named
  positions and the metric that reads them, from the app, `logogram generate` or Python.
- **Integrated gradients** for attribution patching (as in EAP-IG): the gradient is averaged over
  steps between the two prompts' embeddings, which corrects much of what one gradient misses.
  Its estimates are verified by patching like any others.
- **Statistics across many sites:** simultaneous bands that hold for every site of a sweep at
  once, Benjamini–Hochberg q-values, a bootstrap that resamples clusters of prompts (such as
  templates, with a warning when there are too few), and paired differences between two runs on
  the same prompts.
- **The every-feature sweep can choose its strongest features on a seeded share of the prompts
  and report them on the rest**, so their effects aren't biased by the choice.
- **Gemma Scope 2** SAEs and transcoders (`config.json` and `params.safetensors`), EleutherAI
  transcoders, and SAEs on attention heads' outputs. Transcoder features are read from an MLP's
  input and patched in its output.
- **A Python API for notebooks:** `logogram.run`, `load_run`, `list_runs`, `compare_runs`,
  `verify_run`, `check_robustness`, `generate` and `write_dataset`, with tables as pandas or
  Arrow and runs that show their heatmap in a notebook. Importing `logogram` loads no PyTorch.
- **Commands:** `list`, `show`, `compare`, `diff`, `verify`, `robustness`, `export`, `validate`,
  `tasks` and `generate`.
- **In the app,** every new choice is in the experiment form, which says before a run what the
  prompts and the method allow; staged sites can be tested as a circuit, two sites checked for
  their interaction, and a run's strongest sites tested as nested circuits, with a chart of
  faithfulness against circuit size; results show simultaneous bands, q-values, clusters and
  paired comparisons; and **Prompts → Generate a task** writes any task's dataset and offers the
  metric that reads it.
- **The web app** contains errors to the part of the page that failed, with a way to reload it
  and copy the details; undoes and redoes changes to the experiment form (Ctrl/⌘ Z), says when an
  action replaces the form and offers to undo it, and keeps an unsaved form for each project; asks
  before cancelling a run; and lists every shortcut on **?**. Radio groups take one tab stop and
  arrow keys, form fields have visible edges in both themes, and the status line announces a
  run's milestones rather than every tick.

### Fixed

- **Path patching with head receivers and the logits receiver measured only the logits path.**
  The logits receiver now adds its change to what the head receivers changed, so the two together
  measure both paths (checked against each alone).
- A metric or gap that isn't a finite number, as a 16-bit overflow gives, stops the run with the
  reason instead of finishing with empty effects; patched values that aren't finite are reported.
- A failed run's saved error no longer contains this machine's paths (the home folder, the
  Hugging Face cache or the project folder).
- Running out of memory during a run says to lower `execution.batch_size`, and frees the memory.
- Memory: gradient methods no longer keep what weight gradients would need; attribution patching
  computes each layer's terms once per batch instead of once per site; on a GPU, a layer's
  captured activations move to CPU memory when they would take more than a quarter of what is
  free; an SAE's fit is measured a chunk of tokens at a time; and the memory estimate counts what
  gradient methods and path patching hold.
- A deeply nested or very large file in a shared project no longer stops the project or its
  history from opening; it shows up as an error on that run or dataset.
- The server checks which project a tab shows on the SAE routes too, a tab whose session has
  expired stops reconnecting, unloading a model waits for work in progress, and cancelling a run
  no longer waits behind queued analyses.
- The token strip shows a run's own prompts again when you return to its results, instead of
  the experiment form's.
- Heatmaps and the model map draw hover and selection on a layer of their own, and find cells
  through a map, so large sweeps no longer redraw everything on every mouse move.
- Every view has one top-level heading, with the headings below it in order, and the inspector's
  resize handle belongs to the inspector, so screen reader users can move through each page by
  its headings and landmarks.
- Pydantic before 2.10 no longer warns about a field named `model_id` when Logogram starts.
- Text on data colors (heatmap values, the model map's labels and a feature's token highlights)
  is black or white, whichever contrasts more; some mid-tone cells in the black theme had light
  text below the 4.5:1 that small text needs.

### Development

- **A weekly check against published results:** the Golden workflow runs every method on GPT-2
  small's real weights and checks the bundled example against its indirect-object circuit (Wang
  et al. 2022): the name movers, S-inhibition and induction heads restore the answer, the negative
  name movers work against it, and the name movers write the answer most directly.
- CI type-checks the package with pyright, lints with ruff's security, async and blind-except
  rules too, checks the lockfile, keeps coverage at 85% or more, tests at the lowest declared
  dependency versions, audits Python and web dependencies, and pins its actions to commits;
  Dependabot and CodeQL watch them. Typer 0.13 or later is required.
- The browser checks also run in the dark theme, and an axe scan of every view fails on any
  accessibility violation. The inspector's resize handle now says its width to assistive
  technology, and commands wider than their box can be scrolled from the keyboard.
- The privacy check also reads binary files and catches more home paths and short usernames.
- What 0.2 added was checked on GPT-2 small's real weights (see Models in the README).

## 0.1.2 (2026-10-08)

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
