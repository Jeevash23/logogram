# The workbench

An open project fills the workbench. Across the top are the project (and **All projects**, the
logogram at the left), the three workspaces, and the tools: **History**, **Commands** (Ctrl/⌘ K),
**Help** and the appearance. Under them are the open workspace's tabs, the loaded model and the
dataset in use, and **Inspector**. The status line at the bottom shows the job in progress and
the connection to Logogram.

Each workspace has a color, which marks its tab, its progress and its choices: green for
Explore, orange for Experiment, pink for Evidence. Green also means a loaded model. Effects use
a diverging cobalt to amber scale, always with its legend, and nothing else does.

## Explore: look inside the model

* **Model** draws the model map with the model's real dimensions: one row per layer, from layer
  0 at the top, with its residual stream, each attention head, the attention output and the
  MLP. With a run open it is colored by that run's results; **Architecture** shows the structure
  alone. Queries, keys, values, neurons, sets of sites and edges have no cell of their own on the
  map: the run's results show them. **One layer** opens a layer's diagram, connected the way the
  model was measured to add attention and the MLP into the residual stream when it loaded (one
  after the other, or both from the same input). Choose components and **Add to experiment** to
  stage them: the experiment form then measures them, each on its own, or tests them together as
  a circuit. **Save selection** keeps named sites and notes in the project's research notes, and
  **Pin for comparison** keeps a head for Head comparison.
* **Attention** shows where a head attends on the prompts, as a heatmap of query against key
  tokens. **Find heads by what they attend to** scores every head as a previous-token,
  duplicate-token and induction head on seeded random sequences (see
  [find induction heads](guides/find-induction-heads.md)).
* **Head comparison** shows two pinned heads' attention side by side, on the same prompt and the
  same ink scale. Attention weights say where a head reads; whether that matters is for an
  intervention to measure.
* **Layer predictions** reads what each layer would predict through the model's final
  normalization and unembedding, the [logit lens](concepts/logit-lens-and-model-checks.md), for
  models where that reproduces their output.
* **Features** finds a published sparse autoencoder or transcoder by its release and id, in the
  catalogue of SAELens's registry that ships with Logogram, or by its Hugging Face repository and
  folder, loads it, measures how well it fits your prompts, and lists the features that fire on
  each token, with what one feature writes into the logits and the prompts where it fires most
  (see [SAE features](concepts/sae-features.md)).
* **Your text** reads any text without making a dataset: its tokens and the model's most likely
  next tokens after each, what each layer would predict at a chosen token, where a head attends,
  the SAE's features when one is loaded, and **Continue the text**, which shows what the model
  writes next: 1 to 64 new tokens, each the most likely one (a tie goes to the lowest token id),
  the whole text read again for each, and, with an SAE loaded, a second continuation with one of
  its features added (see [steering](concepts/steering.md)). Nothing is saved.
* **Corpus** keeps plain texts in the project, from a file, a Hugging Face dataset's file or a
  dataset's prompts, pages and searches them, and finds where a unit is most active in them: an MLP
  neuron, a feature of the loaded SAE, a saved direction or an attention head, with a histogram of
  its values and the tokens around each example shaded by its value (see
  [corpora and representations](concepts/corpora-and-representations.md)). An example opens in
  **Your text**.
* **Representations** reads the activation at one site and token of each prompt or text, draws
  them on two of their principal components, saves a component or the difference between two
  classes' means as a direction, and exports the activations. It lists the project's directions,
  with what made each, where and on which model, and renames and deletes them.

## Experiment: set up what runs

* **Prompts** lists a dataset's prompt pairs token by token, marks the tokens that differ, and
  says which pairs can't be used, and why. A pair whose prompts have different lengths can be used
  when its positions are labelled in both prompts, and the view shows both prompts' labels (see
  [pairs of different lengths](concepts/activation-patching.md#pairs-of-different-lengths)).
  **Generate a task** writes a seeded dataset for one of six classic tasks (IOI, greater-than,
  docstrings, gendered pronouns, subject-verb agreement, factual recall); **Import JSONL** and
  **Single pair** add your own. Edits to pairs stay in the tab until they are saved as a new
  dataset: a dataset file never changes, since runs record its hash.
* **Baseline** runs the model on the clean and corrupt prompts without intervening, to confirm the
  behavior exists before measuring what carries it.
* **Configure** is the experiment form: the method and its choices, what to sweep, the metric and
  its normalization, the statistics, and the data and model (with the chat template, for models
  made for chat). Beside it, a summary says what will run, how much work it is (for edge
  attribution, the graph's size and the passes and memory a batch needs; for ACDC, the passes
  and about how long they take here), whether a batch fits in memory, whether the baseline was
  checked, and what is still missing, above **Run**.
  While another job runs, **Run** reads **Queue**. Ctrl/⌘ Z and Ctrl/⌘ Shift Z undo and redo
  changes to the form. When an action elsewhere replaces the form (**Patch here**, **Edit and
  rerun**, staged sites), a notice offers to undo it. A form that isn't saved is kept in this
  browser for each project, and comes back when the project opens again, until it is saved, run
  or queued.
* **Spec** shows the complete `spec.json` of the form or of the open run: what the app, `logogram
  run` and the Python API all execute. See [the spec reference](reference/spec.md).

## Evidence: read what happened

* **Results** shows the open run: its title, method and notes, the figures at a glance, the map
  with its scale, a forest plot of the strongest sites with their intervals, and the actions:
  **Check robustness**, **Sweep…**, **Verify top 10 by patching** (for attribution patching),
  **Verify top 20 by path patching** (for edge attribution), **Test the top sites as a circuit**,
  and **Export** (the HTML report, figures and CSV; see [share a result](guides/share-results.md)).
  A run over queries, keys or values shows a layer × head map, a run over neurons each layer's
  strongest neurons beside the distribution of all of them, and an
  [edge attribution](concepts/edge-attribution.md) run its strongest edges drawn between the
  components, its faithfulness curve and their verification by path patching. An
  [ACDC](concepts/acdc.md) run draws the circuit it found the same way, with the edges it kept and
  the circuit's faithfulness, and lists every edge with whether it was kept.
* **Compare runs** puts two runs side by side, or their difference, with paired differences
  prompt by prompt where they measured the same prompts (see
  [compare two runs](guides/compare-runs.md)).
* **Sweeps** shows a sweep: one spec run across several seeds, datasets or checkpoints, and which
  sites keep their sign and size across them (see [sweeps](guides/sweeps.md)).
* **Research notes** keeps saved selections and notes, to reopen, edit or delete. **Save
  selection** (in Model) writes named sites, their exact positions, the model's settings, the run
  they came from and your observations to the project's `research.json`. A note changed meanwhile
  in another tab isn't overwritten: your text stays, to save as a separate note.

## Around the workspaces

* **The inspector**, at the right of Model, Attention and Results, shows the selected component
  or site: on a run, the exact intervention and metric, the effect with its interval, band and
  q-value, the sign flips, the distribution of per-prompt effects and the strongest and weakest
  prompts. On a window narrower than 1,120 pixels, **Inspector** opens it as a panel.
* **The token strip**, under most views, shows the current prompt pair token by token, lined up
  by their labels and last tokens when the two prompts have different lengths. **[** and **]**
  step through the prompts.
* **History** lists every run in the project, the queue, and drafts: search by title, name, id,
  method, model or dataset; rename (F2), star, archive or delete a run from its menu. A run's
  title, star and archive mark live beside its spec, which never changes. Deleting asks first,
  then removes the run's folder for good: there is no trash, since a deleted run kept out of sight
  would still travel with the project when it is shared. Runs derived from a deleted run stay, and
  say so. A run whose spec can't be read is listed as unreadable, with what is wrong and how to
  fix it.
* **The address** keeps your place: the view, and the run it shows or the two it compares. Back
  and Forward move between views, a reload comes back to the same place, and **All projects**
  leads back to the open project as you left it.
* **Commands** (Ctrl/⌘ K) runs any action by name, or jumps to a head (`L9H9`).
* **Help** opens the guide, the walkthrough of the example, the keyboard shortcuts, these pages
  on GitHub, a problem report to paste into an issue, the system check, and About.
