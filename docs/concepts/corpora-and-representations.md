# Corpora and representations

Prompt pairs measure one behavior at a time. Much of interpretability starts elsewhere: with what a
unit of the model responds to in ordinary text, and with how the model's activations arrange
inputs that differ in some way. Logogram reads plain texts, kept in the project as corpora, and the
activations at a site over a dataset's prompts or a corpus's texts. Neither changes the model:
these are readings, not interventions. What they suggest, an intervention tests.

## Corpora

A corpus is `corpora/<name>.jsonl` in the project, one text per line, each with an optional label
(a string, a whole number or true/false) that names its class:

```text
{"text": "The Eiffel Tower is in Paris.", "label": "true"}
{"text": "The Eiffel Tower is in Rome.", "label": "false"}
```

Beside it, `corpora/<name>.json` says where the texts came from, how many there are and the
SHA-256 of the JSONL file. Whatever reads a corpus records that hash, and refuses a corpus that has
changed since.

Texts come from three places, in **Explore → Corpus**, with `logogram corpora import`, or with
`lg.import_corpus`:

* **A file**: plain text (one text per line, or texts separated by blank lines, as you choose),
  JSON Lines or a JSON array of objects, CSV or TSV with a header row, or Parquet (from the command
  line or Python). You name the field or column that holds each text, and optionally the one that
  holds its label. The corpus records the file's name and hash, never its folder.
* **A file of a Hugging Face dataset**: Logogram lists the dataset's files when asked, and
  downloads the one you choose once, at the exact commit its revision resolves to (the main branch
  by default); the corpus records the repository, the file and the commit. Downloads respect
  `HF_HUB_OFFLINE`: offline, a file already in the Hugging Face cache is read from there. In the
  app the download runs as a job, with its progress and a Cancel button.
* **A dataset's prompts**: the clean prompts, the corrupt ones or both (each pair's clean prompt,
  then its corrupt one), labelled by their side, by a field of their meta (such as `template`), or
  not at all. Each text's meta says which pair and side it came from.

Texts keep their characters exactly, spaces included: a line loses only its line break. Rows whose
text is empty are left out, and the import says how many; a row whose text isn't a string, or
whose label isn't a string, a number or true/false, is refused with its row number. A corpus holds
up to 256 MB of texts, and a text up to a million characters.

## Where a unit is most active

**Explore → Corpus** reads a corpus with the loaded model and finds the texts and tokens where one
unit is most active. A unit gives one number at every token:

| Unit | Its value at a token |
|---|---|
| An MLP neuron | Its activation after the nonlinearity (for a gated MLP, the gate's activation times the up projection): what the MLP's output projection reads. Offered where the model was measured, when it loaded, to read its neurons so. |
| A feature of the loaded SAE | Its activation, from the SAE's encoder on the activation the SAE reads. |
| A saved direction | The projection of the activation at the direction's site onto it, signed. For a subspace of several dimensions, the length of the activation's component in it. |
| An attention head | The length of what it writes into the residual stream: its output `z` through its slice of the attention's output projection. |

Each text is read as it is, after the beginning-of-sequence token when that is chosen, and cut to
its first *n* tokens. The beginning-of-sequence token is read but never counted: its activations
are unlike any other token's, and would top every list. Over every token counted, the answer gives:

* the *k* tokens where the unit is largest, each with a window of its text (a number of tokens on
  either side) and the unit's value at every token shown, shaded on the ink ramp;
* for a signed unit (a direction of one dimension), the *k* tokens where it is most negative;
* a histogram of its values, its mean, smallest and largest, and the share of tokens where it is
  above zero.

Every choice is part of the request and comes back with the answer: the corpus and its hash, how
many texts, how many tokens of each, the beginning-of-sequence token, the batch size, *k* and the
context. Texts run in batches of one token length, shortest first, each in the order of its texts,
so no batch needs padding and the same settings always give the same numbers; ties between equal
values go to the earlier text, then the earlier token. One reading covers at most four million
tokens.

**What it shows, and what it doesn't.** The top examples say where a unit is large in this
corpus. They are not what the unit does: a unit can be large for reasons a handful of examples
hide, and a corpus chosen for one theme finds that theme (Bolukbasi et al., 2021). Look at the
whole distribution, read many examples, compare corpora, and test a reading by intervening on the
unit with prompt pairs. An example opens in **Your text**, read at its token, to look further.

## Representations

**Explore → Representations** reads the activation at one site (a residual stream site, an
attention output or an MLP output) at one layer and one token of each prompt or text: the last
token, a token index (negative counts from the end), a labelled position of a dataset's prompts
(such as `S2`), or the mean over the tokens after the beginning-of-sequence token. Each prompt or
text gives one row.

**Principal components.** The rows are centered on their mean and decomposed by a singular value
decomposition in float64. Each component's sign is fixed so that its largest-magnitude loading is
positive, so the same rows always give the same components and coordinates. The answer gives the
share of the variance each of the first ten components explains, and each row's coordinates on
them; the plot shows the rows on any two, each class an ink shape.

**The mean difference between two classes** is the mean of the rows of one class minus the mean of
the rows of another: a dataset's prompts marked by side or by a field of their meta, a corpus's
texts by their labels.

Either can be saved as a [direction](#directions), with its model, site and how it was found. The
mean difference keeps its length before it was made unit length.

**What it shows, and what it doesn't.** Principal components are the directions of greatest
variance, which need not be the directions the model uses: variance follows the inputs chosen,
and two classes can differ along a direction of little variance. A mean difference is the simplest
direction separating two classes, and a strong baseline for linear representations (Marks and
Tegmark, 2023), but that it separates them shows only that the information is there, not that the
model reads it. Steer along a direction, or ablate it, to test whether it matters.

## Directions

A direction is a vector, or a few orthonormal vectors spanning a subspace, saved in the project's
`directions/` folder as safetensors with a JSON file saying what it is: the model it was found on
(by id and revision, or a local model by the hash of its files), the site and layer, how it was
made, and the hash of its tensor file. Principal components and mean differences saved in
Representations are directions, and so are a probe's weights and a learned subspace. The
Representations view lists the project's directions, marks one found on another model than the one
loaded (it may not mean the same thing there), and renames and deletes them; so do
`logogram directions` and its commands. A run that uses a direction keeps its own copy, so renaming
or deleting it never changes a run.

## Exported activations

**Export the activations** in Representations, `logogram activations` and `lg.export_activations`
write the activations at one or more sites and a position to
`activations/<name>.safetensors` in the project: one float32 tensor per site, named
`kind:layer` (such as `resid_post:6`), with one row per prompt or text. Beside it,
`activations/<name>.json` says what they are: the model (with its dtype and weight processing), the
sites and their width, the position, the dataset or corpus read and its hash, each row's index,
side and class, how texts were tokenized, the batch size, and the hash of the tensor file. The same
rows come back from Python as NumPy arrays with `lg.activations`, without writing anything.

```python
import logogram as lg
from safetensors.numpy import load_file

project = lg.open_project("my-project")
model = lg.load_model("openai-community/gpt2")
lg.export_activations(model, project, "ioi-s2", "datasets/ioi.jsonl", "resid_post:6", "label:S2")
acts = load_file(project.root / "activations/ioi-s2.safetensors")["resid_post:6"]  # [rows, width]
```

## References

* Bolukbasi, Pearce, Yuan, Coenen, Reif, Viégas and Wattenberg (2021). *An Interpretability
  Illusion for BERT.* arXiv.
* Bills, Cammarata, Mossing, Tillman, Gao, Goh, Sutskever, Leike, Wu and Saunders (2023).
  *Language models can explain neurons in language models.* OpenAI.
* Marks and Tegmark (2023). *The Geometry of Truth: Emergent Linear Structure in Large Language
  Model Representations of True/False Datasets.* COLM 2024.
* Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv.
* Park, Choe and Veitch (2023). *The Linear Representation Hypothesis and the Geometry of Large
  Language Models.* ICML 2024.
