# Find what a neuron reads

This guide reads a corpus of plain text with GPT-2 small and finds the tokens where one MLP neuron
is most active, then places the IOI prompts on their principal components, saves the difference
between the clean and corrupt prompts as a direction, and finds where that direction is active in
the corpus. [Corpora and representations](../concepts/corpora-and-representations.md) explains
what each step computes.

## 1. Make a corpus

Open a project, load GPT-2 small, and go to **Explore → Corpus**. A corpus comes from one of three
places:

* **Upload a file**: a text file (one text per line, or texts separated by blank lines), JSON Lines
  or a JSON array, or CSV or TSV with a header row. The view reads the file's first rows and offers
  their fields: choose the one that holds each text and, if the texts have classes, the one that
  holds each label.
* **Hugging Face**: type a dataset's id and press **List its files**; Logogram asks Hugging Face
  for its files at that moment, and only then. Choose a Parquet, JSON Lines, CSV or text file,
  name its text column (and its label column, if any), and keep the first few thousand texts.
  **Download and import** downloads the file once, at the exact commit, with its progress below and
  a Cancel button.
* **From prompts**: a dataset's clean or corrupt prompts, or both, labelled by side or by a field
  of their meta.

The same from the command line, in the project's folder:

```text
logogram corpora import notes notes.txt
logogram corpora import reviews --hub OWNER/NAME --file data/train-00000-of-00001.parquet --label-field label --limit 2000
logogram corpora import ioi-sides --dataset datasets/ioi.jsonl --label side
logogram corpora
```

**Show the texts** pages through the corpus a hundred texts at a time; the search finds texts by
their words, labels or meta.

## 2. Find where a neuron is most active

Under **Where a unit is most active**, choose **MLP neuron**, a layer and a neuron. The settings
below say how much is read: the first texts (1,000 by default), the tokens of each (longer texts
are cut), the batch size, how many examples to show and how many tokens of context around each.
**Find where it is most active** reads the texts and answers with:

* a sentence saying what was read and how often the neuron is above zero;
* a histogram of its values over every token counted, with zero marked;
* the tokens where it is largest, each in its window of text, every token shaded on the ink ramp by
  the neuron's value there, the token itself circled.

The beginning-of-sequence token is read but never counted. **Open in Your text** opens an example's
text there, read up to its window's end with the same beginning, at the example's token: what the
model predicts after it and what each layer would.

Read past the first few examples: a neuron large on one kind of token in the top ten may be large
on others further down, or in another corpus. A neuron that seems to respond to something is a
hypothesis, which patching or ablating it on prompt pairs tests.

## 3. Place prompts on their principal components

Go to **Explore → Representations**. Read **A dataset's prompts**: the IOI dataset, clean and
corrupt prompts, each marked by its side. Choose the residual stream leaving layer 6 and the token
labelled `end`, or the last token, and press **Find the principal components**. The plot shows each
prompt on the first two components, clean prompts as filled circles and corrupt ones as filled
squares; choose other components for the axes, hover over a point for its prompt, or move across
the plot with the arrow keys. The bars beside it say how much of the variance each component
explains.

The components follow the variance of these prompts, which need not be what the model uses: see
[what it shows, and what it doesn't](../concepts/corpora-and-representations.md#representations).

## 4. Save a direction, and find where it is active

**Save the difference between two classes' means as a direction** with clean minus corrupt saves
`clean-minus-corrupt-resid_post-6` in the project's directions folder: the model, the site and how
it was made are saved with it, and it is listed under **Directions in this project**, where it can
be renamed or deleted. **Save a component** saves a principal component the same way.

Back in **Corpus**, choose **Direction** and the direction just saved: the answer is now signed, so
it lists the tokens where the projection onto the direction is most negative too.

## 5. Export the activations

**Export the activations** writes the rows to `activations/NAME.safetensors` in the project, one
float32 tensor per site, with `activations/NAME.json` saying what each row is. Check **Every
layer's** to write one tensor per layer. The command line does the same without the app:

```text
logogram activations datasets/ioi.jsonl --site resid_post:6 --position label:end --mark side -o ioi-end --model openai-community/gpt2
```

## From Python

```python
import logogram as lg

project = lg.open_project("~/Documents/Logogram/ioi-example")
model = lg.load_model("openai-community/gpt2")
lg.import_corpus(project, "notes", "notes.txt")
top = lg.top_examples(model, project, "notes", {"kind": "neuron", "layer": 6, "neuron": 1234})
top["top"][0]       # the text and token where it is largest, with its window

pcs = lg.principal_components(model, project, "datasets/ioi.jsonl", "resid_post:6", "label:end", mark="side")
pcs["explained"]    # each component's share of the variance
difference = lg.mean_difference(model, project, "datasets/ioi.jsonl", "resid_post:6", "label:end", "clean", "corrupt", mark="side")
lg.save_direction(project, "clean-minus-corrupt", difference, model=model, site="resid_post:6", method="mean_difference")
acts = lg.activations(model, project, "notes", ["resid_post:6"], "mean")  # NumPy arrays, nothing written
```
