# Study your own fine-tune

A model that isn't on Hugging Face, such as your own fine-tune, can be added from its folder on
this machine. Specs then name it by a name you give and the hash of its files, never by its
folder, so specs, manifests, summaries and reports say nothing about where the files are, and a
spec shared with someone else finds the same model on their machine.

## 1. Add the folder

The folder holds `config.json`, the tokenizer's files, and the weights as safetensors. A folder
whose weights are only pickles (`.bin`, `.pt` or `.pth`) is refused, since opening a pickle can
run code; save the model with `save_pretrained(..., safe_serialization=True)` first.

* In the app: **Load a model → Add a model folder**.
* On the command line: `logogram models add PATH --name my-fine-tune`.
* From Python: `lg.add_local_model(path, "my-fine-tune")`.

Its files are hashed once, with progress for big folders. `logogram models list` lists the
folders added on this machine with their hashes, and `logogram models remove NAME` forgets one
(the folder stays as it is).

## 2. Load and run it

Choose it in the model dialog, or `lg.load_model("my-fine-tune")`. A local model loads and is
checked like any other, on any architecture TransformerLens supports (see
[the model checks](../concepts/logit-lens-and-model-checks.md)). Its spec says:

```json
"model": {
  "id": "my-fine-tune",
  "source": "local",
  "revision": null,
  "sha256": "9f2c…",
  "dtype": "float32",
  "device": "auto",
  "process_weights": true
}
```

## 3. Compare it with its base model

The natural question about a fine-tune is what changed. Run the same spec on the base model and on
the fine-tune, which differ only in `model`, then **Compare runs**: on the same prompts, each site
is compared prompt by prompt, with an interval for the difference (see
[compare two runs](compare-runs.md)). Check both baselines first: a fine-tune that no longer does
the task has a smaller gap to normalize by.

## How the hash is made

The hash is the SHA-256 of a text listing the files loading reads, one line each, sorted by name:
the file's own SHA-256 in lowercase hexadecimal, two spaces and its name, each line ending with a
newline, as `sha256sum` prints them. The files are those in the folder itself, not in subfolders,
that Logogram downloads for a model on Hugging Face: every `*.json`, `*.txt`, `*.model`,
`*.tiktoken` and `*.jinja` file (except hidden files and ONNX, Flax or TensorFlow exports), and the
weights: `model.safetensors`, or else the shards `model.safetensors.index.json` names. Names
compare in Unicode's composed form (NFC), so the hash is the same on every machine and file
system. On Linux, with those names in `files.txt`:

```bash
sha256sum $(cat files.txt) | LC_ALL=C sort -k2 | sha256sum
```

## When files change

The list of local models lives in Logogram's settings folder, beside the recent projects, with
each file's size, modification time and hash. Loading a local model checks its files first: a
file whose size or modification time changed is hashed again, and a folder whose files changed,
or that gained or lost one, is refused, naming them. Changed files make a new model: add the
folder again under a new name. A spec from someone else whose model this machine doesn't have
says so, with the start of the hash it needs and how to add the folder.
