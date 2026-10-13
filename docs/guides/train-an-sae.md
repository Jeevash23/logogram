# Train an SAE

This guide trains a sparse autoencoder on GPT-2 small's residual stream in the middle of the
model, on texts of your own, reads what its training measured, and uses it as a published SAE is
used: its features in **Features**, and in a spec that runs. Then it trains a tuned lens for the
same model and reads the layer predictions through it.
[Training an SAE](../concepts/sae-training.md) and [the tuned lens](../concepts/tuned-lens.md) say
exactly what each step computes.

Train one when no published SAE fits: a fine-tune, a model nobody has published SAEs for, a site
the published ones don't read, or text unlike theirs. Where a published SAE fits, it read far more
text than a project can (see [work with an SAE](work-with-an-sae.md)).

## 1. A corpus

Training reads plain text, not prompt pairs: a **corpus**, the file `corpora/NAME.jsonl` in the
project, one JSON object per line with its `text`:

```json
{"text": "The Eiffel Tower is in Paris, and the Colosseum is in Rome."}
```

Add one to the project from a text or JSONL file, a Hugging Face dataset or a dataset's prompts.
The more text, and the more like the text you will study, the better the SAE: a few thousand texts
of a hundred tokens or more make a first SAE on a laptop. The examples below call the corpus
`openwebtext-10k`, ten thousand texts of web text.

A training records the corpus's hash in the spec it keeps (`training.json`), so training again
from that spec refuses the corpus if its texts have changed since.

## 2. Train the SAE

With GPT-2 small loaded, go to **Explore → Features**. Under **Trained in this project**,
**Train an SAE…** opens the training dialog. It starts from values for the loaded model:

* `resid_post` of the middle layer (layer 6 of 12), with 6,144 features (eight times the
  768-wide residual stream), the 32 largest kept at each token;
* 2,000 steps of 1,024 tokens, at a learning rate of 0.00033 (Gao et al.'s, scaled by the width);
* a tenth of the texts held out, seed 0, the AuxK loss on, and at most 128 tokens of each text.

**Every other setting** shows the rest of the training spec: Adam's betas and epsilon, the
normalization, AuxK's latents, coefficient and when a latent counts as dead, the curve's points,
the tokens of each text and the texts per forward pass. Every value that can change the weights is
in the dialog, and none is filled in silently.

Below the settings, the dialog says what training would take on this machine: how many texts and
tokens it trains on and how often it reads each, the memory it holds at once (the activations of
every token it trains on, in float32) against what is free, and how long it takes, from one
forward pass and one step timed there and then. When it won't fit, lower the steps or the tokens
per step. When the settings can't train (too few tokens for a step, a site the model doesn't
have), it says why before anything runs.

**Train** starts it as a job. The dialog shows each stage, the step, and the loss curve: the
fraction of variance the SAE leaves unexplained on the batches it trains on, with the features
that fire per token. **Close** leaves it training (the status line shows how far it is), and
**Cancel the training** stops it and writes nothing.

## 3. Read what it measured

When it finishes, the dialog shows what the SAE does on the held-out texts, which it never trained
on:

* **Variance explained**: how much of the activations' variance its reconstruction keeps.
* **Features per token**: at most the 32 TopK keeps.
* **Never fire**: the share of features that fire on no held-out token. A large share is wasted
  width; AuxK, more steps or more text help.
* **Loss recovered**: the model's next-token loss with the SAE's reconstruction in place of the
  activation, from 0 (no better than setting it to zero) to 1 (as the model runs).

A low loss recovered means the reconstruction misses what later layers use, whatever the variance
explained says: train longer, on more text, or with more features before reading its features.
The curve shows whether it was still improving when it stopped.

The SAE is saved as `saes/gpt2-resid_post-6`, in SAELens's format, beside the spec it ran
(`training.json`) and what it measured (`metrics.json`). The same spec on the same machine gives
the same weights, bit for bit.

## 4. Use it

**Load this SAE** loads it and opens **Features**; later, it is under **Trained in this project**
with what it measured. Everything then works as with a published SAE (see
[work with an SAE](work-with-an-sae.md)): **Measure the fit** on your prompts, the features that
fire on each token and what each writes, **Your text**, attribution patching over every feature,
patching one, and steering along one.

A spec names it by its folder and the hash of its files:

```json
"experiment": {"kind": "attribution_patching", "direction": "clean_to_corrupt",
               "method": "gradient", "steps": null},
"scope": {"kind": "features", "position": {"kind": "last"}, "top": 50,
          "choose_on": 0.5, "seed": 0},
"sae": {"repo": "gpt2-resid_post-6", "path": "saes/gpt2-resid_post-6", "revision": null,
        "source": "project", "sha256": null}
```

The form fills in the hash of the SAE it loaded; with `sha256` null, the run fills it in when it
starts. Either way the run refuses an SAE whose files have changed since, and its manifest records
which SAE it read. Another project doesn't have it: copy the `saes/NAME` folder along with the
spec.

An SAE trained on another model, or on the model loaded with its weights processed the other way,
loads with a warning: its fit on your prompts says how much that matters.

## 5. Train a tuned lens

In **Explore → Layer predictions**, choose **Tuned lens** as the method, then **Train a tuned
lens…**. The dialog starts from a translator for every layer but the last, 250 steps of 1,024
tokens of AdamW at a learning rate of 0.001 with weight decay 0.001, and the same corpus settings.
It says what training would take, trains as a job with its curve (the KL divergence from the
model's prediction, averaged over the layers), and then shows each layer's held-out KL divergence
through the logit lens and through the tuned lens. The tuned lens should be lower at every layer.

**Read the layer predictions with it** chooses it in Layer predictions: each layer is read through
its translator, and the table marks how each row was read. The experiment's prediction diagnostic
records it by its name and hash in the spec, as `"predictions": {"method": "tuned_lens", "lens":
{"name": "gpt2-lens", "sha256": ...}, ...}`. A lens reads only the model it was trained on, loaded
the same way.

## From the command line

`logogram sae train` takes a training spec, such as the one in
[training an SAE](../concepts/sae-training.md#every-choice), or options that state every choice
(the [command line reference](../reference/cli.md#logogram-sae-train) lists them):

```bash
logogram sae train sae.json
logogram sae train --name gpt2-resid_post-6 --model openai-community/gpt2 --revision none \
  --dtype float32 --device auto --process-weights --corpus openwebtext-10k \
  --site resid_post:6 --d-sae 6144 --k 32 --steps 2000 --batch-tokens 1024 --lr 0.00033 \
  --betas 0.9,0.999 --eps 6.25e-10 --seed 0 --held-out 0.1 --normalize mean_squared_norm \
  --auxk 512,0.03125,102400 --log-every 20 --prepend-bos --max-tokens 128 --model-batch 16
logogram sae list
logogram lens train lens.json
logogram lens list
```

A training prints its progress, then what it measured on the held-out texts and how a spec names
what it trained. Ctrl+C stops it, and nothing is written.

## From Python

```python
import logogram as lg

project = lg.open_project("~/Documents/Logogram/my-project")
model = lg.load_model("openai-community/gpt2")
trained = lg.train_sae(project, "sae.json", model=model)   # a file, its JSON or a dict
trained["metrics"]["held_out"]   # variance explained, L0, the dead share, loss recovered
sae = lg.load_sae(trained["name"], project=project, model=model)
lens = lg.train_lens(project, "lens.json", model=model)
```

`lg.load_sae(NAME, project=project, model=model)` reads an SAE trained in the project from its
files, nothing downloaded, and `lg.run(spec, project, model=model, sae=sae)` runs a spec that
names it.

## References

* Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike and Wu (2024). *Scaling and
  evaluating sparse autoencoders.* arXiv.
* Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman and Steinhardt (2023). *Eliciting
  Latent Predictions from Transformers with the Tuned Lens.* arXiv.
