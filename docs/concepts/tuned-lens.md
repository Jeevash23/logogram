# The tuned lens

The [logit lens](logit-lens-and-model-checks.md#the-logit-lens) reads each layer's residual
stream as if it were the last layer's. Layers don't all write in the basis the final layer reads,
so early layers read poorly that way, and some directions read too strongly. A **tuned lens**
(Belrose et al., 2023) learns, for each layer, an affine map, its **translator**, that carries the
layer's residual stream to what the final layer would make of it, trained so that the readout
predicts what the model predicts. Logogram trains one in the project, on a corpus of its texts,
and **Layer predictions** and the experiment's prediction diagnostic read through it.

## What is computed

For the residual stream `h_l` after layer `l` (`resid_post`) at one token:

```text
T_l(h) = h · W_l + b_l                       the translator: a d_model × d_model weight and a bias
q_l    = softmax(unembed(norm(T_l(h_l))))    what the tuned lens says layer l predicts
p      = softmax(unembed(norm(h_last)))      what the model predicts
```

`norm` is the model's final normalization, recomputed on the translated stream, and `unembed` its
vocabulary projection with its biases and any soft cap on the logits, exactly as the logit lens
reads a layer. With `W_l` the identity and `b_l` zero, the tuned lens is the logit lens.

Each translator is trained to minimize the **KL divergence** from the model's own distribution,
`KL(p ‖ q_l) = Σ p · (log p − log q_l)`, averaged over the tokens of a corpus's texts. The losses of
all layers are summed, and each translator learns from its own layer's only. The last layer has no
translator: its readout is the model's own prediction. Layers without a translator are read as the
logit lens reads them, and each row of the predictions says how it was read.

The model's own distribution is read from the last layer's `resid_post` through the same
normalization and unembedding. The
[model checks](logit-lens-and-model-checks.md#the-checks-on-loading) measure, when a model loads,
that this reproduces the model's output, and lenses are trained and read only on models where it
does.

## How it is trained

1. **The texts.** The corpus is read, refused if it has changed since its hash was pinned. A
   seeded share of its texts (`held_out`) is held out: never trained on, only measured on. Each
   text is tokenized (after the beginning-of-sequence token, with `prepend_bos`) and cut to
   `max_tokens` tokens.
2. **Steps.** `steps` steps of AdamW (`lr`, `betas`, `eps`, `weight_decay`; the same learning rate
   at every step). Each reads the next `batch_tokens` tokens of the training texts, read in a
   seeded order (another once every text has been read), `model_batch` texts per forward pass of
   the model, skipping each text's beginning-of-sequence token: its residual stream is the same
   in every text and unlike any other. The model's weights don't change; only the translators
   learn.
3. **The parameters.** Each translator is learned as `h + h · A_l + b_l`, with `A_l` and `b_l`
   starting at zero, so it starts as the logit lens and weight decay pulls it back towards the
   logit lens rather than towards zero (Belrose et al. start from the identity too). It is saved
   as `W_l = I + A_l`.

The loss curve records, every `log_every` steps, the mean of the steps since its last point: the
KL divergence averaged over the layers, and each layer's.

## What is measured

On the held-out texts, with the lens exactly as saved: for every layer but the last, the mean KL
divergence (in nats per token) from the model's own next-token distribution, through the logit
lens and, for the layers with a translator, through the tuned lens, at every token after the
beginning-of-sequence token. The training dialog, `logogram lens train` and the lens's
`metrics.json` show the two side by side, layer by layer, and `logogram lens list` their means. A
tuned lens should beat the logit lens at every layer; where it doesn't, it was trained too little,
or on too few texts.

## Every choice

A training spec (`logogram_lens_training: 1`) states every choice that can change the
translators, with no default; the [training specs](../reference/training.md#lens-training-spec)
list each field.

```json
{
  "logogram_lens_training": 1,
  "name": "gpt2-lens",
  "model": {"id": "openai-community/gpt2", "source": "hub", "revision": null, "sha256": null,
            "dtype": "float32", "device": "auto", "process_weights": true},
  "corpus": {"name": "openwebtext-10k", "sha256": null},
  "tokenization": {"prepend_bos": true, "max_tokens": 128},
  "layers": null,
  "steps": 250,
  "batch_tokens": 1024,
  "lr": 0.001,
  "betas": [0.9, 0.999],
  "eps": 1e-8,
  "weight_decay": 0.001,
  "seed": 0,
  "held_out": 0.1,
  "log_every": 10,
  "model_batch": 16
}
```

`layers` null gives every layer but the last a translator; a list chooses some. The app starts
from these values (AdamW at 10⁻³ with weight decay 10⁻³, 250 steps), shows every one and lets each
be changed. They are smaller than Belrose et al.'s recipe (SGD with Nesterov momentum, 250 steps of
2¹⁸ tokens each), to train in minutes on a project's corpus; the held-out KL says whether it was
enough.

## Reading with it

In **Explore → Layer predictions**, choose **Tuned lens** as the method and a lens from those
trained in the project. Only lenses trained on the loaded model can be chosen; the others say
which model they read. **Train a tuned lens…** opens the training dialog, and its **Read the
layer predictions with it** reads the prompt through the lens just trained. **Your text** reads
with the logit lens only.

A spec's prediction diagnostic names the lens by its name and the hash of its file
(`lens.safetensors`). A run pins the hash when it starts, refuses a lens whose file has changed
since, and records the lens in its manifest:

```json
"predictions": {
  "method": "tuned_lens",
  "lens": {"name": "gpt2-lens", "sha256": null},
  "prompt_index": 0,
  "which": "clean",
  "position": {"kind": "last"},
  "top_k": 5
}
```

A lens reads only the model it was trained on: the same files (a Hugging Face model's id and
revision, or a local model's hash), loaded with TransformerLens's weight processing the same way,
since processing centers the residual stream. On another model, or one of another width, it is
refused.

## The files

The lens is the folder `lenses/NAME` in the project, written whole or not at all (a cancelled or
failed training writes nothing), never over another:

* `lens.safetensors`: for each layer `L` it trained, `layers.L.weight` (`W_l`, read as
  `h · W_l`) and `layers.L.bias`, in float32.
* `training.json`: the spec as it ran, with the corpus's hash and the model's revision pinned.
* `metrics.json`: the loss curve, each layer's held-out KL divergence before and after, the texts
  and tokens it read, and where it ran (the versions of the libraries and the device).

## Reproducibility

The same spec on the same machine gives bit-identical translators. The split comes from the raw
PCG64 stream, as runs split prompts; the order of the texts comes from a seeded generator on the
CPU, whatever device the model is on; the batches' composition is fixed by the spec; the
translators train in float32; and deterministic algorithms stay on throughout (a training during
which they were turned off is refused). Another machine, device or version of PyTorch can round
differently in the last digits.

## Where it can mislead

* **It is fitted, not read off.** A translator is a probe trained to predict the model's final
  output. What a tuned lens reads at an early layer is what an affine map can extract from that
  layer, which needn't be what the model uses there. Belrose et al. check that their lenses rely
  on directions that matter to the model; Logogram doesn't. Test a reading with an intervention
  before relying on it.
* **Its target is the model's prediction, not the right answer.** The lens learns to agree with
  the final layer, so it shows how early the final prediction can be read, not whether a layer
  knows something the model later loses.
* **The corpus is its world.** A lens trained on prose may read code, another language or a chat
  template poorly; the held-out KL is measured on texts like the corpus's, not on your prompts.
* **A small corpus overfits.** Each translator has `d_model² + d_model` parameters (590,592 for
  GPT-2 small). With few tokens it can fit the training texts and read held-out ones worse than
  the logit lens: compare the two columns. Weight decay pulls it towards the logit lens.
* **A position it never saw.** It trained at every token after the beginning-of-sequence token.
  Read at that token itself, it reads a residual stream it was never trained on.
* **Descriptive, like the logit lens.** It shows what a layer's residual stream points to, not what
  the layer causes.

## References

* Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman and Steinhardt (2023). *Eliciting
  Latent Predictions from Transformers with the Tuned Lens.* arXiv.
* nostalgebraist (2020). *interpreting GPT: the logit lens.* LessWrong.
* Yom Din, Karidi, Choshen and Geva (2024). *Jump to Conclusions: Short-Cutting Transformers With
  Linear Transformations.* LREC-COLING 2024.
