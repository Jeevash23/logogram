# Models

Logogram runs decoder-only language models through TransformerLens 4, which loads well over a
hundred architectures: GPT-2, Pythia, Llama, Mistral, SmolLM, Qwen, Gemma, OLMo, Phi and more. The
model dialog lists starting points, and any other Hugging Face id can be typed in. A model is
checked when it loads rather than trusted to be what its architecture's name says (see
[the model checks](concepts/logit-lens-and-model-checks.md)).

## Which models have been checked

GPT-2 small is the model the bundled examples were made for. Every method but edge attribution
has been run end to end with real weights on GPT-2 small (with a SAELens and an OpenAI SAE),
Pythia-70m (with an EleutherAI SAE) and Qwen 2.5 0.5B, and the test suite runs the sanity checks on
tiny random models of the Llama, Pythia, Qwen 2, Gemma 2 and OLMo 2 families without downloading
anything. Edge attribution has been run on GPT-2 small; it refuses models whose heads share keys
and values, as Qwen's and Llama's do, and models that normalize the attention output after
combining the heads (see [its limits](concepts/edge-attribution.md#limits)). On GPT-2 small,
patching the whole input gives an effect of exactly 1 under every metric, answers of several
tokens included; the logit difference and the log-probability difference, equal in exact
arithmetic for answers of one token, agree to 5 × 10⁻⁷; and reruns are bit-identical. Gemma Scope 2's files were checked on Gemma 3 270M with weight processing off.

A small model may not do a task at all (Pythia-70m prefers the repeated name in IOI), and its runs
then say so. The example's names are single tokens for GPT-2 and Qwen but not all for Pythia: for
another model, generate the prompts in the app with the model loaded, which keeps only words that
are single tokens for it. Some tokenizers, such as Qwen's, have no beginning-of-sequence token;
Logogram then runs prompts without one, and the spec records it. Pythia publishes checkpoints from
throughout training as revisions (`step1000` to `step143000`), so an experiment can be
[swept across training](guides/sweeps.md).

## Memory

Before downloading, Logogram checks that TransformerLens supports the architecture and estimates
the memory needed (the weights, the activations and a margin) against what is free. Processing the
weights needs more for a moment while the model loads, since TransformerLens works on float32
copies, three to four times the float32 size of the weights; the estimate includes them, so a model
that fits only without processing (Qwen 2.5 0.5B on a 6 GB GPU) says so before it loads. The
experiment form's summary says whether a batch of the size chosen fits. Attribution patching,
direct logit attribution and path patching hold every layer's activations at once, so they need
smaller batches than patching does. Edge attribution also holds the gradient at every receiver's
input: the summary says how much memory a batch needs, and a batch that won't fit is refused with
a batch size that will.

## Precision

Use float32 whenever the model fits. float16 and bfloat16 halve the memory and run two to three
times faster on a GPU, but they round. In the IOI example on GPT-2 small and Qwen 2.5, float16
moved effects by at most 0.003 and bfloat16 by up to 0.03: the strongest sites stayed in place, but
effects smaller than about 0.01 changed order. On Pythia-70m, bfloat16 doubled the clean logit
difference and reordered the heads, and float16 overflows (Logogram refuses to load it so). Check
results that matter against float32: **Check robustness** reruns a 16-bit run in float32 and
compares the two.

## Weight processing

`model.process_weights` folds the normalization's weights into the next layer and centers the
weights that write into the residual stream and the unembedding, as TransformerLens does by
default. Logit differences don't change. Value biases are folded into the attention output's bias,
so zero ablation of head outputs and heads' direct effects do change (by up to several logits in
Qwen 2.5, whose value biases are large); a layer's attention output doesn't. If processing changes
a model's predictions, Logogram refuses it and says to load with processing off. SAEs fit the
activations they were trained on, so load the model the way the SAE was trained.

## Devices and GPUs

`model.device` is `auto`, `cpu`, `cuda` (the default GPU), `cuda:N` (one GPU of several, such as
`cuda:1`) or `mps`.

* **Several GPUs.** The model dialog offers each GPU by name and free memory, and the memory
  estimate uses that GPU's free memory. A model loaded on `cuda` serves a spec that asks for
  either `cuda` or the default GPU's number. A spec naming a GPU this machine doesn't have is
  refused with the GPUs it has. `logogram doctor` lists them, each by its `cuda:N`, and each run's
  manifest records the GPU it ran on: its index and its name.
* **Apple Silicon.** **Automatic** runs models on the CPU: TransformerLens reports that Apple's MPS
  can give silently wrong results, and Logogram hasn't been checked on it, so MPS is used only when
  you choose it under **Device**. It is faster; check results that matter on the CPU.

The same spec on another device can differ in the last digits; see
[reproducibility](concepts/runs-and-projects.md#reproducibility).

## Models made for chat, and your own

* [Run a chat model on its own template](guides/chat-models.md): wrap prompts in the model's chat
  template, as the person's message or the start of the reply.
* [Study your own fine-tune](guides/local-models.md): add a model folder on this machine, named in
  specs by the hash of its files.
