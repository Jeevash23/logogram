# The logit lens and the model checks

Logogram runs models through TransformerLens, which rebuilds each architecture from its weights.
A model is not trusted to be what its architecture's name says: when it loads, Logogram measures
what the methods rely on, and offers a method only where the property it needs was measured.

## The checks on loading

On a short fixed input, every model is checked for four things:

* **Predictions.** TransformerLens's version of the model must predict what the original model
  predicts. If processing the weights (folding the normalization into the weights and centering
  them, as TransformerLens does by default) changes the predictions, loading with processed
  weights is refused, with the advice to load with processing off. So is a dtype in which the
  model overflows, such as Pythia in float16.
* **How layers add up.** Logogram measures how each layer adds attention and the MLP into the
  residual stream: one after the other (sequential, as in GPT-2 and Llama), or both from the same
  input (parallel, as in Pythia, GPT-J and Phi). Only sequential layers have a residual stream
  between attention and the MLP (`resid_mid`), and the layer diagram in Explore draws what was
  measured. Whether the attention output is the sum of its heads' writes is measured too: it
  isn't in models that normalize the attention output after combining the heads (Gemma 2 and 3,
  OLMo 2), so single heads' direct effects, and what features on heads' outputs write, are
  refused there.
* **Queries, keys, values and neurons.** Taken from another input, every query, key and value of
  every layer must give that input's attention outputs, and every neuron its MLP outputs; one
  query, key or value replaced alone may change only the heads that read it. Only the sites that
  pass are offered, and the model says why the others aren't: OLMo 2 normalizes every head's
  queries and keys together, so it offers values and neurons only. Whether each head's output is
  its attention pattern times its values, and each MLP's output the neurons' writes plus its bias,
  is measured too, for direct logit attribution.
* **The lens.** The logit lens below is offered when the model's final normalization and
  unembedding, with any soft cap on its logits, reproduce its own output.

[Edge attribution](edge-attribution.md#limits) relies on more, measured the first time it plans
or runs on a model: that giving each receiver its own copy of the residual stream doesn't change
what the model predicts, that every copy is read before its normalization, and that changing one
head's copy changes no other head.

The measured layer structure is kept with each run's results, so a saved run's map can be drawn
without loading the model. A model that looks made for chat (its name says so, such as `-Instruct`, or it stops generating at the token
that ends each turn in its chat template) is flagged too, until its prompts are wrapped in its
[chat template](../guides/chat-models.md).

## The logit lens

The logit lens reads what each layer would predict if the model stopped there: the residual
stream after the layer, at one token, through the model's final normalization and unembedding
(nostalgebraist, 2020). Logogram's **Layer predictions** implements a final-norm logit lens:

* each layer's `resid_post` at the chosen token goes through the loaded model's final
  normalization, recomputed at every layer, and its vocabulary projection, including biases and
  any soft cap on the logits, then a softmax;
* for the clean or the corrupt prompt of a pair, at the last token or an indexed one, with 1 to 20
  of the most likely tokens per layer;
* shown as a trajectory and a table: the answer's and the distractor's probability, the logit
  difference, and the top tokens at each layer.

The lens is descriptive: it shows what a layer's residual stream points to through the final
layers' readout, not what the layer causes, and early layers' predictions aren't calibrated. No
lens is trained, as the tuned lens of Belrose et al. (2023) does. At an earlier position, the
answer and distractor are still the dataset's, not the token expected there.

Measuring adds an optional diagnostic to the experiment form, which a run saves beside its
results as `predictions.json` without changing the intervention:

```json
"predictions": {
  "method": "final_norm_logit_lens",
  "prompt_index": 0,
  "which": "clean",
  "position": {"kind": "last"},
  "top_k": 5
}
```

**Your text** shows the same lens for any text, at any token: choose a token to read it layer by
layer.

## Exactness

The checks run in the tests too, on tiny random models of the Llama, Pythia, Qwen 2, Gemma 2 and
OLMo 2 families, built without downloading anything: patching identities, bit-identical reruns
and the measured layer structure must hold on each.

## References

* nostalgebraist (2020). *interpreting GPT: the logit lens.* LessWrong.
* Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman and Steinhardt (2023). *Eliciting
  Latent Predictions from Transformers with the Tuned Lens.* arXiv.
* Nanda and Bloom (2022). *TransformerLens.* Software.
