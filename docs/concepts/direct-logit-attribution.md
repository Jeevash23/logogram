# Direct logit attribution

At the last token, the residual stream is a sum: the embeddings, plus what every attention head
and MLP wrote into it. The logits read that sum through the final normalization and the
unembedding. Direct logit attribution splits the logit difference into one term per component,
its **direct effect**: what it writes straight into the answer, without passing through later
layers (Elhage et al., 2021).

## What Logogram computes

For the prompts the spec names (`prompts`: the clean or the corrupt ones; there is no default,
since they split differently), Logogram runs the model once and reads, at the last token:

* what each head, attention output and MLP output of the scope wrote into the residual stream;
  a neuron's term is its activation times its row of the MLP's output projection, and a value's
  is what it adds to the last token's attention output through every head that reads it, with
  their attention as it was in the run;
* the final normalization, with its scale held at its value in that run, which makes the map
  from the residual stream to the logits linear;
* the unembedding of the answer minus that of the distractor.

With the scale held, the logit difference is exactly the sum of one term per component plus the
embeddings and the biases, and the results page shows that split: the logit difference, what
attention writes, what the MLPs write, and the embeddings and biases. A component's effect is
its term as a share of the mean logit difference; negative values push toward the distractor.

This is a decomposition of one forward pass, not an intervention: nothing is replaced, so there
are no sign flips or patched probabilities, and it splits the logit difference only.

## What it leaves out

A component's direct effect leaves out everything it does through later components: a head that
moves information for a later head to use has a small direct effect and a large total one.
[Activation patching](activation-patching.md) measures the total, and **Check robustness** offers
the same components patched, so direct and total effects can be read side by side. In the
bundled gendered pronoun example, heads 10.9 and 9.7 of GPT-2 small write about a third and a
fifth of the logit difference directly: the heads Mathwin et al. (2023) found.

Logogram refuses direct logit attribution where it isn't defined:

* for residual stream sites, which no single component writes;
* at positions other than the last token, which the logits don't read (a value at an earlier
  position reaches the last token through the attention, so values take any position);
* for queries and keys, which act on the logits only through the softmax of their head's
  attention;
* for single heads and values in models that normalize the attention output after combining the
  heads (Gemma 2, OLMo 2), where a head's write has no fixed share, and for neurons in models that
  normalize the MLP's output (Gemma 2, OLMo 2);
* for models that soft-cap their logits, which makes the map from the residual stream to the
  logits non-linear.

## When to use it

Use it to find the components that write the answer, at the cost of one forward pass, and to
read how a model's output is assembled. Then patch them to see how much of their effect is
direct.

## In a spec

```json
"experiment": {"kind": "direct_logit_attribution", "prompts": "clean"},
"scope": {"kind": "heads", "position": {"kind": "last"}},
"metric": {"kind": "logit_diff", "normalization": "dataset_gap"}
```

See [the spec reference](../reference/spec.md#experiment-direct_logit_attribution).

## References

* Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.* Transformer
  Circuits Thread.
* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
* Mathwin, Corlouer, Kran, Barez and Nanda (2023). *Identifying a Preliminary Circuit for
  Predicting Gendered Pronouns in GPT-2 Small.*
