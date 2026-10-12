# Attribution patching

[Activation patching](activation-patching.md) runs the model once per site and prompt. That is
exact, and slow for large sweeps: every head at every position of a larger model, or every
feature of an SAE. Attribution patching estimates all of them at once from gradients, then
leaves the estimates to be checked by patching.

## What Logogram computes

For each site, the change that patching it would cause in the [metric](metrics.md) is estimated
to first order:

```text
estimated change = (source activation − receiver activation) · ∂metric/∂activation
```

with the gradient taken at the receiver run. One forward pass on the source prompts and one
forward and backward pass on the receivers cover a whole sweep, so every head at every position
takes seconds even on larger models (Nanda, 2023; Syed, Rager and Conmy, 2023). The estimate is
normalized like patching, so 1 would mean the site alone restores the clean behavior. The
results label every value as an estimate.

A first-order estimate assumes the metric changes linearly between the two activations. It
misses saturation, in attention patterns, normalization and the final softmax, and so can miss an
effect or even give it the wrong sign (Kramár et al., 2024).

## Integrated gradients

With `"method": "integrated_gradients"`, the gradient is averaged over `steps` runs whose input
embeddings lie evenly between the receiver prompt's and the source prompt's (at the midpoints of
equal intervals), as in EAP-IG (Hanna, Pezzelle and Belinkov, 2024), after integrated gradients
(Sundararajan, Taly and Yan, 2017). The estimate then follows the metric along the way from one
prompt to the other instead of only at its start. It costs `steps` forward and backward passes
instead of one; 8 to 16 steps are usual.

On the bundled IOI example with GPT-2 small, the difference shows in the residual stream.
Patching the residual stream at the changed name (S2) entering layer 0 restores the whole answer
(1.00); one gradient estimates −0.17, and 16 steps of integrated gradients 1.00. Over the 48
residual stream sites at the named positions, the rank correlation with patching is 0.92 for 16
steps and 0.34 for one gradient. Head outputs are easier: with 8 steps, the ten strongest
estimated heads rank as patching ranks them (0.98), and patching reverses none of their signs.
The [attribution patching notebook](../../notebooks/attribution-patching.ipynb) measures all of
this.

## Verifying estimates by patching

An estimate says where to look; patching says what is there. **Verify top 10 by patching** on the
results of an attribution patching run patches the ten sites with the largest estimated effects,
for real, in a run of its own that records which run it verifies, and opens the comparison:

* the **rank correlation** between estimate and measurement over those sites;
* any estimate whose **sign** patching confidently reverses (both intervals exclude zero);
* each site's estimate beside its measured effect and interval.

The same is `logogram verify RUN --top 10` on the command line and `lg.verify_run(run, top=10)`
in Python. Verifying only the top estimates can't find a site the estimate missed altogether, so
for residual stream sweeps, where estimates are least reliable, use **Check robustness** to patch
the whole sweep and compare. In the IOI example, the residual stream at S2 in the first layers
has a negative one-gradient estimate where patching restores the whole answer, so verifying the
top estimates would never reach it.

## SAE features

Attribution patching can sweep every feature of an SAE at one position (the `features` scope)
and keep the strongest as the run's sites, ready to verify by patching. Choosing the strongest of
thousands and reporting them on the same prompts biases their effects away from zero (the
winner's curse), so the scope can choose them on a seeded share of the prompts
(`choose_on`, `seed`) and report them on the rest. The results also say how much of the whole
site's estimated effect the features account for; the rest is the SAE's error, and for a skip
transcoder part of it flows through the skip connection. See [SAE features](sae-features.md).

## When to use it

Use attribution patching to search: large sweeps, many positions, SAE features, larger models.
Then verify what it finds by patching, and patch the whole sweep where it matters. Prefer
integrated gradients for the residual stream and wherever one gradient and patching disagree.

## In a spec

```json
"experiment": {
  "kind": "attribution_patching",
  "direction": "clean_to_corrupt",
  "method": "integrated_gradients",
  "steps": 8
}
```

`steps` is `null` with `"method": "gradient"`. See
[the spec reference](../reference/spec.md#experiment-attribution_patching).

## References

* Nanda (2023). *Attribution Patching: Activation Patching At Industrial Scale.* Blog post.
* Syed, Rager and Conmy (2023). *Attribution Patching Outperforms Automated Circuit Discovery.*
  arXiv (BlackboxNLP 2024).
* Kramár, Lieberum, Shah and Nanda (2024). *AtP\*: An efficient and scalable method for
  localizing LLM behaviour to components.* arXiv.
* Sundararajan, Taly and Yan (2017). *Axiomatic Attribution for Deep Networks.* ICML.
* Hanna, Pezzelle and Belinkov (2024). *Have Faith in Faithfulness: Going Beyond Circuit Overlap
  When Finding Model Mechanisms.* COLM.
