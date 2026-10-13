# Distributed alignment search

Activation patching replaces a whole site: a residual stream vector of hundreds of dimensions. A
model may carry the variable that matters (which name to say, say) in a few of those dimensions,
mixed with everything else. Distributed alignment search (DAS; Geiger et al., 2023) learns the
subspace instead of assuming one: it looks for a few directions whose patching alone carries the
effect.

## What Logogram computes

At each site of the scope (a residual stream site, attention output or MLP output, at one token of
each prompt), an orthonormal basis Q of `dimensions` vectors is learned so that patching only the
component within it, from the source prompt into the receiver run,

> receiver activation + ((source activation − receiver activation) · Q) · Qᵀ,

moves the metric toward the source prompt's as far as it can.

**The split.** The pairs are split with a seed into pairs that train the subspace
(`train_fraction`) and held-out pairs that measure it, so no pair measures the subspace it
trained.

**The training.** Q is the orthonormal part of a free matrix (its QR decomposition, in float64 on
the CPU), which starts from a random draw seeded by the seed and the site. Adam (`learning_rate`,
`steps`) maximizes the mean normalized effect over the training pairs, each step over all of them
in their batches: the effect normalized by the training pairs' clean–corrupt gap, as the run
normalizes effects. Batch composition is fixed and every draw is seeded, so a rerun is
bit-identical.

**The measurement.** The held-out pairs are then patched three ways at each site:

* within the **learned subspace**;
* the **whole site** (activation patching: the most any subspace of it can carry);
* with `control`, within a **random subspace** of the same size, drawn with the seed.

Each is a column of the results, normalized like any patching effect. The results also give the
learned subspace's share of the whole site's effect, and the effect on the training pairs after
each step (the training curve).

## Where it can mislead

A learned subspace is chosen to change the output, so it can find one that does, whether or not
the model uses it as a variable: a subspace whose patching activates a dormant pathway, or that
pushes the activation off the distribution the model sees (Makelov et al., 2023). Read the learned
subspace against the whole site and the random subspace, check it on prompts of another template
or task, and test what it means with other methods: a probe for the variable, removing it with
directional ablation, or reading it out while patching elsewhere ([Directions](directions.md)).

Training is gradient ascent from one starting point: another seed can find another subspace.
**Check robustness → Another split of the pairs** reruns with the next seed.

## Saving a learned subspace

Each site's learned basis is kept in the run's `vectors.safetensors`. **Save as direction** in the
results, `logogram directions save RUN SITE NAME` or `run.save_direction(site, name)` saves it as a
project direction of that many dimensions (or one of its vectors, with `row`), to patch along,
remove or read out in other runs.

## In a spec

```json
"experiment": {
  "kind": "das",
  "direction": "clean_to_corrupt",
  "dimensions": 1,
  "train_fraction": 0.5,
  "seed": 0,
  "steps": 100,
  "learning_rate": 0.01,
  "control": true
},
"scope": {"kind": "layer_components", "components": ["resid_post"], "position": {"kind": "label", "label": "S2"}}
```

See [the spec reference](../reference/spec.md#experiment-das).

## References

* Geiger, Wu, Potts, Icard and Goodman (2023). *Finding Alignments Between Interpretable Causal
  Variables and Distributed Neural Representations.* CLeaR.
* Wu, Geiger, Icard, Potts and Goodman (2023). *Interpretability at Scale: Identifying Causal
  Mechanisms in Alpaca.* NeurIPS.
* Makelov, Lange and Nanda (2023). *Is This the Subspace You Are Looking for? An Interpretability
  Illusion for Subspace Activation Patching.* ICLR 2024.
