# Linear probes

Patching asks what a site *does*. A probe asks what can be *read* from it: train a linear
classifier on a site's activations to recover a label, and measure how often it is right on
prompts it never saw (Alain and Bengio, 2016; Belinkov, 2022). Where a probe succeeds, the
information is there, in a form a linear map can read. Where it fails, it may still be there in a
form a linear map can't read, or not at all.

A probe doesn't show that the model uses what it reads. A site can carry a label that nothing
downstream reads, and a probe can find a direction the model never uses. Probing finds candidates;
patching tests them. Save a probe's direction and patch along it, ablate it or steer along it
([Directions](directions.md)) to ask whether the model uses it.

## What Logogram computes

**Examples and labels.** A probe runs on prompt pairs. With `"labels": {"kind": "side"}` each pair
gives two examples, its clean prompt (labelled clean) and its corrupt prompt (labelled corrupt):
where in the model can you tell which prompt it is reading? With `{"kind": "meta", "field":
"template", "prompts": "both"}` each prompt is labelled by a field of its pair's `meta` (the task
generators record each prompt's template, and the gendered-pronoun and subject–verb agreement tasks
record the gender and number), from its clean prompt, its corrupt prompt or both.

**The split.** The pairs, not the examples, are split with a seed (`train_fraction`, `seed`) into
pairs that train the probes and held-out pairs that measure them, so the two prompts of a pair are
never on both sides. Every label among the held-out examples must appear among the training ones;
a run refuses a split that leaves one out, naming it.

**The probe.** At each site of the scope, at one token of each prompt (the last token, a token
index or a labelled position), the activation is standardized with the training examples' mean
and spread, and a logistic regression (a weight vector and a bias per class) is fitted to it by
L-BFGS in float64, starting from zero, for `iterations` iterations, to minimize the mean
cross-entropy plus `regularization` / 2 times the squared length of the weights. With
`regularization` 0 and labels a hyperplane separates, the weights grow until the iterations end:
a little regularization keeps the fit, and the direction, well defined.

**The value.** Each held-out pair's value is the share of its examples the probe labels
correctly: 0, 0.5 or 1 for a pair that gives two examples. A site's mean over the held-out pairs
is the probe's held-out accuracy, with the run's bootstrap interval. The results also give the
accuracy on the training examples (a probe far better there than on held-out examples has
memorized) and chance: the held-out accuracy of always answering the most common training class.

**The control.** With `control`, a second probe at each site is fitted to the training labels
shuffled with the seed, and measured on the true held-out labels: a control task in the spirit of
Hewitt and Liang (2019). It shows how far a probe of that size gets without the information, from
the dimensions alone. **Selectivity**, the probe's accuracy minus its control's, is measured pair
by pair on the same resamples, with its own interval; a probe is selective where that interval
lies above zero.

## Where it can mislead

* A high accuracy says the label is linearly readable, not that the model reads it, and not that
  the probe's direction is the one the model uses: many directions can separate the same
  examples.
* With few examples and wide activations, a probe can separate almost any labels. Read the control
  and the training accuracy beside the held-out accuracy, and use more prompts.
* Labels that every prompt of a template shares (`template`, say) can be read from anything that
  differs between templates, such as the position of the last token. A probe for which prompt it
  is reads, at a site where the two prompts differ by one token, that token's identity.
* The bootstrap resamples held-out pairs. Prompts from one template aren't independent: cluster
  the bootstrap by template (`statistics.cluster`) when they share one.

## Saving a probe's direction

A probe's weights are kept in the run's `vectors.safetensors`, in the activation's own units (the
standardization undone). **Save as direction** in the results, `logogram directions save RUN SITE
NAME` or `run.save_direction(site, name)` saves one site's as a project direction: with two
classes, the second class's weights minus the first's, the direction from the first class toward
the second. See [Directions](directions.md) for what can be done with it.

## In a spec

A probe reads activations, not the model's output, so its spec has no metric (`"metric": null`).

```json
"experiment": {
  "kind": "probe",
  "labels": {"kind": "side"},
  "train_fraction": 0.5,
  "seed": 0,
  "regularization": 0.01,
  "iterations": 100,
  "control": true
},
"scope": {"kind": "layer_position", "site": "resid_post", "positions": "labels"},
"metric": null
```

See [the spec reference](../reference/spec.md#experiment-probe).

## References

* Alain and Bengio (2016). *Understanding Intermediate Layers Using Linear Classifier Probes.*
  arXiv.
* Hewitt and Liang (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP.
* Belinkov (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational
  Linguistics.
