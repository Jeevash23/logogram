# Ablation

Ablation removes a component's contribution and measures what the model loses. "Removes" has to
mean replacing the activation with something, and what it is replaced with changes the result,
so in Logogram the **baseline** is a choice the spec must state. There is no default.

## What Logogram computes

Ablation runs the **clean** prompt and replaces the activation at a site with a baseline value,
then reads the [metric](metrics.md). The effect is normalized by the gap between the corrupt and
the clean prompt:

```text
effect = (ablated − clean) / (corrupt − clean)
```

so 1 means ablating that site alone moves the metric as far as the corrupt prompt does, 0 that it
changes nothing, and a negative value that the model does the task better without it. The gap is
averaged over the dataset or taken per prompt, as the spec's
[normalization](metrics.md#normalization) says.

## Baselines

* **Zero** (`{"kind": "zero"}`) replaces the activation with zeros. It is the simplest, and the
  furthest from anything the model sees: a layer that never outputs zero is pushed off its usual
  range, so a large effect may say more about that than about the component.
* **Mean** (`{"kind": "mean", "reference": "clean" | "corrupt"}`) replaces it with its mean over a
  stated set of prompts: the dataset's clean prompts, or its corrupt ones. When every position is
  replaced, the mean is taken position by position, over the reference prompts of the same
  length; at one position, at each reference prompt's own resolved position. It keeps what all
  prompts share and removes what is particular to each. Wang et al. (2022) mean-ablated heads over
  prompts with three unrelated names, which keep the sentence and remove which name repeats.
* **Resample** (`{"kind": "resample", "pool": …, "donors": 10, "seed": 0}`) replaces it with the
  same activation from other prompts, its donors: each prompt draws `donors` of them from the pool
  without replacement, never itself, with the stated seed. The result is averaged over the
  donors, and the same donors serve every site. Each replacement is something the model really
  computes, on another input; resampling is the replacement causal scrubbing uses (Chan et al.,
  2022).

The baselines answer different questions, so a component that matters under one can matter less
under another. **Check robustness** reruns an ablation with another baseline, or with more
donors, and compares the two.

## Weight processing and zero ablation

TransformerLens's weight processing, a choice in the spec (`model.process_weights`), folds each
head's value bias into the attention output's bias. A head's output `z` is then a different
number with or without it, so zeroing a head's output removes a different amount: in Qwen 2.5,
whose value biases are large, zero-ablating a head gives results that differ by up to several
logits between the two. Zeroing a layer's whole attention output doesn't depend on it, and nor
do mean and resample ablation or patching, which replace an activation with one computed the
same way.

## When to use it

Use ablation when there is no natural corrupt prompt, to ask what the model needs from a
component on the clean prompts, and to replace "the rest of the model" when testing a
[circuit](circuits.md). Prefer mean or resample ablation to zeros. Where a corrupt prompt exists,
[activation patching](activation-patching.md) from corrupt to clean asks a closely related,
better-controlled question.

## In a spec

```json
"experiment": {
  "kind": "ablation",
  "baseline": {"kind": "resample", "pool": "corrupt", "donors": 10, "seed": 0}
}
```

See [the spec reference](../reference/spec.md#experiment-ablation) for every field, and
[baselines](../reference/spec.md#baseline).

## References

* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
* Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris and
  Thomas (2022). *Causal Scrubbing: a method for rigorously testing interpretability hypotheses.*
  Alignment Forum.
* Heimersheim and Nanda (2024). *How to use and interpret activation patching.* arXiv.
