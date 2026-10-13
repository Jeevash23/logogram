# Metrics and normalization

Every method reads one number from the model's output for each prompt: the **metric**. Effects
are changes in it, divided by a gap so that they read the same on any model and task. Both the
metric and how it is normalized are choices the spec states.

## Answers

Each prompt pair names an `answer` and a `distractor`: what the model should say after the clean
prompt, and what it says instead after the corrupt one (in IOI, the indirect object and the
repeated name). Each can be:

* **one token**, usually with its leading space (`" Mary"`, not `"Mary"`);
* **a set of tokens**, any of which counts: a list such as every year after the start year in
  greater-than, read as one answer whose probability is the sum of theirs;
* **a continuation of several tokens**, such as a capital's name: text whose probability is the
  product of each token's, each predicted from the prompt and the tokens before it.

The Prompts view says which pairs' answers are several tokens for the loaded model's tokenizer.

## The metrics

| Metric | `kind` | What it reads |
|---|---|---|
| Logit difference | `logit_diff` | logit(answer) − logit(distractor) at the last token. Single tokens, or sets read through the log of their summed probability |
| Log-probability difference | `logprob_diff` | log P(answer) − log P(distractor). Equal to the logit difference for single tokens; reads continuations too |
| Answer log-probability | `logprob` | log P(answer) |
| Answer probability | `prob` | P(answer) |
| Probability difference | `prob_diff` | P(answer) − P(distractor), as in greater-than (Hanna et al., 2023) |
| KL divergence | `kl` | KL(P_target ‖ P): how far the next-token distribution at the last token is from the clean or the corrupt prompt's own. It reads the whole distribution rather than the answer |

The logit difference is the usual choice (Wang et al., 2022): it is linear in the final residual
stream, so effects add up, and it doesn't saturate the way probabilities do. Probabilities show
what a person would see, but a component can matter a lot while the probability barely moves,
because it was already near 0 or 1. Zhang and Nanda (2024) compare these choices. The KL
divergence, as in Conmy et al. (2023), says whether the intervention restores the whole
prediction, not only the answer; its `target` (the prompt whose distribution it compares with)
has no default.

Whatever the metric, every run also keeps each prompt's probability of the answer and its
**preference**, log P(answer) − log P(distractor) (the logit difference for single tokens). Sign
flips are counted from the preference: the prompts whose preference for the answer changed sign
when the site was intervened on.

## Normalization

An effect is a change in the metric divided by a **gap**, the change that switching prompts
makes:

```text
effect = (intervened − receiver) / gap
```

For patching, the gap is (source − receiver): the metric on the prompt the activation comes from,
minus the metric on the prompt it goes into. For ablation, it is (corrupt − clean). So in every
method 1 means "as far as switching to the other prompt", 0 means no change, and the sign says
which way.

`metric.normalization` says which gap divides:

* **`dataset_gap`** divides every prompt's change by the gap averaged over the dataset. The mean
  effect is then the usual normalized metric, the mean change over the mean gap.
* **`prompt_gap`** divides each prompt's change by its own gap. A prompt whose gap is small then
  gives a large, unstable effect.

Logogram refuses a normalization the gap can't support: a mean gap near zero (the clean and
corrupt prompts give the same metric, so the model may not show the behavior), or, with
`prompt_gap`, a prompt with no gap at all. It warns when the mean gap is small next to its
standard error, or when prompts have gaps below 0.1. The [baseline check](../workbench.md#experiment-set-up-what-runs)
shows the gap before anything runs.

**The change itself** (the map's other view, "change in the logit difference" and so on) is
reported too, in the metric's own units, unnormalized.

## Readouts

A **readout** (`readout`) reads a value inside the model instead of its output, at one token of
each prompt (the last, an index or a labelled position):

* **a neuron**: one MLP neuron's activation after its nonlinearity (for a gated MLP, the gate's
  activation times the up projection);
* **a direction**: an activation's component along a [saved direction](directions.md) of one
  dimension, its dot product with the unit vector, at a residual stream site, attention output or
  MLP output and a layer you choose;
* **an SAE feature**: one feature of the spec's [SAE](sae-features.md), as the SAE encodes the
  activation it reads.

Patching and ablation then ask what drives that value: which earlier heads, MLPs and positions
write the neuron, the direction or the feature. Effects are normalized like any other metric's,
so 1 means the intervention moves the value as far as switching to the other prompt does. A site
later in the model than the value (a later layer, or a later part of the same layer) can't change
it, so its effect is exactly 0, and the run says so.

Readouts work with methods that run the model forward (activation patching, ablation and subspace
patching), on sites rather than on SAE features or sets of sites. Attribution patching and the
other gradient methods read the model's output.

## In a spec

```json
"metric": {"kind": "logit_diff", "normalization": "dataset_gap"}
```

```json
"metric": {"kind": "readout", "unit": {"kind": "neuron", "layer": 9, "neuron": 652},
           "position": {"kind": "last"}, "normalization": "dataset_gap"}
```

```json
"metric": {"kind": "kl", "target": "clean", "normalization": "dataset_gap"}
```

See [the spec reference](../reference/spec.md#metric) for every metric's fields.

## References

* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
* Hanna, Liu and Variengien (2023). *How does GPT-2 compute greater-than?: Interpreting
  mathematical abilities in a pre-trained language model.* NeurIPS.
* Conmy, Mavor-Parker, Lynch, Heimersheim and Garriga-Alonso (2023). *Towards Automated Circuit
  Discovery for Mechanistic Interpretability.* NeurIPS.
* Zhang and Nanda (2024). *Towards Best Practices of Activation Patching in Language Models:
  Metrics and Methods.* ICLR.
