# Controls

A number from an intervention means little alone. A head whose removal moves the answer by 0.3
looks like a finding until five random heads do the same. A circuit that restores the behavior
looks sufficient until any set of that size does. Logogram has a control for each of these
readings. All of them are stated in the spec, so they are chosen before the results are seen.

## A gate on the baseline

An effect describes how a behavior is computed only if the model shows the behavior. A model that
prefers the wrong name on half the clean prompts has no behavior to explain there, and every
effect measured on those prompts describes something else.

Without a gate a run warns about this and goes on. With one, the baseline is measured first and a
run that falls short stops there, before any intervention, and says what it measured. A gate has
two bars and uses either or both.

* `min_correct` is the least share of the clean prompts on which the model must prefer the answer
  to the distractor, such as 0.9.
* `min_gap` is the least gap between the clean and the corrupt prompts in the metric, as the size
  of its mean over prompts.

```json
"gate": {"min_correct": 0.9, "min_gap": 1.0}
```

In the app the gate is its own section of the experiment form, below the metric. A run stopped by
its gate is listed as failed, with the measured share or gap in its message. Lowering a gate
after seeing that message is a choice to report, which is why the gate is saved with the spec.

Methods that hold some prompts out (steering, distributed alignment search) check the gate on the
prompts they measure. A probe reads no output, so it has no gate.

## Random sets beside a set of sites

A [set of sites](circuits.md) is read against sets of the same size chosen by chance. With
`controls`, every set that has sites gets `n` random sets, each intervened on exactly like it. A
random set has one site for each site of the set, of the same kind and at the same position. It
never holds a site of the set itself and never the same component twice. A head that the set
holds at two positions stays one head in the random set, at both positions.

`match` says where each site is redrawn.

* `layer` redraws it among the other heads, key or value heads, or neurons of its own layer. The
  random set then has as many sites in each layer as the set, which controls for depth. A layer
  has one attention output and one MLP output, so a set that holds those can't be matched by
  layer.
* `model` redraws it among the components of its kind in every layer.

```json
"scope": {
  "kind": "site_sets",
  "universe": null,
  "sets": [{"label": "name movers", "complement": false, "sites": [
    {"kind": "head", "layer": 9, "head": 9, "position": {"kind": "all"}},
    {"kind": "head", "layer": 9, "head": 6, "position": {"kind": "all"}},
    {"kind": "head", "layer": 10, "head": 0, "position": {"kind": "all"}}
  ]}],
  "singles": false,
  "controls": {"n": 100, "seed": 0, "match": "layer"}
}
```

The results give, for each set, the mean of its random sets' effects, the range that holds the
run's confidence level of them, and how many are at least as far from zero as the set's own
effect. That count is also given as `p`, (count + 1) / (n + 1). Twenty random sets can show at
best p = 0.048, so draw a hundred or more when the claim rests on it.

Each set draws from its own stream of the seed, so adding a set doesn't change another's random
sets. Every random set is a row of the saved results, with its sites, and can be inspected like
any other.

## Each site alone

Sites patched together can do more than the sum of what each does alone, or less. With `singles`,
each site of a set of two or more is also intervened on alone, in the same run. The results then
give the set's effect beyond the sum of its sites' effects alone, with an interval from the same
resamples. An interval above zero says the sites work together. One below zero says they overlap,
as backup heads do.

`singles` applies to sets that intervene on their sites. A set that keeps its sites and replaces
the rest of the model has no sites alone.

## Donors chosen by a field

[Resample ablation](ablation.md) replaces an activation with the same activation from other
prompts. What a donor shares with the prompt decides what the ablation removes. Donors from the
same template keep the template and vary the names. Donors with a different subject remove the
subject and whatever follows from it.

A resample baseline's `group` names a field of the prompts' `meta` and draws each prompt's donors
only among prompts with the same value of it, or only among those with a different one.

```json
"baseline": {
  "kind": "resample", "pool": "corrupt", "donors": 10, "seed": 0,
  "group": {"field": "template", "match": "same"}
}
```

Running the same sweep with two donor groups and [comparing the runs](../guides/compare-runs.md)
shows which conclusions depend on the choice of donors.

## Two crossed fields in the bootstrap

Prompts built from two lists are tied two ways. In a dataset of names and places, a prompt shares
its name with some prompts and its place with others. A bootstrap over prompts treats them all as
independent, and a bootstrap over one field ignores the other. Both give intervals that are too
narrow.

`statistics.cross` names a second field beside `statistics.cluster`. Each resample then draws
the first field's values with replacement and, independently, the second's. A prompt counts as
often as its two values' draws multiplied (Owen, 2007). See [statistics](statistics.md).

```json
"statistics": {"bootstrap": 2000, "ci": 0.95, "seed": 0, "cluster": "name", "cross": "place"}
```

## Many sites at once

A sweep over 144 heads makes 144 comparisons, and some intervals will exclude zero by chance.
Every finished sweep reports three corrections, described in [statistics](statistics.md). The
simultaneous band holds for all sites together. The q-value bounds the share of chance findings
among the sites reported. The Holm-adjusted p-value is the strictest of the three and bounds the
chance that even one reported site is there by chance.

## References

* Owen (2007). *The pigeonhole bootstrap.* Annals of Applied Statistics.
* Holm (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of
  Statistics.
* Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris and
  Thomas (2022). *Causal Scrubbing: a method for rigorously testing interpretability hypotheses.*
  Alignment Forum.
* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
