# Circuits

Measuring sites one at a time finds the components that matter. A **circuit** is a claim about a
set of them: these components, together, carry the behavior. Logogram tests such a claim by
intervening on whole sets of sites at once, and reads each set against replacing everything, as
Wang et al. (2022) evaluated their IOI circuit by its faithfulness, completeness and minimality.

## Sets of sites

A `site_sets` scope lists **sets**, each intervened on in one forward pass, with one result per
set:

* a set that is not a `complement` intervenes on its sites, together;
* a `complement` set intervenes on every component of the scope's **universe** (every head, or
  every attention and MLP output, at every position) except its sites. A set of heads then reads
  as "keep this circuit, replace the rest". A site at one position keeps only that position of its
  component; the component's other positions are replaced with the rest;
* a `complement` set with no sites replaces the whole universe: **everything**.

The universe can also be made of finer parts: every head's query, key or value, and every MLP
neuron, so a circuit can keep, say, a few heads' values and a few neurons. A universe can't hold a
component and its parts together (a head's output and its values, an MLP's output and its
neurons), since one is made from the other.

The replacement is the spec's experiment: the other prompt's activation
([patching](activation-patching.md)), donors (resample [ablation](ablation.md)), the mean over
prompts of the same length at each position (mean ablation), or zeros. With
[pairs of different lengths](activation-patching.md#pairs-of-different-lengths), sets are patched
at labels or the last token, or zero-ablated: sets that keep their sites and patch the rest, and
mean or resample ablation of sets, need pairs of equal length.

## Faithfulness, completeness, minimality

When the scope has an everything set, every set is also read against it, from the same bootstrap
resamples, so each comes with an interval:

* **Faithfulness**, for a set that keeps sites: the part of the behavior those sites carry alone,
  1 − (the set's change ÷ everything's change). It is 1 when keeping only them loses nothing, and
  0 when it does no better than replacing everything.
* **Share**, for a set that removes sites: its change as a share of everything's. A share near 1
  says removing the sites does what removing everything does: the circuit is complete.
* **Minimality**, one site at a time: for a keeping set one site short of another, the
  faithfulness that site adds.
* **Interaction**: for a set of two sites that are also run alone, the effect of both beyond the
  sum of the two. Backup components show up here, as two sites that break less together than
  apart.

With GPT-2 small, keeping the 26 heads of Wang et al.'s IOI circuit (at every position) and
mean-ablating every other head over prompts with three unrelated names keeps all of the behavior
(faithfulness 1.02); taking head 9.9 out of it costs 0.10 [0.06, 0.14], and ablating 9.9 and 9.6
together breaks less than the two do alone, −0.14 [−0.18, −0.11], as the backup name movers the
paper describes would.

## In the app

* Stage components on the model map, then **Test as a circuit** in the staging tray: the form
  gets the sites kept alone with the rest of the model replaced, the sites removed, and, if you
  like, each site left out in turn.
* With two sites staged, **Check how they interact** runs each alone and both together.
* On a finished run, **Test the top sites as a circuit** keeps its strongest 1, 2, 4, 8 … sites
  alone, for a curve of faithfulness against the number of sites kept.

The results of a run over sets show each set's faithfulness or share, what each site adds and how
sites interact, beside a chart of faithfulness against the number of sites kept.

## When to use it

Use sets once single sites have suggested a circuit: to check that the circuit is enough
(faithfulness near 1), that nothing outside it matters (removing it does what removing everything
does), and that each part earns its place. Read the replacement as part of the claim: a circuit
can be faithful under mean ablation and not under resampling. Conmy et al. (2023) and Hanna,
Pezzelle and Belinkov (2024) discuss how the choice of replacement and metric changes what a
circuit looks like.

## In a spec

```json
"scope": {
  "kind": "site_sets",
  "universe": ["head"],
  "sets": [
    {"label": "everything", "sites": [], "complement": true},
    {"label": "circuit", "complement": true, "sites": [
      {"kind": "head", "layer": 9, "head": 9, "position": {"kind": "all"}},
      {"kind": "head", "layer": 10, "head": 0, "position": {"kind": "all"}}
    ]},
    {"label": "circuit removed", "complement": false, "sites": [
      {"kind": "head", "layer": 9, "head": 9, "position": {"kind": "all"}},
      {"kind": "head", "layer": 10, "head": 0, "position": {"kind": "all"}}
    ]}
  ]
}
```

See [the spec reference](../reference/spec.md#scope-site_sets).

## References

* Wang, Variengien, Conmy, Shlegeris and Steinhardt (2022). *Interpretability in the Wild: a
  Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023.
* Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris and
  Thomas (2022). *Causal Scrubbing: a method for rigorously testing interpretability hypotheses.*
  Alignment Forum.
* Conmy, Mavor-Parker, Lynch, Heimersheim and Garriga-Alonso (2023). *Towards Automated Circuit
  Discovery for Mechanistic Interpretability.* NeurIPS.
* Hanna, Pezzelle and Belinkov (2024). *Have Faith in Faithfulness: Going Beyond Circuit Overlap
  When Finding Model Mechanisms.* COLM.
