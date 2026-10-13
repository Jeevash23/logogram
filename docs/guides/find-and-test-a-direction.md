# Find a direction and test it

A probe can tell you that a site carries something, such as the gender of a sentence's subject.
It can't tell you that the model uses it there. This guide probes GPT-2 small for the subject's
gender at every layer, saves the probe's direction, and asks where the model uses it: by patching
along the direction, by removing it, and against a subspace learned for the purpose.
[Linear probes](../concepts/probes.md), [Directions](../concepts/directions.md) and
[Distributed alignment search](../concepts/das.md) explain what each step computes.

The numbers below are GPT-2 small's (revision `607a30d`), on 240 pairs of the gendered pronoun
task generated with seed 0.

## 1. Prompts whose answer is a gender

In **Experiment → Prompts**, choose **Generate a task**, then **Gendered pronoun**, 240 pairs,
seed 0. Each pair is one sentence about two people of different genders, *So Jason was late for
work again, wasn't* and *So Michelle was late for work again, wasn't*, answered with *he* and
*she*. Each prompt's `meta` records the clean prompt's `gender` and its `template`, and its
named positions mark the `name` and the `end`, the last token, where the pronoun is predicted.

## 2. Probe every layer

In **Configure**, choose **Linear probe**. Under **The probe reads**, choose **A field of the
prompts' meta**, the field `gender`, and **The clean prompts**: the corrupt prompt names someone of
the other gender, so the pair's label fits only its clean prompt. Under **What to sweep**, choose
the residual stream after each layer at the named position `end`, and run:

```json
"experiment": {"kind": "probe",
               "labels": {"kind": "meta", "field": "gender", "prompts": "clean"},
               "train_fraction": 0.5, "seed": 0, "regularization": 0.01,
               "iterations": 100, "control": true},
"scope": {"kind": "layer_components", "components": ["resid_post"],
          "position": {"kind": "label", "label": "end"}},
"metric": null
```

The probe tells the genders apart on every held-out prompt at every layer, from layer 0 on,
while the probe trained on shuffled labels stays near chance (0.46 to 0.56): a selectivity of
about 0.5 everywhere. At the name itself the result is the same. The gender is linearly readable
at the last token from the first layer: layer 0's attention already mixes the name into it.

## 3. Ask where the model uses it

Readable is not used. Patch the same sites instead: **Activation patching**, clean → corrupt,
the residual stream after each layer at `end`, with the logit difference between *he* and *she*.
Patching the whole last-token residual stream moves the answer by at most 0.05 of the gap through
layer 8, then by 0.31 at layer 9, 0.81 at layer 10 and 1.00 at layer 11. The model brings the
gender to the last token late, through the heads that read it from the verb after the name (see
the gendered pronoun example's notes), though a probe reads it there all along.

## 4. Save the direction

In the probe's results, choose **Save as direction** beside `L10 resid post @ end` and name it
`gender-L10`; save `L11` too, and `L4` and `L8` to compare. A probe of two classes saves the
direction from the first class toward the second: here from *female* toward *male*. The same from
the command line is `logogram directions save RUN SITE gender-L10`, and from Python
`run.save_direction(site, "gender-L10")`. The direction is saved in the project's `directions/`
folder, with the model, site and run it came from; a run that uses it pins it by its hash.

## 5. Patch along it

**Subspace patching** copies only the component along the direction from the other prompt, and
keeps everything at right angles to it:

```json
"experiment": {"kind": "subspace_patching", "direction": "clean_to_corrupt",
               "subspace": {"name": "gender-L10", "sha256": null}},
"scope": {"kind": "sites", "sites": [{"kind": "resid_post", "layer": 10,
          "position": {"kind": "label", "label": "end"}}]}
```

| Layer | Whole site | Along the probe's direction |
|---|---|---|
| 4 | 0.01 | 0.003 |
| 8 | 0.05 | 0.011 |
| 10 | 0.81 | 0.38 [0.37, 0.39] |
| 11 | 1.00 | 0.55 [0.54, 0.55] |

Where the model uses the gender, the probe's one direction carries about half of what the whole
768-dimensional residual stream carries. Where it doesn't, the direction carries nothing, though
the probe reads the gender there perfectly.

## 6. Remove it

**Ablation** with the baseline **Remove a direction** runs the clean prompts with the component
along `gender-L10` removed. At layer 10 the answer moves 0.19 [0.17, 0.20] of the way to the
corrupt prompt's, against 0.38 for mean-ablating the whole site over the clean prompts; at layer
11, 0.27 against 0.50. The direction is part of what the model uses at these sites, not all of
it.

## What to take from it

* A probe finds where something can be read, which can be everywhere. Patching finds where the
  model uses it, which here is the last token's residual stream from layer 9 on.
* Testing the probe's direction where the model uses the gender shows that it carries part of
  the effect, and that the rest is elsewhere in the residual stream.
* Steering along this direction does little: the gendered pronoun pairs swap genders both ways,
  so the training pairs' mean difference along the direction nearly cancels, and the run says so.
  Patching along it takes each prompt's own pair's value instead.
* Every number above came with an interval and a per-prompt table; **Check robustness** reruns
  any of these with another seed, split or baseline.
