# Work with an SAE

This guide loads a published sparse autoencoder for GPT-2 small, checks that it fits, reads its
features on the IOI prompts, finds the ones that matter for the answer, patches one, and steers
along one. [SAE features and transcoders](../concepts/sae-features.md) explains what each step
computes.

## 1. Load it

With GPT-2 small loaded and the IOI example open, go to **Explore → Features** and load an SAE.
Joseph Bloom's residual stream SAEs for GPT-2 small are a good start: repository
`jbloom/GPT2-Small-SAEs-Reformatted`, folder `blocks.8.hook_resid_pre`, the residual stream
entering layer 8, with 24,576 features. It downloads once (about 150 MB), from its safetensors
file only.

An SAE fits activations like those it was trained on. These were trained on TransformerLens's
processed weights, which is how Logogram loads models unless told otherwise; for SAEs trained on a
model's own weights, such as Gemma Scope 2's, load the model with **Process weights** off. The fit
in the next step shows which.

## 2. Measure the fit

Press **Measure the fit**. On your prompts, Logogram reports the fraction of the activations'
variance the SAE explains, how many features fire per token, and the logit difference with the
SAE's reconstruction spliced in. A reconstruction that loses much of the logit difference, or
explains little variance, means its features describe something other than what the model does
here: stop and find out why before reading them.

## 3. Read the features

The view lists the features that fire most on each token of the current prompt. Choose one to see
its activation along the prompt, the prompts where it fires most, and what it writes: the tokens
it promotes and suppresses directly, read through the final normalization and the unembedding.
**Your text** does the same for any text you type.

## 4. Find the features that matter

Firing strongly isn't mattering. In **Configure**, choose attribution patching, and under what to
sweep, **every SAE feature** at the last token. Keep the 50 strongest, and choose them on half of
the prompts, reported on the other half, to avoid the winner's curse:

```json
"experiment": {"kind": "attribution_patching", "direction": "clean_to_corrupt",
               "method": "gradient", "steps": null},
"scope": {"kind": "features", "position": {"kind": "last"}, "top": 50,
          "choose_on": 0.5, "seed": 0},
"sae": {"repo": "jbloom/GPT2-Small-SAEs-Reformatted", "path": "blocks.8.hook_resid_pre",
        "revision": null}
```

The results show the strongest features and how much of the site's estimated effect they account
for; the rest is the SAE's error.

## 5. Patch them

**Verify top 10 by patching** patches the ten strongest features for real: each feature of the
corrupt prompt's activation is set to its value on the clean prompt, and the SAE's error is kept.
To patch one feature yourself, name it as a site:

```json
"experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"},
"scope": {"kind": "sites", "sites": [
  {"kind": "sae_feature", "layer": 8, "feature": 1234, "position": {"kind": "last"}}
]}
```

## 6. Steer along one

In a feature's panel, **Steer along this feature** puts steering along its decoder direction in
the form, at the site the SAE writes (here the residual stream entering layer 8), at one token.
Strength 1 is the feature's typical activation; choose the strengths, the split and the random
control, and ask for continuations to see what the steered model writes (see
[steering](../concepts/steering.md)). **Your text → Continue the text** tries a feature on any text
at once, without a run.

## From Python

The [SAE notebook](../../notebooks/sae-features.ipynb) does steps 1 and 3 to 5 with
`lg.load_sae`, `lg.look(model, text, sae=sae)`, `lg.feature_writes` and `lg.run`.

## References

* Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary
  Learning.* Transformer Circuits Thread.
* Marks, Rager, Michaud, Belinkov, Bau and Mueller (2024). *Sparse Feature Circuits: Discovering
  and Editing Interpretable Causal Graphs in Language Models.* ICLR 2025.
