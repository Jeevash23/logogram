# Steering

Patching asks where a behavior lives. Steering asks whether a direction in the residual stream
*causes* it: add the direction while the model runs, at several strengths, and measure how far
the answer moves. Logogram steers along the mean difference between the prompts of each pair
(Turner et al., 2023; Rimsky et al., 2024), or along one SAE feature's decoder direction
(Templeton et al., 2024).

## What Logogram computes

**The direction.** At each steered site (one residual stream site per layer, or the sites the
scope names, at one token of each prompt), the direction is the mean difference between the two
prompts of each pair, from the prompts being steered (`apply_to`) toward the other ones. It is
computed on a seeded training share of the pairs (`train_fraction`, `seed`) and measured only on
the held-out rest, so no prompt receives its own difference.

**The strengths.** Each coefficient multiplies the whole mean difference: 1 adds all of it, 2
twice as much, and negative values push the other way.

**The effect** is normalized like patching: 1 means the steered prompts moved as far as switching
to the other prompt, 0 that nothing changed.

**The control.** With `control`, a random direction of the same length runs alongside at the
same strengths. The results show a layer × strength map with the control's columns muted, each
strength compared with its control prompt by prompt, and, for a selected site, its effect at every
strength with intervals. When no site and strength does more than the control, the results say
so.

A mean difference steers only when the pairs differ the same way. In IOI prompts that mix the
two name orders (ABBA and BABA), the difference at the last token flips with the order, so the
mean cancels and steering does no more than the control. Generate prompts of one order to steer.
**Check robustness** offers another split of the pairs, or the other direction.

## Steering along an SAE feature

With `"direction": {"kind": "sae_feature", "feature": 1234}` and the spec's `sae`, the direction
is that feature's decoder direction, W_dec[f], added where the SAE's output is read into the
model. The scope names that site alone, at one token of each prompt:

* an SAE on a residual stream site adds it to that site of its layer;
* an SAE on an attention output adds it to that layer's attention output, which is the same as
  adding it to the residual stream between attention and the MLP;
* an SAE on an MLP output, and a transcoder (which writes its MLP's output), add it to that MLP's
  output, the same as adding it to the residual stream after the layer;
* an SAE on attention heads' outputs adds it to every head's output `z`, before the output
  projection carries it into the attention output (the scope names that attention output).

**Strength 1 is the feature's typical activation**: the mean of its positive activations at the
steered token of the training pairs' clean and corrupt prompts, since W_dec[f] times *a* is what
the feature writes when its activation is *a*. A run records it, with how often the feature fired
there, and refuses a feature that never fires there. For SAEs that standardize their inputs, each
held-out prompt's vector is also multiplied by the standard deviation of its own activation at
that token. The control is a random direction with the length of the steering vector at
strength 1.

In the app, choose the direction in the experiment form, or press **Steer along this feature** in
a feature's panel in **Explore → Features**.

## What the steered model says

With `continuations` (`{"prompts": 3, "tokens": 20}`), a steering run also continues the first
held-out prompts, in the split's order (up to 8 prompts, up to 64 tokens): without steering, at
each strength, and along the control. Decoding is greedy, the most likely token each time with
ties going to the lowest token id, so the same spec gives the same text. Each forward pass adds
the vector at the steered site and token of the prompt, as when it was measured; the tokens the
model writes aren't steered themselves. The continuations are saved as `continuations.json`,
shown under the results map (each prompt, then one line per strength, the control's muted) and
included in the HTML report.

**Your text → Continue the text** does the same for any text, outside a run: it adds a feature
at a strength you give, in the feature's own units, at every position from a chosen token on,
the new tokens included, and shows the feature's largest activation on the text for scale.

## When to use it

Use steering to test whether a direction you found (a difference between prompts, or a feature)
is enough to move the behavior, and continuations to see what it does to the text beyond the
answer. A result is only as strong as its margin over the random control.

## In a spec

```json
"experiment": {
  "kind": "steering",
  "apply_to": "corrupt",
  "direction": {"kind": "mean_difference"},
  "coefficients": [-1, 1, 2],
  "train_fraction": 0.5,
  "seed": 0,
  "control": true,
  "continuations": null
},
"scope": {"kind": "layer_components", "components": ["resid_pre"], "position": {"kind": "last"}}
```

See [the spec reference](../reference/spec.md#experiment-steering).

## References

* Turner, Thiergart, Udell, Leech, Mini and MacDiarmid (2023). *Activation Addition: Steering
  Language Models Without Optimization.* arXiv.
* Rimsky, Gabrieli, Schulz, Tong, Hubinger and Turner (2024). *Steering Llama 2 via Contrastive
  Activation Addition.* ACL.
* Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from
  Claude 3 Sonnet.* Transformer Circuits Thread.
