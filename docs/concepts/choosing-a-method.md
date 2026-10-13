# Choosing a method

Interpretability questions come in a few kinds, and each kind has methods built for it. The
experiment form groups Logogram's methods by the question they answer; this page says what each
one measures, what it costs, and what it can't tell you, so you can choose before you run.

Every method runs from the same [spec](../reference/spec.md), in the app, with `logogram run` and
from Python, and reports per-prompt values with bootstrap intervals. None of them is a verdict on
its own: a finding that holds under another method, another baseline and other prompts is one
you can trust ([Check a finding's robustness](../guides/check-robustness.md)).

![The experiment form's methods, grouped by the question each answers: where is it computed, how
does it flow, what does a site represent, and what does adding a direction do](../images/methods.png)

## Where is it computed?

Which components, positions and neurons the behavior depends on.

| Method | What it measures | Cost | Watch for |
|---|---|---|---|
| [Activation patching](activation-patching.md) | How far copying one activation from the other prompt moves the output | One forward pass per site and batch | Denoising and noising answer different questions: choose the direction on purpose |
| [Ablation](ablation.md) | What the output loses when an activation is replaced by a baseline | One forward pass per site and batch | The baseline changes the answer: zero ablation pushes activations off their range |
| [Attribution patching](attribution-patching.md) | A first-order estimate of patching at every site, from gradients | Two forward and one backward pass per batch, for every site at once | Estimates can be far off where the model is nonlinear: verify the strongest by patching |
| [Direct logit attribution](direct-logit-attribution.md) | How much each component writes straight into the answer's logit | One forward pass | Leaves out everything a component does through later layers |

## How does it flow?

Which paths carry the information from one component to another, and the circuit they form.

| Method | What it measures | Cost | Watch for |
|---|---|---|---|
| [Path patching](path-patching.md) | A component's effect through chosen receivers only | One forward pass per sender and batch | Holding the other heads is part of the question |
| [Edge attribution](edge-attribution.md) | Every edge of the model's graph estimated from gradients (EAP, EAP-IG), the strongest verified as a circuit | A few passes for all edges, then a curve of patched circuits | Read the faithfulness curve against random circuits of the same size |
| [ACDC](acdc.md) | A circuit found by removing edges one at a time while the output stays close | One forward pass per edge tried | The threshold decides the circuit's size: compare a few |

## What does a site represent?

What can be read from an activation, along which direction, and whether the model uses it.

| Method | What it measures | Cost | Watch for |
|---|---|---|---|
| [Linear probes](probes.md) | Held-out accuracy of a classifier reading a label, beside a control on shuffled labels | One forward pass, then a small fit per site | Readable isn't used: test the probe's direction by patching or removing it |
| [Distributed alignment search](das.md) | The few directions whose patching carries a site's effect, learned on training pairs | Gradient steps per site | A learned subspace can steer from outside what the model uses: compare it with the whole site and a random one |
| [Subspace patching](directions.md) | How much of a site's effect a saved direction carries | One forward pass per site and batch | Compare with patching the whole site |
| [Directional ablation](ablation.md) | What the output loses without a saved direction's component | One forward pass per site and batch | A direction found on some prompts can mean something else on others |
| [Readouts](metrics.md#readouts) | What drives a neuron, a direction or an SAE feature, instead of the output | As patching or ablation | Sites after the readout can't change it |
| [SAE features](sae-features.md) | Patching and attribution of a sparse autoencoder's features | As patching, or attribution, per feature | An SAE that fits badly describes something else: measure the fit first |
| [Corpora and representations](corpora-and-representations.md) | The texts that most activate a neuron, a direction or a feature; principal components of a site | One forward pass over a corpus | Top examples show what a unit responds to most, not everything it does |

## What does adding a direction do?

| Method | What it measures | Cost | Watch for |
|---|---|---|---|
| [Steering](steering.md) | How the output moves when a direction (a mean difference, an SAE feature or a saved direction) is added at several strengths, on held-out prompts, against a random direction | One forward pass per strength and batch | Large strengths break the model in ways a small one doesn't |

## Seeing what a model does

Before choosing a method, look: **Explore → Your text** shows any text's predictions, the logit
lens (each layer's prediction, through a [tuned lens](tuned-lens.md) when you
have trained one), attention and SAE features; **Attention** finds previous-token, duplicate-token
and induction heads; **Corpus** and **Representations** read activations over many texts. These
are explorations, not experiments: they suggest what to test.

## A path through a study

1. Find a behavior and the prompts that show it: a dataset of clean and corrupt pairs, generated
   from a task or written yourself, with a baseline that shows the gap.
2. Locate it: attribution patching over every site, then patching to verify the strongest.
3. Trace it: path patching or edge attribution between the components you found, or ACDC for a
   circuit, checked for faithfulness.
4. Understand the parts: probes, directions and features at the sites that matter, tested by
   patching, removing and steering along them.
5. Check it holds: other seeds, prompts, baselines and model revisions, and a comparison of runs.
