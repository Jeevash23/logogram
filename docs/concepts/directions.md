# Directions

Much of what interpretability finds is a direction: a vector in a site's activation space along
which something varies, such as which name the model will say, the truth of a statement, or a
refusal. Logogram keeps directions in the project, so that one method can find one and others can
test it: find it with a probe, a learned subspace, a principal component or a mean difference;
then patch along it, remove it, steer along it, or read it out while patching elsewhere.

## A direction in the project

A direction is two files in the project's `directions/` folder:

* `<name>.safetensors` holds one tensor, `directions`, `[k, width]`: one row for a direction, k
  rows for a subspace of k dimensions. Logogram saves the rows orthonormal (each of length 1, at
  right angles to the others), so the component of an activation `x` within them is
  `x · Dᵀ · D`.
* `<name>.json` says what it is: the model it was found on (its id and revision, or a local
  model's files by their hash), the site and layer, how it was made (by a probe, a learned
  subspace, a principal component, a mean difference, an import), the run it came from, the hash
  of the tensor file, and the lengths of the vectors before they were made unit length.

A run that uses a direction pins it by the hash of its tensor file and keeps a copy in
`directions/snapshots/`, as runs keep their prompts: saving another direction under the same name
later never changes a run, and a run's spec still finds the direction it used. The run's manifest
lists the directions it used, and its methods paragraph names them.

A direction found on one model is refused on a model of another width, and a run says so when it
was found on another model or revision of the same width: it may not mean the same thing there.

## Finding one

* **A probe** ([Linear probes](probes.md)): save a site's probe as a direction from the results.
  With two classes it points from the first class toward the second.
* **A learned subspace** ([Distributed alignment search](das.md)): save a site's learned subspace,
  or one of its vectors.
* **A principal component**, or **the mean difference between two classes** of prompts or texts,
  from **Explore → Representations**.

## Using one

**Subspace patching** (`"kind": "subspace_patching"`) patches only the component of each site's
activation along the direction from the other prompt of the pair, and keeps everything at right
angles to it: receiver + ((source − receiver) · Dᵀ) · D. Beside patching the whole site (**Check
robustness → Patch the whole site**), it says how much of the site's effect the direction carries.
With every dimension of the space, it is activation patching exactly.

**Directional ablation** (an ablation with `"baseline": {"kind": "projection", ...}`) removes the
component along the direction, x − (x · Dᵀ) · D, or with a `reference` sets it to its mean over the
clean or corrupt prompts. Removing a direction the model uses should hurt the behavior as much as
ablating the whole site does; removing one it doesn't use, not at all.

**Steering along a saved direction** (`"direction": {"kind": "direction", ...}` in a steering
spec) adds the direction's unit vector at the steered sites. Strength 1 moves the activation along
the direction as far as the mean difference between the two prompts of the training pairs does
along it; the run reports that distance. A direction the pairs don't differ along steers nothing,
and the run says so.

**A readout** (`"metric": {"kind": "readout", ...}`) reads an activation's component along the
direction (its dot product with the unit vector) at a site, layer and position you choose, instead
of the model's output. Patching and ablation then ask what drives the direction: which earlier
heads and MLPs write it. See [Metrics](metrics.md#readouts).

Directions are as wide as the residual stream, so they apply to residual stream sites and to
attention and MLP outputs, not to a single head's output, query, key or value, or to neurons.

## Where it can mislead

* A direction found on one set of prompts can mean something else on others. Test it on prompts
  that didn't find it: a probe's and a learned subspace's held-out pairs are a start.
* Patching or removing a direction can change a behavior without the model using that direction
  as a variable, by pushing the activation somewhere it never goes (an interpretability illusion,
  Makelov et al., 2023). Compare with the whole site, and with a random direction.
* Scaling and directional ablation are interventions on the receiver run's own activations, so
  between prompts of different lengths they need no matching (removing a direction to its mean
  does: the mean is taken at the same position of other prompts).

## In a spec

```json
"experiment": {"kind": "subspace_patching", "direction": "clean_to_corrupt",
               "subspace": {"name": "toward-io", "sha256": null}}
```

```json
"experiment": {"kind": "ablation",
               "baseline": {"kind": "projection", "direction": {"name": "toward-io", "sha256": null},
                            "reference": null}}
```

A direction's `sha256` may be null in a spec you write: the run pins it, and its saved spec has the
hash. See [the spec reference](../reference/spec.md).

## References

* Geiger, Wu, Potts, Icard and Goodman (2023). *Finding Alignments Between Interpretable Causal
  Variables and Distributed Neural Representations.* CLeaR.
* Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee and Nanda (2024). *Refusal in Language Models
  Is Mediated by a Single Direction.* NeurIPS.
* Makelov, Lange and Nanda (2023). *Is This the Subspace You Are Looking for? An Interpretability
  Illusion for Subspace Activation Patching.* ICLR 2024.
* Marks and Tegmark (2023). *The Geometry of Truth: Emergent Linear Structure in Large Language
  Model Representations of True/False Datasets.* arXiv.
