# Check a finding's robustness

A finding is stronger when it survives changes that shouldn't matter, and more interesting when it
changes under ones that should. **Check robustness** reruns a finished run with exactly one
methodological choice changed, then compares the two runs.

## In the app

On a finished run's results, press **Check robustness** and choose what to change. The choices
depend on the method:

| The run | What it can change |
|---|---|
| Activation patching | The opposite direction; ablation with each baseline instead |
| Ablation | Another baseline; for resample ablation, more donors or another donor seed |
| Attribution patching | Patch every site for real; integrated gradients instead of one gradient, or the other way |
| Direct logit attribution | The total effect by patching; the split of the other prompts |
| Path patching | Hold or release the MLPs; the whole effect by patching; the opposite direction |
| Steering | Another split of the pairs; the other direction |
| Any run in float16 or bfloat16 | The same run in float32 |

The rerun is an ordinary run that records which run it checks and what it changed. When it
finishes, the comparison opens, and the map flags the sites whose conclusion changed.

## Reading the comparison

* **Rank correlation** (Spearman) of the two runs' effects over the sites both measured: do the
  same sites come out strongest?
* **Overlap of the strongest sites**: how many of the top sites are top in both.
* **Changed conclusions**, flagged on the map: a site whose sign reversed with both intervals
  excluding zero, one that entered or left the top sites, and, for runs on the same prompts, one
  whose paired difference excludes zero.
* **Paired differences** prompt by prompt, with intervals, where the runs measured the same
  prompts (see [compare two runs](compare-runs.md)).

A high correlation with a few flagged sites usually means the finding holds and those sites are
where the two questions differ. Read the flagged sites: the other direction separates components
a behavior *needs* from ones that are *enough*; a different baseline separates a component's role
from the damage of pushing its activation off its usual range; float32 shows what rounding did.

## From the command line and Python

```bash
logogram robustness RUN --experiment '{"kind": "activation_patching", "direction": "corrupt_to_clean"}'
logogram robustness RUN --dtype float32
```

```python
other = lg.check_robustness(run, experiment={"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}})
lg.compare_runs(run, other)
```

All three derive the rerun's spec the same way, so they change exactly the same thing.

## Beyond one rerun

A run's intervals describe its prompts. Whether a finding holds for another seed of the
resampling, another sample of prompts or another checkpoint of the model is a question for a
[sweep](sweeps.md), which runs the spec once per value and shows which sites keep their sign and
size.
