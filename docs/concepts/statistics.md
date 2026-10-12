# Statistics

A mean effect over a few dozen prompts is an estimate, and a sweep makes hundreds of them at
once. Logogram keeps every prompt's value and computes, for every site, how much the estimate
would move with other prompts like these, and how many sites would look strong by chance.

## Per site

For each site, from its per-prompt [effects](metrics.md):

* **n**, the **mean** and the **standard deviation**;
* a **percentile bootstrap interval** at the spec's confidence level (`statistics.ci`, such as
  0.95): the prompts are resampled with replacement `statistics.bootstrap` times, the mean is
  taken in each resample, and the interval runs between the matching percentiles of those means
  (Efron, 1979). The resamples are seeded (`statistics.seed`) and drawn from the raw bit stream
  of NumPy's PCG64 generator, which doesn't change between NumPy versions, and every site of a
  run uses the same resamples, so differences between sites aren't blurred by different draws;
* **sign flips**: the prompts whose preference for the answer (log P(answer) − log P(distractor))
  changed sign under the intervention;
* **opposite sign**: the prompts whose effect has the opposite sign to the mean, rounding noise
  aside. A large mean with many prompts going the other way describes a mixture, not a rule.

The interval describes the prompts: it says how far the mean could move with another sample of
prompts from the same source. It says nothing about another seed of a resample ablation, another
dataset made by the same generator, or another checkpoint of the model. A
[sweep](../guides/sweeps.md) measures those.

## Many sites at once

A head sweep of GPT-2 small tests 144 sites. At 95%, about 7 of them would have intervals that
miss zero even if no head mattered. So every sweep also reports, for each site:

* a **simultaneous band**, which holds for all the sites of the run together at the chosen
  confidence: in that share of the bootstrap resamples, every site lies inside its band. Each
  resample's most extreme site sets how far into the tails the bands reach (the max-rank method
  in Davison and Hinkley, 1997). A band contains the site's ordinary interval;
* a **q-value** (Benjamini and Hochberg, 1995): the smallest false discovery rate at which the
  site would count as a finding, from bootstrap p-values. A two-sided p-value is twice the
  smaller share of resamples on either side of zero, so p < α exactly when the interval at level
  1 − α excludes zero.

A site whose band excludes zero, or whose q-value is small, is a finding that survives the number
of sites tested. The inspector shows both beside the interval.

## Clusters of prompts

Prompts made from one template aren't independent: they share most of their words, and so their
errors. Resampling them one by one makes intervals too narrow. `statistics.cluster` names a field
of the prompts' `meta`, such as `template`, and the bootstrap then resamples whole clusters of
prompts that share its value (Field and Welsh, 2007), which widens the intervals honestly. With
fewer than 10 clusters, the run warns that its intervals are rough: there are too few clusters to
resample. The task generators record each prompt's template in `meta`.

## Comparing runs

Two runs that measured the same prompts are compared **prompt by prompt**: for each site, the
difference between the two runs' effects on each prompt, its mean, and an interval from the same
resamples. That interval is much narrower than the two runs' intervals side by side would
suggest, because the variation between prompts cancels. Steering compares each strength with
its random control the same way. See [compare two runs](../guides/compare-runs.md).

## The winner's curse

Choosing the strongest sites and then reporting their effects on the same prompts biases those
effects away from zero: among many noisy estimates, the largest are partly the luckiest. This
matters most when choosing among thousands, such as an SAE's features. A features sweep can
choose its features on a seeded share of the prompts and report them only on the rest
(`scope.choose_on` and `scope.seed`, see [SAE features](sae-features.md)). A sweep across seeds
or datasets chooses its sites on the first value by default, so that the other values show them
without that bias (see [sweeps](../guides/sweeps.md)).

## Per-prompt values

Every run writes `results.parquet`, one row per site and prompt, with everything the statistics
came from: the receiver, patched and reference values of the metric, the effect, the change, the
answer probability and the preference. `run.per_prompt()` in Python, **Export → Export
per-prompt CSV** in the app and `logogram export RUN` read it; anything Logogram computes can be
recomputed from it.

## References

* Efron (1979). *Bootstrap Methods: Another Look at the Jackknife.* The Annals of Statistics.
* Efron and Tibshirani (1993). *An Introduction to the Bootstrap.* Chapman and Hall.
* Benjamini and Hochberg (1995). *Controlling the False Discovery Rate: A Practical and Powerful
  Approach to Multiple Testing.* Journal of the Royal Statistical Society, Series B.
* Davison and Hinkley (1997). *Bootstrap Methods and their Application.* Cambridge University
  Press.
* Field and Welsh (2007). *Bootstrapping clustered data.* Journal of the Royal Statistical
  Society, Series B.
