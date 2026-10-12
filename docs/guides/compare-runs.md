# Compare two runs

Two runs over the same sites, such as the same sweep on two datasets, in two directions, on two
checkpoints of a model, or an estimate and its verification, are compared site by site, and,
where they measured the same prompts, prompt by prompt.

## In the app

Open **Evidence → Compare runs** and choose **Run A** and **Run B**, or tick two runs in
**History** and press **Compare**. A robustness check and a verification open their comparison
by themselves.

* **Side by side** draws both runs' maps on one shared scale. **Difference (B − A)** draws one map
  of the differences, for runs with the same layout.
* **Top components in either run** lists the strongest sites of both, with each run's effect, its
  rank, and its flags.
* Choose a cell, or a row of the table, to see that site in both runs: each run's effect and
  interval, the difference with its interval, its flags, and a way into each run at that site.
* Above the maps, Logogram lists every choice that differs between the two specs, and says when
  the runs ran on different devices or library versions, which can move results in the last
  digits.

## What the numbers mean

* **Rank correlation** (Spearman) of the two runs' mean effects over the sites both measured.
* **Overlap of the top sites**: how many of each run's strongest sites are among the other's.
* **Flags**, on the map and in the table: `sign` where the sign reverses with both intervals
  excluding zero; `left_top` and `entered_top` where a site leaves or joins the strongest; and
  `differs` where the paired difference excludes zero.
* **The paired difference**: when both runs measured the same prompts, each prompt's effect in B
  minus its effect in A, averaged, with a bootstrap interval from the same resamples. It is much
  narrower than the two runs' intervals side by side would suggest, since the variation between
  prompts cancels; two runs whose intervals overlap can still differ clearly. Runs on different
  prompts (another dataset, another seed of a generator) can't be paired, and are compared by
  their effects and intervals only. See [statistics](../concepts/statistics.md#comparing-runs).

A run that stopped early is compared on what it kept, and says so.

## From the command line and Python

```bash
logogram compare RUN_A RUN_B   # rank correlation, top sites, changed conclusions, paired differences
logogram diff A B              # every choice that differs between two runs or spec files
```

```python
comparison = lg.compare_runs(run_a, run_b)
comparison["spearman"], comparison["top_overlap"], comparison["changes"]
```

Each entry of `changes` holds a site's effect in both runs, its ranks, its flags and its paired
difference.
