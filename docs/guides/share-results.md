# Share a result

A result travels best with everything needed to check it. Logogram gives each finished run a
report anyone can open, figures, the per-prompt values as CSV, and a methods paragraph; and the
project itself is a folder of plain files.

## The report

**Export → Download the report (HTML)** on a finished run, `logogram report RUN` on the command
line, or `run.report("run.html")` in Python writes one HTML file that opens in any browser,
without Logogram:

* the run at a glance (for patching, the clean and corrupt baseline and the gap), and its map,
  drawn as SVG in the effect colors with its scale;
* its strongest sites with their intervals, and every site's numbers;
* for a steering run, what the steered model said;
* for an [edge attribution](../concepts/edge-attribution.md) run, its strongest edges drawn
  between the components instead of a map, its faithfulness curve, and its estimates beside their
  measurement by path patching once a verification has finished;
* the methods paragraph, the complete spec, and where the run came from: the model and its
  revision, the precision and device, the dataset and its hash, and the software versions.

The page runs no script and fetches nothing. It leaves out the name of the CPU or GPU the run used,
unless asked (`logogram report RUN --hardware`, or `run.report(path, hardware=True)`).

## Figures, values and methods

From **Export** on a finished run:

* **Export figure SVG** and **Export figure PNG**: the map as shown, with the value and scale
  chosen (the normalized effect or the change, fitted or ±1). A run over neurons exports its
  figure as SVG only; an edge attribution run has no map, and its report draws its edges instead;
* **Export per-site CSV**: one row per site, with its mean, interval, band, q-value and counts;
  for an edge attribution run, **Export every edge (CSV)** has one row per edge it estimated, not
  only the strongest the results list;
* **Export per-prompt CSV**, with or without each prompt's text: one row per site and prompt,
  everything the statistics came from. Text that a spreadsheet could read as a formula is
  prefixed with an apostrophe;
* **Copy methods** and **Download methods**: the methods paragraph, which says in sentences every
  choice that can change a number, with the software versions and the notes saved with the spec.

The same from the command line and Python:

```bash
logogram export RUN -o values.csv                 # one row per site and prompt
logogram export RUN --sites -o sites.csv          # one row per site
logogram export RUN --prompt-text -o values.csv   # with each prompt's text
logogram methods RUN
```

```python
run.sites()        # one row per site
run.per_prompt()   # one row per site and prompt
run.edges()        # an edge attribution run's every edge, one row each
run.methods()      # the methods paragraph
```

## The project

A project is a folder: commit it, zip it or copy it, and `logogram open` opens it on another
machine. Each run's folder holds its spec, results, summary and manifest, and the prompts it read
are in `datasets/`, with their exact bytes in `datasets/snapshots/`: share the whole project, or
both folders, so each run can be checked against its prompts. `logogram run` on a shared run's spec
reruns it and says whether every per-prompt value matches (see
[runs and projects](../concepts/runs-and-projects.md#reproducibility)).

A run's manifest records the device and GPU model it ran on, so results from different hardware
can be told apart; a shared project shares them too.
