# Queue runs and run them headless

Runs can wait their turn. The queue belongs to the project, so the app and the command line add to
the same one, and it survives restarting Logogram: queue a night's experiments on a remote GPU,
start them, and close your laptop.

## In the app

While a job runs, **Run** in the experiment form reads **Queue**: the experiment waits in line and
starts when the jobs ahead of it end, whether they finish, fail or are cancelled. Queued runs are
listed under **Queued** in **History**, and a queued run can be removed until it starts. Sweeps,
robustness checks and verifications queue the same way.

The queue is kept in the project, in `.logogram/queue.json`, which git ignores. Restarting
`logogram` or closing the project keeps it, and it comes back **paused**: nothing starts until you
press **Start the queue**, or clear it. **Pause the queue** stops it starting more runs; the run in
progress goes on.

## From the command line

```bash
logogram queue add experiments/ioi-head-patching/spec.json   # add a spec to the end of the queue
logogram queue list                                           # what waits, and who runs the queue
logogram queue run                                            # run it to the end, one run at a time
logogram queue clear                                          # take every run out, after asking
```

`logogram queue add` checks what it can before queueing: a spec that can't be read, or whose
dataset is missing, isn't queued. Commands use the project in the current folder or above it, or
`--project PATH`.

`logogram queue run` works through the queue without the app: each run's progress, then how it
ended. It loads each model once and reuses it for the runs that ask for it.

* A run that fails doesn't stop the queue: the next one starts. A run that can't start, because
  its dataset is gone, leaves the queue with the reason.
* Runs added meanwhile, from the command line or in the app, run too.
* Ctrl+C stops the run in progress, which [keeps what finished](../concepts/runs-and-projects.md#a-run-that-stops-early),
  and leaves the rest queued.

One process at a time runs a project's queue. While `logogram queue run` runs it, the app shows the
queue as running in another process and won't start it too; while the app runs it, `logogram
queue run` says so and stops. A process that ended, even by crashing, never leaves the queue
taken.

## Overnight on a remote GPU

On the GPU machine (see [use a remote GPU over SSH](remote-gpu.md)), queue the specs, then run the
queue inside `tmux` or `screen`, so that it keeps going when the SSH connection drops:

```bash
tmux new -s runs
logogram queue run --project ~/Documents/Logogram/my-project
```

Detach with Ctrl+B then D. In the morning, `logogram list` shows how each run ended, and copying
the project folder back brings the runs home. Specs made in the app on your own computer can be
queued there too: in **Experiment → Spec**, **Save without running** writes the form's spec into
`experiments/`; copy the project over, and `logogram queue add` each spec.
