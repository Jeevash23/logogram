# Use a remote GPU over SSH

Logogram can run on a machine you reach over SSH, such as a rented GPU, while you use it in the
browser on your own computer. Everything still runs on the remote machine, which still listens on
127.0.0.1 only: SSH carries the page to you.

The README's [Running on a remote GPU](../../README.md#running-on-a-remote-gpu) is the guide:
installing on the remote machine, starting Logogram there without a browser inside `tmux`,
forwarding the port, opening the printed address, forwarding another local port with
`--forwarded-port`, where projects, models and settings live, and copying runs back.

Three things go with it:

* **Choosing the GPU.** On a machine with several GPUs, the model dialog offers each one by name
  and free memory, and a spec names one as `cuda:N` in `model.device` (`cuda` is the default GPU,
  normally GPU 0). `logogram doctor` lists the GPUs with their numbers, and each run's manifest
  records the GPU it ran on. See [models](../models.md#devices-and-gpus).
* **Running overnight.** Queue specs with `logogram queue add` and run them in `tmux` with
  `logogram queue run`, which keeps going when the connection drops. See
  [queue runs and run them headless](queue.md).
* **Comparing with runs from home.** The manifests record the remote device, and a rerun on
  another device can differ in the last digits: compare per-prompt values before treating two
  runs as the same (see [runs and projects](../concepts/runs-and-projects.md#reproducibility)).
