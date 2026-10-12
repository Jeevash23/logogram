# Security

Logogram runs a web server on your computer that reads and writes project folders and loads
models, so its protections matter. Please report a problem with them privately.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: on the repository's **Security** tab, choose
**Report a vulnerability**, or go straight to
<https://github.com/Jeevash23/logogram/security/advisories/new>. Only the maintainers see the
report. Please don't open a public issue, pull request or discussion about it until a fix is
released.

Say what you did, what happened and what should have happened, with the Logogram version
(`logogram --version`) and your operating system. The problem report from **Help → Report a
problem** in the app, or from `logogram doctor --report`, has both, without names or paths.

The conversation, and the work on a fix, happen in the advisory. When the fix is released, the
advisory is published, crediting you unless you'd rather not be named.

## Supported versions

Fixes go into the newest release on PyPI. Update with `uv tool upgrade logogram` (or as
**Updating** in the README says for how you installed it).

## What counts

Anything that lets someone other than the person who started Logogram, or a web page they open,
do what that person can do with it, or that breaks one of these promises:

* **The session token.** Every HTTP and WebSocket request must carry it. The browser is opened
  with a single-use launch code instead, and the token never goes on a command line. A way to
  read, guess or skip the token, or to use a launch code twice, counts.
* **The Host, Origin and Fetch Metadata checks.** The server listens on 127.0.0.1 only. It
  answers only to the loopback names 127.0.0.1, localhost and [::1], on its own port and the
  forwarded ports it was started with (`--forwarded-port`); it accepts requests only from those
  origins, refuses cross-site requests and sends no CORS headers. A DNS rebinding attack, a
  cross-site request that gets through, or a response another site can read, counts.
* **Writes into project folders.** Projects come from other people. Logogram writes only inside
  the project, never through a symlink that leads out of it, and reads only regular files inside
  it. A project that makes Logogram read or write anywhere else counts.
* **Unpickling and loading files.** Weights and SAEs are loaded from safetensors files only, and
  nothing is ever unpickled. A model, SAE, spec, dataset or run file that makes Logogram run code
  counts.
* **What leaves the computer.** Logogram goes online only for Hugging Face downloads the person
  starts and, if they allow it, a version check that sends nothing but the version. Anything else
  it sends counts, as does a run file or problem report that gives away a person's paths, names
  or machine.

## What doesn't

* Someone who can already run programs as your user account. They can read the token in the
  terminal, and your files.
* A model's own behavior, or a wrong result with no security impact: report those as ordinary
  issues.
* A vulnerability in a dependency that Logogram doesn't expose: report it to that project.
  Dependabot and `pip-audit` watch Logogram's dependencies.
