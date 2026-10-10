#!/usr/bin/env python3
"""Run every Logogram method on a real model and check what must hold.

A release check, and the check for a compute backend Logogram hasn't been verified on (Apple's
MPS). It loads a model from Hugging Face (downloading it once into the usual cache), runs each
method on the bundled IOI example in a temporary project, and checks:

* exact identities: patching the residual stream at the only token that differs restores the
  whole effect, and so does patching the final residual stream; path patching from the last layer
  to the logits equals patching; direct effects add up to the logit difference; the same spec
  twice gives bit-identical results;
* agreement expected of a trained model: attribution patching ranks heads like patching, and
  verifying its strongest estimates reverses no sign;
* that every other method runs (ablations, steering, path patching, layer predictions, and SAE
  features with --sae).

With --compare-device it runs the head sweep again on a second device and reports how far the two
disagree. Nothing is written outside a temporary folder, apart from the Hugging Face cache.

With --golden (GPT-2 small only, at the revision the bundled example pins) it also checks the
bundled example against GPT-2 small's indirect-object circuit as Wang et al. found it
("Interpretability in the Wild", 2022): the clean logit difference is GPT-2 small's; patching the
name movers 9.9 and 10.0, the S-inhibition heads 7.3, 7.9, 8.6 and 8.10, or the induction heads
5.5 and 6.9 restores the answer, and patching the negative name movers 10.7 and 11.10 works
against it, on the normalized scale; and the name movers 9.9, 9.6 and 10.0 have the largest
direct effects on the answer, 10.7 and 11.10 the most negative. The bounds leave room for
differences between machines and library versions, and still catch a sign error, heads mapped to
the wrong layer or index, positions left out, or a broken normalization. --golden-only runs just
the runs these need (a few minutes on a CPU). The weekly Golden workflow
(.github/workflows/golden.yml) runs --golden.

    uv run python scripts/validate_real_weights.py
    uv run python scripts/validate_real_weights.py --golden
    uv run python scripts/validate_real_weights.py --device mps --compare-device cpu
    uv run python scripts/validate_real_weights.py --sae jbloom/GPT2-Small-SAEs-Reformatted \\
        blocks.8.hook_resid_pre

Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The tolerance of an identity in each dtype: float32 rounds in the last digits, 16-bit types in
# the second or third.
TOLERANCE = {"float32": 1e-3, "float16": 2e-2, "bfloat16": 5e-2}


@dataclass
class Ran:
    """A finished run: its summary, its folder, and the spec it ran."""

    summary: dict[str, Any]
    folder: Path
    spec: Any


@dataclass
class Check:
    name: str
    passed: bool | None  # None: skipped
    detail: str
    exact: bool  # holds for any model, not only a trained one


# --golden: GPT-2 small's indirect-object circuit as Wang et al. (2022) found it, as (layer, head).
GOLDEN_MODEL = "openai-community/gpt2"
NAME_MOVERS = ((9, 9), (9, 6), (10, 0))
NEGATIVE_NAME_MOVERS = ((10, 7), (11, 10))
S_INHIBITION = ((7, 3), (7, 9), (8, 6), (8, 10))
INDUCTION = ((5, 5), (6, 9))
# Bounds around what GPT-2 small gives on the bundled example in float32 (in the comments), far
# wider than rounding differences between machines, devices and library versions.
CLEAN_LOGIT_DIFF = (2.4, 3.4)  # 2.90; Wang et al. report 3.56 on their own prompts
PREFERS_ANSWER = 0.75  # the share of clean prompts that prefer the answer: 29 of 32
TOP = 10  # a rank among the 144 heads of the sweep
HELPS = 0.05  # 9.9 0.17, 10.0 0.11, S-inhibition 0.12 to 0.30, induction 0.10 and 0.32
HURTS = -0.1  # 10.7 -0.45, 11.10 -0.21
STRONGEST = (0.15, 1.0)  # 5.5, 0.32


def _head_effects(summary: dict[str, Any], position: str) -> dict[tuple[int, int], float] | None:
    """Each head's mean effect, if ``summary`` holds every head of GPT-2 small at ``position``."""
    effects = {
        (s["layer"], s["head"]): s["effect"]["mean"]
        for s in summary.get("sites", [])
        if s.get("kind") == "head"
        and s.get("position_key") == position
        and s["effect"]["mean"] is not None
    }
    every = {(layer, head) for layer in range(12) for head in range(12)}
    return effects if set(effects) == every else None


def golden_checks(sweep: dict[str, Any], direct: dict[str, Any] | None) -> list[Check]:
    """Check GPT-2 small on the bundled example against its published indirect-object circuit.

    ``sweep`` summarizes patching each head's output from the clean prompt into the corrupt one at
    every position, normalized by the dataset's gap (the example's own spec). ``direct`` holds each
    head's direct effect on the clean logit difference at the last position, the measure Wang et
    al. found the name movers by. Patching a head shows its total effect, which later heads partly
    undo: 9.6 has the second largest direct effect, yet patching it alone lowers the logit
    difference (-0.09), so it is checked by its direct effect only.

    The bounds leave room for differences between machines and library versions; a sign error,
    heads mapped to the wrong layer or index, positions left out, or a broken normalization fails
    them.
    """
    checks: list[Check] = []

    def check(name: str, passed: bool | None, detail: str) -> None:
        checks.append(Check(f"IOI circuit: {name}", passed, detail, exact=False))

    effects = _head_effects(sweep, "all")
    setup = (sweep.get("reference"), sweep.get("receiver"), sweep.get("measure"))
    if (
        effects is None
        or setup != ("clean", "corrupt", "intervention")
        or sweep.get("metric", {}).get("normalization") != "dataset_gap"
    ):
        n = sum(s.get("kind") == "head" for s in sweep.get("sites", []))
        check(
            "the run is a sweep of GPT-2 small's heads",
            False,
            "expected every head of GPT-2 small patched from clean into corrupt at every "
            f"position, normalized by the dataset's gap; got {n} heads",
        )
        return checks
    ranked = sorted(effects, key=lambda k: -effects[k])
    rank = {k: i for i, k in enumerate(ranked, start=1)}

    def shown(heads: tuple[tuple[int, int], ...]) -> str:
        return ", ".join(f"{a}.{b} {effects[a, b]:+.3f} (#{rank[a, b]})" for a, b in heads)

    clean, corrupt = sweep["baseline"]["clean"], sweep["baseline"]["corrupt"]
    clean_ld, corrupt_ld = clean["logit_diff"]["mean"], corrupt["logit_diff"]["mean"]
    n = sweep["n_prompts"]
    lo, hi = CLEAN_LOGIT_DIFF
    check(
        "GPT-2 small prefers the indirect object by its usual margin",
        lo <= clean_ld <= hi and clean["prefers_answer"] >= PREFERS_ANSWER * n and corrupt_ld < 0,
        f"logit difference {clean_ld:.2f} clean (expected {lo} to {hi}) and {corrupt_ld:.2f} "
        f"corrupt; {clean['prefers_answer']} of {n} clean prompts prefer the answer",
    )
    for name, heads, in_top in (
        ("patching name movers 9.9 and 10.0 restores the answer", ((9, 9), (10, 0)), True),
        ("patching S-inhibition heads 7.3, 7.9, 8.6 and 8.10 restores it", S_INHIBITION, True),
        # Their effect comes from the positions before the end (5.5 at the end alone: 0.000).
        ("patching induction heads 5.5 and 6.9 restores it", INDUCTION, False),
    ):
        check(
            name,
            all(effects[k] >= HELPS and (rank[k] <= TOP or not in_top) for k in heads),
            f"{shown(heads)}; expected at least {HELPS}"
            + (f", in the top {TOP}" if in_top else ""),
        )
    check(
        "patching negative name movers 10.7 and 11.10 works against it",
        all(effects[k] <= HURTS for k in NEGATIVE_NAME_MOVERS)
        and ranked[-1] in NEGATIVE_NAME_MOVERS,
        f"{shown(NEGATIVE_NAME_MOVERS)}; expected at most {HURTS}, one of them the most negative",
    )
    (a, b), (lo, hi) = ranked[0], STRONGEST
    check(
        "effects are fractions of the gap between clean and corrupt",
        lo <= effects[a, b] <= hi,
        f"the strongest head, {a}.{b}, restores {effects[a, b]:.3f} (expected {lo} to {hi})",
    )

    names = (
        "name movers 9.9, 9.6 and 10.0 write the answer most directly",
        "negative name movers 10.7 and 11.10 write against it most directly",
    )
    direct_effects = None if direct is None else _head_effects(direct, "last")
    if direct_effects is None or direct.get("measure") != "attribution":
        for name in names:
            # None: the run failed, which is a failed check already.
            check(
                name,
                None if direct is None else False,
                "expected the direct effect of every head of GPT-2 small at the last position",
            )
        return checks
    order = sorted(direct_effects, key=lambda k: -direct_effects[k])

    def listed(heads: list[tuple[int, int]]) -> str:
        return ", ".join(f"{a}.{b} {direct_effects[a, b]:+.3f}" for a, b in heads)

    check(names[0], set(order[:3]) == set(NAME_MOVERS), f"the largest: {listed(order[:3])}")
    check(
        names[1],
        set(order[-2:]) == set(NEGATIVE_NAME_MOVERS),
        f"the most negative: {listed(order[-2:])}",
    )
    return checks


def validate(
    backend: Any,
    project: Any,
    dataset: str,
    *,
    prepend_bos: bool,
    sae_ref: Any = None,
    compare: Any = None,
    golden: bool = False,
    golden_only: bool = False,
    log: Callable[[str], None] = print,
) -> list[Check]:
    """Run the methods on ``backend`` with ``dataset`` (a path in ``project``) and check them.
    ``compare`` is a second backend of the same model on another device, if any. ``golden`` adds
    the checks of GPT-2 small's published circuit (see golden_checks); ``golden_only`` runs only
    what those need."""
    import numpy as np
    import pyarrow.parquet as pq

    from logogram.compare import compare_summaries
    from logogram.datasets import file_sha256
    from logogram.results import largest_change
    from logogram.runner import run_spec
    from logogram.spec import Spec
    from logogram.verify import verification_spec

    info = backend.info
    tol = TOLERANCE[info.dtype]
    checks: list[Check] = []
    sha = file_sha256(project.root / dataset)

    def check(name: str, passed: bool | None, detail: str, exact: bool = True) -> None:
        checks.append(Check(name, passed, detail, exact))
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[passed]
        log(f"{mark}  {name}: {detail}")

    def spec(experiment: dict, scope: dict, model: Any = None, **extra: Any) -> Spec:
        m = (model or backend).info
        return Spec.model_validate(
            {
                "name": "validation",
                "model": {
                    "id": m.id,
                    "revision": m.revision,
                    "dtype": m.dtype,
                    "device": m.device,
                    "process_weights": m.process_weights,
                },
                "dataset": {"path": dataset, "sha256": sha},
                "tokenization": {"prepend_bos": prepend_bos},
                "experiment": experiment,
                "scope": scope,
                "statistics": {"bootstrap": 200, "ci": 0.95, "seed": 0},
                "execution": {"batch_size": 32},
                **extra,
            }
        )

    def run(label: str, s: Spec, model: Any = None, sae: Any = None) -> Ran | None:
        started = time.perf_counter()
        outcome = run_spec(s, project, backend=model or backend, sae=sae)
        seconds = time.perf_counter() - started
        if outcome.status != "finished":
            check(f"{label} runs", False, str(outcome.manifest.get("error")))
            return None
        log(f"      {label}: {seconds:.1f} s")
        return Ran(outcome.summary or {}, outcome.folder, s)

    def effect(outcome: Any, label: str) -> float | None:
        site = next((s for s in outcome.summary["sites"] if s["label"] == label), None)
        return None if site is None else site["effect"]["mean"]

    patch = {"kind": "activation_patching", "direction": "clean_to_corrupt"}
    estimate = {"kind": "attribution_patching", "direction": "clean_to_corrupt"}
    every_head = {"kind": "heads", "position": {"kind": "all"}}
    last = {"kind": "last"}
    n_layers, n_heads = info.n_layers, info.n_heads
    checks_info = info.extra.get("checks") or {}
    check(
        "The model reproduces itself when it loads",
        True,
        f"error {checks_info.get('function', 0):.1e}, {info.extra.get('block_structure')} layers",
    )

    heads = run("Patching every head", spec(patch, every_head))
    direct_per_head = spec(
        {"kind": "direct_logit_attribution", "prompts": "clean"},
        {"kind": "heads", "position": last},
    )
    per_head: Ran | None = None

    def check_golden() -> None:
        if heads is not None:  # a sweep that failed to run is a failed check already
            for c in golden_checks(heads.summary, per_head.summary if per_head else None):
                check(c.name, c.passed, c.detail, exact=False)

    if golden_only:
        per_head = run("Direct effects of every head", direct_per_head)
        check_golden()
        return checks
    resid = run(
        "Patching the residual stream at each named position",
        spec(patch, {"kind": "layer_position", "site": "resid_pre", "positions": "labels"}),
    )
    if resid is not None:
        at_s2, at_end = effect(resid, "L0 resid pre @ S2"), effect(resid, "L0 resid pre @ end")
        if at_s2 is None:
            check("Patching the token that differs restores everything", None, "no S2 label")
        else:
            check(
                "Patching the token that differs restores everything",
                abs(at_s2 - 1) <= tol and abs(at_end or 0) <= tol,
                f"effect {at_s2:.4f} at S2 and {at_end or 0:.4f} at the end, in layer 0",
            )
    final = run(
        "Patching the final residual stream",
        spec(
            patch,
            {
                "kind": "sites",
                "sites": [{"kind": "resid_post", "layer": n_layers - 1, "position": last}],
            },
        ),
    )
    if final is not None:
        value = final.summary["sites"][0]["effect"]["mean"]
        check(
            "Patching the final residual stream restores everything",
            abs(value - 1) <= tol,
            f"effect {value:.4f}",
        )
    if heads is not None:
        again = run("The same head sweep again", heads.spec)
        if again is not None:
            change = largest_change(
                pq.read_table(heads.folder / "results.parquet"),
                pq.read_table(again.folder / "results.parquet"),
            )
            check(
                "The same spec twice gives bit-identical results",
                change == 0.0,
                "identical" if change == 0.0 else f"largest change {change}",
            )

    components = {
        "kind": "layer_components",
        "components": ["attn_out", "mlp_out"],
        "position": {"kind": "all"},
    }
    for name, baseline in (
        ("Zero ablation", {"kind": "zero"}),
        ("Mean ablation", {"kind": "mean", "reference": "corrupt"}),
        ("Resample ablation", {"kind": "resample", "pool": "corrupt", "donors": 4, "seed": 0}),
    ):
        if run(name, spec({"kind": "ablation", "baseline": baseline}, components)) is not None:
            check(f"{name} runs", True, "finished")

    direct = run(
        "Direct logit attribution",
        spec(
            {"kind": "direct_logit_attribution", "prompts": "clean"},
            {"kind": "layer_components", "components": ["attn_out", "mlp_out"], "position": last},
        ),
    )
    best_heads: list[tuple[int, int]] = []
    if direct is not None:
        split = direct.summary["direct"]
        measured = direct.summary["baseline"]["clean"]["logit_diff"]["mean"]
        parts = split["embeddings"] + split["attention"] + split["mlp"] + split["biases"]
        check(
            "Direct effects add up to the logit difference",
            abs(split["logit_diff"] - measured) <= tol * max(1.0, abs(measured))
            and abs(parts - split["logit_diff"]) <= 1e-6 * max(1.0, abs(measured)),
            f"{split['logit_diff']:.4f} split, {measured:.4f} measured",
        )
        per_head = run("Direct effects of every head", direct_per_head)
        if per_head is not None:
            later = [s for s in per_head.summary["sites"] if s["layer"] >= n_layers // 2]
            later.sort(key=lambda s: -(s["effect"]["mean"] or 0))
            best_heads = [(s["layer"], s["head"]) for s in later[:3]]

    estimated = run("Attribution patching of every head", spec(estimate, every_head))
    if estimated is not None and heads is not None:
        cmp = compare_summaries(heads.summary, estimated.summary, top_k=10)
        check(
            "Attribution patching ranks heads like patching",
            (cmp["spearman"] or 0) >= 0.8,
            f"rank correlation {cmp['spearman']:.3f}, top 10 overlap {cmp['top_overlap']}",
            exact=False,
        )
        verified = run(
            "Verifying the top 10 estimates by patching",
            verification_spec(estimated.spec, estimated.summary, 10),
        )
        if verified is not None:
            cmp = compare_summaries(estimated.summary, verified.summary, top_k=5)
            check(
                "Patching reverses none of the top estimates",
                cmp["n_sign_changes"] == 0,
                f"{cmp['n_sign_changes']} reversed, rank correlation {cmp['spearman']:.3f}",
                exact=False,
            )

    steering = {
        "kind": "steering",
        "apply_to": "clean",
        "coefficients": [-1, 1],
        "train_fraction": 0.5,
        "seed": 0,
        "control": True,
    }
    if run(
        "Steering",
        spec(steering, {"kind": "layer_components", "components": ["resid_pre"], "position": last}),
    ):
        check("Steering runs", True, "finished")

    last_heads = [
        {"kind": "head", "layer": n_layers - 1, "head": h, "position": {"kind": "all"}}
        for h in range(n_heads)
    ]
    to_logits = {
        "kind": "path_patching",
        "direction": "clean_to_corrupt",
        "receivers": [{"kind": "logits"}],
        "freeze_mlps": False,
    }
    path = run(
        "Path patching from the last layer to the logits",
        spec(to_logits, {"kind": "sites", "sites": last_heads}),
    )
    patched = run(
        "Patching the last layer's heads", spec(patch, {"kind": "sites", "sites": last_heads})
    )
    if path is not None and patched is not None:
        a = np.asarray(pq.read_table(path.folder / "results.parquet").column("patched_logit_diff"))
        b = np.asarray(
            pq.read_table(patched.folder / "results.parquet").column("patched_logit_diff")
        )
        check(
            "From the last layer, a path to the logits is the whole effect",
            float(np.abs(a - b).max()) <= tol,
            f"largest difference {float(np.abs(a - b).max()):.2e}",
        )
    if best_heads:
        receivers = [
            {"kind": "head", "layer": layer, "head": h, "input": "q"} for layer, h in best_heads
        ]
        to_queries = {
            "kind": "path_patching",
            "direction": "clean_to_corrupt",
            "receivers": receivers,
            "freeze_mlps": False,
        }
        if run("Path patching into the strongest heads' queries", spec(to_queries, every_head)):
            check("Path patching into heads runs", True, f"receivers {best_heads}")

    if info.extra.get("prediction_method"):
        settings = {
            "method": "final_norm_logit_lens",
            "prompt_index": 0,
            "which": "clean",
            "position": last,
            "top_k": 3,
        }
        lens = run(
            "Layer predictions",
            spec(patch, {"kind": "sites", "sites": last_heads[:1]}, predictions=settings),
        )
        if lens is not None:
            report = json.loads((lens.folder / "predictions.json").read_text(encoding="utf-8"))
            mine = report["layers"][-1]["answer_prob"]
            model = lens.summary["per_prompt"]["clean_answer_prob"][0]
            check(
                "The logit lens at the last layer is the model's own prediction",
                abs(mine - model) <= tol * max(model, 1e-6) + 1e-7,
                f"answer probability {mine:.6f} lens, {model:.6f} model",
            )

    if sae_ref is not None:
        from logogram.analysis import sae_fit_report
        from logogram.datasets import load_dataset
        from logogram.sae import load_sae

        sae = load_sae(sae_ref, info.device)
        fit = sae_fit_report(
            backend,
            sae,
            load_dataset(project.root / dataset),
            prepend_bos=prepend_bos,
            batch_size=32,
        )
        check(
            "The SAE fits these activations",
            fit["variance_explained"] > 0.5,
            f"explains {fit['variance_explained']:.1%} of the variance; check weight processing if not",
            exact=False,
        )
        ref = {"repo": sae.repo, "path": sae.path, "revision": sae.revision}
        features = run(
            "Attribution patching of every SAE feature",
            spec(
                estimate,
                {"kind": "features", "position": {"kind": "label", "label": "end"}, "top": 20},
                sae=ref,
            ),
            sae=sae,
        )
        if features is not None:
            verified = run(
                "Patching the top 10 features",
                verification_spec(features.spec, features.summary, 10),
                sae=sae,
            )
            if verified is not None:
                cmp = compare_summaries(features.summary, verified.summary, top_k=5)
                check(
                    "Feature estimates match patching",
                    cmp["n_sign_changes"] == 0 and (cmp["spearman"] or 0) >= 0.8,
                    f"rank correlation {cmp['spearman']:.3f}, {cmp['n_sign_changes']} reversed",
                    exact=False,
                )

    if compare is not None and heads is not None:
        other = run(
            f"Patching every head on {compare.info.device}", spec(patch, every_head, model=compare)
        )
        if other is not None:
            x = np.asarray(pq.read_table(heads.folder / "results.parquet").column("effect"))
            y = np.asarray(pq.read_table(other.folder / "results.parquet").column("effect"))
            cmp = compare_summaries(heads.summary, other.summary, top_k=10)
            agree = TOLERANCE[compare.info.dtype] if compare.info.dtype != "float32" else tol
            check(
                f"{info.device} and {compare.info.device} agree",
                float(np.abs(x - y).max()) <= agree and cmp["top_overlap"] == 10,
                f"largest difference in a per-prompt effect {float(np.abs(x - y).max()):.2e}, "
                f"top 10 overlap {cmp['top_overlap']}",
            )
    if golden:
        check_golden()
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32", choices=sorted(TOLERANCE))
    parser.add_argument(
        "--no-processing", action="store_true", help="Load without weight processing."
    )
    parser.add_argument("--compare-device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--sae", nargs=2, metavar=("REPO", "PATH"), default=None)
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Also check GPT-2 small against its published IOI circuit (Wang et al. 2022).",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="Run only what the --golden checks need: the head sweep and direct effects.",
    )
    args = parser.parse_args()
    golden = args.golden or args.golden_only
    if golden and args.model != GOLDEN_MODEL:
        parser.error(f"--golden checks what is published about {GOLDEN_MODEL}, not {args.model}.")
    if args.golden_only and (args.sae or args.compare_device):
        parser.error(
            "--golden-only runs only what --golden checks; drop --sae and --compare-device."
        )

    from logogram.backends.transformer_lens import load_model
    from logogram.datasets import load_dataset, write_dataset
    from logogram.ioi import generate_ioi
    from logogram.project import Project, example_source
    from logogram.prompts import prepare_prompts
    from logogram.spec import SAERef, Spec

    if golden and args.revision is None:
        # The weights the bundled example pins, so that every run checks the same model.
        example = example_source() / "experiments" / "ioi-head-patching" / "spec.json"
        args.revision = Spec.from_path(example).model.revision

    started = time.perf_counter()

    def load(device: str) -> Any:
        print(f"Loading {args.model} on {device} ({args.dtype})…", flush=True)
        return load_model(
            args.model,
            revision=args.revision,
            dtype=args.dtype,
            device=device,
            process_weights=not args.no_processing,
        )

    backend = load(args.device)
    compare = load(args.compare_device) if args.compare_device else None
    folder = Path(tempfile.mkdtemp(prefix="logogram-validation-"))
    try:
        shutil.copytree(example_source(), folder / "project")
        project = Project.open(folder / "project")
        bos = bool(backend.info.extra.get("bos"))
        dataset = "datasets/ioi.jsonl"
        try:
            prepare_prompts(backend, load_dataset(project.root / dataset), bos)
        except ValueError as exc:
            # The example's names were chosen for GPT-2; make prompts this tokenizer can use.
            print(f"The example's prompts don't fit this tokenizer ({exc}); generating new ones.")
            records = generate_ioi(
                64, seed=0, single_token=lambda w: backend.single_token_id(w) is not None
            )
            dataset = "datasets/ioi-validation.jsonl"
            write_dataset(project.root / dataset, records)
        sae_ref = SAERef(repo=args.sae[0], path=args.sae[1]) if args.sae else None
        checks = validate(
            backend,
            project,
            dataset,
            prepend_bos=bos,
            sae_ref=sae_ref,
            compare=compare,
            golden=golden,
            golden_only=args.golden_only,
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    failed = [c for c in checks if c.passed is False]
    print(
        f"\n{len(checks) - len(failed)} of {len(checks)} checks passed "
        f"in {time.perf_counter() - started:.0f} s on {backend.info.device_name}."
    )
    for c in failed:
        kind = "an identity" if c.exact else "agreement expected of a trained model"
        print(f"FAILED ({kind}): {c.name}: {c.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
