"""The sanity checks every patching implementation must pass, on a tiny random model."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from logogram.datasets import load_dataset
from logogram.engine import run_engine
from logogram.prompts import prepare_prompts
from logogram.results import compute_stats
from logogram.runner import run_spec

TOL = 1e-4


def _prompts(backend, project, name="ioi.jsonl"):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / name), True)


def _whole_layer_sites(n_layers: int) -> list[dict]:
    sites = []
    for layer in range(n_layers):
        for kind in ("resid_pre", "resid_mid", "resid_post"):
            sites.append({"kind": kind, "layer": layer, "position": {"kind": "all"}})
    return sites


@pytest.mark.parametrize("direction", ["clean_to_corrupt", "corrupt_to_clean"])
@pytest.mark.parametrize("normalization", ["dataset_gap", "prompt_gap"])
def test_patching_all_clean_activations_at_a_layer_gives_effect_one(
    tiny_backend, project, spec_factory, direction, normalization
):
    """Replacing the whole residual stream at a layer reproduces the source run exactly."""
    spec = spec_factory(
        experiment={"kind": "activation_patching", "direction": direction},
        scope={"kind": "sites", "sites": _whole_layer_sites(tiny_backend.info.n_layers)},
        metric={"kind": "logit_diff", "normalization": normalization},
    )
    prompts = _prompts(tiny_backend, project)
    result = run_engine(spec, tiny_backend, prompts)
    stats = compute_stats(spec, result)
    np.testing.assert_allclose(stats.effect_mean, 1.0, atol=TOL)
    # The patched run *is* the source run, prompt by prompt.
    np.testing.assert_allclose(
        result.patched_ld, np.broadcast_to(result.reference_ld, result.patched_ld.shape), atol=TOL
    )
    if normalization == "prompt_gap":
        np.testing.assert_allclose(stats.effect, 1.0, atol=1e-3)


def test_patching_corrupt_into_corrupt_gives_zero(tiny_backend, project, spec_factory):
    """Patching an activation into the run it came from changes nothing."""
    prompts = _prompts(tiny_backend, project)
    for scope in (
        {"kind": "heads", "position": {"kind": "all"}},
        {"kind": "heads", "position": {"kind": "last"}},
        {"kind": "layer_components", "position": {"kind": "all"}},
        {"kind": "layer_position", "site": "resid_mid", "positions": "each"},
    ):
        spec = spec_factory(scope=scope)
        result = run_engine(
            spec, tiny_backend, prompts, receiver_override="corrupt", source_override="corrupt"
        )
        stats = compute_stats(spec, result)
        np.testing.assert_allclose(stats.delta, 0.0, atol=TOL)
        np.testing.assert_allclose(stats.effect_mean, 0.0, atol=TOL)
        assert int(stats.sign_flips.sum()) == 0


def test_same_spec_twice_gives_identical_results(tiny_backend, project, spec_factory):
    spec = spec_factory(
        experiment={
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "corrupt", "donors": 3, "seed": 7},
        }
    )
    first = run_spec(spec, project, backend=tiny_backend)
    second = run_spec(spec, project, backend=tiny_backend)
    assert first.status == second.status == "finished"
    a = pq.read_table(first.folder / "results.parquet")
    b = pq.read_table(second.folder / "results.parquet")
    assert a.equals(b)
    sa = json.loads((first.folder / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((second.folder / "summary.json").read_text(encoding="utf-8"))
    sa.pop("run_id"), sb.pop("run_id")
    assert sa == sb
    assert (first.folder / "spec.json").read_text(encoding="utf-8") == (
        second.folder / "spec.json"
    ).read_text(encoding="utf-8")


def test_patching_every_head_matches_the_attention_output(tiny_backend, project, spec_factory):
    """Sum over heads: patching the full attention output equals patching z for every head."""
    import torch

    from logogram.backends.base import Patch

    prompts = _prompts(tiny_backend, project)
    p = prompts[0]
    clean = torch.tensor([p.clean.ids])
    corrupt = torch.tensor([p.corrupt.ids])
    info = tiny_backend.info
    acts = tiny_backend.capture(clean, [("head", 1), ("attn_out", 1)])
    z = acts[("head", 1)]
    attn_patch = Patch(kind="attn_out", layer=1, values=acts[("attn_out", 1)])
    out_attn = tiny_backend.final_logits(corrupt, attn_patch)

    def patch_all_heads(act, hook=None):
        return z.to(act.dtype)

    with torch.no_grad():
        logits = tiny_backend.bridge.run_with_hooks(
            corrupt.to(tiny_backend.device),
            fwd_hooks=[("blocks.1.attn.hook_z", patch_all_heads)],
            return_type="logits",
        )[:, -1].float()
    assert info.n_heads == z.shape[2]
    torch.testing.assert_close(out_attn, logits, atol=1e-4, rtol=1e-4)


def test_resid_mid_patch_changes_the_residual_stream(tiny_backend, project, spec_factory):
    """The MLP acts per position, so patching resid_mid at one position equals patching resid_post
    there. (A patch that only edited the MLP's input would fail this.)"""
    prompts = _prompts(tiny_backend, project)
    s2 = {"kind": "label", "label": "S2"}
    spec = spec_factory(
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "resid_mid", "layer": 0, "position": s2},
                {"kind": "resid_post", "layer": 0, "position": s2},
                {"kind": "resid_mid", "layer": 1, "position": {"kind": "last"}},
                {"kind": "resid_post", "layer": 1, "position": {"kind": "last"}},
            ],
        }
    )
    result = run_engine(spec, tiny_backend, prompts)
    np.testing.assert_allclose(result.patched_ld[0], result.patched_ld[1], atol=TOL)
    np.testing.assert_allclose(result.patched_ld[2], result.patched_ld[3], atol=TOL)
    assert np.abs(result.patched_ld[0] - result.receiver_ld).max() > 1e-3  # it does change things
