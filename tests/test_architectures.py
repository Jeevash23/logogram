"""Other architecture families: what Logogram measures when a model loads, and the sanity checks.

Each family is a tiny random model built locally (see conftest.py), so these tests need no network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import TINY_ARCHITECTURES
from logogram.backends.base import BackendError, Patch
from logogram.backends.transformer_lens import TransformerLensBackend, boot_local
from logogram.datasets import load_dataset
from logogram.engine import run_engine, run_experiment
from logogram.prompts import prepare_prompts
from logogram.results import compute_stats
from logogram.sites import ScopeError

ARCHS = list(TINY_ARCHITECTURES)
STRUCTURE = {
    "llama": "sequential",
    "gpt_neox": "parallel",
    "qwen2": "sequential",
    "gemma2": "sequential",
    "olmo2": "sequential",
}
TOL = 1e-4


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


@pytest.mark.parametrize("arch", ARCHS)
def test_the_structure_is_measured_when_the_model_loads(arch_backend, arch):
    info = arch_backend(arch).info
    checks = info.extra["checks"]
    assert info.extra["block_structure"] == STRUCTURE[arch]
    assert checks["function"] <= checks["tolerance"]
    assert checks["residual"] is not None and checks["residual"] <= checks["tolerance"]
    # Only a layer that adds attention and then the MLP has a residual stream between them.
    assert ("resid_mid" in info.site_kinds) == (STRUCTURE[arch] == "sequential")
    assert info.extra["prediction_method"] == "final_norm_logit_lens"
    assert info.extra["bos"] is True


@pytest.mark.parametrize("arch", ARCHS)
def test_weight_processing_keeps_the_predictions(arch_backend, arch):
    processed = arch_backend(arch, process_weights=True).info.extra["checks"]
    plain = arch_backend(arch, process_weights=False).info.extra["checks"]
    assert processed["function"] <= processed["tolerance"]
    assert plain["function"] <= plain["tolerance"]


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("direction", ["clean_to_corrupt", "corrupt_to_clean"])
def test_patching_a_whole_layer_reproduces_the_source(
    arch_backend, project, spec_factory, arch, direction
):
    backend = arch_backend(arch)
    kinds = [k for k in ("resid_pre", "resid_mid", "resid_post") if k in backend.info.site_kinds]
    sites = [
        {"kind": kind, "layer": layer, "position": {"kind": "all"}}
        for layer in range(backend.info.n_layers)
        for kind in kinds
    ]
    spec = spec_factory(
        experiment={"kind": "activation_patching", "direction": direction},
        scope={"kind": "sites", "sites": sites},
    )
    result = run_engine(spec, backend, _prompts(backend, project))
    stats = compute_stats(spec, result)
    np.testing.assert_allclose(stats.effect_mean, 1.0, atol=TOL)
    np.testing.assert_allclose(
        result.patched, np.broadcast_to(result.reference_metric, result.patched.shape), atol=TOL
    )


@pytest.mark.parametrize("arch", ARCHS)
def test_patching_into_the_same_run_changes_nothing(arch_backend, project, spec_factory, arch):
    backend = arch_backend(arch)
    prompts = _prompts(backend, project)
    for scope in (
        {"kind": "heads", "position": {"kind": "all"}},
        {"kind": "heads", "position": {"kind": "last"}},
        {"kind": "layer_components", "position": {"kind": "all"}},
    ):
        spec = spec_factory(scope=scope)
        result = run_engine(
            spec, backend, prompts, receiver_override="corrupt", source_override="corrupt"
        )
        stats = compute_stats(spec, result)
        np.testing.assert_allclose(stats.delta, 0.0, atol=TOL)


@pytest.mark.parametrize("arch", ARCHS)
def test_every_head_together_equals_the_attention_output(arch_backend, project, arch):
    backend = arch_backend(arch)
    p = _prompts(backend, project)[0]
    clean = torch.tensor([p.clean.ids])
    corrupt = torch.tensor([p.corrupt.ids])
    acts = backend.capture(clean, [("head", 1), ("attn_out", 1)])
    by_attn = backend.final_logits(
        corrupt, Patch(kind="attn_out", layer=1, values=acts[("attn_out", 1)])
    )
    z = acts[("head", 1)]
    assert z.shape[2] == backend.info.n_heads  # query heads, also with grouped-query attention

    def every_head(act, hook=None):
        return z.to(act.dtype)

    with torch.no_grad():
        by_heads = backend.bridge.run_with_hooks(
            corrupt, fwd_hooks=[("blocks.1.attn.hook_z", every_head)], return_type="logits"
        )[:, -1].float()
    torch.testing.assert_close(by_attn, by_heads, atol=1e-4, rtol=1e-4)


def test_a_parallel_model_has_no_residual_stream_between_attention_and_mlp(
    arch_backend, project, spec_factory
):
    backend = arch_backend("gpt_neox")
    spec = spec_factory(scope={"kind": "layer_position", "site": "resid_mid", "positions": "each"})
    with pytest.raises(ScopeError, match="resid_mid"):
        run_engine(spec, backend, _prompts(backend, project))


@pytest.mark.parametrize("process_weights", [True, False])
def test_a_model_whose_predictions_change_is_refused(arch_backend, process_weights):
    folder = arch_backend.folder("llama")
    bridge = boot_local(folder, device="cpu", dtype="float32")
    original = bridge.enable_compatibility_mode

    def compatibility_mode_that_changes_the_model(*args, **kwargs):
        original(*args, **kwargs)
        with torch.no_grad():
            weight = bridge.original_model.get_output_embeddings().weight
            weight.mul_(torch.linspace(0.5, 1.5, weight.shape[0])[:, None])

    bridge.enable_compatibility_mode = compatibility_mode_that_changes_the_model
    expected = "weight processing off" if process_weights else "doesn't reproduce"
    with pytest.raises(BackendError, match=expected):
        TransformerLensBackend.from_bridge(
            bridge,
            model_id="tiny-llama",
            revision="test",
            dtype="float32",
            process_weights=process_weights,
        )


def test_an_unsupported_architecture_is_named_before_downloading():
    from logogram.backends.transformer_lens import architecture_support

    assert architecture_support("LlamaForCausalLM") is None
    assert architecture_support("GPTNeoXForCausalLM") is None
    assert architecture_support("unknown") is None  # a config without architectures: loading says
    note = architecture_support("ImaginaryForCausalLM")
    assert note is not None and "can't load ImaginaryForCausalLM" in note


# -- queries, keys, values and neurons -----------------------------------------------------------

# What each family offers of a head's query, key and value and an MLP neuron, as measured when it
# loads. OLMo 2 normalizes every head's queries (and keys) together, so one head's can't be
# patched alone; its values can.
FINER = {
    "llama": {"q", "k", "v", "neuron"},
    "gpt_neox": {"q", "k", "v", "neuron"},
    "qwen2": {"q", "k", "v", "neuron"},
    "gemma2": {"q", "k", "v", "neuron"},
    "olmo2": {"v", "neuron"},
}
# Gemma 2 and OLMo 2 normalize the MLP's output: it isn't the sum of what its neurons write.
NEURONS_ADD_UP = {"llama": True, "gpt_neox": True, "qwen2": True, "gemma2": False, "olmo2": False}
ALL = {"kind": "all"}
LAST = {"kind": "last"}


def _sets(**sets):
    return {
        "kind": "site_sets",
        "universe": None,
        "sets": [{"label": k, "sites": v, "complement": False} for k, v in sets.items()],
    }


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("process_weights", [True, False])
def test_queries_keys_values_and_neurons_are_measured_when_the_model_loads(
    arch_backend, arch, process_weights
):
    info = arch_backend(arch, process_weights=process_weights).info
    checks = info.extra["checks"]
    tol = checks["tolerance"]
    for name in ("qkv", "v_alone", "values", "neurons"):
        assert checks[name] is not None and checks[name] <= tol, name
    assert {"q", "k", "v", "neuron"} & set(info.site_kinds) == FINER[arch]
    assert (checks["neuron_writes"] <= tol) == NEURONS_ADD_UP[arch]
    notes = info.extra["site_notes"]
    assert set(notes) == {"q", "k", "v", "neuron"} - FINER[arch]
    if arch == "olmo2":
        assert checks["q_alone"] > tol and checks["k_alone"] > tol
        assert "a single head's query can't be patched alone" in notes["q"]
        assert "a single head's key can't be patched alone" in notes["k"]


@pytest.mark.parametrize("arch", ARCHS)
def test_every_query_key_and_value_of_a_layer_together_equal_its_attention_output(
    arch_backend, project, spec_factory, arch
):
    backend = arch_backend(arch)
    info = backend.info
    kv = int(info.extra["n_key_value_heads"])
    qkv = [{"kind": "q", "layer": 1, "head": h, "position": ALL} for h in range(info.n_heads)]
    qkv += [
        {"kind": k, "layer": 1, "head": h, "position": ALL} for k in ("k", "v") for h in range(kv)
    ]
    spec = spec_factory(
        scope=_sets(qkv=qkv, attention=[{"kind": "attn_out", "layer": 1, "position": ALL}])
    )
    prompts = _prompts(backend, project)
    if not FINER[arch] >= {"q", "k", "v"}:
        with pytest.raises(ScopeError, match="can't be patched alone"):
            run_experiment(spec, backend, prompts)
        return
    result = run_experiment(spec, backend, prompts)
    np.testing.assert_allclose(result.patched[0], result.patched[1], atol=TOL)
    assert np.abs(result.patched[0] - result.receiver_metric).max() > 1e-3


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("position", [ALL, {"kind": "label", "label": "S2"}])
def test_every_neuron_of_a_layer_together_equals_its_mlp_output(
    arch_backend, project, spec_factory, arch, position
):
    backend = arch_backend(arch)
    neurons = [
        {"kind": "neuron", "layer": 0, "neuron": n, "position": position}
        for n in range(backend.info.d_mlp)
    ]
    spec = spec_factory(
        scope=_sets(neurons=neurons, mlp=[{"kind": "mlp_out", "layer": 0, "position": position}])
    )
    result = run_experiment(spec, backend, _prompts(backend, project))
    np.testing.assert_allclose(result.patched[0], result.patched[1], atol=TOL)
    assert np.abs(result.patched[0] - result.receiver_metric).max() > 1e-3


@pytest.mark.parametrize("arch", ARCHS)
def test_patching_queries_keys_values_and_neurons_into_the_same_run_changes_nothing(
    arch_backend, project, spec_factory, arch
):
    backend = arch_backend(arch)
    prompts = _prompts(backend, project)[:4]
    scopes = [
        {"kind": "head_inputs", "input": k, "position": LAST}
        for k in ("q", "k", "v")
        if k in FINER[arch]
    ]
    scopes.append({"kind": "neurons", "layers": [1], "position": ALL})
    for scope in scopes:
        spec = spec_factory(scope=scope, execution={"batch_size": 64})
        result = run_engine(
            spec, backend, prompts, receiver_override="corrupt", source_override="corrupt"
        )
        stats = compute_stats(spec, result)
        np.testing.assert_allclose(stats.delta, 0.0, atol=TOL, err_msg=str(scope))


@pytest.mark.parametrize("arch", ["llama", "qwen2", "gemma2", "olmo2"])
def test_a_key_or_value_reaches_exactly_the_query_heads_that_read_it(arch_backend, project, arch):
    """Grouped-query attention: key/value head j is read by query heads j·g … j·g + g − 1, so
    patching its key or value changes their outputs and no other head's."""
    from logogram.backends.transformer_lens import _patch_hooks

    backend = arch_backend(arch)
    p = _prompts(backend, project)[0]
    clean, corrupt = torch.tensor([p.clean.ids]), torch.tensor([p.corrupt.ids])
    n_heads, kv = backend.info.n_heads, int(backend.info.extra["n_key_value_heads"])
    group = n_heads // kv
    assert group > 1

    def z_after(patch):
        seen = {}

        def keep(act, hook=None):
            seen["z"] = act.detach().clone()
            return act

        with torch.no_grad():
            backend.bridge.run_with_hooks(
                corrupt,
                fwd_hooks=[*_patch_hooks(patch), ("blocks.1.attn.hook_z", keep)],
                return_type=None,
            )
        return seen["z"][0]

    own = z_after(None)
    for kind in sorted({"k", "v"} & FINER[arch]):
        acts = backend.capture(clean, [(kind, 1)])[(kind, 1)]
        for unit in range(kv):
            z = z_after(
                Patch(kind=kind, layer=1, values=acts[:, :, unit], heads=torch.tensor([unit]))
            )
            changed = {h for h in range(n_heads) if not torch.equal(z[:, h], own[:, h])}
            assert changed == set(range(unit * group, (unit + 1) * group)), (kind, unit)


@pytest.mark.parametrize("arch", ARCHS)
def test_neurons_have_a_direct_effect_only_where_they_add_up(
    arch_backend, project, spec_factory, arch
):
    backend = arch_backend(arch)
    spec = spec_factory(
        experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
        scope={"kind": "neurons", "layers": [1], "position": LAST},
    )
    prompts = _prompts(backend, project)
    if backend.info.extra.get("logit_soft_cap"):
        with pytest.raises(ScopeError, match="soft-caps"):
            run_experiment(spec, backend, prompts)
    elif NEURONS_ADD_UP[arch]:
        result = run_experiment(spec, backend, prompts)
        assert np.isfinite(result.delta).all()
    else:
        with pytest.raises(ScopeError, match="sum of what its neurons write"):
            run_experiment(spec, backend, prompts)
