"""Queries, keys, values and neurons as sites: the sanity checks of tests/test_sanity.py for them,
what the backend measures about them when a model loads, and the methods that read them.

tests/test_architectures.py runs the same checks on tiny models of other families, grouped-query
attention and gated MLPs included.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from logogram.backends.base import Patch
from logogram.datasets import load_dataset
from logogram.direct import check_direct_sites
from logogram.engine import run_engine, run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.results import compute_stats
from logogram.runner import run_spec
from logogram.sites import ScopeError, expand_scope, resolve_position

TOL = 1e-4
UNIT_CHECKS = ("qkv", "q_alone", "k_alone", "v_alone", "values", "neurons", "neuron_writes")
PATCH = {"kind": "activation_patching", "direction": "clean_to_corrupt"}
ALL = {"kind": "all"}
LAST = {"kind": "last"}


def _prompts(backend, project, limit=None):
    prompts = prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    return prompts if limit is None else prompts[:limit]


def kv_heads(backend) -> int:
    return int(backend.info.extra["n_key_value_heads"])


def qkv_sites(backend, layer: int, position=ALL) -> list[dict]:
    """Every head's query, and every key/value head's key and value, of a layer."""
    sites = [
        {"kind": "q", "layer": layer, "head": h, "position": position}
        for h in range(backend.info.n_heads)
    ]
    for kind in ("k", "v"):
        sites += [
            {"kind": kind, "layer": layer, "head": h, "position": position}
            for h in range(kv_heads(backend))
        ]
    return sites


def neuron_sites(backend, layer: int, position=ALL) -> list[dict]:
    return [
        {"kind": "neuron", "layer": layer, "neuron": n, "position": position}
        for n in range(backend.info.d_mlp)
    ]


def sets_scope(**sets: list[dict]) -> dict:
    return {
        "kind": "site_sets",
        "universe": None,
        "sets": [
            {"label": label, "sites": sites, "complement": False} for label, sites in sets.items()
        ],
    }


# -- what loading measures ---------------------------------------------------------------------


def test_loading_measures_queries_keys_values_and_neurons(tiny_backend):
    info = tiny_backend.info
    checks = info.extra["checks"]
    for name in UNIT_CHECKS:
        assert checks[name] is not None and checks[name] <= checks["tolerance"], name
    assert {"q", "k", "v", "neuron"} <= set(info.site_kinds)
    assert info.extra["site_notes"] == {}


@pytest.mark.parametrize(
    ("measured", "missing", "says"),
    [
        ({"qkv": 0.3}, {"q", "k", "v"}, "didn't reproduce that input's attention output"),
        ({"qkv": None}, {"q", "k", "v"}, "doesn't expose this model's queries"),
        ({"q_alone": 0.2}, {"q"}, "a single head's query can't be patched alone"),
        ({"k_alone": float("inf")}, {"k"}, "values that aren't finite numbers"),
        ({"v_alone": 0.1}, {"v"}, "other than the query heads of its group"),
        ({"neurons": 0.5}, {"neuron"}, "didn't reproduce that input's MLP output"),
        ({"neurons": None}, {"neuron"}, "doesn't expose this model's MLP neurons"),
    ],
)
def test_sites_that_fail_their_measurement_are_refused(
    tiny_model_dir, monkeypatch, measured, missing, says
):
    """Don't trust a model to be what its architecture name says: a query, key, value or neuron
    whose measurement fails when the model loads isn't offered, and the model says why."""
    from logogram.backends import transformer_lens
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    real = transformer_lens.check_units

    def check_units(*args, **kwargs):
        return {**real(*args, **kwargs), **measured}

    monkeypatch.setattr(transformer_lens, "check_units", check_units)
    bridge = boot_local(tiny_model_dir, device="cpu", dtype="float32")
    backend = TransformerLensBackend.from_bridge(
        bridge, model_id="tiny-gpt2", revision="test", dtype="float32", process_weights=True
    )
    info = backend.info
    offered = {"q", "k", "v", "neuron"} & set(info.site_kinds)
    assert offered == {"q", "k", "v", "neuron"} - missing
    assert set(info.extra["site_notes"]) == missing
    kind = sorted(missing)[0]
    assert says in info.extra["site_notes"][kind]
    scope = (
        {"kind": "neurons", "layers": [0], "position": ALL}
        if kind == "neuron"
        else {"kind": "head_inputs", "input": kind, "position": ALL}
    )
    with pytest.raises(ScopeError) as refused:
        expand_scope(_spec(scope), info, [])
    assert str(refused.value) == info.extra["site_notes"][kind]


def _spec(scope: dict, **overrides):
    from conftest import make_spec

    return make_spec(scope=scope, **overrides)


# -- the sanity checks -------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["clean_to_corrupt", "corrupt_to_clean"])
@pytest.mark.parametrize("layer", [0, 1])
def test_patching_every_query_key_and_value_of_a_layer_equals_patching_its_attention_output(
    tiny_backend, project, spec_factory, direction, layer
):
    """The attention output is a function of the layer's queries, keys and values alone: patching
    all of them reproduces patching the attention output, prompt by prompt."""
    spec = spec_factory(
        experiment={"kind": "activation_patching", "direction": direction},
        scope=sets_scope(
            qkv=qkv_sites(tiny_backend, layer),
            attention=[{"kind": "attn_out", "layer": layer, "position": ALL}],
        ),
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    np.testing.assert_allclose(result.patched[0], result.patched[1], atol=TOL)
    assert np.abs(result.patched[0] - result.receiver_metric).max() > 1e-3  # it does change things


@pytest.mark.parametrize(
    "experiment",
    [
        PATCH,
        {"kind": "activation_patching", "direction": "corrupt_to_clean"},
        {"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}},
        {
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "corrupt", "donors": 2, "seed": 3},
        },
    ],
)
@pytest.mark.parametrize("position", [ALL, {"kind": "label", "label": "S2"}, LAST])
def test_patching_every_neuron_of_a_layer_equals_patching_its_mlp_output(
    tiny_backend, project, spec_factory, experiment, position
):
    """The MLP output is its neurons' activations through the output projection, position by
    position: patching every neuron reproduces patching the MLP output, at every position or at
    one. It is affine in them, so a mean over prompts of every neuron is the mean MLP output.
    (Layer 0: a later layer reads what its MLP writes at any position.)"""
    layer = 0
    spec = spec_factory(
        experiment=experiment,
        scope=sets_scope(
            neurons=neuron_sites(tiny_backend, layer, position),
            mlp=[{"kind": "mlp_out", "layer": layer, "position": position}],
        ),
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    np.testing.assert_allclose(result.patched[0], result.patched[1], atol=TOL)
    assert np.abs(result.patched[0] - result.receiver_metric).max() > 1e-3


def test_zeroing_every_neuron_leaves_the_mlp_output_bias(tiny_backend, project, spec_factory):
    """Zero ablation of every neuron isn't zero ablation of the MLP output: the output projection
    still adds its bias. Patching the MLP output with that bias reproduces it."""
    layer = 1
    prompts = _prompts(tiny_backend, project, limit=4)
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope=sets_scope(neurons=neuron_sites(tiny_backend, layer)),
    )
    result = run_experiment(spec, tiny_backend, prompts)
    bias = tiny_backend.bridge.blocks[layer].mlp.b_out.detach()
    for i, p in enumerate(prompts):
        clean = torch.tensor([p.clean.ids])
        values = bias.expand(1, clean.shape[1], -1)
        logits = tiny_backend.final_logits(clean, Patch(kind="mlp_out", layer=layer, values=values))
        expected = float(logits[0, p.answer_id] - logits[0, p.distractor_id])
        assert result.patched[0, i] == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize(
    "scope",
    [
        {"kind": "head_inputs", "input": "q", "position": ALL},
        {"kind": "head_inputs", "input": "k", "position": LAST},
        {"kind": "head_inputs", "input": "v", "position": {"kind": "label", "label": "S2"}},
        {"kind": "neurons", "layers": [1], "position": LAST},
        {"kind": "neurons", "layers": [0], "position": ALL},
    ],
)
def test_patching_corrupt_into_corrupt_gives_zero(tiny_backend, project, spec_factory, scope):
    spec = spec_factory(scope=scope, execution={"batch_size": 64})
    prompts = _prompts(tiny_backend, project, limit=4)
    result = run_engine(
        spec, tiny_backend, prompts, receiver_override="corrupt", source_override="corrupt"
    )
    stats = compute_stats(spec, result)
    np.testing.assert_allclose(stats.delta, 0.0, atol=TOL)
    assert int(stats.sign_flips.sum()) == 0


def test_the_same_spec_twice_gives_identical_results(tiny_backend, project, spec_factory):
    sites = [
        {"kind": "q", "layer": 1, "head": 2, "position": LAST},
        {"kind": "k", "layer": 1, "head": 0, "position": ALL},
        {"kind": "v", "layer": 0, "head": 3, "position": {"kind": "label", "label": "IO"}},
        {"kind": "neuron", "layer": 0, "neuron": 17, "position": LAST},
        {"kind": "neuron", "layer": 1, "neuron": 5, "position": ALL},
    ]
    spec = spec_factory(
        experiment={
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "corrupt", "donors": 2, "seed": 7},
        },
        scope={"kind": "sites", "sites": sites},
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
    # The results name each site's neuron, and -1 where a site has none.
    neurons = a.column("neuron").to_pylist()
    kinds = a.column("kind").to_pylist()
    assert {n for n, k in zip(neurons, kinds, strict=True) if k == "neuron"} == {17, 5}
    assert {n for n, k in zip(neurons, kinds, strict=True) if k != "neuron"} == {-1}
    labels = [s["label"] for s in sa["sites"]]
    assert labels == ["L1 H2 q @ last", "L1 KV0 k", "L0 KV3 v @ IO", "L0 N17 @ last", "L1 N5"]
    assert [s["neuron"] for s in sa["sites"]] == [None, None, None, 17, 5]


# -- grouped-query attention and gated MLPs ----------------------------------------------------


def _z_after(backend, tokens, patch: Patch | None, layer: int) -> torch.Tensor:
    from logogram.backends.transformer_lens import _patch_hooks

    seen = {}

    def keep(act, hook=None):
        seen["z"] = act.detach().clone()
        return act

    hooks = [*_patch_hooks(patch), (f"blocks.{layer}.attn.hook_z", keep)]
    with torch.no_grad():
        backend.bridge.run_with_hooks(tokens, fwd_hooks=hooks, return_type=None)
    return seen["z"]


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_patching_a_key_or_value_changes_every_query_head_that_reads_it_and_no_other(
    arch_backend, tiny_backend, project, arch
):
    """With grouped-query attention, key/value head j is read by query heads j·g … j·g + g − 1:
    patching its key or value changes exactly their outputs. A query changes its own head's."""
    backend = tiny_backend if arch == "gpt2" else arch_backend(arch)
    p = _prompts(backend, project)[0]
    clean, corrupt = torch.tensor([p.clean.ids]), torch.tensor([p.corrupt.ids])
    layer, n_heads, n_kv = 1, backend.info.n_heads, kv_heads(backend)
    group = n_heads // n_kv
    assert (group > 1) == (arch == "llama")
    own = _z_after(backend, corrupt, None, layer)
    for kind, count in (("q", n_heads), ("k", n_kv), ("v", n_kv)):
        acts = backend.capture(clean, [(kind, layer)])[(kind, layer)]
        for unit in range(count):
            patch = Patch(
                kind=kind, layer=layer, values=acts[:, :, unit], heads=torch.tensor([unit])
            )
            z = _z_after(backend, corrupt, patch, layer)
            changed = {h for h in range(n_heads) if not torch.equal(z[0, :, h], own[0, :, h])}
            readers = {unit} if kind == "q" else set(range(unit * group, (unit + 1) * group))
            assert changed == readers, (kind, unit)


def test_a_gated_neuron_is_the_gate_activation_times_the_up_projection(arch_backend, project):
    """In a gated MLP (SwiGLU) a neuron's activation is silu(gate) × up: the input to the down
    projection, which the down projection turns into the MLP's output."""
    backend = arch_backend("llama")
    p = _prompts(backend, project)[0]
    tokens = torch.tensor([p.clean.ids])
    layer = 1
    mlp = backend.bridge.blocks[layer].mlp
    acts = backend.capture(tokens, [("neuron", layer), ("mlp_out", layer)])
    seen = {}

    def keep(act, hook=None):
        seen["in"] = act.detach().clone()
        return act

    with torch.no_grad():
        backend.bridge.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{layer}.mlp.hook_in", keep)], return_type=None
        )
        x = seen["in"].double()
        gate = x @ mlp.W_gate.double()
        up = x @ mlp.W_in.double()
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(acts[("neuron", layer)].double(), expected, atol=1e-5, rtol=1e-4)
        written = acts[("neuron", layer)].double() @ mlp.W_out.double()
        if mlp.b_out is not None:
            written = written + mlp.b_out.double()
        torch.testing.assert_close(written, acts[("mlp_out", layer)].double(), atol=1e-5, rtol=1e-4)
    assert backend.info.extra["activation"] == "silu"


# -- attribution patching ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def smooth_backend(tmp_path_factory):
    """Tiny models with small weights: smooth enough that finite differences measure slopes."""
    from conftest import build_tiny_architecture
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    backends = {}

    def get(arch):
        if arch not in backends:
            folder = tmp_path_factory.mktemp(f"smooth-{arch}")
            build_tiny_architecture(folder, arch, initializer_range=0.05)
            backends[arch] = TransformerLensBackend.from_bridge(
                boot_local(folder, device="cpu", dtype="float32"),
                model_id=f"smooth-{arch}",
                revision="test",
                dtype="float32",
                process_weights=True,
            )
        return backends[arch]

    return get


FD_SITES = [
    {"kind": "q", "layer": 1, "head": 1, "position": LAST},
    {"kind": "k", "layer": 1, "head": 1, "position": {"kind": "label", "label": "IO"}},
    {"kind": "v", "layer": 1, "head": 0, "position": {"kind": "label", "label": "S1"}},
    {"kind": "neuron", "layer": 0, "neuron": 7, "position": LAST},
    {"kind": "neuron", "layer": 1, "neuron": 30, "position": {"kind": "label", "label": "S2"}},
]
HOOK = {
    "q": "blocks.{layer}.attn.hook_q",
    "k": "blocks.{layer}.attn.hook_k",
    "v": "blocks.{layer}.attn.hook_v",
    "neuron": "blocks.{layer}.mlp.hook_post",
}


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_attribution_patching_estimates_match_finite_differences(
    smooth_backend, project, spec_factory, arch
):
    """An estimate is (source − receiver) · gradient: the slope of the metric as the site moves
    from its receiver value toward its source value, which a five-point stencil measures."""
    backend = smooth_backend(arch)
    spec = spec_factory(
        experiment={**PATCH, "kind": "attribution_patching", "method": "gradient", "steps": None},
        scope={"kind": "sites", "sites": FD_SITES},
    )
    prompts = _prompts(backend, project)
    result = run_experiment(spec, backend, prompts)
    group = group_by_length(prompts)[0]
    rows = torch.arange(len(group.members))
    answers = torch.tensor([prompts[i].answer_id for i in group.members])
    distractors = torch.tensor([prompts[i].distractor_id for i in group.members])
    # A step along the patch, as a share of the whole patch: one site's change moves the logits
    # by little, so a smaller step would measure float32's rounding instead of the slope.
    eps = 0.05
    for rs in result.sites:
        site = rs.site
        unit = site.neuron if site.kind == "neuron" else site.head
        key = (site.kind, site.layer)
        receiver = backend.capture(group.corrupt, [key])[key]
        source = backend.capture(group.clean, [key])[key]
        positions = torch.tensor(
            [resolve_position(site.position, prompts[i]) for i in group.members]
        )
        step = (source - receiver)[rows, positions, unit]

        def metric(t, step=step, positions=positions, unit=unit, site=site):
            def nudge(act, hook=None):
                act = act.clone()
                act[rows, positions, unit] += t * step.to(act.dtype)
                return act

            with torch.no_grad():
                logits = backend.bridge.run_with_hooks(
                    group.corrupt,
                    fwd_hooks=[(HOOK[site.kind].format(layer=site.layer), nudge)],
                    return_type="logits",
                )[:, -1].double()
            return (logits[rows, answers] - logits[rows, distractors]).numpy()

        f = {k: metric(k * eps) for k in (-2, -1, 1, 2)}
        numeric = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * eps)
        estimate = result.delta[rs.index, group.members]
        scale = max(1e-3, float(np.abs(numeric).max()))
        np.testing.assert_allclose(
            estimate, numeric, rtol=1e-2, atol=1e-2 * scale, err_msg=f"{arch} {rs.label}"
        )


def test_integrated_gradients_estimate_queries_keys_values_and_neurons(
    tiny_backend, project, spec_factory
):
    spec = spec_factory(
        experiment={
            **PATCH,
            "kind": "attribution_patching",
            "method": "integrated_gradients",
            "steps": 4,
        },
        scope={"kind": "sites", "sites": FD_SITES},
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    assert np.isfinite(result.delta).all()
    assert np.abs(result.delta).max() > 0
    # Patching the corrupt run into itself estimates nothing.
    same = run_experiment(
        spec,
        tiny_backend,
        _prompts(tiny_backend, project),
        receiver_override="corrupt",
        source_override="corrupt",
    )
    np.testing.assert_allclose(same.delta, 0.0, atol=1e-12)


# -- direct logit attribution ------------------------------------------------------------------

DLA = {"kind": "direct_logit_attribution", "prompts": "clean"}


def test_neurons_split_their_mlp_outputs_direct_effect(tiny_backend, project, spec_factory):
    """A neuron's direct effect is its activation times its output row, read through the final
    normalization with its scale held: the neurons of a layer and the output bias add up to the
    MLP output's direct effect."""
    layer = 1
    sites = [
        *neuron_sites(tiny_backend, layer, LAST),
        {"kind": "mlp_out", "layer": layer, "position": LAST},
    ]
    spec = spec_factory(experiment=DLA, scope={"kind": "sites", "sites": sites})
    prompts = _prompts(tiny_backend, project)
    result = run_experiment(spec, tiny_backend, prompts)
    neurons, mlp = result.delta[:-1].sum(0), result.delta[-1]
    bias = tiny_backend.bridge.blocks[layer].mlp.b_out.detach()
    n = tiny_backend.info.n_layers
    for i, p in enumerate(prompts):
        tokens = torch.tensor([p.clean.ids])
        final = tiny_backend.capture(tokens, [("resid_post", n - 1)])[("resid_post", n - 1)]
        changes = tiny_backend.unembed_directions(final[0, -1], bias[None, :])[0]
        biased = float(changes[p.answer_id] - changes[p.distractor_id])
        assert neurons[i] + biased == pytest.approx(mlp[i], abs=1e-4)


def test_values_split_their_heads_direct_effect(tiny_backend, project, spec_factory):
    """Given the attention pattern (which doesn't depend on the values), a head's output at the
    last token is the values it reads weighted by its attention: summed over positions, a key/
    value head's values have the direct effect of the query heads that read it."""
    layer = 1
    sites = [
        {"kind": "v", "layer": layer, "head": h, "position": ALL}
        for h in range(kv_heads(tiny_backend))
    ] + [
        {"kind": "head", "layer": layer, "head": h, "position": LAST}
        for h in range(tiny_backend.info.n_heads)
    ]
    spec = spec_factory(experiment=DLA, scope={"kind": "sites", "sites": sites})
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    n_kv = kv_heads(tiny_backend)
    np.testing.assert_allclose(result.delta[:n_kv], result.delta[n_kv:], atol=1e-6)
    # And by position: the value at each named position, through the attention to it.
    by_position = spec_factory(
        experiment=DLA,
        scope={
            "kind": "sites",
            "sites": [
                {
                    "kind": "v",
                    "layer": layer,
                    "head": 0,
                    "position": {"kind": "label", "label": label},
                }
                for label in ("IO", "S1", "S2")
            ],
        },
    )
    parts = run_experiment(by_position, tiny_backend, _prompts(tiny_backend, project))
    assert np.isfinite(parts.delta).all() and np.abs(parts.delta).max() > 0


@pytest.mark.parametrize("kind", ["q", "k"])
def test_queries_and_keys_have_no_direct_effect(tiny_backend, project, spec_factory, kind):
    spec = spec_factory(
        experiment=DLA,
        scope={"kind": "sites", "sites": [{"kind": kind, "layer": 1, "head": 0, "position": LAST}]},
    )
    with pytest.raises(ScopeError, match="no direct effect"):
        run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))


def test_a_neurons_direct_effect_is_read_at_the_last_token(tiny_backend, project, spec_factory):
    spec = spec_factory(
        experiment=DLA,
        scope={
            "kind": "sites",
            "sites": [
                {
                    "kind": "neuron",
                    "layer": 1,
                    "neuron": 0,
                    "position": {"kind": "label", "label": "S2"},
                }
            ],
        },
    )
    with pytest.raises(ScopeError, match="last token"):
        run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))


def test_neurons_without_a_sum_have_no_direct_effect(tiny_backend, project):
    sites, _ = expand_scope(
        _spec({"kind": "neurons", "layers": [1], "position": LAST}, experiment=DLA),
        tiny_backend.info,
        _prompts(tiny_backend, project),
    )
    info = tiny_backend.info
    checks = {**info.extra["checks"], "neuron_writes": 0.4}
    other = dataclasses.replace(info, extra={**info.extra, "checks": checks})
    with pytest.raises(ScopeError, match="sum of what its neurons write"):
        check_direct_sites(sites[:3], other, _prompts(tiny_backend, project))


# -- scopes, sites and what they say -----------------------------------------------------------


@pytest.mark.parametrize(
    ("site", "message"),
    [
        ({"kind": "q", "layer": 1}, "a q site needs a query head index"),
        ({"kind": "k", "layer": 1}, "a k site needs a key/value head index"),
        ({"kind": "neuron", "layer": 1}, "a neuron site needs a neuron index"),
        ({"kind": "neuron", "layer": 1, "neuron": 3, "head": 0}, "a neuron site has no head index"),
        ({"kind": "head", "layer": 1, "head": 0, "neuron": 3}, "a head site has no neuron index"),
        ({"kind": "mlp_out", "layer": 1, "neuron": 3}, "a mlp_out site has no neuron index"),
        ({"kind": "neuron", "layer": 1, "neuron": -1}, "greater than or equal to 0"),
    ],
)
def test_each_new_site_kind_has_its_own_required_field(site, message):
    from pydantic import ValidationError

    from logogram.spec import Site

    with pytest.raises(ValidationError, match=message):
        Site.model_validate({**site, "position": LAST})


def test_new_scopes_and_universes_are_checked_and_described(spec_factory):
    from pydantic import ValidationError

    from logogram.spec import describe_experiment

    with pytest.raises(ValidationError, match="must not repeat"):
        spec_factory(scope={"kind": "neurons", "layers": [1, 1], "position": LAST})
    with pytest.raises(ValidationError, match="at least 1"):
        spec_factory(scope={"kind": "neurons", "layers": [], "position": LAST})
    with pytest.raises(ValidationError, match="input"):
        spec_factory(scope={"kind": "head_inputs", "input": "z", "position": LAST})
    for universe, message in (
        (["neuron", "mlp_out"], "made from its neurons"),
        (["head", "q"], "made from the heads' queries"),
    ):
        with pytest.raises(ValidationError, match=message):
            spec_factory(
                scope={
                    "kind": "site_sets",
                    "universe": universe,
                    "sets": [{"label": "all", "sites": [], "complement": True}],
                }
            )
    keys = spec_factory(scope={"kind": "head_inputs", "input": "k", "position": ALL})
    assert describe_experiment(keys) == "Patch clean → corrupt · every key, all positions"
    neurons = spec_factory(
        experiment={**PATCH, "kind": "attribution_patching", "method": "gradient", "steps": None},
        scope={"kind": "neurons", "layers": [0, 1], "position": LAST},
    )
    assert describe_experiment(neurons) == (
        "Attribution patching, estimated (clean → corrupt) · every MLP neuron of layers 0, 1, "
        "the last token"
    )


def test_older_specs_read_and_run_as_before(spec_factory):
    """A version 3 spec written before these sites existed reads as it was: its sites have no
    neuron, and it writes the same spec again (with the field, as null)."""
    from logogram.spec import Spec

    spec = spec_factory(
        scope={
            "kind": "sites",
            "sites": [{"kind": "head", "layer": 1, "head": 2, "position": LAST}],
        }
    )
    data = spec.model_dump(mode="json")
    for site in data["scope"]["sites"]:
        assert site.pop("neuron") is None  # as a spec saved before neurons existed has it
    again = Spec.model_validate(data)
    assert again == spec
    assert again.scope.sites[0].neuron is None


def test_scopes_lay_out_heads_key_value_heads_and_neurons(arch_backend, tiny_backend, project):
    backend = arch_backend("llama")
    prompts = _prompts(backend, project)
    sites, layout = expand_scope(
        _spec({"kind": "head_inputs", "input": "k", "position": LAST}), backend.info, prompts
    )
    assert layout["kind"] == "head_inputs" and layout["site"] == "k"
    assert layout["col_title"] == "Key/value head"
    assert [c["query_heads"] for c in layout["cols"]] == [[0, 1], [2, 3]]
    assert len(sites) == backend.info.n_layers * 2
    assert sites[3].label == "L1 KV1 k @ last"
    sites, layout = expand_scope(
        _spec({"kind": "head_inputs", "input": "q", "position": ALL}), backend.info, prompts
    )
    assert len(sites) == backend.info.n_layers * backend.info.n_heads
    assert "query_heads" not in layout["cols"][0]
    sites, layout = expand_scope(
        _spec({"kind": "neurons", "layers": [1, 0], "position": LAST}), backend.info, prompts
    )
    d_mlp = backend.info.d_mlp
    assert layout["kind"] == "neurons" and layout["cols"] == []
    assert [r["label"] for r in layout["rows"]] == ["1", "0"]
    assert all(r["neurons"] == d_mlp for r in layout["rows"])
    assert len(sites) == 2 * d_mlp
    assert (sites[0].layer, sites[0].row, sites[0].col, sites[0].site.neuron) == (1, 0, 0, 0)
    assert sites[d_mlp + 5].label == "L0 N5 @ last"
    assert sites[d_mlp + 5].to_dict()["neuron"] == 5


@pytest.mark.parametrize(
    ("site", "message"),
    [
        ({"kind": "q", "layer": 1, "head": 4}, "Head 4 doesn't exist"),
        (
            {"kind": "k", "layer": 1, "head": 2},
            "Key/value head 2 doesn't exist; this model has 2 key/value heads, each read by 2",
        ),
        (
            {"kind": "neuron", "layer": 0, "neuron": 96},
            "Neuron 96 doesn't exist; this model's MLPs have 96",
        ),
    ],
)
def test_sites_beyond_the_model_are_refused(arch_backend, project, site, message):
    backend = arch_backend("llama")
    spec = _spec({"kind": "sites", "sites": [{**site, "position": LAST}]})
    with pytest.raises(ScopeError, match=message):
        expand_scope(spec, backend.info, _prompts(backend, project))
    with pytest.raises(ScopeError, match="layers"):
        expand_scope(
            _spec({"kind": "neurons", "layers": [0, 2], "position": LAST}),
            backend.info,
            _prompts(backend, project),
        )


def test_sets_keep_neurons_and_replace_the_rest(tiny_backend, project, spec_factory):
    """A complement set over a universe of neurons keeps its neurons and replaces every other
    neuron of the model: keeping none replaces them all, which equals replacing every MLP output."""
    universe_neurons = {
        "kind": "site_sets",
        "universe": ["neuron"],
        "sets": [
            {"label": "everything", "sites": [], "complement": True},
            {
                "label": "keep two",
                "sites": [
                    {"kind": "neuron", "layer": 1, "neuron": 3, "position": ALL},
                    {"kind": "neuron", "layer": 0, "neuron": 9, "position": LAST},
                ],
                "complement": True,
            },
        ],
    }
    universe_mlps = {
        "kind": "site_sets",
        "universe": ["mlp_out"],
        "sets": [{"label": "everything", "sites": [], "complement": True}],
    }
    prompts = _prompts(tiny_backend, project)
    neurons = run_experiment(spec_factory(scope=universe_neurons), tiny_backend, prompts)
    mlps = run_experiment(spec_factory(scope=universe_mlps), tiny_backend, prompts)
    np.testing.assert_allclose(neurons.patched[0], mlps.patched[0], atol=TOL)
    assert np.abs(neurons.patched[1] - neurons.patched[0]).max() > 0


def test_a_set_cant_hold_a_component_and_its_parts(tiny_backend, project, spec_factory):
    spec = spec_factory(
        scope=sets_scope(
            both=[
                {"kind": "mlp_out", "layer": 1, "position": ALL},
                {"kind": "neuron", "layer": 1, "neuron": 2, "position": ALL},
            ]
        )
    )
    with pytest.raises(ScopeError, match="made from them"):
        run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))


def test_path_patching_and_steering_name_what_they_take(tiny_backend, project, spec_factory):
    sites = {
        "kind": "sites",
        "sites": [{"kind": "neuron", "layer": 0, "neuron": 1, "position": LAST}],
    }
    paths = spec_factory(
        experiment={
            "kind": "path_patching",
            "direction": "clean_to_corrupt",
            "receivers": [{"kind": "logits"}],
            "freeze_mlps": False,
        },
        scope=sites,
    )
    with pytest.raises(ScopeError, match="part of a head or an MLP"):
        run_experiment(paths, tiny_backend, _prompts(tiny_backend, project))
    steering = spec_factory(
        experiment={
            "kind": "steering",
            "apply_to": "clean",
            "direction": {"kind": "mean_difference"},
            "coefficients": [1.0],
            "train_fraction": 0.5,
            "seed": 0,
            "control": False,
            "continuations": None,
        },
        scope=sites,
    )
    with pytest.raises(ScopeError, match="neurons can't be steered"):
        run_experiment(steering, tiny_backend, _prompts(tiny_backend, project))


# -- reports, exports and the memory estimate --------------------------------------------------

ATP = {**PATCH, "kind": "attribution_patching", "method": "gradient", "steps": None}


def test_reports_and_exports_name_each_neuron(tiny_backend, project, spec_factory):
    from logogram.exports import results_csv, sites_csv
    from logogram.report import figure_svg, run_report, sites_table

    spec = spec_factory(experiment=ATP, scope={"kind": "neurons", "layers": [1], "position": LAST})
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished"
    summary = outcome.summary
    d_mlp = tiny_backend.info.d_mlp
    assert summary["layout"]["kind"] == "neurons"
    assert summary["layout"]["rows"] == [{"key": "1", "label": "1", "neurons": d_mlp}]
    assert [s["neuron"] for s in summary["sites"]] == list(range(d_mlp))
    page = run_report(spec, summary, outcome.manifest, outcome.run_id)
    assert "Every neuron, layer by layer" in page
    assert "All neurons, by mean normalized effect" in page and "Layer 1" in page
    figure = figure_svg(spec, summary)
    assert figure.startswith("<svg") and f"of {d_mlp} neurons measured" in figure
    rows = sites_table(summary)
    assert [r["neuron"] for r in rows] == list(range(d_mlp))
    assert sites_csv(rows).splitlines()[0].split(",")[6] == "neuron"
    header = next(iter(results_csv(outcome.folder / "results.parquet"))).strip().split(",")
    assert "neuron" in header


def test_a_report_of_keys_names_the_query_heads_of_each_group(arch_backend, project, spec_factory):
    from logogram.report import run_report

    backend = arch_backend("llama")
    spec = spec_factory(
        model={"id": "tiny-llama", "revision": "test", "device": "cpu"},
        scope={"kind": "head_inputs", "input": "k", "position": LAST},
    )
    outcome = run_spec(spec, project, backend=backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert outcome.summary["model"]["n_key_value_heads"] == 2
    page = run_report(spec, outcome.summary, outcome.manifest, outcome.run_id)
    assert "Layer × Key/value head" in page
    assert "query heads by key/value head: 0: 0, 1; 1: 2, 3" in page


def test_the_memory_estimate_counts_what_a_scopes_sites_hold(tiny_backend, app_client):
    from logogram.spec import HeadsScope, NeuronsScope
    from logogram.system import estimate_run_memory, scope_footprint, site_bytes

    shape = {
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 12,
        "d_model": 768,
        "d_head": 64,
        "d_mlp": 3072,
        "seq_len": 20,
    }
    neurons = scope_footprint(NeuronsScope(layers=[3, 4], position=LAST), **shape)
    assert (neurons.sites, neurons.width, neurons.neuron_layers) == (6144, 3072, 2)
    heads = scope_footprint(HeadsScope(position=ALL), **shape)
    assert (heads.sites, heads.width, heads.neuron_layers) == (144, 768, 0)
    held = {"prompts": 200, "bootstrap": 1000, "batch_size": 64, "seq_len": 20, "dtype": "float32"}
    # Each site's values per prompt and bootstrap resamples, and a layer of source activations
    # cached for every prompt: 8 · 6,144 · (5 · 200 + 6 · 1,000) + 200 · 20 · 3,072 · 4 bytes.
    assert site_bytes(neurons, method="activation_patching", **held) == 344_064_000 + 49_152_000
    # Attribution patching keeps a float64 product per neuron, token and prompt of a batch, with
    # the activations and gradients it multiplies: 2 layers · 64 · 20 · 3,072 · (8 + 3 · 4).
    assert site_bytes(neurons, method="attribution_patching", **held) == 344_064_000 + 157_286_400
    assert site_bytes(neurons, method="activation_patching", **held) > 10 * site_bytes(
        heads, method="activation_patching", **held
    )
    memory = estimate_run_memory(
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_mlp=3072,
        d_vocab=50257,
        dtype="float32",
        device="cpu",
        batch_size=64,
        seq_len=20,
        all_layers=True,
        footprint=neurons,
        method="attribution_patching",
        prompts=200,
        bootstrap=1000,
    )
    assert memory.sites == 6144 and memory.site_bytes > 0
    assert memory.total > memory.activations + memory.site_bytes
    assert "The 6,144 sites hold about" in memory.explanation
    # Through the server, for the loaded model.
    client, headers = app_client
    body = {"batch_size": 8, "seq_len": 16, "method": "activation_patching", "prompts": 100}
    plain = client.post("/api/models/run-memory", json=body, headers=headers).json()
    scoped = client.post(
        "/api/models/run-memory",
        json={
            **body,
            "bootstrap": 1000,
            "scope": {"kind": "neurons", "layers": [0, 1], "position": LAST},
        },
        headers=headers,
    ).json()
    assert plain["sites"] == 0 and scoped["sites"] == 2 * tiny_backend.info.d_mlp
    assert scoped["total"] > plain["total"]
    bad = {**body, "scope": {"kind": "neurons", "layers": [], "position": LAST}}
    assert client.post("/api/models/run-memory", json=bad, headers=headers).status_code == 422


@pytest.fixture
def app_client(tiny_backend, tmp_path, monkeypatch):
    """The app with the tiny model loaded, and the headers its requests need."""
    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="test-token", port=8765), serve_web=False)
    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer test-token", "origin": "http://127.0.0.1:8765"}
    return client, headers
