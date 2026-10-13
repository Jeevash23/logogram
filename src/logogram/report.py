"""A run as words and as one HTML file that anyone can open, without Logogram.

The methods paragraph says every choice that can change a number in sentences, from the executed
spec and the saved provenance; the app's "Copy methods", the CLI and the Python API all take it
from here. The report holds the run's result map, its strongest sites with their intervals, every
site's numbers, the baseline, the methods paragraph, the spec and where the run came from. It
needs no network and runs no script: styles are inline and the map is SVG. It leaves out the
hardware's model names unless asked, since a report is meant to be passed around.
"""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any

from logogram import __version__
from logogram.schema import stopped_text
from logogram.spec import (
    ACDC,
    Ablation,
    ActivationPatching,
    AttributionPatching,
    DirectLogitAttribution,
    EdgeAttribution,
    EdgesScope,
    FeatureDirection,
    MeanBaseline,
    PathPatching,
    Spec,
    Steering,
    ZeroBaseline,
    describe_experiment,
    describe_metric,
    describe_sae_site,
    receiver_text,
)

STRONGEST = 12


# -- the methods paragraph -------------------------------------------------------------------


def _arrow(direction: str) -> str:
    return (
        "from the clean prompt into the corrupt one"
        if direction == "clean_to_corrupt"
        else "from the corrupt prompt into the clean one"
    )


def _baseline(b: Any) -> str:
    if isinstance(b, ZeroBaseline):
        return "zeros"
    if isinstance(b, MeanBaseline):
        return f"the activation's mean over the {b.reference} prompts"
    return (
        f"the same activation in {b.donors} other {b.pool} prompts drawn without replacement "
        f"(seed {b.seed}), averaged over donors"
    )


def _method(spec: Spec, summary: dict[str, Any] | None) -> str:
    e = spec.experiment
    if isinstance(e, Ablation):
        return f"Each site's activation was replaced by {_baseline(e.baseline)}."
    if isinstance(e, ActivationPatching):
        return f"Each site's activation was copied {_arrow(e.direction)} of each pair."
    if isinstance(e, DirectLogitAttribution):
        text = (
            f"Direct logit attribution of the {e.prompts} prompts: each component's output at the "
            "last token, read through the final normalization with its scale held at its value "
            "in the run."
        )
        direct = (summary or {}).get("direct")
        if direct:

            def f(key: str) -> str:
                value = direct.get(key)
                return "unknown" if value is None else f"{value:.4f}"

            text += (
                f" The mean logit difference was {f('logit_diff')}: attention {f('attention')}, "
                f"MLPs {f('mlp')}, embeddings {f('embeddings')} and biases {f('biases')}."
            )
        return text
    if isinstance(e, PathPatching):
        if isinstance(spec.scope, EdgesScope):
            n = len(spec.scope.edges)
            return (
                f"Path patching {_arrow(e.direction)}, one edge at a time ({n} edges): each edge's "
                "sender was patched while every other attention head and every MLP was held at "
                "its value in the receiver run, so the change reached the edge's receiver only "
                "directly; the receiver's recorded input (a head's query, key or value, an MLP's "
                "input before its normalization, or the final residual stream) was then patched "
                "into an unchanged run."
            )
        receivers = ", ".join(receiver_text(r) for r in e.receivers)
        held = ", and the MLPs were held too" if e.freeze_mlps else ", with the MLPs recomputed"
        return (
            f"Path patching {_arrow(e.direction)}, into {receivers}: each sender was patched while "
            f"every other attention head was held at its value in the receiver run{held}; the "
            "receivers' recorded inputs were then patched into an unchanged run."
        )
    if isinstance(e, EdgeAttribution):
        return _edge_method(spec, e, summary)
    if isinstance(e, ACDC):
        return _acdc_method(spec, e, summary)
    if isinstance(e, Steering):
        return _steering_method(spec, e, summary)
    assert isinstance(e, AttributionPatching)
    how = (
        f"the gradient of the metric averaged over {e.steps} runs whose input embeddings lie "
        "evenly between the receiver's and the source's (integrated gradients)"
        if e.method == "integrated_gradients"
        else "the gradient of the metric at the receiver run"
    )
    return (
        f"Attribution patching {_arrow(e.direction)}: a first-order estimate of each site's "
        f"patching effect, (source − receiver activation) · {how}, not a patched forward pass."
    )


def _edge_method(spec: Spec, e: EdgeAttribution, summary: dict[str, Any] | None) -> str:
    edges = (summary or {}).get("edges") or {}
    size = (
        f"{edges['n_edges']:,} edges ({edges['n_senders']} senders, {edges['n_receivers']} "
        "receivers)"
        if edges
        else "the edges of the scope"
    )
    how = (
        f"the metric's gradient averaged over {e.steps} runs whose input embeddings lie evenly "
        "between the receiver's and the source's (integrated gradients, EAP-IG)"
        if e.method == "integrated_gradients"
        else "the metric's gradient at the receiver run (EAP)"
    )
    text = (
        f"Edge attribution {_arrow(e.direction)} over {size}: the graph's nodes are the "
        "embeddings, every attention head (its output through its slice of the output "
        "projection) and every MLP, and an edge runs from a node's output to a later head's "
        "query, key or value input, a later MLP's input, or the logits. Each edge's estimate is "
        f"(the sender's output on the source prompt − on the receiver prompt) · {how}, summed "
        "over positions, where the gradient is taken with respect to the receiver's own copy of "
        "the residual stream, before its normalization, through that normalization with its "
        "scale recomputed from the copy (not held), as patching the edge recomputes it. It is a "
        "first-order estimate, not a patched forward pass."
    )
    if e.curve is not None:
        c = e.curve
        patched = (
            "the chosen edges carried the source prompt's values and every other edge the "
            "receiver run's own"
            if c.patch == "chosen"
            else "the chosen edges kept the receiver run's values and every other edge carried "
            "the source prompt's"
        )
        text += (
            f" A faithfulness curve then patched edges for real: for each k of "
            f"{', '.join(f'{k:,}' for k in c.counts)}, the receiver prompt ran once with the k "
            f"strongest edges chosen, where {patched}; each patched edge added its sender's "
            "difference (source − this pass) into its receiver's own input only. Faithfulness is "
            "the share of the gap from no edge chosen to every edge chosen that the chosen edges "
            f"close. The control chose k edges at random ({c.control.draws} draw"
            f"{'s' if c.control.draws != 1 else ''}, seed {c.control.seed}); the edges were "
            "chosen and measured on the same prompts."
        )
        curve = edges.get("curve") or {}
        difference = curve.get("source_difference")
        if difference is not None:
            text += (
                f" Patching every edge reproduced the source run's metric within {difference:.2g}."
            )
    return text


def _acdc_method(spec: Spec, e: ACDC, summary: dict[str, Any] | None) -> str:
    from logogram.acdc import ORDER_TEXT

    acdc = (summary or {}).get("acdc") or {}
    size = (
        f"{acdc['n_edges']:,} edges ({acdc['n_senders']} senders, {acdc['n_receivers']} receivers)"
        if acdc
        else "the edges of the scope"
    )
    receiver, source = (
        ("corrupt", "clean") if e.direction == "clean_to_corrupt" else ("clean", "corrupt")
    )
    counted = (
        f"toward the {source} prompts' value"
        if e.rule == "toward_source"
        else "in either direction"
    )
    text = (
        f"ACDC (automated circuit discovery) over {size}: the graph's nodes are the embeddings, "
        "every attention head (its output through its slice of the output projection) and every "
        "MLP, and an edge runs from a node's output to a later head's query, key or value input, "
        "a later MLP's input, or the logits, each receiver reading its own copy of the residual "
        f"stream, before its normalization. The {receiver} prompts ran with every edge in the "
        f"circuit, and the edges were tested one at a time in {acdc.get('order') or ORDER_TEXT}. "
        f"Removing an edge made it carry its sender's output on the {source} prompt into its "
        f"receiver (the sender's difference, {source} − this pass, added into the receiver's own "
        "input), while the edges of the circuit carried what their senders output in that pass. "
        "An edge stayed removed when removing it moved the mean metric over the prompts "
        f"{counted} by less than {e.threshold:g}; otherwise it was put back. Each edge's value is "
        "the change its removal made when it was tested."
    )
    if acdc:
        f = acdc.get("faithfulness") or {}

        def value(key: str) -> str:
            v = f.get(key)
            return "undefined" if v is None else f"{v:.3f}"

        level = round(spec.statistics.ci * 100)
        text += (
            f" The circuit kept {acdc['n_kept']:,} of the {acdc['n_edges']:,} edges. Its "
            f"faithfulness, the share of the gap from the {source} run (0) to the {receiver} run "
            f"with every edge (1) that it keeps, was {value('mean')} ({level}% interval "
            f"{value('lo')} to {value('hi')})."
        )
    return text


def _steering_method(spec: Spec, e: Steering, summary: dict[str, Any] | None) -> str:
    other = "corrupt" if e.apply_to == "clean" else "clean"
    strengths = ", ".join(f"{c:g}" for c in e.coefficients)
    steering = (summary or {}).get("steering") or {}
    split = (
        f"a training split of {round(e.train_fraction * 100)}% of the pairs (split seed {e.seed})"
    )
    if isinstance(e.direction, FeatureDirection):
        f = e.direction.feature
        sae = spec.sae
        named = (
            f" of the SAE {sae.repo}{'/' + sae.path if sae.path else ''}"
            f"{f' at revision {sae.revision}' if sae.revision else ''}"
            if sae
            else ""
        )
        feature = steering.get("feature") or {}
        where = (
            f", {describe_sae_site(feature['site'], feature['layer'], feature['transcoder'])}"
            if feature
            else ""
        )
        text = (
            f"Steering along an SAE feature: the decoder direction W_dec[{f}] of feature {f}"
            f"{named} was added to the {e.apply_to} prompts where the SAE writes{where}, at "
            f"strengths {strengths}. Adding W_dec[{f}] times a is what the feature writes when "
            "its activation is a, so strength 1 is the feature's typical activation: the mean of "
            "its positive activations at the steered token of the clean and corrupt prompts of "
            f"{split}."
        )
        if feature.get("standardized"):
            text += (
                " The SAE standardizes its inputs, so each prompt's vector was also multiplied by "
                "the standard deviation of its own activation at that token, as the SAE's "
                "decoding does."
            )
        values = feature.get("activations") or []
        if values:
            rows = (summary or {}).get("layout", {}).get("rows") or []
            parts = []
            for i, value in enumerate(values):
                fired = (feature.get("fired") or [None])[i]
                measured = (feature.get("measured") or [None])[i]
                at = f" at {rows[i]['label']}" if len(values) > 1 and i < len(rows) else ""
                parts.append(
                    f"{value:.4g}{at} (positive in {fired} of the {measured} training activations)"
                )
            text += f" The typical activation was {'; '.join(parts)}."
    else:
        text = (
            f"Steering: a direction was added to the {e.apply_to} prompts, the mean of ({other} − "
            f"{e.apply_to}) activations at each site over {split}, at strengths {strengths}."
        )
    text += (
        " A random direction of the same length as the steering vector at strength 1 was run as a "
        "control."
        if e.control
        else " No random control was run."
    )
    if steering:
        train = ", ".join(str(i) for i in steering.get("train", []))
        what = (
            "The typical activation"
            if isinstance(e.direction, FeatureDirection)
            else "The directions"
        )
        text += (
            f" {what} came from dataset prompts {train}; effects were measured on the "
            f"{len(steering.get('test', []))} held-out prompts."
        )
    if e.continuations is not None:
        c = e.continuations
        text += (
            f" Greedy continuations of {c.tokens} new token{'s' if c.tokens != 1 else ''} (the "
            "most likely token each time, ties to the lowest token id) were recorded for the "
            f"first {c.prompts} held-out prompt{'s' if c.prompts != 1 else ''}: without steering, "
            f"at each strength{' and with the control' if e.control else ''}. Each forward pass "
            "added the vector at the steered site and token of the prompt, as when it was "
            "measured; the generated tokens weren't steered themselves."
        )
    return text


def _statistics(spec: Spec, summary: dict[str, Any] | None) -> str:
    st = spec.statistics
    stats = (summary or {}).get("statistics") or {}
    if st.cluster:
        clusters = stats.get("clusters")
        counted = f" ({clusters} clusters)" if clusters else ""
        over = f"over clusters of prompts with the same {st.cluster}{counted}"
    else:
        over = "over prompts"
    text = (
        f"Intervals are {st.ci * 100:g}% percentile bootstrap intervals {over}, from "
        f"{st.bootstrap} resamples with seed {st.seed}."
    )
    if stats.get("multiple_comparisons"):
        text += f" {stats['multiple_comparisons']}"
    return text


def _quoted(text: str) -> str:
    """Text as JSON writes it: in quotes, with its newlines and other controls escaped."""
    return json.dumps(text, ensure_ascii=False)


def _tokenized(spec: Spec, summary: dict[str, Any] | None) -> str:
    """How the prompts were tokenized: with or without a beginning-of-sequence token, or wrapped
    in the model's own chat template, and then exactly as what text."""
    template = spec.tokenization.chat_template
    if template is None:
        bos = "with" if spec.tokenization.prepend_bos else "without"
        return f"tokenized {bos} a beginning-of-sequence token"
    chat = (summary or {}).get("chat_template") or {}
    sha = f" (SHA-256 {chat['sha256']})" if chat.get("sha256") else ""
    system = (
        f", after the system message {_quoted(template.system)}"
        if template.system is not None
        else ""
    )
    if template.kind == "user_turn":
        text = (
            f"wrapped in the model's own chat template{sha} as the person's message{system}, "
            "followed by the template's opening of the model's reply, so the answer is the "
            "reply's first token"
        )
    else:
        text = (
            f"wrapped in the model's own chat template{sha} as the start of the model's reply "
            f"to the person's message {_quoted(template.user)}{system}, so the model goes on "
            "from the prompt"
        )
    if "before" in chat:
        after = f" + {_quoted(chat['after'])}" if chat.get("after") else ""
        text += f": each prompt was read as {_quoted(chat['before'])} + the prompt{after}"
    return text + ", tokenized whole without adding special tokens"


def methods_text(
    spec: Spec, summary: dict[str, Any] | None, manifest: dict[str, Any] | None, run_id: str
) -> str:
    """A copyable methods paragraph: every choice that can change a number, in sentences."""
    manifest = manifest or {}
    n = (summary or {}).get("n_prompts")
    m = spec.model
    revision = (
        f"revision {m.revision}"
        if m.revision
        else "an unpinned revision (the main branch when it ran)"
    )
    device = (manifest.get("device") or {}).get("type") or m.device
    weights = (
        "with TransformerLens's weight processing (LayerNorm weights folded, writing weights and "
        "the unembedding centered), which leaves logit differences unchanged"
        if m.process_weights
        else "with its original weights, unprocessed"
    )
    d = spec.dataset
    # A run reads a snapshot of its dataset; say where the prompts came from.
    source = (manifest.get("dataset") or {}).get("source_path") or d.path
    pairs = (
        "The prompt pairs were" if n is None else f"{n} prompt pair{' was' if n == 1 else 's were'}"
    )
    pinned = f" (SHA-256 {d.sha256})" if d.sha256 else " (not pinned by a checksum)"
    first = f", the first {d.limit} of the file" if d.limit else ""
    normalized = (
        "Effects were normalized by the mean clean–corrupt gap over the dataset"
        if spec.metric.normalization == "dataset_gap"
        else "Effects were normalized by each prompt's own clean–corrupt gap"
    )
    formula = ((summary or {}).get("metric") or {}).get("normalized_effect")
    model = (
        f"{m.id}, a local model whose files hash to SHA-256 {m.sha256} (Logogram's hash of the "
        "files it loads)"
        if m.source == "local"
        else f"{m.id} at {revision}"
    )
    paragraphs = [
        f"{describe_experiment(spec)}.",
        f"The model was {model}, run in {m.dtype} on {device}, {weights}.",
        f"{pairs} read from {source}{pinned}{first}, and {_tokenized(spec, summary)}.",
        f"The metric, for each prompt, was {describe_metric(spec.metric)}. "
        f"{normalized}{f': {formula}' if formula else ''}.",
        f"Prompts ran in batches of {spec.execution.batch_size}. {_statistics(spec, summary)}",
        _method(spec, summary),
    ]
    # A run that stopped early: how far it got, and what its numbers cover.
    partial = (summary or {}).get("partial")
    if partial:
        ended = "failed" if partial.get("status") == "failed" else "was cancelled"
        paragraphs.append(
            f"The run {ended} before it finished. {partial['note']} Their statistics are computed "
            "as for a finished run. " + " ".join(partial.get("left_out") or [])
        )
    # What was expected, as the spec that ran states it, so the result can be read against it.
    if spec.notes.strip():
        paragraphs.append(f"Notes saved with the spec: {spec.notes.strip()}")
    versions = manifest.get("versions") or {}
    software = ", ".join(f"{k} {v}" for k, v in versions.items() if v)
    paragraphs.append(f"The run is {run_id}." + (f" Software: {software}." if software else ""))
    paragraphs += [f"Note from the run: {w}" for w in (summary or {}).get("warnings", [])]
    return "\n\n".join(paragraphs)


# -- the map ---------------------------------------------------------------------------------

# The effect scale, symmetric around zero: cobalt for positive, amber for negative, as in the app.
_POSITIVE = (37, 87, 222)
_NEGATIVE = (200, 120, 20)
_NEUTRAL = (236, 236, 236)


def effect_color(value: float | None, bound: float) -> str:
    if value is None or not math.isfinite(value):
        return "#ffffff"
    t = max(-1.0, min(1.0, value / bound)) if bound > 0 else 0.0
    end = _POSITIVE if t >= 0 else _NEGATIVE
    mix = [round(n + abs(t) * (e - n)) for n, e in zip(_NEUTRAL, end, strict=True)]
    return "#" + "".join(f"{c:02x}" for c in mix)


def _fmt(x: float | None, digits: int = 3, sign: bool = True) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return text.replace("-", "−")


def heatmap_svg(
    summary: dict[str, Any],
    cell: int = 18,
    labels: bool = True,
    value: str = "effect",
    unit: bool = False,
) -> str:
    """A run's means (the normalized ``effect``, or the change in the metric, ``delta``) as an SVG
    grid in the effect colors on a symmetric scale, fitted to the largest value or fixed at ±1
    (``unit``, effects only), with a legend, axis labels and each cell's value on hover. A run
    over every neuron of some layers has too many sites for a grid: see :func:`neurons_svg`. An
    edge attribution run has no grid: its strongest edges are drawn as a graph instead."""
    layout = summary["layout"]
    if layout.get("kind") == "neurons":
        return neurons_svg(summary, value=value, unit=unit)
    if layout.get("kind") == "edges":
        return edge_graph_svg(summary)
    rows, cols = layout["rows"], layout["cols"]
    # Small maps get bigger cells, so a few heads don't make a postage stamp.
    cell = max(cell, min(36, 480 // max(1, len(cols)), 480 // max(1, len(rows))))
    by_cell = {(s["row"], s["col"]): s for s in summary["sites"]}
    finite = [
        abs(s[value]["mean"])
        for s in summary["sites"]
        if s[value]["mean"] is not None and math.isfinite(s[value]["mean"])
    ]
    bound = 1.0 if unit and value == "effect" else (max(finite) if finite else 1.0)
    what = "mean normalized effect" if value == "effect" else "mean change in the metric"
    left = 72
    top = 64 if labels else 20
    # Wide enough for the legend and its label, however few the columns.
    width = left + max(cell * len(cols), 300) + 16
    height = top + cell * len(rows) + 54
    esc = html.escape
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="10" '
        f'role="img" aria-label="{esc(layout["row_title"])} by {esc(layout["col_title"] or "site")}: '
        f'{what}">'
    ]
    if labels:
        for c, col in enumerate(cols):
            x = left + c * cell + cell / 2
            parts.append(
                f'<text x="{x:.1f}" y="{top - 6}" transform="rotate(-50 {x:.1f} {top - 6})" '
                f'fill="currentColor">{esc(str(col.get("label", c))[:14])}</text>'
            )
    for r, row in enumerate(rows):
        parts.append(
            f'<text x="{left - 6}" y="{top + r * cell + cell * 0.68:.1f}" text-anchor="end" '
            f'fill="currentColor">{esc(str(row.get("label", r))[:12])}</text>'
        )
        for c in range(len(cols)):
            site = by_cell.get((r, c))
            mean = site[value]["mean"] if site else None
            fill = effect_color(mean, bound) if site else "none"
            title = (
                f"<title>{esc(site['label'])}: {_fmt(mean)} "
                f"[{_fmt(site[value]['lo'])} to {_fmt(site[value]['hi'])}]</title>"
                if site
                else ""
            )
            parts.append(
                f'<rect x="{left + c * cell}" y="{top + r * cell}" width="{cell - 1}" '
                f'height="{cell - 1}" fill="{fill}">{title}</rect>'
            )
    # The legend: the scale from −bound to +bound.
    parts.append(_legend(left, top + cell * len(rows) + 18, bound, what))
    parts.append("</svg>")
    return "".join(parts)


def _legend(left: int, y: int, bound: float, what: str) -> str:
    """The effect scale from −bound to +bound, with what it measures."""
    return (
        '<defs><linearGradient id="effect-scale">'
        f'<stop offset="0" stop-color="{effect_color(-1, 1)}"/>'
        f'<stop offset="0.5" stop-color="{effect_color(0, 1)}"/>'
        f'<stop offset="1" stop-color="{effect_color(1, 1)}"/></linearGradient></defs>'
        f'<rect x="{left}" y="{y}" width="120" height="8" fill="url(#effect-scale)"/>'
        f'<text x="{left}" y="{y + 22}" fill="currentColor">{_fmt(-bound, 2)}</text>'
        f'<text x="{left + 60}" y="{y + 22}" text-anchor="middle" fill="currentColor">0</text>'
        f'<text x="{left + 120}" y="{y + 22}" text-anchor="end" fill="currentColor">'
        f"{_fmt(bound, 2)}</text>"
        f'<text x="{left + 132}" y="{y + 8}" fill="currentColor">{what}</text>'
    )


# Per layer of a neurons run: how many of its strongest neurons the figure lists, and how many
# bins its distribution has.
NEURONS_LISTED = 6
NEURON_BINS = 48


def neurons_svg(summary: dict[str, Any], value: str = "effect", unit: bool = False) -> str:
    """A run over every neuron of some layers, layer by layer: the distribution of all its
    neurons' means (a histogram, each bar in the effect color of its middle, its height the
    number of neurons there on a square-root scale so a few strong neurons stay visible beside
    thousands near zero), and its strongest neurons with their intervals."""
    layout = summary["layout"]
    rows = layout["rows"]
    by_row: dict[int, list[dict[str, Any]]] = {}
    for site in summary["sites"]:
        by_row.setdefault(site["row"], []).append(site)

    def mean(site: dict[str, Any]) -> float | None:
        x = site[value]["mean"]
        return x if x is not None and math.isfinite(x) else None

    finite = [abs(m) for site in summary["sites"] if (m := mean(site)) is not None]
    bound = 1.0 if unit and value == "effect" else (max(finite) if finite else 1.0)
    bound = bound or 1.0
    what = "mean normalized effect" if value == "effect" else "mean change in the metric"

    def strongest_of(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            (site for site in sites if mean(site) is not None),
            key=lambda site: (-abs(mean(site) or 0.0), site["index"]),
        )[:NEURONS_LISTED]

    # The listed neurons' intervals share one track, wide enough for every one of them.
    listed = [site for r in range(len(rows)) for site in strongest_of(by_row.get(r, []))]
    reach = [
        abs(x)
        for site in listed
        for x in (site[value]["lo"], site[value]["hi"], mean(site))
        if x is not None and math.isfinite(x)
    ]
    track = max(reach) if reach else bound
    esc = html.escape
    left, hist_w, hist_h, gap = 72, 300, 64, 28
    list_x = left + hist_w + 36
    track_x, track_w = list_x + 92, 200
    row_h = max(hist_h, NEURONS_LISTED * 13) + gap
    top = 22
    width = track_x + track_w + 70
    height = top + row_h * len(rows) + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="10" '
        f'role="img" aria-label="Every neuron of {len(rows)} layer{"s" if len(rows) != 1 else ""}:'
        f' {what}, layer by layer">',
        f'<text x="{left}" y="{top - 8}" fill="currentColor">All neurons, by {what}</text>',
        f'<text x="{list_x}" y="{top - 8}" fill="currentColor">The strongest, with their '
        "intervals</text>",
    ]
    for r, row in enumerate(rows):
        y0 = top + r * row_h
        sites = by_row.get(r, [])
        means = [m for site in sites if (m := mean(site)) is not None]
        parts.append(
            f'<text x="{left - 8}" y="{y0 + hist_h / 2:.1f}" text-anchor="end" '
            f'fill="currentColor">Layer {esc(str(row.get("label", r)))}</text>'
        )
        # The distribution: bins over −bound..+bound, bars on a square-root scale.
        counts = [0] * NEURON_BINS
        for m in means:
            b = int((max(-bound, min(bound, m)) + bound) / (2 * bound) * NEURON_BINS)
            counts[min(NEURON_BINS - 1, b)] += 1
        tallest = math.sqrt(max(counts) or 1)
        bin_w = hist_w / NEURON_BINS
        for b, count in enumerate(counts):
            if not count:
                continue
            h = max(1.0, math.sqrt(count) / tallest * hist_h)
            middle = -bound + (b + 0.5) * 2 * bound / NEURON_BINS
            parts.append(
                f'<rect x="{left + b * bin_w:.1f}" y="{y0 + hist_h - h:.1f}" '
                f'width="{max(1.0, bin_w - 0.5):.1f}" height="{h:.1f}" '
                f'fill="{effect_color(middle, bound)}"><title>{count} neuron'
                f"{'s' if count != 1 else ''} between {_fmt(middle - bound / NEURON_BINS)} and "
                f"{_fmt(middle + bound / NEURON_BINS)}</title></rect>"
            )
        zero_x = left + hist_w / 2
        parts.append(
            f'<line x1="{zero_x}" y1="{y0}" x2="{zero_x}" y2="{y0 + hist_h}" '
            'stroke="currentColor" stroke-width="0.5"/>'
            f'<line x1="{left}" y1="{y0 + hist_h}" x2="{left + hist_w}" y2="{y0 + hist_h}" '
            'stroke="currentColor" stroke-width="0.5"/>'
            f'<text x="{left}" y="{y0 + hist_h + 12}" fill="currentColor">'
            f"{len(means):,} of {len(sites):,} neurons measured</text>"
        )
        # The strongest neurons, each with its interval on a shared track.
        strongest = strongest_of(sites)

        def at(x: float) -> float:
            return track_x + (max(-track, min(track, x)) + track) / (2 * track) * track_w

        parts.append(
            f'<line x1="{at(0):.1f}" y1="{y0}" x2="{at(0):.1f}" '
            f'y2="{y0 + NEURONS_LISTED * 13}" stroke="currentColor" stroke-width="0.5"/>'
        )
        for i, site in enumerate(strongest):
            y = y0 + i * 13 + 9
            m = mean(site) or 0.0
            lo, hi = site[value]["lo"], site[value]["hi"]
            label = f"N{site.get('neuron')}" if site.get("neuron") is not None else site["label"]
            parts.append(
                f'<text x="{list_x}" y="{y}" fill="currentColor">{esc(label)}</text>'
                f'<text x="{track_x - 6}" y="{y}" text-anchor="end" fill="currentColor">'
                f"{_fmt(m, 2)}</text>"
            )
            if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
                parts.append(
                    f'<line x1="{at(lo):.1f}" y1="{y - 3}" x2="{at(hi):.1f}" y2="{y - 3}" '
                    'stroke="currentColor" stroke-width="1.5"/>'
                )
            parts.append(
                f'<circle cx="{at(m):.1f}" cy="{y - 3}" r="3" fill="{effect_color(m, bound)}" '
                f'stroke="currentColor" stroke-width="0.75"><title>{esc(site["label"])}: '
                f"{_fmt(m)} [{_fmt(lo)} to {_fmt(hi)}]</title></circle>"
            )
    # The track's ends, under the last layer's list.
    end_y = top + row_h * (len(rows) - 1) + NEURONS_LISTED * 13 + 12
    parts.append(
        f'<text x="{track_x}" y="{end_y}" fill="currentColor">{_fmt(-track, 2)}</text>'
        f'<text x="{track_x + track_w / 2:.1f}" y="{end_y}" text-anchor="middle" '
        'fill="currentColor">0</text>'
        f'<text x="{track_x + track_w}" y="{end_y}" text-anchor="end" fill="currentColor">'
        f"{_fmt(track, 2)}</text>"
    )
    parts.append(_legend(left, top + row_h * len(rows) + 4, bound, what))
    parts.append("</svg>")
    return "".join(parts)


# -- edges -----------------------------------------------------------------------------------

# How many of the strongest edges the graph drawing shows.
GRAPH_EDGES = 60


def _node_place(node: dict[str, Any], n_layers: int) -> tuple[int, float | None]:
    """A node's column (the embeddings first, then each layer's heads and MLP, the logits last)
    and its head (None for a node drawn in the middle of its column)."""
    kind = node.get("kind")
    if kind == "embed":
        return 0, None
    if kind == "logits":
        return 2 * n_layers + 1, None
    layer = int(node.get("layer") or 0)
    if kind == "mlp":
        return 2 + 2 * layer, None
    return 1 + 2 * layer, float(node.get("head") or 0)


def edge_graph_svg(
    summary: dict[str, Any], top: int = GRAPH_EDGES, only: list[int] | None = None
) -> str:
    """The strongest edges drawn between the model's nodes, layers left to right: each edge in
    the effect colors on a symmetric scale (with a legend), wider when stronger, the strongest on
    top where edges cross. A head's query, key and value inputs end at the top, middle and bottom
    of it. ``only``: the edges that may be drawn (an ACDC run's circuit), by index."""
    model = summary.get("model") or {}
    sites = [s for s in summary["sites"] if s.get("edge") and s["effect"]["mean"] is not None]
    if only is not None:
        chosen = set(only)
        sites = [s for s in sites if s["index"] in chosen]
    strongest = sorted(sites, key=lambda s: (-abs(s["effect"]["mean"]), s["index"]))[:top]
    n_layers = int(model.get("n_layers") or 1)
    n_heads = int(model.get("n_heads") or 1)
    # With grouped-query attention a key or value is a key/value head's, read by every query head
    # of its group: its edges end in the middle of the group.
    group = max(1, n_heads // int(model.get("n_key_value_heads") or n_heads))
    columns = 2 * n_layers + 2
    left, right, top_pad = 40, 24, 16
    width = max(640, 36 * columns + left + right)
    rows_h = max(180, 16 * n_heads)
    height = top_pad + rows_h + 70
    step = (width - left - right) / max(1, columns - 1)
    bound = max((abs(s["effect"]["mean"]) for s in strongest), default=1.0) or 1.0
    esc = html.escape
    legend = (
        "normalized change in the metric when ACDC removed the edge"
        if summary.get("acdc")
        else "estimated normalized effect of patching the edge alone"
    )

    def place(node: dict[str, Any], inp: str | None = None) -> tuple[float, float]:
        col, head = _node_place(node, n_layers)
        y = top_pad + rows_h / 2 if head is None else top_pad + (head + 0.5) * rows_h / n_heads
        y += {"q": -3.0, "k": 0.0, "v": 3.0}.get(inp or "", 0.0)
        return left + col * step, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="10" '
        f'role="img" aria-label="The {len(strongest)} strongest edges, layers left to right">'
    ]
    # An SVG paints in order: the weakest first, so the strongest are on top where edges cross.
    for s in reversed(strongest):
        e = s["edge"]
        receiver = e["receiver"]
        x1, y1 = place(e["sender"])
        shared = receiver.get("input") in ("k", "v") and group > 1
        x2, y2 = place(
            {"kind": "logits"}
            if receiver["kind"] == "logits"
            else {"kind": "mlp", "layer": receiver["layer"]}
            if receiver["kind"] == "mlp"
            else {
                "kind": "head",
                "layer": receiver["layer"],
                "head": receiver["head"] * group + (group - 1) / 2 if shared else receiver["head"],
            },
            receiver.get("input"),
        )
        mean = s["effect"]["mean"]
        bend = (x2 - x1) * 0.45
        parts.append(
            f'<path d="M{x1:.1f} {y1:.1f} C{x1 + bend:.1f} {y1:.1f} {x2 - bend:.1f} {y2:.1f} '
            f'{x2:.1f} {y2:.1f}" fill="none" stroke="{effect_color(mean, bound)}" '
            f'stroke-width="{1 + 3 * abs(mean) / bound:.2f}" stroke-linecap="round">'
            f"<title>{esc(s['label'])}: {_fmt(mean)} [{_fmt(s['effect']['lo'])} to "
            f"{_fmt(s['effect']['hi'])}]</title></path>"
        )
    # The nodes, over the edges: heads as dots, MLPs as squares, and the two ends of the model.
    for layer in range(n_layers):
        x = left + (1 + 2 * layer) * step
        for h in range(n_heads):
            y = top_pad + (h + 0.5) * rows_h / n_heads
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="currentColor"/>')
        xm, ym = left + (2 + 2 * layer) * step, top_pad + rows_h / 2
        parts.append(
            f'<rect x="{xm - 3:.1f}" y="{ym - 3:.1f}" width="6" height="6" fill="currentColor"/>'
        )
        parts.append(
            f'<text x="{(x + xm) / 2:.1f}" y="{top_pad + rows_h + 14}" text-anchor="middle" '
            f'fill="currentColor">L{layer}</text>'
        )
    for col, name in ((0, "embeddings"), (columns - 1, "logits")):
        x, y = left + col * step, top_pad + rows_h / 2
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="currentColor"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top_pad + rows_h + 14}" text-anchor="middle" '
            f'fill="currentColor">{name}</text>'
        )
    ly = top_pad + rows_h + 34
    parts.append(
        '<defs><linearGradient id="edge-scale">'
        f'<stop offset="0" stop-color="{effect_color(-1, 1)}"/>'
        f'<stop offset="0.5" stop-color="{effect_color(0, 1)}"/>'
        f'<stop offset="1" stop-color="{effect_color(1, 1)}"/></linearGradient></defs>'
        f'<rect x="{left}" y="{ly}" width="120" height="8" fill="url(#edge-scale)"/>'
        f'<text x="{left}" y="{ly + 22}" fill="currentColor">{_fmt(-bound, 2)}</text>'
        f'<text x="{left + 60}" y="{ly + 22}" text-anchor="middle" fill="currentColor">0</text>'
        f'<text x="{left + 120}" y="{ly + 22}" text-anchor="end" fill="currentColor">'
        f"{_fmt(bound, 2)}</text>"
        f'<text x="{left + 132}" y="{ly + 8}" fill="currentColor">{legend}</text></svg>'
    )
    return "".join(parts)


def curve_svg(curve: dict[str, Any], n_edges: int) -> str:
    """Faithfulness against the number of edges chosen (on a log scale), for the strongest edges
    (in ink) and for edges chosen at random (dashed), each with its interval as a band."""
    points = [(p, q) for p, q in zip(curve["strongest"], curve["random"], strict=True)]
    if not points:
        return ""
    width, height = 640, 300
    left, right, top, bottom = 48, 20, 14, 46
    ks = [p["k"] for p, _ in points] + ([n_edges] if curve["patch"] == "chosen" else [])
    lo_k, hi_k = math.log10(min(ks)), math.log10(max(max(ks), min(ks) * 10))
    values = [0.0, 1.0]
    for p, q in points:
        for f in (p["faithfulness"], q["faithfulness"]):
            values += [v for v in (f["lo"], f["hi"], f["mean"]) if v is not None]
    y_lo, y_hi = min(values), max(values)
    pad = (y_hi - y_lo) * 0.06 or 0.1
    y_lo, y_hi = y_lo - pad, y_hi + pad

    def x(k: float) -> float:
        return left + (math.log10(k) - lo_k) / (hi_k - lo_k or 1) * (width - left - right)

    def y(v: float) -> float:
        return top + (y_hi - v) / (y_hi - y_lo) * (height - top - bottom)

    def series(which: int, dashed: bool, color: str) -> str:
        pts = [(pq[0]["k"], pq[which]["faithfulness"]) for pq in points]
        band = [(k, f) for k, f in pts if f["lo"] is not None and f["hi"] is not None]
        out = ""
        if len(band) >= 2:
            upper = " ".join(f"{x(k):.1f},{y(f['hi']):.1f}" for k, f in band)
            lower = " ".join(f"{x(k):.1f},{y(f['lo']):.1f}" for k, f in reversed(band))
            out += f'<polygon points="{upper} {lower}" fill="{color}" fill-opacity="0.12"/>'
        line = " ".join(f"{x(k):.1f},{y(f['mean']):.1f}" for k, f in pts if f["mean"] is not None)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out += f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6"{dash}/>'
        for k, f in pts:
            if f["mean"] is not None:
                out += (
                    f'<circle cx="{x(k):.1f}" cy="{y(f["mean"]):.1f}" r="2.4" fill="{color}">'
                    f"<title>{k:,} edges: {_fmt(f['mean'])} [{_fmt(f['lo'])} to "
                    f"{_fmt(f['hi'])}]</title></circle>"
                )
        return out

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif" font-size="10" '
        'role="img" aria-label="Faithfulness against the number of edges">'
    ]
    for v in (0.0, 1.0):
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
            f'stroke="currentColor" stroke-opacity="0.25"/>'
            f'<text x="{left - 6}" y="{y(v) + 3:.1f}" text-anchor="end" fill="currentColor">'
            f"{v:g}</text>"
        )
    decade = math.floor(lo_k)
    while decade <= hi_k:
        k = 10**decade
        if lo_k - 1e-9 <= decade:
            parts.append(
                f'<text x="{x(k):.1f}" y="{height - bottom + 16}" text-anchor="middle" '
                f'fill="currentColor">{k:,}</text>'
            )
        decade += 1
    parts.append(series(1, True, "#8e8e96"))
    parts.append(series(0, False, "currentColor"))
    every = curve.get("every") or {}
    if curve["patch"] == "chosen" and (every.get("faithfulness") or {}).get("mean") is not None:
        mean = every["faithfulness"]["mean"]
        parts.append(
            f'<circle cx="{x(n_edges):.1f}" cy="{y(mean):.1f}" r="3.2" fill="none" '
            f'stroke="currentColor"><title>Every edge: {_fmt(mean)}</title></circle>'
        )
    parts.append(
        f'<text x="{(left + width - right) / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        'fill="currentColor">edges chosen (log scale)</text></svg>'
    )
    return "".join(parts)


def _acdc_html(summary: dict[str, Any], ci_level: int) -> str:
    """An ACDC run's circuit, drawn, with how many edges it kept and how faithful it is."""
    acdc = summary["acdc"]
    f = acdc.get("faithfulness") or {}
    drawn = min(GRAPH_EDGES, acdc["n_kept"])
    some = f"the {drawn} with the largest changes" if drawn < acdc["n_kept"] else "each"
    return (
        "<section><h2>The circuit ACDC found, layers left to right</h2><figure>"
        f"{edge_graph_svg(summary, only=acdc['kept'])}<figcaption>The {acdc['n_kept']:,} edges "
        f"ACDC kept of {acdc['n_edges']:,}, {some} drawn in the color of the change its removal "
        f"made when it was tested. Hover an edge for its {ci_level}% interval.</figcaption>"
        "</figure><p class='lede'>The circuit's faithfulness, the share of the gap from the "
        f"{summary['reference']} run (0) to the {summary['receiver']} run with every edge (1) "
        f"that it keeps: {_fmt(f.get('mean'), sign=False)} ({ci_level}% interval "
        f"{_fmt(f.get('lo'), sign=False)} to {_fmt(f.get('hi'), sign=False)}).</p></section>"
    )


def _edges_html(summary: dict[str, Any], ci_level: int, verification: dict[str, Any] | None) -> str:
    """An edge attribution run's graph, its faithfulness curve, and its estimates against path
    patching when a verification ran; an ACDC run's circuit."""
    esc = html.escape
    if summary.get("acdc"):
        return _acdc_html(summary, ci_level)
    edges = summary.get("edges") or {}
    n_edges = edges.get("n_edges", len(summary["sites"]))
    out = (
        f"<section><h2>The strongest edges, layers left to right</h2><figure>"
        f"{edge_graph_svg(summary)}<figcaption>The {min(GRAPH_EDGES, len(summary['sites']))} "
        f"strongest of {n_edges:,} edges. Hover an edge for its {ci_level}% interval."
        "</figcaption></figure></section>"
    )
    curve = edges.get("curve")
    if curve:
        rows = "".join(
            f"<tr><td class='num'>{p['k']:,}</td>"
            f"<td class='num'>{_fmt(p['faithfulness']['mean'])}</td>"
            f"<td class='num'>{_fmt(p['faithfulness']['lo'])} to "
            f"{_fmt(p['faithfulness']['hi'])}</td>"
            f"<td class='num'>{_fmt(q['faithfulness']['mean'])}</td>"
            f"<td class='num'>{_fmt(q['faithfulness']['lo'])} to "
            f"{_fmt(q['faithfulness']['hi'])}</td></tr>"
            for p, q in zip(curve["strongest"], curve["random"], strict=True)
        )
        patched = (
            "the chosen edges carry the source prompt's values"
            if curve["patch"] == "chosen"
            else "every edge but the chosen ones carries the source prompt's values"
        )
        out += (
            f"<section><h2>How much of the behavior the strongest edges carry</h2><figure>"
            f"{curve_svg(curve, n_edges)}<figcaption>Faithfulness of the k strongest edges (ink) "
            f"and of k edges chosen at random (dashed), with {ci_level}% intervals: {patched}. 0 "
            "is no edge chosen, 1 every edge.</figcaption></figure>"
            "<table><thead><tr><th class='num'>Edges</th><th class='num'>Strongest</th>"
            f"<th class='num'>{ci_level}% interval</th><th class='num'>Random</th>"
            f"<th class='num'>{ci_level}% interval</th></tr></thead><tbody>{rows}</tbody></table>"
            "</section>"
        )
    if verification and verification.get("rows"):
        rho = verification.get("spearman")
        rows = "".join(
            f"<tr><td>{esc(r['label'])}</td>"
            f"<td class='num'>{_fmt(r['estimate']['mean'])}</td>"
            f"<td class='num'>{_fmt(r['measured']['mean'])}</td>"
            f"<td class='num'>{_fmt(r['measured']['lo'])} to {_fmt(r['measured']['hi'])}</td>"
            "</tr>"
            for r in verification["rows"]
        )
        out += (
            "<section><h2>Estimates against path patching</h2><p class='lede'>"
            f"Run {esc(verification['run'])} path patched the {verification['n']} strongest edges "
            "one at a time. Spearman's rank correlation of estimate and measurement: "
            f"{'—' if rho is None else _fmt(rho, 3, sign=False)}; "
            f"{verification['same_sign']} of {verification['n']} have the same sign.</p>"
            "<table><thead><tr><th>Edge</th><th class='num'>Estimate</th>"
            f"<th class='num'>Path patching</th><th class='num'>{ci_level}% interval</th></tr>"
            f"</thead><tbody>{rows}</tbody></table></section>"
        )
    return out


def figure_svg(
    spec: Spec, summary: dict[str, Any], value: str = "effect", unit: bool = False
) -> str:
    """The run's map as a standalone vector figure on white, titled with what was run: for a
    paper or a slide."""
    inner = heatmap_svg(summary, cell=22, value=value, unit=unit)
    match = re.match(r'<svg [^>]*width="(\d+)" height="(\d+)"', inner)
    width, height = (int(match.group(1)), int(match.group(2))) if match else (600, 400)
    head = 58
    total_w, total_h = max(width, 520), height + head
    body = inner[inner.index(">") + 1 : inner.rindex("</svg>")]
    esc = html.escape
    # A run that stopped early says so on the figure too: the empty cells weren't measured.
    partial = summary.get("partial")
    stopped = f" · {stopped_text(partial)}" if partial else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}" font-family="system-ui, sans-serif" font-size="10" '
        'color="#0a0a0b">'
        f'<rect width="{total_w}" height="{total_h}" fill="#ffffff"/>'
        f'<text x="16" y="24" font-size="15" font-weight="600" fill="currentColor">{esc(spec.name)}</text>'
        f'<text x="16" y="42" font-size="11" fill="#5d5d66">{esc(summary.get("description", ""))} · '
        f"n = {summary.get('n_prompts', '—')}{esc(stopped)}</text>"
        f'<g transform="translate(0 {head})">{body}</g></svg>\n'
    )


# -- the report ------------------------------------------------------------------------------

_STYLE = """
:root { color-scheme: light dark; --bg: #fff; --text: #0a0a0b; --muted: #5d5d66; --line: #e3e3e8; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #000; --text: #f5f5f7; --muted: #a3a3ad; --line: #26262c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 40px 24px 64px; }
h1 { font-size: 30px; line-height: 1.15; margin: 4px 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 15px; margin: 0 0 10px; }
section { padding-top: 22px; margin-top: 22px; border-top: 1px solid var(--line); }
section.flush { border-top: 0; padding-top: 0; }
.eyebrow, .muted, caption, figcaption { color: var(--muted); }
.eyebrow { font-size: 12px; margin: 0; }
.lede { margin: 0; max-width: 760px; }
.figures { display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid var(--text); }
.figures div { flex: 1 1 180px; padding: 12px 16px 12px 0; }
.figures dt { color: var(--muted); font-size: 12px; }
.figures dd { margin: 2px 0 0; font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 5px 10px 5px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; font-size: 12px; color: var(--muted); }
td.num, th.num { text-align: right; }
figure { margin: 0; overflow-x: auto; }
pre { white-space: pre-wrap; font: 12px/1.5 ui-monospace, Menlo, Consolas, monospace; margin: 0; }
details summary { cursor: pointer; margin: 8px 0; }
.methods p { max-width: 780px; }
dl.provenance { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }
dl.provenance dt { color: var(--muted); }
dl.provenance dd { margin: 0; overflow-wrap: anywhere; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
h3 { font-size: 13px; margin: 18px 0 4px; }
table.continuations { margin: 6px 0 4px; }
table.continuations caption { text-align: left; color: var(--muted); font-size: 12px; padding: 4px 0; }
table.continuations th { width: 9em; white-space: nowrap; }
table.continuations tr.control th, table.continuations tr.control td { color: var(--muted); }
.tok { display: inline-block; margin: 0 2px 2px 0; padding: 0 5px; border: 1px solid var(--line);
  border-radius: 3px; white-space: pre; font-size: 13px; }
"""


def _site_rows(sites: list[dict[str, Any]], ci_level: int, kept: set[int] | None = None) -> str:
    """The sites' statistics as a table. ``kept``: an ACDC run's circuit, whose edges the table
    marks kept and every other edge removed. Its values are the changes the removals made, between
    circuits rather than patched runs of the whole model, so it counts no sign flips."""
    esc = html.escape
    rows = []
    for s in sites:
        band = s.get("band") or {}
        circuit = "" if kept is None else f"<td>{'kept' if s['index'] in kept else 'removed'}</td>"
        flips = "" if kept is not None else f'<td class="num">{s.get("sign_flips", "—")}</td>'
        rows.append(
            "<tr>"
            f"<td>{esc(s['label'])}</td>"
            f"{circuit}"
            f'<td class="num">{_fmt(s["effect"]["mean"])}</td>'
            f'<td class="num">{_fmt(s["effect"]["lo"])} to {_fmt(s["effect"]["hi"])}</td>'
            f'<td class="num">'
            f"{_fmt(band.get('lo')) + ' to ' + _fmt(band.get('hi')) if band else '—'}</td>"
            f'<td class="num">{_fmt(s.get("q"), 3, sign=False)}</td>'
            f"{flips}"
            f'<td class="num">{s.get("n", "—")}</td>'
            "</tr>"
        )
    head = (
        "<th>Site</th><th class='num'>Effect</th>"
        if kept is None
        else "<th>Edge</th><th>Circuit</th><th class='num'>Change when removed</th>"
    )
    flips_head = "" if kept is not None else "<th class='num'>Sign flips</th>"
    return (
        f"<table><thead><tr>{head}"
        f"<th class='num'>{ci_level}% interval</th><th class='num'>Band, all sites</th>"
        f"<th class='num'>q-value</th>{flips_head}<th class='num'>n</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def visible_token(token: str) -> str:
    """A token as the app's token strip writes it: spaces as ·, line breaks as ⏎, tabs as ⇥."""
    return token.replace("\n", "⏎").replace("\t", "⇥").replace(" ", "·")


def _tokens_html(tokens: list[str]) -> str:
    return "".join(
        f'<span class="tok">{html.escape(visible_token(t)) or "∅"}</span>' for t in tokens
    )


def _continuations_html(continuations: dict[str, Any] | None) -> str:
    """Each continued prompt as written, then what the model says without steering and at each
    strength (the random control's lines marked), one table per steered site."""
    if not continuations or not continuations.get("prompts"):
        return ""
    esc = html.escape
    rows = continuations.get("rows") or []
    several = len(rows) > 1
    n = continuations.get("tokens")
    parts = [
        "<section><h2>What the steered model says</h2>"
        f"<p class='muted'>Greedy continuations of {n} new tokens (the most likely token each "
        "time, ties to the lowest token id) of the first held-out prompts. Each forward pass adds "
        "the vector at the steered site and token of the prompt, as when it was measured; the "
        "generated tokens aren't steered themselves. Tokens are shown as the model splits them: "
        "· is a space, ⏎ a line break.</p>"
    ]
    for prompt in continuations["prompts"]:
        parts.append(f"<h3>Prompt {prompt['index']}</h3><p class='lede'>{esc(prompt['text'])}</p>")
        for row in rows:
            # The random control's lines are named as its columns are ("random ×1"), and muted.
            lines = [("Without steering", prompt["unsteered"]["tokens"], False)] + [
                (s["key"], s["tokens"], s["control"])
                for s in prompt["steered"]
                if s["row"] == row["row"]
            ]
            caption = f"<caption>At {esc(row['label'])}</caption>" if several else ""
            body = "".join(
                f"<tr class='{'control' if control else 'steered'}'>"
                f"<th scope='row'>{esc(label)}</th>"
                f"<td class='said'>{_tokens_html(tokens)}</td></tr>"
                for label, tokens, control in lines
            )
            parts.append(f"<table class='continuations'>{caption}<tbody>{body}</tbody></table>")
    parts.append("</section>")
    return "".join(parts)


def run_report(
    spec: Spec,
    summary: dict[str, Any],
    manifest: dict[str, Any] | None,
    run_id: str,
    *,
    hardware: bool = False,
    continuations: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
) -> str:
    """One finished run as a self-contained HTML page. ``continuations`` is the run's
    continuations.json, for a steering run that recorded what the model says; ``verification``
    an edge attribution run's estimates against path patching (logogram.edges)."""
    manifest = manifest or {}
    esc = html.escape
    sites = [s for s in summary["sites"] if s["effect"]["mean"] is not None]
    strongest = sorted(sites, key=lambda s: -abs(s["effect"]["mean"]))
    # An ACDC run lists its circuit first, each edge marked kept or removed.
    kept = set(summary["acdc"]["kept"]) if summary.get("acdc") else None
    if kept is not None:
        strongest = [s for s in strongest if s["index"] in kept] + [
            s for s in strongest if s["index"] not in kept
        ]
    ci_level = round(spec.statistics.ci * 100)
    metric = summary.get("metric") or {}
    baseline = summary.get("baseline") or {}
    label = metric.get("label") or "metric"

    def stat(group: dict[str, Any] | None) -> str:
        return _fmt((group or {}).get("mean"), 2)

    clean = baseline.get("clean") or {}
    corrupt = baseline.get("corrupt") or {}
    figures = [
        ("Prompts", str(summary.get("n_prompts", "—"))),
        (f"Clean {label}", stat(clean.get("metric") or clean.get("logit_diff"))),
        (f"Corrupt {label}", stat(corrupt.get("metric") or corrupt.get("logit_diff"))),
        (
            "Gap the effects are measured against",
            _fmt((baseline.get("gap") or {}).get("mean"), 2, sign=False),
        ),
    ]
    acdc = summary.get("acdc")
    if acdc:
        f = acdc.get("faithfulness") or {}
        figures.append(("Edges kept", f"{acdc['n_kept']:,} of {acdc['n_edges']:,}"))
        figures.append(
            (
                "Faithfulness of the circuit",
                f"{_fmt(f.get('mean'), 2, sign=False)} [{_fmt(f.get('lo'), 2, sign=False)}, "
                f"{_fmt(f.get('hi'), 2, sign=False)}]",
            )
        )
    elif strongest:
        top = strongest[0]
        figures.append(
            ("Strongest effect", f"{esc(top['label'])} {_fmt(top['effect']['mean'], 2)}")
        )
    feature = (summary.get("steering") or {}).get("feature")
    if feature and feature.get("activations"):
        # What strength 1 is along an SAE feature: its typical activation where it is steered.
        typical = ", ".join(f"{a:.3g}" for a in feature["activations"])
        figures.append((f"Strength 1: feature {feature['feature']}'s typical activation", typical))
    figure_html = "".join(f"<div><dt>{esc(k)}</dt><dd>{v}</dd></div>" for k, v in figures)

    layout = summary.get("layout") or {}
    has_map = layout.get("kind") not in (None, "sites", "site_sets", "edges")
    caption = (
        "For each layer, the distribution of every neuron's mean normalized effect over prompts, "
        f"and its strongest neurons with their {ci_level}% intervals."
        if layout.get("kind") == "neurons"
        else f"Each cell is a site's mean normalized effect over prompts; hover for its "
        f"{ci_level}% interval.{_readers(layout)}"
    )
    title = (
        "Every neuron, layer by layer"
        if layout.get("kind") == "neurons"
        else f"{layout.get('row_title', '')} × {layout.get('col_title') or 'site'}"
    )
    map_html = (
        f"<section><h2>{esc(title)}</h2><figure>{heatmap_svg(summary)}<figcaption>"
        f"{esc(caption)} {esc(metric.get('normalized_effect', ''))}</figcaption></figure></section>"
        if has_map
        else ""
    )
    if layout.get("kind") == "edges":
        map_html = _edges_html(summary, ci_level, verification)
    edges = summary.get("edges") or summary.get("acdc") or {}
    unit = "edges" if layout.get("kind") == "edges" else "sites"
    listed = (
        f"The circuit's edges and the strongest removed, {len(summary['sites']):,} of"
        if summary.get("acdc")
        else f"The {len(summary['sites']):,} strongest of"
    )
    strongest_heading = (
        "The edges ACDC kept, then the largest changes it removed"
        if kept is not None
        else f"The strongest {unit}"
    )
    every = (
        f"{listed} {edges['n_edges']:,} edges (edges.parquet holds every edge's statistics)"
        if edges and edges["n_edges"] > len(summary["sites"])
        else f"The {len(summary['sites'])} {unit} that finished"
        if partial_of(summary)
        else f"All {len(summary['sites'])} {unit}"
    )
    methods = "".join(
        f"<p>{esc(p)}</p>" for p in methods_text(spec, summary, manifest, run_id).split("\n\n")
    )
    warnings = "".join(f"<li>{esc(w)}</li>" for w in summary.get("warnings", []))
    notes = (
        f"<section><h2>What was expected</h2><p class='lede'>{esc(spec.notes)}</p></section>"
        if spec.notes.strip()
        else ""
    )
    # A run that stopped early: say so before anything else, and what its numbers cover.
    partial = summary.get("partial")
    stopped = ""
    if partial:
        ended = "failed" if partial.get("status") == "failed" else "was cancelled"
        reason = manifest.get("error")
        items = "".join(f"<li>{esc(item)}</li>" for item in partial.get("left_out") or [])
        stopped = (
            f"<section class='stopped'><h2>This run {ended} before it finished</h2>"
            f"<p class='lede'>{esc(partial['note'])} Nothing is drawn for the rest. Their "
            "statistics are computed as for a finished run."
            + (f" {esc(reason)}" if reason else "")
            + "</p>"
            + (f"<ul>{items}</ul>" if items else "")
            + "</section>"
        )

    model = manifest.get("model") or {}
    device = manifest.get("device") or {}
    dataset = manifest.get("dataset") or {}
    pinned_at = model.get("revision") or spec.model.revision or "an unpinned revision"
    provenance: list[tuple[str, str]] = [
        ("Run", run_id),
        ("Finished", manifest.get("finished_at") or "—"),
        (
            "Model",
            f"{spec.model.id}, a local model, files SHA-256 {spec.model.sha256}"
            if spec.model.source == "local"
            else f"{spec.model.id} at {pinned_at}",
        ),
        ("Architecture", model.get("architecture") or "—"),
        ("Precision", spec.model.dtype),
        ("Device", device.get("type") or spec.model.device),
    ]
    if hardware and device.get("name"):
        provenance.append(("Hardware", device["name"]))
    chat = manifest.get("chat_template") or {}
    if spec.tokenization.chat_template is not None:
        how = (
            "the person's message"
            if spec.tokenization.chat_template.kind == "user_turn"
            else "the start of the model's reply"
        )
        sha = f", SHA-256 {chat['sha256']}" if chat.get("sha256") else ""
        provenance.append(("Chat template", f"the model's own, prompts as {how}{sha}"))
    provenance += [
        ("Dataset", f"{dataset.get('path') or spec.dataset.path}, {dataset.get('n', '—')} prompts"),
        ("Dataset SHA-256", dataset.get("sha256") or spec.dataset.sha256 or "not pinned"),
    ]
    versions = manifest.get("versions") or {}
    provenance += [(k, v) for k, v in versions.items() if v]
    provenance_html = "".join(f"<dt>{esc(k)}</dt><dd>{esc(str(v))}</dd>" for k, v in provenance)

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(spec.name)} · Logogram report</title><style>{_STYLE}</style></head>"
        "<body><main>"
        f"<header><p class='eyebrow'>Logogram report · run {esc(run_id)}</p>"
        f"<h1>{esc(spec.name)}</h1><p class='lede'>{esc(summary.get('description', ''))}</p></header>"
        f"{stopped}"
        f"{notes}"
        f"<section class='flush'><dl class='figures'>{figure_html}</dl></section>"
        f"{map_html}"
        f"<section><h2>{strongest_heading}</h2>"
        f"{_site_rows(strongest[:STRONGEST], ci_level, kept)}"
        f"<p class='muted'>{'Changes' if kept is not None else 'Effects'} are normalized: 1 is as "
        "far as switching to the other prompt, "
        f"0 is no change. The band holds for every {unit[:-1]} of the sweep at once; the q-value "
        f"is the share of false findings expected among {unit} this strong.</p></section>"
        f"{_continuations_html(continuations)}"
        f"<section class='methods'><h2>Methods</h2>{methods}</section>"
        + (f"<section><h2>Notes from the run</h2><ul>{warnings}</ul></section>" if warnings else "")
        + f"<section><h2>Every {unit[:-1]}</h2><details><summary>{every}</summary>"
        + f"{_site_rows(summary['sites'], ci_level, kept)}</details></section>"
        f"<section><h2>Where it came from</h2><dl class='provenance'>{provenance_html}</dl>"
        f"<details><summary>The spec</summary><pre>{esc(spec.to_json())}</pre></details></section>"
        f"<footer>Made with Logogram {esc(__version__)}. The per-prompt values behind every "
        "number are in the run's results.parquet"
        + (", and every edge's statistics in its edges.parquet" if edges else "")
        + ".</footer></main></body></html>\n"
    )


def _readers(layout: dict[str, Any]) -> str:
    """For a map of keys or values with grouped-query attention: which query heads read each
    key/value head."""
    cols = layout.get("cols") or []
    if layout.get("kind") != "head_inputs" or not any(
        len(c.get("query_heads") or []) > 1 for c in cols
    ):
        return ""
    groups = "; ".join(f"{c['label']}: {', '.join(str(h) for h in c['query_heads'])}" for c in cols)
    what = "key" if layout.get("site") == "k" else "value"
    return (
        f" Each column is a key/value head, whose {what} every query head of its group reads "
        f"(query heads by key/value head: {groups})."
    )


def partial_of(summary: dict[str, Any]) -> dict[str, Any] | None:
    return summary.get("partial")


def sites_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per site: its mean, intervals, q-value and counts, as the report lists them."""
    rows = []
    for s in summary["sites"]:
        band = s.get("band") or {}
        rows.append(
            {
                "index": s["index"],
                "label": s["label"],
                "kind": s["kind"],
                "layer": s["layer"],
                "head": s.get("head"),
                "feature": s.get("feature"),
                "neuron": s.get("neuron"),
                "position": s["position_key"],
                "variant": s.get("variant_key"),
                "n": s["n"],
                "effect": s["effect"]["mean"],
                "effect_sd": s["effect"]["sd"],
                "effect_lo": s["effect"]["lo"],
                "effect_hi": s["effect"]["hi"],
                "band_lo": band.get("lo"),
                "band_hi": band.get("hi"),
                "q": s.get("q"),
                "delta": s["delta"]["mean"],
                "patched_metric": s.get("patched_metric"),
                "sign_flips": s["sign_flips"],
                "opposite_sign": s["opposite_sign"],
            }
        )
    return rows


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)
