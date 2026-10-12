#!/usr/bin/env python3
"""Build the catalogue of published SAEs that Logogram ships, from SAELens's registry, and check
a sample of it on the real files.

People find SAEs by their SAELens release and id. SAELens (https://github.com/decoderesearch/SAELens,
MIT licence) lists them in ``sae_lens/pretrained_saes.yaml``, which ships inside the sae-lens
package; this script reads that file from a sae-lens wheel or installation it is pointed at,
never from the network, and records where it came from (sae-lens's version, the file's SHA-256)
and SAELens's copyright and permission notice, as its licence asks.

``build`` reads, for each release, its Hugging Face repository's file list at its current commit,
and each SAE's configuration there, and writes ``src/logogram/backends/sae_catalogue.json``: for
every SAE its release and id, the model it was trained on (as SAELens names it, and the Hugging
Face ids it goes by), its repository and folder, the hook it reads, its width and the architecture
its configuration states, and whether Logogram reads it exactly or why not. That is decided by
Logogram's own readers from the configuration (``logogram.backends.saes.read_config``, the checks
reading the files runs, on parameters without values), from the files a folder holds
(``unreadable_files``: pickles, NumPy archives, cross-layer parameters, formats not read yet), and
from settings SAELens's registry reads an SAE with that its configuration doesn't state. SAELens's
own files can hold parameters their configuration doesn't list (its earlier ones hold a
scaling_factor), so for that format the build also reads each weights file's header, and the
values of a scaling_factor, by range requests: no weights are downloaded.

``verify`` then checks a sample on the real files and writes what it found into the catalogue: the
header of the weights files of a few SAEs of every release (every parameter's name, shape and
type, read as Logogram reads the weights, must give the catalogue's verdict, site, layer and
width), and for one SAE of every release whose model runs here (GPT-2 small, Pythia-70m and Gemma
3 270M, the last through an ungated copy whose weights files have Google's SHA-256), the whole
files, read by Logogram's reader and run on the model, with and without TransformerLens's weight
processing, to measure the share of its activations' variance the SAE explains. A check that
disagrees with the catalogue fails the command.

Logogram never runs this script: the catalogue is package data, and at run time Logogram contacts
nothing but the Hugging Face downloads a person starts and the opt-in update check. Everything
fetched here goes into the cache folder given (``--cache``), so a rebuild reads the same commits
without the network unless ``--refresh`` asks for the repositories' current ones.

    uv run python scripts/sae_catalogue.py build --sae-lens sae_lens-6.54.0-py3-none-any.whl \\
        --cache CACHE
    uv run python scripts/sae_catalogue.py verify --cache CACHE --budget-mb 2500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "logogram" / "backends" / "sae_catalogue.json"
REGISTRY = "sae_lens/pretrained_saes.yaml"

# SAELens configuration settings that change what an SAE computes. When SAELens's registry reads
# an SAE with one of these set otherwise than its configuration, its files alone don't say what it
# computes. (Others, such as the dataset or how the model was loaded, change nothing it computes.)
COMPUTING = frozenset(
    {
        "architecture",
        "activation_fn",
        "activation_fn_str",
        "activation_fn_kwargs",
        "k",
        "apply_b_dec_to_input",
        "normalize_activations",
        "finetuning_scaling_factor",
        "reshape_activations",
        "hook_name",
        "hook_point",
        "hook_head_index",
        "d_in",
        "d_sae",
    }
)

# Formats SAELens converts that Logogram doesn't read, in words, by SAELens's conversion function,
# when the files alone don't say what they are.
NOT_READ = {
    "mwhanna_transcoder": (
        "These transcoders are published as one safetensors file per layer, configured at the top "
        "of their repository (circuit-tracer's format, which SAELens converts), a format Logogram "
        "doesn't read yet."
    ),
}

TL_SITES = {
    "hook_resid_pre": "resid_pre",
    "hook_resid_mid": "resid_mid",
    "hook_resid_post": "resid_post",
    "hook_attn_out": "attn_out",
    "hook_mlp_out": "mlp_out",
    "attn.hook_z": "head",
    "hook_z": "head",
    "ln2.hook_normalized": "mlp_in",
    "mlp.hook_in": "mlp_in",
    "hook_mlp_in": "mlp_in",
}


def log(*parts: Any) -> None:
    print(*parts, file=sys.stderr, flush=True)


# -- the registry ------------------------------------------------------------------------------


@dataclass
class Registry:
    version: str
    licence: str
    notice: str
    text: bytes
    releases: dict[str, Any]


def read_registry(source: Path) -> Registry:
    """SAELens's registry, version and licence from a sae-lens wheel, an installation (the
    site-packages folder holding sae_lens and its .dist-info) or the sae_lens package folder."""
    import yaml

    metadata = notice = ""
    if source.is_file():
        with zipfile.ZipFile(source) as wheel:
            names = wheel.namelist()
            text = wheel.read(REGISTRY)
            for name in names:
                if name.endswith(".dist-info/METADATA"):
                    metadata = wheel.read(name).decode("utf-8")
                if re.search(r"\.dist-info/(?:licenses/)?LICEN[CS]E[^/]*$", name):
                    notice = wheel.read(name).decode("utf-8")
    else:
        package = source / "sae_lens" if (source / "sae_lens").is_dir() else source
        text = (package / "pretrained_saes.yaml").read_bytes()
        for dist in sorted(package.parent.glob("sae_lens-*.dist-info")):
            metadata = (dist / "METADATA").read_text(encoding="utf-8")
            licences = [p for p in dist.rglob("LICEN[CS]E*") if p.is_file()]
            if licences:
                notice = licences[0].read_text(encoding="utf-8")
    version = re.search(r"^Version: *(\S+)", metadata, re.M)
    licence = re.search(r"^License-Expression: *(.+)$", metadata, re.M) or re.search(
        r"^License: *(.+)$", metadata, re.M
    )
    if not (version and licence and notice.strip()):
        raise SystemExit(
            f"{source} has no sae-lens version, licence or licence notice: point --sae-lens at a "
            "sae-lens wheel, or at the site-packages folder it is installed in."
        )
    releases = yaml.safe_load(text)
    if not isinstance(releases, dict):
        raise SystemExit(f"{REGISTRY} in {source} isn't a mapping of releases.")
    return Registry(
        version=version.group(1),
        licence=licence.group(1).strip(),
        notice=notice.strip() + "\n",
        text=text,
        releases=releases,
    )


def registry_saes(release: dict[str, Any]) -> list[dict[str, Any]]:
    """A release's SAEs, one per id (a later entry replaces an earlier one, as SAELens reads it),
    keeping only what SAELens reads of each."""
    by_id: dict[str, dict[str, Any]] = {}
    for sae in release.get("saes") or []:
        path = sae.get("path") if isinstance(sae, dict) else None
        # Paths may be numbers (a cross-layer transcoder's layer, 0 among them).
        if (
            isinstance(sae, dict)
            and isinstance(sae.get("id"), str)
            and path is not None
            and str(path) != ""
        ):
            by_id[sae["id"]] = {
                "id": sae["id"],
                "path": str(sae["path"]),
                "l0": sae.get("l0"),
            }
    return list(by_id.values())


# -- Hugging Face ------------------------------------------------------------------------------


class Hub:
    """Repository listings and small files from Hugging Face, kept in the cache folder."""

    def __init__(self, cache: Path, refresh: bool) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.cache = cache
        self.refresh = refresh
        (cache / "repos").mkdir(parents=True, exist_ok=True)
        self.models_file = cache / "models.json"
        self.models: dict[str, list[str]] = (
            json.loads(self.models_file.read_text(encoding="utf-8"))
            if self.models_file.is_file()
            else {}
        )

    def _retry(self, fn: Callable[[], Any], what: str) -> Any:
        for attempt in range(8):
            try:
                return fn()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (429, 500, 502, 503, 504) or "timed out" in str(exc).lower():
                    delay = min(60.0, 2.0**attempt)
                    log(f"  {what}: {status or 'timeout'}, again in {delay:.0f}s")
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"{what}: gave up after several tries")

    def repo(self, repo_id: str) -> dict[str, Any]:
        """A repository's commit, files (name: size), gating and licence."""
        path = self.cache / "repos" / f"{repo_id.replace('/', '__')}.json"
        if path.is_file() and not self.refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        try:
            info = self._retry(
                lambda: self.api.model_info(repo_id, files_metadata=True), f"list {repo_id}"
            )
        except (RepositoryNotFoundError, GatedRepoError) as exc:
            data = {"id": repo_id, "error": type(exc).__name__}
        else:
            card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
            licence = card.get("license") if card is not None else None
            data = {
                "id": info.id,
                "sha": info.sha,
                "gated": bool(info.gated),
                "licence": licence if isinstance(licence, str) else None,
                "files": {s.rfilename: int(s.size or 0) for s in info.siblings or []},
            }
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        return data

    def file(self, repo_id: str, sha: str, name: str) -> Path | None:
        """A small file of a repository at a commit, or None if it isn't there."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, GatedRepoError

        try:
            return Path(
                self._retry(
                    lambda: hf_hub_download(
                        repo_id, name, revision=sha, cache_dir=self.cache / "hub"
                    ),
                    f"fetch {repo_id}/{name}",
                )
            )
        except (EntryNotFoundError, GatedRepoError):
            return None

    def fetch_all(self, wanted: Iterable[tuple[str, str, str]], workers: int = 8) -> None:
        todo = sorted(set(wanted))
        log(f"Reading {len(todo)} configuration files")
        done = 0

        def one(item: tuple[str, str, str]) -> None:
            self.file(*item)

        with ThreadPoolExecutor(workers) as pool:
            for _ in pool.map(one, todo):
                done += 1
                if done % 500 == 0:
                    log(f"  {done} of {len(todo)}")

    def _range(self, repo_id: str, sha: str, name: str, first: int, last: int) -> bytes:
        from huggingface_hub import get_session, hf_hub_url
        from huggingface_hub.utils import build_hf_headers

        url = hf_hub_url(repo_id, name, revision=sha)

        def get() -> bytes:
            response = get_session().get(
                url, headers={**build_hf_headers(), "Range": f"bytes={first}-{last}"}, timeout=60
            )
            response.raise_for_status()
            return response.content

        return self._retry(get, f"read {repo_id}/{name}")

    def header(self, repo_id: str, sha: str, name: str) -> dict[str, Any]:
        """A safetensors file's header: its length, and each tensor's dtype, shape and offsets,
        read with two range requests rather than downloading the file."""
        path = self.cache / "headers" / repo_id.replace("/", "__") / sha / f"{name}.header.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        size = int.from_bytes(self._range(repo_id, sha, name, 0, 7), "little")
        tensors = json.loads(self._range(repo_id, sha, name, 8, 8 + size - 1).decode("utf-8"))
        tensors.pop("__metadata__", None)
        data = {"size": size, "tensors": tensors}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        return data

    def tensor(self, repo_id: str, sha: str, name: str, tensor: str) -> Any:
        """One small tensor of a safetensors file, read with a range request."""
        import torch

        header = self.header(repo_id, sha, name)
        entry = header["tensors"][tensor]
        dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
        begin, end = entry["data_offsets"]
        path = self.cache / "headers" / repo_id.replace("/", "__") / sha / f"{name}.{tensor}.bin"
        if not path.is_file():
            base = 8 + header["size"]
            path.write_bytes(self._range(repo_id, sha, name, base + begin, base + end - 1))
        raw = bytearray(path.read_bytes())
        assert len(raw) == end - begin, f"{name}: {tensor} came back cut short"
        values = torch.frombuffer(raw, dtype=dtypes[entry["dtype"]]).clone()
        return values.reshape(entry["shape"]).float()

    def model_ids(self, model: str) -> list[str]:
        """The Hugging Face ids a model SAELens names goes by: the repository's own first, then
        the names that lead to it (TransformerLens's official name, the registry's own)."""
        if model in self.models:
            return self.models[model]
        from transformer_lens.supported_models import get_official_model_name

        try:
            official = get_official_model_name(model)
        except ValueError:
            if "/" not in model:
                raise SystemExit(
                    f"TransformerLens doesn't know the model {model!r}, so its Hugging Face id "
                    "can't be found: add it to the aliases in this script."
                ) from None
            official = model
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            info = self._retry(lambda: self.api.model_info(official), f"model {official}")
            canonical = str(info.id)
        except RepositoryNotFoundError:
            log(f"  Hugging Face has no model {official}; keeping the name as it is")
            canonical = official
        ids = list(dict.fromkeys([canonical, official] + ([model] if "/" in model else [])))
        self.models[model] = ids
        self.models_file.write_text(json.dumps(self.models, indent=1, sort_keys=True), "utf-8")
        return ids


# -- one SAE -----------------------------------------------------------------------------------


@dataclass
class Row:
    id: str
    folder: str
    hook: str | None = None
    site: str | None = None
    layer: int | None = None
    width: int | None = None
    architecture: str | None = None
    kind: str = "sae"
    reads: str = "exact"
    reason: str | None = None
    registry: dict[str, Any] | None = None
    l0: float | None = None
    verified: dict[str, Any] | None = None
    # What a check on the files needs: the weights file, the configuration as read and its file.
    weights: str | None = field(default=None, repr=False)
    config: dict[str, Any] | None = field(default=None, repr=False)
    training: dict[str, Any] | None = field(default=None, repr=False)

    def refuse(self, reason: str, reads: str = "refused") -> None:
        self.reads, self.reason = reads, reason

    def stored(self, reasons: dict[str, int]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "folder": self.folder,
            "hook": self.hook,
            "site": self.site,
            "layer": self.layer,
            "width": self.width,
            "architecture": self.architecture,
            "kind": self.kind,
            "reads": self.reads,
        }
        if self.reason is not None:
            out["why"] = reasons.setdefault(self.reason, len(reasons))
        if self.registry:
            out["registry"] = self.registry
        if self.l0 is not None:
            out["l0"] = self.l0
        if self.verified:
            out["verified"] = self.verified
        return out


def tl_site(hook: str | None) -> tuple[str | None, int | None]:
    """The site and layer of a TransformerLens hook name, as far as it says."""
    match = re.fullmatch(r"blocks\.(\d+)\.(.+)", hook or "")
    if match is None:
        return None, None
    return TL_SITES.get(match.group(2)), int(match.group(1))


def width_of(text: str) -> int | None:
    """A width as Gemma Scope and others write it in folder names: width_16k, width_1m."""
    match = re.search(r"width_(\d+)([km])", text)
    if match is None:
        return None
    n = int(match.group(1))
    return n * 1024 if match.group(2) == "k" else n * 1024 * 1024


def json_file(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def yaml_file(path: Path | None) -> dict[str, Any] | None:
    import yaml

    if path is None:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class Context:
    """One release as the build reads it."""

    name: str
    conversion: str
    release: dict[str, Any]
    repo: dict[str, Any]
    hub: Hub

    @property
    def sha(self) -> str:
        return str(self.repo["sha"])

    @property
    def files(self) -> dict[str, int]:
        return self.repo.get("files") or {}

    def inside(self, path: str) -> list[str]:
        """The files in a folder of the repository (nested ones too), or the one file a path
        names. (A cross-layer transcoder's layer is a number naming files at the top.)"""
        if self.conversion == "mntss_clt_layer_transcoder":
            layer_files = (f"W_enc_{path}.safetensors", f"W_dec_{path}.safetensors")
            return [n for n in layer_files if n in self.files]
        prefix = f"{path.strip('/')}/"
        names = [n[len(prefix) :] for n in self.files if n.startswith(prefix)]
        if names:
            return names
        return [path.rsplit("/", 1)[-1]] if path.strip("/") in self.files else []

    def config(self, name: str) -> Path | None:
        return self.hub.file(self.repo["id"], self.sha, name) if name in self.files else None

    def wanted(self, sae: dict[str, Any]) -> list[str]:
        """The configuration files this SAE's description needs."""
        path = sae["path"].strip("/")
        names = {
            "gemma_3": [f"{path}/config.json"],
            "dictionary_learning_1": [f"{path}/config.json"],
            "llama_scope": [f"{path}/hyperparams.json"],
            "llama_scope_r1_distill": [f"{path}/config.json"],
            "connor_rob_hook_z": [path.split(".pt")[0] + "_cfg.json"],
            "mwhanna_transcoder": ["wandb-config.yaml", "wanb-config.yaml", "config.yaml"],
            "mntss_clt_layer_transcoder": ["config.yaml"],
            "temporal": [f"{path}/conf.yaml"],
        }.get(self.conversion, [f"{path}/cfg.json", "config.json"])
        return [n for n in names if n in self.files]


def describe(ctx: Context, sae: dict[str, Any]) -> Row:
    """What the catalogue says of one SAE."""
    from logogram.backends.base import BackendError
    from logogram.backends.saes import read_config, unreadable_files, weights_file

    row = Row(id=sae["id"], folder=sae["path"].strip("/"))
    if isinstance(sae.get("l0"), int | float) and sae["l0"] > 0:
        row.l0 = round(float(sae["l0"]), 2)
    files = ctx.inside(row.folder)
    describe_format(ctx, row, sae)
    if not files:
        row.refuse(
            f"SAELens's registry names {row.folder}, which isn't in {ctx.repo['id']} at the "
            "catalogued commit."
        )
        return row
    weights = weights_file(files)
    if weights is None:
        reason = NOT_READ.get(ctx.conversion) or unreadable_files(files)
        assert reason is not None
        row.refuse(reason)
        return row
    cfg = json_file(ctx.config(f"{row.folder}/{_config_name(weights)}"))
    if cfg is None:
        row.refuse(
            "Its configuration couldn't be read when the catalogue was made; Logogram checks it "
            "when you load it.",
            reads="unknown",
        )
        return row
    training = None
    if weights == "sae.safetensors" and "transcode" not in cfg:
        training = json_file(ctx.config("config.json"))
    row.weights, row.config, row.training = weights, cfg, training
    settings = registry_settings(ctx.release, cfg)
    if settings:
        from logogram.backends.catalogue import registry_reason

        row.registry = settings
        row.refuse(registry_reason(ctx.name, row.id, settings))
        return row
    tensors = None
    if weights in HEADER_FORMATS:
        # Its configuration doesn't list every parameter its weights file holds (SAELens's earlier
        # files hold a scaling_factor too): read the file's header, and the values of a
        # scaling_factor, which decide whether it is read exactly.
        tensors = file_tensors(ctx, f"{row.folder}/{weights}")
        row.verified = {"revision": ctx.sha, "read": "header", "fit": None}
    try:
        params = read_config(weights, cfg, row.folder, training, tensors=tensors)
    except BackendError as exc:
        row.refuse(str(exc))
        return row
    row.site, row.layer, row.width = params.site, params.layer, params.d_sae
    row.kind = (
        "sae"
        if not params.transcoder
        else "skip_transcoder"
        if params.W_skip is not None
        else "transcoder"
    )
    return row


def _config_name(weights: str) -> str:
    from logogram.backends.saes import CONFIG_FILES

    return CONFIG_FILES[weights]


# Formats whose configuration doesn't say everything their weights file holds, so the build reads
# each file's header too: SAELens's own.
HEADER_FORMATS = ("sae_weights.safetensors",)
# Parameters whose values decide whether an SAE is read exactly, read by a range request.
DECIDING = ("scaling_factor",)


def file_tensors(ctx: Context, name: str) -> dict[str, Any]:
    """The parameters a weights file holds, by name and shape, with the values of those that
    decide whether it is read exactly."""
    header = ctx.hub.header(ctx.repo["id"], ctx.sha, name)["tensors"]
    tensors: dict[str, Any] = {k: tuple(v["shape"]) for k, v in header.items()}
    for deciding in DECIDING:
        if deciding in header:
            tensors[deciding] = ctx.hub.tensor(ctx.repo["id"], ctx.sha, name, deciding)
    return tensors


def registry_settings(release: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """The settings SAELens's registry reads an SAE with, on top of its configuration, that change
    what it computes and that its configuration states otherwise (or not at all)."""
    overrides = release.get("config_overrides") or {}
    meta = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
    differ = {}
    for key, value in overrides.items():
        if key in COMPUTING and cfg.get(key, meta.get(key)) != value:
            differ[key] = value
    return differ


def describe_format(ctx: Context, row: Row, sae: dict[str, Any]) -> None:
    """The hook, width, architecture and kind its configuration states (or, where its publisher
    wrote none, what SAELens's conversion states of it)."""
    conv, path = ctx.conversion, row.folder
    if conv in ("sae_lens", "sparsify"):
        cfg = json_file(ctx.config(f"{path}/cfg.json")) or {}
        meta = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
        overrides = ctx.release.get("config_overrides") or {}
        row.hook = cfg.get("hook_name") or cfg.get("hook_point") or meta.get("hook_name")
        row.hook = row.hook or overrides.get("hook_name")
        row.width = cfg.get("d_sae") if isinstance(cfg.get("d_sae"), int) else None
        fn = cfg.get("activation_fn_str") or cfg.get("activation_fn")
        architecture = cfg.get("architecture") or ("topk" if fn == "topk" else "standard")
        row.architecture = architecture if cfg else None
    elif conv == "gemma_3":
        cfg = json_file(ctx.config(f"{path}/config.json")) or {}
        row.hook = cfg.get("hf_hook_point_in")
        row.width = cfg.get("width") if isinstance(cfg.get("width"), int) else width_of(path)
        row.architecture = cfg.get("architecture")
        if cfg.get("type") in ("transcoder", "clt"):
            row.kind = "skip_transcoder" if cfg.get("affine_connection") else "transcoder"
    elif conv in ("gemma_2", "gemma_2_transcoder"):
        repo = ctx.repo["id"]
        n = re.search(r"layer_(\d+)", path)
        if conv == "gemma_2_transcoder":
            row.hook = f"blocks.{n.group(1)}.ln2.hook_normalized" if n else None
            row.kind, row.architecture = "transcoder", "jumprelu_transcoder"
        else:
            kind = (
                "hook_resid_post"
                if "res" in repo
                else "hook_mlp_out"
                if "mlp" in repo
                else "attn.hook_z"
            )
            row.hook = (
                "hook_embed"
                if "embedding" in path
                else f"blocks.{n.group(1)}.{kind}"
                if n
                else None
            )
            row.architecture = "jumprelu"
        row.width = width_of(path)
    elif conv == "dictionary_learning_1":
        cfg = json_file(ctx.config(f"{path}/config.json")) or {}
        trainer = cfg.get("trainer") if isinstance(cfg.get("trainer"), dict) else {}
        if isinstance(trainer.get("layer"), int):
            row.hook = f"blocks.{trainer['layer']}.hook_resid_post"
        row.width = trainer.get("dict_size") if isinstance(trainer.get("dict_size"), int) else None
        row.architecture = trainer.get("dict_class")
    elif conv in ("llama_scope", "llama_scope_r1_distill"):
        name = "hyperparams.json" if conv == "llama_scope" else "config.json"
        cfg = json_file(ctx.config(f"{path}/{name}")) or {}
        row.hook = cfg.get("hook_point_in")
        d_sae = cfg.get("d_sae")
        if not isinstance(d_sae, int) and isinstance(cfg.get("d_model"), int):
            factor = cfg.get("expansion_factor")
            d_sae = cfg["d_model"] * factor if isinstance(factor, int) else None
        row.width = d_sae if isinstance(d_sae, int) else None
        row.architecture = cfg.get("act_fn") or cfg.get("sae_type") or "jumprelu"
    elif conv == "qwen_scope":
        match = re.search(r"-W(\d+)K-L0_(\d+)", ctx.repo["id"])
        n = re.search(r"layer(\d+)", path)
        row.hook = f"blocks.{n.group(1)}.hook_resid_post" if n else None
        row.width = int(match.group(1)) * 1024 if match else None
        row.architecture = f"topk (k = {match.group(2)})" if match else "topk"
    elif conv == "mwhanna_transcoder":
        wandb = yaml_file(ctx.config("wandb-config.yaml")) or yaml_file(
            ctx.config("wanb-config.yaml")
        )
        n = re.search(r"layer_(\d+)", path)
        row.hook = f"blocks.{n.group(1)}.mlp.hook_in" if n else None
        feature = (wandb or {}).get("d_feature")
        if isinstance(feature, dict) and isinstance(feature.get("value"), int):
            row.width = feature["value"]
        row.kind, row.architecture = "transcoder", "transcoder"
    elif conv == "mntss_clt_layer_transcoder":
        cfg = yaml_file(ctx.config("config.yaml")) or {}
        if isinstance(cfg.get("feature_input_hook"), str):
            row.hook = f"blocks.{path}.{cfg['feature_input_hook']}"
        row.kind, row.architecture = "transcoder", "cross-layer transcoder"
    elif conv == "connor_rob_hook_z":
        cfg = json_file(ctx.config(path.split(".pt")[0] + "_cfg.json")) or {}
        row.hook = cfg.get("act_name")
        row.width = cfg.get("dict_size") if isinstance(cfg.get("dict_size"), int) else None
        row.architecture = "standard"
    elif conv == "temporal":
        cfg = yaml_file(ctx.config(f"{path}/conf.yaml")) or {}
        overrides = ctx.release.get("config_overrides") or {}
        n = re.search(r"layer_(\d+)", path)
        row.hook = overrides.get("hook_name") or (
            f"blocks.{n.group(1)}.hook_resid_post" if n else None
        )
        llm, sae_cfg = cfg.get("llm") or {}, cfg.get("sae") or {}
        if isinstance(llm.get("dimin"), int) and isinstance(sae_cfg.get("exp_factor"), int | float):
            row.width = int(llm["dimin"] * sae_cfg["exp_factor"])
        row.architecture = "temporal"
    elif conv == "goodfire":
        known = {
            "Llama-3.3-70B-Instruct-SAE-l50.pt": ("blocks.50.hook_resid_post", 65536),
            "Llama-3.1-8B-Instruct-SAE-l19.pth": ("blocks.19.hook_resid_post", 65536),
        }
        row.hook, row.width = known.get(path, (None, None))
        row.architecture = "standard"
    elif conv == "deepseek_r1":
        n = re.search(r"l(\d+)", path)
        row.hook = f"blocks.{n.group(1)}.hook_resid_post" if n else None
        row.width, row.architecture = 4096 * 16, "standard"
    site, n_layer = tl_site(row.hook)
    if site is None and row.hook:
        from logogram.backends.base import BackendError
        from logogram.backends.saes import site_of_hf_point

        try:
            site, n_layer = site_of_hf_point(row.hook)
        except BackendError:
            site, n_layer = None, None
    if site == "mlp_in":
        site = "mlp_out"  # a transcoder: it reads the MLP's input and writes its output
        if row.kind == "sae":
            row.kind = "transcoder"
    row.site, row.layer = site, n_layer


# -- the build ---------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> None:
    registry = read_registry(Path(args.sae_lens))
    hub = Hub(Path(args.cache), refresh=args.refresh)
    log(f"sae-lens {registry.version}: {len(registry.releases)} releases")
    contexts: list[tuple[Context, list[dict[str, Any]]]] = []
    for name, release in registry.releases.items():
        conversion = release.get("conversion_func") or "sae_lens"
        repo = hub.repo(release["repo_id"])
        ctx = Context(name, conversion, release, repo, hub)
        contexts.append((ctx, registry_saes(release)))
    wanted = []
    for ctx, saes in contexts:
        if "sha" not in ctx.repo:
            continue
        for sae in saes:
            wanted += [(ctx.repo["id"], ctx.sha, n) for n in ctx.wanted(sae)]
    hub.fetch_all(wanted)
    previous = previous_checks(Path(args.out))
    reasons: dict[str, int] = {}
    releases = []
    counts: Counter[str] = Counter()
    for ctx, saes in contexts:
        rows = []
        for sae in saes:
            if "sha" not in ctx.repo:
                row = Row(id=sae["id"], folder=sae["path"].strip("/"))
                row.refuse(
                    f"Its repository, {ctx.repo['id']}, couldn't be read when the catalogue was "
                    f"made ({ctx.repo.get('error')}); Logogram checks it when you load it.",
                    reads="unknown",
                )
            else:
                row = describe(ctx, sae)
                # A check an earlier run made on these very files, if it read more than this one.
                kept = previous.get((ctx.name, row.id))
                same = bool(kept) and kept.get("revision") == ctx.sha
                if same and (kept.get("read") == "weights" or row.verified is None):
                    row.verified = kept
            counts[row.reads] += 1
            rows.append(row)
        releases.append(
            {
                "release": ctx.name,
                "model": ctx.release["model"],
                "hf_models": hub.model_ids(ctx.release["model"]),
                # As Hugging Face names the repository now (a renamed one redirects).
                "repo": ctx.repo.get("id") or ctx.release["repo_id"],
                "revision": ctx.repo.get("sha"),
                "licence": ctx.repo.get("licence"),
                "gated": bool(ctx.repo.get("gated")),
                "saes": [r.stored(reasons) for r in rows],
            }
        )
    catalogue = {
        "catalogue": 1,
        "source": {
            "package": "sae-lens",
            "version": registry.version,
            "file": REGISTRY,
            "sha256": hashlib.sha256(registry.text).hexdigest(),
            "licence": registry.licence,
            "notice": registry.notice,
        },
        "reasons": list(reasons),
        "releases": releases,
    }
    write_catalogue(Path(args.out), catalogue)
    log(f"Wrote {args.out}: {sum(counts.values())} SAEs, {dict(counts)}")


def previous_checks(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """The checks on real files an earlier catalogue recorded, by release and id."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (r["release"], s["id"]): s["verified"]
        for r in data.get("releases", [])
        for s in r["saes"]
        if s.get("verified")
    }


def write_catalogue(path: Path, catalogue: dict[str, Any]) -> None:
    """The catalogue as JSON, one SAE to a line, after checking it against Logogram's schema."""
    from logogram.backends.catalogue import parse_catalogue

    parse_catalogue(catalogue)

    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))

    lines = ["{", f'  "catalogue": {catalogue["catalogue"]},', '  "source": {']
    source = list(catalogue["source"].items())
    lines += [
        f"    {dumps(k)}: {dumps(v)}{',' if i < len(source) - 1 else ''}"
        for i, (k, v) in enumerate(source)
    ]
    lines += ["  },", '  "reasons": [']
    reasons = catalogue["reasons"]
    lines += [f"    {dumps(r)}{',' if i < len(reasons) - 1 else ''}" for i, r in enumerate(reasons)]
    lines += ["  ],", '  "releases": [']
    for i, release in enumerate(catalogue["releases"]):
        head = {k: v for k, v in release.items() if k != "saes"}
        lines.append("    {" + dumps(head)[1:-1] + ', "saes": [')
        saes = release["saes"]
        lines += [f"      {dumps(s)}{',' if j < len(saes) - 1 else ''}" for j, s in enumerate(saes)]
        lines.append("    ]}" + ("," if i < len(catalogue["releases"]) - 1 else ""))
    lines += ["  ]", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- the checks on real files ------------------------------------------------------------------

# Models small enough to run here, by the Hugging Face id a release names, and what to load: the
# model itself, or an ungated copy of a gated one, used only if its weights files have the same
# SHA-256 as the original's.
RUNNABLE = {
    "openai-community/gpt2": "openai-community/gpt2",
    "EleutherAI/pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
    "google/gemma-3-270m": "unsloth/gemma-3-270m",
    "google/gemma-3-270m-it": "unsloth/gemma-3-270m-it",
}
# Weights files larger than this (in MiB) are checked only by their header.
LARGEST_MB = 300
FLOATS = {"F64", "F32", "F16", "BF16"}


@dataclass
class Check:
    release: dict[str, Any]
    sae: dict[str, Any]
    ctx: Context
    weights: bool = False

    @property
    def name(self) -> str:
        return f"{self.release['release']} {self.sae['id']}"


def verify(args: argparse.Namespace) -> None:
    """Check a sample of the catalogue on the real files, and record each check in it."""
    path = Path(args.out)
    catalogue = json.loads(path.read_text(encoding="utf-8"))
    hub = Hub(Path(args.cache), refresh=False)
    checks = plan(catalogue, hub, random.Random(args.seed), args.per_release, args.budget_mb)
    log(f"Checking {len(checks)} SAEs, {sum(c.weights for c in checks)} of them with their weights")
    failures: list[str] = []
    for check in checks:
        try:
            check.sae["verified"] = check_header(check, catalogue["reasons"])
        except AssertionError as exc:
            failures.append(f"{check.name}: {exc}")
    text = Path(args.text).read_text(encoding="utf-8") if args.text else None
    models: dict[tuple[str, bool], Any] = {}
    for check in [c for c in checks if c.weights and c.sae.get("verified")]:
        try:
            check.sae["verified"] = check_weights(check, text, models, Path(args.cache))
        except AssertionError as exc:
            failures.append(f"{check.name}: {exc}")
    write_catalogue(path, catalogue)
    verified = Counter(
        s["verified"]["read"] for r in catalogue["releases"] for s in r["saes"] if s.get("verified")
    )
    log(f"The catalogue records checks on the files of {dict(verified)} SAEs")
    if failures:
        log("These disagree with the catalogue:")
        for failure in failures:
            log(f"  {failure}")
        raise SystemExit(1)


def plan(
    catalogue: dict[str, Any], hub: Hub, rng: random.Random, per_release: int, budget_mb: int
) -> list[Check]:
    """A few SAEs of every release in a format Logogram reads (those it reads exactly, and one it
    refuses for what its configuration says), and the weights of one per release whose model runs
    here, as far as the budget goes."""
    checks = []
    for release in catalogue["releases"]:
        repo = hub.repo(release["repo"])
        if release["revision"] is None or repo.get("sha") != release["revision"]:
            continue
        ctx = Context(release["release"], "", {}, repo, hub)
        exact = [s for s in release["saes"] if s["reads"] == "exact"]
        refused = [s for s in release["saes"] if s["reads"] == "refused" and readable(ctx, s)]
        picked = rng.sample(exact, min(per_release, len(exact)))
        picked += rng.sample(refused, min(1, len(refused)))
        for i, sae in enumerate(picked):
            check = Check(release, sae, ctx)
            if i == 0 and sae["reads"] == "exact" and release["hf_models"][0] in RUNNABLE:
                size = ctx.files.get(f"{sae['folder']}/{readable(ctx, sae)}", 0) / 2**20
                if size <= min(LARGEST_MB, budget_mb):
                    check.weights, budget_mb = True, budget_mb - int(size)
            checks.append(check)
    return checks


def readable(ctx: Context, sae: dict[str, Any]) -> str | None:
    from logogram.backends.saes import weights_file

    return weights_file(ctx.inside(sae["folder"]))


def check_header(check: Check, reasons: list[str]) -> dict[str, Any]:
    """The SAE's weights file's header (each parameter's name, shape and type), read as Logogram
    reads the weights: it must give the catalogue's verdict, and the same site, layer and width."""
    from logogram.backends.base import BackendError
    from logogram.backends.saes import CONFIG_FILES, read_config

    ctx, sae = check.ctx, check.sae
    weights = readable(ctx, sae)
    assert weights is not None, "its folder holds no format Logogram reads"
    cfg = json_file(ctx.config(f"{sae['folder']}/{CONFIG_FILES[weights]}"))
    assert cfg is not None, "its configuration can't be read"
    training = None
    if weights == "sae.safetensors" and "transcode" not in cfg:
        training = json_file(ctx.config("config.json"))
    name = f"{sae['folder']}/{weights}"
    header = ctx.hub.header(ctx.repo["id"], ctx.sha, name)["tensors"]
    dtypes = {v["dtype"] for v in header.values()}
    assert dtypes <= FLOATS, f"its weights hold {sorted(dtypes - FLOATS)}"
    checked = {"revision": ctx.sha, "read": "header", "fit": None}
    try:
        params = read_config(weights, cfg, sae["folder"], training, tensors=file_tensors(ctx, name))
    except BackendError as exc:
        stated = reasons[sae["why"]] if sae.get("why") is not None else None
        assert str(exc) == stated, f"its files are refused ({exc})"
        return checked
    if sae.get("registry"):
        return checked  # read as its files say; SAELens's registry reads it otherwise
    assert sae["reads"] == "exact", "its files are read, but the catalogue refuses it"
    found = (params.site, params.layer, params.d_sae)
    stated = (sae["site"], sae["layer"], sae["width"])
    assert found == stated, f"its files read as {found}, the catalogue says {stated}"
    return checked


def check_weights(
    check: Check, text: str | None, models: dict[tuple[str, bool], Any], cache: Path
) -> dict[str, Any]:
    """Download the SAE's files, read them whole as Logogram reads them, and run the SAE on its
    model, with TransformerLens's weight processing and without: the share of the variance of the
    activations it explains, and how many features fire per token, over the text (the setting it
    fits better is recorded)."""
    import shutil

    from huggingface_hub import hf_hub_download

    from logogram.backends.saes import CONFIG_FILES, read_sae
    from logogram.sae import SAE

    ctx, sae = check.ctx, check.sae
    weights = readable(ctx, sae)
    assert weights is not None
    root = cache / "weights" / ctx.repo["id"].replace("/", "__")
    for name in (CONFIG_FILES[weights], weights):
        hf_hub_download(ctx.repo["id"], f"{sae['folder']}/{name}", revision=ctx.sha, local_dir=root)
    params = read_sae(root / sae["folder"], sae["folder"])
    found = (params.site, params.layer, params.d_sae)
    stated = (sae["site"], sae["layer"], sae["width"])
    assert found == stated, f"its files read as {found}, the catalogue says {stated}"
    checked: dict[str, Any] = {"revision": ctx.sha, "read": "weights", "fit": None}
    model_id = check.release["hf_models"][0]
    if text is not None:
        for processed in (True, False):
            backend = load(model_id, processed, models)
            if backend is None:
                continue
            loaded = SAE(repo=ctx.repo["id"], path=sae["folder"], revision=ctx.sha, params=params)
            fit = measure(backend, loaded, text)
            log(
                f"  {check.name}: weights processed {processed}: variance explained "
                f"{fit['variance_explained']:.3f}, {fit['l0']:.1f} features per token"
            )
            best = checked["fit"]
            if best is None or fit["variance_explained"] > best["variance_explained"]:
                checked["fit"] = {"model": RUNNABLE[model_id], "process_weights": processed, **fit}
    shutil.rmtree(root, ignore_errors=True)
    return checked


def load(model_id: str, processed: bool, models: dict[tuple[str, bool], Any]) -> Any:
    """The model (or its ungated copy), loaded as Logogram loads it, once; None if it can't be."""
    key = (model_id, processed)
    if key not in models:
        for other in [k for k in models if k[0] != model_id]:
            loaded = models.pop(other)
            if loaded is not None:
                loaded.close()
        copy = RUNNABLE[model_id]
        if copy != model_id and not same_weights(model_id, copy):
            log(f"  {copy}'s weights aren't {model_id}'s; not run")
            models[key] = None
            return None
        from logogram.api import load_model
        from logogram.backends.base import BackendError

        try:
            models[key] = load_model(copy, device="cpu", process_weights=processed)
        except BackendError as exc:
            log(f"  {copy} with weight processing {processed} doesn't load: {exc}")
            models[key] = None
    return models[key]


def same_weights(original: str, copy: str) -> bool:
    """Whether a copy's safetensors files have the original's SHA-256 (Hugging Face lists the
    hashes of a gated repository's files without access to them)."""
    from huggingface_hub import HfApi

    def hashes(repo: str) -> dict[str, str]:
        info = HfApi().model_info(repo, files_metadata=True)
        return {
            s.rfilename: s.lfs.sha256
            for s in info.siblings or []
            if s.rfilename.endswith(".safetensors") and s.lfs is not None
        }

    theirs = hashes(original)
    return bool(theirs) and theirs == hashes(copy)


WINDOW = 64


def measure(backend: Any, sae: Any, text: str) -> dict[str, Any]:
    """The SAE's fit on the model's activations over the first 1,024 tokens of the text, read in
    windows of 64 tokens, each begun with the model's beginning-of-sequence token where it has one
    (prompt-length, as SAEs are trained on short contexts and as the Features view reads prompts),
    from each window's first real token on."""
    import torch

    from logogram.sae import fit_on

    bos = backend.bos_token_id is not None
    ids = backend.tokenize(text, prepend_bos=False).ids[:1024]
    start = 1 if bos else 0
    step = WINDOW - start
    rows = [
        ([backend.bos_token_id] if bos else []) + ids[i : i + step]
        for i in range(0, len(ids) - step + 1, step)
    ]
    tokens = torch.tensor(rows, device=backend.device)
    reads, writes = (sae.site_in, sae.layer), (sae.site, sae.layer)
    with backend.lock, torch.no_grad():
        captured = backend.capture(tokens, list(dict.fromkeys([reads, writes])))
    acts = sae.reads(captured[reads][:, start:].float())
    acts = acts.reshape(-1, acts.shape[-1])
    targets = None
    if sae.transcoder:
        out = sae.flat(captured[writes][:, start:].float())
        targets = out.reshape(-1, out.shape[-1])
    fit = fit_on(sae, acts, targets)
    return {
        "tokens": int(fit["tokens"]),
        "variance_explained": round(float(fit["variance_explained"]), 4),
        "l0": round(float(fit["l0"]), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Write the catalogue from a sae-lens wheel or installation.")
    b.add_argument("--sae-lens", required=True, help="A sae-lens wheel, or where it is installed.")
    b.add_argument("--cache", required=True, help="A folder for what is read from Hugging Face.")
    b.add_argument("--out", default=str(OUT), help="The catalogue to write.")
    b.add_argument(
        "--refresh", action="store_true", help="Read the repositories' current commits again."
    )
    v = sub.add_parser("verify", help="Check a sample of the catalogue on the real files.")
    v.add_argument("--cache", required=True, help="The folder the build read into.")
    v.add_argument("--out", default=str(OUT), help="The catalogue to check and record in.")
    v.add_argument("--per-release", type=int, default=2, help="SAEs per release to check.")
    v.add_argument("--budget-mb", type=int, default=2500, help="Weights to download, in MiB.")
    v.add_argument("--seed", type=int, default=0, help="Seeds the choice of SAEs.")
    v.add_argument("--text", help="A plain-text file to measure fits on.")
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
