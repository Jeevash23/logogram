"""The catalogue of published SAEs: SAELens's registry, and what Logogram reads of each SAE in it.

People find SAEs by their SAELens release and id (``gpt2-small-res-jb``,
``blocks.8.hook_resid_pre``), while Logogram loads them from a Hugging Face repository and folder.
``sae_catalogue.json`` beside this module maps one to the other for every SAE in SAELens's
pretrained-SAE registry, and records, for each, the hook it reads, its width and the architecture
its configuration states, and whether Logogram reads it exactly or, if not, why, as Logogram's own
readers decide from its configuration (:func:`logogram.backends.saes.read_config`). Some were also
checked on their files; those say how and at which commit.

``scripts/sae_catalogue.py`` writes it from a sae-lens wheel or installation and the SAEs'
configurations on Hugging Face. Logogram itself only reads it: nothing here touches the network.
A spec still records the repository, folder and revision an SAE was loaded from, never its release.

SAELens's registry is published under the MIT licence; its copyright and permission notice travel
with the catalogue (``source.notice``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from logogram.backends.base import BackendError

CATALOGUE_FILE = "sae_catalogue.json"
Reads = Literal["exact", "refused", "unknown"]
Kind = Literal["sae", "transcoder", "skip_transcoder"]
# Sites in words, for searching ("residual", "attention", "mlp", "heads").
SITE_WORDS = {
    "resid_pre": "residual stream before the layer resid_pre",
    "resid_mid": "residual stream between attention and mlp resid_mid",
    "resid_post": "residual stream after the layer resid_post",
    "attn_out": "attention output attn_out",
    "mlp_out": "mlp output mlp_out",
    "mlp_in": "mlp input mlp_in",
    "head": "attention heads' outputs z head",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Fit(_Strict):
    """How well it fit the model's activations when it was checked: the share of their variance
    it explains and how many features fire per token, over this many tokens of plain text."""

    model: str = Field(min_length=1)
    process_weights: bool
    tokens: int = Field(ge=1)
    variance_explained: float
    l0: float = Field(ge=0)


class Verified(_Strict):
    """A check on the SAE's files at one commit. ``header``: its weights file's header (the name
    and shape of every parameter) read as Logogram reads the weights, agreeing with the
    configuration. ``weights``: its files downloaded and read whole. ``fit`` when it was also
    run on its model."""

    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    read: Literal["header", "weights"]
    fit: Fit | None = None


class StoredSAE(_Strict):
    id: str = Field(min_length=1)
    folder: str = Field(min_length=1)
    hook: str | None
    site: str | None
    layer: int | None = Field(ge=0)
    width: int | None = Field(ge=1)
    architecture: str | None
    kind: Kind
    reads: Reads
    # An index into the catalogue's reasons: why it isn't read exactly (or why that is unknown).
    why: int | None = Field(default=None, ge=0)
    # Settings SAELens's registry reads it with that change what it computes and that its own
    # configuration doesn't state: its files alone don't say what it computes.
    registry: dict[str, Any] | None = None
    l0: float | None = None
    verified: Verified | None = None


class Release(_Strict):
    release: str = Field(min_length=1)
    # The model as SAELens's registry names it, and the Hugging Face ids it goes by (the
    # repository's own first).
    model: str = Field(min_length=1)
    hf_models: list[str] = Field(min_length=1)
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    # The commit whose files the configurations were read from; None if they couldn't be read.
    revision: str | None = Field(pattern=r"^[0-9a-f]{40}$")
    licence: str | None
    gated: bool
    saes: list[StoredSAE] = Field(min_length=1)


class Source(_Strict):
    package: Literal["sae-lens"]
    version: str = Field(min_length=1)
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    licence: str = Field(min_length=1)
    notice: str = Field(min_length=1)


class CatalogueFile(_Strict):
    catalogue: Literal[1]
    source: Source
    reasons: list[str]
    releases: list[Release]


@dataclass(frozen=True)
class Entry:
    """One SAE of the catalogue, with what its release says of it."""

    release: str
    id: str
    model: str
    hf_models: tuple[str, ...]
    repo: str
    folder: str
    revision: str | None
    licence: str | None
    gated: bool
    hook: str | None
    site: str | None
    layer: int | None
    width: int | None
    architecture: str | None
    kind: str
    reads: str
    reason: str | None
    registry: dict[str, Any] | None
    l0: float | None
    verified: Verified | None

    @property
    def exact(self) -> bool:
        return self.reads == "exact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": self.release,
            "id": self.id,
            "model": self.model,
            "hf_model": self.hf_models[0],
            "repo": self.repo,
            "folder": self.folder,
            "revision": self.revision,
            "licence": self.licence,
            "gated": self.gated,
            "hook": self.hook,
            "site": self.site,
            "layer": self.layer,
            "width": self.width,
            "architecture": self.architecture,
            "kind": self.kind,
            "reads": self.reads,
            "reason": self.reason,
            "l0": self.l0,
            "verified": None if self.verified is None else self.verified.model_dump(),
        }

    def haystack(self) -> str:
        """What a search reads: release, id, folder, hook, site, layer, width and architecture."""
        parts = [self.release, self.id, self.folder, self.hook or "", self.architecture or ""]
        if self.site:
            parts.append(SITE_WORDS.get(self.site, self.site))
        if self.layer is not None:
            parts.append(f"layer {self.layer}")
        if self.width is not None:
            parts.append(str(self.width))
        parts.append(self.kind.replace("_", " "))
        return " ".join(parts).lower()


@dataclass(frozen=True)
class Catalogue:
    source: Source
    entries: tuple[Entry, ...]

    def for_model(self, model_id: str) -> list[Entry]:
        """The SAEs trained on a model, matched by its Hugging Face id (any it goes by, in any
        case). A model loaded from a folder, or a copy under another name, matches nothing."""
        wanted = model_id.strip().casefold()
        return [e for e in self.entries if wanted in (m.casefold() for m in e.hf_models)]

    def find(self, release: str, sae_id: str) -> Entry:
        """The SAE with this release and id, as SAELens names it."""
        in_release = [e for e in self.entries if e.release == release]
        if not in_release:
            close = sorted({e.release for e in self.entries if release.lower() in e.release})
            hint = f" Did you mean {_listed(close[:5])}?" if close else ""
            raise BackendError(
                f"SAELens {self.source.version}'s registry has no release {release!r}.{hint} "
                "List the SAEs for a model with `logogram saes MODEL`."
            )
        for e in in_release:
            if e.id == sae_id:
                return e
        ids = [e.id for e in in_release]
        shown = _listed(ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")
        raise BackendError(f"The release {release} has no SAE {sae_id!r}. Its ids: {shown}.")

    def at(self, repo: str, folder: str) -> list[Entry]:
        """The catalogued SAEs published at this repository and folder (one, or several
        releases' names for the same files)."""
        key = (repo.casefold(), folder.strip("/"))
        return [e for e in self.entries if (e.repo.casefold(), e.folder.strip("/")) == key]


def _listed(names: list[str]) -> str:
    quoted = [repr(n) for n in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def parse_catalogue(data: dict[str, Any]) -> Catalogue:
    """A catalogue from its JSON, checked against its schema."""
    stored = CatalogueFile.model_validate(data)
    entries = []
    for r in stored.releases:
        for s in r.saes:
            if s.why is not None and s.why >= len(stored.reasons):
                raise ValueError(f"{r.release}/{s.id} names a reason the catalogue doesn't have")
            if (s.reads == "exact") != (s.why is None):
                raise ValueError(f"{r.release}/{s.id}: an SAE not read exactly needs a reason")
            entries.append(
                Entry(
                    release=r.release,
                    id=s.id,
                    model=r.model,
                    hf_models=tuple(r.hf_models),
                    repo=r.repo,
                    folder=s.folder,
                    revision=r.revision,
                    licence=r.licence,
                    gated=r.gated,
                    hook=s.hook,
                    site=s.site,
                    layer=s.layer,
                    width=s.width,
                    architecture=s.architecture,
                    kind=s.kind,
                    reads=s.reads,
                    reason=None if s.why is None else stored.reasons[s.why],
                    registry=s.registry,
                    l0=s.l0,
                    verified=s.verified,
                )
            )
    return Catalogue(source=stored.source, entries=tuple(entries))


@lru_cache(maxsize=1)
def load_catalogue() -> Catalogue:
    """The catalogue that ships with Logogram (read once, from the installed package)."""
    text = resources.files("logogram.backends").joinpath(CATALOGUE_FILE).read_text("utf-8")
    return parse_catalogue(json.loads(text))


# -- search ------------------------------------------------------------------------------------

_LAYER = re.compile(r"^(?:layer|l)[ _=:-]?(\d+)$")
_WIDTH = re.compile(r"^(\d+(?:\.\d+)?)([km])$")


def _width_matches(width: int | None, number: float, unit: str) -> bool:
    """Whether "16k", "65k" or "1m" names this width: in thousands, cut short or rounded (as
    folder names write 65,536 as 65k), or in powers of two (64k)."""
    if width is None:
        return False
    scale = 1000 if unit == "k" else 1_000_000
    binary = 1024 if unit == "k" else 1024 * 1024
    return number in (width // scale, round(width / scale), round(width / binary))


def search(entries: list[Entry], query: str) -> list[Entry]:
    """The entries every word of the query matches. ``layer 8`` (or ``layer8``, ``l8``) keeps one
    layer, a width such as ``16k`` or ``1m`` keeps that width (in thousands or powers of two), a
    number alone keeps that layer or that width, and any other word must appear in the release,
    id, folder, hook, site, architecture or kind."""
    words = query.lower().replace(",", " ").split()
    tests = []
    i = 0
    while i < len(words):
        word = words[i]
        if word == "layer" and i + 1 < len(words) and words[i + 1].isdigit():
            word = f"layer{words[i + 1]}"
            i += 1
        i += 1
        layer, width = _LAYER.match(word), _WIDTH.match(word)
        if layer:
            n = int(layer.group(1))
            tests.append(lambda e, n=n: e.layer == n)
        elif width:
            number, unit = float(width.group(1)), width.group(2)
            tests.append(
                lambda e, w=word, x=number, u=unit: (
                    _width_matches(e.width, x, u) or w in e.haystack()
                )
            )
        elif word.isdigit():
            n = int(word)
            tests.append(lambda e, n=n: e.layer == n or e.width == n)
        else:
            tests.append(lambda e, w=word: w in e.haystack())
    return [e for e in entries if all(t(e) for t in tests)]


def order(entries: list[Entry]) -> list[Entry]:
    """Those Logogram reads exactly first, then those it can't tell, then those it refuses; in
    the registry's order otherwise."""
    rank = {"exact": 0, "unknown": 1, "refused": 2}
    return sorted(entries, key=lambda e: rank[e.reads])


def registry_reason(release: str, sae_id: str, settings: dict[str, Any]) -> str:
    """Why an SAE can't be read exactly when SAELens's registry reads it with ``settings`` that
    its own configuration doesn't state."""
    stated = ", ".join(f"{k} = {v!r}" for k, v in sorted(settings.items()))
    return (
        f"SAELens's registry reads this SAE ({release}, {sae_id}) with {stated}, which its own "
        "configuration doesn't state, so its files alone don't say what it computes."
    )


def registry_refusal(repo: str, folder: str) -> str | None:
    """Why an SAE at this repository and folder can't be read exactly from its files alone, when
    SAELens's registry reads it with settings its configuration doesn't state; None otherwise."""
    for entry in load_catalogue().at(repo, folder):
        if entry.registry:
            return registry_reason(entry.release, entry.id, entry.registry)
    return None
