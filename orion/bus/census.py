from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "CensusResult",
    "compute_census",
    "load_channel_catalog_names",
    "normalize_channel_name",
]

_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "channels.yaml"


@dataclass(frozen=True)
class CensusResult:
    declared_silent: list[str]
    undeclared_active: list[str]


def _is_wildcard(name: str) -> bool:
    return name.endswith("*")


def _wildcard_prefix(name: str) -> str:
    return name[:-1]


def compute_census(catalog_names: set[str], active_channels: dict[str, float]) -> CensusResult:
    """Diff the declared channel catalog against channels observed to be
    carrying real traffic (Phase 1's velocity counters).

    Wildcard-aware: ~25 entries in orion/bus/channels.yaml are prefix
    patterns (e.g. "orion:exec:result:*", "orion:cortex:result*" -- both the
    ":*" and bare "*" suffix forms appear) covering dynamic per-request reply
    channels like "orion:exec:result:LLMGatewayService:<uuid>". Matching
    catalog_names by exact string equality only would put every one of those
    reply channels in undeclared_active as a false positive, since no two
    request/reply cycles share a literal channel name.

    A wildcard catalog entry counts as silent only if nothing in
    active_channels matches its prefix -- not just its own literal string,
    since a wildcard entry never appears verbatim as a live channel name.

    Bare-star entries (e.g. "orion:cortex:result*", with no colon before the
    star) produce a looser prefix than colon-star entries (e.g.
    "orion:exec:result:*") -- "orion:cortex:result*" would also match a
    hypothetical unrelated "orion:cortex:resultset" channel. This faithfully
    reflects what channels.yaml itself declares; no colliding literal exists
    in the catalog today.
    """
    declared_silent: list[str] = []
    covered_active: set[str] = set()
    for name in sorted(catalog_names):
        if _is_wildcard(name):
            prefix = _wildcard_prefix(name)
            matches = [ch for ch in active_channels if ch.startswith(prefix)]
        else:
            matches = [name] if name in active_channels else []
        covered_active.update(matches)
        if not matches:
            declared_silent.append(name)

    undeclared_active = sorted(ch for ch in active_channels if ch not in covered_active)
    return CensusResult(declared_silent=declared_silent, undeclared_active=undeclared_active)


def load_channel_catalog_names(catalog_path: str | Path | None = None) -> set[str]:
    """Load every declared channel name (exact and wildcard) from
    orion/bus/channels.yaml. Defaults to the file living right next to this
    module -- __file__-relative, so resolution doesn't depend on the calling
    process's CWD or which service imports this (unlike
    services/orion-bus/app/bus_observer.py's own near-identical loader, which
    needs a multi-base-path fallback because it's reaching for a file outside
    its own service directory; this one never has to reach anywhere).

    A near-duplicate of that function exists in bus_observer.py -- not
    consolidated here to avoid a cross-service refactor out of scope for this
    patch, but any new consumer of catalog names should prefer this shared
    orion.bus.census version over reaching into another service's internals.
    """
    path = Path(catalog_path) if catalog_path else _DEFAULT_CATALOG_PATH
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names: set[str] = set()
    for ch in data.get("channels") or []:
        if isinstance(ch, dict) and ch.get("name"):
            names.add(str(ch["name"]))
    return names


def normalize_channel_name(channel: str, catalog_names: set[str]) -> str:
    """Collapse a live channel name to its matching wildcard catalog entry,
    if any, else return it unchanged.

    Exists to fix a real, live-found problem: a consumer that creates one
    graph node (or any other per-channel record) per literal channel string
    it observes -- without this normalization -- creates one node per dynamic
    per-request reply channel (e.g.
    "orion:exec:result:LLMGatewayService:<uuid>") instead of one node per
    real, bounded catalog entry. Found live 2026-07-24: services/orion-bus-mirror's
    synaptic graph had ~9K Channel nodes against a 264-entry declared catalog,
    almost entirely from this exact pattern.

    If more than one wildcard matches (e.g. both a broad umbrella like
    "orion:exec:result:*" and a narrower per-service pattern like
    "orion:exec:result:LLMGatewayService:*" -- both real, both present in
    channels.yaml today), the narrowest (longest prefix) match wins -- the
    more specific grouping is the more useful one for anything downstream
    that cares which service's replies are hot, not just that "some" reply
    channel is.

    An exact literal catalog entry is checked FIRST and short-circuits any
    wildcard matching, even if a wildcard sibling would also prefix-match it
    -- caught in review: channels.yaml has real cases (e.g.
    "orion:exec:result:LLMGatewayService" as its own standalone entry,
    alongside the "orion:exec:result:*" umbrella and/or a
    "...LLMGatewayService:*" wildcard) where a literal, already-bounded,
    already-declared channel would otherwise get merged into a broader
    bucket -- exactly backwards from why this function exists. A channel
    that's already a real catalog entry needs no collapsing at all.
    """
    if channel in catalog_names:
        return channel
    matches = [
        name
        for name in catalog_names
        if _is_wildcard(name) and channel.startswith(_wildcard_prefix(name))
    ]
    if not matches:
        return channel
    return max(matches, key=lambda name: len(_wildcard_prefix(name)))
