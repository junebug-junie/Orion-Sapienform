from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CensusResult", "compute_census"]


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
