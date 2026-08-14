"""Startup gate: every channel this service subscribes to must have a route.

## Why this exists

Twice on 2026-08-13/14, `orion-sql-writer` ran for hours subscribed to a live
channel it had no route for. Every event on that channel was written to
``bus_fallback_log`` instead of its table. Nothing looked wrong:

- the container was healthy and had restarted cleanly,
- the producer was ticking on schedule and logging real values,
- ``PUBSUB NUMSUB`` reported a live subscriber,
- no error was raised -- ``handle_envelope`` logs a single WARNING per event
  and moves on.

Both incidents were found only because someone happened to query row counts.
The cause was mundane and will recur: the subscribe list lives in a durable,
branch-independent ``.env``, while the route lives in code. Deploy this
service from a branch that predates a new route -- which any concurrent
session can do, and this repo has 100+ active worktrees -- and the two fall
out of step silently.

Root ``CLAUDE.md`` is explicit that the fix for a failure seen twice is a
deterministic gate, not a louder reminder. This is that gate.

## What it checks

For each channel in ``effective_subscribe_channels``, resolve its
``message_kind`` from ``orion/bus/channels.yaml`` and confirm the kind is
handled by EITHER of the two real dispatch paths in ``handle_envelope``:

1. ``settings.route_map`` -> a SQL model, or
2. ``orion.evidence_index.ingest._ADAPTERS_BY_KIND`` -> the "adapter-only"
   branch that writes ``evidence_units``.

Both are required. A first version of this gate checked only ``route_map``
and immediately flagged ``orion:evidence:markdown:ingest`` and
``orion:evidence:parsed:ingest`` -- which are perfectly healthy, handled by
path 2. Confirmed against real data rather than assumed: both kinds have ZERO
rows in ``bus_fallback_log`` while ``evidence_units`` held 20,248 rows updated
seconds earlier. A gate that cries wolf on working channels is worse than no
gate, because it teaches people to skip the output -- which is how the
original silent WARNING went unnoticed for hours in the first place.

Deliberately NOT fatal. Refusing to boot would turn a partial data-loss window
into a total outage, and would let one stale channel entry take down
persistence for the other 69. It logs at ERROR with the specific kinds, which
is what turns a silent failure into a visible one -- the whole point.

Channels with no ``message_kind`` in the registry are skipped, not flagged:
several are pattern/broadcast channels whose payload kind is not declared
there, and flagging them would produce standing noise that trains people to
ignore this check -- the same way the silent WARNING was already being ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("sql-writer.route_coverage")

_CHANNELS_YAML_RELATIVE = Path("orion") / "bus" / "channels.yaml"


def _find_channels_yaml() -> Path | None:
    """Locate orion/bus/channels.yaml by walking UP from this file.

    Not a fixed `parents[N]`. The repo layout puts this module at
    services/orion-sql-writer/app/ (depth 3 from the root) while the image
    copies it to /app/app/ with the registry at /app/orion/ (depth 1). A
    hardcoded depth is correct in exactly one of those and an IndexError in
    the other -- which is precisely how the first version of this file
    crash-looped the service on deploy: `parents[3]` raised at IMPORT time,
    taking down the whole worker before it could log anything.

    Resolved lazily, per call, for the same reason. A startup diagnostic must
    not be able to break startup, and module-level path arithmetic can.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _CHANNELS_YAML_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def _channel_kinds(channels_yaml: Path | None = None) -> dict[str, str]:
    """{channel_name: message_kind} for every channel that declares one."""
    import yaml

    path = channels_yaml or _find_channels_yaml()
    if path is None:
        logger.warning("route_coverage_registry_not_found relative=%s", _CHANNELS_YAML_RELATIVE)
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        # The registry is not readable from wherever this is running. Return
        # empty rather than raise: this is a diagnostic, and it must never be
        # the reason the service fails to start.
        logger.warning("route_coverage_registry_unreadable path=%s", path)
        return {}
    kinds: dict[str, str] = {}
    for entry in doc.get("channels") or []:
        if not isinstance(entry, dict):
            continue
        name, kind = entry.get("name"), entry.get("message_kind")
        if name and kind:
            kinds[name] = kind
    return kinds


def _adapter_kinds() -> set[str]:
    """Kinds handled by the evidence-unit adapter path.

    Read from the real registry rather than hardcoded, so a newly-registered
    adapter does not become a false positive here the day it ships.
    """
    try:
        from orion.evidence_index.ingest import _ADAPTERS_BY_KIND

        return set(_ADAPTERS_BY_KIND)
    except Exception:
        logger.warning("route_coverage_adapter_registry_unreadable")
        return set()


def find_unrouted_channels(
    subscribed: list[str],
    route_map: dict[str, str],
    channels_yaml: Path | None = None,
    adapter_kinds: set[str] | None = None,
) -> list[tuple[str, str]]:
    """[(channel, message_kind)] for subscribed channels no dispatch path handles.

    Pure and injectable so the gate is testable without a live registry, a
    live settings object, or a live database.
    """
    kinds = _channel_kinds(channels_yaml)
    handled_by_adapter = _adapter_kinds() if adapter_kinds is None else adapter_kinds
    unrouted = []
    for channel in subscribed:
        kind = kinds.get(channel)
        if kind is None:
            continue  # not declared in the registry -- see module docstring
        if kind not in route_map and kind not in handled_by_adapter:
            unrouted.append((channel, kind))
    return unrouted


def log_route_coverage(
    subscribed: list[str],
    route_map: dict[str, str],
    channels_yaml: Path | None = None,
    adapter_kinds: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Emit the startup verdict. Returns the offenders so callers can assert."""
    unrouted = find_unrouted_channels(subscribed, route_map, channels_yaml, adapter_kinds)
    if not unrouted:
        logger.info("route_coverage_ok subscribed=%d all_routed=true", len(subscribed))
        return []
    for channel, kind in unrouted:
        logger.error(
            "route_coverage_missing channel=%s kind=%s -- events on this channel will be "
            "written to bus_fallback_log instead of their table. This service is almost "
            "certainly running an image that predates the route; rebuild and redeploy it.",
            channel,
            kind,
        )
    logger.error(
        "route_coverage_summary unrouted=%d of subscribed=%d", len(unrouted), len(subscribed)
    )
    return unrouted
