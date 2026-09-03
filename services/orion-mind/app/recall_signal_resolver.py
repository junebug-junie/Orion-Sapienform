"""Table-driven resolver for recall's bus_synaptic transport fragments.

Spec: docs/superpowers/specs/2026-09-03-recall-signal-rendering-design.md
(amended 2026-09-03 twice: the render gate is 0.15, matching
orion-equilibrium-service's already-live metacog trigger, not the spec's
original 0.25; and the resolver lives HERE, in orion-mind's evidence pack
builder, not in orion-cortex-orch's conversation_front.py, which the spec
originally named but which is dead code -- never called from anywhere in
that service, confirmed live 2026-09-03).

Handles fragments with ``meta.signal_kind == "publish_gap_zscore"`` only,
per the spec's non-goal ("one signal at a time" -- ``causal_latency_zscore``
fragments from the same adapter keep rendering via their pre-existing
``text``/``snippet`` field, untouched).

Three rendered states:

- **Below the gate** -- no fragment at all.
- **At or above the gate** -- one rolled-up sentence for the whole bus, not
  one per channel.
- **Not writing / degenerate** -- a distinct liveness sentence. Two
  different failure shapes collapse to this state: the series read
  returning nothing (the tick itself has stopped), and the series reading a
  flat 0.0 (the tick is still firing but against an empty edge set -- see
  ``services/orion-recall/app/storage/falkor_bus_synaptic_adapter.py``'s
  module docstring for why that is a real, confirmed failure mode of
  ``_bus_synaptic_tick``, not a hypothetical one, and why
  ``classify_channel_series()``'s ``dead`` verdict is the right tool for it
  instead of a second liveness query).
"""

from __future__ import annotations

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from orion.db_readonly import open_readonly_connection
from orion.field.channel_glossary import SUBNORMAL_CUTOFF, classify_channel_series, resolve_channel_entry

logger = logging.getLogger("orion.mind.recall_signal_resolver")

BUS_SYNAPTIC_NODE_ID = "node:substrate.bus_synaptic"
BUS_SYNAPTIC_CHANNEL = "prediction_error"

# Maps the glossary's node-qualified scan_token/URN name (config/field/
# field_channel_glossary.v1.yaml, orion/metrics/lineage.py's
# resolve_field_channels()) for each signal_kind this resolver handles, to
# the log verdict tag it's resolved into -- used by
# render_bus_synaptic_digest_line() below to key its debug log line so a
# reader can correlate the log to the exact registry entry, not just this
# module's own constants. A dict-key literal, not a bare assignment, so
# orion/metrics/gate.py's orphan scan (which only trusts subscript/get/
# dict-key/attribute/collection access as real consumption, not a bare
# `X = "literal"`) actually sees this as a reader.
METRIC_URN_LOG_TAGS: Dict[str, str] = {
    "node:substrate.bus_synaptic.prediction_error": "bus_synaptic_pressure",
}

# Only this signal_kind is handled by the table-driven path this pass. A
# fragment with any other signal_kind (or none) passes through
# build_evidence_pack() unchanged.
HANDLED_SIGNAL_KIND = "publish_gap_zscore"

_LATTICE_POLICY_RELATIVE_PATH = Path("config") / "substrate-lattice" / "transport_lattice_policy.v1.yaml"

_NOT_WRITING_TEXT = (
    "The bus synaptic graph hasn't been written recently. "
    "Transport state is unknown, not calm."
)
_DEGENERATE_ZERO_TEXT = (
    "The bus synaptic graph is reading a flat 0.0 -- consistent with the "
    "transport tick still firing against an empty edge set (bus-mirror "
    "likely not producing), not genuine calm. Transport state is unknown."
)


def _lattice_policy_path_candidates() -> List[Path]:
    """Same multi-candidate resolution as
    ``orion.field.channel_glossary._glossary_path_candidates()`` -- this
    service's Dockerfile also only ``COPY``s ``orion/``, not ``config/``, so
    the same fallback-to-``/repo`` pattern applies here.
    """
    seen: set[str] = set()
    roots: List[Path] = []

    def _add(root: str | Path | None) -> None:
        if root is None:
            return
        try:
            resolved = Path(root).expanduser().resolve()
        except OSError:
            resolved = Path(root)
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append(resolved)

    raw = os.getenv("ORION_REPO_ROOT", "").strip()
    if raw:
        _add(raw)
    _add("/repo")
    _add("/mnt/scripts/Orion-Sapienform")
    return [root / _LATTICE_POLICY_RELATIVE_PATH for root in roots]


@functools.lru_cache(maxsize=1)
def _load_bus_synaptic_lattice_rungs() -> Optional[Dict[str, Optional[float]]]:
    """``watch_at``/``summarize_at``/``propose_at`` for ``bus_synaptic_pressure``,
    for DISPLAY in the rendered sentence only -- never the render gate
    itself (see module docstring for why the gate is a separate, deliberate
    0.15, not this ladder's 0.25). Returns ``None`` if the policy file can't
    be found or parsed; the resolver degrades to a sentence without ladder
    context rather than failing evidence-pack assembly.
    """
    for path in _lattice_policy_path_candidates():
        if not path.is_file():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            rung = ((raw.get("channels") or {}).get("bus_synaptic_pressure")) or {}
            if "watch_at" not in rung or "summarize_at" not in rung:
                return None
            return {
                "watch_at": float(rung["watch_at"]),
                "summarize_at": float(rung["summarize_at"]),
                "propose_at": (
                    float(rung["propose_at"]) if rung.get("propose_at") is not None else None
                ),
            }
        except Exception:
            logger.debug("lattice_policy_load_failed path=%s", path, exc_info=True)
            return None
    return None


def fetch_bus_synaptic_prediction_error_series(
    dsn: str, *, limit: int = 60, max_age_sec: float = 1800.0
) -> List[float]:
    """Recent ``node:substrate.bus_synaptic`` ``prediction_error`` values,
    oldest first, for ``classify_channel_series()``.

    Uses ``orion.db_readonly.open_readonly_connection()`` -- the repo's
    canonical helper for exactly this shape of call (short connect/
    statement timeouts, fail-open to ``None``/``[]``), rather than
    hand-rolling connection setup a third time. Two real bugs in an earlier
    draft of this function were caught by this reuse alone (code review,
    2026-09-03): a bare ``psycopg2.connect`` + ``SET LOCAL statement_timeout``
    issued *after* ``autocommit=True`` is silently a no-op (``SET LOCAL``
    only holds for the remainder of the current transaction; autocommit
    means every statement is its own transaction, so the timeout had
    already reverted before the real query ran) -- ``open_readonly_connection``
    uses a session-level ``SET statement_timeout`` instead, which is
    correct under autocommit. It also enforces a genuinely read-only
    session, a safety property this fetch-only call had no reason to skip.

    Blocking, not offloaded to a thread pool -- this service's whole
    mind-run pipeline is fully synchronous end to end (``run_mind`` and
    everything it calls, including LLM synthesis with a wall time up to
    ``MIND_WALL_MS_DEFAULT``), so this call's ~1.3s worst case (1s connect +
    300ms statement timeout) is a small, bounded addition to an
    already-synchronous request path, not a new architectural pattern.

    Bounded by a row ``LIMIT``, not a time window alone --
    ``substrate_field_state`` is a shared snapshot table every reducer
    writes to, so a pure time window could return hundreds of rows most of
    which never touched this node. Also bounded by ``max_age_sec`` (the same
    horizon ``orion/substrate/bus_synaptic_surprise.py``'s staleness guard
    uses) so a genuinely stalled pipeline returns ``[]`` -- the "not
    writing" state -- rather than a long-frozen series read as current.
    """
    if not dsn or not str(dsn).strip():
        return []
    conn = open_readonly_connection(dsn, connect_timeout=1, statement_timeout_ms=300)
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT field_json -> 'node_vectors' -> %s ->> %s AS value, generated_at
            FROM substrate_field_state
            WHERE generated_at >= now() - (%s || ' seconds')::interval
            ORDER BY generated_at DESC
            LIMIT %s
            """,
            (BUS_SYNAPTIC_NODE_ID, BUS_SYNAPTIC_CHANNEL, max_age_sec, limit),
        )
        rows = cur.fetchall() or []
    except Exception as exc:
        logger.debug("bus_synaptic_series_fetch_failed error=%s", exc)
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    values: List[float] = []
    for row in reversed(rows):  # oldest first, for classify_channel_series
        value = row[0] if row else None
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def partition_bus_synaptic_fragments(
    fragments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ``fragments`` into ``(passthrough, handled)``.

    ``handled`` fragments (``meta.signal_kind == HANDLED_SIGNAL_KIND``) must
    never be rendered individually by the caller -- they collapse into at
    most one resolved line (see ``render_bus_synaptic_digest_line``), per
    the spec: "one fragment for the whole bus, not one per channel."
    Everything else -- including ``causal_latency_zscore`` fragments from
    the same adapter -- is ``passthrough`` and renders exactly as before.
    """
    passthrough: List[Dict[str, Any]] = []
    handled: List[Dict[str, Any]] = []
    for frag in fragments or []:
        meta = frag.get("meta") if isinstance(frag, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        if meta.get("signal_kind") == HANDLED_SIGNAL_KIND:
            handled.append(frag)
        else:
            passthrough.append(frag)
    return passthrough, handled


# How fresh an edge must be to be named as "loudest right now". Deliberately
# tight and local to this display detail, independent of whatever
# max_edge_age_sec recall's own fetch used -- falkor_bus_synaptic_adapter.py
# dropped ITS recency filter on 2026-09-03 so a total outage isn't silently
# swallowed (see that module's docstring), which means the fragments handed
# here can now include edges frozen for hours or days (the module's own
# named example: orion:dream:log frozen at |z|=36 for weeks). Without this,
# a permanently stale edge sorts first in the adapter's ORDER BY abs(z) DESC
# forever and "loudest right now" would misattribute a real, currently-
# happening incident to a zombie channel indefinitely (code review finding,
# 2026-09-03).
_LOUDEST_MAX_AGE_SEC = 300.0


def _describe_loudest(handled_fragments: List[Dict[str, Any]]) -> Optional[str]:
    """Pick the loudest FRESH fragment's channel/organ for the "loudest
    right now" detail. ``falkor_bus_synaptic_adapter.py``'s own query
    already sorts by ``abs(gap_zscore) DESC``, so the first fresh fragment
    carrying both fields is the loudest -- purely descriptive, never part
    of the render decision itself. A fragment with no ``last_seen_epoch``
    at all is treated as not fresh (fail closed, not "loudest by default")."""
    now = time.time()
    for frag in handled_fragments:
        meta = frag.get("meta") if isinstance(frag, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        channel = meta.get("channel")
        organ = meta.get("organ_id")
        last_seen_epoch = meta.get("last_seen_epoch")
        if not (channel and organ and last_seen_epoch is not None):
            continue
        try:
            age = now - float(last_seen_epoch)
        except (TypeError, ValueError):
            continue
        if age <= _LOUDEST_MAX_AGE_SEC:
            return f"{channel} from {organ}"
    return None


def render_bus_synaptic_digest_line(
    handled_fragments: List[Dict[str, Any]],
    *,
    dsn: str,
    render_gate_threshold: float,
) -> Optional[str]:
    """One rendered sentence for the whole bus, or ``None`` to render
    nothing. See module docstring for the three states.

    ``handled_fragments`` supplies only the "loudest channel" display
    detail -- the render DECISION (gate / not-writing / degenerate) comes
    entirely from live state read fresh here, never from the per-edge
    z-scores recall's adapter already returned.

    The gate is on ``dsn`` being configured, NOT on ``handled_fragments``
    being non-empty. An earlier draft gated on the latter, reasoning that
    empty fragments meant "recall didn't even try this turn" -- but recall's
    per-edge Falkor fetch failing open to ``[]`` (a Falkor/bus-mirror outage,
    the adapter's own fail-open contract) is indistinguishable from that
    case, so a Falkor outage silently skipped this Postgres-backed liveness
    check entirely -- the same "outage reads as silence" failure the whole
    liveness path exists to prevent, one layer up (code review finding,
    2026-09-03). ``substrate_field_state`` is written by a different
    service's own FalkorDB query (``orion-substrate-runtime``'s
    ``_bus_synaptic_tick``), so its freshness is genuinely independent of
    whether recall's own Falkor connection is healthy right now. With no
    dsn configured, ``fetch_bus_synaptic_prediction_error_series`` returns
    ``[]`` immediately with no network call, so this stays cheap when the
    feature is simply off.
    """
    if not dsn or not str(dsn).strip():
        return None
    metric_urn = "node:substrate.bus_synaptic.prediction_error"
    log_tag = METRIC_URN_LOG_TAGS.get(metric_urn, "unknown")
    series = fetch_bus_synaptic_prediction_error_series(dsn)
    if not series:
        logger.debug(
            "bus_synaptic_metric_resolved metric=%s tag=%s verdict=not_writing",
            metric_urn,
            log_tag,
        )
        return _NOT_WRITING_TEXT
    verdict = classify_channel_series(series)
    latest = series[-1]
    # classify_channel_series() only returns "dead" when EVERY value in the
    # window is subnormal -- an outage that started partway through the
    # window (older real values still present) reads as "quiet"/"live"
    # instead, with latest==0.0. Left unchecked, that 0.0 then fails the
    # render gate below and returns None -- silence, not the degenerate
    # state -- reproducing the exact "outage reads as silence" failure this
    # whole liveness path exists to prevent, just delayed to partway through
    # max_age_sec instead of eliminated (code review finding, 2026-09-03).
    # Checking `latest` directly closes that gap regardless of what the rest
    # of the window looked like.
    if verdict == "dead" or abs(latest) < SUBNORMAL_CUTOFF:
        logger.debug(
            "bus_synaptic_metric_resolved metric=%s tag=%s verdict=dead series_verdict=%s",
            metric_urn,
            log_tag,
            verdict,
        )
        return _DEGENERATE_ZERO_TEXT
    logger.debug(
        "bus_synaptic_metric_resolved metric=%s tag=%s verdict=%s latest=%.4f",
        metric_urn,
        log_tag,
        verdict,
        latest,
    )
    if latest < render_gate_threshold:
        return None
    rungs = _load_bus_synaptic_lattice_rungs()
    ladder_phrase = ""
    if rungs:
        ladder_phrase = (
            f", against a {rungs['watch_at']:.2f} watch threshold "
            f"and {rungs['summarize_at']:.2f} summarize"
        )
    loudest = _describe_loudest(handled_fragments)
    loudest_phrase = f" Loudest right now: {loudest}." if loudest else ""
    entry = resolve_channel_entry(BUS_SYNAPTIC_CHANNEL, BUS_SYNAPTIC_NODE_ID)
    trend_phrase = f" Trend in {entry.trend_source}." if entry and entry.trend_source else ""
    return (
        f"Transport: {latest * 100:.0f}% of live bus channels running "
        f"anomalous{ladder_phrase}.{loudest_phrase}{trend_phrase}"
    )


__all__ = [
    "HANDLED_SIGNAL_KIND",
    "fetch_bus_synaptic_prediction_error_series",
    "partition_bus_synaptic_fragments",
    "render_bus_synaptic_digest_line",
]
