"""Signal history: the five substrate nodes that are NOT prediction errors.

This module imports no third-party package itself, but importing it runs orion/substrate/__init__.py,
which does (pydantic and more).

Design: docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md (PR #3781).

Five nodes still write their value into the prediction-error slot, and none of them is a forecast
miss: three alarms (cabinet, strain, street) and two camera scene-change scores (room, walkway).
They move to their own table, ``substrate_node_signal_history``
(services/orion-sql-db/manual_migration_node_signal_history_v1.sql), with an honest ``kind``.

Patch 1 (this module's first use) is additive:
  * orion-substrate-runtime DUAL-WRITES these five nodes: the old
    ``substrate_node_prediction_error_history`` row is still written, and the same reading also
    lands in the new table (``signal_samples``).
  * ``READ_BOTH_*`` SQL and ``merge_signal_rows`` read the union of both tables. Later patches
    switch readers to them; until the old rows age out of the PE table (7-day prune, no DELETE),
    a reader must union both or it sees a short window.
No reader uses the new table yet, so Orion's behavior does not change in Patch 1.

Patch 2 (alarms off the prediction-error slot) moves the three alarms (``MOVED_SIGNAL_NODE_IDS``):
  * their writer puts the value in node metadata ``signal_value`` / ``signal_kind`` and in this
    table only. No ``prediction_error`` key, no PE-table row (CI: scripts/check_pe_alarm_readers.py).
  * every history reader reads ``HISTORY_WITH_MOVED_SOURCE_SQL``: PE rows for the other nodes, and
    the read-both union for the moved ids, so the week of old PE rows still counts as alarm
    history until the 7-day prune ages it out (no DELETE).
  * a reader of a node's value (``node_signal_value``) reads ``signal_value`` for a moved id and
    never its stale ``prediction_error`` key (kill means kill, materializer resurrection).
Patch 3 (cameras off the prediction-error slot, 2026-11-24) moves the two cameras too:
  * their value is camera scene change (orion/substrate/scene_change.py), in nats, written to
    ``signal_value`` (unclamped) with ``signal_kind`` ``scene_change`` and ``signal_scale``
    (``v2`` / ``v2_partial``; ``v1`` under the legacy_z rollback scorer).
  * their history rows carry ``scale``; every reader reads them signal_only with
    ``scale IN ('v2','v2_partial')`` (orion/prediction/signal_history_sql.py).
  * a consumer that needs a 0..1 size (the dynamics pressure seed, node salience) uses
    ``signal_unit_value``: nats / the 4.67-nat cap for a camera.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from orion.prediction.alarm_nodes import ALARM_NODE_IDS

# Stdlib-only half (light readers import it without orion.substrate's heavy package init).
from orion.prediction.signal_history_sql import (  # noqa: F401  (re-export)
    ALARM_GLOSSARY_CHANNEL,
    CALM_CHECK_SCALES,
    CAMERA_NODE_IDS,
    CURRENT_SCALES,
    SCALE_PREDICATE_SQL,
    SCALE_V1,
    SCALE_V2,
    SCALE_V2_PARTIAL,
    SCALES,
    SCENE_CHANGE_GLOSSARY_CHANNEL,
    is_camera_scene_change_node,
    HISTORY_GLOSSARY_CHANNELS,
    HISTORY_READ_MODE_BY_NODE_ID,
    HISTORY_WITH_MOVED_SOURCE_SQL,
    MOVED_SIGNAL_NODE_IDS,
    READ_BOTH,
    SIGNAL_ONLY,
    PE_HISTORY_TABLE,
    SIGNAL_HISTORY_TABLE,
    glossary_channel_for,
    history_read_mode,
    history_table_for,
    history_with_moved_source_sql,
    is_moved_signal_node,
)


SIGNAL_KIND_ALARM = "alarm"
SIGNAL_KIND_SCENE_CHANGE = "scene_change"
SIGNAL_KINDS: frozenset[str] = frozenset({SIGNAL_KIND_ALARM, SIGNAL_KIND_SCENE_CHANGE})

# Same ids as orion.schemas.attention_candidate.PERCEPTION_NODE_ID / PERCEPTION_WALKWAY_NODE_ID
# (pinned by a test; not imported, to keep this module stdlib-only).
SCENE_CHANGE_NODE_IDS: frozenset[str] = CAMERA_NODE_IDS

# ln(289) - 1: the largest camera scene-change value (orion/substrate/scene_change.py
# SCENE_CHANGE_CAP_NATS, pinned equal by a test; a literal here keeps this module light).
SCENE_CHANGE_CAP_NATS = 4.666426688112432

SIGNAL_KIND_BY_NODE_ID: dict[str, str] = {
    **{node_id: SIGNAL_KIND_ALARM for node_id in ALARM_NODE_IDS},
    **{node_id: SIGNAL_KIND_SCENE_CHANGE for node_id in SCENE_CHANGE_NODE_IDS},
}
SIGNAL_NODE_IDS: tuple[str, ...] = tuple(sorted(SIGNAL_KIND_BY_NODE_ID))


# Node metadata keys a moved node carries instead of ``prediction_error``.
SIGNAL_VALUE_KEY = "signal_value"
SIGNAL_KIND_KEY = "signal_kind"
# Which scale a camera's signal_value is on (v1 legacy z / v2_partial / v2); the history sampler
# copies it into the row's ``scale``. Alarms do not write it (their rows are 'v1').
SIGNAL_SCALE_KEY = "signal_scale"
PREDICTION_ERROR_KEY = "prediction_error"


Row = tuple[str, datetime, float]
SignalRow = tuple[str, datetime, float, str, str]  # node_id, observed_at, value, kind, scale


def signal_kind_for(node_id: object) -> str | None:
    """``alarm`` / ``scene_change`` for the five signal nodes, else None."""
    return SIGNAL_KIND_BY_NODE_ID.get(str(node_id or ""))


def node_value_key(node_id: object) -> str:
    """The metadata key that holds this node's value: ``signal_value`` for a moved node."""
    return SIGNAL_VALUE_KEY if is_moved_signal_node(node_id) else PREDICTION_ERROR_KEY


def node_signal_value(node_id: object, metadata: object) -> object:
    """The raw stored value of a substrate value node, or None.

    A moved node is read from ``signal_value`` only: a stale ``prediction_error`` key left on it
    (an old write, a materializer resurrection) is never read. Raw (unconverted), like
    ``metadata.get("prediction_error")`` was."""
    if not isinstance(metadata, dict):
        return None
    return metadata.get(node_value_key(node_id))


def signal_unit_value(node_id: object, raw: object) -> float:
    """A signal node's value on a 0..1 size scale, for a consumer that needs one (the dynamics
    pressure seed, node salience). A camera's scene change is nats (0 .. 4.67): divided by the cap.
    Every other value is returned as is (an alarm is already 0..1). Non-numeric reads 0.0."""
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    if is_camera_scene_change_node(node_id):
        return max(0.0, min(1.0, value / SCENE_CHANGE_CAP_NATS))
    return value


def split_history_samples(samples: Iterable[Row]) -> tuple[list[Row], list[SignalRow]]:
    """(PE-table rows, signal-table rows) for one batch of history samples.

    A moved node's reading (alarms since Patch 2, cameras since Patch 3) goes to the signal table
    only. Every other node goes to the PE table only."""
    samples = list(samples)
    pe_rows: list[Row] = []
    for node_id, observed_at, value in samples:
        if not is_moved_signal_node(node_id):
            pe_rows.append((str(node_id), observed_at, float(value)))
    return pe_rows, signal_samples(samples)


def signal_samples(
    samples: Iterable[Row], scales: dict[tuple[str, datetime], str] | None = None
) -> list[SignalRow]:
    """The subset of history samples that belong in the signal table, tagged with kind and scale.

    ``scales`` maps (node_id, observed_at) to the reading's ``signal_scale``; a missing or unknown
    scale is 'v1' (every alarm row, and a camera row written by the legacy scorer)."""
    out: list[SignalRow] = []
    for node_id, observed_at, value in samples:
        kind = signal_kind_for(node_id)
        if kind is not None:
            scale = (scales or {}).get((str(node_id), observed_at), SCALE_V1)
            if scale not in SCALES:
                scale = SCALE_V1
            out.append((str(node_id), observed_at, float(value), kind, scale))
    return out


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def merge_signal_rows(pe_rows: Iterable[Row], signal_rows: Iterable[Row]) -> list[Row]:
    """Union of both tables for the signal nodes, one row per (node_id, observed_at).

    The signal table wins a tie (it is the successor; in Patch 1 the values are identical).
    PE rows for any other node are dropped: this is the signal nodes' history, not all history.
    Sorted by node_id, then observed_at, like the store's fetches.
    """
    merged: dict[tuple[str, datetime], float] = {}
    for node_id, observed_at, value in pe_rows:
        if signal_kind_for(node_id) is None:
            continue
        merged[(str(node_id), _aware(observed_at))] = float(value)
    for node_id, observed_at, value in signal_rows:
        merged[(str(node_id), _aware(observed_at))] = float(value)
    return [(n, t, v) for (n, t), v in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1]))]


# --- SQL (sqlalchemy text(); psycopg2 binds a Python list to ANY(:node_ids)) -------------------

INSERT_SIGNAL_HISTORY_SQL = f"""
INSERT INTO {SIGNAL_HISTORY_TABLE} (node_id, observed_at, value, kind, scale)
VALUES (:node_id, :observed_at, :value, :kind, :scale)
ON CONFLICT (node_id, observed_at) DO NOTHING
"""

PRUNE_SIGNAL_HISTORY_SQL = f"""
DELETE FROM {SIGNAL_HISTORY_TABLE}
WHERE observed_at < :older_than
"""

# One row per (node_id, observed_at) across both tables; the signal table wins (pref 0).
_BOTH_SINCE = f"""
SELECT DISTINCT ON (node_id, observed_at) node_id, observed_at, value
FROM (
    SELECT node_id, observed_at, value, 0 AS pref
    FROM {SIGNAL_HISTORY_TABLE}
    WHERE observed_at >= :since AND node_id = ANY(:node_ids)
    UNION ALL
    SELECT node_id, observed_at, value, 1 AS pref
    FROM {PE_HISTORY_TABLE}
    WHERE observed_at >= :since AND node_id = ANY(:node_ids)
) both_tables
ORDER BY node_id, observed_at, pref
"""

READ_BOTH_SINCE_SQL = _BOTH_SINCE

READ_BOTH_COUNT_SQL = f"SELECT count(*) FROM ({_BOTH_SINCE}) rows_both"

READ_BOTH_EVENT_TIMES_SQL = (
    f"SELECT observed_at FROM ({_BOTH_SINCE}) rows_both WHERE value > 0 ORDER BY observed_at"
)
