"""Stdlib-only half of orion/substrate/signal_history.py: the table names, which signal nodes have
moved off the prediction-error slot, and the one history source every reader uses.

Light on purpose: orion/introspect/node_history.py and orion-thought's self-readings cue read history
without importing orion.substrate (its package init is heavy). orion/substrate/signal_history.py
re-exports all of this.

Design: docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md (PR #3808).
"""
from __future__ import annotations

import re
from typing import Iterable

from orion.prediction.alarm_nodes import ALARM_NODE_IDS

SIGNAL_HISTORY_TABLE = "substrate_node_signal_history"
PE_HISTORY_TABLE = "substrate_node_prediction_error_history"

# How a history reader reads each node that has left the prediction-error slot: a per-node rule,
# so a later patch flips one node without touching the readers (design rev 3, PR #3827).
#   read_both    the signal table plus the node's old PE rows, deduped on (node_id, observed_at),
#                signal table preferred. Patch 2: the three alarms, until their PE rows age out.
#   signal_only  the signal table only, never the PE table, and only rows of the current scale
#                (``scale IN ('v2','v2_partial')``). Patch 3 (2026-11-24): the two cameras, because
#                old- and new-scale camera rows must never share a ranking window -- so the cameras
#                are deliberately NOT in a read-both union.
# A node absent from the map is read from the PE table only (every real prediction-error node).
READ_BOTH = "read_both"
SIGNAL_ONLY = "signal_only"
READ_MODES: frozenset[str] = frozenset({READ_BOTH, SIGNAL_ONLY})

# The two camera scene-change nodes (PE retire Patch 3). Same ids as
# orion.schemas.attention_candidate.PERCEPTION_NODE_ID / PERCEPTION_WALKWAY_NODE_ID (pinned by a test).
PERCEPTION_NODE_ID = "node:substrate.perception"
PERCEPTION_WALKWAY_NODE_ID = "node:substrate.perception_walkway"
CAMERA_NODE_IDS: frozenset[str] = frozenset({PERCEPTION_NODE_ID, PERCEPTION_WALKWAY_NODE_ID})

# substrate_node_signal_history.scale (manual_migration_node_signal_history_scale_v1.sql).
#   v1          every row from before Patch 3 and every alarm row; for a camera, the old clipped z.
#   v2_partial  camera scene change while its 24 h stage-2 window is still filling: ranked by the
#               contests, EXCLUDED from calm checks and win share (eval: scale = 'v2' only).
#   v2          camera scene change with a full stage-2 window.
SCALE_V1 = "v1"
SCALE_V2 = "v2"
SCALE_V2_PARTIAL = "v2_partial"
SCALES: frozenset[str] = frozenset({SCALE_V1, SCALE_V2, SCALE_V2_PARTIAL})
# What a history reader of a signal_only node reads (the contests rank against both).
CURRENT_SCALES: tuple[str, ...] = (SCALE_V2, SCALE_V2_PARTIAL)
# What a calm check or a win-share count reads: full-window rows only.
CALM_CHECK_SCALES: tuple[str, ...] = (SCALE_V2,)
SCALE_PREDICATE_SQL = "scale IN (" + ", ".join(f"'{s}'" for s in CURRENT_SCALES) + ")"

HISTORY_READ_MODE_BY_NODE_ID: dict[str, str] = {
    **{node_id: READ_BOTH for node_id in sorted(ALARM_NODE_IDS)},
    **{node_id: SIGNAL_ONLY for node_id in sorted(CAMERA_NODE_IDS)},
}

# The signal nodes whose writer has left the prediction-error slot. Every generic prediction-error
# reader skips these ids' PE side (their stale ``prediction_error`` key, their week of old PE rows)
# the way it skips RETIRED_PREDICTION_ERROR_NODE_IDS; alarm-aware readers get them back through the
# history source below and ``orion.substrate.signal_history.node_signal_value``.
MOVED_SIGNAL_NODE_IDS: frozenset[str] = frozenset(HISTORY_READ_MODE_BY_NODE_ID)


def history_read_mode(node_id: object) -> str | None:
    """``read_both`` / ``signal_only`` for a moved node, None for a PE-table node."""
    return HISTORY_READ_MODE_BY_NODE_ID.get(str(node_id or ""))


def is_moved_signal_node(node_id: object) -> bool:
    """True for a signal node that no longer writes the prediction-error slot (Patch 2: the three
    alarms; Patch 3: the two cameras)."""
    return str(node_id or "") in MOVED_SIGNAL_NODE_IDS


def is_camera_scene_change_node(node_id: object) -> bool:
    """True for the two camera scene-change nodes (room, walkway)."""
    return str(node_id or "") in CAMERA_NODE_IDS


# The glossary channel of a moved node's entry (config/field/field_channel_glossary.v1.yaml):
# ``channel: alarm`` for the three alarms, ``channel: scene_change`` for the two cameras. Every
# other node's value is described under ``prediction_error``.
ALARM_GLOSSARY_CHANNEL = "alarm"
SCENE_CHANGE_GLOSSARY_CHANNEL = "scene_change"
HISTORY_GLOSSARY_CHANNELS: frozenset[str] = frozenset(
    {"prediction_error", ALARM_GLOSSARY_CHANNEL, SCENE_CHANGE_GLOSSARY_CHANNEL}
)


def glossary_channel_for(node_id: object) -> str:
    """The glossary channel that describes this node's value: ``scene_change`` for a camera,
    ``alarm`` for a moved alarm, else ``prediction_error`` (every real prediction-error node)."""
    if is_camera_scene_change_node(node_id):
        return SCENE_CHANGE_GLOSSARY_CHANNEL
    return ALARM_GLOSSARY_CHANNEL if is_moved_signal_node(node_id) else "prediction_error"


def history_table_for(node_id: object) -> str:
    """The table a reading of this node is read from, for a provenance ref (``<table>:<node>@<ts>``).

    For a read_both node this names the successor table even for a reading that, until its old
    PE rows age out (~2026-12-01), may come from the PE table: the ref names the source the
    reader reads (both tables, deduped on the same primary key), not the physical row."""
    return SIGNAL_HISTORY_TABLE if is_moved_signal_node(node_id) else PE_HISTORY_TABLE


# --- One history source for every reader (Patch 2) ----------------------------------------------
# A derived table with no bind parameters (the moved ids are code constants, inlined as literals),
# so the same text works under sqlalchemy ``:name``, psycopg2 ``%s`` and asyncpg ``$1``. Columns:
# node_id, observed_at, value. PE rows for every node except the moved ids, plus the read-both
# union for the moved ids (signal table preferred; old PE rows keep the 7-day window full until
# they age out). Postgres pushes an outer ``WHERE node_id = ... AND observed_at >= ...`` into both
# branches (both are DISTINCT ON / UNION ALL columns), so a reader keeps its indexes.

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _id_array(ids: Iterable[str]) -> str:
    ids = sorted(ids)
    for node_id in ids:
        if not _SAFE_ID.match(node_id):
            raise ValueError(f"unsafe node id for an SQL literal: {node_id!r}")
    return "ARRAY[" + ", ".join(f"'{node_id}'" for node_id in ids) + "]::text[]"


def history_with_moved_source_sql(read_modes: dict[str, str] | None = None) -> str:
    """The derived table for a per-node read rule (default ``HISTORY_READ_MODE_BY_NODE_ID``)."""
    modes = HISTORY_READ_MODE_BY_NODE_ID if read_modes is None else read_modes
    unknown = set(modes.values()) - READ_MODES
    if unknown:
        raise ValueError(f"unknown history read mode(s): {sorted(unknown)}")
    moved = _id_array(modes) if modes else None
    both = [n for n, m in modes.items() if m == READ_BOTH]
    signal_only = [n for n, m in modes.items() if m == SIGNAL_ONLY]
    branches = [
        f"""    SELECT node_id, observed_at, value
    FROM {PE_HISTORY_TABLE}"""
        + (f"\n    WHERE node_id <> ALL({moved})" if moved else "")
    ]
    if both:
        ids = _id_array(both)
        branches.append(f"""    SELECT node_id, observed_at, value FROM (
        SELECT DISTINCT ON (node_id, observed_at) node_id, observed_at, value
        FROM (
            SELECT node_id, observed_at, value, 0 AS pref
            FROM {SIGNAL_HISTORY_TABLE}
            WHERE node_id = ANY({ids})
            UNION ALL
            SELECT node_id, observed_at, value, 1 AS pref
            FROM {PE_HISTORY_TABLE}
            WHERE node_id = ANY({ids})
        ) both_tables
        ORDER BY node_id, observed_at, pref
    ) read_both_rows""")
    if signal_only:
        # Current-scale rows only (Patch 3): the cameras' old clipped-z rows (scale 'v1') never
        # share a ranking window with their scene-change rows. Pinned by a unit test and by
        # scripts/check_pe_alarm_readers.py (any other SQL naming the signal table with a camera
        # id must carry a scale predicate too).
        branches.append(f"""    SELECT node_id, observed_at, value
    FROM {SIGNAL_HISTORY_TABLE}
    WHERE node_id = ANY({_id_array(signal_only)}) AND {SCALE_PREDICATE_SQL}""")
    return "(\n" + "\n    UNION ALL\n".join(branches) + "\n) node_history"


HISTORY_WITH_MOVED_SOURCE_SQL = history_with_moved_source_sql()
