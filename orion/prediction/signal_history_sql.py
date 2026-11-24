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
#   signal_only  the signal table only, never the PE table. Patch 3 moves the two cameras here
#                (new-scale rows only), because old- and new-scale camera rows must never share a
#                ranking window -- so the cameras are deliberately NOT in a read-both union.
# A node absent from the map is read from the PE table only (every real prediction-error node,
# and the cameras until Patch 3).
READ_BOTH = "read_both"
SIGNAL_ONLY = "signal_only"
READ_MODES: frozenset[str] = frozenset({READ_BOTH, SIGNAL_ONLY})
HISTORY_READ_MODE_BY_NODE_ID: dict[str, str] = {node_id: READ_BOTH for node_id in sorted(ALARM_NODE_IDS)}

# The signal nodes whose writer has left the prediction-error slot. Every generic prediction-error
# reader skips these ids' PE side (their stale ``prediction_error`` key, their week of old PE rows)
# the way it skips RETIRED_PREDICTION_ERROR_NODE_IDS; alarm-aware readers get them back through the
# history source below and ``orion.substrate.signal_history.node_signal_value``.
MOVED_SIGNAL_NODE_IDS: frozenset[str] = frozenset(HISTORY_READ_MODE_BY_NODE_ID)


def history_read_mode(node_id: object) -> str | None:
    """``read_both`` / ``signal_only`` for a moved node, None for a PE-table node."""
    return HISTORY_READ_MODE_BY_NODE_ID.get(str(node_id or ""))


def is_moved_signal_node(node_id: object) -> bool:
    """True for a signal node that no longer writes the prediction-error slot (Patch 2: alarms)."""
    return str(node_id or "") in MOVED_SIGNAL_NODE_IDS


# The glossary channel of a moved node's entry (config/field/field_channel_glossary.v1.yaml,
# ``channel: alarm``). Every other node's value is still described under ``prediction_error``.
ALARM_GLOSSARY_CHANNEL = "alarm"
HISTORY_GLOSSARY_CHANNELS: frozenset[str] = frozenset({"prediction_error", ALARM_GLOSSARY_CHANNEL})


def glossary_channel_for(node_id: object) -> str:
    """The glossary channel that describes this node's value: ``alarm`` for a moved alarm, else
    ``prediction_error`` (the cameras until Patch 3, and every real prediction-error node)."""
    return ALARM_GLOSSARY_CHANNEL if is_moved_signal_node(node_id) else "prediction_error"


def history_table_for(node_id: object) -> str:
    """The table a reading of this node lives in, for a provenance ref (``<table>:<node>@<ts>``)."""
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
        branches.append(f"""    SELECT node_id, observed_at, value
    FROM {SIGNAL_HISTORY_TABLE}
    WHERE node_id = ANY({_id_array(signal_only)})""")
    return "(\n" + "\n    UNION ALL\n".join(branches) + "\n) node_history"


HISTORY_WITH_MOVED_SOURCE_SQL = history_with_moved_source_sql()
