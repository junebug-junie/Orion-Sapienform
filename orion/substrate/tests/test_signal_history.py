"""orion/substrate/signal_history.py: the five non-PE nodes, their kinds, and the read-both merge
(PR #3781 Patch 1)."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.prediction.alarm_nodes import ALARM_NODE_IDS
from orion.schemas.attention_candidate import PERCEPTION_NODE_ID, PERCEPTION_WALKWAY_NODE_ID
from orion.substrate.signal_history import (
    INSERT_SIGNAL_HISTORY_SQL,
    READ_BOTH_COUNT_SQL,
    READ_BOTH_EVENT_TIMES_SQL,
    READ_BOTH_SINCE_SQL,
    SIGNAL_KIND_ALARM,
    SIGNAL_KIND_SCENE_CHANGE,
    SIGNAL_KINDS,
    SIGNAL_NODE_IDS,
    merge_signal_rows,
    signal_kind_for,
    signal_samples,
)

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "services/orion-sql-db/manual_migration_node_signal_history_v1.sql"
T0 = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


def test_exactly_the_five_nodes_with_honest_kinds():
    assert set(SIGNAL_NODE_IDS) == set(ALARM_NODE_IDS) | {PERCEPTION_NODE_ID, PERCEPTION_WALKWAY_NODE_ID}
    assert len(SIGNAL_NODE_IDS) == 5
    for node_id in ALARM_NODE_IDS:
        assert signal_kind_for(node_id) == SIGNAL_KIND_ALARM
    assert signal_kind_for(PERCEPTION_NODE_ID) == SIGNAL_KIND_SCENE_CHANGE
    assert signal_kind_for(PERCEPTION_WALKWAY_NODE_ID) == SIGNAL_KIND_SCENE_CHANGE
    # Real forecast-surprise and other substrate nodes are not signal nodes.
    for other in ("node:prediction.body", "node:substrate.execution", "node:substrate.perception.x", None, ""):
        assert signal_kind_for(other) is None


def test_signal_samples_keeps_only_the_five_and_tags_kind():
    rows = signal_samples([
        ("node:substrate.cabinet", T0, 0.0),
        ("node:substrate.execution", T0, 0.4),
        (PERCEPTION_WALKWAY_NODE_ID, T0, 0.25),
        ("node:prediction.body", T0, 1.2),
    ])
    assert rows == [
        ("node:substrate.cabinet", T0, 0.0, "alarm"),
        (PERCEPTION_WALKWAY_NODE_ID, T0, 0.25, "scene_change"),
    ]


def test_merge_dedupes_on_key_signal_wins_other_nodes_dropped_naive_is_utc():
    naive = T0.replace(tzinfo=None)
    pe = [
        ("node:substrate.strain", naive, 1.0),
        ("node:substrate.strain", T0 + timedelta(minutes=1), 0.0),
        ("node:substrate.execution", T0, 0.9),
    ]
    sig = [
        ("node:substrate.strain", T0, 0.5),  # same key as the naive PE row: signal wins
        ("node:substrate.cabinet", T0, 0.0),
    ]
    assert merge_signal_rows(pe, sig) == [
        ("node:substrate.cabinet", T0, 0.0),
        ("node:substrate.strain", T0, 0.5),
        ("node:substrate.strain", T0 + timedelta(minutes=1), 0.0),
    ]


def test_read_both_sql_unions_both_tables_and_prefers_signal():
    for sql in (READ_BOTH_SINCE_SQL, READ_BOTH_COUNT_SQL, READ_BOTH_EVENT_TIMES_SQL):
        assert "substrate_node_signal_history" in sql
        assert "substrate_node_prediction_error_history" in sql
        assert "UNION ALL" in sql and "DISTINCT ON (node_id, observed_at)" in sql
        assert sql.count("node_id = ANY(:node_ids)") == 2
    assert "ORDER BY node_id, observed_at, pref" in READ_BOTH_SINCE_SQL
    assert "value > 0" in READ_BOTH_EVENT_TIMES_SQL
    assert "ON CONFLICT (node_id, observed_at) DO NOTHING" in INSERT_SIGNAL_HISTORY_SQL


def test_migration_matches_module_contract():
    sql = MIGRATION.read_text()
    assert sql.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-substrate-runtime"
    low = sql.lower()
    assert "create table if not exists substrate_node_signal_history" in low
    for col in ("node_id      text", "observed_at  timestamptz", "value        double precision",
                "kind         text", "recorded_at  timestamptz"):
        assert col in low
    assert "primary key (node_id, observed_at)" in low
    kinds = set(re.findall(r"'([a-z_]+)'", low.split("check (kind in", 1)[1].split(")", 1)[0]))
    assert kinds == set(SIGNAL_KINDS)
    # Additive only (post-pull auto-apply): no DO block, no GRANT, no DELETE/DROP.
    for banned in ("do $$", "grant ", "delete ", "drop "):
        assert banned not in "\n".join(l for l in low.splitlines() if not l.startswith("--"))
