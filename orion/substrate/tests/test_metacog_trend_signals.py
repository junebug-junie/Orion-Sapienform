from __future__ import annotations

import os
import pytest
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from orion.substrate.metacog_trend_signals import (
    build_recent_trend_signals_cue,
    latest_biometrics_induction_by_node,
    latest_node_prediction_errors,
    most_notable_trend_channel,
)

NODE_IDS = ("node:substrate.execution", "node:substrate.biometrics")


def _engine_with_first_row(row: dict | None) -> MagicMock:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = row
    return engine


def _engine_with_all_rows(rows: list[dict]) -> MagicMock:
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.all.return_value = rows
    return engine


# --- latest_node_prediction_errors ---


def test_prediction_errors_returns_real_values() -> None:
    engine = _engine_with_first_row(
        {
            "generated_at": datetime.now(timezone.utc),
            "node_0": "0.297",
            "node_1": "0.155",
        }
    )
    result = latest_node_prediction_errors(engine, NODE_IDS)
    assert result == {
        "node:substrate.execution": 0.297,
        "node:substrate.biometrics": 0.155,
    }


def test_prediction_errors_none_when_no_row() -> None:
    engine = _engine_with_first_row(None)
    result = latest_node_prediction_errors(engine, NODE_IDS)
    assert result == {nid: None for nid in NODE_IDS}


def test_prediction_errors_none_per_node_when_node_absent() -> None:
    engine = _engine_with_first_row(
        {"generated_at": datetime.now(timezone.utc), "node_0": None, "node_1": "0.4"}
    )
    result = latest_node_prediction_errors(engine, NODE_IDS)
    assert result["node:substrate.execution"] is None
    assert result["node:substrate.biometrics"] == 0.4


def test_prediction_errors_all_none_when_stale() -> None:
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    engine = _engine_with_first_row(
        {"generated_at": stale, "node_0": "0.297", "node_1": "0.155"}
    )
    result = latest_node_prediction_errors(engine, NODE_IDS)
    assert result == {nid: None for nid in NODE_IDS}


def test_prediction_errors_empty_node_ids_returns_empty_dict() -> None:
    engine = MagicMock()
    assert latest_node_prediction_errors(engine, ()) == {}
    engine.connect.assert_not_called()


# --- latest_biometrics_induction_by_node ---


def test_induction_by_node_returns_metrics_within_max_age() -> None:
    now = datetime.now(timezone.utc)
    engine = _engine_with_all_rows(
        [
            {"node": "athena", "metrics": {"cpu": {"trend": 0.55}}, "ts": now},
            {
                "node": "atlas",
                "metrics": {"gpu_util": {"trend": 0.51}},
                "ts": now - timedelta(seconds=30),
            },
        ]
    )
    result = latest_biometrics_induction_by_node(engine, ("athena", "atlas", "circe"))
    assert result["athena"] == {"cpu": {"trend": 0.55}}
    assert result["atlas"] == {"gpu_util": {"trend": 0.51}}
    assert "circe" not in result


def test_induction_by_node_drops_stale_rows() -> None:
    now = datetime.now(timezone.utc)
    engine = _engine_with_all_rows(
        [
            {
                "node": "circe",
                "metrics": {"cpu": {"trend": 0.5}},
                "ts": now - timedelta(seconds=600),
            }
        ]
    )
    result = latest_biometrics_induction_by_node(engine, ("circe",), max_age_sec=180.0)
    assert result == {}


def test_induction_by_node_empty_nodes_returns_empty_dict() -> None:
    engine = MagicMock()
    assert latest_biometrics_induction_by_node(engine, ()) == {}
    engine.connect.assert_not_called()


# --- most_notable_trend_channel ---


def test_most_notable_trend_channel_picks_max_deviation() -> None:
    metrics = {
        "cpu": {"trend": 0.505, "level": 0.3, "spike_rate": 0.0, "volatility": 0.06},
        "gpu_util": {"trend": 0.7, "level": 0.1, "spike_rate": 0.1, "volatility": 0.2},
        "mem": {"trend": 0.5, "level": 0.05},
    }
    channel, values = most_notable_trend_channel(metrics)
    assert channel == "gpu_util"
    assert values["trend"] == 0.7


def test_most_notable_trend_channel_none_when_no_valid_trend() -> None:
    assert most_notable_trend_channel({"cpu": {"level": 0.1}}) is None
    assert most_notable_trend_channel({}) is None


# --- build_recent_trend_signals_cue ---


def test_build_recent_trend_signals_cue_shape() -> None:
    cue = build_recent_trend_signals_cue(
        prediction_errors={
            "node:substrate.execution": 0.297,
            "node:substrate.biometrics": 0.155,
        },
        induction_by_node={
            "athena": {"cpu": {"trend": 0.505, "level": 0.31, "spike_rate": 0.0, "volatility": 0.06}},
        },
    )
    assert cue["prediction_error"]["node:substrate.execution"] == 0.297
    assert cue["biometrics_induction"]["athena"]["channel"] == "cpu"
    assert cue["biometrics_induction"]["athena"]["trend"] == 0.505


def test_build_recent_trend_signals_cue_skips_nodes_with_no_notable_channel() -> None:
    cue = build_recent_trend_signals_cue(
        prediction_errors={},
        induction_by_node={"athena": {"cpu": {"level": 0.1}}},
    )
    assert cue["biometrics_induction"] == {}


# --- query shape must stay indexable ---------------------------------------


def test_induction_query_is_a_lateral_top_1_per_node_not_distinct_on() -> None:
    """Two independent traps, both measured live 2026-09-03, both pinned here.

    1. `varchar::timestamptz` is not IMMUTABLE, so Postgres refuses to index
       that expression -- a cast in the ORDER BY makes the ordering
       unindexable by construction.
    2. `DISTINCT ON (node)` must consume its entire sorted input and Postgres
       has no loose index scan, so it kept choosing a parallel seq scan +
       external merge sort over all 138,927 matching rows *even with the
       index present*: 911ms and ~150MB of temp spill, versus 0.47ms for the
       LATERAL form.

    Trap 2 is why this asserts on the query SHAPE and not merely on the
    absence of a cast: removing the cast alone left the query just as slow,
    which is precisely the "fix" that would otherwise look correct.
    """
    engine = _engine_with_all_rows([])
    latest_biometrics_induction_by_node(engine, ["athena"])

    sql = " ".join(str(engine.connect().__enter__().execute.call_args[0][0]).split()).upper()

    assert "DISTINCT ON" not in sql, f"DISTINCT ON cannot use the index: {sql}"
    assert "CROSS JOIN LATERAL" in sql, sql
    # Token boundary, not substring: `"LIMIT 1" in "LIMIT 10"` is True, and
    # LIMIT 10 is WORSE than the original bug -- the reader below assigns
    # `out[node] = metrics` per row, so the LAST row wins and a silent
    # LIMIT 10 hands back the OLDEST of ten instead of the newest.
    assert re.search(r"\bLIMIT 1\b", sql), (
        f"lateral must take exactly one row per node: {sql}"
    )

    # The ordering the index serves. DESC is load-bearing: ASC would return
    # the OLDEST row per node -- a wrong answer that still looks like a
    # working query.
    order_by = sql.split("ORDER BY", 1)
    assert len(order_by) == 2, f"no ORDER BY in statement: {sql}"
    clause = order_by[1]
    assert "::TIMESTAMPTZ" not in clause, (
        f"ORDER BY casts timestamp, which no index can serve: {clause!r}"
    )
    assert "B.TIMESTAMP DESC" in clause, clause


def test_induction_query_still_casts_the_projected_timestamp() -> None:
    """The max-age filter compares `ts` as a datetime, so the projection casts.

    Pinned separately from the indexability assertions above so that "drop the
    cast" cannot be applied wholesale to make those pass.
    """
    engine = _engine_with_all_rows([])
    latest_biometrics_induction_by_node(engine, ["athena"])

    sql = " ".join(str(engine.connect().__enter__().execute.call_args[0][0]).split()).upper()
    select_list = sql.split("FROM", 1)[0]
    assert "TIMESTAMP::TIMESTAMPTZ AS TS" in select_list, select_list


# --- real-Postgres lane -----------------------------------------------------
#
# The two tests above drive a MagicMock and assert on SQL *text*. The two
# things that can actually break at runtime are invisible to them: whether
# `unnest(CAST(:nodes AS text[]))` binds at all, and whether it binds for both
# callers' parameter shapes. The Hub passes a list; cortex-exec
# (services/orion-cortex-exec/app/metacog_trend_reader.py) passes a tuple, and
# psycopg2 adapts a tuple to a composite record, not an array:
#
#   ProgrammingError: cannot cast type record to text[]
#
# The single `list(nodes)` call in latest_biometrics_induction_by_node is what
# normalises that. Delete it and every mocked test in this repo stays green
# while cortex-exec's metacog trend cue goes permanently dark -- and because
# that reader fails open, nothing reports it.

_LOCAL_DATABASE_URL = os.getenv(
    "ORION_TEST_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"
)


def _local_postgres_engine():
    try:
        from sqlalchemy import create_engine
    except ImportError:
        return None
    try:
        engine = create_engine(_LOCAL_DATABASE_URL, connect_args={"connect_timeout": 2})
        with engine.connect():
            pass
        return engine
    except Exception:
        return None


_ENGINE = _local_postgres_engine()


@pytest.mark.skipif(_ENGINE is None, reason="local Postgres not reachable")
@pytest.mark.parametrize(
    "nodes",
    [
        pytest.param(["athena"], id="hub-single-element-list"),
        pytest.param(("athena", "atlas", "circe"), id="cortex-exec-tuple"),
        pytest.param(["athena", "athena"], id="duplicate-node-names"),
        pytest.param(["definitely-not-a-node"], id="absent-node"),
    ],
)
def test_induction_query_binds_against_real_postgres(nodes) -> None:
    """Real psycopg2 binding, real table. Rows may be empty; binding may not fail."""
    out = latest_biometrics_induction_by_node(_ENGINE, nodes, max_age_sec=10**9)
    assert isinstance(out, dict)
    assert set(out).issubset({str(n) for n in nodes})


@pytest.mark.skipif(_ENGINE is None, reason="local Postgres not reachable")
def test_induction_query_returns_the_newest_row_per_node() -> None:
    """The LATERAL must pick the newest row, not merely *a* row.

    Cross-checked against an independent MAX() over the same table rather than
    against the query's own ordering -- a LIMIT/ordering mistake that made the
    query return the oldest row would otherwise agree with itself.
    """
    from sqlalchemy import text as _text

    with _ENGINE.connect() as conn:
        nodes = [
            r[0]
            for r in conn.execute(
                _text("SELECT DISTINCT node FROM orion_biometrics_induction WHERE node IS NOT NULL LIMIT 3")
            ).all()
        ]
        if not nodes:
            pytest.skip("no biometrics induction rows to compare against")
        expected = {
            r[0]: r[1]
            for r in conn.execute(
                _text(
                    "SELECT node, MAX(timestamp) FROM orion_biometrics_induction "
                    "WHERE node = ANY(:nodes) GROUP BY node"
                ),
                {"nodes": list(nodes)},
            ).all()
        }
        actual = {
            r[0]: r[1]
            for r in conn.execute(
                _text(
                    """
                    SELECT n.node, latest.timestamp
                    FROM unnest(CAST(:nodes AS text[])) AS n(node)
                    CROSS JOIN LATERAL (
                        SELECT b.timestamp FROM orion_biometrics_induction b
                        WHERE b.node = n.node ORDER BY b.timestamp DESC LIMIT 1
                    ) AS latest
                    """
                ),
                {"nodes": list(nodes)},
            ).all()
        }
    assert actual == expected, "LATERAL did not return the newest row per node"
