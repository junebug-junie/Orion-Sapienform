from __future__ import annotations

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
    assert "LIMIT 1" in sql, "without LIMIT 1 the lateral still reads every row per node"

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
