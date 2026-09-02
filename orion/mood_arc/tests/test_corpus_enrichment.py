"""Tests for orion/mood_arc/corpus_enrichment.py's asof/forward-fill join logic.

Pure logic, synthetic timestamps -- no Postgres connection anywhere in this file. Most
fetch_*() functions (real SQL against real tables) are exercised by hand against live
Postgres as part of the v4 retrain, not here (see orion/mood_arc/README.md's v4 section) --
they are thin, directly-inspectable query wrappers, and a mocked-cursor test of them would
mostly just be re-asserting the SQL string back at itself. fetch_attention_self_model() is
the one exception (a fake cursor/connection, not live Postgres): it has real per-row
branching logic (splitting a combined row into independent per-field series) worth locking
in directly, found by code review to matter for correct LOCF behavior.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.mood_arc.corpus_enrichment import (
    asof_forward_fill,
    enrich_corpus,
    fetch_all_series,
    fetch_attention_self_model,
)
from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1


def _row(t: datetime, channels: dict[str, float] | None = None, idx: int = 0) -> FieldChannelCorpusRowV1:
    return FieldChannelCorpusRowV1(generated_at=t, tick_id=f"tick_{idx}", channels=channels or {})


def _rows(n: int, *, start: datetime, step_sec: float = 2.0) -> list[FieldChannelCorpusRowV1]:
    return [_row(start + timedelta(seconds=step_sec * i), idx=i) for i in range(n)]


START = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_asof_forward_fill_carries_last_known_value_forward() -> None:
    rows = _rows(6, start=START, step_sec=10.0)  # ticks at :00 :10 :20 :30 :40 :50
    series = [
        (START + timedelta(seconds=5), {"action_warrant": 0.1}),
        (START + timedelta(seconds=25), {"action_warrant": 0.9}),
    ]
    missing = asof_forward_fill(rows, series)
    # tick :00 is strictly before the series' first entry (:05) -- left absent.
    assert "action_warrant" not in rows[0].channels
    assert missing == 1
    # ticks :10, :20 fall after :05's reading, before :25's -- carry 0.1 forward.
    assert rows[1].channels["action_warrant"] == 0.1
    assert rows[2].channels["action_warrant"] == 0.1
    # ticks :30, :40, :50 fall after :25's reading -- carry 0.9 forward.
    assert rows[3].channels["action_warrant"] == 0.9
    assert rows[4].channels["action_warrant"] == 0.9
    assert rows[5].channels["action_warrant"] == 0.9


def test_asof_forward_fill_is_exact_boundary_inclusive() -> None:
    """A corpus tick exactly AT a series timestamp must take that reading, not the prior one --
    matches filter_rows_by_min_generated_at's own >=, boundary-inclusive convention elsewhere
    in this module's sibling fit_encoder.py."""
    rows = [_row(START, idx=0)]
    series = [(START, {"swear_frequency": 0.02}), (START + timedelta(seconds=1), {"swear_frequency": 0.5})]
    missing = asof_forward_fill(rows, series)
    assert missing == 0
    assert rows[0].channels["swear_frequency"] == 0.02


def test_asof_forward_fill_empty_series_leaves_every_row_missing() -> None:
    rows = _rows(4, start=START)
    missing = asof_forward_fill(rows, [])
    assert missing == 4
    assert all("anything" not in r.channels for r in rows)


def test_asof_forward_fill_all_rows_before_first_reading_reports_full_miss() -> None:
    """This is the case the ALL ROWS MISSING flag in fit_encoder.py's enrich-corpus output
    exists to catch -- a real query result that never actually lines up with the corpus
    window (wrong table, wrong column, or a lookback window that didn't reach far enough
    back), not a genuinely calm signal."""
    rows = _rows(5, start=START)
    series = [(START + timedelta(days=1), {"doc_semantic_drift": 0.3})]
    missing = asof_forward_fill(rows, series)
    assert missing == len(rows)
    assert all("doc_semantic_drift" not in r.channels for r in rows)


def test_asof_forward_fill_merges_multiple_channels_per_series_entry() -> None:
    """attention_self_model's heartbeat + per-domain prediction-error can arrive in the same
    reading -- both keys must land in the same tick's channels dict."""
    rows = _rows(2, start=START, step_sec=60.0)
    series = [(START, {"heartbeat_mean_ratio": 0.97, "prediction_error_execution": 0.05})]
    asof_forward_fill(rows, series)
    assert rows[0].channels["heartbeat_mean_ratio"] == 0.97
    assert rows[0].channels["prediction_error_execution"] == 0.05
    assert rows[1].channels["heartbeat_mean_ratio"] == 0.97


def test_asof_forward_fill_does_not_clobber_existing_channels() -> None:
    """The join must ADD keys to a row already carrying field-digester channels (cpu_pressure
    etc.) -- not replace the dict."""
    rows = [_row(START, channels={"cpu_pressure": 0.4}, idx=0)]
    asof_forward_fill(rows, [(START, {"action_warrant": 0.7})])
    assert rows[0].channels == {"cpu_pressure": 0.4, "action_warrant": 0.7}


def test_asof_forward_fill_series_need_not_be_pre_sorted() -> None:
    rows = _rows(4, start=START, step_sec=10.0)  # ticks :00 :10 :20 :30
    unsorted_series = [
        (START + timedelta(seconds=25), {"x": 2.0}),
        (START + timedelta(seconds=5), {"x": 1.0}),
    ]
    asof_forward_fill(rows, unsorted_series)
    assert rows[0].channels.get("x") is None  # tick :00, before :05
    assert rows[1].channels["x"] == 1.0  # tick :10, after :05, before :25
    assert rows[2].channels["x"] == 1.0  # tick :20, still before :25
    assert rows[3].channels["x"] == 2.0  # tick :30, after :25


class _FakeCursor:
    """Shared cursor test double for both this file's mocked-connection tests: records every
    executed query's SQL text (for assertions on which table/columns a fetch_*() call
    touches) and returns a pre-set `rows` list from fetchall() (for asserting how a fetch_*()
    call parses real-shaped row tuples). One class covers both needs -- consolidated from two
    near-identical inline fakes (found by code review, 2026-09-02)."""

    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self._conn.queries.append(query)

    def fetchall(self):
        return self._conn.rows


class _FakeConn:
    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.queries: list[str] = []

    def cursor(self):
        return _FakeCursor(self)


def test_fetch_attention_self_model_splits_into_independent_per_field_series() -> None:
    """Regression test for the finding this fixed: a DB row carrying only ONE of
    heartbeat_mean_ratio/prediction_error_by_domain must not blank out the OTHER field for
    that timestamp -- each field needs its own independent series so asof_forward_fill()'s
    single-nearest-entry lookup gives each field correct last-observation-carried-forward
    regardless of what else the nearest row happened to contain."""
    t0 = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 0, 0, 30, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 1, 0, 1, 0, tzinfo=timezone.utc)
    rows = [
        # t0: both groups present.
        (t0, "0.9", {"execution": 0.1, "chat": 0.2, "biometrics": 0.3, "bus_synaptic": 0.4}),
        # t1: heartbeat ONLY -- prediction_error_by_domain absent this row.
        (t1, "0.95", None),
        # t2: prediction_error ONLY -- heartbeat_mean_ratio absent this row.
        (t2, None, {"execution": 0.5, "chat": 0.6, "biometrics": 0.7, "bus_synaptic": 0.8}),
    ]
    conn = _FakeConn(rows)

    series = fetch_attention_self_model(conn, t0, t2)

    assert [v for _, v in series["heartbeat_mean_ratio"]] == [
        {"heartbeat_mean_ratio": 0.9},
        {"heartbeat_mean_ratio": 0.95},
    ]
    assert [v for _, v in series["prediction_error_execution"]] == [
        {"prediction_error_execution": 0.1},
        {"prediction_error_execution": 0.5},
    ]

    # The real regression: asof-joining onto a tick at t2 must still carry heartbeat's t1
    # value forward (LOCF), not lose it because t2's own DB row had no heartbeat reading.
    corpus_rows = [_row(t2, idx=0)]
    asof_forward_fill(corpus_rows, series["heartbeat_mean_ratio"])
    asof_forward_fill(corpus_rows, series["prediction_error_execution"])
    assert corpus_rows[0].channels["heartbeat_mean_ratio"] == 0.95  # carried forward from t1
    assert corpus_rows[0].channels["prediction_error_execution"] == 0.5  # t2's own reading


def test_enrich_corpus_empty_rows_is_a_no_op() -> None:
    class _NeverCalledConn:
        def cursor(self):  # pragma: no cover -- must never be reached
            raise AssertionError("enrich_corpus must not query Postgres for an empty row list")

    assert enrich_corpus([], _NeverCalledConn(), lookback_hours=48.0) == {}


def test_fetch_all_series_only_wires_dense_enough_signals() -> None:
    """Regression lock on the v4 scope decision: git/pr/graph_delta, dev_economics,
    doc_semantic_drift, and swear_frequency are real, tested, implemented fetch_*()
    functions above, but must NOT be called by fetch_all_series() -- their real cadence
    (~16min-11.6h) is far coarser than fit_encoder.py's ~60s default window, so
    forward-filling them in adds dimensionality with no real per-window trajectory signal
    (confirmed empirically: an earlier version of this module that DID wire them in failed
    the floor gate). Only action_warrant and attention_self_model (dense enough) belong in
    the default join until a per-window "context" representation is designed for the rest."""

    conn = _FakeConn()
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    until = datetime(2026, 9, 2, tzinfo=timezone.utc)

    series = fetch_all_series(conn, since, until)

    # attention_self_model contributes 5 SEPARATE series (heartbeat_mean_ratio + 4
    # prediction_error_{domain} channels), not one combined entry -- see
    # fetch_attention_self_model()'s docstring for why (independent LOCF per field).
    assert set(series.keys()) == {
        "action_warrant",
        "heartbeat_mean_ratio",
        "prediction_error_execution",
        "prediction_error_chat",
        "prediction_error_biometrics",
        "prediction_error_bus_synaptic",
    }
    joined_queries = " ".join(conn.queries).lower()
    for forbidden_table in (
        "juniper_affective_state_log",
        "substrate_codebase_delta_log",
        "dev_economics_ledger_log",
        "doc_semantic_drift_log",
    ):
        assert forbidden_table not in joined_queries, (
            f"fetch_all_series() must not query {forbidden_table} -- these signals are "
            "deferred (too sparse for the current window), not silently reintroduced"
        )
