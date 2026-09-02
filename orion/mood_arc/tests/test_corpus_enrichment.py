"""Tests for orion/mood_arc/corpus_enrichment.py's asof/forward-fill join logic.

Pure logic, synthetic timestamps -- no Postgres connection anywhere in this file. The
fetch_*() functions themselves (real SQL against real tables) are exercised by hand against
live Postgres as part of the v4 retrain, not here (see orion/mood_arc/README.md's v4 section)
-- they are thin, directly-inspectable query wrappers, and a mocked-cursor test of them would
mostly just be re-asserting the SQL string back at itself.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.mood_arc.corpus_enrichment import asof_forward_fill, enrich_corpus, fetch_all_series
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

    class _RecordingConn:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def cursor(self):
            conn = self

            class _Cur:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *exc):
                    return False

                def execute(self_inner, query, params):
                    conn.queries.append(query)

                def fetchall(self_inner):
                    return []

            return _Cur()

    conn = _RecordingConn()
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    until = datetime(2026, 9, 2, tzinfo=timezone.utc)

    series = fetch_all_series(conn, since, until)

    assert set(series.keys()) == {"action_warrant", "attention_self_model"}
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
