"""Tests for orion/metrics/liveness.py -- phase 5 of the metric semantic layer.

DB access is mocked throughout (a fake conn/cursor) so this stays a fast gate
test, not a periodic eval. The real path is exercised manually against the
live Postgres instance (see the module docstring for the live numbers found
doing that on 2026-08-19/20) and via `scripts/check_metric_lineage.py --metric`.
"""
from __future__ import annotations

from unittest import mock

import pytest

from orion.field.channel_glossary import classify_channel_series
from orion.metrics.liveness import (
    CONNECT_TIMEOUT_SECONDS,
    DEFAULT_POSTGRES_URI,
    STATEMENT_TIMEOUT_MS,
    FlatColumnSource,
    LivenessOutcome,
    ScalarFieldSource,
    ThroughputSource,
    _classify_unbounded_series,
    _normalize_to_unit_scale,
    _resolve_source_kind,
    _worst_of,
    has_registered_source,
    liveness_for_node,
    open_readonly_connection,
    resolve_dsn,
    resolved_host,
)


class _FakeNode:
    """Minimal stand-in for orion.metrics.lineage.MetricNode -- only the
    fields liveness_for_node()/has_registered_source() actually read."""

    def __init__(
        self,
        *,
        name: str,
        schema_id: str | None = None,
        metric_field: str | None = None,
        surface: str | None = None,
    ):
        self.name = name
        self.schema_id = schema_id
        self.metric_field = metric_field
        self.surface = surface


class _FakeCursor:
    def __init__(self, fetchall_result=None, fetchone_result=None):
        self.fetchall_result = fetchall_result or []
        self.fetchone_result = fetchone_result
        self.executed: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return self.fetchall_result

    def fetchone(self):
        return self.fetchone_result


class _FakeConn:
    def __init__(self, cursors: list[_FakeCursor]):
        self._cursors = list(cursors)
        self.closed = False
        self.autocommit = False

    def cursor(self):
        return self._cursors.pop(0)

    def close(self):
        self.closed = True


# --------------------------------------------------------------- resolve_dsn


def test_resolve_dsn_prefers_postgres_uri(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://a/b")
    monkeypatch.setenv("DATABASE_URL", "postgresql://c/d")
    assert resolve_dsn() == "postgresql://a/b"


def test_resolve_dsn_falls_back_through_env_keys(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://c/d")
    assert resolve_dsn() == "postgresql://c/d"


def test_resolve_dsn_default_when_nothing_set(monkeypatch):
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_URL"):
        monkeypatch.delenv(key, raising=False)
    assert resolve_dsn() == DEFAULT_POSTGRES_URI


# ----------------------------------------------------------------- resolved_host


def test_resolved_host_reports_host_and_port(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:secret@myhost:5432/db")
    assert resolved_host() == "myhost:5432"


def test_resolved_host_never_leaks_credentials(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://user:supersecret@myhost:5432/db")
    assert "supersecret" not in resolved_host()
    assert "user" not in resolved_host()


def test_resolved_host_default_is_localhost(monkeypatch):
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_URL"):
        monkeypatch.delenv(key, raising=False)
    assert resolved_host() == "localhost:55432"


# ------------------------------------------------------- open_readonly_connection


def test_open_readonly_connection_returns_none_when_psycopg2_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fail_import(name, *a, **kw):
        if name == "psycopg2":
            raise ImportError("no psycopg2 here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fail_import)
    assert open_readonly_connection("postgresql://x") is None


def test_open_readonly_connection_returns_none_on_connect_failure():
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.side_effect = Exception("connection refused")
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        assert open_readonly_connection("postgresql://x") is None


def test_open_readonly_connection_closes_and_returns_none_if_not_readonly():
    cur = _FakeCursor(fetchone_result=("off",))
    conn = _FakeConn([cur])
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        result = open_readonly_connection("postgresql://x")
    assert result is None
    assert conn.closed is True


def test_open_readonly_connection_passes_connect_and_statement_timeout():
    cur = _FakeCursor(fetchone_result=("on",))
    conn = _FakeConn([cur])
    fake_psycopg2 = mock.Mock()
    fake_psycopg2.connect.return_value = conn
    with mock.patch.dict("sys.modules", {"psycopg2": fake_psycopg2}):
        result = open_readonly_connection("postgresql://x")
    assert result is conn
    fake_psycopg2.connect.assert_called_once_with(
        "postgresql://x", connect_timeout=CONNECT_TIMEOUT_SECONDS
    )
    # SET statement_timeout must actually be issued on the session --
    # regression for the "connect_timeout alone doesn't bound a hung query"
    # finding.
    executed_queries = [q for q, _params in cur.executed]
    assert any("statement_timeout" in q for q in executed_queries)


# ------------------------------------------------------------- ScalarFieldSource


def test_scalar_field_source_filters_none_and_builds_query():
    # Rows come back DESC (most-recent-first, so a LIMIT keeps the newest
    # rows -- see the class docstring); fetch() reverses to ASC for the
    # classifier, which expects chronological order.
    source = ScalarFieldSource(
        table="t", json_column="j", ts_column="ts", window_hours=1.0
    )
    cur = _FakeCursor(fetchall_result=[(0.3,), (None,), (0.1,)])
    conn = _FakeConn([cur])
    values, truncated = source.fetch(conn, "myfield")
    assert values == [0.1, 0.3]
    assert truncated is False
    query, params = cur.executed[0]
    assert "t" in query and "j" in query and "ts" in query and "DESC" in query
    assert params == ("myfield", 1.0, "myfield", 50_001)  # MAX_ROWS + 1, see fetch()'s docstring


def test_scalar_field_source_not_truncated_at_exactly_max_rows():
    """Regression: an earlier version used LIMIT MAX_ROWS with
    `len(rows) >= MAX_ROWS`, which could not distinguish "exactly MAX_ROWS
    matching rows (complete)" from "more than MAX_ROWS (truncated)" -- both
    looked identical since the fetch itself was capped at MAX_ROWS."""
    source = ScalarFieldSource(
        table="t", json_column="j", ts_column="ts", window_hours=1.0
    )
    cur = _FakeCursor(fetchall_result=[(1.0,)] * 50_000)  # exactly MAX_ROWS
    conn = _FakeConn([cur])
    values, truncated = source.fetch(conn, "myfield")
    assert truncated is False
    assert len(values) == 50_000


def test_scalar_field_source_reports_truncated_when_over_max_rows():
    source = ScalarFieldSource(
        table="t", json_column="j", ts_column="ts", window_hours=1.0
    )
    cur = _FakeCursor(fetchall_result=[(1.0,)] * 50_001)  # MAX_ROWS + 1
    conn = _FakeConn([cur])
    values, truncated = source.fetch(conn, "myfield")
    assert truncated is True
    assert len(values) == 50_000  # the extra fetched row is dropped


# ------------------------------------------------------------- FlatColumnSource


def test_flat_column_source_filters_none_and_builds_query():
    # Same DESC-then-reverse-to-ASC contract as ScalarFieldSource, but no
    # JSONB extraction -- a plain column read.
    source = FlatColumnSource(table="t", ts_column="ts", window_hours=48.0)
    cur = _FakeCursor(fetchall_result=[(0.3,), (None,), (0.1,)])
    conn = _FakeConn([cur])
    values, truncated = source.fetch(conn, "level")
    assert values == [0.1, 0.3]
    assert truncated is False
    query, params = cur.executed[0]
    assert "t" in query and "ts" in query and "level" in query and "DESC" in query
    assert "->>" not in query  # not a JSONB extraction
    assert params == (48.0, 50_001)


def test_flat_column_source_extra_where_is_appended_to_query():
    source = FlatColumnSource(table="t", ts_column="ts", window_hours=48.0)
    cur = _FakeCursor(fetchall_result=[(0.1,)])
    conn = _FakeConn([cur])
    source.fetch(conn, "level", extra_where="AND confidence > 0")
    query, params = cur.executed[0]
    assert "AND confidence > 0" in query
    assert params == (48.0, 50_001)  # extra_where carries no params of its own


def test_flat_column_source_no_extra_where_by_default():
    source = FlatColumnSource(table="t", ts_column="ts", window_hours=48.0)
    cur = _FakeCursor(fetchall_result=[(0.1,)])
    conn = _FakeConn([cur])
    source.fetch(conn, "confidence")
    query, _params = cur.executed[0]
    assert "confidence > 0" not in query


def test_flat_column_source_reports_truncated_when_over_max_rows():
    source = FlatColumnSource(table="t", ts_column="ts", window_hours=48.0)
    cur = _FakeCursor(fetchall_result=[(1.0,)] * 50_001)
    conn = _FakeConn([cur])
    values, truncated = source.fetch(conn, "confidence")
    assert truncated is True
    assert len(values) == 50_000


# ------------------------------------------------------------- ThroughputSource


def test_throughput_source_returns_bucket_counts():
    source = ThroughputSource(
        table="t", ts_column="ts", window_hours=1.0, bucket_hours=1.0 / 60
    )
    cur = _FakeCursor(fetchall_result=[(0.0,), (3.0,), (5.0,)])
    conn = _FakeConn([cur])
    values = source.fetch(conn)
    assert values == [0.0, 3.0, 5.0]
    query, params = cur.executed[0]
    assert params == (1.0, 1.0 / 60, 1.0 / 60)


# ------------------------------------------------------------------- _worst_of


def test_worst_of_all_live_is_live():
    assert _worst_of(["live", "live", "live"]) == "live"


def test_worst_of_live_and_quiet_mix_is_live_not_quiet():
    """The bug found live 2026-08-19: an earlier severity table ranked quiet
    above live, so 4 live stages + 1 expectedly-quiet slow stage rolled up to
    QUIET -- misleadingly making a healthy ladder look concerning."""
    assert _worst_of(["live", "live", "live", "live", "quiet"]) == "live"


def test_worst_of_all_quiet_is_quiet():
    assert _worst_of(["quiet", "quiet"]) == "quiet"


def test_worst_of_dead_beats_live_and_quiet():
    assert _worst_of(["live", "live", "dead", "quiet"]) == "dead"


def test_worst_of_never_produced_beats_dead():
    assert _worst_of(["dead", "never_produced", "live"]) == "never_produced"


def test_worst_of_ratchet_suspect_beats_clean_but_loses_to_dead():
    assert _worst_of(["live", "ratchet_suspect"]) == "ratchet_suspect"
    assert _worst_of(["dead", "ratchet_suspect"]) == "dead"


# ------------------------------------------------------------- _resolve_source_kind


def test_resolve_source_kind_attention_self_model_field():
    node = _FakeNode(name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence")
    assert _resolve_source_kind(node) == "attention_self_model"


def test_resolve_source_kind_ladder_signal_node():
    node = _FakeNode(name="l7_l11_ladder", schema_id=None, metric_field=None)
    assert _resolve_source_kind(node) == "ladder"


def test_resolve_source_kind_none_for_unrelated_node():
    node = _FakeNode(name="cpu_pressure", schema_id=None, metric_field=None)
    assert _resolve_source_kind(node) is None


def test_resolve_source_kind_repair_pressure_level_and_confidence():
    for field in ("level", "confidence"):
        node = _FakeNode(name="repair_pressure", surface="organ_signal", metric_field=field)
        assert _resolve_source_kind(node) == "repair_pressure"


def test_resolve_source_kind_repair_pressure_none_for_uncovered_dimension():
    """Only level/confidence are wired -- the other 10 canonical dimensions
    (coherence, tension, etc.) were never confirmed to exist as real columns
    on the backing table, so they must stay NOT_COMPUTED, not guessed."""
    node = _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="trust_rupture")
    assert _resolve_source_kind(node) is None


def test_resolve_source_kind_repair_pressure_none_for_wrong_surface():
    """Same name/field, wrong surface -- must not match. Guards against a
    future organ_signal-shaped name collision with an unrelated inner_state
    or field_channel node."""
    node = _FakeNode(name="repair_pressure", surface="inner_state", metric_field="level")
    assert _resolve_source_kind(node) is None


def test_has_registered_source_and_liveness_for_node_agree_on_every_kind():
    """Regression for the routing-drift finding: has_registered_source() and
    liveness_for_node() both delegate to _resolve_source_kind() now, so they
    cannot silently disagree the way two independent conditional copies
    could. Exercises both functions across a representative node set and
    asserts they always agree on "is this covered"."""
    nodes = [
        _FakeNode(name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence"),
        _FakeNode(name="l7_l11_ladder", schema_id=None, metric_field=None),
        _FakeNode(name="cpu_pressure", schema_id=None, metric_field=None),
        _FakeNode(name="attention_self_model.v1", schema_id=None, metric_field=None),
        _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="level"),
        _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="trust_rupture"),
    ]
    for node in nodes:
        assert has_registered_source(node) == (_resolve_source_kind(node) is not None)


# ------------------------------------------------------------- has_registered_source


def test_has_registered_source_true_for_attention_self_model_field():
    node = _FakeNode(name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence")
    assert has_registered_source(node) is True


def test_has_registered_source_true_for_ladder_signal_node():
    node = _FakeNode(name="l7_l11_ladder", schema_id=None, metric_field=None)
    assert has_registered_source(node) is True


def test_has_registered_source_false_for_unrelated_node():
    node = _FakeNode(name="cpu_pressure", schema_id=None, metric_field=None)
    assert has_registered_source(node) is False


def test_has_registered_source_true_for_repair_pressure_level():
    node = _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="level")
    assert has_registered_source(node) is True


def test_has_registered_source_false_for_repair_pressure_uncovered_dimension():
    node = _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="coherence")
    assert has_registered_source(node) is False


def test_has_registered_source_false_for_attention_self_model_signal_level_node():
    """The parent signal node itself (metric_field=None) has no scalar to
    sample -- only its per-field children do."""
    node = _FakeNode(name="attention_self_model.v1", schema_id=None, metric_field=None)
    assert has_registered_source(node) is False


# ------------------------------------------------------------- liveness_for_node


def test_liveness_for_node_computes_attention_self_model_field():
    node = _FakeNode(name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence")
    cur = _FakeCursor(fetchall_result=[(0.1,), (0.5,), (0.9,)])
    conn = _FakeConn([cur])
    outcome = liveness_for_node(node, conn)
    assert isinstance(outcome, LivenessOutcome)
    assert outcome.verdict == "live"
    assert outcome.sample_count == 3
    assert outcome.truncated is False
    assert "confidence" in outcome.detail


def test_liveness_for_node_broadcast_lane_age_reports_ratchet_suspect_for_a_stuck_lane():
    """CORRECTION 2026-08-20 (a third review round caught a round-2
    regression): broadcast_lane_age_sec is intentionally NOT routed through
    the unbounded/ratchet-downgrading classifier the ladder uses. Unlike a
    row count, this age is EXPECTED to reset toward zero on every real
    broadcast-lane refresh (schema's own `broadcast_lane_stale` sibling
    field exists for exactly this). A monotonic climb across the whole
    window means the lane never refreshed once -- a genuine stuck-lane
    signal, and downgrading it to `live` would be a false-positive-healthy
    verdict strictly worse than the honest NOT_COMPUTED this field had
    before phase 5 shipped."""
    node = _FakeNode(name="broadcast_lane_age_sec", schema_id="AttentionSelfModelV1", metric_field="broadcast_lane_age_sec")
    # fetch() expects DESC (newest-first) rows and reverses to chronological
    # ASC -- supplying DESC order here so the classified series is really
    # [5, 12, 18, 25, 30] chronologically: climbing, never resetting.
    stuck_lane_age_desc = [(30.0,), (25.0,), (18.0,), (12.0,), (5.0,)]
    cur = _FakeCursor(fetchall_result=stuck_lane_age_desc)
    conn = _FakeConn([cur])
    outcome = liveness_for_node(node, conn)
    assert outcome.verdict == "ratchet_suspect"


def test_liveness_for_node_broadcast_lane_age_with_real_resets_reads_live():
    """The healthy counterpart: an age series with real resets (sawtooth,
    matching a lane that actually refreshes) is not monotonic and correctly
    reads as ordinary live variance, no special-casing needed."""
    node = _FakeNode(name="broadcast_lane_age_sec", schema_id="AttentionSelfModelV1", metric_field="broadcast_lane_age_sec")
    # DESC input reversed to chronological ASC [0.5, 2.0, 0.1, 3.5, 0.2, 4.0].
    healthy_sawtooth_desc = [(4.0,), (0.2,), (3.5,), (0.1,), (2.0,), (0.5,)]
    cur = _FakeCursor(fetchall_result=healthy_sawtooth_desc)
    conn = _FakeConn([cur])
    outcome = liveness_for_node(node, conn)
    assert outcome.verdict == "live"


def test_liveness_for_node_computes_ladder_throughput_rollup():
    node = _FakeNode(name="l7_l11_ladder", schema_id=None, metric_field=None)
    # 5 stages, each its own cursor -- fetch() opens one cursor per stage.
    cursors = [_FakeCursor(fetchall_result=[(1.0,), (2.0,), (3.0,)]) for _ in range(5)]
    conn = _FakeConn(cursors)
    outcome = liveness_for_node(node, conn)
    assert isinstance(outcome, LivenessOutcome)
    assert outcome.verdict == "live"
    assert "substrate_proposal_frames=" in outcome.detail


def test_liveness_for_node_returns_none_for_unregistered_node():
    node = _FakeNode(name="cpu_pressure", schema_id=None, metric_field=None)
    conn = _FakeConn([])
    assert liveness_for_node(node, conn) is None


def test_liveness_for_node_computes_repair_pressure_field():
    """confidence is NOT gated -- how often the appraiser has zero
    confidence is itself real liveness information, not noise to filter."""
    node = _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="confidence")
    cur = _FakeCursor(fetchall_result=[(0.65,), (0.087,), (0.194,)])
    conn = _FakeConn([cur])
    outcome = liveness_for_node(node, conn)
    assert isinstance(outcome, LivenessOutcome)
    assert outcome.verdict == "live"
    assert outcome.sample_count == 3
    assert "repair_pressure_appraisal_log.confidence" in outcome.detail
    query, _params = cur.executed[0]
    assert "confidence > 0" not in query


def test_liveness_for_node_repair_pressure_level_is_confidence_gated():
    """Regression for the floor-domination finding (scripts/analysis/
    measure_metacog_trend_baseline.py, 2026-07-30): `level` read ungated is
    contaminated by the appraiser's confidence==0.0 default. The query must
    carry the confidence>0 gate; without this test a future edit could drop
    it silently and reintroduce the exact contamination that finding
    documents."""
    node = _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="level")
    cur = _FakeCursor(fetchall_result=[(0.12,), (0.19,)])
    conn = _FakeConn([cur])
    outcome = liveness_for_node(node, conn)
    query, _params = cur.executed[0]
    assert "confidence > 0" in query
    assert "gated" in outcome.detail


def test_liveness_for_node_repair_pressure_dead_when_all_zero():
    """A real, plausible failure mode for this signal: even among
    confidence-gated (trustworthy) readings, level never actually
    elevates -- the appraiser fires with real evidence but never detects a
    rupture."""
    node = _FakeNode(name="repair_pressure", surface="organ_signal", metric_field="level")
    cur = _FakeCursor(fetchall_result=[(0.0,), (0.0,), (0.0,)])
    conn = _FakeConn([cur])
    outcome = liveness_for_node(node, conn)
    assert outcome.verdict == "dead"


def test_liveness_for_node_ladder_sample_count_is_real_row_sum_not_bucket_count():
    """Regression: an earlier version counted non-empty buckets (~1 per
    bucket regardless of how many rows it held), not actual rows -- so a
    ~2s-cadence stage with ~28 rows/bucket understated its real volume by
    ~28x. sample_count must be a true row total, comparable to the scalar
    source's `len(values)`."""
    node = _FakeNode(name="l7_l11_ladder", schema_id=None, metric_field=None)
    cursors = [_FakeCursor(fetchall_result=[(10.0,), (0.0,), (20.0,)]) for _ in range(5)]
    conn = _FakeConn(cursors)
    outcome = liveness_for_node(node, conn)
    assert outcome.sample_count == 5 * 30  # 10+0+20 per stage, 5 stages


def test_liveness_for_node_propagates_query_failure():
    """liveness_for_node() does NOT swallow a query-time DB error itself --
    scripts/check_metric_lineage.py's _liveness_for_nodes is the one place
    that catches it and degrades to UNKNOWN. Verifies the exception is not
    silently eaten here, which would make that outer catch dead code."""
    node = _FakeNode(name="confidence", schema_id="AttentionSelfModelV1", metric_field="confidence")

    class _BrokenCursor(_FakeCursor):
        def execute(self, query, params=None):
            raise RuntimeError("connection reset by peer")

    conn = _FakeConn([_BrokenCursor()])
    with pytest.raises(RuntimeError):
        liveness_for_node(node, conn)


# --------------------------------------------------------- _normalize_to_unit_scale


def test_normalize_scales_by_own_max():
    assert _normalize_to_unit_scale([5.0, 10.0, 20.0]) == [0.25, 0.5, 1.0]


def test_normalize_leaves_all_zero_series_unchanged():
    """Dividing by a zero max would raise -- an all-zero series (dead) must
    pass through unchanged so classify_channel_series still sees it as dead,
    not crash."""
    assert _normalize_to_unit_scale([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_normalize_leaves_empty_series_unchanged():
    assert _normalize_to_unit_scale([]) == []


def test_normalize_alone_does_not_fix_ratchet_false_positive():
    """Documents why _classify_unbounded_series() exists instead of just
    calling classify_channel_series(_normalize_to_unit_scale(series)):
    rescaling to unit scale does NOT stop a monotonic series from tripping
    ratchet_suspect, because climbed = last-first is measured against the
    series' own new max of 1.0 -- almost any monotonic climb clears
    LIVE_VARIANCE_THRESHOLD=0.05 regardless of the original scale."""
    raw_counts = [5.0, 12.0, 18.0, 25.0, 30.0]  # non-decreasing, climbs by 25
    assert classify_channel_series(raw_counts) == "ratchet_suspect"
    assert classify_channel_series(_normalize_to_unit_scale(raw_counts)) == "ratchet_suspect"


def test_classify_unbounded_series_downgrades_ratchet_to_live():
    """The actual fix: a monotonic climb -- unremarkable for a pipeline
    throughput count or an unbounded age, unlike a never-decaying [0,1]
    channel -- reads as LIVE, not as a suspected stuck accumulator."""
    raw_counts = [5.0, 12.0, 18.0, 25.0, 30.0]
    assert _classify_unbounded_series(raw_counts) == "live"


def test_classify_unbounded_series_still_detects_dead():
    assert _classify_unbounded_series([0.0, 0.0, 0.0, 0.0]) == "dead"


def test_classify_unbounded_series_still_detects_never_produced():
    assert _classify_unbounded_series([]) == "never_produced"


def test_statement_timeout_ms_is_a_positive_bound():
    assert STATEMENT_TIMEOUT_MS > 0


# --------------------------------------------------- repair_pressure invariant


def test_repair_pressure_signal_kind_is_unique_across_organ_registry():
    """`_resolve_source_kind`'s repair_pressure branch disambiguates only by
    (surface, name, metric_field) -- no check against producer_service or
    organ identity -- relying on this invariant: exactly one organ in
    ORGAN_REGISTRY declares a "repair_pressure" signal_kind (confirmed
    2026-09-04). If a second organ ever adds one, this fails loudly instead
    of silently routing that organ's liveness queries to
    repair_pressure_appraisal_log -- the wrong table entirely."""
    from orion.signals.registry import ORGAN_REGISTRY

    organs_with_repair_pressure = [
        organ_id
        for organ_id, entry in ORGAN_REGISTRY.items()
        if "repair_pressure" in entry.signal_kinds
    ]
    assert organs_with_repair_pressure == ["graph_cognition"], (
        "a second organ now declares a 'repair_pressure' signal_kind -- "
        "orion.metrics.liveness._resolve_source_kind's repair_pressure branch "
        "must be updated to also check producer_service/organ identity "
        "before this can be trusted again"
    )
