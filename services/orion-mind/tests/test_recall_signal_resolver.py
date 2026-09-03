"""recall_signal_resolver.py: the render-gate resolver for recall's
bus_synaptic transport fragments.

Fixture values are hand-computed, not read back from the implementation --
in particular the render-gate threshold (0.15) is asserted against a literal
0.15/0.25 pair, not against settings.RECALL_TRANSPORT_RENDER_GATE_THRESHOLD,
so a future accidental revert of the 0.25->0.15 fix would fail this suite.

conftest.py's autouse fixture re-binds `app` to this service before every
test in this directory.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def _recent_epoch(offset_sec: float = 60.0) -> float:
    return time.time() - offset_sec


def _fragment(
    *,
    channel="orion:vision:edge:health",
    organ="vision-edge",
    signal_kind="publish_gap_zscore",
    last_seen_epoch=None,
):
    return {
        "id": f"bus_synaptic_publish:{organ}:{channel}",
        "source": "bus_synaptic_anomaly",
        "text": "",
        "meta": {
            "organ_id": organ,
            "channel": channel,
            "signal_kind": signal_kind,
            "zscore": 7.1,
            "last_seen_epoch": _recent_epoch(30.0) if last_seen_epoch is None else last_seen_epoch,
        },
    }


# --------------------------------------------------- partition


def test_partition_separates_handled_from_passthrough():
    from app.recall_signal_resolver import partition_bus_synaptic_fragments

    handled = _fragment()
    causal = _fragment(signal_kind="causal_latency_zscore")
    other = {"id": "x", "meta": {}}
    passthrough, bus_synaptic = partition_bus_synaptic_fragments([handled, causal, other])
    assert bus_synaptic == [handled]
    assert passthrough == [causal, other]


def test_partition_handles_missing_meta():
    from app.recall_signal_resolver import partition_bus_synaptic_fragments

    frag = {"id": "no-meta"}
    passthrough, bus_synaptic = partition_bus_synaptic_fragments([frag])
    assert passthrough == [frag]
    assert bus_synaptic == []


def test_partition_empty_input():
    from app.recall_signal_resolver import partition_bus_synaptic_fragments

    assert partition_bus_synaptic_fragments([]) == ([], [])
    assert partition_bus_synaptic_fragments(None) == ([], [])


# --------------------------------------------------- fetch_bus_synaptic_prediction_error_series


def test_series_fetch_empty_dsn_returns_empty():
    from app.recall_signal_resolver import fetch_bus_synaptic_prediction_error_series

    assert fetch_bus_synaptic_prediction_error_series("") == []
    assert fetch_bus_synaptic_prediction_error_series(None) == []  # type: ignore[arg-type]


def test_series_fetch_no_connection_returns_empty(monkeypatch):
    """open_readonly_connection() itself fails open to None on any
    connect/auth/read-only-enforcement failure -- confirm this function
    treats that as [] rather than raising."""
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "open_readonly_connection", lambda *a, **k: None)
    assert resolver.fetch_bus_synaptic_prediction_error_series("postgresql://x") == []


def test_series_fetch_uses_the_canonical_readonly_helper_with_bounded_timeouts(monkeypatch):
    """Regression test (code review, 2026-09-03): an earlier draft hand-rolled
    psycopg2.connect + SET LOCAL statement_timeout after autocommit=True,
    which is silently a no-op under autocommit (SET LOCAL only holds for the
    remainder of the current transaction). open_readonly_connection() uses a
    session-level SET statement_timeout instead, which is correct."""
    import app.recall_signal_resolver as resolver

    calls = []

    def _fake_open(dsn, *, connect_timeout=None, statement_timeout_ms=None):
        calls.append({"dsn": dsn, "connect_timeout": connect_timeout, "statement_timeout_ms": statement_timeout_ms})
        return _FakeConn([])

    monkeypatch.setattr(resolver, "open_readonly_connection", _fake_open)
    resolver.fetch_bus_synaptic_prediction_error_series("postgresql://x")

    assert len(calls) == 1
    assert calls[0]["dsn"] == "postgresql://x"
    assert calls[0]["connect_timeout"] is not None
    assert calls[0]["statement_timeout_ms"] is not None


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        self.closed = True


def test_series_fetch_reverses_to_oldest_first(monkeypatch):
    import app.recall_signal_resolver as resolver

    rows = [("0.30", "t3"), ("0.20", "t2"), ("0.10", "t1")]
    fake_conn = _FakeConn(rows)
    monkeypatch.setattr(resolver, "open_readonly_connection", lambda *a, **k: fake_conn)

    out = resolver.fetch_bus_synaptic_prediction_error_series("postgresql://x")

    assert out == [0.10, 0.20, 0.30]
    assert fake_conn.closed


def test_series_fetch_skips_null_and_unparseable_values(monkeypatch):
    import app.recall_signal_resolver as resolver

    rows = [("not-a-float", "t3"), (None, "t2"), ("0.05", "t1")]
    fake_conn = _FakeConn(rows)
    monkeypatch.setattr(resolver, "open_readonly_connection", lambda *a, **k: fake_conn)

    out = resolver.fetch_bus_synaptic_prediction_error_series("postgresql://x")

    assert out == [0.05]


def test_series_fetch_query_exception_still_closes_connection_and_fails_open(monkeypatch):
    import app.recall_signal_resolver as resolver

    class _RaisingCursor:
        def execute(self, *a, **k):
            raise RuntimeError("query failed")

    class _ConnWithRaisingCursor:
        def __init__(self):
            self.closed = False

        def cursor(self):
            return _RaisingCursor()

        def close(self):
            self.closed = True

    fake_conn = _ConnWithRaisingCursor()
    monkeypatch.setattr(resolver, "open_readonly_connection", lambda *a, **k: fake_conn)

    out = resolver.fetch_bus_synaptic_prediction_error_series("postgresql://x")

    assert out == []
    assert fake_conn.closed


# --------------------------------------------------- render_bus_synaptic_digest_line


@pytest.fixture(autouse=True)
def _no_lattice_policy_by_default(monkeypatch):
    """Most render tests don't care about the ladder-rung display detail --
    default to "policy file not found" so those tests aren't coupled to a
    real config/ path. Tests that DO care override this explicitly."""
    import app.recall_signal_resolver as resolver

    resolver._load_bus_synaptic_lattice_rungs.cache_clear()
    monkeypatch.setattr(resolver, "_lattice_policy_path_candidates", lambda: [Path("/nonexistent")])
    yield
    resolver._load_bus_synaptic_lattice_rungs.cache_clear()


def test_render_returns_none_without_touching_postgres_when_dsn_unconfigured(monkeypatch):
    import app.recall_signal_resolver as resolver

    called = []
    monkeypatch.setattr(
        resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: called.append(1) or [0.5]
    )
    out = resolver.render_bus_synaptic_digest_line([], dsn="", render_gate_threshold=0.15)
    assert out is None
    assert called == []  # must not even touch Postgres when the feature is off


def test_render_checks_postgres_even_with_no_handled_fragments_when_dsn_is_set(monkeypatch):
    """Regression test (code review, 2026-09-03): an earlier draft gated the
    entire liveness check on handled_fragments being non-empty, reasoning
    that empty meant "recall didn't try this turn". But recall's per-edge
    Falkor fetch fails open to [] on a Falkor/bus-mirror outage too --
    indistinguishable from "didn't try" by that signal alone -- so a real
    outage silently skipped this Postgres-backed check entirely, exactly
    the "outage reads as silence" failure this liveness path exists to
    prevent. The gate is now the dsn being configured, not the fragment
    list's emptiness -- substrate_field_state is written by a DIFFERENT
    service's own Falkor query, so its freshness doesn't depend on
    recall's Falkor connection being healthy right now."""
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [])
    out = resolver.render_bus_synaptic_digest_line([], dsn="postgresql://x", render_gate_threshold=0.15)
    assert out == resolver._NOT_WRITING_TEXT


def test_render_not_writing_when_series_empty(monkeypatch):
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [])
    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)
    assert out == resolver._NOT_WRITING_TEXT
    assert "unknown, not calm" in out


def test_render_degenerate_when_series_is_flat_zero(monkeypatch):
    import app.recall_signal_resolver as resolver

    # The exact failure mode confirmed live in _bus_synaptic_tick: the tick
    # keeps firing and writing a real, fresh 0.0 when the edge set is empty.
    monkeypatch.setattr(
        resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.0, 0.0, 0.0, 0.0]
    )
    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)
    assert out == resolver._DEGENERATE_ZERO_TEXT
    assert "not genuine calm" in out


def test_render_degenerate_when_only_the_tail_is_zero(monkeypatch):
    """Regression test (code review, 2026-09-03): classify_channel_series()
    only returns "dead" when EVERY value in the window is subnormal. An
    outage that started partway through the window -- older real values
    still present, only the most recent rows are 0.0 -- reads as "quiet" or
    "live" instead, with latest==0.0. Left unchecked, that 0.0 then fails
    the render gate and returns None: silence, not the degenerate state --
    the exact "outage reads as calm/silent" failure this whole liveness
    path exists to prevent, just delayed to partway through the window
    instead of eliminated."""
    import app.recall_signal_resolver as resolver

    series = [0.02, 0.03, 0.02, 0.0, 0.0]  # real history, then the outage starts
    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: series)
    from orion.field.channel_glossary import classify_channel_series

    assert classify_channel_series(series) != "dead"  # sanity: this really is the gap, not a redundant case

    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)
    assert out == resolver._DEGENERATE_ZERO_TEXT


def test_render_none_below_gate(monkeypatch):
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.10, 0.12, 0.14])
    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)
    assert out is None


def test_render_fires_at_the_gate_not_the_old_spec_value(monkeypatch):
    """The whole point of the handoff: 0.20 (between 0.15 and 0.25) must
    render, because equilibrium already reflects at 0.15."""
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.10, 0.15, 0.20])
    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)
    assert out is not None
    assert "Transport:" in out
    assert "20%" in out


def test_render_includes_loudest_channel(monkeypatch):
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.30])
    frag = _fragment(channel="orion:vision:edge:health", organ="vision-edge")
    out = resolver.render_bus_synaptic_digest_line([frag], dsn="x", render_gate_threshold=0.15)
    assert "orion:vision:edge:health from vision-edge" in out


def test_render_does_not_attribute_loudest_to_a_frozen_edge(monkeypatch):
    """Regression test (code review, 2026-09-03): falkor_bus_synaptic_
    adapter.py dropped its recency filter on the publish-gap query, so a
    permanently frozen edge (e.g. a dead channel stuck at a high z-score for
    weeks) can now sort first in the adapter's own ORDER BY abs(z) DESC and
    stay there forever. Without a freshness check here, "loudest right now"
    would misattribute a real, currently-happening incident to that zombie
    channel indefinitely."""
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.30])
    frozen = _fragment(
        channel="orion:dream:log", organ="orion-dream", last_seen_epoch=time.time() - 30 * 86400
    )
    out = resolver.render_bus_synaptic_digest_line([frozen], dsn="x", render_gate_threshold=0.15)
    assert out is not None
    assert "Loudest right now" not in out
    assert "orion:dream:log" not in out


def test_render_skips_a_frozen_edge_and_names_the_next_fresh_one(monkeypatch):
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.30])
    frozen = _fragment(
        channel="orion:dream:log", organ="orion-dream", last_seen_epoch=time.time() - 30 * 86400
    )
    fresh = _fragment(channel="orion:vision:edge:health", organ="vision-edge")
    out = resolver.render_bus_synaptic_digest_line([frozen, fresh], dsn="x", render_gate_threshold=0.15)
    assert "orion:vision:edge:health from vision-edge" in out
    assert "orion:dream:log" not in out


def test_render_includes_ladder_rungs_when_policy_loads(monkeypatch, tmp_path):
    import app.recall_signal_resolver as resolver

    policy = tmp_path / "transport_lattice_policy.v1.yaml"
    policy.write_text(
        "channels:\n"
        "  bus_synaptic_pressure:\n"
        "    watch_at: 0.25\n"
        "    summarize_at: 0.50\n"
        "    propose_at: 0.75\n"
    )
    resolver._load_bus_synaptic_lattice_rungs.cache_clear()
    monkeypatch.setattr(resolver, "_lattice_policy_path_candidates", lambda: [policy])
    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.31])

    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)

    assert "0.25 watch threshold" in out
    assert "0.50 summarize" in out
    resolver._load_bus_synaptic_lattice_rungs.cache_clear()


def test_render_degrades_gracefully_without_lattice_policy(monkeypatch):
    import app.recall_signal_resolver as resolver

    # Already the default fixture state (path candidates -> nonexistent).
    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.31])
    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)
    assert out is not None
    assert "watch threshold" not in out


def test_render_includes_trend_source_from_glossary(monkeypatch):
    import app.recall_signal_resolver as resolver

    monkeypatch.setattr(resolver, "fetch_bus_synaptic_prediction_error_series", lambda *a, **k: [0.31])
    out = resolver.render_bus_synaptic_digest_line([_fragment()], dsn="x", render_gate_threshold=0.15)
    assert "substrate_field_state" in out
