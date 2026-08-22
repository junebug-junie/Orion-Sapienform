from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import field_channel_glossary_routes  # noqa: E402


def _field_state_row(overrides: dict) -> dict:
    node_vectors = {
        "node:athena": {
            "availability": 1.0,
            "cpu_pressure": 0.0,
            "expected_offline_suppression": 1.0,
            "stream_backlog_health": 1.0,
            "delivery_confidence": 1.0,
        }
    }
    node_vectors["node:athena"].update(overrides)
    return {
        "schema_version": "field.state.v1",
        "generated_at": "2026-07-20T00:00:00+00:00",
        "tick_id": "tick-test",
        "node_vectors": node_vectors,
        "capability_vectors": {},
        "edges": [],
        "recent_perturbations": [],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    field_channel_glossary_routes._engine_instance = None
    app = FastAPI()
    app.include_router(field_channel_glossary_routes.router)
    return TestClient(app)


def _fake_engine_with_rows(field_jsons: list[dict]):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value = [(json.dumps(fj),) for fj in field_jsons]
    return fake_engine


def test_channels_endpoint_returns_38_raw_channels_plus_1_derived(client):
    r = client.get("/api/field-channel-glossary/channels")
    assert r.status_code == 200
    body = r.json()
    # 38 raw pre-merge channels + 1 derived scalar (tension_deviation_pressure,
    # 2026-08-16) -- see that entry's own comment in config/field/
    # field_channel_glossary.v1.yaml for why it's listed here despite not
    # being one of the 38.
    assert len(body["channels"]) == 39
    assert len(body["categories"]) == 7
    channels_by_name = {c["channel"]: c for c in body["channels"]}
    assert "tension_deviation_pressure" in channels_by_name
    assert channels_by_name["tension_deviation_pressure"]["level"] == ["field"]
    # No self_state_dimension: that field means "routes through
    # CHANNEL_DIMENSION_MAP's merge", which this scalar deliberately does
    # not do -- see the yaml entry's own comment.
    assert channels_by_name["tension_deviation_pressure"]["self_state_dimension"] is None


def test_health_endpoint_classifies_dead_channel(client):
    rows = [_field_state_row({"cpu_pressure": 0.0}) for _ in range(5)]
    fake_engine = _fake_engine_with_rows(rows)

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")

    assert r.status_code == 200
    body = r.json()
    assert body["window_hours"] == 1
    assert body["row_count"] == 5
    by_channel = {c["channel"]: c for c in body["channels"]}
    assert by_channel["cpu_pressure"]["verdict"] == "dead"
    assert by_channel["cpu_pressure"]["clean"] is False


def test_health_endpoint_classifies_live_channel(client):
    values = [0.1, 0.4, 0.15, 0.6, 0.2]
    rows = [_field_state_row({"cpu_pressure": v}) for v in values]
    fake_engine = _fake_engine_with_rows(rows)

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")

    body = r.json()
    by_channel = {c["channel"]: c for c in body["channels"]}
    assert by_channel["cpu_pressure"]["verdict"] == "live"
    assert by_channel["cpu_pressure"]["clean"] is True
    assert by_channel["cpu_pressure"]["sample_count"] == 5


def test_health_endpoint_defaults_invalid_hours_to_default(client):
    fake_engine = _fake_engine_with_rows([_field_state_row({})])

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=999")

    assert r.status_code == 200
    assert r.json()["window_hours"] == field_channel_glossary_routes.DEFAULT_HOURS


def test_build_channel_series_quiet_zero_channel_is_dead_not_never_produced():
    # stream_backlog_pressure is never in PRESSURE_CHANNELS/HIGHER_IS_BETTER_CHANNELS,
    # so a tick where every source reads exactly 0.0 means
    # collect_field_channel_pressures() drops it from the merge -- but the
    # raw node vector still genuinely carries the key at 0.0 (reconcile
    # seeds it). build_channel_series() must record that as a real 0.0
    # sample, not silently treat the channel as absent/never-wired.
    rows = [_field_state_row({"stream_backlog_pressure": 0.0}) for _ in range(3)]
    series, _stamps, _ticks, unparsable = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["stream_backlog_pressure"]
    )
    assert unparsable == 0
    assert series["stream_backlog_pressure"] == [0.0, 0.0, 0.0]
    assert field_channel_glossary_routes.classify_channel_series(series["stream_backlog_pressure"]) == "dead"


def test_build_channel_series_genuinely_never_produced_when_key_absent_everywhere():
    rows = [_field_state_row({}) for _ in range(3)]
    series, _stamps, _ticks, _unparsable = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["stream_backlog_pressure"]
    )
    assert series["stream_backlog_pressure"] == []
    assert field_channel_glossary_routes.classify_channel_series(series["stream_backlog_pressure"]) == "never_produced"


def test_build_channel_series_counts_unparsable_rows_separately_from_dead():
    rows = [_field_state_row({}), {"not": "a valid field state"}, _field_state_row({})]
    series, _stamps, _ticks, unparsable = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]
    )
    assert unparsable == 1
    assert len(series["cpu_pressure"]) == 2


# --- tension_deviation_pressure (2026-08-16) --------------------------------


def _field_state_row_with_tension(deviation_pressure: float) -> dict:
    row = _field_state_row({})
    row["tension_deviation_pressure"] = deviation_pressure
    return row


def test_build_channel_series_always_collects_tension_deviation_pressure():
    """Unlike every other series here, this one is not gated on
    known_channels/raw_keys presence -- it lives directly on FieldStateV1
    and is always a real float, never missing."""
    rows = [_field_state_row_with_tension(v) for v in (0.0, 0.3, 0.9)]
    series, stamps, tick_times, unparsable = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]  # deliberately NOT listing it
    )
    assert unparsable == 0
    assert series["tension_deviation_pressure"] == [0.0, 0.3, 0.9]
    # No channel-merge provenance winner to time -- see build_channel_series()'s
    # own docstring.
    assert stamps["tension_deviation_pressure"] == [None, None, None]
    assert len(tick_times["tension_deviation_pressure"]) == 3


def test_health_endpoint_classifies_tension_deviation_pressure(client):
    # Non-monotonic on purpose (unlike a plain ramp up to the max) -- a
    # monotonically non-decreasing climb of this size would classify as
    # ratchet_suspect instead of live (classify_channel_series()'s own
    # rule), and this test wants the ordinary "real variance" path.
    chronological_values = [0.0, 0.3, 0.0, 0.85, 0.1]  # oldest -> newest
    # _fake_engine_with_rows simulates the route's real `ORDER BY
    # generated_at DESC` query, so rows must be handed in newest-first --
    # see test_health_endpoint_reverses_desc_rows_back_to_chronological_order.
    rows = [_field_state_row_with_tension(v) for v in reversed(chronological_values)]
    fake_engine = _fake_engine_with_rows(rows)

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")

    body = r.json()
    by_channel = {c["channel"]: c for c in body["channels"]}
    row = by_channel["tension_deviation_pressure"]
    assert row["verdict"] == "live"
    assert row["sample_count"] == 5
    assert row["min"] == 0.0
    assert row["max"] == 0.85
    assert row["last"] == 0.1


def test_health_endpoint_reverses_desc_rows_back_to_chronological_order(client):
    # The route issues `ORDER BY generated_at DESC LIMIT :row_cap` -- when
    # Postgres has already truncated to ROW_CAP rows (simulated here by the
    # mock returning exactly that many, newest-first), the route must
    # reverse them back to ascending chronological order before building
    # series. Otherwise "last" (meant to be the most recent reading) would
    # actually be the oldest one in the truncated set, and a monotonically
    # climbing real signal would look like it was falling.
    chronological_values = [0.1, 0.4, 0.7]  # oldest -> newest
    rows_desc_from_db = [
        _field_state_row({"cpu_pressure": v}) for v in reversed(chronological_values)
    ]
    fake_engine = _fake_engine_with_rows(rows_desc_from_db)

    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")

    by_channel = {c["channel"]: c for c in r.json()["channels"]}
    cpu = by_channel["cpu_pressure"]
    # If the DESC rows were used as-is (bug), "last" would read 0.1 (the
    # oldest tick, first in the DESC list) instead of 0.7 (the true most
    # recent reading).
    assert cpu["last"] == pytest.approx(0.7)
    assert cpu["min"] == pytest.approx(0.1)
    assert cpu["max"] == pytest.approx(0.7)


# --------------------------------------------------------------------------
# R2 regime readout (PR #1633), wired into /health alongside the verdict
# --------------------------------------------------------------------------

def test_regime_splits_the_window_and_declares_the_span() -> None:
    """The split IS the window: `regime.window_seconds` must report the sliced
    span, not the requested `hours`, or a consumer reads a 15-minute reading as
    an hour-long one.

    Hand-computed: 100 samples, REGIME_WINDOW_FRACTION 0.25 -> split at 75, so
    the window is the last 25 and the baseline the first 75.
      window_seconds = 3600 * (25/100) = 900.0
    """
    from scripts.field_channel_glossary_routes import _regime_for

    values = [0.1 + (i % 10) * 0.01 for i in range(100)]
    r = _regime_for("cpu_pressure", values, 1)

    assert r.sample_count == 25
    assert r.window_seconds == 900.0
    # The baseline half is real, so the relative axes are populated rather
    # than None.
    assert r.level_percentile is not None
    assert r.drift is not None


def test_regime_separates_two_channels_the_verdict_calls_the_same_word() -> None:
    """The whole reason R2 exists. Both series below classify as "quiet"
    (max - median <= 0.05), but one is idling near zero and the other is
    running near its ceiling.

    Hand-computed: flat 0.02 vs flat 0.81, 100 samples each.
    """
    from orion.field.channel_glossary import classify_channel_series
    from scripts.field_channel_glossary_routes import _regime_for

    idle = [0.02] * 100
    loaded = [0.81] * 100
    assert classify_channel_series(idle) == classify_channel_series(loaded) == "quiet"

    # Flat series have no producer writes to infer, so both read no_new_input;
    # the LEVEL axis is what distinguishes them and it survives.
    assert _regime_for("cpu_pressure", idle, 1).level == 0.02
    assert _regime_for("cpu_pressure", loaded, 1).level == 0.81


def test_regime_survives_an_empty_series_without_fabricating() -> None:
    from scripts.field_channel_glossary_routes import _regime_for

    r = _regime_for("absent_channel", [], 1)
    assert r.sample_count == 0
    assert r.regime == "insufficient_samples"
    assert r.dispersion is None
    assert r.drift is None
    # Falls back to the full requested span rather than 0.0, which would read
    # as "measured over no time at all".
    assert r.window_seconds == 3600.0


# --- producer write timestamps behind the merged value (2026-08-14) ---------
#
# The panel reported `refresh_evidence: value_ratio` for all 38 channels
# because build_channel_series() discarded node_vector_updated_at. The
# value-ratio fallback is blind in the subnormal range, where decay stops
# producing a 0.92 ratio. Measured across four consecutive 1-hour windows:
# 22-25 of 38 channels move to the timestamp path and 3-12 verdicts change,
# in both directions -- e.g. `stream_backlog_pressure` was called `static`
# while a producer really was writing it. Both figures are window-sensitive.


def _row_with_stamps(vectors: dict, stamps: dict, generated_at: str) -> dict:
    return {
        "schema_version": "field.state.v1",
        "generated_at": generated_at,
        "tick_id": "tick-test",
        "node_vectors": vectors,
        "node_vector_updated_at": stamps,
        "capability_vectors": {},
        "edges": [],
        "recent_perturbations": [],
    }


def test_stamps_are_aligned_with_series_and_take_the_winners_time():
    """max() picks one source; the honest timestamp is THAT source's."""
    rows = [
        _row_with_stamps(
            {
                "node:athena": {"cpu_pressure": 0.2},
                "node:circe": {"cpu_pressure": 0.9},  # wins the max()
            },
            {
                "node:athena": {"cpu_pressure": "2026-08-14T00:00:00+00:00"},
                "node:circe": {"cpu_pressure": f"2026-08-14T0{i}:30:00+00:00"},
            },
            generated_at=f"2026-08-14T0{i}:31:00+00:00",
        )
        for i in range(1, 4)
    ]
    series, stamps, ticks, _ = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]
    )
    assert len(stamps["cpu_pressure"]) == len(series["cpu_pressure"]) == 3
    # circe's stamps, not athena's -- athena never won the merge.
    assert [s.hour for s in stamps["cpu_pressure"]] == [1, 2, 3]


def test_frozen_winner_reads_no_write_not_producer_written():
    """The live false positive, as a regression.

    One source, one frozen write time, values drifting in the subnormal range
    where ratio inference cannot tell decay from a write. The timestamp path
    must say no_write_in_window.
    """
    rows = [
        _row_with_stamps(
            {"node:athena": {"cpu_pressure": v}},
            {"node:athena": {"cpu_pressure": "2026-08-14T00:00:00+00:00"}},
            generated_at=f"2026-08-14T06:{i:02d}:00+00:00",
        )
        for i, v in enumerate([3e-323, 2.5e-323, 3e-323, 2.5e-323, 3e-323, 2.5e-323,
                               3e-323, 2.5e-323, 3e-323, 2.5e-323, 3e-323, 2.5e-323])
    ]
    series, stamps, ticks, _ = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]
    )
    regime = field_channel_glossary_routes._regime_for(
        "cpu_pressure", series["cpu_pressure"], 1, stamps["cpu_pressure"], ticks["cpu_pressure"]
    )
    assert regime.refresh_evidence == "timestamp"
    assert regime.refresh_state == "no_write_in_window"

    # Without the stamps -- the shipped behaviour until now -- the same series
    # is misread as a live producer.
    inferred = field_channel_glossary_routes._regime_for(
        "cpu_pressure", series["cpu_pressure"], 1, None, None
    )
    assert inferred.refresh_evidence == "value_ratio"
    assert inferred.refresh_state == "producer_written"


def test_advancing_winner_stamp_reads_producer_written():
    rows = [
        _row_with_stamps(
            {"node:athena": {"cpu_pressure": 0.4}},
            {"node:athena": {"cpu_pressure": f"2026-08-14T06:{i:02d}:00+00:00"}},
            generated_at=f"2026-08-14T06:{i:02d}:30+00:00",
        )
        for i in range(12)
    ]
    series, stamps, ticks, _ = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]
    )
    regime = field_channel_glossary_routes._regime_for(
        "cpu_pressure", series["cpu_pressure"], 1, stamps["cpu_pressure"], ticks["cpu_pressure"]
    )
    assert regime.refresh_evidence == "timestamp"
    assert regime.refresh_state == "producer_written"


def test_missing_stamps_fall_back_rather_than_inventing_one():
    """A node with no node_vector_updated_at entry yields None, and the regime
    degrades to inference instead of claiming a write."""
    rows = [_field_state_row({"cpu_pressure": 0.3}) for _ in range(4)]
    series, stamps, ticks, _ = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]
    )
    assert all(s is None for s in stamps["cpu_pressure"])
    regime = field_channel_glossary_routes._regime_for(
        "cpu_pressure", series["cpu_pressure"], 1, stamps["cpu_pressure"], ticks["cpu_pressure"]
    )
    assert regime.refresh_evidence == "value_ratio"


def test_health_endpoint_reports_timestamp_evidence_end_to_end(client):
    rows = [
        _row_with_stamps(
            {"node:athena": {"cpu_pressure": 0.4}},
            {"node:athena": {"cpu_pressure": "2026-08-14T00:00:00+00:00"}},
            generated_at=f"2026-08-14T06:{i:02d}:00+00:00",
        )
        for i in range(12)
    ]
    fake_engine = _fake_engine_with_rows(rows)
    with patch.object(field_channel_glossary_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/field-channel-glossary/health?hours=1")
    by_channel = {c["channel"]: c for c in r.json()["channels"]}
    assert by_channel["cpu_pressure"]["regime"]["refresh_evidence"] == "timestamp"
    assert by_channel["cpu_pressure"]["regime"]["refresh_state"] == "no_write_in_window"


def test_sparse_in_window_stamp_is_producer_written_not_absence():
    """The BLOCKER this rule replaced, end to end.

    `execution_friction` live: 2 stamped samples of 437 (0.5% coverage), both
    at one in-window time. The previous advance-based rule needed two distinct
    increasing stamps, so it reported `no_write_in_window` AND flagged it
    authoritative -- a confidently wrong verdict, worse than the value_ratio
    blindness the timestamp path exists to replace.
    """
    rows = []
    for i in range(12):
        stamped = i == 9  # one lone in-window write, the rest unstamped
        rows.append(
            _row_with_stamps(
                {"node:athena": {"cpu_pressure": 0.3}},
                {"node:athena": {"cpu_pressure": "2026-08-14T06:09:00+00:00"}}
                if stamped
                else {},
                generated_at=f"2026-08-14T06:{i:02d}:00+00:00",
            )
        )
    series, stamps, ticks, _ = field_channel_glossary_routes.build_channel_series(
        rows, known_channels=["cpu_pressure"]
    )
    regime = field_channel_glossary_routes._regime_for(
        "cpu_pressure", series["cpu_pressure"], 1, stamps["cpu_pressure"], ticks["cpu_pressure"]
    )
    assert regime.refresh_evidence == "timestamp"
    assert regime.refresh_state == "producer_written"


def test_two_frozen_sources_at_different_times_is_not_a_write():
    """The mutant the first review found surviving: nothing distinguished the
    old `len(set(stamps)) > 1` rule from its replacement.

    Two nodes alternate as merge winner, each frozen at its OWN old write time.
    Distinct stamps, no write. Asserted in BOTH winner orderings, because the
    intermediate `advance` rule was order-dependent and passed only one of them.
    """
    for first_winner_is_newer in (True, False):
        rows = []
        for i in range(12):
            hi, lo = (0.9, 0.2) if (i % 2 == 0) == first_winner_is_newer else (0.2, 0.9)
            rows.append(
                _row_with_stamps(
                    {"node:athena": {"cpu_pressure": hi}, "node:circe": {"cpu_pressure": lo}},
                    {
                        "node:athena": {"cpu_pressure": "2026-08-13T01:00:00+00:00"},
                        "node:circe": {"cpu_pressure": "2026-08-13T02:00:00+00:00"},
                    },
                    generated_at=f"2026-08-14T06:{i:02d}:00+00:00",
                )
            )
        series, stamps, ticks, _ = field_channel_glossary_routes.build_channel_series(
            rows, known_channels=["cpu_pressure"]
        )
        assert len(set(s for s in stamps["cpu_pressure"] if s)) == 2, "must be distinct"
        regime = field_channel_glossary_routes._regime_for(
            "cpu_pressure", series["cpu_pressure"], 1,
            stamps["cpu_pressure"], ticks["cpu_pressure"],
        )
        assert regime.refresh_evidence == "timestamp"
        assert regime.refresh_state == "no_write_in_window", first_winner_is_newer
