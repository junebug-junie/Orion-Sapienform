"""Column shape and payload normalization for biometrics.cluster.v1 -> orion_biometrics_cluster.

`_write_row` filters payload keys against the mapper's columns, so a field with no
column is dropped SILENTLY rather than raising -- the same failure
test_biometrics_summary_sql_shape.py was written for. These tests assert against the
real mapper and the real normalizer, never against source text.

The stakes here are specific: this table is the ONLY store of the fleet power total.
Per-node rows cannot substitute, because circe reaches the PDU only through a proxied
reading athena takes on its behalf, which is deliberately never written into circe's
own row.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import Float, inspect as sa_inspect

from app.models.biometrics_cluster import BiometricsClusterSQL
from app.worker import _normalize_biometrics_cluster_payload

from orion.schemas.telemetry.biometrics import BiometricsClusterV1


def _columns() -> dict:
    return {c.key: c for c in sa_inspect(BiometricsClusterSQL).columns}


def test_every_payload_field_survives_to_a_column_or_is_deliberately_mapped() -> None:
    """No silent drops. `timestamp` is the one renamed field (-> observed_at); every
    other payload field must land somewhere or _write_row discards it without error."""
    cols = _columns()
    renamed = {"timestamp": "observed_at"}
    for field in BiometricsClusterV1.model_fields:
        target = renamed.get(field, field)
        assert target in cols, f"payload field {field!r} has no column ({target!r})"


def test_the_two_clocks_are_separate_columns() -> None:
    """observed_at (when the fleet was measured) and created_at (when the row landed)
    are different clocks. A backlog replay writes old observations at a new wall time,
    so settlement MUST range over observed_at and retention MUST age by created_at.
    Collapsing them would silently corrupt both."""
    cols = _columns()
    assert "observed_at" in cols and "created_at" in cols
    # (an `is not` comparison here would be trivially true -- two Column() calls are
    # always distinct objects -- so the real assertions are the type checks below)
    for name in ("observed_at", "created_at"):
        assert cols[name].type.timezone is True, f"{name} must be timezone-aware"


def test_watts_are_real_columns_not_only_jsonb() -> None:
    """Settlement is a time-range aggregate over watts. JSONB extraction inside a range
    predicate cannot use a btree the way a float column can."""
    cols = _columns()
    for name in ("pdu_watts", "chassis_watts", "gpu_watts_total", "gpu_count"):
        assert name in cols, f"{name} should be hoisted out of measurements"
        assert cols[name].nullable, f"{name} must be nullable -- absent means UNMEASURED"
        # Type matters: a String column would pass a presence check while defeating the
        # entire reason these are hoisted (a btree range scan over watts).
        assert isinstance(cols[name].type, Float), f"{name} must be Float, got {cols[name].type}"


def test_provenance_and_honesty_columns_are_persisted() -> None:
    """Reading a fleet total without these is reading a partial sum as a complete one."""
    cols = _columns()
    for name in ("measurements_missing", "measurements_proxied", "nodes_absent"):
        assert name in cols


def test_normalizer_maps_timestamp_to_observed_at_and_hoists_watts() -> None:
    observed = datetime(2026, 8, 28, 4, 30, tzinfo=timezone.utc)
    out = _normalize_biometrics_cluster_payload(
        {
            "timestamp": observed,
            "measurements": {
                "pdu_watts": 1526.0,
                "chassis_watts": 1457.0,
                "gpu_watts_total": 448.39,
                "gpu_count": 8.0,
                "temp_c_max": 61.0,
            },
            "measurements_proxied": {"circe": ["chassis_watts", "pdu_watts"]},
        }
    )
    assert out["observed_at"] == observed
    assert "timestamp" not in out
    assert out["pdu_watts"] == 1526.0
    assert out["chassis_watts"] == 1457.0
    assert out["gpu_watts_total"] == pytest.approx(448.39)
    assert out["gpu_count"] == 8.0
    # measurements is still stored whole -- it carries keys the columns do not.
    assert out["measurements"]["temp_c_max"] == 61.0
    assert out["measurements_proxied"] == {"circe": ["chassis_watts", "pdu_watts"]}


def test_an_unmeasured_key_stays_none_and_never_becomes_zero() -> None:
    """A key absent from measurements means UNMEASURED. Defaulting it to 0.0 would
    invent a reading no instrument produced, and every downstream sum would treat an
    unseen machine as a machine drawing nothing -- the exact failure
    measurements_missing exists to prevent."""
    out = _normalize_biometrics_cluster_payload(
        {"timestamp": datetime.now(timezone.utc), "measurements": {"chassis_watts": 500.0}}
    )
    assert out["chassis_watts"] == 500.0
    assert out.get("pdu_watts") is None
    assert out.get("gpu_watts_total") is None


def test_normalizer_tolerates_a_payload_with_no_measurements() -> None:
    out = _normalize_biometrics_cluster_payload({"timestamp": datetime.now(timezone.utc)})
    assert "observed_at" in out
    assert out.get("pdu_watts") is None


def test_the_normalizer_rejects_a_bool_on_the_unvalidated_path() -> None:
    """Scoped honestly: this covers a DIRECT call, not the production path.

    bool is a subclass of int, so `isinstance(True, (int, float))` is True and an
    unguarded hoist would write 1.0 W. The guard below stops that for any caller that
    reaches the normalizer without pydantic in front of it."""
    out = _normalize_biometrics_cluster_payload(
        {"timestamp": datetime.now(timezone.utc), "measurements": {"pdu_watts": True}}
    )
    assert out.get("pdu_watts") is None


def test_KNOWN_GAP_a_bool_is_coerced_to_1_watt_before_the_normalizer_runs() -> None:
    """Records a real gap rather than implying it is closed.

    An earlier version of the test above claimed the normalizer protects production
    from a bool in a watts field. It does not. `_write` validates through
    BiometricsClusterV1 BEFORE `_write_row` runs, and the schema declares
    `measurements: Optional[Dict[str, float]]`, so pydantic coerces True -> 1.0 and the
    normalizer's guard never sees a bool on the real path. The old test passed only
    because it bypassed validation -- a detector shaped by the investigation that built
    it rather than by the production path.

    Fixing this properly means a field_validator on BiometricsClusterV1 rejecting bool,
    which tightens a live channel contract and belongs in its own patch. Until then this
    test pins the ACTUAL behaviour so nobody re-reads the guard as protection it is not
    providing."""
    validated = BiometricsClusterV1.model_validate(
        {"measurements": {"pdu_watts": True, "chassis_watts": 3}}
    )
    assert validated.measurements == {"pdu_watts": 1.0, "chassis_watts": 3.0}

    out = _normalize_biometrics_cluster_payload(validated.model_dump())
    assert out["pdu_watts"] == 1.0  # a fabricated reading, and currently unavoidable here


def test_the_kind_routes_to_this_model() -> None:
    """Asserts on the TABLE the route resolves to, rather than on object identity.

    The tablename IS the contract here; class identity is incidental to it. Pointing the
    route at BiometricsSummarySQL still fails this test, which is the regression that
    matters.

    On the double-import question: review could not reproduce an identity mismatch for
    MODEL_MAP across three randomized full-suite runs, and I do not claim one here. But
    the effect is real elsewhere and is demonstrated in
    test_write_row_actually_invokes_the_normalizer -- passing that test's own imported
    class into `_write_row` makes `sql_model_cls is BiometricsClusterSQL` evaluate False
    under the full suite, skipping the normalizer entirely. So identity comparisons
    across module boundaries are genuinely fragile in this suite; the earlier version of
    this docstring simply pinned the claim to the wrong symbol."""
    from app.settings import Settings  # noqa: F401
    from app.worker import MODEL_MAP

    model_cls, schema_cls = MODEL_MAP["BiometricsClusterSQL"]
    assert model_cls.__tablename__ == BiometricsClusterSQL.__tablename__ == "orion_biometrics_cluster"
    assert schema_cls.__name__ == BiometricsClusterV1.__name__ == "BiometricsClusterV1"

    assert Settings().route_map.get("biometrics.cluster.v1") == "BiometricsClusterSQL"


def test_the_table_is_retention_managed() -> None:
    """Bounded from the first commit that creates it. Every unbounded table in this
    service became a problem later rather than never."""
    from app.grammar_truth import GRAMMAR_RETENTION_TABLES

    assert "orion_biometrics_cluster" in {t for t, _ in GRAMMAR_RETENTION_TABLES}


def test_the_channel_is_actually_subscribed_from_the_real_env() -> None:
    """THE BLOCKER THIS FILE ORIGINALLY MISSED ENTIRELY.

    `SQL_WRITER_SUBSCRIBE_CHANNELS` REPLACES the Python default wholesale, unlike
    `route_map`, which merges over DEFAULT_ROUTE_MAP. Adding the channel to
    settings.py's default alone therefore produced: a correct route, a correct model,
    working retention, a created table, ten green tests -- and no subscription. The
    writer would never have received a single biometrics.cluster.v1 envelope, and
    nothing in this file would have said so.

    Asserting on `route_map` gives false comfort precisely because route_map merges and
    this field does not. So this asserts the effective channel list, built from the
    checked-in .env_example rather than from the Python default."""
    import json
    from pathlib import Path

    from app.settings import Settings

    example = Path(__file__).resolve().parents[1] / ".env_example"
    raw = None
    for line in example.read_text().splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            raw = line.split("=", 1)[1].strip()
            break
    assert raw is not None, "SQL_WRITER_SUBSCRIBE_CHANNELS missing from .env_example"
    assert "orion:biometrics:cluster" in json.loads(raw), (
        "the operator contract must list the channel; the code default does not "
        "survive an env override of this field"
    )

    # And belt-and-braces: even a stale operator list cannot drop it, same guarantee
    # already applied to orion:autonomy:action:outcome for this exact failure.
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert "orion:biometrics:cluster" in stale.effective_subscribe_channels


def test_write_row_actually_invokes_the_normalizer(monkeypatch) -> None:
    """THE SECOND BLOCKER: every other test here calls the normalizer DIRECTLY.

    Mutation-checked in review: deleting the two lines in `_write_row` that dispatch to
    `_normalize_biometrics_cluster_payload` left all ten tests green, while in
    production `timestamp` would never be renamed, `observed_at` would stay NULL, and
    every insert would die on NOT NULL. This drives the real `_write_row` so that hook
    is covered."""
    from app import worker

    merged = {}

    class _FakeSession:
        def merge(self, obj):
            merged["obj"] = obj
            return obj

        def add(self, obj):
            merged["obj"] = obj

        def commit(self):
            merged["committed"] = True

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(worker, "get_session", lambda: _FakeSession())

    payload = BiometricsClusterV1.model_validate(
        {
            "timestamp": datetime(2026, 8, 28, 4, 30, tzinfo=timezone.utc),
            "measurements": {"pdu_watts": 1526.0, "chassis_watts": 1457.0},
            "measurements_proxied": {"circe": ["chassis_watts", "pdu_watts"]},
        }
    ).model_dump()

    # Pass WORKER's own class object, not this module's import of it. Under the full
    # suite these are two distinct objects with the same qualified name -- confirmed
    # here, where passing the local import made `sql_model_cls is BiometricsClusterSQL`
    # inside _write_row evaluate False, the normalizer never ran, and observed_at came
    # back None. Production has a single import path so the dispatch holds there, but a
    # test that supplies its own class is not exercising the production comparison.
    assert worker._write_row(worker.BiometricsClusterSQL, payload) is True

    row = merged["obj"]
    assert row.observed_at == datetime(2026, 8, 28, 4, 30, tzinfo=timezone.utc)
    assert row.pdu_watts == 1526.0
    assert row.chassis_watts == 1457.0
    assert row.measurements_proxied == {"circe": ["chassis_watts", "pdu_watts"]}
