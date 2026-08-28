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
from sqlalchemy import inspect as sa_inspect

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
    assert cols["observed_at"] is not cols["created_at"]
    for name in ("observed_at", "created_at"):
        assert cols[name].type.timezone is True, f"{name} must be timezone-aware"


def test_watts_are_real_columns_not_only_jsonb() -> None:
    """Settlement is a time-range aggregate over watts. JSONB extraction inside a range
    predicate cannot use a btree the way a float column can."""
    cols = _columns()
    for name in ("pdu_watts", "chassis_watts", "gpu_watts_total", "gpu_count"):
        assert name in cols, f"{name} should be hoisted out of measurements"
        assert cols[name].nullable, f"{name} must be nullable -- absent means UNMEASURED"


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


def test_a_bool_is_not_accepted_as_watts() -> None:
    """bool is a subclass of int in Python; True would silently become 1.0 W."""
    out = _normalize_biometrics_cluster_payload(
        {"timestamp": datetime.now(timezone.utc), "measurements": {"pdu_watts": True}}
    )
    assert out.get("pdu_watts") is None


def test_the_kind_routes_to_this_model() -> None:
    """Asserts on the TABLE the route resolves to, not object identity.

    `is` fails here under the full suite while passing in isolation: the model module
    gets imported under two module identities, so MODEL_MAP holds a class with the same
    name and a different id. That is a property of this service's test import setup, not
    of the routing -- and identity would make this test a false alarm that a future
    reader "fixes" by weakening it. What actually matters is that biometrics.cluster.v1
    lands in orion_biometrics_cluster and validates against BiometricsClusterV1."""
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
