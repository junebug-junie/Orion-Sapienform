"""FastAPI HTTP surface tests -- review gap: only the underlying
svc.stats()/svc.latest_h1_dict() methods were tested directly, never
main.py's actual HTTP wiring (/health, /h1)."""
from __future__ import annotations

import os

os.environ.setdefault("ORION_BUS_ENABLED", "false")

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_ok_and_stats() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "orion-heartbeat"
    assert "events_absorbed" in body
    assert "allowlisted_organs" in body


def test_h1_endpoint_before_first_computation() -> None:
    with TestClient(app) as client:
        resp = client.get("/h1")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ok": False, "reason": "no_h1_computed_yet"}


def test_health_endpoint_exposes_live_tuning_and_band_edges() -> None:
    """A read-only consumer must be able to explain the mechanism from the
    RUNNING service, not from a mirrored copy of .env_example that may not
    match this container -- the same config-drift failure that silently
    reverted two live flags on 2026-07-29."""
    from app.substrate.reconstruction import verdict_thresholds

    with TestClient(app) as client:
        body = client.get("/health").json()

    config = body["config"]
    for key in (
        "n_trajectories",
        "gamma",
        "base_decay_prob",
        "decay_spread_sensitivity",
        "decay_interval_sec",
        "h1_interval_sec",
    ):
        assert key in config, key
    assert config["high_ratio"] == verdict_thresholds()["high_ratio"]
    assert config["low_ratio"] == verdict_thresholds()["low_ratio"]
    assert config["std_mixed"] == verdict_thresholds()["std_mixed"]
    assert config["std_redundant_max"] == verdict_thresholds()["std_redundant_max"]
    assert config["bulk_low"] == verdict_thresholds()["bulk_low"]
    assert config["bulk_redundant_min"] == verdict_thresholds()["bulk_redundant_min"]
    assert body["absorb_queue_maxsize"] >= 1
    assert set(body["organ_site_map"]) == set(body["allowlisted_organs"])
    for retired in ("reheat_strength", "reheat_prob_scale", "decay_reheat_interval_sec"):
        assert retired not in config, retired


def test_health_has_no_reheat_surface_and_names_the_input_rules() -> None:
    """#3838: the bus_synaptic reheat was removed outright -- no stale
    `last_reheat` block may linger on /health. The input-rule counters and
    the reasons for every unrouted producer must be visible instead."""
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert "last_reheat" not in body
    assert body["seat_order"] == [
        "orion-hub",
        "orion-biometrics",
        "orion-cortex-exec",
        "orion-bus",
        "orion-harness-governor",
    ]
    for key in ("events_skipped_input", "events_skipped_input_by_reason", "inputs_absorbed_by_kind", "inputs_unlisted_role"):
        assert key in body, key
    assert body["unrouted_producer_reasons"]["orion-sql-writer"]
