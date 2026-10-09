"""The evidence bundle an urgent curiosity run starts with.

Hardware and pool readings only. Every section is bounded: a failing or slow
reader becomes `{"error": ...}` and the rest of the bundle still arrives, and
the whole bundle fits the seed's byte cap.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.curiosity_urgent import URGENT_EVIDENCE_MAX_BYTES, CuriosityUrgentSeedV1
from scripts import urgent_evidence as ue

NOW = datetime(2026, 9, 28, 20, 0, 0, tzinfo=timezone.utc)


def _cooling_row() -> dict:
    return {
        "ts": NOW - timedelta(seconds=5),
        "cooling_watts": 410.0,
        "cooling_volts": 120.0,
        "switch_on": True,
        "controller_ready": True,
        "device_online": True,
        "payload_json": {"state": {"stale": False}, "provenance": {"sample_age_sec": 3.0}},
    }


def _snapshot(node: str) -> dict:
    return {
        "nodes": {
            node: {
                "as_of": NOW.isoformat(),
                "freshness_s": 2.0,
                "status": "ok",
                "summary": {
                    "measurements": {"temp_c_max": 71.0, "fan_pct_max": 60.0, "chassis_watts": 640.0},
                    "peak_pressure": 0.8,
                    "peak_pressure_channel": "power",
                    "constraint": "power",
                    "pressures": {"power": 0.8},
                },
            }
        }
    }


def _raw_recent(node: str) -> dict:
    return {
        "items": [
            {
                "timestamp": NOW.isoformat(),
                "raw": {
                    "gpu": {
                        "gpus": [
                            {
                                "index": 0,
                                "name": "V100",
                                "utilization_gpu": 97,
                                "memory_used_mb": 30000,
                                "memory_total_mb": 32000,
                                "power_draw_watts": 240.0,
                                "processes": [{"pid": 1, "name": "llama-server"}],
                            }
                        ]
                    }
                },
            }
        ]
    }


class _Feed:
    def __init__(self, state):
        self.state = state

    def snapshot(self):
        return {"version": 7, "state": self.state, "events": [{"event": "granted"}] * 3}


_POOL_STATE = {
    "generated_at": NOW.isoformat(),
    "mode": "enforce",
    "config_digest": "abc",
    "cards": [{"card": "athena:0", "vram_gb": 32.0, "swap_state": "idle", "swapped_in": ["agent"], "lent": False}],
    "roles": [{"role": "agent", "kind": "llm", "cards": ["athena:0"], "url": "http://x", "status": "confirmed"}],
    "leases": [
        {"lease_id": "l1", "request_id": "r1", "holder": "curiosity:abc", "work_class": "curiosity",
         "priority": "background", "kind": "hold", "status": "granted", "role": "agent",
         "created_at": NOW.isoformat(), "granted_at": NOW.isoformat()},
        {"lease_id": "l0", "request_id": "r0", "holder": "old", "work_class": "chat",
         "priority": "interactive", "kind": "request", "status": "released", "role": "agent",
         "created_at": NOW.isoformat()},
    ],
    "queue_depth": {"agent": 1},
    "backlog_depth": {},
}


@pytest.fixture
def readers(monkeypatch):
    """Every reader answers. Tests override one at a time."""
    monkeypatch.setattr(ue.settings, "CABINET_AMBIENT_HISTORY_NODE", "athena")
    monkeypatch.setattr(ue.settings, "CABINET_SENSORS_STALE_AFTER_SEC", 30.0)
    monkeypatch.setattr(ue, "_now_utc", lambda: NOW)

    async def latest(*, node):
        assert node == "athena"
        return _cooling_row()

    async def history(*, node, hours):
        assert node == "athena" and hours == 1
        return [
            {"t": NOW - timedelta(minutes=90), "temp_c": 30.0},  # outside the 60 min window
            {"t": NOW - timedelta(minutes=30), "temp_c": 33.25},
            {"t": NOW - timedelta(minutes=1), "temp_c": None},
            {"t": NOW - timedelta(seconds=10), "temp_c": 35.5},
        ]

    async def snapshot(node):
        return _snapshot(node)

    async def raw_recent(node, *, limit=10):
        return _raw_recent(node)

    monkeypatch.setattr(ue.cabinet_cooling_routes, "_latest_query", latest)
    monkeypatch.setattr(ue.cabinet_sensors_routes, "_history_query", history)
    monkeypatch.setattr(ue.biometrics_node_client, "fetch_snapshot", snapshot)
    monkeypatch.setattr(ue.biometrics_node_client, "fetch_raw_recent", raw_recent)
    monkeypatch.setattr(ue.gpu_pool_routes, "feed", _Feed(_POOL_STATE))
    return monkeypatch


def test_every_section_has_its_success_shape(readers) -> None:
    bundle = asyncio.run(ue.collect_evidence())

    assert set(bundle) == {"cooling", "cabinet_trend", "hosts", "gpus", "pool", "collected_at"}
    assert bundle["collected_at"] == NOW.isoformat()

    cooling = bundle["cooling"]
    assert cooling["node"] == "athena" and cooling["ok"] is True
    assert cooling["sample"]["cooling_watts"] == 410.0 and cooling["sample"]["switch_on"] is True
    assert cooling["sample_age_sec"] == 3.0

    trend = bundle["cabinet_trend"]
    assert trend["node"] == "athena" and trend["window_min"] == 60
    assert [p["temp_c"] for p in trend["points"]] == [33.25, 35.5]

    athena = bundle["hosts"]["athena"]
    assert athena["measurements"]["temp_c_max"] == 71.0
    assert athena["measurements"]["fan_pct_max"] == 60.0
    assert athena["measurements"]["chassis_watts"] == 640.0
    assert athena["status"] == "ok" and athena["peak_pressure_channel"] == "power"
    assert set(bundle["hosts"]) == {"athena", "circe"}

    card = bundle["gpus"]["athena"][0]
    assert card["utilization_gpu"] == 97 and card["power_draw_watts"] == 240.0
    assert card["memory_used_mb"] == 30000 and card["memory_total_mb"] == 32000
    assert "trend" not in card

    pool = bundle["pool"]
    assert pool["mode"] == "enforce" and pool["queue_depth"] == {"agent": 1}
    assert [lease["lease_id"] for lease in pool["leases"]] == ["l1"]  # released lease is not active
    assert "events" not in pool

    json.dumps(bundle)  # JSON-native throughout


def test_evidence_is_hardware_and_pool_only(readers) -> None:
    """No chat, memory, or journal keys anywhere in the bundle."""
    blob = json.dumps(asyncio.run(ue.collect_evidence())).lower()
    for forbidden in ("journal", "memory_crystall", "chat_history", "user_message", "messages"):
        assert forbidden not in blob


def test_one_failing_section_becomes_an_error_and_the_rest_arrive(readers) -> None:
    async def boom(*, node):
        raise RuntimeError("DATABASE_URL is not configured")

    readers.setattr(ue.cabinet_cooling_routes, "_latest_query", boom)
    bundle = asyncio.run(ue.collect_evidence())

    assert bundle["cooling"] == {"error": "RuntimeError: DATABASE_URL is not configured"}
    assert bundle["cabinet_trend"]["points"]
    assert bundle["hosts"]["athena"]["measurements"]
    assert bundle["pool"]["leases"]


def test_one_unreachable_node_does_not_blank_the_other(readers) -> None:
    async def snapshot(node):
        if node == "circe":
            raise ue.biometrics_node_client.BiometricsNodeClientError("circe unreachable")
        return _snapshot(node)

    readers.setattr(ue.biometrics_node_client, "fetch_snapshot", snapshot)
    bundle = asyncio.run(ue.collect_evidence())
    assert bundle["hosts"]["circe"] == {"error": "BiometricsNodeClientError: circe unreachable"}
    assert bundle["hosts"]["athena"]["measurements"]["temp_c_max"] == 71.0


def test_a_slow_section_times_out_instead_of_holding_the_bundle(readers) -> None:
    async def slow(*, node, hours):
        await asyncio.sleep(5)
        return []

    readers.setattr(ue.cabinet_sensors_routes, "_history_query", slow)
    bundle = asyncio.run(ue.collect_evidence(per_section_timeout=0.05))
    assert bundle["cabinet_trend"]["error"].startswith("TimeoutError")
    assert bundle["cooling"]["ok"] is True


def test_no_pool_state_yet_is_said_not_invented(readers) -> None:
    readers.setattr(ue.gpu_pool_routes, "feed", _Feed(None))
    bundle = asyncio.run(ue.collect_evidence())
    assert "error" in bundle["pool"]


def test_an_oversized_bundle_is_trimmed_under_the_seed_cap(readers) -> None:
    async def huge(*, node, hours):
        start = NOW - timedelta(minutes=59)
        return [{"t": start + timedelta(seconds=i * 0.5), "temp_c": 30.0 + i / 1000} for i in range(7000)]

    readers.setattr(ue.cabinet_sensors_routes, "_history_query", huge)
    bundle = asyncio.run(ue.collect_evidence())

    trend = bundle["cabinet_trend"]
    assert trend["trimmed_points"] > 0
    assert trend["points"]  # trimmed, not emptied
    # Oldest dropped first: the newest reading survives.
    assert trend["points"][-1]["temp_c"] == pytest.approx(30.0 + 6999 / 1000)
    assert bundle["cooling"]["ok"] is True and bundle["hosts"]["athena"]["measurements"]

    seed = CuriosityUrgentSeedV1(
        incident_id="ab" * 16, question="Is athena overheating?", trigger="manual",
        evidence=bundle, requested_at=NOW,
    )
    assert len(json.dumps(seed.evidence, separators=(",", ":")).encode()) <= URGENT_EVIDENCE_MAX_BYTES


def test_trim_drops_whole_sections_when_the_trend_is_not_enough() -> None:
    bundle = {
        "cooling": {"ok": True},
        "cabinet_trend": {"points": []},
        "hosts": {"athena": {"blob": "x" * 40_000}},
        "gpus": {},
        "pool": {},
        "collected_at": NOW.isoformat(),
    }
    trimmed = ue.trim_to_cap(bundle)
    assert ue.encoded_size(trimmed) <= URGENT_EVIDENCE_MAX_BYTES
    assert trimmed["hosts"]["error"].startswith("trimmed")
    assert trimmed["cooling"] == {"ok": True}


def test_gpu_cards_carry_pool_derived_lanes(readers) -> None:
    """Stage 5.5: the urgent bundle labels a circe card from the pool's own state, the same label
    Juniper sees in the biometrics modal; athena (no pool) stays unassigned."""
    state = {**_POOL_STATE, "host": "circe",
             "cards": [{"card": "gpu0", "index": 0, "vram_gb": 32.0, "swap_state": "idle",
                        "swapped_in": [], "lent": False}],
             "roles": [{"role": "chat", "kind": "llm", "cards": ["gpu0"], "url": "http://x",
                        "status": "confirmed"}]}
    readers.setattr(ue.gpu_pool_routes, "feed", _Feed(state))

    bundle = asyncio.run(ue.collect_evidence())

    assert bundle["gpus"]["circe"][0]["lane"] == "chat"
    assert bundle["gpus"]["circe"][0]["lane_assigned"] is True
    assert bundle["gpus"]["athena"][0]["lane"] == "unassigned"
