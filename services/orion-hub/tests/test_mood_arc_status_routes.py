from __future__ import annotations

import sys
from pathlib import Path

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

from scripts import mood_arc_status_routes  # noqa: E402
from scripts.field_digester_client import FieldDigesterClientError  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(mood_arc_status_routes.router)
    return TestClient(app)


# --------------------------------------------------------------- /live


def test_live_relays_field_channel_anomaly_block(monkeypatch) -> None:
    async def _fake_fetch_health():
        return {
            "status": "ok",
            "service": "orion-field-digester",
            "field_channel_anomaly": {
                "enabled": True,
                "encoder_version": "v4",
                "last_live_enrichment_fields": ["action_warrant"],
            },
        }

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["encoder_version"] == "v4"
    assert body["last_live_enrichment_fields"] == ["action_warrant"]


def test_live_reports_unreachable_without_a_500(monkeypatch) -> None:
    """Absence must read as absence -- never a 500, never a silently-empty
    success (CLAUDE.md's 'no empty-shell cognition')."""

    async def _fake_fetch_health():
        raise FieldDigesterClientError("http://x/health unreachable")

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is False
    assert "error" in body


def test_live_handles_a_null_field_channel_anomaly_block(monkeypatch) -> None:
    """Review finding (2026-09-04): `.get(key, default)` only substitutes on
    a MISSING key, not a present-but-null one -- `**None` raises TypeError."""

    async def _fake_fetch_health():
        return {"status": "ok", "service": "orion-field-digester", "field_channel_anomaly": None}

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["enabled"] is False


def test_live_handles_scorer_disabled(monkeypatch) -> None:
    async def _fake_fetch_health():
        return {"status": "ok", "service": "orion-field-digester", "field_channel_anomaly": {"enabled": False}}

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    body = resp.json()
    assert body["reachable"] is True
    assert body["enabled"] is False


# --------------------------------------------------------- /phi-v2-inventory


def test_phi_v2_inventory_reports_both_legacy_signals() -> None:
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    assert resp.status_code == 200
    body = resp.json()
    ids = {s["signal_id"] for s in body["legacy_signals"]}
    assert ids == {"phi_heuristic.valence", "phi_intrinsic_reward.v1"}


def test_phi_v2_inventory_confirms_producer_is_dead() -> None:
    """orion-spark-introspector was deleted 2026-07-28 -- this must be a
    live filesystem check, not trust the registry's composition_status
    (which this repo's own convention deliberately does NOT use to encode
    liveness -- see orion/inner_state_registry.md)."""
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    body = resp.json()
    for signal in body["legacy_signals"]:
        assert signal["found_in_registry"] is True
        assert signal["producer_service"] == "orion-spark-introspector"
        assert signal["producer_service_exists"] is False
        assert signal["last_note"]  # non-empty, not fabricated


def test_phi_v2_inventory_names_the_design_doc() -> None:
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    body = resp.json()["design_doc"]
    assert body["exists"] is True
    assert body["path"] == "docs/superpowers/specs/2026-08-21-phi-v2-design.md"
    assert body["status"] is not None
    assert "not implemented" in body["status"].lower()


def test_phi_v2_inventory_checks_manual_cli_exists() -> None:
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    assert resp.json()["manual_cli_exists"] is True


# --------------------------------------------------------- _first_sentences


@pytest.mark.parametrize(
    "text,count,expected",
    [
        ("One. Two. Three.", 2, "One. Two."),
        ("Just one sentence with no trailing period", 2, "Just one sentence with no trailing period"),
        ("Wrapped\n    across  lines. Second sentence.", 1, "Wrapped across lines."),
    ],
)
def test_first_sentences(text: str, count: int, expected: str) -> None:
    assert mood_arc_status_routes._first_sentences(text, count) == expected
