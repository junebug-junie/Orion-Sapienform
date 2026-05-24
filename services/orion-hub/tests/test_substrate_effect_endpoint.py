from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.substrate_effect_cache import (
    SubstrateEffectSnapshot,
    substrate_effect_cache,
)
from scripts.substrate_effect_pipeline import run_substrate_effect_pipeline


@pytest.fixture()
def client():
    from scripts.main import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_cache():
    substrate_effect_cache.clear()
    yield


def test_unknown_turn_returns_404(client):
    response = client.get("/api/chat/turn/does-not-exist/substrate-effect")
    assert response.status_code == 404


def test_high_pressure_turn_returns_view_with_repair_concrete(client):
    summary, _ = run_substrate_effect_pipeline(
        turn_id="turn-high",
        message_id=None,
        user_text=(
            "you gave me garbage directions, stop, build me a design spec for claude, "
            "arsonist pov only, nuts and bolts"
        ),
        source_id="conv-high",
        contract_before={"mode": "default"},
    )
    assert summary is not None  # sanity: pipeline did run
    response = client.get("/api/chat/turn/turn-high/substrate-effect")
    assert response.status_code == 200
    body = response.json()
    assert body["turn_id"] == "turn-high"
    assert body["outcome"]["appraisal_kind"] == "repair_pressure"
    assert body["outcome"]["level_label"] in {"HIGH", "MEDIUM"}
    assert body["behavior_delta"]["changed"] in (True, False)
    assert isinstance(body["outcome"]["summary"], str) and body["outcome"]["summary"]
    assert isinstance(body["causal_chain"], list)
    assert isinstance(body["evidence_cards"], list)


def test_known_turn_with_no_effect_returns_valid_empty_view(client):
    snap = SubstrateEffectSnapshot(
        turn_id="turn-empty",
        message_id=None,
        user_text="",
        appraisal=None,
        signal=None,
        evidence=[],
        contract_before={"mode": "default"},
        contract_after={"mode": "default"},
        causal_molecule_ids=[],
    )
    substrate_effect_cache.store(snap)
    response = client.get("/api/chat/turn/turn-empty/substrate-effect")
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"]["appraisal_kind"] == "none"
    assert body["outcome"]["level"] == 0.0
    assert body["evidence_cards"] == []
    assert body["causal_chain"] == []
    assert "No substrate effect" in body["outcome"]["summary"]


def test_recent_effects_endpoint_returns_newest_first(client):
    for text, turn in [
        ("first benign", "t1"),
        ("you gave me garbage directions, stop, build me a design spec for claude", "t2"),
        ("third benign", "t3"),
    ]:
        run_substrate_effect_pipeline(
            turn_id=turn,
            message_id=None,
            user_text=text,
            source_id=f"conv-{turn}",
            contract_before={"mode": "default"},
        )
    response = client.get("/api/substrate-effect/recent?limit=10")
    assert response.status_code == 200
    rows = response.json().get("rows")
    assert isinstance(rows, list)
    assert [r["turn_id"] for r in rows][:3] == ["t3", "t2", "t1"]
    row_t2 = next(r for r in rows if r["turn_id"] == "t2")
    assert row_t2["chip_label"]
    assert row_t2["level_label"] in {"HIGH", "MEDIUM", "LOW", "NONE"}
