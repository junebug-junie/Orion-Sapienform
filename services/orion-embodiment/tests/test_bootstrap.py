from __future__ import annotations

import io
import json

import scripts.bootstrap_orion_agent as boot
from scripts.bootstrap_orion_agent import (
    _build_persona,
    _fetch_self_model,
    _load_town_card_blurb,
    upsert_env_keys,
)


def test_upsert_env_keys_replaces_and_adds(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AITOWN_WORLD_ID=w1\nAITOWN_ORION_PLAYER_ID=old\nSECRET=keepme\n", encoding="utf-8")
    upsert_env_keys(env, {"AITOWN_ORION_PLAYER_ID": "p9", "AITOWN_ORION_AGENT_ID": "a9"})
    text = env.read_text(encoding="utf-8")
    assert "AITOWN_ORION_PLAYER_ID=p9" in text
    assert "AITOWN_ORION_AGENT_ID=a9" in text
    assert "SECRET=keepme" in text
    assert "AITOWN_WORLD_ID=w1" in text
    assert "old" not in text


def test_upsert_env_keys_is_idempotent(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AITOWN_ORION_PLAYER_ID=p9\n", encoding="utf-8")
    upsert_env_keys(env, {"AITOWN_ORION_PLAYER_ID": "p9"})
    assert env.read_text(encoding="utf-8").count("AITOWN_ORION_PLAYER_ID=p9") == 1


def test_load_town_card_blurb_reads_override(tmp_path, monkeypatch):
    card = tmp_path / "orion_town_card.txt"
    card.write_text("Orion is a synthetic mind\nliving across the mesh.\n", encoding="utf-8")
    monkeypatch.setenv("EMBODIMENT_TOWN_CARDS_PATH", str(card))
    blurb = _load_town_card_blurb()
    assert blurb == "Orion is a synthetic mind living across the mesh."


def test_load_town_card_blurb_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBODIMENT_TOWN_CARDS_PATH", str(tmp_path / "nope.txt"))
    assert _load_town_card_blurb() is None


def test_build_persona_prefers_card_over_self_model(tmp_path, monkeypatch):
    card = tmp_path / "orion_town_card.txt"
    card.write_text("Orion is a distributed digital mind across the mesh.", encoding="utf-8")
    monkeypatch.setenv("EMBODIMENT_TOWN_CARDS_PATH", str(card))
    # self_state_url is ignored when a card is present (no network call).
    persona = _build_persona("http://unused:9999", "f1")
    assert persona.persona_source == "card"
    assert persona.identity_blurb == "Orion is a distributed digital mind across the mesh."
    assert persona.spritesheet == "f1"


def test_build_persona_falls_back_when_no_card(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBODIMENT_TOWN_CARDS_PATH", str(tmp_path / "absent.txt"))
    # No self-state URL → minimal fallback persona (never empty-shell).
    persona = _build_persona("", "f2")
    assert persona.persona_source == "fallback"
    assert persona.identity_blurb
    assert persona.spritesheet == "f2"


def test_build_persona_card_provenance_has_content_hash(tmp_path, monkeypatch):
    card = tmp_path / "orion_town_card.txt"
    card.write_text("Orion is a distributed digital mind.", encoding="utf-8")
    monkeypatch.setenv("EMBODIMENT_TOWN_CARDS_PATH", str(card))
    persona = _build_persona("http://unused:9999", "f1")
    assert persona.provenance["source"] == "town_cards.yaml"
    assert len(persona.provenance["content_sha256"]) == 12
    assert persona.provenance["chars"] == len("Orion is a distributed digital mind.")


def test_fetch_self_model_requests_latest_endpoint(monkeypatch):
    """The runtime exposes GET /latest — guard against the old /self-model/latest."""
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        payload = {"overall_condition": "steady", "trajectory_condition": "improving"}
        return _Resp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(boot.urllib.request, "urlopen", fake_urlopen)
    out = _fetch_self_model("http://orion-self-state-runtime:8118/")
    assert seen["url"] == "http://orion-self-state-runtime:8118/latest"
    assert out["identity_summary"] == "steady"
    assert out["anchor_strategy"] == "improving"
