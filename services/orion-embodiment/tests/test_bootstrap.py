from __future__ import annotations

from scripts.bootstrap_orion_agent import upsert_env_keys


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
