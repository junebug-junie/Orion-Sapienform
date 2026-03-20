from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_agent_chain_settings_accepts_legacy_channel_env(monkeypatch):
    monkeypatch.delenv("AGENT_CHAIN_REQUEST_CHANNEL", raising=False)
    monkeypatch.delenv("AGENT_CHAIN_RESULT_PREFIX", raising=False)
    monkeypatch.setenv("CHANNEL_AGENT_CHAIN_INTAKE", "orion:exec:request:AgentChainService:legacy")
    monkeypatch.setenv("CHANNEL_AGENT_CHAIN_REPLY_PREFIX", "orion:exec:result:AgentChainService:legacy")

    from app.settings import AgentChainSettings

    cfg = AgentChainSettings()
    assert cfg.agent_chain_request_channel == "orion:exec:request:AgentChainService:legacy"
    assert cfg.agent_chain_result_prefix == "orion:exec:result:AgentChainService:legacy"


def test_agent_chain_settings_prefers_canonical_env_over_legacy(monkeypatch):
    monkeypatch.setenv("AGENT_CHAIN_REQUEST_CHANNEL", "orion:exec:request:AgentChainService:canonical")
    monkeypatch.setenv("AGENT_CHAIN_RESULT_PREFIX", "orion:exec:result:AgentChainService:canonical")
    monkeypatch.setenv("CHANNEL_AGENT_CHAIN_INTAKE", "orion:exec:request:AgentChainService:legacy")
    monkeypatch.setenv("CHANNEL_AGENT_CHAIN_REPLY_PREFIX", "orion:exec:result:AgentChainService:legacy")

    from app.settings import AgentChainSettings

    cfg = AgentChainSettings()
    assert cfg.agent_chain_request_channel == "orion:exec:request:AgentChainService:canonical"
    assert cfg.agent_chain_result_prefix == "orion:exec:result:AgentChainService:canonical"


def test_agent_chain_settings_llm_and_cognition_defaults(monkeypatch):
    monkeypatch.delenv("LLM_REQUEST_CHANNEL", raising=False)
    monkeypatch.delenv("LLM_REPLY_PREFIX", raising=False)
    monkeypatch.delenv("COGNITION_BASE_DIR", raising=False)

    from app.settings import AgentChainSettings

    cfg = AgentChainSettings()
    assert cfg.llm_request_channel == "orion:exec:request:LLMGatewayService"
    assert cfg.llm_reply_prefix == "orion:llm:reply"
    assert cfg.cognition_base_dir == "/app/orion/cognition"
