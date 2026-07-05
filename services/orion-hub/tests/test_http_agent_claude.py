from __future__ import annotations

import pytest

from scripts.agent_claude_input import prepare_agent_claude_input


@pytest.mark.asyncio
async def test_http_agent_claude_collects_events(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import api_routes

    async def fake_run_turn_from_settings(**kwargs):
        yield {"type": "step", "step": {"type": "assistant", "raw": {"type": "assistant"}}}
        yield {"type": "final", "llm_response": "HTTP done.", "metadata": {"exit_code": 0}}

    monkeypatch.setattr(api_routes, "run_turn_from_settings", fake_run_turn_from_settings)
    monkeypatch.setattr(api_routes.settings, "HUB_AGENT_CLAUDE_ENABLED", True, raising=False)

    result = await api_routes._run_agent_claude_http(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-http",
    )
    assert result["llm_response"] == "HTTP done."
    assert len(result["claude_steps"]) == 1
    assert prepare_agent_claude_input("x").prompt == "x"
