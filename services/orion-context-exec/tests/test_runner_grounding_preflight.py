"""Integration test: runner blocks fake-engine answer success for recall questions."""

from __future__ import annotations

import pytest

from app.runner import ContextExecRunner
from app.grounding_eval import is_placeholder_investigation_summary
from orion.schemas.context_exec import ContextExecRequestV1


@pytest.mark.asyncio
async def test_runner_fake_engine_recall_question_blocks_answer_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_fake_organs_enabled", False)
    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", False)
    monkeypatch.setattr(cfg, "rlm_engine", "fake")

    runner = ContextExecRunner()
    req = ContextExecRequestV1(
        text="do you recall where my mom lives?",
        mode="general_investigation",
        llm_profile="quick",
    )
    run = await runner.run(req)
    assert run.status == "ok"
    answer_eval = run.runtime_debug.get("answer_evaluation") or {}
    assert answer_eval.get("runtime_status") == "ok"
    assert answer_eval.get("answer_status") == "failed_fake_engine_selected"
    assert answer_eval.get("grounding_status") == "skipped"
    assert not is_placeholder_investigation_summary(run.final_text)
    assert "not a grounded investigation" in run.final_text
