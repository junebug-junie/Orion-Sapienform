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
    assert "Blocked: fake engine selected" in run.final_text
    organ_status = run.runtime_debug.get("organ_status") or {}
    assert isinstance(organ_status.get("recall"), dict)


@pytest.mark.asyncio
async def test_runner_recall_timeout_records_organ_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.organ_runtime import OrganRuntime
    from app.organ_status import record_recall
    from app.settings import settings as cfg

    monkeypatch.setattr(cfg, "context_exec_fake_organs_enabled", False)
    monkeypatch.setattr(cfg, "context_exec_agent_synthesis_enabled", False)
    monkeypatch.setattr(cfg, "rlm_engine", "fake")

    async def _fake_recall_query(self, query, *, profile="assist.light.v1", limit=None):
        payload = {
            "hits": [],
            "error": "Timeout reading from 100.92.216.81:6379",
            "query": query,
        }
        if self.organ_status is not None:
            record_recall(self.organ_status, payload)
        return payload

    monkeypatch.setattr(OrganRuntime, "recall_query", _fake_recall_query)

    runner = ContextExecRunner()
    req = ContextExecRequestV1(
        text="do you recall where my mom lives?",
        mode="general_investigation",
        llm_profile="quick",
    )
    run = await runner.run(req)
    recall = (run.runtime_debug.get("organ_status") or {}).get("recall") or {}
    assert recall.get("attempted") is True
    assert recall.get("ok") is False
    assert recall.get("hit_count") == 0
    assert "Timeout reading from" in str(recall.get("error"))
