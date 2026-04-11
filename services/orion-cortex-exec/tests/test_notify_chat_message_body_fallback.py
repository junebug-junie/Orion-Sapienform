"""Notify skill must populate body_text from bridge context when skill_args omit it (verb_runtime path)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
from uuid import uuid4

from orion.core.verbs.base import VerbContext
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest

from app.verb_adapters import NotifyChatMessageVerb


def _plan(*, raw_user: str) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="skills.system.notify_chat_message.v1", steps=[]),
        args=PlanExecutionArgs(request_id=str(uuid4()), extra={}),
        context={
            "raw_user_text": raw_user,
            "user_message": raw_user,
            "metadata": {"capability_bridge_user_text": raw_user},
            "messages": [{"role": "user", "content": raw_user}],
        },
    )


def test_notify_verb_fills_body_from_context_when_skill_args_empty() -> None:
    captured: dict = {}

    class _Accepted:
        ok = True
        status = "accepted"
        detail = None
        notification_id = "nid-test"

    def _capture_send(req):  # noqa: ANN001
        captured["body"] = req.body_text
        return _Accepted()

    async def _run() -> None:
        verb = NotifyChatMessageVerb()
        ctx = VerbContext(request_id="rid", meta={"correlation_id": "corr-x"})
        payload = _plan(raw_user='Send a notification saying "hello from test".')
        mock_inst = MagicMock()
        mock_inst.send = _capture_send
        with patch("app.verb_adapters.NotifyClient", return_value=mock_inst):
            out, _fx = await verb.execute(ctx, payload)
        assert out.ok is True
        assert out.final_text
        assert "hello from test" in out.final_text or "Send a notification" in out.final_text
        assert captured.get("body")

    asyncio.run(_run())
