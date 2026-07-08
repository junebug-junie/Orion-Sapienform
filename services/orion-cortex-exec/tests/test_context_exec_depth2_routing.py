from __future__ import annotations

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.context_exec import ContextExecRunV1

from app import settings as settings_mod
from app.supervisor import Supervisor


class _FakeContextExecClient:
    def __init__(self, run: ContextExecRunV1):
        self.run_obj = run
        self.calls = 0

    async def run(self, **kwargs):
        self.calls += 1
        return self.run_obj


@pytest.mark.asyncio
async def test_context_exec_dispatch_selected(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "context_exec_enabled", True)
    from orion.schemas.cortex.schemas import ExecutionPlan

    sup = Supervisor(bus=object())  # type: ignore[arg-type]
    fake = _FakeContextExecClient(
        ContextExecRunV1(
            run_id="r1",
            status="ok",
            mode="general_investigation",
            text="grounded answer",
            final_text="grounded answer",
            artifact={},
            artifact_type=None,
            runtime_debug={"engine": "context_exec"},
        )
    )
    sup.context_exec_client = fake  # type: ignore[method-assign]

    step = await sup._context_exec_escalation(
        source=ServiceRef(name="cortex-exec", version="test", node="test"),
        correlation_id="corr-1",
        ctx={"mode": "agent", "messages": [{"role": "user", "content": "probe"}]},
        packs=[],
    )
    assert step.step_name == "context_exec"
    assert step.result["ContextExecService"]["final_text"] == "grounded answer"
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_context_exec_disabled_fail_closed(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "context_exec_enabled", False)
    sup = Supervisor(bus=object())  # type: ignore[arg-type]
    step = await sup._context_exec_escalation(
        source=ServiceRef(name="cortex-exec", version="test", node="test"),
        correlation_id="corr-2",
        ctx={"mode": "agent", "messages": [{"role": "user", "content": "probe"}]},
        packs=[],
    )
    assert step.status == "fail"
    assert "removed" in (step.result["ContextExecService"]["text"] or "").lower()
