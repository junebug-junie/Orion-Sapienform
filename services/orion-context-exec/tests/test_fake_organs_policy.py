from __future__ import annotations

import pytest

from app.runner import ContextExecRunner, FAKE_ORGANS
from orion.schemas.context_exec import ContextExecRequestV1


@pytest.mark.asyncio
async def test_fake_organs_disabled_returns_no_default_evidence(monkeypatch):
    from app import settings as settings_mod
    from app import trace_tools

    trace_tools.clear_trace_store()
    monkeypatch.setattr(settings_mod.settings, "context_exec_fake_organs_enabled", False)
    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None
    runner = ContextExecRunner()
    run = await runner.run(
        ContextExecRequestV1(
            text="Where did the Denver claim come from?",
            mode="belief_provenance",
        )
    )
    assert run.status == "ok"
    assert run.findings_bundle is None or len(run.findings_bundle.findings) == 0
    assert run.runtime_debug.get("fake_organs_enabled") is False


@pytest.mark.asyncio
async def test_fake_organs_enabled_injects_harness_evidence(monkeypatch):
    from app import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "context_exec_fake_organs_enabled", True)
    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None
    runner = ContextExecRunner()
    run = await runner.run(
        ContextExecRequestV1(
            text="Where did the Denver claim come from?",
            mode="belief_provenance",
        )
    )
    assert run.findings_bundle is not None
    assert len(run.findings_bundle.findings) >= 1
