"""Integration: conceptual turn skips repo in investigation_v2 sweep."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _ctx_app_modules():
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(CTX_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(CTX_ROOT))


@pytest.mark.asyncio
async def test_conceptual_turn_skips_repo_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    _ctx_app_modules()
    from app.investigation_v2 import run_investigation_v2
    from app.settings import settings
    from orion.cognition.answer_contract_normalize import heuristic_answer_contract
    from orion.schemas.context_exec import ContextExecRequestV1, SourceStatus, context_exec_permissions_for_llm_profile

    monkeypatch.setattr(settings, "context_exec_real_repo_enabled", True)
    monkeypatch.setattr(settings, "context_exec_real_recall_enabled", False)
    monkeypatch.setattr(settings, "context_exec_real_trace_enabled", False)

    runtime = MagicMock()
    runtime.bus = None
    runtime.recall_query = MagicMock(return_value={"hits": []})
    runtime.repo_grep = MagicMock(
        side_effect=AssertionError("repo_grep must not run for conceptual turns")
    )
    runtime.traces_search = MagicMock(return_value=[])

    namespace = MagicMock()
    namespace.memory.search_claims = MagicMock(return_value=[])
    namespace.repo.grep = MagicMock(
        side_effect=AssertionError("namespace.repo.grep must not run for conceptual turns")
    )

    text = "hey buddy, why do you give shallow responses like this?"
    req = ContextExecRequestV1(
        text=text,
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        answer_contract=heuristic_answer_contract(text),
    )
    artifact = await run_investigation_v2(req, namespace, runtime, {})
    assert artifact["evidence"]["repo"]["status"] == SourceStatus.skipped.value
    runtime.repo_grep.assert_not_called()
