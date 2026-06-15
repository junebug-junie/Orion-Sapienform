"""Schema tests for investigation_v2 mode and profile permissions."""

from __future__ import annotations

from orion.schemas.context_exec import (
    ContextExecRequestV1,
    context_exec_permissions_for_llm_profile,
)


def test_context_exec_request_accepts_investigation_v2_mode() -> None:
    req = ContextExecRequestV1(text="probe", mode="investigation_v2")
    assert req.mode == "investigation_v2"


def test_context_exec_permissions_for_llm_profile_agent_read_repo() -> None:
    perms = context_exec_permissions_for_llm_profile("agent")
    assert perms.read_repo is True
