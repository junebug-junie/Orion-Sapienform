"""Schema validation for context-exec llm_profile."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.schemas.context_exec import ContextExecRequestV1


def test_context_exec_request_accepts_valid_llm_profiles() -> None:
    for profile in ("chat", "quick", "agent", "metacog"):
        req = ContextExecRequestV1(text="probe", llm_profile=profile)
        assert req.llm_profile == profile


def test_context_exec_request_normalizes_llm_profile_case() -> None:
    req = ContextExecRequestV1(text="probe", llm_profile="AGENT")
    assert req.llm_profile == "agent"


def test_context_exec_request_omitted_llm_profile_allowed() -> None:
    req = ContextExecRequestV1(text="probe")
    assert req.llm_profile is None


@pytest.mark.parametrize("bad", ["http://evil", "circe-32b", "latents", ""])
def test_context_exec_request_rejects_invalid_llm_profile(bad: str) -> None:
    with pytest.raises(ValidationError):
        ContextExecRequestV1(text="probe", llm_profile=bad)
