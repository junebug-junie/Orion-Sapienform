"""Request-scoped leases reach FCC without leaking across concurrent turns."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from orion.harness.fcc_motor import _build_subprocess_env
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.llm.resource_lease import LEASE_HEADER, decode_lease_header
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1
from orion.schemas.resource_admission import ResourceLeaseV1


def token():
    now = datetime.now(timezone.utc)
    return ResourceLeaseV1(run_id="run-one", demand_id="run-one:turn", lease_id="lease-one",
        resource_key="llm.route.agent", lane="agent", backend_key="http://worker:8000", generation=2,
        granted_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=60))


def test_subprocess_lease_is_scoped_and_existing_headers_survive(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"X-Trace: trace\n{LEASE_HEADER.lower()}: stale")
    lease = token().model_dump(mode="json")
    protected = _build_subprocess_env(fcc_server_url="http://gateway:8080", auth_token="tok", resource_lease=lease)
    headers = protected["ANTHROPIC_CUSTOM_HEADERS"].splitlines()
    assert headers[0] == "X-Trace: trace"
    assert decode_lease_header(headers[1].split(": ", 1)[1]) == lease
    unprotected = _build_subprocess_env(fcc_server_url="http://gateway:8080", auth_token="tok")
    assert unprotected["ANTHROPIC_CUSTOM_HEADERS"] == "X-Trace: trace"


@pytest.mark.asyncio
async def test_motor_receives_admitted_timeout_and_lease():
    captured = {}
    async def motor(**kwargs):
        captured.update(kwargs)
        yield {"type": "error", "error": "synthetic stop", "error_code": "test"}
    lease = token()
    request = HarnessRunRequestV1(correlation_id="run-one", thought_event=make_thought(), user_message="study",
        permissions=ContextExecPermissionV1(), answer_contract=AnswerContract(), resource_lease=lease,
        inference_timeout_sec=42, fcc_model_label="Agent - Orion")
    runner = HarnessRunner(AsyncMock(), fcc_runner=motor, fcc_timeout_sec=999)
    await runner.run(request)
    assert captured["timeout_sec"] == 42
    assert captured["resource_lease"] == lease.model_dump(mode="json")


def test_leased_turn_targets_gateway_directly_without_proxy_credentials(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_GATEWAY_URL", "http://llm-gateway:8210/")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "external-proxy-secret")
    lease = token().model_dump(mode="json")
    protected = _build_subprocess_env(fcc_server_url="http://fcc-proxy:8082", auth_token="fcc-secret", resource_lease=lease)
    assert protected["ANTHROPIC_BASE_URL"] == "http://llm-gateway:8210"
    assert protected["ANTHROPIC_AUTH_TOKEN"] == "orion-resource-lease"
    assert "ANTHROPIC_API_KEY" not in protected
    legacy = _build_subprocess_env(fcc_server_url="http://fcc-proxy:8082", auth_token="fcc-secret")
    assert legacy["ANTHROPIC_BASE_URL"] == "http://fcc-proxy:8082"
    assert legacy["ANTHROPIC_AUTH_TOKEN"] == "fcc-secret"
