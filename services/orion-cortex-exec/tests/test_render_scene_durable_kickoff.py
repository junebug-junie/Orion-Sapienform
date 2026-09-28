"""RenderSceneVerb submits a `reverie.visual` durable run and returns a pending
settlement instead of blocking on thought's /visual-chain/run-once.

Design: docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md
(Kickoff). The verb never claims an outcome it has not seen: a confirmed submit
is `outcome="unknown"` + `settlement.state="pending"`; a failed submit is
`not_submitted` with no fallback to the direct call (no double render).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import pytest

import app.verb_adapters as verb_adapters
from app.verb_adapters import RenderSceneVerb
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.core.verbs.base import VerbContext
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.reverie_visual_run import reverie_visual_run_id

SOURCE = ServiceRef(name="orion-cortex-orch", version="0.1.0", node="athena")


class _FakeBus:
    """cortex-orch's durable ingress: captures the kickoff, replies with a
    CortexClientResult carrying the runner's receipt (or a scripted fault)."""

    def __init__(self, *, mode: str = "accepted") -> None:
        self.codec = OrionCodec()
        self.mode = mode
        self.calls: list[dict] = []

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        self.calls.append({"channel": request_channel, "payload": envelope.payload, "timeout_sec": timeout_sec})
        if self.mode == "rpc_error":
            raise TimeoutError("cortex-orch did not answer")
        durable = DurableRunRequestV1.model_validate(envelope.payload["context"]["metadata"]["durable_run"])
        receipt = {
            "schema_version": "durable.run.receipt.v1",
            "run_id": durable.run_id,
            "status": "accepted",
            "workflow_kind": durable.workflow,
            "requested_resource": durable.admission.resource,
            "workflow": durable.workflow,
        }
        if self.mode == "wrong_run":
            receipt["run_id"] = "reverie-visual-someone-else"
        if self.mode == "fail":
            payload = {"ok": False, "status": "fail", "error": {"message": "invalid durable admission receipt", "type": "AdmissionUnconfirmed"}}
        else:
            payload = {"ok": True, "status": "accepted", "metadata": {"durable_run": receipt}}
        reply = BaseEnvelope(kind="cortex.orch.result", source=SOURCE, correlation_id=envelope.correlation_id, payload=payload)
        return {"data": self.codec.encode(reply)}


def _payload(skill_args: dict) -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="skills.imagination.render_scene.v1", steps=[]),
        args=PlanExecutionArgs(request_id="req-1", extra={"skill_args": skill_args}),
        context={"metadata": {}},
    )


def _run(bus, skill_args: dict):
    ctx = VerbContext(meta={"bus": bus, "correlation_id": "corr-1"})
    output, effects = asyncio.run(RenderSceneVerb().execute(ctx, _payload(skill_args)))
    assert effects == []
    return output, json.loads(output.final_text)


@pytest.fixture(autouse=True)
def _durable_on(monkeypatch):
    monkeypatch.setattr(verb_adapters.settings, "render_scene_durable_enabled", True)
    monkeypatch.setattr(verb_adapters.settings, "render_scene_retry_window_sec", 0.0)


@pytest.fixture
def _no_direct_call(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("the direct /visual-chain/run-once call must not run")

    monkeypatch.setattr(verb_adapters, "_http_json_post", _boom)


def test_confirmed_submit_returns_pending_settlement(_no_direct_call):
    bus = _FakeBus()
    output, result = _run(bus, {"dispatch_id": "dispatch-abc", "proposal_id": "p1", "decision_id": "d1"})

    assert output.ok is True and output.status == "ok"
    assert result["outcome"] == "unknown"
    assert result["ran"] is False and result["refused"] is False
    settlement = result["settlement"]
    assert settlement["state"] == "pending"
    assert settlement["durable_run_id"] == reverie_visual_run_id("dispatch-abc")
    submitted = datetime.fromisoformat(settlement["submitted_at"])
    deadline = datetime.fromisoformat(settlement["deadline_at"])
    # Retry window defaults to the visual baseline interval.
    assert deadline - submitted == timedelta(seconds=5400)

    [call] = bus.calls
    assert call["channel"] == verb_adapters.self_study_module.CORTEX_ORCH_REQUEST_CHANNEL
    durable = DurableRunRequestV1.model_validate(call["payload"]["context"]["metadata"]["durable_run"])
    assert durable.workflow == "reverie.visual"
    assert durable.run_id == reverie_visual_run_id("dispatch-abc")
    assert durable.brief.visual_request.dispatch_id == "dispatch-abc"
    assert durable.brief.visual_request.proposal_id == "p1"
    assert durable.brief.visual_request.decision_id == "d1"
    assert durable.brief.visual_request.correlation_id == "corr-1"
    assert durable.admission.resource == "service.route.diffusion"
    assert durable.admission.preferred_lane == "diffusion"
    assert durable.admission.deadline_at == deadline


def test_retry_window_setting_overrides_baseline(monkeypatch, _no_direct_call):
    monkeypatch.setattr(verb_adapters.settings, "render_scene_retry_window_sec", 1200.0)
    _, result = _run(_FakeBus(), {"dispatch_id": "dispatch-abc"})
    settlement = result["settlement"]
    delta = datetime.fromisoformat(settlement["deadline_at"]) - datetime.fromisoformat(settlement["submitted_at"])
    assert delta == timedelta(seconds=1200)


def test_retry_window_never_outlives_thoughts_attempt_max_age(monkeypatch, _no_direct_call):
    from orion.schemas.reverie_visual_run import REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC

    monkeypatch.setattr(verb_adapters.settings, "render_scene_retry_window_sec", 86400.0)
    _, result = _run(_FakeBus(), {"dispatch_id": "dispatch-abc"})
    settlement = result["settlement"]
    delta = datetime.fromisoformat(settlement["deadline_at"]) - datetime.fromisoformat(settlement["submitted_at"])
    assert delta == timedelta(seconds=REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC)


@pytest.mark.parametrize(
    ("mode", "reason_prefix"),
    [
        ("rpc_error", "TimeoutError"),
        ("fail", "not_accepted:fail"),
        ("wrong_run", "receipt_run_id_mismatch"),
    ],
)
def test_unconfirmed_submit_is_not_submitted_and_never_falls_back(mode, reason_prefix, _no_direct_call):
    output, result = _run(_FakeBus(mode=mode), {"dispatch_id": "dispatch-abc"})

    assert output.ok is False and output.status == "unavailable"
    assert result["outcome"] == "unknown"
    assert result["settlement"]["state"] == "not_submitted"
    assert result["settlement"]["reason"].startswith(reason_prefix)


def test_missing_bus_is_not_submitted(_no_direct_call):
    output, result = _run(None, {"dispatch_id": "dispatch-abc"})
    assert output.ok is False
    assert result["settlement"] == {"state": "not_submitted", "durable_run_id": reverie_visual_run_id("dispatch-abc"), "reason": "missing_bus"}


def _direct_reply(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def _fake_post(url, *, body, timeout_sec):
        calls.append({"url": url, "body": body})
        return {"outcome": "produced", "ran": True, "refused": False, "chain_id": "chain-1"}

    monkeypatch.setattr(verb_adapters, "_http_json_post", _fake_post)
    return calls


def test_manual_run_without_dispatch_id_keeps_the_direct_path(monkeypatch):
    calls = _direct_reply(monkeypatch)
    bus = _FakeBus()
    output, result = _run(bus, {})
    assert bus.calls == []
    assert calls and calls[0]["url"].endswith("/visual-chain/run-once")
    assert result["outcome"] == "produced"
    assert "settlement" not in result
    assert output.ok is True


def test_setting_off_restores_the_direct_path(monkeypatch):
    monkeypatch.setattr(verb_adapters.settings, "render_scene_durable_enabled", False)
    calls = _direct_reply(monkeypatch)
    bus = _FakeBus()
    _, result = _run(bus, {"dispatch_id": "dispatch-abc"})
    assert bus.calls == []
    assert calls[0]["body"]["dispatch_id"] == "dispatch-abc"
    assert result["outcome"] == "produced"
    assert "settlement" not in result


def test_kickoff_timeout_stays_under_the_dispatch_route_budget():
    """The submit RPC must finish well inside the dispatch route's own RPC budget,
    or dispatch severs a submit that may have succeeded."""
    from pathlib import Path

    import yaml

    from app.durable_kickoff import DEFAULT_KICKOFF_TIMEOUT_SEC

    repo = Path(verb_adapters.__file__).resolve().parents[3]
    policy = yaml.safe_load((repo / "config/execution_dispatch/execution_dispatch_policy.v1.yaml").read_text())
    route = policy["template_to_cortex"]["render_scene"]
    assert DEFAULT_KICKOFF_TIMEOUT_SEC < float(route["rpc_timeout_sec"])
