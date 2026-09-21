"""_call_self_study_reflect_llm's durable-first-then-fallback path (GPU2
elastic-burst arc, 2026-09-21, step 3 of 3): when SELF_STUDY_REFLECT_DURABLE_ENABLED
is on, dispatch as a durable run and wait synchronously for its completion
event, same external contract (`list[dict] | None`) as the direct RPC always
had. A failed/unconfirmed dispatch falls back to the direct RPC unchanged;
an accepted-but-failed durable run does NOT also fall back (no double spend
of the reflect timeout budget).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.schemas.durable_run import DurableRunStateV1  # noqa: E402

import app.self_study as self_study  # noqa: E402

SOURCE = ServiceRef(name="orion-cortex-exec", version="0.1.0", node="athena")


class _FakePubSub:
    """References the bus itself rather than snapshotting its envelope list
    at subscribe time -- a test can append/replace `bus._state_envelopes`
    AFTER subscribing (simulating the completion event arriving later, which
    is the real timing every one of these tests needs) and iter_messages
    still sees it."""

    def __init__(self, bus: "_FakeBus"):
        self._bus = bus

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeBus:
    """Supports both the dispatch RPC (rpc_request) and the completion wait
    (subscribe/iter_messages), the two things `_call_self_study_reflect_llm`'s
    durable path needs beyond the plain RPC the direct path already used."""

    def __init__(
        self,
        *,
        dispatch_status: str = "accepted",
        dispatch_rpc_error: bool = False,
        state_envelopes: list[BaseEnvelope] | None = None,
        direct_findings: list[dict] | None = None,
    ) -> None:
        self.codec = OrionCodec()
        self.rpc_calls: list[str] = []
        self._dispatch_status = dispatch_status
        self._dispatch_rpc_error = dispatch_rpc_error
        self._state_envelopes = state_envelopes or []
        self._direct_findings = direct_findings if direct_findings is not None else [{"reflection_kind": "pattern", "title": "t", "description": "d", "concept_kinds": ["runtime_boundary"]}]

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        payload = envelope.payload
        is_durable_kickoff = isinstance(payload, dict) and "durable_run" in (payload.get("context") or {}).get("metadata", {})
        self.rpc_calls.append("durable_kickoff" if is_durable_kickoff else "direct")
        if is_durable_kickoff:
            if self._dispatch_rpc_error:
                raise RuntimeError("cortex down")
            result = BaseEnvelope(
                kind="cortex.orch.result", source=SOURCE, correlation_id=envelope.correlation_id,
                payload={"status": self._dispatch_status},
            )
            return {"data": self.codec.encode(result)}
        # direct verb-dispatch RPC (the original path)
        result = BaseEnvelope(
            kind="cortex.orch.result", source=SOURCE, correlation_id=envelope.correlation_id,
            payload={"ok": True, "status": "success", "final_text": json.dumps({"findings": self._direct_findings})},
        )
        return {"data": self.codec.encode(result)}

    def subscribe(self, channel):
        return _FakePubSub(self)

    async def iter_messages(self, pubsub: _FakePubSub):
        sent = 0
        while True:
            envelopes = self._state_envelopes
            while sent < len(envelopes):
                yield {"data": self.codec.encode(envelopes[sent])}
                sent += 1
            await asyncio.sleep(0.01)  # poll for a later-arriving event, same as a real pubsub waiting


def _completed_envelope(run_id: str, *, ok: bool, findings: list[dict] | None = None, error: str | None = None) -> BaseEnvelope:
    corr = "11111111-1111-1111-1111-111111111111"
    state = DurableRunStateV1(
        run_id=run_id, workflow="self_study.reflect", thread_id=run_id, node="finish",
        status="completed", correlation_id=corr,
        detail={"line": "reflect", "llm_call_ok": ok, "llm_call_error": error, "findings": findings or [], "attempts": 1},
    )
    return BaseEnvelope(kind="durable.run.state.v1", source=SOURCE, correlation_id=corr, payload=state.model_dump(mode="json"))


def _snapshot_and_concepts():
    snapshot = self_study.build_self_snapshot(observed_at="2026-03-21T00:00:00+00:00")
    concepts = self_study.induce_self_concepts(snapshot)
    return snapshot, concepts


def test_durable_disabled_by_default_goes_straight_to_the_direct_rpc(monkeypatch):
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_DURABLE_ENABLED", False)
    bus = _FakeBus()
    snapshot, concepts = _snapshot_and_concepts()
    findings = asyncio.run(self_study._call_self_study_reflect_llm(
        bus=bus, source=SOURCE, snapshot=snapshot, concepts=concepts, correlation_id="c1",
    ))
    assert bus.rpc_calls == ["direct"]
    assert findings and findings[0]["title"] == "t"


def test_durable_enabled_accepted_and_completed_returns_findings_from_the_event(monkeypatch):
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_DURABLE_ENABLED", True)
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_TIMEOUT_SEC", 2.0)
    snapshot, concepts = _snapshot_and_concepts()

    async def scenario():
        # The run_id is minted inside _dispatch_reflect_durable_run, so we
        # can't know it ahead of time -- capture it from the dispatch RPC
        # call itself via a bus that inspects the kickoff payload.
        captured: dict = {}

        class _CapturingBus(_FakeBus):
            async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
                payload = envelope.payload
                if isinstance(payload, dict) and "durable_run" in (payload.get("context") or {}).get("metadata", {}):
                    captured["run_id"] = payload["context"]["metadata"]["durable_run"]["run_id"]
                return await super().rpc_request(request_channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec)

        bus = _CapturingBus(dispatch_status="accepted")

        async def call():
            return await self_study._call_self_study_reflect_llm(
                bus=bus, source=SOURCE, snapshot=snapshot, concepts=concepts, correlation_id="c2",
            )

        task = asyncio.create_task(call())
        # Give _dispatch_reflect_durable_run a tick to run and populate `captured`.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if "run_id" in captured:
                break
        assert "run_id" in captured
        bus._state_envelopes = [_completed_envelope(captured["run_id"], ok=True, findings=[{"reflection_kind": "pattern", "title": "durable finding", "description": "d", "concept_kinds": ["runtime_boundary"]}])]
        return await task, bus

    result, bus = asyncio.run(scenario())
    assert bus.rpc_calls == ["durable_kickoff"]  # never fell back to the direct path
    assert result == [{"reflection_kind": "pattern", "title": "durable finding", "description": "d", "concept_kinds": ["runtime_boundary"]}]


def test_a_failed_dispatch_falls_back_to_the_direct_rpc(monkeypatch):
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_DURABLE_ENABLED", True)
    bus = _FakeBus(dispatch_rpc_error=True)
    snapshot, concepts = _snapshot_and_concepts()
    findings = asyncio.run(self_study._call_self_study_reflect_llm(
        bus=bus, source=SOURCE, snapshot=snapshot, concepts=concepts, correlation_id="c3",
    ))
    assert bus.rpc_calls == ["durable_kickoff", "direct"]
    assert findings and findings[0]["title"] == "t"


def test_an_unconfirmed_dispatch_falls_back_to_the_direct_rpc(monkeypatch):
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_DURABLE_ENABLED", True)
    bus = _FakeBus(dispatch_status="rejected")
    snapshot, concepts = _snapshot_and_concepts()
    findings = asyncio.run(self_study._call_self_study_reflect_llm(
        bus=bus, source=SOURCE, snapshot=snapshot, concepts=concepts, correlation_id="c4",
    ))
    assert bus.rpc_calls == ["durable_kickoff", "direct"]
    assert findings and findings[0]["title"] == "t"


def test_an_accepted_dispatch_whose_run_fails_does_not_double_dispatch(monkeypatch):
    """Once a dispatch IS accepted, a genuinely failed durable outcome must
    NOT trigger the direct RPC too -- that would double-spend the reflect
    timeout budget on two separate attempts for one logical reflection."""
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_DURABLE_ENABLED", True)
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_TIMEOUT_SEC", 2.0)
    snapshot, concepts = _snapshot_and_concepts()

    async def scenario():
        captured: dict = {}

        class _CapturingBus(_FakeBus):
            async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
                payload = envelope.payload
                if isinstance(payload, dict) and "durable_run" in (payload.get("context") or {}).get("metadata", {}):
                    captured["run_id"] = payload["context"]["metadata"]["durable_run"]["run_id"]
                return await super().rpc_request(request_channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec)

        bus = _CapturingBus(dispatch_status="accepted")

        async def call():
            return await self_study._call_self_study_reflect_llm(
                bus=bus, source=SOURCE, snapshot=snapshot, concepts=concepts, correlation_id="c5",
            )

        task = asyncio.create_task(call())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if "run_id" in captured:
                break
        bus._state_envelopes = [_completed_envelope(captured["run_id"], ok=False, error="llm_call_failed")]
        return await task, bus

    result, bus = asyncio.run(scenario())
    assert bus.rpc_calls == ["durable_kickoff"]  # no fallback
    assert result is None


def test_an_accepted_dispatch_with_no_completion_event_times_out_to_none(monkeypatch):
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_DURABLE_ENABLED", True)
    monkeypatch.setattr(self_study, "SELF_STUDY_REFLECT_TIMEOUT_SEC", 0.2)
    bus = _FakeBus(dispatch_status="accepted")
    snapshot, concepts = _snapshot_and_concepts()
    result = asyncio.run(self_study._call_self_study_reflect_llm(
        bus=bus, source=SOURCE, snapshot=snapshot, concepts=concepts, correlation_id="c6",
    ))
    assert bus.rpc_calls == ["durable_kickoff"]  # no fallback -- accepted means trust the durable path
    assert result is None
