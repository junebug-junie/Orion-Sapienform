"""The compactors hand their LLM digest calls to an admitted ``compactor.digest`` durable run.

Real submission path (``real_durable_submit``): no LLM call happens in cortex-orch; the request is
a DurableRunRequestV1 on agent/background admission with a deadline derived from the window; the
run_id is deterministic per input; a failed previous run gets the next generation; a completed one
is reported as already done; a missing receipt fails the dispatch. Finalize path: a
``workflow_request.durable_digest`` writes the card + journal without fetching or calling an LLM.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app import workflow_runtime as wr
from app.workflow_runtime import execute_chat_workflow
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.compactor_digest_run import CompactorDigestResultV1
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult
from orion.schemas.durable_run import DurableRunRequestV1

NOW = datetime(2026, 9, 29, 12, 10, tzinfo=timezone.utc)  # 06:10 MDT


class _Bus:
    enabled = True

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, channel: str, envelope: Any) -> None:
        self.published.append((channel, envelope))

    def journal_payloads(self) -> list[dict]:
        return [env.payload if isinstance(env.payload, dict) else env.payload.model_dump(mode="json")
                for ch, env in self.published if ch == "orion:journal:write"]


class _Result:
    def __init__(self, *, ok: bool = True, final_text: str = "", metadata: dict | None = None):
        self.ok = ok
        self.error = None
        self.output = {"result": {"status": "success", "final_text": final_text, "metadata": metadata or {}}}


def _req(workflow_id: str, **workflow_extra) -> CortexClientRequest:
    return CortexClientRequest.model_validate({
        "mode": "brain", "route_intent": "none", "options": {}, "packs": [],
        "recall": {"enabled": False, "required": False},
        "context": {
            "messages": [], "raw_user_text": "workflow please", "user_message": "workflow please",
            "session_id": "sid-1", "user_id": "user-1", "trace_id": "trace-1",
            "metadata": {"workflow_request": {"workflow_id": workflow_id, **workflow_extra}},
        },
    })


def _scheduled(workflow_id: str, *, notify_on: str = "failure", run_id: str = "sched-run-1") -> CortexClientRequest:
    return _req(workflow_id, scheduled_dispatch={"run_id": run_id, "source": "orion-actions"},
                execution_policy={"workflow_id": workflow_id, "invocation_mode": "immediate",
                                  "notify_on": notify_on,
                                  "schedule": {"kind": "recurring", "cadence": "daily", "hour_local": 6,
                                               "minute_local": 10, "timezone": "America/Denver"}})


def _prs(n: int) -> list[dict]:
    return [{"number": 1000 + i, "title": f"PR {i}", "body": "detail " * 1300, "merged_at": "2026-09-28T18:00:00Z",
             "url": f"https://github.com/acme/widgets/pull/{1000 + i}"} for i in range(n)]


def _fetch_only(items: list[dict]):
    calls: list[str] = []

    async def fake(*args, **kwargs):
        req = kwargs["client_request"]
        calls.append(req.verb)
        if req.verb == "skills.repo.github_recent_prs.v1":
            return _Result(final_text=json.dumps({"available": True, "repo": "acme/widgets", "items": items,
                                                  "window_mode": "window"}))
        raise AssertionError(f"cortex-orch must not call {req.verb} in-process")
    return fake, calls


class _Dispatch:
    """Stands in for ``durable_runs.dispatch_durable_run`` (the receipt RPC to orion-durable-runs)."""

    def __init__(self, statuses: list[str] | None = None, ok: bool = True) -> None:
        self.statuses = list(statuses or ["waiting_resource"])
        self.ok = ok
        self.requests: list[DurableRunRequestV1] = []
        self.kwargs: list[dict] = []

    async def __call__(self, *, bus, source, req, correlation_id, admission_enabled, receipt_timeout_sec, **_):
        request = DurableRunRequestV1.model_validate(req.context.metadata["durable_run"])
        self.requests.append(request)
        self.kwargs.append({"admission_enabled": admission_enabled, "receipt_timeout_sec": receipt_timeout_sec})
        if not self.ok:
            return CortexClientResult(ok=False, mode="brain", verb="durable:compactor.digest", status="fail",
                                      steps=[], correlation_id=correlation_id,
                                      error={"message": "invalid durable admission receipt", "run_id": request.run_id})
        status = self.statuses.pop(0) if self.statuses else "waiting_resource"
        return CortexClientResult(ok=True, mode="brain", verb="durable:compactor.digest", status="accepted",
                                  steps=[], correlation_id=correlation_id,
                                  metadata={"durable_run": {"run_id": request.run_id, "status": status,
                                                            "workflow_kind": request.workflow}})


@pytest.fixture
def fixed_now(monkeypatch):
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is None else NOW.astimezone(tz)

    monkeypatch.setattr(wr, "datetime", _FixedDatetime)


@pytest.fixture
def notifications(monkeypatch):
    sent: list[dict] = []

    async def _notify(**kwargs):
        sent.append(kwargs)

    monkeypatch.setattr(wr, "_emit_workflow_notify", _notify)
    return sent


@pytest.fixture(autouse=True)
def _cards(monkeypatch):
    cards: list[dict] = []

    async def _persist(**kwargs):
        cards.append(kwargs)
        return "00000000-0000-0000-0000-0000000000aa"

    monkeypatch.setattr(wr, "persist_github_compactor_memory_card", _persist)
    monkeypatch.setattr(wr, "persist_chat_history_compactor_memory_card", _persist)
    return cards


def _run(req, fake, bus=None):
    return asyncio.run(execute_chat_workflow(
        bus=bus or _Bus(), source=ServiceRef(name="cortex-orch"), req=req,
        correlation_id="00000000-0000-0000-0000-00000000c0de", causality_chain=[], trace={},
        call_verb_runtime=fake))


@pytest.mark.real_durable_submit
def test_github_day_submits_admitted_durable_run_and_makes_no_llm_call(monkeypatch, fixed_now, notifications):
    dispatch = _Dispatch()
    monkeypatch.setattr(wr, "dispatch_durable_run", dispatch)
    fake, calls = _fetch_only(_prs(40))
    bus = _Bus()

    result = _run(_scheduled("github_compactor_pass", notify_on="completion"), fake, bus)

    assert calls == ["skills.repo.github_recent_prs.v1"]            # the fetch only; no digest verb
    assert result.ok is True and result.status == "accepted"
    assert bus.journal_payloads() == []                            # nothing finalized yet
    assert notifications == []                                     # "completed" would be a lie now
    [request] = dispatch.requests
    assert request.workflow == "compactor.digest"
    assert request.admission.resource == "llm.route.agent" and request.admission.priority == "background"
    brief = request.brief
    # Deadline from the window, not from now: the Denver day's end + 24h (so identical re-submits).
    window_end = datetime.fromisoformat(brief.finalize["window_end_utc"])
    assert window_end.isoformat().startswith("2026-09-29T05:59:59")
    assert request.admission.deadline_at - window_end == timedelta(hours=24)
    assert brief.kind == "github" and brief.llm_route == "agent" and brief.window_label == "2026-09-28"
    covered = sorted(item["number"] for chunk in brief.inputs for item in chunk["items"])
    assert covered == [1000 + i for i in range(40)] and len(brief.inputs) > 1
    assert brief.finalize["coverage"]["total_count"] == 40 and brief.finalize["repo"] == "acme/widgets"
    assert brief.finalize["execution_policy"]["notify_on"] == "completion"
    assert "schedule" not in brief.finalize["execution_policy"]   # no volatile schedule spec in the input
    wf = result.metadata["workflow"]
    assert wf["status"] == "accepted" and wf["executed"] is False
    assert wf["durable_run"]["run_id"] == request.run_id and wf["durable_run"]["chunk_count"] == len(brief.inputs)
    assert dispatch.kwargs[0]["admission_enabled"] is wr.get_settings().durable_admission_enabled


@pytest.mark.real_durable_submit
def test_run_id_is_deterministic_per_window_and_input(monkeypatch, fixed_now):
    dispatch = _Dispatch(statuses=["waiting_resource"] * 3)
    monkeypatch.setattr(wr, "dispatch_durable_run", dispatch)
    fake, _ = _fetch_only(_prs(3))
    _run(_scheduled("github_compactor_pass", run_id="claim-a"), fake)
    _run(_scheduled("github_compactor_pass", run_id="claim-b"), fake)   # a retry claim: new schedule run id
    changed, _ = _fetch_only(_prs(4))
    _run(_scheduled("github_compactor_pass", run_id="claim-c"), changed)
    first, again, other = dispatch.requests
    assert first.run_id == again.run_id and first.correlation_id == again.correlation_id
    assert first.run_id.startswith("compactor:github_compactor_pass:2026-09-28:acme-widgets:")
    assert other.run_id != first.run_id


@pytest.mark.real_durable_submit
def test_failed_previous_run_gets_next_generation_and_completed_is_idempotent(monkeypatch, fixed_now, notifications):
    fake, _ = _fetch_only(_prs(2))
    retry = _Dispatch(statuses=["failed", "cancelled", "waiting_resource"])
    monkeypatch.setattr(wr, "dispatch_durable_run", retry)
    accepted = _run(_scheduled("github_compactor_pass"), fake)
    ids = [r.run_id for r in retry.requests]
    assert ids[1] == f"{ids[0]}:g2" and ids[2] == f"{ids[0]}:g3"
    assert accepted.metadata["workflow"]["durable_run"]["generation"] == 3

    done = _Dispatch(statuses=["completed"])
    monkeypatch.setattr(wr, "dispatch_durable_run", done)
    result = _run(_scheduled("github_compactor_pass", notify_on="completion"), fake)
    assert result.ok is True and result.status == "success"
    assert result.metadata["workflow"]["status"] == "completed"
    assert "already finalized" in result.metadata["workflow"]["main_result"]


@pytest.mark.real_durable_submit
def test_no_receipt_fails_the_dispatch_loudly(monkeypatch, fixed_now):
    monkeypatch.setattr(wr, "dispatch_durable_run", _Dispatch(ok=False))
    fake, _ = _fetch_only(_prs(2))
    with pytest.raises(wr.WorkflowExecutionError, match="compactor_durable_submit_failed"):
        _run(_scheduled("github_compactor_pass"), fake)


@pytest.mark.real_durable_submit
def test_generations_are_bounded(monkeypatch, fixed_now):
    monkeypatch.setattr(wr, "dispatch_durable_run", _Dispatch(statuses=["failed"] * 10))
    fake, _ = _fetch_only(_prs(2))
    with pytest.raises(wr.WorkflowExecutionError, match="compactor_durable_generations_exhausted"):
        _run(_scheduled("github_compactor_pass"), fake)


@pytest.mark.real_durable_submit
def test_past_deadline_redispatch_still_finds_a_completed_run_but_refuses_a_fresh_one(monkeypatch):
    """Day 2026-09-28's deadline (window end + 24h) passed by 2026-10-02. A completed run whose
    terminal row orion-actions missed is still reported; a fresh row is refused."""
    late = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)

    class _Late(datetime):
        @classmethod
        def now(cls, tz=None):
            return late if tz is None else late.astimezone(tz)

    monkeypatch.setattr(wr, "datetime", _Late)
    brief_kwargs = dict(kind="github", workflow_id="github_compactor_pass", window_label="2026-09-28",
                        inputs=[{"items": [{"number": 1}]}], timeout_sec=660.0, session_id="s",
                        finalize={"repo": "acme/widgets"})
    from orion.schemas.compactor_digest_run import CompactorDigestRunBriefV1
    brief = CompactorDigestRunBriefV1(**brief_kwargs)
    past = datetime(2026, 9, 30, 6, 0, tzinfo=timezone.utc)

    async def submit(dispatch):
        monkeypatch.setattr(wr, "dispatch_durable_run", dispatch)
        return await wr._submit_compactor_digest_run(bus=_Bus(), source=ServiceRef(name="o"), correlation_id="c",
                                                     brief=brief, deadline_at=past)

    done = asyncio.run(submit(_Dispatch(statuses=["completed"])))
    assert done["status"] == "completed"
    with pytest.raises(wr.WorkflowExecutionError, match="compactor_window_deadline_passed"):
        asyncio.run(submit(_Dispatch(statuses=["waiting_resource"])))


@pytest.mark.real_durable_submit
def test_already_finalized_redispatch_does_not_notify_twice(monkeypatch, fixed_now, notifications):
    monkeypatch.setattr(wr, "dispatch_durable_run", _Dispatch(statuses=["completed"]))
    fake, _ = _fetch_only(_prs(2))
    result = _run(_scheduled("github_compactor_pass", notify_on="completion"), fake)
    assert result.status == "success" and notifications == []


def test_failed_finalize_does_not_notify_per_retry(monkeypatch, notifications):
    async def no_calls(*args, **kwargs):
        raise AssertionError("no calls")

    async def broken(**kwargs):
        raise wr.WorkflowExecutionError("journal_write_bus_disabled")

    monkeypatch.setattr(wr, "_publish_journal_entry_write_or_fail", broken)
    durable = _durable_result()
    policy = {**durable.finalize["execution_policy"], "notify_on": "failure"}
    req = _req("github_compactor_pass", durable_digest=durable.model_dump(mode="json"), execution_policy=policy)
    with pytest.raises(wr.WorkflowExecutionError):
        _run(req, no_calls)
    assert notifications == []


@pytest.mark.real_durable_submit
def test_github_quiet_day_finalizes_inline_without_a_durable_run(monkeypatch, fixed_now):
    dispatch = _Dispatch()
    monkeypatch.setattr(wr, "dispatch_durable_run", dispatch)
    fake, _ = _fetch_only([])
    bus = _Bus()
    result = _run(_scheduled("github_compactor_pass"), fake, bus)
    assert dispatch.requests == []
    assert result.metadata["workflow"]["status"] == "completed"
    assert len(bus.journal_payloads()) == 1                        # the quiet day is still journaled


def _durable_result(**overrides) -> CompactorDigestResultV1:
    body = "Full day narrative. " + "every PR accounted for; " * 2000
    base = {
        "run_id": "compactor:github_compactor_pass:2026-09-28:acme-widgets:abc123def456",
        "kind": "github", "workflow_id": "github_compactor_pass", "window_label": "2026-09-28",
        "digest": {"card_summary": "Day of merges.", "journal_title": "Repo day", "journal_body": body,
                   "pr_refs": ["#1000", "#1001"]},
        "chunk_count": 3, "merge_mode": "llm_merge", "llm_route": "agent", "gpu_roles": ["agent-gpu2"],
        "attempts": [{"step": "chunk_1_of_3", "ok": False, "error": "empty_completion"},
                     {"step": "chunk_1_of_3", "ok": True, "role": "agent-gpu2"}],
        "finalize": {"repo": "acme/widgets", "window_label": "2026-09-28", "window_mode": "day",
                     "window_start_utc": "2026-09-28T06:00:00+00:00", "window_end_utc": "2026-09-29T05:59:59+00:00",
                     "calendar_date": "2026-09-28", "timezone_name": "America/Denver", "lookback_days": 1,
                     "merged_pr_count": 2, "author": "user-1",
                     "coverage": {"total_count": 2, "covered_count": 2, "input_truncated": False,
                                  "truncated_pr_numbers": [], "fetch_window_unconfirmed": False},
                     "github_pages_fetched": 1, "github_page_cap_hit": False,
                     "execution_policy": {"workflow_id": "github_compactor_pass", "invocation_mode": "immediate",
                                          "notify_on": "failure"}},
    }
    base.update(overrides)
    return CompactorDigestResultV1.model_validate(base)


def _finalize_req(result: CompactorDigestResultV1) -> CortexClientRequest:
    return _req(result.workflow_id, durable_digest=result.model_dump(mode="json"),
                execution_policy=result.finalize.get("execution_policy"))


def test_finalize_path_writes_card_and_untrimmed_journal_without_fetch_or_llm(_cards):
    async def no_calls(*args, **kwargs):
        raise AssertionError("finalize must not fetch or call an LLM")

    bus = _Bus()
    durable = _durable_result()
    result = _run(_finalize_req(durable), no_calls, bus)
    wf = result.metadata["workflow"]
    [journal] = bus.journal_payloads()
    assert journal["body"] == durable.digest["journal_body"]         # untrimmed
    assert journal["entry_id"] == wr.stable_github_compactor_journal_entry_id(
        workflow_id="github_compactor_pass", calendar_date="2026-09-28", repo="acme/widgets")
    assert _cards[0]["merged_pr_count"] == 2 and wf["card_id"]
    assert wf["status"] == "completed" and wf["durable_run_id"] == durable.run_id
    assert wf["digest_merge_mode"] == "llm_merge" and wf["digest_chunk_count"] == 3
    assert wf["digest_gpu_roles"] == ["agent-gpu2"] and len(wf["digest_attempts"]) == 2
    assert wf["total_count"] == wf["covered_count"] == 2 and wf["window_mode"] == "day"
    # The durable digest is not echoed back whole into the reply.
    assert result.metadata["workflow_request"]["durable_digest"] == {"run_id": durable.run_id}


def test_finalize_replay_is_an_upsert_not_a_second_entry():
    async def no_calls(*args, **kwargs):
        raise AssertionError("no calls")

    bus = _Bus()
    durable = _durable_result()
    _run(_finalize_req(durable), no_calls, bus)
    _run(_finalize_req(durable), no_calls, bus)
    ids = {p["entry_id"] for p in bus.journal_payloads()}
    assert len(ids) == 1


def test_finalize_rejects_invalid_or_mismatched_durable_digest():
    async def no_calls(*args, **kwargs):
        raise AssertionError("no calls")

    bad = _req("github_compactor_pass", durable_digest={"run_id": "x"})
    with pytest.raises(wr.WorkflowExecutionError, match="compactor_durable_digest_invalid"):
        _run(bad, no_calls)
    chat = _durable_result(kind="chat", workflow_id="chat_history_compactor_pass",
                           digest={"card_summary": "c", "journal_title": "t", "journal_body": "b", "turn_refs": []})
    with pytest.raises(wr.WorkflowExecutionError, match="workflow_mismatch"):
        _run(_req("github_compactor_pass", durable_digest=chat.model_dump(mode="json")), no_calls)


def test_chat_finalize_rebuilds_window_and_writes_indexed_card(_cards):
    async def no_calls(*args, **kwargs):
        raise AssertionError("no calls")

    durable = _durable_result(
        kind="chat", workflow_id="chat_history_compactor_pass",
        digest={"card_summary": "Chat day.", "journal_title": "Chat day", "journal_body": "We talked. " * 3000,
                "turn_refs": ["corr-1"]},
        finalize={"window": {"mode": "day", "compactor_index": "chat_compactor:day:2026-09-28",
                             "window_start": "2026-09-28T06:00:00+00:00", "window_end": "2026-09-29T05:59:59+00:00",
                             "lookback_seconds": 86399, "lookback_hours": None, "calendar_date": "2026-09-28",
                             "timezone_name": "America/Denver"},
                  "window_label": "2026-09-28", "turn_count": 120, "selection_strategy": "time_bound_recent_n",
                  "coverage": {"total_count": 120, "covered_count": 120, "input_truncated": False},
                  "author": "user-1"})
    bus = _Bus()
    result = _run(_finalize_req(durable), no_calls, bus)
    wf = result.metadata["workflow"]
    assert _cards[0]["window"].compactor_index == "chat_compactor:day:2026-09-28"
    assert _cards[0]["turn_count"] == 120
    [journal] = bus.journal_payloads()
    assert journal["body"] == durable.digest["journal_body"]
    assert wf["turn_count"] == 120 and wf["total_count"] == 120 and wf["durable_run_id"] == durable.run_id
