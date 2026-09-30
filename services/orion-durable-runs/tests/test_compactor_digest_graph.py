"""compactor.digest: a compactor day's LLM digest calls, checkpointed one call per node run, under a
GPU pool hold.

Real LangGraph interrupts + checkpointer around the graph, with the fake pool from
test_admitted_graph (queued until ``grant()``) and a fake cortex-orch that answers the digest verb
and the finalize workflow request. Proves: a busy pool is a checkpointed wait, never an attempt; a
restart resumes at the first undigested chunk; a failed call is a bounded attempt for THAT call; a
merge that fails or drops refs falls back to the join; a failed finalize is retried without
re-running any LLM call; the run_id/brief contract.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from test_admitted_graph import CFG, World
from app.admitted_graph import AdmissionDeps, HoldLost
from app.compactor_digest_graph import build_compactor_digest_graph, finish_detail
from orion.cognition.compactor import map_reduce
from orion.cognition.github_compactor.digest import build_github_compactor_digest_inputs
from orion.schemas.compactor_digest_run import (
    COMPACTOR_DIGEST_WORKFLOW, CompactorDigestResultV1, CompactorDigestRunBriefV1,
)
from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.durable_run import DurableRunRequestV1


class Crash(BaseException):
    """A process dying mid-node: not an Exception, so no graph handler swallows it."""


def _prs(n: int) -> list[dict]:
    return [{"number": 1000 + i, "title": f"PR {i}", "body": (f"PR {1000 + i}. " + "detail " * 1300),
             "merged_at": "2026-09-28T18:00:00Z"} for i in range(n)]


def _brief(n_prs: int = 40, **overrides) -> dict:
    inputs, coverage = build_github_compactor_digest_inputs({"repo": "acme/widgets", "items": _prs(n_prs),
                                                             "window_mode": "day", "calendar_date": "2026-09-28"})
    brief = {"kind": "github", "workflow_id": "github_compactor_pass", "window_label": "2026-09-28",
             "inputs": inputs, "timeout_sec": 5.0, "session_id": "orion-actions",
             "finalize": {"repo": "acme/widgets", "coverage": coverage,
                          "execution_policy": {"workflow_id": "github_compactor_pass", "notify_on": "failure"}}}
    brief.update(overrides)
    return CompactorDigestRunBriefV1.model_validate(brief).model_dump(mode="json")


def initial(**brief_overrides) -> dict:
    return {"run_id": "study-001", "correlation_id": "trace-001", "workflow": COMPACTOR_DIGEST_WORKFLOW,
            "attempt": 0, "admission": {"resource": "llm.route.agent"}, "brief": _brief(**brief_overrides)}


LONG_BODY = "Merged narrative. " + "every PR accounted for; " * 2000   # ~48k chars: stored whole


class Orch:
    """Fake cortex-orch. ``fail`` maps a call label to how many times it fails first."""

    def __init__(self, world, fail=None, merge="ok", finalize_ok=None, crash_on=None):
        self.world = world
        self.fail = dict(fail or {})
        self.merge = merge
        self.finalize_ok = list(finalize_ok or [True])
        self.crash_on = crash_on
        self.calls: list[str] = []
        self.finalized: list[CompactorDigestResultV1] = []

    async def rpc(self, payload, *, timeout_sec, label):
        request = CortexClientRequest.model_validate(payload)        # what orch would validate
        if label == "finalize":
            wr = request.context.metadata["workflow_request"]
            assert "durable_run" not in request.context.metadata and wr["workflow_id"] == "github_compactor_pass"
            self.finalized.append(CompactorDigestResultV1.model_validate(wr["durable_digest"]))
            ok = self.finalize_ok.pop(0) if self.finalize_ok else True
            if not ok:
                return {"ok": False, "status": "fail", "error": {"message": "journal_write_bus_disabled"}}
            return {"ok": True, "status": "success", "metadata": {"workflow": {
                "card_id": "card-1", "journal_body_chars": len(LONG_BODY),
                "journal_entry": {"entry_id": "entry-2026-09-28"}}}}
        # A digest call: carries the run's hold, the agent route, never a workflow re-entry.
        assert request.verb == "github_compactor_digest_v1"
        assert request.options["gpu_lease"] == self.world.ref() and request.options["llm_route"] == "agent"
        assert "workflow_request" not in request.context.metadata
        self.calls.append(label)
        if label == self.crash_on:
            self.crash_on = None
            raise Crash()
        if self.fail.get(label, 0) > 0:
            self.fail[label] -= 1
            return {"ok": True, "status": "success", "final_text": "", "metadata": {}}   # empty completion
        gi = request.context.metadata["github_compactor_input"]
        if "partial_digests" in gi:
            refs = [r for part in gi["partial_digests"] for r in part["pr_refs"]]
            if self.merge == "drop_refs":
                refs = refs[:1]
            digest = {"card_summary": "Day of merges.", "journal_title": "Repo day", "journal_body": LONG_BODY,
                      "pr_refs": refs}
        else:
            refs = [f"#{item['number']}" for item in gi["items"]]
            digest = {"card_summary": "part", "journal_title": "part", "journal_body": " ".join(refs), "pr_refs": refs}
        return {"ok": True, "status": "success", "final_text": "", "metadata": {"github_compactor_digest": digest}}


def graph(world, saver, orch, max_attempts=3):
    return build_compactor_digest_graph(
        orch.rpc,
        AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                      now=lambda: world.now, max_attempts=max_attempts),
        saver)


def _chunks() -> int:
    return len(_brief()["inputs"])


def test_busy_pool_waits_checkpointed_then_every_chunk_and_the_merge_run_under_the_hold():
    async def run():
        world, saver = World(), InMemorySaver()
        orch = Orch(world)
        await asyncio.wait_for(graph(world, saver, orch).ainvoke(initial(), CFG), 1)
        snap = await graph(world, saver, orch).aget_state(CFG)
        assert snap.next == ("resource_wait",) and orch.calls == [] and snap.values.get("attempt", 0) == 0
        # Still busy after a "restart": asking again is a wait, never an attempt.
        await graph(world, saver, orch).ainvoke(Command(resume={}), CFG)
        assert (await graph(world, saver, orch).aget_state(CFG)).values.get("attempt", 0) == 0
        world.grant()
        result = await graph(world, saver, orch).ainvoke(Command(resume={}), CFG)
        n = _chunks()
        assert n > 1
        assert orch.calls == [f"chunk_{i}_of_{n}" for i in range(1, n + 1)] + ["merge"]
        assert result["status"] == "completed" and world.releases == ["completed"]
        [final] = orch.finalized
        assert final.merge_mode == "llm_merge" and final.chunk_count == n
        assert final.digest["journal_body"] == LONG_BODY                  # untrimmed
        assert final.gpu_roles == ["agent-gpu2"] and all(a["ok"] for a in final.attempts)
        assert final.finalize["repo"] == "acme/widgets"                    # echoed verbatim
        detail = finish_detail(result)
        assert detail["journal_entry_id"] == "entry-2026-09-28" and detail["card_id"] == "card-1"
        assert detail["merge_mode"] == "llm_merge" and detail["chunk_count"] == n

    asyncio.run(run())


def test_restart_mid_day_resumes_at_the_first_undigested_chunk():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        n = _chunks()
        orch = Orch(world, crash_on=f"chunk_2_of_{n}")
        with pytest.raises(Crash):
            await graph(world, saver, orch).ainvoke(initial(), CFG)
        snap = await graph(world, saver, orch).aget_state(CFG)
        assert snap.next == ("digest",) and len(snap.values["partials"]) == 1   # chunk 1 checkpointed
        result = await graph(world, saver, orch).ainvoke(None, CFG)
        assert orch.calls.count(f"chunk_1_of_{n}") == 1                    # never re-digested
        assert orch.calls.count(f"chunk_2_of_{n}") == 2                    # the interrupted call, replayed
        assert result["status"] == "completed"

    asyncio.run(run())


def test_failed_call_is_a_bounded_attempt_for_that_call_and_resets_after_success():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        n = _chunks()
        orch = Orch(world, fail={f"chunk_1_of_{n}": 1, f"chunk_2_of_{n}": 1})
        await graph(world, saver, orch, max_attempts=2).ainvoke(initial(), CFG)
        for _ in range(4):   # each failure hands the hold back and waits for it again
            snap = await graph(world, saver, orch).aget_state(CFG)
            if not snap.next:
                break
            world.grant()
            await graph(world, saver, orch, max_attempts=2).ainvoke(Command(resume={}), CFG)
        state = (await graph(world, saver, orch).aget_state(CFG)).values
        # Two failures, but each on a different call: max_attempts=2 was never exhausted.
        assert state["status"] == "completed"
        assert [a["ok"] for a in state["call_log"][:4]] == [False, True, False, True]
        assert "empty_completion" in state["call_log"][0]["error"]
        assert world.releases.count("attempt_failed") == 2

    asyncio.run(run())


def test_chunk_that_exhausts_its_attempts_fails_the_run_without_finalize():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        n = _chunks()
        orch = Orch(world, fail={f"chunk_1_of_{n}": 99})
        await graph(world, saver, orch, max_attempts=2).ainvoke(initial(), CFG)
        world.grant()
        result = await graph(world, saver, orch, max_attempts=2).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "failed" and "empty_completion" in result["last_error"]
        assert orch.finalized == []

    asyncio.run(run())


def test_merge_that_exhausts_its_attempts_joins_the_chunk_digests_and_finishes():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        orch = Orch(world, fail={"merge": 99})
        await graph(world, saver, orch, max_attempts=2).ainvoke(initial(), CFG)
        world.grant()
        result = await graph(world, saver, orch, max_attempts=2).ainvoke(Command(resume={}), CFG)
        assert result["status"] == "completed"
        [final] = orch.finalized
        assert final.merge_mode == "concatenated" and final.merge_skipped_reason.startswith("merge_failed:")
        assert all(f"#{1000 + i}" in final.digest["journal_body"] for i in range(40))

    asyncio.run(run())


def test_merge_that_drops_refs_falls_back_to_the_join():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        orch = Orch(world, merge="drop_refs")
        result = await graph(world, saver, orch).ainvoke(initial(), CFG)
        assert result["status"] == "completed"
        [final] = orch.finalized
        assert final.merge_mode == "concatenated" and final.merge_skipped_reason.startswith("merge_dropped_refs:")
        assert final.attempts[-1]["error"].startswith("refs_missing:")

    asyncio.run(run())


def test_merge_input_over_budget_makes_no_merge_call(monkeypatch):
    monkeypatch.setattr(map_reduce, "DIGEST_INPUT_CHAR_BUDGET", 50)

    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        orch = Orch(world)
        result = await graph(world, saver, orch).ainvoke(initial(), CFG)
        assert "merge" not in orch.calls and result["status"] == "completed"
        assert orch.finalized[0].merge_skipped_reason == "merge_input_over_budget"

    asyncio.run(run())


def test_single_chunk_day_is_one_call():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        orch = Orch(world)
        result = await graph(world, saver, orch).ainvoke(initial(n_prs=1), CFG)
        assert orch.calls == ["single"] and result["status"] == "completed"
        assert orch.finalized[0].merge_mode == "single"

    asyncio.run(run())


def test_failed_finalize_is_retried_without_rerunning_any_llm_call():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        orch = Orch(world, finalize_ok=[False, True])
        with pytest.raises(RuntimeError, match="compactor_finalize_failed"):
            await graph(world, saver, orch).ainvoke(initial(), CFG)
        calls_before = list(orch.calls)
        assert (await graph(world, saver, orch).aget_state(CFG)).next == ("finalize",)
        result = await graph(world, saver, orch).ainvoke(None, CFG)       # the driver's checkpoint resume
        assert result["status"] == "completed" and orch.calls == calls_before
        assert len(orch.finalized) == 2 and orch.finalized[0].digest == orch.finalized[1].digest

    asyncio.run(run())


def test_lost_hold_mid_call_is_not_an_attempt():
    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()

        async def execute(state, node):
            world.pool_status = "queued"
            raise HoldLost("gpu_hold_lost")

        world.execute = execute
        result = await graph(world, saver, Orch(world)).ainvoke(initial(), CFG)
        assert result["status"] == "waiting_resource" and result.get("attempt", 0) == 0
        assert not result.get("call_log")

    asyncio.run(run())


# ------------------------------------------------------------------ contract

def test_request_requires_admission_and_matching_brief():
    brief = _brief(n_prs=1)
    with pytest.raises(ValidationError, match="require durable resource admission"):
        DurableRunRequestV1.model_validate({"run_id": "compactor:x:1", "workflow": "compactor.digest",
                                            "correlation_id": "c", "brief": brief})
    ok = DurableRunRequestV1.model_validate({"run_id": "compactor:x:1", "workflow": "compactor.digest",
                                             "correlation_id": "c", "brief": brief, "admission": {}})
    assert isinstance(ok.brief, CompactorDigestRunBriefV1)
    with pytest.raises(ValidationError):
        DurableRunRequestV1.model_validate({"run_id": "compactor:x:1", "workflow": "reading.turn",
                                            "correlation_id": "c", "brief": brief, "admission": {}})


def test_brief_refuses_reserved_routes_and_kind_mismatch():
    with pytest.raises(ValidationError, match="reserved interactive lane"):
        _brief(n_prs=1, llm_route="chat")
    with pytest.raises(ValidationError, match="must agree"):
        _brief(n_prs=1, kind="chat")


def test_schemas_registered():
    from orion.schemas.registry import resolve

    assert resolve("CompactorDigestRunBriefV1") is CompactorDigestRunBriefV1
    assert resolve("CompactorDigestResultV1") is CompactorDigestResultV1


def test_admission_runtime_registers_the_graph():
    from app import admission_runtime as ar

    assert ar.WORK_NODES[COMPACTOR_DIGEST_WORKFLOW] == {"digest"}


def test_heavy_day_many_chunks_completes_under_the_runtime_config():
    """digest loops once per call with no interrupt between; the driver's own config
    (AdmissionRuntime.config -> LangGraph's default recursion limit, 10007 in 1.2.11) must let a
    heavy day's dozens of calls through in one invocation."""
    from app import admission_runtime as ar

    async def run():
        world, saver = World(), InMemorySaver()
        world.grant()
        orch = Orch(world)
        many = _brief(n_prs=1)
        many["inputs"] = [{"items": [{"number": 2000 + i}], "chunk_index": i, "chunk_count": 40} for i in range(40)]
        state = {**initial(), "brief": many}
        cfg = ar.AdmissionRuntime.config(CFG["configurable"]["thread_id"])

        async def rpc(payload, *, timeout_sec, label):
            if label == "finalize":
                return await orch.rpc(payload, timeout_sec=timeout_sec, label=label)
            gi = payload["context"]["metadata"]["github_compactor_input"]
            orch.calls.append(label)
            refs = [f"#{i['number']}" for i in gi.get("items", [])] or \
                [r for p in gi.get("partial_digests", []) for r in p["pr_refs"]]
            return {"ok": True, "metadata": {"github_compactor_digest": {
                "card_summary": "c", "journal_title": "t", "journal_body": " ".join(refs), "pr_refs": refs}}}

        g = build_compactor_digest_graph(rpc, AdmissionDeps(world.register, world.lease, world.execute,
                                                            world.release, world.event, now=lambda: world.now),
                                         saver)
        result = await g.ainvoke(state, cfg)
        assert result["status"] == "completed" and len(orch.calls) >= 40

    asyncio.run(run())


def test_finalize_honours_pause_but_not_a_passed_deadline():
    from app.admitted_graph import RunControlPending, WorkflowDeadline

    async def run(guard_exc):
        world, saver = World(), InMemorySaver()
        world.grant()
        orch = Orch(world)

        async def guard(state):
            raise guard_exc

        g = build_compactor_digest_graph(orch.rpc, AdmissionDeps(
            world.register, world.lease, world.execute, world.release, world.event,
            now=lambda: world.now, guard=guard), saver)
        try:
            result = await g.ainvoke(initial(n_prs=1), CFG)
        except RunControlPending:
            return None, orch
        return result, orch

    paused, orch = asyncio.run(run(RunControlPending("paused")))
    assert paused is None and orch.finalized == []                 # no card/journal while paused
    late, orch = asyncio.run(run(WorkflowDeadline("workflow_deadline")))
    assert late["status"] == "completed" and len(orch.finalized) == 1   # the day's work is not thrown away
