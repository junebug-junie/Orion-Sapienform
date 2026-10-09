"""Unified-turn latency L4: stance_context_prepare (consumer side, cortex-exec).

docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md, L4.

The stance context build runs once per turn: either in the prepare (overlapping
orion-mind) or, when the prepare failed / never arrived, inline in stance_react.
Never both -- a second build writes a second chat_stance_belief_log row, a
second cortex_turn attention row and makes a second chat-lane probe call.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from typing import Any, Dict

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.cognition.plan_loader import build_plan_for_verb
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.stance_context_prepare import (
    STANCE_CONTEXT_PREPARE_REQUEST_KIND,
    STANCE_CONTEXT_PREPARE_RESULT_KIND,
    STANCE_PREPARE_REQUESTED_CTX_KEY,
    StanceContextPrepareRequestV1,
    StanceContextPrepareResultV1,
    stance_context_prepare_channel,
)

# Some suites in this directory delete and re-import ``app.*``. Both sides of
# the seam resolve each other through sys.modules at call time, so the modules
# are looked up per test, not bound at import.
sp: Any = None


async def prepare_brain_reply_context(ctx: Dict[str, Any]) -> Any:
    return await importlib.import_module("app.executor").prepare_brain_reply_context(ctx)


CORR = "7f0c2a52-6a8f-4b7e-9d1f-2b0e0e6c1a11"
SOURCE = ServiceRef(name="orion-thought", node="test", version="1.0")


class FakeBuild:
    """Stands in for build_chat_stance_inputs. Counts builds and the side-effect
    rows a real build writes (belief log, cortex_turn attention row)."""

    def __init__(self, *, delay: float = 0.0, fail_first: bool = False) -> None:
        self.delay = delay
        self.fail_first = fail_first
        self.calls = 0
        self.rows: list[tuple[str, str]] = []
        self.verbs: list[Any] = []

    async def __call__(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        self.verbs.append(ctx.get("verb"))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_first and self.calls == 1:
            raise RuntimeError("falkor_down")
        corr = str(ctx.get("correlation_id"))
        self.rows.append(("chat_stance_belief_log", corr))
        self.rows.append(("cortex_turn", corr))
        ctx.setdefault("debug", {})["stance_build"] = "done"
        ctx["chat_reasoning_summary"] = {"summary": "s"}
        ctx["autonomy_slice"] = {"slice": 1}
        inputs = {"identity": {"orion": ["o"]}, "user_message_seen": ctx.get("user_message")}
        ctx["chat_stance_inputs"] = inputs
        return inputs


@pytest.fixture(autouse=True)
def _modules(monkeypatch):
    global sp
    sp = importlib.import_module("app.stance_prepare")
    monkeypatch.setattr(sp, "_CACHE", sp.StancePrepareCache())
    yield


@pytest.fixture
def fake_build(monkeypatch):
    build = FakeBuild()
    executor = importlib.import_module("app.executor")
    monkeypatch.setattr(executor, "build_chat_stance_inputs", build)
    monkeypatch.setattr(
        executor, "_inject_identity_context", lambda ctx: ctx.setdefault("orion_identity_summary", ["o"])
    )
    return build


def _plan_request(**context: Any) -> PlanExecutionRequest:
    ctx = {
        "session_id": "sess-1",
        "user_message": "where is our work heading?",
        "metadata": {"correlation_id": CORR, "mode": "brain"},
        "debug": {"from_request": True},
    }
    ctx.update(context)
    return PlanExecutionRequest(
        plan=build_plan_for_verb("stance_react", mode="brain"),
        args=PlanExecutionArgs(request_id=CORR, trigger_source="orion-thought", extra={"mode": "brain"}),
        context=ctx,
    )


def _prepare_env() -> tuple[BaseEnvelope, StanceContextPrepareRequestV1]:
    req = StanceContextPrepareRequestV1(correlation_id=CORR, plan_request=_plan_request())
    env = BaseEnvelope(
        kind=STANCE_CONTEXT_PREPARE_REQUEST_KIND,
        source=SOURCE,
        correlation_id=CORR,
        reply_to="orion:cortex:exec:stance_prepare_result:abc",
        payload=req.model_dump(mode="json"),
    )
    return env, req


def _stance_ctx(*, requested: bool = True) -> Dict[str, Any]:
    """What main.handle + the router build for the stance_react request."""
    env, req = _prepare_env()
    ctx = sp.build_prepare_ctx(env, req)
    ctx["mind_coloring"] = {"items": ["x"]}
    ctx["debug"] = {"from_request": True, "recall_profile": "stance"}
    if requested:
        ctx[STANCE_PREPARE_REQUESTED_CTX_KEY] = True
    return ctx


async def _prepare() -> StanceContextPrepareResultV1:
    env, req = _prepare_env()
    return await sp.run_stance_context_prepare(env, req)


# --- one build per turn ------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_then_stance_builds_once(fake_build) -> None:
    result = await _prepare()
    assert result.status == "ready" and result.build_ms is not None

    ctx = _stance_ctx()
    inputs = await prepare_brain_reply_context(ctx)

    assert fake_build.calls == 1
    assert inputs == ctx["chat_stance_inputs"]
    assert inputs["user_message_seen"] == "where is our work heading?"
    # Everything the build wrote to ctx crossed over; the request's own keys survived.
    assert ctx["chat_reasoning_summary"] == {"summary": "s"}
    assert ctx["autonomy_slice"] == {"slice": 1}
    assert ctx["debug"] == {"from_request": True, "recall_profile": "stance", "stance_build": "done"}
    assert ctx["mind_coloring"] == {"items": ["x"]}
    assert ctx["correlation_id"] == CORR
    assert ctx["stance_prepare_overlap"]["outcome"] == "used"
    assert STANCE_PREPARE_REQUESTED_CTX_KEY not in ctx
    # Side-effect rows: exactly once for this turn.
    assert fake_build.rows == [("chat_stance_belief_log", CORR), ("cortex_turn", CORR)]
    # The probe and the attention frame read ctx["verb"].
    assert fake_build.verbs == ["stance_react"]


@pytest.mark.asyncio
async def test_stance_awaits_an_in_flight_prepare_instead_of_building(fake_build) -> None:
    fake_build.delay = 0.3
    prepare_task = asyncio.create_task(_prepare())
    await asyncio.sleep(0.05)  # prepare registered, build running

    ctx = _stance_ctx()
    started = time.perf_counter()
    await prepare_brain_reply_context(ctx)
    waited = time.perf_counter() - started
    assert (await prepare_task).status == "ready"

    assert fake_build.calls == 1
    assert len(fake_build.rows) == 2
    assert 0.15 < waited < 1.5
    assert ctx["stance_prepare_overlap"]["outcome"] == "used"
    assert ctx["stance_prepare_overlap"]["wait_ms"] > 100


@pytest.mark.asyncio
async def test_prepare_arriving_after_stance_is_waited_for(fake_build) -> None:
    ctx = _stance_ctx()
    stance_task = asyncio.create_task(prepare_brain_reply_context(ctx))
    await asyncio.sleep(0.3)  # stance is waiting; prepare not here yet
    assert not stance_task.done()
    assert (await _prepare()).status == "ready"
    await stance_task

    assert fake_build.calls == 1
    assert ctx["stance_prepare_overlap"]["outcome"] == "used"


@pytest.mark.asyncio
async def test_failed_prepare_falls_back_to_inline_build(fake_build) -> None:
    fake_build.fail_first = True
    result = await _prepare()
    assert result.status == "failed" and "falkor_down" in (result.error or "")

    ctx = _stance_ctx()
    inputs = await prepare_brain_reply_context(ctx)

    assert fake_build.calls == 2  # the failed prepare, then inline
    assert isinstance(inputs, dict)
    assert ctx["stance_prepare_overlap"]["outcome"] == "failed"
    assert fake_build.rows == [("chat_stance_belief_log", CORR), ("cortex_turn", CORR)]


@pytest.mark.asyncio
async def test_absent_prepare_builds_inline_after_the_wait_and_a_late_prepare_does_not_build(
    fake_build, monkeypatch
) -> None:
    monkeypatch.setattr(sp, "ABSENT_WAIT_SEC", 0.2)
    ctx = _stance_ctx()
    started = time.perf_counter()
    await prepare_brain_reply_context(ctx)
    assert 0.15 < time.perf_counter() - started < 1.0
    assert ctx["stance_prepare_overlap"]["outcome"] == "absent"
    assert fake_build.calls == 1

    late = await _prepare()
    assert late.status == "duplicate" and late.error == "abandoned"
    assert fake_build.calls == 1
    assert len(fake_build.rows) == 2


def test_absent_wait_is_two_seconds() -> None:
    assert sp.ABSENT_WAIT_SEC == 2.0
    assert sp.CACHE_TTL_SEC == 120.0


@pytest.mark.asyncio
async def test_no_marker_means_no_wait_and_no_cache_lookup(fake_build) -> None:
    ctx = _stance_ctx(requested=False)
    started = time.perf_counter()
    await prepare_brain_reply_context(ctx)
    assert time.perf_counter() - started < 0.1
    assert fake_build.calls == 1
    assert "stance_prepare_overlap" not in ctx


# --- cache: consume once, expire ---------------------------------------------


@pytest.mark.asyncio
async def test_cache_entry_is_consumed_once(fake_build) -> None:
    await _prepare()
    first = await sp.get_cache().take(CORR, absent_wait_sec=0.01)
    second = await sp.get_cache().take(CORR, absent_wait_sec=0.01)
    assert first.outcome == "used" and first.delta is not None
    assert second.outcome == "consumed" and second.delta is None
    # A duplicate prepare for a consumed turn does not build again.
    assert (await _prepare()).status == "duplicate"
    assert fake_build.calls == 1


@pytest.mark.asyncio
async def test_cache_entry_expires_after_ttl(fake_build, monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(sp, "_CACHE", sp.StancePrepareCache(clock=lambda: now[0]))
    await _prepare()
    assert len(sp.get_cache()) == 1
    now[0] += sp.CACHE_TTL_SEC + 1
    taken = await sp.get_cache().take(CORR, absent_wait_sec=0.01)
    assert taken.outcome == "absent"
    assert len(sp.get_cache()) == 0


# --- RPC handler + contract --------------------------------------------------


@pytest.mark.asyncio
async def test_handler_replies_with_registered_result_kind(fake_build) -> None:
    env, _ = _prepare_env()
    out = await sp.handle_stance_context_prepare(env)
    assert out is not None
    assert out.kind == STANCE_CONTEXT_PREPARE_RESULT_KIND
    result = StanceContextPrepareResultV1.model_validate(out.payload)
    assert result.status == "ready" and result.correlation_id == CORR


@pytest.mark.asyncio
async def test_handler_rejects_invalid_payload_without_building(fake_build) -> None:
    env = BaseEnvelope(
        kind=STANCE_CONTEXT_PREPARE_REQUEST_KIND,
        source=SOURCE,
        correlation_id=CORR,
        reply_to="orion:cortex:exec:stance_prepare_result:abc",
        payload={"correlation_id": CORR},
    )
    out = await sp.handle_stance_context_prepare(env)
    assert StanceContextPrepareResultV1.model_validate(out.payload).status == "failed"
    assert fake_build.calls == 0


@pytest.mark.parametrize(
    "exec_channel,expected",
    [
        ("orion:cortex:exec:request", "orion:cortex:exec:stance_prepare"),
        ("orion:cortex:exec:request:chat", "orion:cortex:exec:stance_prepare:chat"),
        ("orion:cortex:exec:request:background", "orion:cortex:exec:stance_prepare:background"),
        ("orion:something:else", None),
        ("orion:cortex:exec:request:a:b", None),
    ],
)
def test_prepare_channel_follows_the_exec_lane(exec_channel, expected) -> None:
    assert stance_context_prepare_channel(exec_channel) == expected


def test_prepare_ctx_matches_the_exec_ctx_and_sets_the_verb() -> None:
    env, req = _prepare_env()
    ctx = sp.build_prepare_ctx(env, req)
    assert ctx["verb"] == "stance_react"
    assert ctx["mode"] == "brain"
    assert ctx["correlation_id"] == CORR
    assert ctx["user_message"] == "where is our work heading?"
    assert ctx["plan_metadata"] == (req.plan_request.plan.metadata or {})
    assert STANCE_PREPARE_REQUESTED_CTX_KEY not in ctx


# --- review follow-ups -------------------------------------------------------


def _bare_plan_request() -> PlanExecutionRequest:
    """orion-thought's real shape: no ``debug`` in the request context."""
    return PlanExecutionRequest(
        plan=build_plan_for_verb("stance_react", mode="brain"),
        args=PlanExecutionArgs(request_id=CORR, trigger_source="orion-thought", extra={"mode": "brain"}),
        context={"session_id": "sess-1", "user_message": "where is our work heading?", "mode": "brain"},
    )


@pytest.mark.asyncio
async def test_router_path_keeps_its_own_debug_keys_and_builds_once(fake_build, monkeypatch) -> None:
    """Drives the real main-ctx builder + PlanRunner.run_plan brain hook; only the
    build and the LLM step are faked. The build adds a brand-new ``debug`` dict in
    the prepare ctx; the router's own debug.recall_* entries must survive."""
    from unittest.mock import AsyncMock

    from orion.schemas.cortex.schemas import StepExecutionResult

    router = importlib.import_module("app.router")
    exec_ctx = importlib.import_module("app.exec_ctx")
    plan_request = _bare_plan_request()

    env = BaseEnvelope(
        kind=STANCE_CONTEXT_PREPARE_REQUEST_KIND,
        source=SOURCE,
        correlation_id=CORR,
        reply_to="orion:cortex:exec:stance_prepare_result:abc",
        payload=StanceContextPrepareRequestV1(correlation_id=CORR, plan_request=plan_request).model_dump(mode="json"),
    )
    assert (await sp.handle_stance_context_prepare(env)) is not None

    stance_context = dict(plan_request.context)
    stance_context[STANCE_PREPARE_REQUESTED_CTX_KEY] = True
    stance_req = plan_request.model_copy(update={"context": stance_context})
    ctx = exec_ctx.build_exec_ctx(
        payload_context=stance_context, req=stance_req, trace_id=CORR, parent_event_id=None, corr_id=CORR
    )
    monkeypatch.setattr(
        router,
        "call_step_services",
        AsyncMock(
            return_value=StepExecutionResult(
                status="success",
                verb_name="stance_react",
                step_name="llm_stance_react",
                order=0,
                result={"LLMGatewayService": {"content": "{}"}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            )
        ),
    )
    monkeypatch.setattr(router, "assemble_stance_grounding", AsyncMock(return_value=None))
    res = await router.PlanRunner().run_plan(
        bus=object(), source=SOURCE, req=stance_req, correlation_id=CORR, ctx=ctx
    )

    assert fake_build.calls == 1
    assert ctx["debug"]["stance_build"] == "done"
    assert "recall_gating_reason" in ctx["debug"] and "recall_profile_source" in ctx["debug"]
    assert ctx["stance_prepare_overlap"]["outcome"] == "used"
    assert (res.metadata or {}).get("stance_prepare_overlap", {}).get("outcome") == "used"


@pytest.mark.asyncio
async def test_non_canonical_payload_correlation_id_still_hits(fake_build) -> None:
    req = StanceContextPrepareRequestV1(correlation_id=CORR.upper(), plan_request=_plan_request())
    env = BaseEnvelope(
        kind=STANCE_CONTEXT_PREPARE_REQUEST_KIND,
        source=SOURCE,
        correlation_id=CORR.upper(),
        payload=req.model_dump(mode="json"),
    )
    assert (await sp.run_stance_context_prepare(env, req)).status == "ready"
    ctx = _stance_ctx()  # correlation_id is the normalized lowercase form
    await prepare_brain_reply_context(ctx)
    assert fake_build.calls == 1
    assert ctx["stance_prepare_overlap"]["outcome"] == "used"


def test_nested_in_place_mutation_is_transferred() -> None:
    before_ctx = {"options": {"a": {"b": 1}}, "keep": 1}
    before = sp._snapshot(before_ctx)
    before_ctx["options"]["a"]["b"] = 2
    delta = sp.compute_ctx_delta(before, before_ctx)
    target = {"options": {"a": {"b": 1}, "own": True}, "keep": 1}
    sp.apply_ctx_delta(target, delta)
    assert target["options"]["a"]["b"] == 2
    assert target["options"]["own"] is True
