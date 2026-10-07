"""Unified-turn latency L1-L3 (docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md).

L1: finalize/repair verbs skip the brain reply-context build at both the router
    and the step-time check, and the skip flag is read from ctx and options.
L2: the synchronous half of the stance build runs on a dedicated worker, so the
    event loop keeps answering other handlers during a build.
L3: identity_yaml/self_definition/autonomy are not write-through; the signal
    probe tells the gateway its real wait budget.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import app.chat_stance as chat_stance_module
from app.executor import (
    BRAIN_REPLY_CONTEXT_SKIP_VERBS,
    _should_prepare_brain_reply_context,
    brain_reply_context_skipped,
    call_step_services,
    prepare_brain_reply_context,
)
from app.router import PlanRunner
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.schemas.cortex.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanExecutionArgs,
    PlanExecutionRequest,
    StepExecutionResult,
)

FINALIZE_VERBS = ("harness_finalize_reflect", "orion_response_repair")

# Some suites in this directory delete and re-import ``app.*`` modules, so a
# dotted-path monkeypatch ("app.executor.x") can hit a different module object
# than the functions imported above. Patch the globals those functions actually
# resolve names in.
EXECUTOR_NS = call_step_services.__globals__
ROUTER_NS = PlanRunner.run_plan.__globals__
STANCE_NS = EXECUTOR_NS["build_chat_stance_inputs"].__globals__
SOURCE = ServiceRef(name="test", node="test", version="1.0")


# ---------------------------------------------------------------------------
# L1 -- one skip helper for both paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", FINALIZE_VERBS + ("introspect_spark", "memory_graph_suggest"))
def test_skip_helper_covers_finalize_and_repair_verbs(verb: str) -> None:
    assert verb in BRAIN_REPLY_CONTEXT_SKIP_VERBS
    assert brain_reply_context_skipped(verb, {"mode": "brain"}) is True
    assert brain_reply_context_skipped(verb.upper(), {"mode": "brain"}) is True


def test_skip_helper_reads_flag_from_ctx_and_from_options() -> None:
    assert brain_reply_context_skipped("stance_react", {}) is False
    assert brain_reply_context_skipped("stance_react", {"skip_brain_reply_context": True}) is True
    # options passed explicitly (router) and read from ctx["options"] (step time)
    assert brain_reply_context_skipped("stance_react", {}, {"skip_brain_reply_context": True}) is True
    assert brain_reply_context_skipped("stance_react", {"options": {"skip_brain_reply_context": True}}) is True


@pytest.mark.parametrize("verb", FINALIZE_VERBS)
def test_step_time_check_skips_finalize_verbs(verb: str) -> None:
    step = ExecutionStep(verb_name=verb, step_name=f"llm_{verb}", order=0, services=["LLMGatewayService"])
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is False


def test_step_time_check_still_prepares_stance_react() -> None:
    step = ExecutionStep(verb_name="stance_react", step_name="llm_stance_react", order=0, services=["LLMGatewayService"])
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is True


def _brain_ctx() -> dict:
    return {
        "mode": "brain",
        "messages": [{"role": "user", "content": "hello"}],
        "raw_user_text": "hello",
        "session_id": "s-test",
    }


async def _run_step(verb: str, ctx: dict, content: str = '{"verdict":"aligned"}'):
    step = ExecutionStep(
        step_name=f"llm_{verb}",
        verb_name=verb,
        services=["LLMGatewayService"],
        order=0,
        prompt_template="{{ raw_user_text }}",
    )
    with patch.object(
        EXECUTOR_NS["LLMGatewayClient"],
        "chat",
        new=AsyncMock(return_value=ChatResponsePayload(content=content)),
    ):
        return await call_step_services(
            bus=MagicMock(), source=SOURCE, step=step, ctx=ctx, correlation_id=str(uuid4())
        )


@pytest.mark.parametrize("verb", FINALIZE_VERBS)
def test_step_time_path_never_calls_build_chat_stance_inputs(monkeypatch, verb: str) -> None:
    build = AsyncMock(return_value={})
    monkeypatch.setitem(EXECUTOR_NS, "build_chat_stance_inputs", build)
    result = asyncio.run(_run_step(verb, _brain_ctx()))
    assert result.status == "success"
    assert build.await_count == 0


def test_step_time_path_still_builds_for_stance_react(monkeypatch) -> None:
    """Positive control: the same harness does reach the build for stance_react."""
    build = AsyncMock(return_value={})
    monkeypatch.setitem(EXECUTOR_NS, "build_chat_stance_inputs", build)
    asyncio.run(_run_step("stance_react", _brain_ctx(), content='{"imperative":"x"}'))
    assert build.await_count == 1


def _plan_request(verb: str) -> PlanExecutionRequest:
    plan = ExecutionPlan(
        verb_name=verb,
        label=verb,
        description="",
        category="x",
        priority="normal",
        interruptible=True,
        can_interrupt_others=False,
        timeout_ms=1000,
        max_recursion_depth=1,
        metadata={"mode": "brain"},
        steps=[
            ExecutionStep(
                verb_name=verb,
                step_name=f"llm_{verb}",
                description="",
                order=0,
                services=["LLMGatewayService"],
                requires_memory=False,
            )
        ],
    )
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(request_id="r1", extra={"mode": "brain"}),
        context={"mode": "brain", "raw_user_text": "hi"},
    )


def _run_router(monkeypatch, verb: str, ctx: dict) -> AsyncMock:
    prepare = AsyncMock(return_value=None)
    monkeypatch.setitem(ROUTER_NS, "prepare_brain_reply_context", prepare)
    monkeypatch.setitem(
        ROUTER_NS,
        "call_step_services",
        AsyncMock(
            return_value=StepExecutionResult(
                status="success",
                verb_name=verb,
                step_name=f"llm_{verb}",
                order=0,
                result={"LLMGatewayService": {"content": "ok"}},
                latency_ms=1,
                node="n",
                logs=[],
                error=None,
            )
        ),
    )
    monkeypatch.setitem(ROUTER_NS, "assemble_stance_grounding", AsyncMock(return_value=None))
    asyncio.run(
        PlanRunner().run_plan(
            bus=object(),
            source=SOURCE,
            req=_plan_request(verb),
            correlation_id=str(uuid4()),
            ctx=ctx,
        )
    )
    return prepare


@pytest.mark.parametrize("verb", FINALIZE_VERBS)
def test_router_skips_prepare_for_finalize_verbs(monkeypatch, caplog, verb: str) -> None:
    caplog.set_level("INFO")
    prepare = _run_router(monkeypatch, verb, {"mode": "brain", "raw_user_text": "hi"})
    assert prepare.await_count == 0
    assert "router_skip_prepare_brain_reply_context" in caplog.text


def test_router_reads_skip_flag_from_options(monkeypatch) -> None:
    """Before L1 the router read the flag from ctx only; memory_graph_suggest-style
    callers set it in options."""
    prepare = _run_router(
        monkeypatch,
        "chat_general",
        {"mode": "brain", "raw_user_text": "hi", "options": {"skip_brain_reply_context": True}},
    )
    assert prepare.await_count == 0


def test_router_still_prepares_for_stance_react(monkeypatch) -> None:
    prepare = _run_router(monkeypatch, "stance_react", {"mode": "brain", "raw_user_text": "hi"})
    assert prepare.await_count == 1


async def _fake_probe(ctx) -> None:
    ctx["current_turn_llm_signals"] = [{"phrase": "Zephyr Bridge", "type": "concept"}]


def test_unified_turn_writes_exactly_one_cortex_turn_row(monkeypatch) -> None:
    """The stance leg publishes the cortex_turn attention-schema row; the
    finalize and repair legs of the same turn no longer add duplicates."""
    published: list[str | None] = []

    async def _record(frame, *, leg=None):
        published.append(leg)
        return True

    async def _no_trace(frame):
        return True

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setitem(STANCE_NS, "publish_attention_schema", _record)
    monkeypatch.setitem(STANCE_NS, "persist_chat_attention_salience_trace", _no_trace)
    monkeypatch.setitem(STANCE_NS, "populate_current_turn_llm_signals", _fake_probe)

    async def _turn() -> None:
        stance_ctx = {
            "mode": "brain",
            "verb": "stance_react",
            "user_message": "I am planning around Zephyr Bridge.",
            "skip_unified_beliefs": True,
            "stance_inputs": {"user_message": "I am planning around Zephyr Bridge.", "utterance_origin": "juniper"},
        }
        await prepare_brain_reply_context(stance_ctx)
        for verb in FINALIZE_VERBS:
            await _run_step(verb, _brain_ctx(), content="final voice")

    asyncio.run(_turn())
    assert published == ["stance_react"]


# ---------------------------------------------------------------------------
# L2 -- the build no longer freezes the loop
# ---------------------------------------------------------------------------


def test_concurrent_handler_completes_during_slow_build(monkeypatch) -> None:
    worker_threads: list[str] = []

    def _slow_hydrate_and_unify(ctx):
        worker_threads.append(threading.current_thread().name)
        time.sleep(2.0)
        return None

    monkeypatch.delenv("ORION_CURIOSITY_FRAME_ENABLED", raising=False)
    monkeypatch.setitem(STANCE_NS, "_hydrate_and_unify_beliefs", _slow_hydrate_and_unify)

    async def _scenario() -> tuple[float, bool]:
        build = asyncio.create_task(
            STANCE_NS["build_chat_stance_inputs"]({"verb": "chat_general", "skip_unified_beliefs": True})
        )
        await asyncio.sleep(0.05)  # let the build reach the worker
        t0 = time.monotonic()
        await asyncio.sleep(0.01)  # a second "handler"
        elapsed = time.monotonic() - t0
        still_building = not build.done()
        await build
        return elapsed, still_building

    elapsed, still_building = asyncio.run(_scenario())
    assert still_building, "the fake 2 s build finished before the concurrent handler ran"
    assert elapsed < 0.5
    assert worker_threads and worker_threads[0].startswith("stance-build")


def test_stance_worker_is_single_threaded_and_ordered() -> None:
    executor = chat_stance_module._stance_build_executor()
    assert executor._max_workers == 1
    order: list[int] = []

    def _step(i: int) -> None:
        time.sleep(0.02)
        order.append(i)

    async def _many() -> None:
        await asyncio.gather(*(chat_stance_module._run_on_stance_worker(_step, i) for i in range(5)))

    asyncio.run(_many())
    assert order == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# L3 -- registry tiers and probe budget
# ---------------------------------------------------------------------------


def _producer(registry, producer_id: str):
    return next(p for p in registry.producers if p.producer_id == producer_id)


def test_identity_yaml_is_ephemeral_in_both_registries() -> None:
    from orion.cognition.projection_builder import build_projection_unification_registry

    for registry in (chat_stance_module._build_unification_registry(), build_projection_unification_registry()):
        entry = _producer(registry, "identity_yaml")
        assert entry.trust_tier.write_through is False
        assert entry.trust_tier.name == "snapshot_ephemeral"
        assert entry.pull_on_cold is False
        assert entry in registry.ephemeral_ctx_producers_for_anchor("orion")
        # Deliberately unchanged: non-write-through + pull_on_cold would make
        # goals flicker between cold and warm turns (review finding).
        autonomy = _producer(registry, "autonomy")
        assert autonomy.trust_tier.name == "graphdb_durable"
        assert autonomy.pull_on_cold is True


def test_self_definition_is_ephemeral() -> None:
    entry = _producer(chat_stance_module._build_unification_registry(), "self_definition")
    assert entry.trust_tier.write_through is False
    assert entry.pull_on_cold is False


def test_identity_yaml_no_longer_degrades_orion_anchor_on_falkor() -> None:
    """Live failure before L3: 'durable writes support concept, evidence, entity
    nodes only; got node_kind=state_snapshot' on every cold turn."""
    from orion.cognition.projection_builder import build_projection_unification_registry
    from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
    from orion.graph.falkor_client import RecordingFalkorClient
    from orion.substrate.relational import CognitiveUnificationLayer, ProducerRegistryV1

    registry = ProducerRegistryV1(
        producers=[_producer(build_projection_unification_registry(), "identity_yaml")]
    )
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=RecordingFalkorClient(),
        hydrate=False,
    )
    layer = CognitiveUnificationLayer(registry=registry, store=store)
    ctx = {
        "orion_identity_summary": ["I am Orion."],
        "juniper_relationship_summary": ["Juniper is my collaborator."],
        "response_policy_summary": ["Speak plainly."],
    }
    beliefs = layer.beliefs_for_stance(anchors=("orion",), ctx=ctx)
    assert "identity_yaml" not in beliefs.degraded_producers
    assert beliefs.anchors["orion"].degraded is False
    sources = [getattr(n, "snapshot_source", None) for n in beliefs.anchors["orion"].snapshots]
    assert "identity_yaml" in sources


def test_signal_probe_sends_its_real_wait_budget_to_the_gateway(monkeypatch) -> None:
    import app.current_turn_llm_signals as signals_module

    sent: list = []

    class _Bus:
        codec = SimpleNamespace(
            decode=lambda data: SimpleNamespace(ok=True, envelope=SimpleNamespace(payload={"content": "[]"}))
        )

        async def rpc_request(self, channel, env, **kwargs):
            sent.append((env, kwargs))
            return {"data": b""}

    monkeypatch.setattr(signals_module.settings, "current_turn_signal_probe_timeout_sec", 3.0)
    asyncio.run(signals_module._llm_call(_Bus(), prompt="p"))
    env, kwargs = sent[0]
    assert env.payload["options"]["gateway_read_timeout_sec"] == 3.0
    assert kwargs["timeout_sec"] == 3.0


def test_identity_lines_from_ephemeral_snapshot_match_the_ctx_fallback() -> None:
    """Spec acceptance: stance identity lines unchanged. Before L3 the snapshot
    write failed and _project_identity_from_beliefs fell back to ctx; now it
    reads the ephemeral snapshot. Both must give the same kernel."""
    from orion.cognition.projection_builder import build_projection_unification_registry
    from orion.substrate.relational import CognitiveUnificationLayer, ProducerRegistryV1
    from orion.substrate.store import InMemorySubstrateGraphStore

    ctx = {
        "orion_identity_summary": [f"Orion line {i}" for i in range(14)] + ["Orion line 3"],
        "juniper_relationship_summary": ["Juniper is my collaborator.", "  "],
        "response_policy_summary": ["Speak plainly."],
    }
    layer = CognitiveUnificationLayer(
        registry=ProducerRegistryV1(producers=[_producer(build_projection_unification_registry(), "identity_yaml")]),
        store=InMemorySubstrateGraphStore(),
    )
    beliefs = layer.beliefs_for_stance(anchors=("orion",), ctx=dict(ctx))
    from_snapshot = chat_stance_module._project_identity_from_beliefs(beliefs, dict(ctx))
    from_fallback = chat_stance_module._project_identity_from_beliefs(None, dict(ctx))
    assert any(getattr(n, "snapshot_source", None) == "identity_yaml" for n in beliefs.anchors["orion"].snapshots)
    assert from_snapshot == from_fallback
    assert len(from_snapshot["orion_identity_summary"]) == 10


# ---------------------------------------------------------------------------
# L6 step 2 -- concept_induction is ephemeral in both registries
# ---------------------------------------------------------------------------


def test_concept_induction_is_ephemeral_and_bound_in_both_registries() -> None:
    import functools

    from orion.cognition.projection_builder import build_projection_unification_registry
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    for registry in (
        chat_stance_module._build_unification_registry(concept_store=store),
        build_projection_unification_registry(concept_store=store),
    ):
        entry = _producer(registry, "concept_induction")
        assert entry.trust_tier.write_through is False
        assert entry.trust_tier.name == "concept_induced"
        assert entry.pull_on_cold is True
        assert isinstance(entry.adapter_fn, functools.partial)
        assert entry.adapter_fn.keywords == {"store": store}
