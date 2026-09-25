"""RPC-health coverage for orion-harness-governor: the dispatch bus is what gets
published, cortex-exec :background RPC outcomes land on it, and the FCC motor's
subprocess wall time lands as hop fcc:<served_model> (success / timeout / skipped)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.harness.cortex_client import HarnessCortexClient
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.cortex.types import ExecutionStep


class _InlinePubSub:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    async def subscribe(self, *channels: str) -> None:
        return None

    async def unsubscribe(self, *channels: str) -> None:
        return None

    async def close(self) -> None:
        return None

    async def listen(self):
        while True:
            yield await self.queue.get()


class _InlineRedis:
    def __init__(self) -> None:
        self.pubsub_obj = _InlinePubSub()

    def pubsub(self):
        return self.pubsub_obj

    async def close(self) -> None:
        return None


def _plan() -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="orion_reflect",
            steps=[ExecutionStep(verb_name="orion_reflect", step_name="s", order=0, services=["LLMGatewayService"])],
        ),
        args=PlanExecutionArgs(request_id=str(uuid4())),
    )


def _motor(**kw):
    base = dict(fcc_elapsed_sec=12.5, grounding_status="grounded", fcc_served_model="qwen3.5-27b", exit_code=0)
    base.update(kw)
    return SimpleNamespace(**base)


def _hops(bus: OrionBusAsync) -> dict:
    return bus.get_rpc_health_snapshot().channel_latency


def _bus() -> OrionBusAsync:
    return OrionBusAsync("redis://unused:6379/0", enabled=False)


def test_settings_defaults_match_env_example() -> None:
    from app.settings import HarnessGovernorSettings

    fields = HarnessGovernorSettings.model_fields
    assert fields["rpc_health_publish_enabled"].default is True
    assert fields["rpc_health_publish_interval_sec"].default == 30.0
    assert fields["rpc_health_channel_latency_enabled"].default is True
    from pathlib import Path

    env = (Path(__file__).resolve().parents[1] / ".env_example").read_text()
    assert "RPC_HEALTH_PUBLISH_ENABLED=true" in env
    assert "RPC_HEALTH_PUBLISH_INTERVAL_SEC=30" in env
    assert "RPC_HEALTH_CHANNEL_LATENCY_ENABLED=true" in env


def test_fcc_hop_success_on_normal_exit() -> None:
    from app.bus_listener import record_fcc_hop

    bus = _bus()
    record_fcc_hop(bus, _motor())
    hop = _hops(bus)["fcc:qwen3.5-27b"]
    assert (hop.success_count, hop.timeout_count) == (1, 0)
    assert hop.max_ms == pytest.approx(12500.0)


@pytest.mark.parametrize("code", ["fcc_timeout", "fcc_stream_stalled"])
def test_fcc_hop_timeout_on_timeout_kill(code: str) -> None:
    from app.bus_listener import record_fcc_hop

    bus = _bus()
    record_fcc_hop(bus, _motor(grounding_status=code, fcc_elapsed_sec=7200.0, exit_code=None))
    hop = _hops(bus)["fcc:qwen3.5-27b"]
    assert (hop.success_count, hop.timeout_count) == (0, 1)
    assert hop.max_ms is None  # a timeout's ceiling never enters latency stats


def test_fcc_hop_nonzero_exit_is_still_a_round_trip_and_missing_model_falls_back() -> None:
    from app.bus_listener import record_fcc_hop

    bus = _bus()
    record_fcc_hop(bus, _motor(grounding_status="fcc_nonzero_exit", exit_code=1, fcc_served_model=None))
    assert _hops(bus)["fcc:unknown"].success_count == 1


@pytest.mark.parametrize(
    "motor",
    [
        _motor(fcc_elapsed_sec=None),
        _motor(grounding_status="fcc_spawn_failed"),
        _motor(grounding_status="fcc_bad_model_label"),
        _motor(grounding_status="fcc_lane_context_too_small"),
        _motor(grounding_status="fcc_mcp_github_missing"),
        _motor(grounding_status="fcc_nonzero_exit", exit_code=-9),  # Hub cancel SIGKILL
    ],
)
def test_fcc_hop_skipped_without_a_real_subprocess_round_trip(motor) -> None:
    from app.bus_listener import record_fcc_hop

    bus = _bus()
    record_fcc_hop(bus, motor)
    assert _hops(bus) == {}


def test_fcc_hop_never_raises() -> None:
    from app.bus_listener import record_fcc_hop

    record_fcc_hop(object(), _motor())  # recorder without the API: swallowed


@pytest.mark.asyncio
async def test_handle_run_records_fcc_hop_on_the_bus_it_was_given() -> None:
    """Wiring: handle_harness_run_request records the motor leg on its own `bus`
    (main.py passes the published dispatch bus)."""
    from app import bus_listener
    from orion.harness.tests.fixtures import make_thought
    from orion.schemas.harness_finalize import HarnessRunRequestV1

    bus = _bus()
    bus.publish = AsyncMock()  # type: ignore[method-assign]
    runner = SimpleNamespace(
        run=AsyncMock(
            return_value=SimpleNamespace(
                draft_text="",
                draft_molecule=None,
                step_count=0,
                exit_code=None,
                compliance_verdict="failed",
                grounding_status="fcc_timeout",
                grammar_receipts=[],
                fcc_served_model="m1",
                fcc_elapsed_sec=30.0,
                reading_receipts=[],
                source_fetches=[],
            )
        )
    )
    # model_construct: the fields irrelevant to this wiring check (user_message,
    # permissions, answer_contract) are not needed once validation is patched out.
    req = HarnessRunRequestV1.model_construct(correlation_id="c-9", thought_event=make_thought(), mode="orion")
    with patch.object(bus_listener, "validate_harness_run_request", return_value=None):
        await bus_listener.handle_harness_run_request(
            bus, req, reply_to="orion:harness:run:result:c-9", runner=runner,
            cortex_client=AsyncMock(), substrate_client=AsyncMock(),
        )
    assert _hops(bus)["fcc:m1"].timeout_count == 1


@pytest.mark.asyncio
async def test_exec_background_rpc_lands_on_dispatch_bus() -> None:
    bus = OrionBusAsync("redis://unused:6379/0")
    fake = _InlineRedis()

    async def _reply(channel, env):
        await fake.pubsub_obj.queue.put({"type": "message", "data": b"{}"})

    client = HarnessCortexClient(
        bus, request_channel="orion:cortex:exec:request:background", result_prefix="orion:exec:result", timeout_sec=1.0
    )
    with patch.object(bus, "_create_pubsub_redis", return_value=fake):
        bus.publish = _reply  # type: ignore[assignment]
        try:  # whatever the reply decodes to, the RPC round trip itself completed
            await client.execute_plan(_plan(), correlation_id=str(uuid4()))
        except RuntimeError:
            pass
        bus.publish = AsyncMock()  # type: ignore[assignment]  # no reply -> timeout
        with patch.object(bus, "_emit_rpc_timeout_grammar", AsyncMock()):
            with pytest.raises(TimeoutError):
                await client.execute_plan(_plan(), correlation_id=str(uuid4()), timeout_sec=0.05)
    hop = _hops(bus)["orion:cortex:exec:request:background"]
    assert (hop.success_count, hop.timeout_count) == (1, 1)


@pytest.mark.asyncio
async def test_publisher_built_from_settings_publishes_dispatch_bus_window() -> None:
    from app import main as gov_main

    bus = _bus()
    bus.publish = AsyncMock()  # type: ignore[method-assign]
    pub = gov_main.build_rpc_health_publisher(lambda: bus)
    assert pub.enabled is True
    pub._kwargs["interval_sec"] = 0.01
    bus.record_hop_success("fcc:m1", 5.0)
    pub.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if bus.publish.await_count:
            break
    await pub.stop()
    payload = bus.publish.await_args_list[0].args[1].payload
    assert payload["service"] == "orion-harness-governor"
    assert payload["instance"] == "main"
    assert payload["channel_latency"]["fcc:m1"]["success_count"] == 1


def test_fcc_hop_timeout_before_first_assistant_event_keys_by_probed_model() -> None:
    """A stall before the CLI echoes a model leaves fcc_served_model None; the timeout
    must still land under the model's key (probed pre-run), not fcc:unknown."""
    from app.bus_listener import record_fcc_hop

    bus = _bus()
    record_fcc_hop(
        bus,
        _motor(grounding_status="fcc_stream_stalled", fcc_served_model=None, probed_served_model="qwen3.5-27b"),
    )
    hops = _hops(bus)
    assert "fcc:unknown" not in hops
    assert hops["fcc:qwen3.5-27b"].timeout_count == 1


@pytest.mark.parametrize("code", ["fcc_stream_line_limit", "fcc_draft_length_ceiling_exceeded"])
def test_fcc_hop_skips_motor_output_limit_kills(code: str) -> None:
    from app.bus_listener import record_fcc_hop

    bus = _bus()
    record_fcc_hop(bus, _motor(grounding_status=code))
    assert _hops(bus) == {}


@pytest.mark.asyncio
async def test_hub_cancel_through_real_runner_is_not_recorded_as_success() -> None:
    """Drive the real HarnessRunner with the event fcc_motor emits after a Hub cancel's
    SIGKILL (fcc_nonzero_exit, metadata.exit_code=-9). Before the runner read exit_code
    from error events, motor.exit_code stayed None and the cancel was recorded as a
    successful (truncated) round trip."""
    from typing import Any, AsyncIterator

    from app.bus_listener import record_fcc_hop
    from orion.harness.runner import HarnessRunner
    from orion.harness.tests.fixtures import make_thought
    from orion.schemas.cognition.answer_contract import AnswerContract
    from orion.schemas.context_exec import ContextExecPermissionV1
    from orion.schemas.harness_finalize import HarnessRunRequestV1

    async def _cancelled(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "error",
            "error": "claude exited with code -9",
            "error_code": "fcc_nonzero_exit",
            "metadata": {"exit_code": -9, "fcc_served_model": None},
            "llm_response": "",
        }

    request = HarnessRunRequestV1(
        correlation_id="c-cancel",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    probe = AsyncMock(return_value="qwen3.5-27b")
    motor = await HarnessRunner(AsyncMock(), fcc_runner=_cancelled, served_model_probe=probe).run(request)
    assert motor.exit_code == -9
    assert motor.probed_served_model == "qwen3.5-27b"

    bus = _bus()
    record_fcc_hop(bus, motor)
    assert _hops(bus) == {}
