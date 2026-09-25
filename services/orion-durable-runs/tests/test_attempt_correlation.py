"""Late cancellations cannot address a successor lease's FCC subprocess."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.admission_runtime import AdmissionRuntime
from app.graph import Deps, make_nodes, turn_correlation_id
from orion.harness import fcc_motor
from orion.schemas.durable_run import CuriosityTurnResultV1


def state(generation=1, fence=0):
    # A granted GPU pool hold's ref (stage 4.5): lease_id + generation fence the turn identity.
    return {"run_id": "study-generation-001", "correlation_id": "original-study-lineage", "attempt": 0,
        "brief": {"prompt": "Investigate the evidence", "session_id": "study-session", "timeout_sec": 100},
        "lease": {"lease_id": "hold-1", "generation": generation, "role": "agent",
                  "holder": "durable-runs:study-generation-001"},
        **({"turn_fence": fence} if fence else {})}


def test_harness_identity_changes_per_lease_but_run_lineage_stays_stable():
    async def scenario():
        seen = []
        async def turn(request):
            seen.append(request)
            return CuriosityTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id,
                                         text="A supported finding", debug={"harness_step_count": 14})
        node = make_nodes(Deps(turn, AsyncMock(), AsyncMock(), AsyncMock()))["harness_turn"]
        first, second = state(1), state(2)
        first_result = await node(first)
        await node(first)  # duplicate same lease keeps one transport identity
        second_result = await node(second)
        assert seen[0].correlation_id == seen[1].correlation_id
        assert seen[0].correlation_id != seen[2].correlation_id
        assert {request.run_id for request in seen} == {first["run_id"]}
        assert first["correlation_id"] == second["correlation_id"] == "original-study-lineage"
        for result, request in ((first_result, seen[0]), (second_result, seen[2])):
            assert result["debug"]["turn_correlation_id"] == request.correlation_id
            assert result["debug"]["parent_correlation_id"] == "original-study-lineage"
            assert result["debug"]["harness_step_count"] == 14
    asyncio.run(scenario())


def test_delayed_old_cancel_does_not_kill_new_registered_or_future_process(monkeypatch):
    monkeypatch.setattr(fcc_motor, "_ACTIVE", {})
    monkeypatch.setattr(fcc_motor, "_PENDING_CANCEL", set())
    first, second = state(1), state(2)
    runtime = object.__new__(AdmissionRuntime)
    runtime.runner = SimpleNamespace(_publish=AsyncMock(return_value=True), _corr_for_admission=lambda value: value)
    asyncio.run(runtime._cancel_harness(first, "lease_lost"))
    channel, kind, cancel, envelope_corr = runtime.runner._publish.await_args.args
    assert channel == "orion:harness:run:cancel" and kind == "harness.run.cancel.v1"
    assert cancel.correlation_id == envelope_corr == turn_correlation_id(first)
    successor = MagicMock()
    fcc_motor._register_process(turn_correlation_id(second), successor)
    fcc_motor.cancel_fcc_turn(cancel.correlation_id)  # old process not registered yet
    successor.kill.assert_not_called()
    late_old_process = MagicMock()
    fcc_motor._register_process(turn_correlation_id(first), late_old_process)
    late_old_process.kill.assert_called_once()
    successor.kill.assert_not_called()
    assert fcc_motor._ACTIVE[turn_correlation_id(second)] is successor


def test_legacy_cancellation_identity_remains_unchanged():
    legacy = state()
    legacy.pop("lease")
    assert turn_correlation_id(legacy) == legacy["correlation_id"]
    runtime = object.__new__(AdmissionRuntime)
    runtime.runner = SimpleNamespace(_publish=AsyncMock(return_value=True), _corr_for_admission=lambda value: value)
    asyncio.run(runtime._cancel_harness(legacy, "cancel"))
    assert runtime.runner._publish.await_args.args[2].correlation_id == legacy["correlation_id"]


def test_a_restart_fence_under_the_same_hold_generation_gets_a_new_identity():
    """A pool hold keeps lease_id + generation across a durable-runs restart; the restarted driver
    bumps turn_fence so the replay never shares the fenced turn's identity (a late cancel for the
    old one cannot kill it). Fence 0 is the pre-fence identity, so old checkpoints are unchanged."""
    same_hold = [turn_correlation_id(state(1, fence)) for fence in (0, 1, 2)]
    assert len(set(same_hold)) == 3
    assert turn_correlation_id(state(1, 0)) == turn_correlation_id(state(1))
