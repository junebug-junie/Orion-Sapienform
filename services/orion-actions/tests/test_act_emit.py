"""Body economy A5: daily metacog (`infer`) and in-process journal (`make`) acts.

Driven through the real lifespan closures (`_execute_daily`, `_dispatch_journal`), with the bus,
cortex RPC and notify faked, the same way test_daily_retry_gate.py drives the daily path.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app import act_emit
from app import main as actions_main
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

METACOG_JSON = json.dumps({
    "date": "2026-10-22", "timezone": "America/Denver", "thinking_patterns": ["p"], "blindspots": [],
    "course_correction": "tighten grounding", "tomorrow_experiment": "run one bounded check",
    "tomorrow_experiment_skill_id": None, "confidence": 0.6,
})


def _grammar_events(hunter) -> list:
    out = []
    for c in hunter.bus.publish.call_args_list:
        env = c.args[1]
        if getattr(env, "kind", "") == "grammar.event.v1":
            assert c.args[0] == "orion:grammar:event"
            out.append(env.payload)
    return out


class _Harness:
    def __init__(self, *, plan_payload=None, orch_payload=None, rpc_error=None, notify_ok=True, emit=True):
        self.plan_payload = plan_payload
        self.orch_payload = orch_payload
        self.rpc_error = rpc_error
        self.notify_ok = notify_ok
        self.emit = emit
        self.hunters: list = []
        self.sent: list = []  # outbound cortex request envelopes (their correlation ids lease the GPU)

    async def run(self, tmp_path, body):
        harness = self

        class _Hunter:
            def __init__(self, *a, **k):
                self.bus = AsyncMock()
                self.bus.codec = MagicMock()
                self.bus.codec.decode.return_value = SimpleNamespace(
                    ok=True, error=None, envelope=SimpleNamespace(payload=harness.plan_payload))
                harness.hunters.append(self)

            async def start(self):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass

        rpc_bus = MagicMock()

        async def _rpc(channel, env, **k):
            harness.sent.append(env)
            if harness.rpc_error is not None:
                raise harness.rpc_error
            return {"data": b"stub"}

        rpc_bus.rpc_request = _rpc
        rpc_bus.codec.decode.return_value = SimpleNamespace(
            ok=True, error=None, envelope=SimpleNamespace(payload=harness.orch_payload))
        rpc_bus.close = AsyncMock()

        async def _fork(_bus):
            return rpc_bus

        real_create_task = asyncio.create_task

        def _noop_task(coro, *a, **k):
            coro.close()

            async def _idle():
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass

            return real_create_task(_idle())

        async def _plan_rpc(**kwargs):
            harness.sent.append(kwargs["envelope_factory"]("orion:exec:result:test", 1))
            if harness.rpc_error is not None:
                raise harness.rpc_error
            return {"data": b"stub"}

        accepted = SimpleNamespace(ok=self.notify_ok, status="ok", detail=None if self.notify_ok else "down",
                                   notification_id=None)
        draft = MagicMock()
        draft.model_dump.return_value = {"mode": "digest"}
        write = MagicMock()
        write.model_dump.return_value = {"entry_id": "entry-1"}
        with (
            patch.object(actions_main, "Hunter", _Hunter),
            patch("orion.core.bus.rpc_fork.fork_rpc_client", _fork),
            patch("orion.notify.client.NotifyClient", return_value=MagicMock()),
            patch.object(actions_main.asyncio, "create_task", side_effect=_noop_task),
            patch.object(actions_main, "_rpc_request_with_retry", _plan_rpc),
            patch.object(actions_main, "_publish_daily_outputs", lambda **k: {"generic": accepted}),
            patch.object(actions_main, "draft_from_cortex_result", lambda p: draft),
            patch.object(actions_main, "guard_walkway_draft", lambda d, t: (d, [])),
            patch.object(actions_main, "build_write_payload", lambda *a, **k: write),
            patch.object(actions_main.settings, "actions_scheduler_cursor_store_path", str(tmp_path / "c.json")),
            patch.object(actions_main.settings, "actions_journaling_enabled", True),
            patch.object(actions_main.settings, "actions_act_emit_enabled", self.emit),
        ):
            async with actions_main.lifespan(actions_main.app):
                await body(actions_main.app)
        return _grammar_events(self.hunters[0])


async def _metacog_tick(app, local_date="2026-10-23"):
    from datetime import datetime, timezone

    return await app.state.scheduled_daily_tick(
        action_name=actions_main.ACTION_DAILY_METACOG_V1, due=True, local_date=local_date,
        now_utc=datetime(2026, 10, 24, 2, 15, tzinfo=timezone.utc), forced_date=None,
    )


def _journal_env(summary="a thing happened worth writing down") -> BaseEnvelope:
    return BaseEnvelope(kind="orion.actions.trigger.journal.v1", source=ServiceRef(name="test"),
                        correlation_id=str(uuid4()), payload={"summary": summary})


# --- daily metacog -------------------------------------------------------------------------------

def test_daily_metacog_success_opens_and_closes_one_infer_act(tmp_path):
    h = _Harness(plan_payload={"result": {"status": "success", "final_text": METACOG_JSON}})
    events = asyncio.run(h.run(tmp_path, _metacog_tick))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    opened, closed = events
    assert opened["act"]["actor"] == "actions.daily_metacog" and opened["act"]["verb"] == "infer"
    assert opened["act"]["act_id"] == closed["act"]["act_id"]
    assert opened["provenance"]["source_service"] == "orion-actions"
    assert closed["act"]["outcome"]["value"] == 1.0
    assert closed["act"]["cost"]["seconds"] >= 0.0
    assert set(closed["act"]["cost"]) == {"seconds", "gpu_seconds", "tokens", "watts"}  # V1 only
    assert opened["correlation_id"] == closed["correlation_id"] == str(h.sent[0].correlation_id)


def test_daily_metacog_failure_closes_zero(tmp_path):
    h = _Harness(plan_payload={"result": {"status": "fail", "final_text": None, "steps": [
        {"step_name": "draft", "status": "fail", "error": "daily_metacog_prompt_over_limit"}]}})
    events = asyncio.run(h.run(tmp_path, _metacog_tick))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    assert events[1]["act"]["outcome"]["value"] == 0.0
    assert events[1]["act"]["outcome"]["note"].startswith("raised:RuntimeError")


def test_daily_metacog_notify_failure_closes_zero(tmp_path):
    h = _Harness(plan_payload={"result": {"status": "success", "final_text": METACOG_JSON}}, notify_ok=False)
    events = asyncio.run(h.run(tmp_path, _metacog_tick))
    assert events[-1]["act"]["outcome"]["value"] == 0.0
    assert "notify_failed" in events[-1]["act"]["outcome"]["note"]


def test_daily_pulse_is_not_an_act(tmp_path):
    from datetime import datetime, timezone

    async def pulse(app):
        await app.state.scheduled_daily_tick(
            action_name=actions_main.ACTION_DAILY_PULSE_V1, due=True, local_date="2026-10-23",
            now_utc=datetime(2026, 10, 24, 14, 30, tzinfo=timezone.utc), forced_date=None)

    h = _Harness(plan_payload={"result": {"status": "fail", "final_text": None}})
    assert asyncio.run(h.run(tmp_path, pulse)) == []


# --- journal -------------------------------------------------------------------------------------

def test_journal_compose_success_opens_and_closes_one_make_act(tmp_path):
    env = _journal_env()

    async def body(app):
        await app.state.bus_handler(env)

    h = _Harness(orch_payload={"ok": True})
    events = asyncio.run(h.run(tmp_path, body))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    opened, closed = events
    assert opened["act"]["actor"] == "actions.journal" and opened["act"]["verb"] == "make"
    # The act's id is the compose request's id (the one its GPU leases carry), not the trigger's.
    assert opened["correlation_id"] == closed["correlation_id"] == str(h.sent[0].correlation_id)
    assert opened["correlation_id"] != str(env.correlation_id)
    assert closed["act"]["outcome"]["value"] == 1.0 and "entry_id=entry-1" in closed["act"]["outcome"]["note"]


def test_journal_compose_failure_closes_zero(tmp_path):
    async def body(app):
        await app.state.bus_handler(_journal_env())

    h = _Harness(orch_payload={"ok": False, "error": "gpu_pool_unavailable:deadline"})
    events = asyncio.run(h.run(tmp_path, body))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    assert events[1]["act"]["outcome"]["value"] == 0.0
    assert "journal_compose_failed" in events[1]["act"]["outcome"]["note"]


def test_journal_rpc_exception_closes_zero(tmp_path):
    async def body(app):
        await app.state.bus_handler(_journal_env())

    h = _Harness(rpc_error=TimeoutError("RPC timeout"))
    events = asyncio.run(h.run(tmp_path, body))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    assert events[1]["act"]["outcome"]["note"].startswith("raised:TimeoutError")


def test_journal_cooldown_skip_opens_nothing(tmp_path):
    env = _journal_env("same summary")

    async def body(app):
        await app.state.bus_handler(env)
        await app.state.bus_handler(_journal_env("same summary"))  # deduped by cooldown

    h = _Harness(orch_payload={"ok": True})
    events = asyncio.run(h.run(tmp_path, body))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]


def test_redelivered_trigger_is_the_same_act(tmp_path):
    """Same trigger, same correlation id -> same act id (the writer is first-close-wins)."""
    a = act_emit.journal_act("k", "corr-1", trigger_kind="manual", horizon=480.0)
    b = act_emit.journal_act("k", "corr-1", trigger_kind="manual", horizon=480.0)
    c = act_emit.journal_act("k", "corr-2", trigger_kind="manual", horizon=480.0)
    assert a.act_id == b.act_id != c.act_id


def test_flag_off_emits_nothing(tmp_path):
    async def body(app):
        await app.state.bus_handler(_journal_env())
        await _metacog_tick(app)

    h = _Harness(orch_payload={"ok": True},
                 plan_payload={"result": {"status": "success", "final_text": METACOG_JSON}}, emit=False)
    assert asyncio.run(h.run(tmp_path, body)) == []


def test_world_pulse_journal_goes_durable_and_emits_no_actions_act(tmp_path):
    """Ownership rule: a durable journal.compose run is counted by orion-durable-runs only."""
    from datetime import datetime, timezone

    from orion.schemas.world_pulse import (
        DailyWorldPulseSectionsV1, DailyWorldPulseV1, WorldPulseRunResultV1, WorldPulseRunV1,
    )

    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    payload = WorldPulseRunResultV1(
        run=WorldPulseRunV1(run_id="wp-1", date="2026-10-23", started_at=now, completed_at=now,
                            status="completed", dry_run=False),
        digest=DailyWorldPulseV1(run_id="wp-1", date="2026-10-23", generated_at=now, title="t",
                                 executive_summary="s", sections=DailyWorldPulseSectionsV1(),
                                 orion_analysis_layer="deterministic", created_at=now),
    ).model_dump(mode="json")
    env = BaseEnvelope(kind="world.pulse.run.result.v1", source=ServiceRef(name="orion-world-pulse"),
                       correlation_id=str(uuid4()), payload=payload)

    async def body(app):
        with patch.object(actions_main, "submit_durable_run_via_cortex", AsyncMock(return_value=None)):
            await app.state.bus_handler(env)

    h = _Harness(orch_payload={"ok": True})
    events = asyncio.run(h.run(tmp_path, body))
    assert events == []


@pytest.mark.parametrize("timeout,expected", [(420.0, 480.0)])
def test_journal_horizon_is_the_compose_deadline(timeout, expected):
    assert act_emit.journal_horizon_sec(timeout) == expected
    assert act_emit.daily_metacog_horizon_sec(timeout) == 4 * timeout + 60.0


def test_trace_closes_once_and_survives_a_failing_publish():
    sent = []

    async def bad(event):
        raise ConnectionError("bus gone")

    async def good(event):
        sent.append(event)

    act = act_emit.journal_act("k", "c", trigger_kind="manual", horizon=480.0)
    t = act_emit.ActTrace(bad, act, component="x", correlation_id="c")
    asyncio.run(t.open())
    asyncio.run(t.close(True, "ok"))  # no raise

    t2 = act_emit.ActTrace(good, act, component="x", correlation_id="c")
    asyncio.run(t2.open())
    asyncio.run(t2.close(True, "first"))
    asyncio.run(t2.close(False, "second"))
    assert [e.event_kind for e in sent] == ["act_opened", "act_closed"]
    assert sent[1].act.outcome.note == "first"


def test_two_composes_sharing_a_trigger_id_get_distinct_cost_keys(tmp_path):
    """Walkway jobs in one scheduler tick share the parent correlation id; their acts must not."""
    shared = str(uuid4())

    def env(summary):
        return BaseEnvelope(kind="orion.actions.trigger.journal.v1", source=ServiceRef(name="test"),
                            correlation_id=shared, payload={"summary": summary})

    async def body(app):
        await app.state.bus_handler(env("day rollup"))
        await app.state.bus_handler(env("week rollup"))

    h = _Harness(orch_payload={"ok": True})
    events = asyncio.run(h.run(tmp_path, body))
    closed = [e for e in events if e["event_kind"] == "act_closed"]
    assert len(closed) == 2
    assert closed[0]["correlation_id"] != closed[1]["correlation_id"]
    assert closed[0]["act"]["act_id"] != closed[1]["act"]["act_id"]
    assert [str(e.correlation_id) for e in h.sent] == [c["correlation_id"] for c in closed]


def test_journal_cancellation_closes_zero_and_propagates(tmp_path):
    async def body(app):
        with pytest.raises(asyncio.CancelledError):
            await app.state.bus_handler(_journal_env())

    h = _Harness(rpc_error=asyncio.CancelledError())
    events = asyncio.run(h.run(tmp_path, body))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    assert events[1]["act"]["outcome"]["value"] == 0.0
    assert events[1]["act"]["outcome"]["note"].startswith("raised:CancelledError")


def test_daily_metacog_cancellation_closes_zero_and_propagates(tmp_path):
    async def body(app):
        with pytest.raises(asyncio.CancelledError):
            await _metacog_tick(app)

    h = _Harness(rpc_error=asyncio.CancelledError())
    events = asyncio.run(h.run(tmp_path, body))
    assert [e["event_kind"] for e in events] == ["act_opened", "act_closed"]
    assert events[1]["act"]["outcome"]["note"].startswith("raised:CancelledError")
