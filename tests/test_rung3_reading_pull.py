"""Thinking rung 3 follow-up: the reading pull, and the principled holdback rule.

Pins, each against real numbers from the live data of 2026-10-21:
- holdback is drawn only when the act's outcome can occur without the act (its object pre-exists);
  a painting (object minted by its dispatch) is never held back, a reading pull is;
- the route declares the hub.world_pulse_read `look` act with a queue object and the derived
  12000 s horizon; the proposal template exists and is read-only;
- Hub's pull gates, in order: kill switch, the lane's own Wallet A gate, the derived daily cap (8),
  arousal strained, thermal hot -- unknown arousal / thermal changes nothing;
- the allocator scores a pull on the look act's own open->close cost (679 s live), the lane's own
  open reads do not hold it, a doer refusal writes no episode and holds the route for the doer's
  retry_after_sec, and both arms are precommitted with the seed Hub named.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-execution-dispatch-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

import app.worker as worker_mod  # noqa: E402
from app.worker import ExecutionDispatchRuntimeWorker  # noqa: E402
from orion.autonomy import act_episodes  # noqa: E402
from orion.autonomy.prediction import EffectPosterior  # noqa: E402
from orion.autonomy.thermal_gate import thermal_state  # noqa: E402
from orion.execution_dispatch.policy import READ_PULL_EXECUTOR_VERB, load_execution_dispatch_policy  # noqa: E402
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1  # noqa: E402
from orion.schemas.world_pulse_read_pull import WorldPulseReadPullResultV1  # noqa: E402
from orion.world_pulse_read import pull  # noqa: E402

NOW = datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)
LOOK_CELL = ("act:look", "hub.world_pulse_read", "artifact_produced", 0)
# Live 2026-10-21: substrate_action_effect_posterior for the reading act (n=119, mean 0.696), and the
# look act's p50 open->close over its 119 closes.
LIVE_LOOK = EffectPosterior(mean=0.696, variance=0.0021, n=119)
LIVE_LOOK_COST = 679.3
# Live: distinct seeds whose Stage 1 durable turn started, per UTC day, 2026-09-26..10-20.
LIVE_DAILY_READS = [1, 12, 15, 27, 5, 6, 6, 8, 8, 5, 5, 6, 2, 2, 23, 6, 4, 8, 46, 49, 45, 22, 13, 27, 15]


# --- the holdback rule -----------------------------------------------------------------------------

def test_holdback_is_drawn_only_when_the_object_exists_without_the_act() -> None:
    assert act_episodes.holdback_rule("dispatch") == (
        False, "holdback_exempt:outcome_requires_act:object_minted_by_dispatch")
    assert act_episodes.holdback_rule("queue") == (
        True, "holdback_drawn:outcome_observable_without_act:object_preexists")
    with pytest.raises(ValueError):
        act_episodes.holdback_rule("painting")   # a name is not a rule


def test_routes_declare_where_their_object_comes_from() -> None:
    policy = load_execution_dispatch_policy(REPO / "config/execution_dispatch/execution_dispatch_policy.v1.yaml")
    refs = policy.act_refs_by_cortex_verb()
    paint = refs["skills.imagination.render_scene.v1"]
    read = refs[READ_PULL_EXECUTOR_VERB]
    assert paint.object_source == "dispatch"
    assert (read.verb, read.actor, read.signal_id, read.object_source, read.horizon_sec) == (
        "look", "hub.world_pulse_read", "artifact_produced", "queue", 12000.0)
    route = policy.template_to_cortex["read_world_pulse"]
    assert (route.dispatch_kind, route.allowed_scope) == ("observe", "inspect_only")


def test_painting_object_really_is_minted_by_its_dispatch() -> None:
    """The fact the 'dispatch' declaration rests on: the painting run id is a function of the
    dispatch id, so no dispatch means no run to observe."""
    from orion.schemas.reverie_visual_run import reverie_visual_run_id

    assert reverie_visual_run_id("d:1") != reverie_visual_run_id("d:2")
    assert reverie_visual_run_id("d:1") == reverie_visual_run_id("d:1")


def test_the_read_template_is_a_read_only_observe_proposal() -> None:
    from orion.proposals.policy import load_proposal_policy

    tpl = load_proposal_policy(REPO / "config/proposals/proposal_policy.v1.yaml").proposal_templates["read_world_pulse"]
    assert (tpl.kind, tpl.required_policy_gate, tpl.target_id) == ("observe", "read_only", "queue:world_pulse_read")
    assert tpl.dimensions == {} and tpl.expected_signal is None


# --- Hub's pull gates ----------------------------------------------------------------------------

def _gates(**kw):
    base = dict(enabled=True, wallet_reason=None, wallet_retry_sec=None, pulled_today=0, daily_cap=8,
                arousal_level=None, thermal=None, now=NOW)
    base.update(kw)
    return pull.pull_block_reason(pull.PullGateInputs(**base))


def test_daily_cap_is_derived_from_the_lane_median_reading_day() -> None:
    assert pull.derive_daily_cap(LIVE_DAILY_READS) == pull.DERIVED_DAILY_CAP == 8
    assert pull.derive_daily_cap([0, 0, 3, 5, 100]) == 5          # zero days and bursts do not set it
    with pytest.raises(ValueError):
        pull.derive_daily_cap([0, 0])


def test_gates_in_order_and_unknown_changes_nothing() -> None:
    hot = thermal_state(temp_c=33.52, age_sec=1.0)        # live cabinet 2026-10-21 05:18Z
    assert _gates().open
    assert _gates(arousal_level=None, thermal=thermal_state(temp_c=None, age_sec=None)).open
    assert _gates(enabled=False, arousal_level="strained").refusal == "disabled"
    assert _gates(wallet_reason="refund_backoff", wallet_retry_sec=1800.0).retry_after_sec == 1800.0
    cap = _gates(pulled_today=8, arousal_level="strained")
    assert cap.refusal == "daily_cap" and cap.retry_after_sec == pytest.approx(21 * 3600.0)
    assert _gates(arousal_level="strained", thermal=hot).refusal == "arousal_strained"
    assert _gates(arousal_level="engaged", thermal=hot).refusal == "thermal_hot"
    assert _gates(arousal_level="calm", thermal=thermal_state(temp_c=29.0, age_sec=1.0)).open


# --- Hub's pull(): check has no side effect, act marks the seed ------------------------------------

class _Conn:
    def __init__(self, *, head=None, prior=None, pulled_today=0, mark_ok=True):
        self.head, self.prior, self.pulled_today, self.mark_ok = head, prior, pulled_today, mark_ok
        self.marks: list[tuple] = []

    async def fetchval(self, sql, *args):
        if "information_schema.columns" in sql:
            return 3
        if sql == pull.COUNT_STARTED_PULLS_SQL:
            return self.pulled_today
        raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        if sql == pull.SELECT_PULL_CANDIDATE_SQL:
            return self.head
        if "WHERE pull_dispatch_id = $1" in sql:
            return self.prior
        if sql == pull.MARK_PULL_SQL:
            self.marks.append(args)
            return self.head if self.mark_ok else None
        raise AssertionError(sql)


SEED = {"seed_id": "digest_item:r:i:h", "kind": "digest_item", "run_id": "r", "url": "https://example.org/a",
        "title": "t", "section": "s", "item_id": "i", "request_json": None}


def _pipeline(conn, *, arousal=None, temp=None):
    import importlib

    sys.path.insert(0, str(REPO / "services" / "orion-hub"))
    mod = importlib.import_module("scripts.world_pulse_read_pipeline")
    p = object.__new__(mod.WorldPulseReadPipeline)
    p.enabled, p.pull_enabled, p.pull_daily_cap = True, True, 8
    p.window_start_hour = p.window_end_hour = 0
    p._tz_loaded, p._tz, p.tick_interval_sec = True, timezone.utc, 300.0
    p._bus = None
    p._pull_arousal_reader = None if arousal is None else SimpleNamespace(
        refresh=AsyncMock(), level=lambda now=None: arousal)
    p._pull_thermal_reader = None if temp is None else (lambda: (temp, 1.0))

    class _Pool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self_inner):
                    return conn

                async def __aexit__(self_inner, *a):
                    return False
            return _Ctx()

    p._pool_provider = lambda: _Pool()
    p._read_claimed = AsyncMock(return_value="waiting_resource")
    return p


def _req(op, arm=None, dispatch_id="d:1"):
    from orion.schemas.world_pulse_read_pull import WorldPulseReadPullRequestV1

    return WorldPulseReadPullRequestV1(op=op, dispatch_id=dispatch_id, arm=arm)


@pytest.mark.asyncio
async def test_check_refuses_on_strained_arousal_without_touching_the_queue() -> None:
    conn = _Conn(head=SEED)
    out = await _pipeline(conn, arousal="strained").pull(_req("check"))
    assert (out.state, out.refusal, out.retry_after_sec) == ("refused", "arousal_strained", 360.0)
    assert conn.marks == []


@pytest.mark.asyncio
async def test_check_names_the_lane_head_and_marks_nothing() -> None:
    conn = _Conn(head=SEED)
    out = await _pipeline(conn, arousal="calm", temp=27.0).pull(_req("check"))
    assert (out.state, out.seed_id, out.object_ref) == ("ready", SEED["seed_id"], f"read_seed:{SEED['seed_id']}")
    assert conn.marks == []


@pytest.mark.asyncio
async def test_hot_room_refuses_a_pull() -> None:
    out = await _pipeline(_Conn(head=SEED), temp=33.52).pull(_req("check"))
    assert (out.state, out.refusal) == ("refused", "thermal_hot")


@pytest.mark.asyncio
async def test_treated_act_marks_and_starts_one_read_control_only_marks() -> None:
    import asyncio

    conn = _Conn(head=SEED)
    p = _pipeline(conn)
    treated = await p.pull(_req("act", "treated", "d:t"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert (treated.state, treated.seed_id, treated.pulled_today) == ("started", SEED["seed_id"], 1)
    assert conn.marks[-1] == (SEED["seed_id"], "d:t", "treated")
    p._read_claimed.assert_awaited_once()
    control = await _pipeline(conn).pull(_req("act", "control", "d:c"))
    assert (control.state, control.arm) == ("control_recorded", "control")
    assert conn.marks[-1] == (SEED["seed_id"], "d:c", "control")


@pytest.mark.asyncio
async def test_a_replayed_act_reports_its_first_mark_never_a_second_seed() -> None:
    conn = _Conn(head=SEED, prior={"seed_id": "earlier", "pull_arm": "control"})
    out = await _pipeline(conn).pull(_req("act", "treated", "d:r"))
    assert (out.state, out.seed_id, out.arm) == ("control_recorded", "earlier", "control")
    assert conn.marks == []


@pytest.mark.asyncio
async def test_the_cap_counts_started_pulls() -> None:
    out = await _pipeline(_Conn(head=SEED, pulled_today=8)).pull(_req("check"))
    assert (out.state, out.refusal, out.pulled_today) == ("refused", "daily_cap", 8)


# --- dispatch runtime ----------------------------------------------------------------------------

def _read(dispatch_id: str = "dispatch:proposal:read_world_pulse:t1") -> ExecutionDispatchCandidateV1:
    return ExecutionDispatchCandidateV1(
        dispatch_id=dispatch_id, source_decision_id=f"pd:{dispatch_id}", source_proposal_id=f"p:{dispatch_id}",
        dispatch_status="prepared_for_dispatch", dispatch_mode="dispatch_read_only", dispatch_kind="observe",
        target_id="queue:world_pulse_read", target_kind="system", cortex_verb=READ_PULL_EXECUTOR_VERB,
        cortex_mode="brain", risk_score=0.05, confidence_score=0.9)


def _frame(*cands) -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:test", source_policy_frame_id="policy.frame:test",
        source_proposal_frame_id="proposal.frame:test", generated_at=NOW,
        dispatch_mode="dispatch_read_only", candidates=list(cands))


def _worker(monkeypatch, *, posteriors=None, enabled=True):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = ExecutionDispatchRuntimeWorker()
    monkeypatch.setattr(worker._settings, "orion_dispatch_act_candidates_enabled", True, raising=False)
    monkeypatch.setattr(worker._settings, "orion_dispatch_reading_pull_enabled", enabled, raising=False)
    store = MagicMock()
    store.load_action_cost_estimates.return_value = {}
    store.load_effect_posteriors.return_value = {LOOK_CELL: LIVE_LOOK} if posteriors is None else posteriors
    store.act_duration_p50_sec.return_value = LIVE_LOOK_COST
    store.act_in_flight.return_value = False
    store.act_episode_in_flight.return_value = False
    store.act_control_stats.return_value = (0, 0)
    store.load_dispatch_result_by_dispatch_id.return_value = None
    worker._store = store
    return worker


MOTOR = SimpleNamespace(remaining_sec=1e6)


def test_a_pull_is_scored_on_the_look_act_and_its_measured_work_time(monkeypatch) -> None:
    """Below the unchanged information floor, then admitted on the proven path (LB 0.604 > 0.5)."""
    worker = _worker(monkeypatch)
    out = worker._log_allocator_preview(_frame(_read()), MOTOR)
    assert [c.dispatch_id for c in out.admitted] == ["dispatch:proposal:read_world_pulse:t1"]
    assert out.admitted[0].cost_sec == LIVE_LOOK_COST
    worker._store.act_duration_p50_sec.assert_called_once_with(
        actor="hub.world_pulse_read", verb="look", signal_id="artifact_produced", lookback_hours=720)
    reason = worker._last_proven_reasons["dispatch:proposal:read_world_pulse:t1"]
    assert reason.startswith("admitted:proven_effect:lb=0.604:ref=0.500:act_prior")


def test_the_lane_own_open_reads_do_not_hold_a_pull(monkeypatch) -> None:
    worker = _worker(monkeypatch)
    worker._log_allocator_preview(_frame(_read()), MOTOR)
    assert worker._store.act_in_flight.call_args.kwargs["include_open_acts"] is False


def test_kill_switch_and_doer_hold_refuse_before_the_allocator(monkeypatch) -> None:
    off = _worker(monkeypatch, enabled=False)
    assert off._log_allocator_preview(_frame(_read()), MOTOR).refusals_by_reason() == {"reading_pull_disabled": 1}
    held = _worker(monkeypatch)
    held._reading_pull_hold = (datetime.now(timezone.utc) + timedelta(seconds=300), "arousal_strained")
    assert held._log_allocator_preview(_frame(_read()), MOTOR).refusals_by_reason() == {
        "doer_refused:arousal_strained": 1}


def _send_worker(monkeypatch, replies):
    worker = _worker(monkeypatch)
    worker._derive_motor_budget = MagicMock(return_value=SimpleNamespace(
        remaining_sec=1e6, mode="enforcing", exhausted=False, spent_sec=0.0, allowance_sec=1e6, pace=1.0,
        projected_day_sec=0.0))
    worker._derive_daily_risk_cap = MagicMock(return_value=(1e9, {}))
    worker._store.sum_risk_dispatched_today.return_value = 0.0
    monkeypatch.setattr(worker._settings, "orion_dispatch_holdback_fraction", 0.2, raising=False)
    monkeypatch.setattr(worker._settings, "orion_dispatch_allocator_enforce", True, raising=False)
    fake_bus = MagicMock(connect=AsyncMock(), close=AsyncMock(), publish=AsyncMock())
    monkeypatch.setattr(worker_mod, "OrionBusAsync", MagicMock(return_value=fake_bus))
    monkeypatch.setattr(worker_mod.RPC_HEALTH_SINK, "absorb_bus", MagicMock())
    calls = []

    async def rpc(bus, req):
        calls.append(req)
        return replies[(req.op, req.arm)]

    worker._read_pull_rpc = rpc
    worker._record_dispatch_status = MagicMock()
    rows = []
    worker._store.insert_act_episode.side_effect = lambda row: rows.append(row) or True
    return worker, calls, rows


def _reply(op, state, arm=None, **kw):
    return WorldPulseReadPullResultV1(op=op, dispatch_id="x", state=state, arm=arm, **kw)


READY = _reply("check", "ready", seed_id="s1", object_ref="read_seed:s1")


@pytest.mark.asyncio
async def test_a_refused_check_writes_no_episode_and_holds_the_route(monkeypatch) -> None:
    worker, calls, rows = _send_worker(monkeypatch, {("check", None): _reply(
        "check", "refused", refusal="arousal_strained", retry_after_sec=360.0)})
    out = await worker._send_prepared_candidates(_frame(_read("d:r")))
    assert rows == [] and [c.op for c in calls] == ["check"]
    assert [c.blocked_by for c in out.blocked_candidates] == [["doer_refused:arousal_strained"]]
    assert worker._reading_pull_hold[1] == "arousal_strained"


@pytest.mark.asyncio
async def test_treated_pull_precommits_the_checked_seed_then_acts(monkeypatch) -> None:
    worker, calls, rows = _send_worker(monkeypatch, {
        ("check", None): READY,
        ("act", "treated"): _reply("act", "started", "treated", seed_id="s1", object_ref="read_seed:s1")})
    monkeypatch.setattr(worker, "_stable_draw", lambda dispatch_id: 0.9)
    out = await worker._send_prepared_candidates(_frame(_read("d:t")))
    assert [(c.op, c.arm) for c in calls] == [("check", None), ("act", "treated")]
    assert [(r["arm"], r["object_ref"], r["outcome_value"]) for r in rows] == [("treated", "read_seed:s1", None)]
    assert rows[0]["expected"]["holdback"] == "holdback_drawn:outcome_observable_without_act:object_preexists"
    assert rows[0]["horizon_at"] - rows[0]["decided_at"] == timedelta(seconds=12000)
    worker._store.set_act_episode_object.assert_called_once_with("d:t", "read_seed:s1")
    assert [c.dispatch_id for c in out.dispatched_candidates] == ["d:t"]
    saved = worker._store.save_dispatch_result.call_args.kwargs
    assert saved["latency_ms"] is None and saved["result_json"]["settlement"]["seed_id"] == "s1"


@pytest.mark.asyncio
async def test_control_pull_is_withheld_but_its_seed_is_marked(monkeypatch) -> None:
    worker, calls, rows = _send_worker(monkeypatch, {
        ("check", None): READY,
        ("act", "control"): _reply("act", "control_recorded", "control", seed_id="s1", object_ref="read_seed:s1")})
    monkeypatch.setattr(worker, "_stable_draw", lambda dispatch_id: 0.1)
    out = await worker._send_prepared_candidates(_frame(_read("d:c")))
    assert [(c.op, c.arm) for c in calls] == [("check", None), ("act", "control")]
    assert [(r["arm"], r["outcome_value"]) for r in rows] == [("control", None)]   # resolved later, not 0
    assert out.dispatched_candidates == []
    assert [c.blocked_by for c in out.blocked_candidates] == [["act_holdback"]]
    worker._store.set_act_episode_object.assert_called_once_with("d:c", "read_seed:s1")


@pytest.mark.asyncio
async def test_a_refusal_at_act_voids_the_decision(monkeypatch) -> None:
    worker, calls, rows = _send_worker(monkeypatch, {
        ("check", None): READY,
        ("act", "treated"): _reply("act", "refused", "treated", refusal="no_candidate", retry_after_sec=300.0)})
    monkeypatch.setattr(worker, "_stable_draw", lambda dispatch_id: 0.9)
    await worker._send_prepared_candidates(_frame(_read("d:v")))
    worker._store.void_act_episode.assert_called_once_with("d:v", "doer_refused:no_candidate")
