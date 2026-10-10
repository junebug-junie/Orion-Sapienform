"""Durable `reverie.visual` stage handlers (app/visual_steps.py) against an in-memory
attempt store and faked hops. The real SQL of the stage store is covered by
test_visual_steps_db.py against a disposable PostgreSQL."""
from __future__ import annotations

import copy
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    NEEDS_GENERATE,
    ReverieVisualStepRequestV1,
    ReverieVisualStepResultV1,
)

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
RUN_ID = "reverie-visual-test"
REQUEST = VisualRunRequestV1(dispatch_id="dispatch-1", proposal_id="proposal-1")


def _png() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 32, 32)


class FakeAttemptStore:
    """Mirrors the stage-store contract of app/store.py closely enough for step logic."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.claim_replay: dict | None = None
        self.claims: list[str] = []
        self.finished: list[tuple[str, dict]] = []
        self.execution_receipts: list = []
        self.stage_writes = 0

    def add(self, *, attempt_id="attempt-1", request=REQUEST, outcome="active", stage=None, result=None):
        self.rows[attempt_id] = {
            "dispatch_id": request.dispatch_id, "attempt_id": attempt_id, "outcome": outcome,
            "request_json": request.model_dump(mode="json"), "result_json": result,
            "stage_json": dict(stage or {}),
        }
        return self.rows[attempt_id]

    def load_visual_attempt_for_dispatch(self, request):
        for row in self.rows.values():
            if row["dispatch_id"] == request.dispatch_id:
                if row["request_json"] != request.model_dump(mode="json"):
                    return {**copy.deepcopy(row), "request_mismatch": True}
                return copy.deepcopy(row)
        return None

    def load_visual_attempt(self, attempt_id):
        row = self.rows.get(attempt_id)
        return copy.deepcopy(row) if row else None

    def update_visual_stage(self, attempt_id, mutate, *, release_abandoned=False):
        row = self.rows.get(attempt_id)
        if row is None:
            return None
        new = mutate(copy.deepcopy(row["stage_json"]), copy.deepcopy(row))
        if new is not None:
            row["stage_json"] = new
            self.stage_writes += 1
        if release_abandoned and row["outcome"] == "unknown" and row["stage_json"].get("abandoned_at"):
            row["outcome"] = "abandoned"
        return copy.deepcopy(row)

    def claim_visual_attempt(self, request, *, retry_sec, now, abandoned_in_flight_window_sec=None,
                             attempt_max_age_sec=None):
        assert abandoned_in_flight_window_sec and abandoned_in_flight_window_sec > 0
        assert attempt_max_age_sec and attempt_max_age_sec > 5400
        if self.claim_replay is not None:
            return None, self.claim_replay
        attempt_id = f"attempt-{len(self.claims) + 1}"
        self.claims.append(attempt_id)
        self.add(attempt_id=attempt_id, request=request)
        return attempt_id, None

    def finish_visual_attempt(self, attempt_id, result):
        self.finished.append((attempt_id, result))
        self.rows[attempt_id]["outcome"] = result["outcome"]
        self.rows[attempt_id]["result_json"] = result

    def persist_visual_execution_receipt(self, chain_id, receipt):
        self.execution_receipts.append((chain_id, receipt))
        return True

    def load_visual_retry_after_sec(self, now):
        return 420.0


@pytest.fixture
def env(monkeypatch, tmp_path):
    from app import store, visual_chain, visual_steps

    fake = FakeAttemptStore()
    for name in ("load_visual_attempt_for_dispatch", "load_visual_attempt", "update_visual_stage",
                 "claim_visual_attempt", "finish_visual_attempt", "persist_visual_execution_receipt",
                 "load_visual_retry_after_sec"):
        monkeypatch.setattr(store, name, getattr(fake, name))
    monkeypatch.setattr(visual_steps, "_unrecorded_renders", {})
    monkeypatch.setattr(visual_chain.settings, "thermal_gate_enabled", False)
    monkeypatch.setattr(visual_chain.settings, "visual_chain_enabled", False)
    monkeypatch.setattr(visual_chain.settings, "visual_chain_interpretation_enabled", False)
    monkeypatch.setattr(visual_chain.settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_visual_chain_continuity_state",
                        lambda **kw: ("a fox by the fire", 1, 4, "prior-chain"))
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: "a real thought")
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    generate_calls: list[str] = []

    def fake_generate(prompt, *, base_url, timeout_sec):
        generate_calls.append(prompt)
        return _png()

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    holds: list = []

    async def fake_validate(bus, ref, *, source, expected_holder=None, timeout_sec=None):
        holds.append((ref, expected_holder))
        if ref.holder != expected_holder:
            raise visual_steps.LeaseUnavailable("gpu_lease_holder_mismatch", ref.lease_id)

    monkeypatch.setattr(visual_steps, "validate_hold_ref", fake_validate)
    return SimpleNamespace(store=fake, vc=visual_chain, steps=visual_steps, generate_calls=generate_calls,
                           holds=holds, tmp_path=tmp_path)


def _req(step, *, attempt_id=None, holder=None, request=REQUEST, correlation_id="corr-1"):
    lease = None
    if step == "generate":
        from orion.gpu_pool.client import durable_run_holder
        lease = GpuLeaseRefV1(lease_id="lease-1", generation=1, role="diffusion",
                              holder=holder or durable_run_holder(RUN_ID))
    return ReverieVisualStepRequestV1(run_id=RUN_ID, correlation_id=correlation_id, step=step,
                                      visual_request=request, attempt_id=attempt_id, gpu_lease=lease)


async def _run(env, req, **kw):
    # A bus: generate attaches its diffusion lease through it (conftest's granting fake pool).
    return await env.steps.run_visual_step(AsyncMock(), req, now_fn=lambda: NOW, **kw)


async def _prepared(env):
    result = await _run(env, _req("prepare"))
    assert result.status == "done", result
    return result.attempt_id


async def _generated(env):
    attempt_id = await _prepared(env)
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert result.status == "done", result
    return attempt_id, result


# ── prepare ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prepare_claims_and_freezes_the_plan(env):
    result = await _run(env, _req("prepare"))
    assert result.status == "done"
    assert result.attempt_id == env.store.claims[0]
    stage = env.store.rows[result.attempt_id]["stage_json"]
    assert stage["stage"] == "prepared"
    plan = stage["plan"]
    assert "a fox by the fire" in plan["prompt"]
    assert plan["continuity_streak"] == 2  # next value, recorded for the next run
    assert plan["context_slot_rotation"] == 5
    assert plan["context_slot_used"] == "context"
    assert plan["prior_chain_id"] == "prior-chain"
    assert env.generate_calls == []  # prepare never touches the GPU


@pytest.mark.asyncio
async def test_prepare_replay_returns_frozen_plan_without_rereading_or_reinterpreting(env, monkeypatch):
    attempt_id = await _prepared(env)
    frozen = copy.deepcopy(env.store.rows[attempt_id]["stage_json"]["plan"])

    def boom(**kw):
        raise AssertionError("replayed prepare must not re-read context or advance rotation")

    monkeypatch.setattr(env.vc, "load_latest_visual_chain_continuity_state", boom)
    interpret = AsyncMock(side_effect=AssertionError("replayed prepare must not re-interpret"))
    monkeypatch.setattr(env.vc, "interpret_context_for_visual", interpret)
    replay = await _run(env, _req("prepare", correlation_id="corr-2"))
    assert replay.status == "done"
    assert replay.attempt_id == attempt_id
    assert env.store.claims == [attempt_id]  # no second claim
    assert env.store.rows[attempt_id]["stage_json"]["plan"] == frozen
    interpret.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_final_attempt_is_terminal_with_its_outcome(env):
    env.store.add(outcome="failed", result={"outcome": "failed", "reason": "generation_failed"})
    result = await _run(env, _req("prepare"))
    assert result.status == "terminal"
    assert result.outcome == "failed"
    assert result.reason == "generation_failed"


@pytest.mark.asyncio
async def test_prepare_reuses_an_unknown_attempt_for_the_same_dispatch(env):
    env.store.add(attempt_id="legacy-attempt", outcome="unknown",
                  result={"outcome": "unknown", "reason": "execution_unresolved"})
    result = await _run(env, _req("prepare"))
    assert result.status == "done"
    assert result.attempt_id == "legacy-attempt"
    assert env.store.claims == []


@pytest.mark.asyncio
async def test_prepare_dispatch_request_mismatch_is_terminal(env):
    env.store.add(request=REQUEST.model_copy(update={"proposal_id": "other"}))
    result = await _run(env, _req("prepare"))
    assert (result.status, result.outcome, result.reason) == ("terminal", "failed", "dispatch_request_mismatch")


def _baseline_request(monkeypatch, env, *, observed_at, due_at):
    from orion.reverie.baseline import VisualBaselinePolicy
    from orion.schemas.reverie_visual import VisualBaselineEligibilityV1

    policy = VisualBaselinePolicy(enabled=True)
    monkeypatch.setattr(env.steps, "load_baseline_policy", lambda: policy)
    need = VisualBaselineEligibilityV1(need_id="need-1", observed_at=observed_at, due_at=due_at,
                                       policy_id=policy.policy_id)
    return REQUEST.model_copy(update={"visual_baseline": need})


@pytest.mark.asyncio
async def test_prepare_waits_while_legacy_worker_enabled(env, monkeypatch):
    request = _baseline_request(monkeypatch, env, observed_at=NOW, due_at=NOW - timedelta(seconds=1))
    monkeypatch.setattr(env.vc.settings, "visual_chain_enabled", True)
    result = await _run(env, _req("prepare", request=request))
    assert (result.status, result.reason) == ("retry", "legacy_visual_worker_enabled")
    assert result.retry_after_sec == 300.0
    assert env.store.claims == []


@pytest.mark.asyncio
async def test_prepare_retried_long_after_the_baseline_was_observed_still_claims(env, monkeypatch):
    # The run waited out retries for hours: the request is still the same request.
    observed = NOW - timedelta(hours=2)
    request = _baseline_request(monkeypatch, env, observed_at=observed, due_at=observed - timedelta(seconds=1))
    result = await _run(env, _req("prepare", request=request))
    assert result.status == "done", result
    assert env.store.claims == [result.attempt_id]


@pytest.mark.asyncio
async def test_prepare_claims_when_due_after_thought_observed_activity(env, monkeypatch):
    # First activation: the scheduler sets due_at = its own now, later than thought's
    # observed_at. That need is due, not "not due".
    observed = NOW - timedelta(seconds=40)
    request = _baseline_request(monkeypatch, env, observed_at=observed,
                                due_at=observed + timedelta(milliseconds=40))
    result = await _run(env, _req("prepare", request=request))
    assert result.status == "done", result
    assert env.store.claims == [result.attempt_id]


@pytest.mark.asyncio
async def test_prepare_before_due_waits_until_due(env, monkeypatch):
    # A few seconds of clock skew with proposal-runtime: wait, do not fail.
    request = _baseline_request(monkeypatch, env, observed_at=NOW, due_at=NOW + timedelta(seconds=5))
    result = await _run(env, _req("prepare", request=request))
    assert (result.status, result.reason) == ("retry", "visual_baseline_not_due")
    assert result.retry_after_sec == 5.0
    assert env.store.claims == []


@pytest.mark.asyncio
async def test_prepare_structurally_ineligible_baseline_is_terminal(env, monkeypatch):
    from orion.schemas.reverie_visual import VisualBaselineEligibilityV1

    request = _baseline_request(monkeypatch, env, observed_at=NOW, due_at=NOW - timedelta(seconds=1))
    wrong_policy = VisualBaselineEligibilityV1.model_validate(
        {**request.visual_baseline.model_dump(), "policy_id": "some-other-policy"})
    request = request.model_copy(update={"visual_baseline": wrong_policy})
    result = await _run(env, _req("prepare", request=request))
    assert (result.status, result.outcome, result.reason) == (
        "terminal", "failed", "visual_baseline_policy_mismatch")
    assert env.store.claims == []


@pytest.mark.asyncio
async def test_legacy_run_once_claim_applies_the_same_release_rules(env, monkeypatch):
    import json

    from app import main, store
    from orion.reverie import baseline

    monkeypatch.setattr(baseline, "load_baseline_policy", lambda: baseline.VisualBaselinePolicy(enabled=True))
    seen = {}

    def claim(request, **kw):
        seen.update(kw)
        return None, {"ok": True, "ran": False, "outcome": "deferred_busy", "reason": "attempt_unresolved"}

    monkeypatch.setattr(store, "claim_visual_attempt", claim)
    body = json.loads((await main.visual_chain_run_once(VisualRunRequestV1())).body)
    assert body["reason"] == "attempt_unresolved"
    assert seen["abandoned_in_flight_window_sec"] == env.steps._in_flight_window_sec()
    assert seen["attempt_max_age_sec"] == env.vc.settings.visual_chain_attempt_max_age_sec


def test_attempt_max_age_default_outlasts_the_durable_retry_window():
    from app.settings import ThoughtSettings
    from orion.execution_dispatch.visual_settlement import DEFAULT_RETRY_WINDOW_SEC
    from orion.schemas.reverie_visual_run import REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC

    default = ThoughtSettings.model_fields["visual_chain_attempt_max_age_sec"].default
    assert default == 7200.0 and default > DEFAULT_RETRY_WINDOW_SEC
    # cortex-exec clamps every run's retry window to this cap.
    assert default > REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC >= DEFAULT_RETRY_WINDOW_SEC


@pytest.mark.asyncio
@pytest.mark.parametrize("replay,status,reason,retry_after", [
    ({"outcome": "deferred_busy", "reason": "attempt_unresolved", "attempt_id": "other"}, "retry",
     "attempt_unresolved", None),
    ({"outcome": "deferred_busy", "reason": "retry_cooldown"}, "retry", "retry_cooldown", 420.0),
    ({"outcome": "already_satisfied", "reason": "activity_changed"}, "terminal", "activity_changed", None),
])
async def test_prepare_claim_refusals(env, replay, status, reason, retry_after):
    env.store.claim_replay = replay
    result = await _run(env, _req("prepare"))
    assert (result.status, result.reason, result.retry_after_sec) == (status, reason, retry_after)
    if status == "terminal":
        assert result.outcome == "already_satisfied"


@pytest.mark.asyncio
async def test_missing_stage_column_is_a_retry_never_a_recompute(env, monkeypatch):
    from app import store

    def unavailable(*a, **kw):
        raise store.VisualStageStoreUnavailable("stage_json missing")

    monkeypatch.setattr(store, "load_visual_attempt_for_dispatch", unavailable)
    result = await _run(env, _req("prepare"))
    assert (result.status, result.reason) == ("retry", "stage_store_unavailable")
    assert env.store.claims == []


# ── generate ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_attaches_under_the_runs_hold(env, gpu_pool):
    """GPU pool stage 5.4: the diffusion call attaches under the run's validated hold (no second
    wait, never durable-runs /capacity) and the child lease covers exactly the diffusion call."""
    attempt_id, result = await _generated(env)
    assert result.status == "done"
    [call] = gpu_pool.calls
    [(ref, _)] = env.holds[-1:]
    assert call["hold"] == ref and call["work_class"] == "diffusion"
    assert len(env.generate_calls) == 1


@pytest.mark.asyncio
async def test_generate_records_the_image_and_reports_its_work_time(env):
    attempt_id, result = await _generated(env)
    stage = env.store.rows[attempt_id]["stage_json"]
    assert stage["stage"] == "generated"
    assert result.artifact_sha256 == stage["artifact"]["sha256"]
    assert Path(stage["artifact"]["path"]).exists()
    assert result.elapsed_sec is not None and result.elapsed_sec >= 0
    assert len(env.generate_calls) == 1
    assert env.generate_calls[0] == stage["plan"]["prompt"]


@pytest.mark.asyncio
async def test_generate_replay_with_recorded_artifact_makes_no_gpu_call(env, monkeypatch):
    attempt_id, first = await _generated(env)

    def boom(*a, **kw):
        raise AssertionError("replayed generate must not call diffusion")

    monkeypatch.setattr(env.vc, "call_diffusion_generate", boom)
    replay = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-2"))
    assert replay.status == "done"
    assert replay.artifact_sha256 == first.artifact_sha256
    assert replay.elapsed_sec == env.store.rows[attempt_id]["stage_json"]["artifact"]["elapsed_sec"]


@pytest.mark.asyncio
async def test_generate_in_flight_guard_blocks_a_second_diffusion_call(env, monkeypatch):
    attempt_id = await _prepared(env)
    env.store.rows[attempt_id]["stage_json"].update(
        stage="generating", generating_started_at=(NOW - timedelta(seconds=30)).isoformat())
    monkeypatch.setattr(env.vc, "call_diffusion_generate",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("second diffusion call")))
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", "generate_in_flight")
    window = 2 * env.steps.visual_step_generate_deadline_sec()
    assert result.retry_after_sec == pytest.approx(window - 30)


@pytest.mark.asyncio
async def test_generate_stale_in_flight_marker_proceeds(env):
    attempt_id = await _prepared(env)
    window = 2 * env.steps.visual_step_generate_deadline_sec()
    env.store.rows[attempt_id]["stage_json"].update(
        stage="generating", generating_started_at=(NOW - timedelta(seconds=window + 1)).isoformat())
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert result.status == "done"
    assert len(env.generate_calls) == 1


@pytest.mark.asyncio
async def test_generate_thermal_refusal_is_a_retry_and_writes_no_chain_row(env, monkeypatch):
    attempt_id = await _prepared(env)
    persisted = []
    monkeypatch.setattr(env.vc, "persist_reverie_visual_chain", lambda c: persisted.append(c) or True)
    monkeypatch.setattr(env.vc.settings, "thermal_gate_enabled", True)
    monkeypatch.setattr(env.vc, "_thermal_state", "normal")
    monkeypatch.setattr(env.vc, "read_cabinet_temp_c", lambda: (36.0, 1.0))
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", "thermal_refused")
    assert result.retry_after_sec and result.retry_after_sec > 0
    assert persisted == []
    assert env.generate_calls == []
    stage = env.store.rows[attempt_id]["stage_json"]
    assert stage["stage"] == "prepared"
    assert stage["deferrals"][-1]["reason"] == "thermal_refused"


@pytest.mark.asyncio
async def test_generate_hold_ref_mismatch_is_retry_and_never_touches_the_gpu(env):
    attempt_id = await _prepared(env)
    result = await _run(env, _req("generate", attempt_id=attempt_id, holder="durable-runs:someone-else"))
    assert (result.status, result.reason) == ("retry", "hold_invalid:gpu_lease_holder_mismatch")
    assert env.generate_calls == []
    assert env.store.rows[attempt_id]["stage_json"]["stage"] == "prepared"


@pytest.mark.asyncio
async def test_generate_requires_prepare(env):
    env.store.add()
    result = await _run(env, _req("generate", attempt_id="attempt-1"))
    assert (result.status, result.reason) == ("retry", "not_prepared")
    assert env.generate_calls == []


@pytest.mark.asyncio
async def test_generate_for_another_dispatch_is_attempt_mismatch(env):
    env.store.add(attempt_id="foreign", request=VisualRunRequestV1(dispatch_id="other-dispatch"))
    result = await _run(env, _req("generate", attempt_id="foreign"))
    assert (result.status, result.outcome, result.reason) == ("terminal", "failed", "attempt_mismatch")


@pytest.mark.asyncio
async def test_generate_resource_deferral_and_failure_are_retries_without_chain_rows(env, monkeypatch):
    attempt_id = await _prepared(env)
    persisted = []
    monkeypatch.setattr(env.vc, "persist_reverie_visual_chain", lambda c: persisted.append(c) or True)

    def deferred(*a, **kw):
        raise env.vc.DiffusionResourceDeferred("controller_displacement")

    monkeypatch.setattr(env.vc, "call_diffusion_generate", deferred)
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", "resource_deferred:controller_displacement")

    def broken(*a, **kw):
        raise env.vc.DiffusionGenerationError("HTTP 500")

    monkeypatch.setattr(env.vc, "call_diffusion_generate", broken)
    result = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-2"))
    assert (result.status, result.reason) == ("retry", "generation_failed:HTTP 500")
    stage = env.store.rows[attempt_id]["stage_json"]
    assert stage["stage"] == "prepared"  # a finished failure never blocks the next generate
    assert [d["reason"] for d in stage["deferrals"]] == [
        "resource_deferred:controller_displacement", "generation_failed:HTTP 500"]
    assert persisted == []


@pytest.mark.asyncio
async def test_generate_busy_lock_is_deferred_busy(env):
    attempt_id = await _prepared(env)
    async with env.vc._visual_chain_lock:
        result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", "deferred_busy")
    assert env.generate_calls == []


@pytest.mark.asyncio
async def test_generate_deadline_leaves_the_work_running_until_it_records_its_exit(env, monkeypatch):
    import asyncio

    attempt_id = await _prepared(env)
    monkeypatch.setattr(env.steps, "visual_step_generate_deadline_sec", lambda: 0.5)
    release = asyncio.Event()
    diffusion_calls = []

    async def slow(prompt, *, correlation_id, hold=None, bus=None):
        diffusion_calls.append(prompt)
        await release.wait()
        return _png()

    monkeypatch.setattr(env.vc, "generate_visual_bytes", slow)
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", "generate_deadline_exceeded")
    assert env.store.rows[attempt_id]["stage_json"]["stage"] == "generating"
    assert env.vc._visual_chain_lock.locked()  # the card is still busy; so is the lock

    again = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-2"))
    assert (again.status, again.reason) == ("retry", "generate_in_flight")

    release.set()
    await asyncio.gather(*list(env.steps._generate_tasks))
    assert not env.vc._visual_chain_lock.locked()
    assert env.store.rows[attempt_id]["stage_json"]["stage"] == "generated"
    replay = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-3"))
    assert replay.status == "done" and replay.artifact_sha256
    assert len(diffusion_calls) == 1


@pytest.mark.asyncio
async def test_hung_diffusion_releases_the_lock_at_the_in_flight_ceiling(env, monkeypatch):
    import asyncio

    attempt_id = await _prepared(env)
    monkeypatch.setattr(env.steps, "visual_step_generate_deadline_sec", lambda: 0.05)
    monkeypatch.setattr(env.steps, "_in_flight_window_sec", lambda: 0.3)

    async def hung(prompt, *, correlation_id, hold=None, bus=None):
        await asyncio.sleep(30)

    monkeypatch.setattr(env.vc, "generate_visual_bytes", hung)
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert result.reason == "generate_deadline_exceeded"
    await asyncio.gather(*list(env.steps._generate_tasks))
    assert not env.vc._visual_chain_lock.locked()
    stage = env.store.rows[attempt_id]["stage_json"]
    assert stage["stage"] == "prepared"
    assert stage["deferrals"][-1]["reason"] == "generate_wedged"


@pytest.mark.asyncio
async def test_generate_does_not_start_when_the_attempt_is_abandoned_meanwhile(env, monkeypatch):
    attempt_id = await _prepared(env)

    async def thermal_then_abandon():
        row = env.store.rows[attempt_id]
        row["outcome"] = "abandoned"
        row["stage_json"]["abandoned_at"] = NOW.isoformat()
        return {"state": "disabled", "allows_gpu_work": True}

    monkeypatch.setattr(env.vc, "thermal_gate_snapshot", thermal_then_abandon)
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", "stage_changed")
    assert env.generate_calls == []
    assert env.store.rows[attempt_id]["stage_json"]["stage"] == "prepared"


@pytest.mark.asyncio
async def test_generate_does_not_rerender_when_another_generate_finished_first(env, monkeypatch):
    attempt_id = await _prepared(env)

    async def thermal_then_concurrent_generate():
        env.store.rows[attempt_id]["stage_json"].update(stage="generated", artifact={"sha256": "a" * 64})
        return {"state": "disabled", "allows_gpu_work": True}

    monkeypatch.setattr(env.vc, "thermal_gate_snapshot", thermal_then_concurrent_generate)
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", "stage_changed")
    assert env.generate_calls == []
    assert env.store.rows[attempt_id]["stage_json"]["artifact"] == {"sha256": "a" * 64}


def _failing_generated_writes(env, monkeypatch, failures):
    """Make the stage write that records a finished render raise `failures[0]` times."""
    real = env.store.update_visual_stage

    def flaky(attempt_id, mutate, *, release_abandoned=False):
        row = env.store.rows.get(attempt_id)
        probe = mutate(copy.deepcopy(row["stage_json"]), copy.deepcopy(row)) if row else None
        if probe is not None and probe.get("stage") == "generated" and failures[0] > 0:
            failures[0] -= 1
            raise RuntimeError("db blip")
        return real(attempt_id, mutate, release_abandoned=release_abandoned)

    monkeypatch.setattr(env.steps.store, "update_visual_stage", flaky)
    monkeypatch.setattr(env.steps, "_STAGE_WRITE_BACKOFF_SEC", 0.0)


@pytest.mark.asyncio
async def test_generate_retries_a_transient_stage_write_instead_of_losing_the_render(env, monkeypatch):
    attempt_id = await _prepared(env)
    failures = [env.steps._STAGE_WRITE_ATTEMPTS - 1]
    _failing_generated_writes(env, monkeypatch, failures)
    result = await _run(env, _req("generate", attempt_id=attempt_id))
    assert result.status == "done", result
    assert failures == [0]
    assert env.store.rows[attempt_id]["stage_json"]["artifact"]["sha256"] == result.artifact_sha256
    assert len(env.generate_calls) == 1


@pytest.mark.asyncio
async def test_unrecorded_render_is_adopted_by_the_next_generate_not_rerendered(env, monkeypatch):
    attempt_id = await _prepared(env)
    failures = [env.steps._STAGE_WRITE_ATTEMPTS]
    _failing_generated_writes(env, monkeypatch, failures)
    first = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (first.status, first.reason) == ("retry", "stage_store_unavailable")
    assert first.artifact_sha256 and first.attempt_id == attempt_id
    assert Path(env.steps._unrecorded_renders[attempt_id]["artifact"]["path"]).exists()
    assert env.store.rows[attempt_id]["stage_json"]["stage"] == "generating"
    env.steps._unrecorded_renders[attempt_id]["artifact"]["elapsed_sec"] = 42.5

    # Still inside the in-flight window, yet the verified file is adopted, not re-rendered.
    again = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-2"))
    assert again.status == "done", again
    assert again.artifact_sha256 == first.artifact_sha256
    assert len(env.generate_calls) == 1
    # The earlier try's GPU time, not the adopting step's near-zero wall time.
    assert again.elapsed_sec == 42.5
    stage = env.store.rows[attempt_id]["stage_json"]
    assert (stage["stage"], stage["artifact"]["sha256"]) == ("generated", first.artifact_sha256)
    assert attempt_id not in env.steps._unrecorded_renders


@pytest.mark.asyncio
async def test_reprepare_of_unreadable_plan_never_rewinds_a_generated_attempt(env):
    attempt_id, first = await _generated(env)
    env.store.rows[attempt_id]["stage_json"]["plan"] = "unreadable-after-a-schema-change"
    generate = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-2"))
    assert (generate.status, generate.reason) == ("retry", "not_prepared")

    prepare = await _run(env, _req("prepare", correlation_id="corr-3"))
    assert prepare.status == "done", prepare
    assert env.store.rows[attempt_id]["stage_json"]["stage"] == "generated"
    again = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-4"))
    assert again.status == "done", again
    assert again.artifact_sha256 == first.artifact_sha256
    assert len(env.generate_calls) == 1  # the recorded image is replayed, never re-rendered


@pytest.mark.asyncio
async def test_unrecorded_render_whose_file_is_gone_is_not_adopted(env, monkeypatch):
    attempt_id = await _prepared(env)
    _failing_generated_writes(env, monkeypatch, [env.steps._STAGE_WRITE_ATTEMPTS])
    first = await _run(env, _req("generate", attempt_id=attempt_id))
    Path(env.steps._unrecorded_renders[attempt_id]["artifact"]["path"]).unlink()
    again = await _run(env, _req("generate", attempt_id=attempt_id, correlation_id="corr-2"))
    # Falls through to the in-flight guard: never reports an image it cannot verify.
    assert (again.status, again.reason) == ("retry", "generate_in_flight")
    assert attempt_id not in env.steps._unrecorded_renders
    assert first.artifact_sha256 and len(env.generate_calls) == 1


@pytest.mark.asyncio
async def test_run_once_judges_a_generate_lock_holder_by_its_own_deadline(env, caplog):
    import logging
    import time

    async with env.vc.visual_chain_single_flight(deadline_sec=1000.0) as held:
        assert held
        env.vc._visual_chain_started_at = time.monotonic() - 400  # past run-once's 300s
        with caplog.at_level(logging.INFO, logger=env.vc.logger.name):
            assert await env.vc.run_visual_chain_once(None) is None
    assert not [r for r in caplog.records if "PAST its" in r.getMessage()]


# ── caption ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_caption_before_generate_needs_generate(env):
    attempt_id = await _prepared(env)
    result = await _run(env, _req("caption", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", NEEDS_GENERATE)


@pytest.mark.asyncio
async def test_caption_with_missing_file_resets_to_prepared_and_needs_generate(env):
    attempt_id, _ = await _generated(env)
    Path(env.store.rows[attempt_id]["stage_json"]["artifact"]["path"]).unlink()
    result = await _run(env, _req("caption", attempt_id=attempt_id))
    assert (result.status, result.reason) == ("retry", NEEDS_GENERATE)
    stage = env.store.rows[attempt_id]["stage_json"]
    assert stage["stage"] == "prepared"
    assert "artifact" not in stage


def _capture_production(env, monkeypatch):
    persisted, acknowledged = [], []

    def ack(chain, artifact):
        from orion.schemas.reverie_visual import VisualProductionReceiptV1
        acknowledged.append(artifact)
        return VisualProductionReceiptV1(chain_id=chain.chain_id, attempt_id=chain.chain_id,
                                         sha256=artifact.sha256, bytes=artifact.bytes,
                                         path=artifact.path, produced_at=artifact.created_at)

    monkeypatch.setattr(env.vc, "persist_reverie_visual_chain", lambda c: persisted.append(c) or True)
    monkeypatch.setattr(env.vc, "acknowledge_visual_production", ack)
    monkeypatch.setattr(env.vc, "upload_to_percept_store", lambda data, **kw: "e" * 64)
    return persisted, acknowledged


def _caption(env, monkeypatch, text):
    captioner = AsyncMock(return_value=text)
    monkeypatch.setattr(env.vc, "request_caption", captioner)
    return captioner


@pytest.mark.asyncio
async def test_caption_happy_path_writes_the_production_row(env, monkeypatch):
    attempt_id, generated = await _generated(env)
    persisted, acknowledged = _capture_production(env, monkeypatch)
    _caption(env, monkeypatch, "a fox asleep in the embers")
    result = await _run(env, _req("caption", attempt_id=attempt_id))

    assert result.status == "done"
    assert (result.outcome, result.chain_id) == ("produced", attempt_id)
    assert result.artifact_sha256 == generated.artifact_sha256
    [chain] = persisted
    assert chain.chain_id == attempt_id  # chain_id == attempt_id invariant
    assert chain.terminal_reason == "max_steps"
    assert chain.prior_description == "a fox asleep in the embers"
    plan = env.store.rows[attempt_id]["stage_json"]["plan"]
    for key in ("prompt", "continuity_streak", "context_slot_rotation", "context_slot_used",
                "context_text", "continuity_reset"):
        assert chain.chain_json[key] == plan[key]
    assert chain.chain_json["run_request"] == REQUEST.model_dump(mode="json")
    assert chain.chain_json["production_receipt"]["attempt_id"] == attempt_id
    assert acknowledged[0].chain_id == attempt_id
    [(receipt_chain, receipt)] = env.store.execution_receipts
    assert receipt_chain == attempt_id
    assert receipt.outcome == "produced" and receipt.artifact_persisted
    assert result.execution_receipt == receipt.model_dump(mode="json")
    [(finished_id, finished)] = env.store.finished
    assert finished_id == attempt_id and finished["outcome"] == "produced"
    assert env.generate_calls and len(env.generate_calls) == 1


@pytest.mark.asyncio
async def test_caption_retry_after_ack_failure_does_not_recaption(env, monkeypatch):
    attempt_id, _ = await _generated(env)
    persisted, _ = _capture_production(env, monkeypatch)
    monkeypatch.setattr(env.vc, "acknowledge_visual_production", lambda c, a: None)
    _caption(env, monkeypatch, "a quiet ember")
    first = await _run(env, _req("caption", attempt_id=attempt_id))
    assert (first.status, first.reason) == ("retry", "acknowledge_failed")
    assert env.store.rows[attempt_id]["stage_json"]["caption"]["description"] == "a quiet ember"
    assert env.store.finished == []
    persisted, _ = _capture_production(env, monkeypatch)
    recaption = _caption(env, monkeypatch, "a different caption")
    second = await _run(env, _req("caption", attempt_id=attempt_id, correlation_id="c2"))
    assert second.status == "done"
    recaption.assert_not_called()
    assert persisted[-1].prior_description == "a quiet ember"


@pytest.mark.asyncio
async def test_caption_replay_after_production_returns_the_recorded_result(env, monkeypatch):
    attempt_id, _ = await _generated(env)
    _capture_production(env, monkeypatch)
    _caption(env, monkeypatch, "x")
    first = await _run(env, _req("caption", attempt_id=attempt_id))
    monkeypatch.setattr(env.vc, "persist_reverie_visual_chain",
                        lambda c: (_ for _ in ()).throw(AssertionError("no second write")))
    replay = await _run(env, _req("caption", attempt_id=attempt_id, correlation_id="c2"))
    assert replay.status == "done"
    assert (replay.outcome, replay.chain_id, replay.artifact_sha256) == ("produced", attempt_id,
                                                                          first.artifact_sha256)


# ── abandon / exceptions ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abandon_closes_the_attempt(env, monkeypatch):
    from app import store

    calls = []

    def abandon(attempt_id, *, dispatch_id, reason, now, in_flight_window_sec, request_json=None):
        calls.append((attempt_id, dispatch_id, reason))
        row = env.store.rows[attempt_id]
        row["outcome"] = "abandoned"
        row["result_json"] = {"outcome": "unknown", "reason": reason}
        return copy.deepcopy(row)

    monkeypatch.setattr(store, "abandon_visual_attempt", abandon)
    attempt_id = await _prepared(env)
    result = await _run(env, _req("abandon", attempt_id=attempt_id))
    assert (result.status, result.outcome, result.reason) == ("done", "unknown", "run_abandoned")
    assert calls == [(attempt_id, "dispatch-1", "run_abandoned")]
    # A later step on the abandoned attempt ends the run instead of doing work.
    later = await _run(env, _req("generate", attempt_id=attempt_id))
    assert (later.status, later.outcome) == ("terminal", "unknown")
    assert env.generate_calls == []


@pytest.mark.asyncio
async def test_abandon_without_attempt_id_resolves_the_attempt_by_dispatch(env, monkeypatch):
    from app import store

    calls = []

    def abandon(attempt_id, *, dispatch_id, reason, now, in_flight_window_sec, request_json=None):
        calls.append((attempt_id, dispatch_id, request_json))
        [row] = [r for r in env.store.rows.values() if r["dispatch_id"] == dispatch_id] or [None]
        if row is None:
            return None
        row["outcome"] = "abandoned"
        row["result_json"] = {"outcome": "unknown", "reason": reason}
        return copy.deepcopy(row)

    monkeypatch.setattr(store, "abandon_visual_attempt", abandon)
    # prepare's reply was lost: the run never learned the attempt id.
    missing = await _run(env, _req("abandon"))
    assert (missing.status, missing.reason, missing.attempt_id) == ("done", "attempt_missing", None)
    attempt_id = await _prepared(env)
    result = await _run(env, _req("abandon"))
    assert (result.status, result.outcome, result.attempt_id) == ("done", "unknown", attempt_id)
    assert calls[-1] == (None, "dispatch-1", REQUEST.model_dump(mode="json"))
    assert env.store.rows[attempt_id]["outcome"] == "abandoned"


@pytest.mark.asyncio
async def test_unexpected_exception_is_a_retry(env, monkeypatch):
    from app import store

    monkeypatch.setattr(store, "load_visual_attempt", lambda *a: (_ for _ in ()).throw(KeyError("x")))
    result = await _run(env, _req("caption", attempt_id="attempt-1"))
    assert (result.status, result.reason) == ("retry", "step_exception:KeyError")
    assert isinstance(result, ReverieVisualStepResultV1)


# ── dream hop mode (design 2026-10-10-dream-carry-through) ──────────────────

from orion.schemas.reverie_visual_run import DreamHopImageV1  # noqa: E402

DREAM_PROMPT = "a staircase of moths climbing into a lantern that is also the moon"
DREAM_HOP = DreamHopImageV1(carry_run_id="dream-carry-abc", hop_index=1, prompt=DREAM_PROMPT)
DREAM_REQUEST = VisualRunRequestV1(dispatch_id="dream-carry:dream-carry-abc:1")


def _dream_req(step, *, attempt_id=None, hop=DREAM_HOP, correlation_id="dream-corr-1"):
    req = _req(step, attempt_id=attempt_id, request=DREAM_REQUEST, correlation_id=correlation_id)
    return req.model_copy(update={"dream_hop": hop})


def _forbid_waking_planning(env, monkeypatch):
    """A dream hop must never read or advance waking continuity, rotation or interpret."""
    def boom(*a, **kw):
        raise AssertionError("dream hop read waking planning state")

    monkeypatch.setattr(env.vc, "compute_visual_plan", AsyncMock(side_effect=AssertionError("compute_visual_plan")))
    monkeypatch.setattr(env.vc, "interpret_context_for_visual", AsyncMock(side_effect=AssertionError("interpret")))
    for name in ("load_latest_visual_chain_continuity_state", "load_latest_reverie_interpretation",
                 "load_latest_self_study_reflection", "load_latest_memory_crystallization"):
        monkeypatch.setattr(env.vc, name, boom)


def _forbid_production_writes(env, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("dream hop wrote a waking production/chain row")

    monkeypatch.setattr(env.vc, "persist_reverie_visual_chain", boom)
    monkeypatch.setattr(env.vc, "acknowledge_visual_production", boom)
    monkeypatch.setattr(env.vc, "build_production_chain", boom)
    monkeypatch.setattr(env.vc, "upload_to_percept_store", lambda data, **kw: "e" * 64)


async def _dream_generated(env):
    prepared = await _run(env, _dream_req("prepare"))
    assert prepared.status == "done", prepared
    generated = await _run(env, _dream_req("generate", attempt_id=prepared.attempt_id))
    assert generated.status == "done", generated
    return prepared.attempt_id, generated


@pytest.mark.asyncio
async def test_dream_hop_prepare_freezes_the_prompt_verbatim_without_waking_planning(env, monkeypatch):
    _forbid_waking_planning(env, monkeypatch)
    result = await _run(env, _dream_req("prepare"))
    assert result.status == "done", result
    stage = env.store.rows[result.attempt_id]["stage_json"]
    plan = stage["plan"]
    assert plan["prompt"] == DREAM_PROMPT  # verbatim: no continuity prefix, no context slot
    assert (plan["prior_description"], plan["prior_chain_id"], plan["effective_prior"]) == (None, None, None)
    assert (plan["continuity_streak"], plan["context_slot_rotation"]) == (0, 0)
    assert (plan["context_slot_used"], plan["context_slot_interpreted"], plan["context_text"]) == (None, None, None)
    assert plan["context_selection"] is None
    assert stage["dream_hop"] == {"carry_run_id": "dream-carry-abc", "hop_index": 1}
    assert env.store.claims == [result.attempt_id]  # the same claim (one open attempt, cooldown)
    # Generate paints exactly that prompt.
    generated = await _run(env, _dream_req("generate", attempt_id=result.attempt_id))
    assert generated.status == "done"
    assert env.generate_calls == [DREAM_PROMPT]


@pytest.mark.asyncio
async def test_dream_hop_caption_returns_what_was_seen_and_writes_no_waking_rows(env, monkeypatch):
    attempt_id, generated = await _dream_generated(env)
    _forbid_production_writes(env, monkeypatch)
    captioner = _caption(env, monkeypatch, "moths settling on a pale lamp")
    result = await _run(env, _dream_req("caption", attempt_id=attempt_id))

    assert result.status == "done", result
    assert (result.outcome, result.caption) == ("produced", "moths settling on a pale lamp")
    assert result.artifact_sha256 == generated.artifact_sha256
    assert result.chain_id is None and result.execution_receipt is None
    assert env.store.execution_receipts == []
    [(finished_id, finished)] = env.store.finished
    assert finished_id == attempt_id
    assert finished["outcome"] == "produced" and finished["caption"] == "moths settling on a pale lamp"
    assert finished["artifact_sha256"] == generated.artifact_sha256
    assert finished["dream_hop"] == {"carry_run_id": "dream-carry-abc", "hop_index": 1}
    assert finished["durable_run_id"] == RUN_ID and "execution_receipt" not in finished

    # Replay (lost reply): same caption, no second caption call.
    replay = await _run(env, _dream_req("caption", attempt_id=attempt_id, correlation_id="dream-corr-2"))
    assert (replay.status, replay.outcome, replay.caption) == ("done", "produced", "moths settling on a pale lamp")
    assert replay.artifact_sha256 == generated.artifact_sha256 and replay.chain_id is None
    captioner.assert_awaited_once()


@pytest.mark.asyncio
async def test_waking_painting_after_a_dream_hop_sees_unchanged_continuity(env, monkeypatch):
    chains: list = []

    def continuity(**kw):
        if not chains:
            return ("a fox by the fire", 1, 4, "prior-chain")
        last = chains[-1]
        return (last.prior_description, last.chain_json["continuity_streak"],
                last.chain_json["context_slot_rotation"], last.chain_id)

    monkeypatch.setattr(env.vc, "load_latest_visual_chain_continuity_state", continuity)
    _capture_production(env, monkeypatch)
    monkeypatch.setattr(env.vc, "persist_reverie_visual_chain", lambda c: chains.append(c) or True)
    _caption(env, monkeypatch, "moths settling on a pale lamp")

    attempt_id, _ = await _dream_generated(env)
    dream = await _run(env, _dream_req("caption", attempt_id=attempt_id))
    assert (dream.status, dream.outcome) == ("done", "produced")
    assert chains == []  # the dream hop wrote no chain row

    waking = await _run(env, _req("prepare"))
    plan = env.store.rows[waking.attempt_id]["stage_json"]["plan"]
    assert plan["prior_description"] == "a fox by the fire"  # not the dream's caption
    assert plan["prior_chain_id"] == "prior-chain"
    assert (plan["continuity_streak"], plan["context_slot_rotation"]) == (2, 5)
    assert "dream_hop" not in env.store.rows[waking.attempt_id]["stage_json"]


@pytest.mark.asyncio
async def test_dream_hop_empty_caption_is_a_retry_and_recaptions(env, monkeypatch):
    attempt_id, _ = await _dream_generated(env)
    _forbid_production_writes(env, monkeypatch)
    _caption(env, monkeypatch, "   ")
    first = await _run(env, _dream_req("caption", attempt_id=attempt_id))
    assert (first.status, first.reason, first.caption) == ("retry", "caption_empty", None)
    assert env.store.finished == []
    stage = env.store.rows[attempt_id]["stage_json"]
    assert "caption" not in stage  # never cached
    assert stage["deferrals"][-1]["reason"] == "caption_empty"

    recaption = _caption(env, monkeypatch, "a lantern full of wings")
    second = await _run(env, _dream_req("caption", attempt_id=attempt_id, correlation_id="dream-corr-2"))
    assert (second.status, second.caption) == ("done", "a lantern full of wings")
    recaption.assert_awaited_once()


@pytest.mark.asyncio
async def test_dream_hop_cached_empty_caption_is_recaptioned(env, monkeypatch):
    attempt_id, _ = await _dream_generated(env)
    _forbid_production_writes(env, monkeypatch)
    env.store.rows[attempt_id]["stage_json"]["caption"] = {"description": None, "captioned_at": NOW.isoformat()}
    recaption = _caption(env, monkeypatch, "a lantern full of wings")
    result = await _run(env, _dream_req("caption", attempt_id=attempt_id))
    assert (result.status, result.caption) == ("done", "a lantern full of wings")
    recaption.assert_awaited_once()


@pytest.mark.asyncio
async def test_dream_hop_request_for_a_different_hop_is_dispatch_request_mismatch(env, monkeypatch):
    attempt_id, _ = await _dream_generated(env)
    _forbid_production_writes(env, monkeypatch)
    _caption(env, monkeypatch, "x")
    other = DreamHopImageV1(carry_run_id="dream-carry-abc", hop_index=3, prompt=DREAM_PROMPT)
    reprompt = DREAM_HOP.model_copy(update={"prompt": "a different picture"})
    for hop in (other, reprompt, None):
        for step in ("prepare", "generate", "caption"):
            req = _dream_req(step, attempt_id=None if step == "prepare" else attempt_id, hop=hop)
            result = await _run(env, req)
            assert (result.status, result.outcome, result.reason) == (
                "terminal", "failed", "dispatch_request_mismatch"), (hop, step, result)
    assert env.store.finished == []
    # Still mismatched once produced: another hop never receives this hop's picture.
    done = await _run(env, _dream_req("caption", attempt_id=attempt_id))
    assert done.status == "done"
    late = await _run(env, _dream_req("caption", attempt_id=attempt_id, hop=other))
    assert (late.status, late.reason) == ("terminal", "dispatch_request_mismatch")


@pytest.mark.asyncio
async def test_dream_hop_closed_attempt_replays_its_caption(env, monkeypatch):
    attempt_id, _ = await _dream_generated(env)
    _forbid_production_writes(env, monkeypatch)
    _caption(env, monkeypatch, "moths settling on a pale lamp")
    await _run(env, _dream_req("caption", attempt_id=attempt_id))
    replay = await _run(env, _dream_req("generate", attempt_id=attempt_id))
    assert (replay.status, replay.outcome, replay.caption, replay.chain_id) == (
        "terminal", "produced", "moths settling on a pale lamp", None)


def test_waking_step_reply_omits_dream_only_fields():
    result = ReverieVisualStepResultV1(run_id=RUN_ID, correlation_id="c", step="caption", status="done",
                                       outcome="produced", chain_id="a-1", attempt_id="a-1",
                                       execution_receipt={"gate_reason": "max_steps", "detail": None})
    payload = result.model_dump(mode="json", exclude_none=True)
    assert "caption" not in payload
    assert payload["execution_receipt"] == {"gate_reason": "max_steps", "detail": None}  # nested Nones kept
    assert ReverieVisualStepResultV1.model_validate(payload) == result
