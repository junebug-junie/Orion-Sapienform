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

    def claim_visual_attempt(self, request, *, retry_sec, now, abandoned_in_flight_window_sec=None):
        assert abandoned_in_flight_window_sec and abandoned_in_flight_window_sec > 0
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
    monkeypatch.setattr(visual_chain.settings, "thermal_gate_enabled", False)
    monkeypatch.setattr(visual_chain.settings, "visual_chain_gpu2_capacity_enabled", False)
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
    return await env.steps.run_visual_step(None, req, now_fn=lambda: NOW, **kw)


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


@pytest.mark.asyncio
async def test_prepare_refused_while_legacy_worker_enabled(env, monkeypatch):
    from orion.reverie.baseline import load_baseline_policy
    from orion.schemas.reverie_visual import VisualBaselineEligibilityV1

    policy = load_baseline_policy()
    need = VisualBaselineEligibilityV1(need_id="need-1", observed_at=datetime.now(timezone.utc),
                                       due_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                                       policy_id=policy.policy_id)
    monkeypatch.setattr(env.vc.settings, "visual_chain_enabled", True)
    result = await _run(env, _req("prepare", request=REQUEST.model_copy(update={"visual_baseline": need})))
    assert (result.status, result.outcome) == ("terminal", "failed")
    assert result.reason in {"legacy_visual_worker_enabled", "visual_baseline_disabled"}
    assert env.store.claims == []


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

    async def slow(prompt, *, correlation_id):
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

    async def hung(prompt, *, correlation_id):
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

    def abandon(attempt_id, *, dispatch_id, reason, now, in_flight_window_sec):
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
async def test_unexpected_exception_is_a_retry(env, monkeypatch):
    from app import store

    monkeypatch.setattr(store, "load_visual_attempt", lambda *a: (_ for _ in ()).throw(KeyError("x")))
    result = await _run(env, _req("caption", attempt_id="attempt-1"))
    assert (result.status, result.reason) == ("retry", "step_exception:KeyError")
    assert isinstance(result, ReverieVisualStepResultV1)
