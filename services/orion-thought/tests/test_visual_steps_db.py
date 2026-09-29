"""Durable `reverie.visual` stages against isolated PostgreSQL (the real stage store).

Set ORION_VISUAL_TEST_DATABASE_URL to a disposable database (never production).
Each test gets its own schema with the chain, attempt and attempt-stage migrations.
"""
from __future__ import annotations

import os
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

from orion.gpu_pool.client import durable_run_holder
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.reverie_visual import ReverieVisualChainV1, VisualRunRequestV1
from orion.schemas.reverie_visual_run import ReverieVisualStepRequestV1

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
RUN_ID = "reverie-visual-db"
_MIGRATIONS = ("manual_migration_reverie_visual_chain.sql", "manual_migration_reverie_visual_attempt.sql",
               "manual_migration_reverie_visual_attempt_stage.sql")


def _schema_engine(monkeypatch, migrations):
    from app import store

    url = os.environ.get("ORION_VISUAL_TEST_DATABASE_URL")
    if not url:
        pytest.skip("isolated PostgreSQL URL required")
    schema = "visual_steps_test_" + uuid4().hex
    bootstrap = create_engine(url)
    with bootstrap.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    bootstrap.dispose()
    engine = create_engine(url, connect_args={"options": f"-c search_path={schema}"})
    root = Path(__file__).resolve().parents[3]
    with engine.begin() as conn:
        for filename in migrations:
            conn.execute(text((root / "services/orion-sql-db" / filename).read_text()))
    monkeypatch.setattr(store, "_get_engine", lambda: engine)
    return store, engine


@pytest.fixture
def db(monkeypatch, tmp_path):
    from app import visual_chain, visual_steps

    store, engine = _schema_engine(monkeypatch, _MIGRATIONS)
    settings = visual_chain.settings
    monkeypatch.setattr(settings, "thermal_gate_enabled", False)
    monkeypatch.setattr(settings, "visual_chain_enabled", False)
    monkeypatch.setattr(settings, "visual_chain_interpretation_enabled", False)
    monkeypatch.setattr(settings, "visual_chain_storage_dir", str(tmp_path))
    monkeypatch.setattr(visual_chain, "load_latest_reverie_interpretation", lambda **kw: "a real thought")
    monkeypatch.setattr(visual_chain, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(visual_chain, "load_latest_memory_crystallization", lambda **kw: None)
    renders = []

    def fake_generate(prompt, *, base_url, timeout_sec):
        renders.append(prompt)
        return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 32, 32)
                + len(renders).to_bytes(2, "big"))

    monkeypatch.setattr(visual_chain, "call_diffusion_generate", fake_generate)
    monkeypatch.setattr(visual_chain, "upload_to_percept_store",
                        lambda data, **kw: __import__("hashlib").sha256(data).hexdigest())
    monkeypatch.setattr(visual_chain, "request_caption", AsyncMock(return_value="embers under a low sky"))

    async def validate(bus, ref, *, source, expected_holder=None, timeout_sec=None):
        assert ref.holder == expected_holder

    monkeypatch.setattr(visual_steps, "validate_hold_ref", validate)
    yield SimpleNamespace(store=store, engine=engine, vc=visual_chain, steps=visual_steps, renders=renders)
    engine.dispose()


def _req(step, dispatch_id="dispatch-1", attempt_id=None):
    lease = None
    if step == "generate":
        lease = GpuLeaseRefV1(lease_id="lease-1", generation=1, role="diffusion",
                              holder=durable_run_holder(RUN_ID))
    return ReverieVisualStepRequestV1(
        run_id=RUN_ID, correlation_id=f"{step}-corr", step=step, attempt_id=attempt_id, gpu_lease=lease,
        visual_request=VisualRunRequestV1(dispatch_id=dispatch_id, proposal_id="proposal-1"),
    )


async def _step(db, step, *, at=NOW, **kw):
    return await db.steps.run_visual_step(AsyncMock(), _req(step, **kw), now_fn=lambda: at)


def _attempt(engine, attempt_id):
    with engine.connect() as conn:
        return conn.execute(text("SELECT outcome, stage_json, result_json FROM reverie_visual_attempt "
                                 "WHERE attempt_id=:id"), {"id": attempt_id}).mappings().one()


async def _produce(db, dispatch_id="dispatch-1", at=NOW):
    prepared = await _step(db, "prepare", dispatch_id=dispatch_id, at=at)
    assert prepared.status == "done", prepared
    attempt_id = prepared.attempt_id
    generated = await _step(db, "generate", dispatch_id=dispatch_id, attempt_id=attempt_id, at=at)
    assert generated.status == "done", generated
    captioned = await _step(db, "caption", dispatch_id=dispatch_id, attempt_id=attempt_id, at=at)
    assert captioned.status == "done", captioned
    return attempt_id, captioned


@pytest.mark.asyncio
async def test_missing_stage_column_refuses_before_claiming(monkeypatch):
    from app import visual_steps

    _, engine = _schema_engine(monkeypatch, _MIGRATIONS[:2])
    result = await visual_steps.run_visual_step(None, _req("prepare"), now_fn=lambda: NOW)
    assert (result.status, result.reason) == ("retry", "stage_store_unavailable")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM reverie_visual_attempt")).scalar() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_full_run_writes_one_production_row_keyed_by_attempt(db):
    attempt_id, captioned = await _produce(db)
    assert captioned.chain_id == attempt_id
    with db.engine.connect() as conn:
        rows = conn.execute(text("SELECT chain_id, terminal_reason, prior_description, chain_json "
                                 "FROM reverie_visual_chain")).mappings().all()
    [row] = rows
    assert row["chain_id"] == attempt_id
    assert row["terminal_reason"] == "max_steps"
    assert row["prior_description"] == "embers under a low sky"
    assert row["chain_json"]["production_receipt"]["attempt_id"] == attempt_id
    assert row["chain_json"]["execution_receipt"]["outcome"] == "produced"
    attempt = _attempt(db.engine, attempt_id)
    assert attempt["outcome"] == "produced"
    assert attempt["result_json"]["durable_run_id"] == RUN_ID
    assert attempt["stage_json"]["plan"]["prompt"] == db.renders[0]
    assert db.store.load_visual_activity().last_success_chain_id == attempt_id
    # Every replay is a no-op that reports the recorded outcome.
    assert (await _step(db, "prepare")).outcome == "produced"
    assert (await _step(db, "generate", attempt_id=attempt_id)).outcome == "produced"
    replay = await _step(db, "caption", attempt_id=attempt_id)
    assert (replay.status, replay.outcome, replay.chain_id) == ("done", "produced", attempt_id)
    assert len(db.renders) == 1


@pytest.mark.asyncio
async def test_replayed_prepare_keeps_the_frozen_plan(db, monkeypatch):
    first = await _step(db, "prepare")
    frozen = _attempt(db.engine, first.attempt_id)["stage_json"]["plan"]
    monkeypatch.setattr(db.vc, "load_latest_reverie_interpretation", lambda **kw: "a different thought")
    replay = await _step(db, "prepare", at=NOW + timedelta(minutes=5))
    assert replay.attempt_id == first.attempt_id
    assert _attempt(db.engine, first.attempt_id)["stage_json"]["plan"] == frozen


@pytest.mark.asyncio
async def test_continuity_survives_a_deferral_row(db, monkeypatch):
    attempt_id, _ = await _produce(db)
    produced_state = db.store.load_latest_visual_chain_continuity_state()
    assert produced_state[0] == "embers under a low sky"
    plan = db.vc.VisualPlan.from_json(_attempt(db.engine, attempt_id)["stage_json"]["plan"])
    assert produced_state[1:] == (plan.continuity_streak, plan.context_slot_rotation)

    # A legacy thermal refusal and a resource deferral write rows with no continuity state.
    monkeypatch.setattr(db.vc.settings, "thermal_gate_enabled", True)
    monkeypatch.setattr(db.vc, "evaluate_thermal_gate", AsyncMock(return_value=SimpleNamespace(
        state="hot", temp_c=36.0, age_sec=1.0, reason="test_reading", allows_gpu_work=False,
        degraded=False)))
    refused = await db.vc.run_visual_chain_once(None, now_fn=lambda: NOW + timedelta(hours=1))
    assert refused is not None and refused.terminal_reason == "thermal_refused"
    deferred = db.vc.build_resource_deferred_chain(
        "deferred-chain", plan, "controller_displacement",
        thermal_gate={"state": "disabled"}, run_request=None,
        now_fn=lambda: NOW + timedelta(hours=2))
    assert db.store.persist_reverie_visual_chain(deferred)

    assert db.store.load_latest_visual_chain_continuity_state() == produced_state
    next_plan = await db.vc.compute_visual_plan(None, chain_id="next")
    assert next_plan.continuity_streak == plan.continuity_streak + 1
    assert next_plan.context_slot_rotation == plan.context_slot_rotation + 1
    assert next_plan.prior_chain_id == attempt_id


@pytest.mark.asyncio
async def test_abandon_releases_the_claim_and_is_idempotent(db):
    prepared = await _step(db, "prepare")
    abandoned = await _step(db, "abandon", attempt_id=prepared.attempt_id)
    assert (abandoned.status, abandoned.outcome, abandoned.reason) == ("done", "unknown", "run_abandoned")
    attempt = _attempt(db.engine, prepared.attempt_id)
    assert attempt["outcome"] == "abandoned"
    assert attempt["result_json"]["outcome"] == "unknown"
    again = await _step(db, "abandon", attempt_id=prepared.attempt_id, at=NOW + timedelta(minutes=1))
    assert (again.status, again.outcome) == ("done", "unknown")
    assert _attempt(db.engine, prepared.attempt_id)["stage_json"]["abandoned_at"] == NOW.isoformat()
    # The abandoned attempt no longer blocks the next dispatch (after the retry gap).
    later = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(hours=1))
    assert later.status == "done" and later.attempt_id != prepared.attempt_id


@pytest.mark.asyncio
async def test_generate_timeout_then_abandon_is_released_by_the_generate_itself(db, monkeypatch):
    import asyncio

    prepared = await _step(db, "prepare")
    attempt_id = prepared.attempt_id
    monkeypatch.setattr(db.steps, "visual_step_generate_deadline_sec", lambda: 0.5)
    monkeypatch.setattr(db.steps, "_in_flight_window_sec", lambda: 660.0)
    release = asyncio.Event()
    real_generate = db.vc.generate_visual_bytes

    async def slow(prompt, *, correlation_id, hold=None, bus=None):
        await release.wait()
        return await real_generate(prompt, correlation_id=correlation_id, hold=hold, bus=bus)

    monkeypatch.setattr(db.vc, "generate_visual_bytes", slow)
    timed_out = await _step(db, "generate", attempt_id=attempt_id)
    assert timed_out.reason == "generate_deadline_exceeded"

    # The run's deadline passed: durable-runs abandons once and never sends generate again.
    held = await _step(db, "abandon", attempt_id=attempt_id, at=NOW + timedelta(seconds=10))
    assert (held.outcome, held.reason) == ("unknown", "generate_in_flight")
    assert _attempt(db.engine, attempt_id)["outcome"] == "unknown"
    blocked = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(seconds=20))
    assert (blocked.status, blocked.reason) == ("retry", "attempt_unresolved")

    release.set()
    await asyncio.gather(*list(db.steps._generate_tasks))
    assert _attempt(db.engine, attempt_id)["outcome"] == "abandoned"
    later = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(hours=1))
    assert later.status == "done"
    assert len(db.renders) == 1


@pytest.mark.asyncio
async def test_abandoned_in_flight_attempt_whose_process_died_is_released_after_the_window(db):
    prepared = await _step(db, "prepare")
    attempt_id = prepared.attempt_id

    def died_mid_generate(stage, _row):
        stage.update(stage="generating", generating_started_at=NOW.isoformat())
        return stage

    db.store.update_visual_stage(attempt_id, died_mid_generate)
    await _step(db, "abandon", attempt_id=attempt_id, at=NOW + timedelta(seconds=10))
    assert _attempt(db.engine, attempt_id)["outcome"] == "unknown"
    window = 2 * db.steps.visual_step_generate_deadline_sec()
    inside = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(seconds=window - 5))
    assert (inside.status, inside.reason) == ("retry", "attempt_unresolved")
    after = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(hours=1))
    assert after.status == "done"
    assert _attempt(db.engine, attempt_id)["outcome"] == "abandoned"


@pytest.mark.asyncio
async def test_legacy_claim_is_not_blocked_by_a_dead_durable_leftover(db):
    # Rollback to run-once: the same release rule applies on the legacy claim.
    prepared = await _step(db, "prepare")

    def died_mid_generate(stage, _row):
        stage.update(stage="generating", generating_started_at=NOW.isoformat())
        return stage

    db.store.update_visual_stage(prepared.attempt_id, died_mid_generate)
    await _step(db, "abandon", attempt_id=prepared.attempt_id, at=NOW + timedelta(seconds=10))
    window = 2 * db.steps.visual_step_generate_deadline_sec()
    legacy_id, legacy_replay = db.store.claim_visual_attempt(
        VisualRunRequestV1(dispatch_id="legacy"), retry_sec=600, now=NOW + timedelta(hours=1),
        abandoned_in_flight_window_sec=window, attempt_max_age_sec=7200.0)
    assert legacy_replay is None and legacy_id
    assert _attempt(db.engine, prepared.attempt_id)["outcome"] == "abandoned"


@pytest.mark.asyncio
async def test_abandon_without_attempt_id_resolves_by_dispatch(db):
    nothing = await _step(db, "abandon")
    assert (nothing.status, nothing.reason) == ("done", "attempt_missing")
    prepared = await _step(db, "prepare")
    # prepare's reply was lost: abandon names only the dispatch.
    abandoned = await _step(db, "abandon", at=NOW + timedelta(seconds=5))
    assert (abandoned.status, abandoned.outcome, abandoned.attempt_id) == ("done", "unknown", prepared.attempt_id)
    assert _attempt(db.engine, prepared.attempt_id)["outcome"] == "abandoned"
    later = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(hours=1))
    assert later.status == "done"


@pytest.mark.asyncio
async def test_abandon_by_dispatch_refuses_a_different_request_under_that_dispatch(db):
    prepared = await _step(db, "prepare")
    other = ReverieVisualStepRequestV1(
        run_id=RUN_ID, correlation_id="abandon-corr", step="abandon",
        visual_request=VisualRunRequestV1(dispatch_id="dispatch-1", proposal_id="another-proposal"))
    result = await db.steps.run_visual_step(None, other, now_fn=lambda: NOW)
    assert (result.status, result.reason) == ("terminal", "attempt_mismatch")
    assert _attempt(db.engine, prepared.attempt_id)["outcome"] == "active"


@pytest.mark.asyncio
async def test_stuck_active_attempt_expires_after_max_age(db):
    prepared = await _step(db, "prepare")  # abandon never arrives
    max_age = db.vc.settings.visual_chain_attempt_max_age_sec
    inside = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(seconds=max_age - 60))
    assert (inside.status, inside.reason) == ("retry", "attempt_unresolved")
    after = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(seconds=max_age + 1))
    assert after.status == "done" and after.attempt_id != prepared.attempt_id
    expired = _attempt(db.engine, prepared.attempt_id)
    assert expired["outcome"] == "abandoned"
    assert (expired["result_json"]["outcome"], expired["result_json"]["reason"]) == ("unknown", "attempt_expired")
    # The expired run itself ends instead of doing more work.
    late = await _step(db, "generate", attempt_id=prepared.attempt_id, at=NOW + timedelta(seconds=max_age + 2))
    assert (late.status, late.outcome, late.reason) == ("terminal", "unknown", "attempt_expired")


@pytest.mark.asyncio
async def test_expiry_never_releases_a_generate_still_in_flight(db):
    prepared = await _step(db, "prepare")
    max_age = db.vc.settings.visual_chain_attempt_max_age_sec
    at = NOW + timedelta(seconds=max_age + 1)

    def generating_now(stage, _row):
        stage.update(stage="generating", generating_started_at=(at - timedelta(seconds=30)).isoformat())
        return stage

    db.store.update_visual_stage(prepared.attempt_id, generating_now)
    blocked = await _step(db, "prepare", dispatch_id="dispatch-2", at=at)
    assert (blocked.status, blocked.reason) == ("retry", "attempt_unresolved")
    assert _attempt(db.engine, prepared.attempt_id)["outcome"] == "active"


@pytest.mark.asyncio
async def test_expiry_reconciles_a_produced_attempt_instead_of_releasing_it(db):
    attempt_id, _ = await _produce(db)
    with db.engine.begin() as conn:  # the finish write was lost after production
        conn.execute(text("UPDATE reverie_visual_attempt SET outcome='active' WHERE attempt_id=:id"),
                     {"id": attempt_id})
    later = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(days=1))
    assert later.status == "done"
    produced = _attempt(db.engine, attempt_id)
    assert produced["outcome"] == "produced"
    assert produced["result_json"]["reason"] == "production_reconciled"


def test_legacy_claim_expiry_without_the_stage_column(monkeypatch):
    store, engine = _schema_engine(monkeypatch, _MIGRATIONS[:2])
    first, _ = store.claim_visual_attempt(VisualRunRequestV1(dispatch_id="legacy-1"), retry_sec=600, now=NOW,
                                          abandoned_in_flight_window_sec=660.0, attempt_max_age_sec=7200.0)
    assert first
    blocked = store.claim_visual_attempt(VisualRunRequestV1(dispatch_id="legacy-2"), retry_sec=600,
                                         now=NOW + timedelta(hours=1), abandoned_in_flight_window_sec=660.0,
                                         attempt_max_age_sec=7200.0)
    assert blocked[1]["reason"] == "attempt_unresolved"
    second, replay = store.claim_visual_attempt(VisualRunRequestV1(dispatch_id="legacy-2"), retry_sec=600,
                                                now=NOW + timedelta(days=2), abandoned_in_flight_window_sec=660.0,
                                                attempt_max_age_sec=7200.0)
    assert replay is None and second
    assert store.replay_visual_attempt(VisualRunRequestV1(dispatch_id="legacy-1"))["reason"] == "attempt_expired"
    engine.dispose()


@pytest.mark.asyncio
async def test_unparseable_generate_start_is_held_not_a_broken_claim(db):
    prepared = await _step(db, "prepare")

    def garbled(stage, _row):
        stage.update(stage="generating", generating_started_at="not-a-time",
                     abandoned_at=NOW.isoformat())
        return stage

    db.store.update_visual_stage(prepared.attempt_id, garbled)
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE reverie_visual_attempt SET outcome='unknown' WHERE attempt_id=:id"),
                     {"id": prepared.attempt_id})
    later = await _step(db, "prepare", dispatch_id="dispatch-2", at=NOW + timedelta(hours=1))
    assert (later.status, later.reason) == ("retry", "attempt_unresolved")


@pytest.mark.asyncio
async def test_generate_for_foreign_attempt_is_refused(db):
    prepared = await _step(db, "prepare")
    foreign = await _step(db, "generate", dispatch_id="dispatch-other", attempt_id=prepared.attempt_id)
    assert (foreign.status, foreign.reason) == ("terminal", "attempt_mismatch")
    abandon = await _step(db, "abandon", dispatch_id="dispatch-other", attempt_id=prepared.attempt_id)
    assert (abandon.status, abandon.reason) == ("terminal", "attempt_mismatch")
    assert _attempt(db.engine, prepared.attempt_id)["outcome"] == "active"
    assert db.renders == []
