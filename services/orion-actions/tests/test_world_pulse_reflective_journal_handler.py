from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.settings import Settings  # noqa: E402
from app.world_pulse_journal import (  # noqa: E402
    build_world_pulse_journal_trigger,
    world_pulse_journal_skip_reason,
)
from orion.schemas.world_pulse import (  # noqa: E402
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def _result(*, status: str = "completed", dry_run: bool = False, with_digest: bool = True) -> WorldPulseRunResultV1:
    now = datetime(2026, 5, 20, tzinfo=timezone.utc)
    digest = None
    if with_digest:
        digest = DailyWorldPulseV1(
            run_id="wp-1",
            date="2026-05-20",
            generated_at=now,
            title="Pulse",
            executive_summary="Summary text.",
            sections=DailyWorldPulseSectionsV1(),
            orion_analysis_layer="deterministic",
            created_at=now,
        )
    return WorldPulseRunResultV1(
        run=WorldPulseRunV1(
            run_id="wp-1",
            date="2026-05-20",
            started_at=now,
            status=status,
            dry_run=dry_run,
        ),
        digest=digest,
    )


def test_world_pulse_journal_disabled_by_default() -> None:
    cfg = Settings()
    assert cfg.actions_world_pulse_journal_enabled is False
    assert cfg.actions_world_pulse_run_dry_run is True
    assert cfg.actions_world_pulse_journal_allow_dry_run is False


def test_skip_reason_when_disabled() -> None:
    assert world_pulse_journal_skip_reason(_result(), enabled=False) == "world_pulse_journal_disabled"


def test_skip_reason_dry_run_unless_allowed() -> None:
    assert world_pulse_journal_skip_reason(_result(dry_run=True), enabled=True, allow_dry_run=False) == "world_pulse_dry_run"
    assert world_pulse_journal_skip_reason(_result(dry_run=True), enabled=True, allow_dry_run=True) is None


def test_skip_reason_missing_digest() -> None:
    assert world_pulse_journal_skip_reason(_result(with_digest=False), enabled=True) == "world_pulse_missing_digest"


def test_build_trigger_when_eligible() -> None:
    assert world_pulse_journal_skip_reason(_result(), enabled=True) is None
    trigger = build_world_pulse_journal_trigger(_result())
    assert trigger.trigger_kind == "world_pulse_digest"
    assert trigger.source_ref == "wp-1"


# --- retry of failed world_pulse_digest composes (2026-09-29) -----------------------

import asyncio  # noqa: E402
from datetime import timedelta  # noqa: E402
from uuid import uuid4  # noqa: E402

from app.pending_journal_store import PendingJournalStore, backoff_for_attempts  # noqa: E402
from app.world_pulse_journal import (  # noqa: E402
    drain_pending_world_pulse_journals,
    handle_world_pulse_run_result_journal,
    is_retryable_journal_error,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402

_T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
_SRC = ServiceRef(name="orion-actions", version="test")
_GPU_ERR = RuntimeError("journal_compose_failed:{'message': 'gpu_pool_unavailable:deadline'}")


def _retry_settings(**overrides) -> Settings:
    base = {
        "ACTIONS_WORLD_PULSE_JOURNAL_ENABLED": True,
        "ACTIONS_JOURNALING_ENABLED": True,
        "ACTIONS_WORLD_PULSE_JOURNAL_RETRY_ENABLED": True,
        "ACTIONS_WORLD_PULSE_JOURNAL_RETRY_MAX_AGE_HOURS": 12,
    }
    base.update(overrides)
    return Settings(**base)


def _env_for(result: WorldPulseRunResultV1) -> BaseEnvelope:
    return BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="orion-world-pulse", version="0.1.0"),
        correlation_id=str(uuid4()),
        payload=result.model_dump(mode="json"),
    )


class _FakeDispatch:
    """Mirrors main._dispatch_journal's contract: True on success; on a compose error
    awaits on_failure(exc) and returns False; on cooldown/disabled returns False
    without calling on_failure. Counts journals actually written per run_id."""

    def __init__(self, script: list) -> None:
        self.script = list(script)  # each item: "ok" | "skip" | Exception
        self.calls = 0
        self.written: dict[str, int] = {}

    async def __call__(self, parent, *, trigger, audit_action, dedupe_key, on_failure=None, **kw):
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if step == "ok":
            self.written[trigger.source_ref] = self.written.get(trigger.source_ref, 0) + 1
            return True
        if step == "skip":
            return False
        if on_failure is not None:
            await on_failure(step)
        return False


class _Audit:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, env, **kw):
        self.calls.append(kw)


def _handle(env, cfg, dispatch, audit, store, now=_T0):
    return asyncio.run(
        handle_world_pulse_run_result_journal(
            env, settings=cfg, dispatch_journal=dispatch, audit=audit, retry_store=store, now_fn=lambda: now
        )
    )


def _drain(store, cfg, dispatch, audit, now):
    return asyncio.run(
        drain_pending_world_pulse_journals(
            store=store, settings=cfg, dispatch_journal=dispatch, audit=audit, source=_SRC, now=now
        )
    )


def test_retryable_error_classification() -> None:
    assert is_retryable_journal_error(_GPU_ERR)
    assert is_retryable_journal_error(TimeoutError("RPC timeout waiting on x"))
    assert is_retryable_journal_error(ValueError("cortex_orch_missing_final_text"))
    assert is_retryable_journal_error(RuntimeError("cortex_orch_decode_failed:bad"))
    assert not is_retryable_journal_error(RuntimeError("journal_compose_failed:unknown_verb"))


def test_failure_enqueues_with_backoff(tmp_path) -> None:
    store = PendingJournalStore(tmp_path / "pending_journals.json")
    audit = _Audit()
    _handle(_env_for(_result()), _retry_settings(), _FakeDispatch([_GPU_ERR]), audit, store)
    entry = store.get("wp-1")
    assert entry is not None
    assert entry.attempts == 1
    assert entry.next_at_dt == _T0 + timedelta(minutes=5)
    assert entry.first_failed_at_dt == _T0
    assert "gpu_pool_unavailable" in entry.last_error
    assert any(c.get("status") == "retry_scheduled" for c in audit.calls)


def test_non_retryable_outcomes_do_not_enqueue(tmp_path) -> None:
    store = PendingJournalStore(tmp_path / "pending_journals.json")
    # cooldown / journaling disabled: dispatch returns False without an error
    _handle(_env_for(_result()), _retry_settings(), _FakeDispatch(["skip"]), _Audit(), store)
    assert store.get("wp-1") is None
    # deterministic compose error
    _handle(_env_for(_result()), _retry_settings(), _FakeDispatch([RuntimeError("bad_trigger")]), _Audit(), store)
    assert store.get("wp-1") is None
    # retry feature off: nothing enqueued, dispatch receives no on_failure
    seen: dict = {}

    async def spy(parent, **kw):
        seen.update(kw)
        return False

    _handle(_env_for(_result()), _retry_settings(ACTIONS_WORLD_PULSE_JOURNAL_RETRY_ENABLED=False), spy, _Audit(), store)
    assert "on_failure" not in seen
    assert store.pending() == []


def test_drain_retries_until_success_and_never_duplicates(tmp_path) -> None:
    store = PendingJournalStore(tmp_path / "pending_journals.json")
    cfg = _retry_settings()
    dispatch = _FakeDispatch([_GPU_ERR, _GPU_ERR, "ok"])
    _handle(_env_for(_result()), cfg, dispatch, _Audit(), store)
    # not due yet
    assert _drain(store, cfg, dispatch, _Audit(), _T0 + timedelta(minutes=4)) == []
    assert _drain(store, cfg, dispatch, _Audit(), _T0 + timedelta(minutes=5)) == [("wp-1", "rescheduled")]
    entry = store.get("wp-1")
    assert entry.attempts == 2
    assert entry.next_at_dt == _T0 + timedelta(minutes=5) + backoff_for_attempts(2)
    assert _drain(store, cfg, dispatch, _Audit(), entry.next_at_dt) == [("wp-1", "completed")]
    assert store.get("wp-1") is None
    assert store.is_completed("wp-1")
    assert dispatch.written == {"wp-1": 1}
    # further drains and a redelivered run result must not write again
    assert _drain(store, cfg, dispatch, _Audit(), _T0 + timedelta(hours=3)) == []
    audit = _Audit()
    _handle(_env_for(_result()), cfg, dispatch, audit, store)
    assert dispatch.written == {"wp-1": 1}
    assert audit.calls[-1]["reason"] == "world_pulse_journal_already_written"


def test_pending_survives_restart(tmp_path) -> None:
    path = tmp_path / "pending_journals.json"
    store = PendingJournalStore(path)
    cfg = _retry_settings()
    _handle(_env_for(_result()), cfg, _FakeDispatch([_GPU_ERR]), _Audit(), store)
    reloaded = PendingJournalStore(path)
    entry = reloaded.get("wp-1")
    assert entry is not None and entry.attempts == 1
    dispatch = _FakeDispatch(["ok"])
    assert _drain(reloaded, cfg, dispatch, _Audit(), _T0 + timedelta(minutes=6)) == [("wp-1", "completed")]
    assert dispatch.written == {"wp-1": 1}
    assert PendingJournalStore(path).is_completed("wp-1")


def test_gives_up_after_max_age(tmp_path) -> None:
    store = PendingJournalStore(tmp_path / "pending_journals.json")
    cfg = _retry_settings(ACTIONS_WORLD_PULSE_JOURNAL_RETRY_MAX_AGE_HOURS=12)
    _handle(_env_for(_result()), cfg, _FakeDispatch([_GPU_ERR]), _Audit(), store)
    dispatch = _FakeDispatch(["ok"])
    audit = _Audit()
    out = _drain(store, cfg, dispatch, audit, _T0 + timedelta(hours=12, minutes=1))
    assert out == [("wp-1", "gave_up")]
    assert dispatch.calls == 0
    assert store.get("wp-1") is None
    assert audit.calls[-1]["action_name"] == "world_pulse_journal_gave_up"


def test_drain_drops_when_not_dispatched(tmp_path) -> None:
    store = PendingJournalStore(tmp_path / "pending_journals.json")
    cfg = _retry_settings()
    _handle(_env_for(_result()), cfg, _FakeDispatch([_GPU_ERR]), _Audit(), store)
    out = _drain(store, cfg, _FakeDispatch(["skip"]), _Audit(), _T0 + timedelta(minutes=5))
    assert out == [("wp-1", "dropped_not_dispatched")]
    assert store.pending() == []


def test_drain_disabled_is_noop(tmp_path) -> None:
    store = PendingJournalStore(tmp_path / "pending_journals.json")
    _handle(_env_for(_result()), _retry_settings(), _FakeDispatch([_GPU_ERR]), _Audit(), store)
    off = _retry_settings(ACTIONS_WORLD_PULSE_JOURNAL_RETRY_ENABLED=False)
    assert _drain(store, off, _FakeDispatch(["ok"]), _Audit(), _T0 + timedelta(hours=1)) == []
    assert store.get("wp-1") is not None


def test_retry_settings_defaults() -> None:
    cfg = Settings()
    assert cfg.actions_world_pulse_journal_retry_enabled is True
    assert cfg.actions_world_pulse_journal_retry_max_age_hours == 12.0
