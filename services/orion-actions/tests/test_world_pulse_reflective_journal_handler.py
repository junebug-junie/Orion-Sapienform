from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

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


# --- durable submission (2026-09-30): journal.compose admitted run ---------------------------

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402
from uuid import uuid4  # noqa: E402

from app.world_pulse_journal import (  # noqa: E402
    build_world_pulse_journal_run_request,
    handle_world_pulse_run_result_journal,
    next_local_midnight,
    submit_durable_run_via_cortex,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402

_NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)   # 06:00 Denver


def _cfg(**kw) -> Settings:
    base = {"ACTIONS_WORLD_PULSE_JOURNAL_ENABLED": True, "ACTIONS_JOURNALING_ENABLED": True}
    return Settings(**{**base, **kw})


def _env(result: WorldPulseRunResultV1) -> BaseEnvelope:
    return BaseEnvelope(kind="world.pulse.run.result.v1", source=ServiceRef(name="orion-world-pulse"),
                        correlation_id=str(uuid4()), payload=result.model_dump(mode="json"))


class _Audit:
    def __init__(self):
        self.calls = []

    async def __call__(self, env, **kw):
        self.calls.append(kw)


def _handle(env, *, submit, cfg=None, audit=None, now=_NOW):
    audit = audit or _Audit()
    slept = []

    async def sleep(sec):
        slept.append(sec)

    asyncio.run(handle_world_pulse_run_result_journal(
        env, settings=cfg or _cfg(), submit=submit, audit=audit, llm_route="quick_background",
        now_fn=lambda: now, sleep=sleep))
    return audit, slept


def test_request_is_deterministic_admitted_and_bounded_to_local_midnight() -> None:
    r1 = build_world_pulse_journal_run_request(_result(), settings=_cfg(), llm_route="quick_background", now=_NOW)
    r2 = build_world_pulse_journal_run_request(_result(), settings=_cfg(), llm_route="quick_background",
                                               now=_NOW + timedelta(hours=3))
    assert r1.run_id == "world-pulse-journal:wp-1"
    assert r1.workflow == "journal.compose"
    # Redelivery resubmits the identical request (durable-runs ignores deadline_at on a resubmit).
    d1 = r1.model_dump(mode="json", exclude={"requested_at"})
    d2 = r2.model_dump(mode="json", exclude={"requested_at"})
    d1["admission"].pop("deadline_at"); d2["admission"].pop("deadline_at")
    assert d1 == d2
    assert r1.brief.entry_id == r2.brief.entry_id and r1.correlation_id == r2.correlation_id
    adm = r1.admission
    assert adm.resource == "llm.route.quick_background" and adm.preferred_lane == "quick_background"
    assert adm.priority == "background"
    assert adm.deadline_at == datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)   # midnight Denver (MDT)
    assert r1.brief.trigger.trigger_kind == "world_pulse_digest"
    assert r1.brief.body_appendix is None and r1.brief.body_appendix_markers == []   # no followups
    assert r1.brief.recall_profile == "journal.world_pulse.grounded.v1"


def test_next_local_midnight_late_evening() -> None:
    late = datetime(2026, 10, 1, 5, 30, tzinfo=timezone.utc)   # 23:30 Denver, Sep 30
    assert next_local_midnight(late, "America/Denver") == datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)


def test_eligible_result_submits_once_and_audits() -> None:
    submitted = []

    async def submit(request):
        submitted.append(request)
        return None

    audit, slept = _handle(_env(_result()), submit=submit)
    assert len(submitted) == 1 and slept == []
    assert audit.calls[-1]["status"] == "submitted"
    assert audit.calls[-1]["extra"]["durable_run_id"] == "world-pulse-journal:wp-1"


def _comparable(request):
    """What durable-runs' store compares on a resubmit (orion/durable_admission/store.py submit):
    the whole request minus requested_at and admission.deadline_at."""
    data = request.model_dump(mode="json", exclude_none=True)
    data.pop("requested_at", None)
    data["admission"].pop("deadline_at", None)
    return data


def test_redelivery_resubmits_an_identical_request() -> None:
    seen = []

    async def submit(request):
        seen.append(_comparable(request))
        return None

    _handle(_env(_result()), submit=submit)
    _handle(_env(_result()), submit=submit, now=_NOW + timedelta(hours=1))   # new envelope, later clock
    assert len(seen) == 2 and seen[0] == seen[1]


def test_curiosity_followups_travel_as_prerendered_text() -> None:
    from orion.schemas.world_pulse import CuriosityFollowupV1

    result = _result()
    result.digest.curiosity_followups = [CuriosityFollowupV1.model_validate({
        "section": "science", "query": "fusion ignition", "driving_gap": "missing",
        "articles": [{"title": "Ignition", "url": "https://example.org/ignition", "salience": 0.9}],
    })]
    req = build_world_pulse_journal_run_request(result, settings=_cfg(), llm_route="quick_background", now=_NOW)
    assert "## Orion went looking" in req.brief.body_appendix
    assert req.brief.body_appendix_markers == ["https://example.org/ignition"]
    assert "world_pulse_result" not in req.brief.model_dump()


def test_submit_retries_briefly_then_reports_failure() -> None:
    calls = []

    async def submit(request):
        calls.append(1)
        return "TimeoutError: RPC timeout"

    audit, slept = _handle(_env(_result()), submit=submit)
    assert len(calls) == 5 and slept == [10.0, 30.0, 90.0, 270.0]
    assert audit.calls[-1]["status"] == "failed"
    assert audit.calls[-1]["reason"].startswith("durable_submit_failed:")


def test_ineligible_results_do_not_submit() -> None:
    async def submit(request):
        raise AssertionError("must not submit")

    audit, _ = _handle(_env(_result(dry_run=True)), submit=submit)
    assert audit.calls[-1]["reason"] == "world_pulse_dry_run"
    audit, _ = _handle(_env(_result()), submit=submit, cfg=_cfg(ACTIONS_JOURNALING_ENABLED=False))
    assert audit.calls[-1]["reason"] == "journaling_disabled"
    audit, _ = _handle(_env(_result()), submit=submit,
                       cfg=Settings(ACTIONS_WORLD_PULSE_JOURNAL_ENABLED=False))
    assert audit.calls == []


def _bus_replying(payload):
    codec = OrionCodec()
    reply = BaseEnvelope(kind="cortex.orch.result", source=ServiceRef(name="orion-cortex-orch"),
                         correlation_id=str(uuid4()), payload=payload)
    return SimpleNamespace(codec=codec, rpc_request=AsyncMock(return_value={"data": codec.encode(reply)}))


def test_submit_via_cortex_checks_the_receipt() -> None:
    request = build_world_pulse_journal_run_request(_result(), settings=_cfg(), llm_route="quick_background", now=_NOW)
    good = {"ok": True, "status": "accepted", "metadata": {"durable_run": {
        "run_id": request.run_id, "workflow_kind": "journal.compose",
        "requested_resource": "llm.route.quick_background", "status": "waiting_resource"}}}
    bus = _bus_replying(good)
    src = ServiceRef(name="orion-actions")
    assert asyncio.run(submit_durable_run_via_cortex(bus=bus, source=src, request=request,
                                                     request_channel="orion:cortex:request")) is None
    sent = bus.rpc_request.await_args.args[1]
    assert sent.payload["context"]["metadata"]["durable_run"]["run_id"] == request.run_id
    wrong = {**good, "metadata": {"durable_run": {**good["metadata"]["durable_run"], "run_id": "other"}}}
    assert asyncio.run(submit_durable_run_via_cortex(bus=_bus_replying(wrong), source=src, request=request,
                                                     request_channel="c")) == "receipt_run_id_mismatch"
    refused = {"ok": False, "status": "fail", "error": {"message": "durable admission is disabled at Cortex"}}
    assert asyncio.run(submit_durable_run_via_cortex(bus=_bus_replying(refused), source=src, request=request,
                                                     request_channel="c")).startswith("not_accepted:fail")


def test_actions_no_longer_composes_world_pulse_in_process() -> None:
    import inspect

    from app import main as actions_main

    src = inspect.getsource(actions_main)
    assert "merge_world_pulse_curiosity_into_draft" not in src
    assert "world_pulse_result" not in src
