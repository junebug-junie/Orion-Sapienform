from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import main as actions_main  # noqa: E402
from app.settings import Settings  # noqa: E402
from app.world_pulse_journal import handle_world_pulse_run_result_journal  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.journaler import cooldown_key_for_trigger  # noqa: E402
from orion.schemas.world_pulse import (  # noqa: E402
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def _run_result_payload(*, run_id: str = "wp-integration-1", dry_run: bool = False) -> dict:
    now = datetime(2026, 5, 20, tzinfo=timezone.utc)
    digest = DailyWorldPulseV1(
        run_id=run_id,
        date="2026-05-20",
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="Integration summary.",
        sections=DailyWorldPulseSectionsV1(),
        orion_analysis_layer="deterministic",
        created_at=now,
    )
    return WorldPulseRunResultV1(
        run=WorldPulseRunV1(
            run_id=run_id,
            date="2026-05-20",
            started_at=now,
            completed_at=now,
            status="completed",
            dry_run=dry_run,
        ),
        digest=digest,
    ).model_dump(mode="json")


def _envelope(*, run_id: str = "wp-integration-1", dry_run: bool = False) -> BaseEnvelope:
    return BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="orion-world-pulse", version="0.1.0"),
        correlation_id=str(uuid4()),
        payload=_run_result_payload(run_id=run_id, dry_run=dry_run),
    )


@pytest.mark.asyncio
async def test_handle_envelope_routes_world_pulse_run_result_to_dispatch_journal() -> None:
    """Exercise app.state.bus_handler (handle_envelope) for world.pulse.run.result.v1."""
    env = _envelope(run_id="wp-route-99")
    captured: dict = {}

    async def capture_dispatch(parent, *, trigger, audit_action: str, dedupe_key: str, reason: str | None = None, **kwargs):
        captured["trigger"] = trigger
        captured["audit_action"] = audit_action
        captured["dedupe_key"] = dedupe_key
        captured.update(kwargs)
        return True

    cfg = Settings(
        ACTIONS_WORLD_PULSE_JOURNAL_ENABLED=True,
        ACTIONS_JOURNALING_ENABLED=True,
    )

    class _NoopHunter:
        def __init__(self, *args, **kwargs) -> None:
            # Async bus surface: lifespan awaits connect()/close() on it.
            self.bus = AsyncMock()

        async def start(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

    async def intercept(env: BaseEnvelope, **kw):
        return await handle_world_pulse_run_result_journal(
            env,
            settings=cfg,
            dispatch_journal=capture_dispatch,
            audit=AsyncMock(),
        )

    _real_create_task = asyncio.create_task

    def _noop_create_task(coro, *args, **kwargs):
        async def _cancellable() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

        return _real_create_task(_cancellable())

    with (
        patch.object(actions_main, "Hunter", _NoopHunter),
        patch("orion.notify.client.NotifyClient", return_value=MagicMock()),
        patch.object(actions_main.asyncio, "create_task", side_effect=_noop_create_task),
        patch("app.main.handle_world_pulse_run_result_journal", side_effect=intercept) as routed,
    ):
        async with actions_main.lifespan(actions_main.app):
            handler = actions_main.app.state.bus_handler
            assert handler is not None
            await handler(env)

    routed.assert_awaited_once()
    assert routed.await_args.args[0].kind == "world.pulse.run.result.v1"

    trigger = captured["trigger"]
    assert trigger.trigger_kind == "world_pulse_digest"
    assert trigger.source_kind == "world_pulse"
    assert trigger.source_ref == "wp-route-99"
    assert captured["dedupe_key"] == cooldown_key_for_trigger(trigger)
    assert captured["audit_action"] == "journal.world_pulse_digest"


def test_handle_envelope_does_not_fall_through_to_collapse_for_world_pulse_kind() -> None:
    env = _envelope()
    collapse_called = False

    async def fake_collapse(_env: BaseEnvelope) -> None:
        nonlocal collapse_called
        collapse_called = True

    async def noop_world_pulse(_env: BaseEnvelope) -> bool:
        return True

    async def route_like_handle_envelope(envelope: BaseEnvelope) -> None:
        kind = str(envelope.kind or "")
        if kind == "world.pulse.run.result.v1":
            if await noop_world_pulse(envelope):
                return
        await fake_collapse(envelope)

    asyncio.run(route_like_handle_envelope(env))
    assert collapse_called is False
