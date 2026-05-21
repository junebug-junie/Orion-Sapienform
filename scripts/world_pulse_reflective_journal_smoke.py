#!/usr/bin/env python3
"""No-live-services smoke for World Pulse → reflective journal runtime contract."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dry-run", action="store_true", help="Set ACTIONS_WORLD_PULSE_JOURNAL_ALLOW_DRY_RUN=true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    actions_dir = root / "services" / "orion-actions"
    env = dict(os.environ)
    if args.allow_dry_run:
        env["ACTIONS_WORLD_PULSE_JOURNAL_ALLOW_DRY_RUN"] = "true"

    cmd = [
        sys.executable,
        "-c",
        _SMOKE_BODY,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(actions_dir),
        env={**env, "PYTHONPATH": f"{actions_dir}{os.pathsep}{root}"},
        check=False,
    )
    return int(proc.returncode)


_SMOKE_BODY = r'''
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from app.settings import Settings
from app.world_pulse_journal import (
    build_world_pulse_journal_trigger,
    handle_world_pulse_run_result_journal,
    world_pulse_journal_skip_reason,
)
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler import cooldown_key_for_trigger
from orion.schemas.world_pulse import (
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def fixture_result(*, run_id: str, dry_run: bool) -> WorldPulseRunResultV1:
    now = datetime(2026, 5, 20, tzinfo=timezone.utc)
    digest = DailyWorldPulseV1(
        run_id=run_id,
        date="2026-05-20",
        generated_at=now,
        title="Smoke Pulse",
        executive_summary="Smoke executive summary.",
        sections=DailyWorldPulseSectionsV1(),
        orion_analysis_layer="deterministic",
        created_at=now,
    )
    return WorldPulseRunResultV1(
        run=WorldPulseRunV1(
            run_id=run_id,
            date="2026-05-20",
            started_at=now,
            status="completed",
            dry_run=dry_run,
        ),
        digest=digest,
    )


async def run_smoke() -> None:
    import os
    allow = os.environ.get("ACTIONS_WORLD_PULSE_JOURNAL_ALLOW_DRY_RUN", "").lower() in {"1", "true", "yes"}
    run_id = "wp-smoke-run-1"
    result = fixture_result(run_id=run_id, dry_run=False)
    assert WorldPulseRunResultV1.model_validate(result.model_dump(mode="json"))

    settings = Settings(
        ACTIONS_WORLD_PULSE_JOURNAL_ENABLED=True,
        ACTIONS_WORLD_PULSE_JOURNAL_ALLOW_DRY_RUN=allow,
    )
    assert world_pulse_journal_skip_reason(result, enabled=True, allow_dry_run=allow) is None

    trigger = build_world_pulse_journal_trigger(result)
    assert trigger.trigger_kind == "world_pulse_digest"
    assert trigger.source_kind == "world_pulse"
    assert trigger.source_ref == run_id
    dedupe_key = cooldown_key_for_trigger(trigger)
    assert dedupe_key == f"actions:journal:world_pulse_digest:world_pulse:{run_id}"

    dry = fixture_result(run_id=run_id, dry_run=True)
    assert world_pulse_journal_skip_reason(dry, enabled=True, allow_dry_run=False) == "world_pulse_dry_run"

    dispatched = []

    async def capture_dispatch(env, *, trigger, audit_action: str, dedupe_key: str, reason=None):
        dispatched.append((trigger, audit_action, dedupe_key))

    from uuid import uuid4
    env = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="smoke", version="0.0.0"),
        correlation_id=str(uuid4()),
        payload=result.model_dump(mode="json"),
    )
    assert await handle_world_pulse_run_result_journal(
        env,
        settings=settings,
        dispatch_journal=capture_dispatch,
        audit=AsyncMock(),
    )
    assert len(dispatched) == 1
    t, action, key = dispatched[0]
    assert t.source_ref == run_id
    assert action == "journal.world_pulse_digest"
    assert key == dedupe_key

    skip_calls = []

    async def audit_skip(_env, *, status, event_id, action_name, reason=None, extra=None):
        skip_calls.append(str(reason))

    env_dry = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="smoke", version="0.0.0"),
        correlation_id=str(uuid4()),
        payload=dry.model_dump(mode="json"),
    )
    await handle_world_pulse_run_result_journal(
        env_dry,
        settings=settings,
        dispatch_journal=capture_dispatch,
        audit=audit_skip,
    )
    if not allow:
        assert skip_calls == ["world_pulse_dry_run"]
        assert len(dispatched) == 1

    print("world_pulse_reflective_journal_smoke: ok")


asyncio.run(run_smoke())
'''


if __name__ == "__main__":
    raise SystemExit(main())
