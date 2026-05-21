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
