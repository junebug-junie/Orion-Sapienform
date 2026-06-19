from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.remediator import execute_remediation
from app.roster import NEVER_REMEDIATE_IDS, ProbeConfig, ProbeMode, RosterEntry


def _entry(*, entry_id: str = "landing-pad", auto_remediate: bool = True) -> RosterEntry:
    return RosterEntry(
        id=entry_id,
        heartbeat_name=entry_id,
        compose_dir="orion-landing-pad",
        compose_service="orion-landing-pad",
        include_bus_env=False,
        auto_remediate=auto_remediate,
        probe=ProbeConfig(mode=ProbeMode.http, ready_url="http://svc/ready"),
    )


@pytest.mark.asyncio
async def test_execute_remediation_runs_compose() -> None:
    with patch("app.remediator._run_command", new=AsyncMock(return_value=(0, ""))):
        result = await execute_remediation(_entry(), repo_root="/repo", tier=1)
    assert result.ok is True
    assert result.command


@pytest.mark.asyncio
async def test_execute_remediation_blocked_for_notify() -> None:
    result = await execute_remediation(_entry(entry_id="notify"), repo_root="/repo", tier=1)
    assert result.ok is False
    assert "notify" in NEVER_REMEDIATE_IDS
