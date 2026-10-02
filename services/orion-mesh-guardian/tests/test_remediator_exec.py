from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import httpx

from app.remediator import deployed_from_elsewhere, docker_cli_selfcheck, execute_remediation
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
    with patch("app.remediator._run_command", new=AsyncMock(return_value=(0, ""))), patch(
        "app.remediator.deployed_from_elsewhere", new=AsyncMock(return_value=None)
    ):
        result = await execute_remediation(_entry(), repo_root="/repo", tier=1)
    assert result.ok is True
    assert result.command


@pytest.mark.asyncio
async def test_execute_remediation_blocked_for_notify() -> None:
    result = await execute_remediation(_entry(entry_id="notify"), repo_root="/repo", tier=1)
    assert result.ok is False
    assert "notify" in NEVER_REMEDIATE_IDS


def _docker(labels_by_container: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/containers/json"
        assert "com.docker.compose.service=orion-landing-pad" in request.url.params["filters"]
        return httpx.Response(200, json=[{"Labels": labels} for labels in labels_by_container])

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_deployed_from_this_checkout_is_allowed() -> None:
    wd = {"com.docker.compose.project.working_dir": "/mnt/scripts/Orion-Sapienform/services/orion-landing-pad"}
    assert await deployed_from_elsewhere(_entry(), repo_root="/mnt/scripts/Orion-Sapienform", transport=_docker([wd])) is None


@pytest.mark.asyncio
async def test_not_yet_deployed_is_allowed() -> None:
    assert await deployed_from_elsewhere(_entry(), repo_root="/mnt/scripts/Orion-Sapienform", transport=_docker([])) is None


@pytest.mark.asyncio
async def test_worktree_deploy_is_refused_and_named() -> None:
    worktree = "/mnt/scripts/Orion-Sapienform-some-fix/services/orion-landing-pad"
    transport = _docker([{"com.docker.compose.project.working_dir": worktree}])
    assert await deployed_from_elsewhere(_entry(), repo_root="/mnt/scripts/Orion-Sapienform", transport=transport) == worktree


@pytest.mark.asyncio
async def test_unreadable_docker_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no socket")

    result = await deployed_from_elsewhere(
        _entry(), repo_root="/mnt/scripts/Orion-Sapienform", transport=httpx.MockTransport(handler)
    )
    assert result is not None and result.startswith("<unreadable")


@pytest.mark.asyncio
async def test_refused_remediation_runs_nothing() -> None:
    run = AsyncMock(return_value=(0, ""))
    with patch("app.remediator._run_command", new=run), patch(
        "app.remediator.deployed_from_elsewhere", new=AsyncMock(return_value="/elsewhere")
    ):
        result = await execute_remediation(_entry(), repo_root="/repo", tier=2)
    assert result.ok is False
    assert "remediation_refused" in result.stderr_tail and "/elsewhere" in result.stderr_tail
    run.assert_not_called()


@pytest.mark.asyncio
async def test_selfcheck_reports_an_unreachable_daemon() -> None:
    err = "Error response from daemon: client version 1.43 is too old. Minimum supported API version is 1.44"
    with patch("app.remediator._run_command", new=AsyncMock(return_value=(1, err))):
        assert await docker_cli_selfcheck(repo_root="/repo") == err
    with patch("app.remediator._run_command", new=AsyncMock(side_effect=FileNotFoundError("docker"))):
        assert "FileNotFoundError" in await docker_cli_selfcheck(repo_root="/repo")
    with patch("app.remediator._run_command", new=AsyncMock(return_value=(0, ""))):
        assert await docker_cli_selfcheck(repo_root="/repo") is None
