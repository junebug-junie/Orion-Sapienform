"""circe's container restart census, carried on this controller's heartbeat.

Why here: 2026-10-14..23, orion-circe-affectgpt-worker crash-looped 601 times on circe (CUDA OOM at
model load, ~144 W for 9 days) and nothing alerted. orion-mesh-guardian already detects crash loops
(RestartCount rising), but it only reads athena's docker socket. This controller is the one circe
service that already holds circe's docker socket and already publishes a heartbeat, so the census
rides that heartbeat (``SystemHealthV1.details["container_restarts"]``) to the equilibrium snapshot,
which the guardian already consumes. No new channel, no new socket, no SSH.

Read-only: ``docker ps -aq`` + ``docker inspect``. Refreshed by its own loop (not inside the
heartbeat's bounded details hook, which gives up after ~3 s), so the heartbeat only copies the last
read.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from orion.schemas.telemetry.system_health import CONTAINER_RESTARTS_DETAIL_KEY, ContainerRestartsV1

from .compose import SafeCommandRunner

INSPECT_FORMAT = "{{.Id}} {{.Name}} {{.RestartCount}}"

_latest: ContainerRestartsV1 | None = None


def parse_inspect(text: str) -> dict[str, int]:
    """``<id> /<name> <count>`` lines -> {"<name>@<id12>": count}. Same keying as the guardian's
    local reader: a container recreated under the same name starts a new history."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].isdigit():
            counts[f"{parts[1].lstrip('/')}@{parts[0][:12]}"] = int(parts[2])
    return counts


def read_restart_counts(runner: SafeCommandRunner) -> dict[str, int]:
    listed = runner.run(["docker", "ps", "-aq", "--no-trunc"])
    if listed.returncode != 0:
        raise RuntimeError(f"docker ps rc={listed.returncode}: {listed.stderr.strip()[:200]}")
    ids = listed.stdout.split()
    if not ids:
        return {}
    inspected = runner.run(["docker", "inspect", "--format", INSPECT_FORMAT, *ids])
    counts = parse_inspect(inspected.stdout)
    # A container removed between ps and inspect makes inspect exit 1 but still print the rest.
    if inspected.returncode != 0 and not counts:
        raise RuntimeError(f"docker inspect rc={inspected.returncode}: {inspected.stderr.strip()[:200]}")
    return counts


def refresh(runner: SafeCommandRunner, *, now: Callable[[], datetime] | None = None) -> ContainerRestartsV1:
    """One read. Never raises: a failed read is published as ``error`` with empty counts, so the
    guardian reports this host as blind instead of calm."""
    global _latest
    read_at = (now or (lambda: datetime.now(timezone.utc)))()
    try:
        _latest = ContainerRestartsV1(read_at=read_at, counts=read_restart_counts(runner))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[HOST] container_census_failed error={exc!r}")
        _latest = ContainerRestartsV1(read_at=read_at, error=f"{type(exc).__name__}: {exc}"[:300])
    return _latest


def heartbeat_details() -> dict[str, Any]:
    """The chassis heartbeat hook. Nothing until the first read: absent reads as blind, not calm."""
    latest = _latest
    if latest is None:
        return {}
    return {CONTAINER_RESTARTS_DETAIL_KEY: latest.model_dump(mode="json")}


async def census_loop(interval_sec: float, *, timeout_sec: float = 30.0) -> None:
    runner = SafeCommandRunner(allowed_commands={"docker"}, timeout_sec=timeout_sec)
    first = True
    while True:
        result = await asyncio.to_thread(refresh, runner)
        if first:
            logger.info(
                f"[HOST] container_census_first_read containers={len(result.counts)} error={result.error}"
            )
            first = False
        await asyncio.sleep(interval_sec)


def reset_for_tests() -> None:
    global _latest
    _latest = None
