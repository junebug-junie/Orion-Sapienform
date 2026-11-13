"""The container restart census this controller carries on its heartbeat (circe crash loops)."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import pytest

from test_api import _load, main_module  # noqa: F401  (shared module loader for this suite)

census = _load("container_census")

from orion.schemas.telemetry.system_health import (  # noqa: E402
    CONTAINER_RESTARTS_DETAIL_KEY,
    ContainerRestartsV1,
    SystemHealthV1,
)

NOW = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)


class _Runner:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[str]] = []

    def run(self, command, **_):
        self.calls.append(command)
        rc, out, err = self.replies.pop(0)
        return subprocess.CompletedProcess(command, rc, out, err)


@pytest.fixture(autouse=True)
def _fresh():
    census.reset_for_tests()
    yield
    census.reset_for_tests()


def test_parse_inspect_keys_by_name_and_short_id():
    text = "5e1f0c3a9b2dffff /orion-circe-affectgpt-worker 601\nabcdef1234567890 /orion-circe-biometrics 0\njunk\n"
    assert census.parse_inspect(text) == {
        "orion-circe-affectgpt-worker@5e1f0c3a9b2d": 601,
        "orion-circe-biometrics@abcdef123456": 0,
    }


def test_read_is_read_only_ps_then_inspect():
    runner = _Runner([(0, "aaa\nbbb\n", ""), (0, "aaa111111111 /x 3\nbbb222222222 /y 0\n", "")])
    assert census.read_restart_counts(runner) == {"x@aaa111111111": 3, "y@bbb222222222": 0}
    assert runner.calls[0] == ["docker", "ps", "-aq"]
    assert runner.calls[1][:4] == ["docker", "inspect", "--format", census.INSPECT_FORMAT]
    assert runner.calls[1][4:] == ["aaa", "bbb"]


def test_container_removed_between_ps_and_inspect_keeps_the_rest():
    runner = _Runner([(0, "aaa\nbbb\n", ""), (1, "aaa111111111 /x 3\n", "Error: No such object: bbb")])
    assert census.read_restart_counts(runner) == {"x@aaa111111111": 3}


def test_no_containers_is_an_empty_census_not_an_error():
    snap = census.refresh(_Runner([(0, "", "")]), now=lambda: NOW)
    assert snap.counts == {} and snap.error is None


def test_failed_read_is_published_as_error_not_as_calm():
    snap = census.refresh(_Runner([(1, "", "Cannot connect to the Docker daemon")]), now=lambda: NOW)
    assert snap.counts == {} and "Cannot connect" in snap.error
    detail = census.heartbeat_details()[CONTAINER_RESTARTS_DETAIL_KEY]
    assert detail["error"] and detail["counts"] == {}


def test_heartbeat_is_empty_before_the_first_read():
    assert census.heartbeat_details() == {}


def test_heartbeat_detail_round_trips_through_the_bus_contract():
    census.refresh(_Runner([(0, "aaa\n", ""), (0, "aaa111111111 /x 3\n", "")]), now=lambda: NOW)
    details = census.heartbeat_details()
    hb = SystemHealthV1(service="gpu-lane-controller", node="circe", boot_id="b", last_seen_ts=NOW, details=details)
    back = ContainerRestartsV1.model_validate(hb.model_dump(mode="json")["details"][CONTAINER_RESTARTS_DETAIL_KEY])
    assert back.counts == {"x@aaa111111111": 3} and back.read_at == NOW


def test_census_ships_on_and_rides_the_actuator_heartbeat(monkeypatch):
    s = type(main_module.settings)(_env_file=None)
    assert s.GPU_LANE_CONTAINER_CENSUS_ENABLED is True
    assert s.GPU_LANE_CONTAINER_CENSUS_INTERVAL_SEC == 30.0
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTAINER_CENSUS_ENABLED", True)
    hunter = main_module.build_actuator_chassis()
    assert hunter._heartbeat_details is main_module.container_census.heartbeat_details
    monkeypatch.setattr(main_module.settings, "GPU_LANE_CONTAINER_CENSUS_ENABLED", False)
    assert main_module.build_actuator_chassis()._heartbeat_details is None


def test_census_loop_keeps_reading_after_an_error(monkeypatch):
    import asyncio

    calls = []

    def fake_refresh(runner):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return ContainerRestartsV1(read_at=NOW)

    monkeypatch.setattr(census, "refresh", fake_refresh)

    async def run():
        task = asyncio.create_task(census.census_loop(0.0))
        while len(calls) < 3:
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert len(calls) >= 3
