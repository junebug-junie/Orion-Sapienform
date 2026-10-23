"""Remote hosts' crash loops: census from the equilibrium snapshot, host-qualified, blind != calm."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from app import service as service_module
from app.service import MeshGuardianService
from app.settings import Settings
from app.stability import (
    REMOTE_CENSUS_STALE_SEC,
    CrashLoopTracker,
    qualify,
    remote_restart_counts,
)
from orion.schemas.telemetry.system_health import CONTAINER_RESTARTS_DETAIL_KEY

T0 = 1_792_800_000.0  # ~2026-10-23


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _svc(node: str, counts: dict[str, int] | None, *, read_at: float, error: str | None = None,
         service: str = "gpu-lane-controller"):
    details = {}
    if counts is not None or error is not None:
        details[CONTAINER_RESTARTS_DETAIL_KEY] = {"read_at": _iso(read_at), "counts": counts or {}, "error": error}
    return SimpleNamespace(service=service, node=node, details=details)


def _census(services, *, now=T0, expected=("circe",), since=T0 - 3600):
    return remote_restart_counts(services, expected_hosts=expected, local_node="athena", now=now,
                                 watching_since=since)


class TestRemoteCensus:
    def test_fresh_census_is_host_qualified(self) -> None:
        got = _census([_svc("circe", {"orion-circe-affectgpt-worker@abc": 7}, read_at=T0 - 20)])
        assert got.counts == {"circe/orion-circe-affectgpt-worker@abc": 7}
        assert got.alerts == []

    def test_same_container_name_on_two_hosts_never_collides(self) -> None:
        got = _census([
            _svc("circe", {"orion-x@aaa": 1}, read_at=T0),
            _svc("hecate", {"orion-x@aaa": 50}, read_at=T0),
        ])
        assert got.counts == {"circe/orion-x@aaa": 1, "hecate/orion-x@aaa": 50}
        tracker = CrashLoopTracker()
        tracker.observe({**qualify("athena", {"orion-x@aaa": 0}), **got.counts}, T0)
        alerts = tracker.observe({"athena/orion-x@aaa": 0, "circe/orion-x@aaa": 4, "hecate/orion-x@aaa": 50}, T0 + 60)
        assert [a.key for a in alerts] == ["crash_loop:circe/orion-x"]
        assert "`docker logs orion-x` on circe" in alerts[0].message
        assert alerts[0].context["host"] == "circe"

    def test_silent_host_reads_blind_not_calm(self) -> None:
        got = _census([])
        assert got.counts == {}
        assert [(a.kind, a.subject, a.severity) for a in got.alerts] == [("container_watch_blind", "circe", "warning")]
        assert "no container census" in got.alerts[0].message

    def test_stale_census_reads_blind_and_feeds_nothing(self) -> None:
        got = _census([_svc("circe", {"svc@a": 3}, read_at=T0 - REMOTE_CENSUS_STALE_SEC - 1)])
        assert got.counts == {}
        assert got.alerts[0].kind == "container_watch_blind"
        assert got.alerts[0].context["last_census_age_sec"] == REMOTE_CENSUS_STALE_SEC + 1

    def test_failed_read_reads_blind_with_the_error(self) -> None:
        got = _census([_svc("circe", None, read_at=T0 - 5, error="RuntimeError: docker ps rc=1")])
        assert got.counts == {}
        assert "docker ps rc=1" in got.alerts[0].message

    def test_boot_grace_before_the_first_census_can_arrive(self) -> None:
        assert _census([], since=T0 - 10).alerts == []
        assert len(_census([], since=T0 - REMOTE_CENSUS_STALE_SEC).alerts) == 1

    def test_unexpected_host_is_watched_but_never_blind(self) -> None:
        got = _census([_svc("hecate", {"a@1": 2}, read_at=T0)], expected=())
        assert got.counts == {"hecate/a@1": 2} and got.alerts == []
        assert _census([], expected=()).alerts == []

    def test_local_node_is_skipped(self) -> None:
        # athena's own containers come from its socket, never twice.
        assert _census([_svc("athena", {"a@1": 2}, read_at=T0)], expected=()).counts == {}

    def test_malformed_detail_is_ignored(self) -> None:
        svc = SimpleNamespace(service="x", node="circe", details={CONTAINER_RESTARTS_DETAIL_KEY: {"counts": "no"}})
        assert _census([svc]).counts == {}

    def test_freshest_reporter_wins_for_the_blind_message(self) -> None:
        got = _census([
            _svc("circe", None, read_at=T0 - 900, error="old"),
            _svc("circe", None, read_at=T0 - 10, error="new", service="other"),
        ])
        assert "new" in got.alerts[0].message and "other" in got.alerts[0].message

    def test_two_reporters_on_one_host_freshest_wins_no_merge(self) -> None:
        got = _census([
            _svc("circe", {"w@1": 9}, read_at=T0 - 10),
            _svc("circe", {"w@1": 7}, read_at=T0 - 40, service="other"),  # older, lower: must not win
        ])
        assert got.counts == {"circe/w@1": 9}

    def test_newer_producer_field_is_ignored_not_blinding(self) -> None:
        svc = _svc("circe", {"w@1": 2}, read_at=T0 - 5)
        svc.details[CONTAINER_RESTARTS_DETAIL_KEY]["host_uptime_s"] = 123
        got = _census([svc])
        assert got.counts == {"circe/w@1": 2} and got.alerts == []

    def test_malformed_census_card_says_malformed(self) -> None:
        svc = SimpleNamespace(service="x", node="circe", details={CONTAINER_RESTARTS_DETAIL_KEY: {"read_at": "no"}})
        got = _census([svc])
        assert "malformed census" in got.alerts[0].message
        assert got.alerts[0].context["last_census_age_sec"] is None

    def test_future_read_at_is_clock_skew_not_fresh(self) -> None:
        got = _census([_svc("circe", {"w@1": 2}, read_at=T0 + 600)])
        assert got.counts == {}
        assert "clock skew" in got.alerts[0].message

    def test_small_future_read_at_is_tolerated(self) -> None:
        assert _census([_svc("circe", {"w@1": 2}, read_at=T0 + 20)]).counts == {"circe/w@1": 2}


class TestTrackerScope:
    def test_unread_host_keeps_its_history(self) -> None:
        tracker = CrashLoopTracker()
        tracker.observe({"athena/a@1": 0, "circe/b@2": 0}, T0)
        # circe blind for a cycle: its history must survive.
        tracker.observe({"athena/a@1": 0}, T0 + 60, read_hosts={"athena"})
        alerts = tracker.observe({"athena/a@1": 0, "circe/b@2": 3}, T0 + 120, read_hosts={"athena", "circe"})
        assert [a.key for a in alerts] == ["crash_loop:circe/b"]

    def test_read_host_drops_removed_containers(self) -> None:
        tracker = CrashLoopTracker()
        tracker.observe({"circe/b@2": 10}, T0)
        tracker.observe({}, T0 + 60, read_hosts={"circe"})
        # Re-appearing at 13 is a fresh baseline, not 3 restarts.
        assert tracker.observe({"circe/b@2": 13}, T0 + 120, read_hosts={"circe"}) == []


class _Rec:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish_transition(self, *, service_id: str, heartbeat_name: str, event: dict) -> None:
        self.events.append({"service_id": service_id, **event})


def _guardian(monkeypatch, local_counts):
    guardian = MeshGuardianService(Settings(NODE_NAME="athena", MESH_GUARDIAN_REMOTE_CONTAINER_HOSTS="circe"))
    guardian.attention = _Rec()

    async def fake_counts():
        if isinstance(local_counts, Exception):
            raise local_counts
        return local_counts

    monkeypatch.setattr(service_module, "_docker_restart_counts", fake_counts)
    return guardian


class TestServiceCrashLoopCheck:
    def test_circe_crash_loop_alerts_even_when_local_socket_fails(self, monkeypatch) -> None:
        guardian = _guardian(monkeypatch, RuntimeError("docker socket missing"))
        for i, count in enumerate([0, 2, 4]):
            now = T0 + 60 * i
            guardian.latest_snapshot = SimpleNamespace(
                services=[_svc("circe", {"orion-circe-affectgpt-worker@c0ffee": count}, read_at=now - 15)]
            )
            alerts, info = asyncio.run(guardian._check_crash_loops(now))
        assert [a.key for a in alerts] == ["crash_loop:circe/orion-circe-affectgpt-worker"]
        assert info["local"] == "error:RuntimeError"

    def test_blind_circe_is_a_card_through_the_full_cycle(self, monkeypatch) -> None:
        guardian = _guardian(monkeypatch, {"orion-athena-hub@1": 0})
        guardian.latest_snapshot = SimpleNamespace(services=[])
        first, _ = asyncio.run(guardian._check_crash_loops(T0 - REMOTE_CENSUS_STALE_SEC))
        assert first == []  # boot grace, started by the check itself
        guardian.latest_snapshot = SimpleNamespace(services=[])
        alerts, info = asyncio.run(guardian._check_crash_loops(T0))
        assert [(a.kind, a.subject) for a in alerts] == [("container_watch_blind", "circe")]
        assert info["blind"] == ["circe"] and info["containers"] == 1

    def test_watch_off_reads_local_only(self, monkeypatch) -> None:
        guardian = _guardian(monkeypatch, {"x@1": 0})
        guardian.settings.remote_container_watch_enabled = False
        guardian.latest_snapshot = None
        alerts, info = asyncio.run(guardian._check_crash_loops(T0 + 10_000))
        assert alerts == [] and "remote" not in info

    def test_settings_ship_on_with_circe_expected(self) -> None:
        s = Settings(_env_file=None)
        assert s.remote_container_watch_enabled is True
        assert s.remote_container_host_list() == ["circe"]
