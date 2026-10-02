from __future__ import annotations

import asyncio
import logging

import pytest

from app import service as service_module
from app.attention import AttentionPublisher
from app.service import MeshGuardianService, _docker_restart_counts, _parse_restart_counts, _save_config
from app.settings import Settings
from app.stability import (
    CRASH_LOOP_WINDOW_SEC,
    REALERT_AFTER_SEC,
    SNAPSHOT_STALE_SEC,
    SNAPSHOT_STUCK_SEC,
    AlertGate,
    CounterRiseTracker,
    CrashLoopTracker,
    graph_inflation_alert,
    slow_consumer_alert,
    snapshot_alerts,
)

T0 = 1_790_900_000.0


class TestCrashLoopTracker:
    def test_steady_restart_count_never_alerts(self) -> None:
        tracker = CrashLoopTracker()
        for i in range(200):
            assert tracker.observe({"svc": 5}, T0 + i * 60) == []

    def test_three_restarts_inside_the_window_alerts(self) -> None:
        tracker = CrashLoopTracker()
        assert tracker.observe({"svc": 10}, T0) == []
        assert tracker.observe({"svc": 12}, T0 + 1200) == []
        alerts = tracker.observe({"svc": 13}, T0 + 2400)
        assert [a.kind for a in alerts] == ["crash_loop"]
        assert alerts[0].context["restarts_in_window"] == 3

    def test_slow_restarts_outside_the_window_do_not_alert(self) -> None:
        tracker = CrashLoopTracker()
        step = CRASH_LOOP_WINDOW_SEC  # one restart per window length
        for i in range(6):
            assert tracker.observe({"svc": i}, T0 + i * step) == []

    def test_recreated_container_resets_instead_of_counting_negative(self) -> None:
        tracker = CrashLoopTracker()
        tracker.observe({"svc": 581}, T0)
        assert tracker.observe({"svc": 0}, T0 + 60) == []
        assert tracker.observe({"svc": 2}, T0 + 120) == []


class TestCounterRise:
    def test_first_value_is_only_a_baseline(self) -> None:
        tracker = CounterRiseTracker()
        assert tracker.observe(581) == 0
        assert tracker.observe(581) == 0
        assert tracker.observe(583) == 2

    def test_redis_restart_rebaselines(self) -> None:
        tracker = CounterRiseTracker()
        tracker.observe(581)
        assert tracker.observe(0) == 0
        assert tracker.observe(1) == 1

    def test_slow_consumer_alert_only_on_rise(self) -> None:
        assert slow_consumer_alert("bus-redis", 0, 581) == []
        assert [a.kind for a in slow_consumer_alert("bus-redis", 1, 582)] == ["slow_consumer_kill"]


class TestSnapshotAlerts:
    HEALTHY = {
        "rdb_bgsave_in_progress": 0,
        "rdb_current_bgsave_time_sec": -1,
        "rdb_last_bgsave_status": "ok",
        "rdb_last_save_time": T0 - 70,
        "rdb_changes_since_last_save": 7363,
    }

    def test_healthy_is_quiet(self) -> None:
        assert snapshot_alerts("x", self.HEALTHY, save_config="3600 1", now=T0) == []

    def test_a_normal_in_progress_save_is_quiet(self) -> None:
        info = {**self.HEALTHY, "rdb_bgsave_in_progress": 1, "rdb_current_bgsave_time_sec": 2}
        assert snapshot_alerts("x", info, save_config="3600 1", now=T0) == []

    def test_stuck_save_alerts_critical(self) -> None:
        info = {**self.HEALTHY, "rdb_bgsave_in_progress": 1, "rdb_current_bgsave_time_sec": SNAPSHOT_STUCK_SEC}
        alerts = snapshot_alerts("falkordb", info, save_config="3600 1", now=T0)
        assert [(a.kind, a.severity) for a in alerts] == [("snapshot_stuck", "critical")]

    def test_failed_save_alerts(self) -> None:
        info = {**self.HEALTHY, "rdb_last_bgsave_status": "err"}
        assert [a.kind for a in snapshot_alerts("x", info, save_config="3600 1", now=T0)] == ["snapshot_failed"]

    def test_overdue_only_when_snapshots_are_configured(self) -> None:
        info = {**self.HEALTHY, "rdb_last_save_time": T0 - SNAPSHOT_STALE_SEC}
        assert [a.kind for a in snapshot_alerts("x", info, save_config="3600 1", now=T0)] == ["snapshot_overdue"]
        assert snapshot_alerts("x", info, save_config="", now=T0) == []

    def test_overdue_needs_unsaved_changes(self) -> None:
        info = {**self.HEALTHY, "rdb_last_save_time": T0 - SNAPSHOT_STALE_SEC, "rdb_changes_since_last_save": 0}
        assert snapshot_alerts("x", info, save_config="3600 1", now=T0) == []


class TestGraphInflation:
    def test_healthy_graph_is_quiet(self) -> None:
        assert graph_inflation_alert("g", 242, 332) == []

    def test_inflated_graph_alerts(self) -> None:
        assert [a.kind for a in graph_inflation_alert("g", 118_112, 332)] == ["graph_inflation"]

    def test_unknown_catalog_never_alerts(self) -> None:
        assert graph_inflation_alert("g", 118_112, 0) == []


class TestAlertGate:
    def test_same_key_is_suppressed_until_realert_window(self) -> None:
        gate = AlertGate()
        alert = graph_inflation_alert("g", 118_112, 332)
        assert len(gate.admit(alert, T0)) == 1
        assert gate.admit(alert, T0 + 60) == []
        assert len(gate.admit(alert, T0 + REALERT_AFTER_SEC)) == 1


class _FakeRedis:
    def __init__(self, *, stats=None, persistence=None, save="3600 1", graph_count=None) -> None:
        self._stats = stats or {}
        self._persistence = persistence or {}
        self._save = save
        self._graph_count = graph_count

    async def info(self, section: str):
        return self._stats if section == "stats" else self._persistence

    async def config_get(self, key: str):
        return {"save": self._save}

    async def execute_command(self, *args):
        if self._graph_count is None:
            raise ConnectionError("falkordb down")
        return [["count(c)"], [[self._graph_count]], ["stats"]]

    async def aclose(self) -> None:
        return None


class _FakeBus:
    def __init__(self, redis) -> None:
        self.redis = redis


class _RecordingAttention:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish_transition(self, *, service_id: str, heartbeat_name: str, event: dict) -> None:
        self.events.append({"service_id": service_id, "heartbeat_name": heartbeat_name, **event})


def _guardian(monkeypatch, *, bus_redis, falkordb, restart_counts) -> tuple[MeshGuardianService, _RecordingAttention]:
    guardian = MeshGuardianService(Settings())
    guardian.bus = _FakeBus(bus_redis)
    guardian._falkordb = falkordb
    attention = _RecordingAttention()
    guardian.attention = attention

    async def fake_counts():
        if isinstance(restart_counts, Exception):
            raise restart_counts
        return restart_counts

    monkeypatch.setattr(service_module, "_docker_restart_counts", fake_counts)
    return guardian, attention


class TestServiceCycle:
    def test_one_broken_source_does_not_blind_the_others(self, monkeypatch) -> None:
        stuck = {"rdb_bgsave_in_progress": 1, "rdb_current_bgsave_time_sec": 138_655, "rdb_last_bgsave_status": "ok"}
        guardian, attention = _guardian(
            monkeypatch,
            bus_redis=_FakeRedis(stats={"client_output_buffer_limit_disconnections": 0}, persistence=stuck),
            falkordb=_FakeRedis(persistence=stuck, graph_count=None),  # graph query raises
            restart_counts=RuntimeError("docker socket missing"),
        )

        alerts = asyncio.run(guardian.run_stability_checks(T0))

        # Docker and the FalkorDB graph query both fail; the bus check must
        # still run, and FalkorDB's own stuck save must survive its graph
        # query failing (review finding: one try used to discard both).
        assert [(a.subject, a.kind) for a in alerts] == [
            ("bus-redis", "snapshot_stuck"),
            ("falkordb", "snapshot_stuck"),
        ]
        assert [e["service_id"] for e in attention.events] == ["bus-redis", "falkordb"]

    def test_alerts_become_attention_cards_once(self, monkeypatch) -> None:
        stuck = {"rdb_bgsave_in_progress": 1, "rdb_current_bgsave_time_sec": 138_655, "rdb_last_bgsave_status": "ok"}
        guardian, attention = _guardian(
            monkeypatch,
            bus_redis=_FakeRedis(stats={"client_output_buffer_limit_disconnections": 0}),
            falkordb=_FakeRedis(persistence=stuck, graph_count=242),
            restart_counts={"svc": 0},
        )

        asyncio.run(guardian.run_stability_checks(T0))
        asyncio.run(guardian.run_stability_checks(T0 + 60))

        assert [(e["service_id"], e["context"]["event"]) for e in attention.events] == [
            ("falkordb", "snapshot_stuck")
        ]
        assert attention.events[0]["severity"] == "critical"
        assert attention.events[0]["heartbeat_name"] == "stability"


class TestAttentionDeliveryIsChecked:
    def test_undelivered_card_is_logged_as_error(self, caplog) -> None:
        publisher = AttentionPublisher(Settings())

        class _Rejected:
            ok = False
            detail = "connection refused"

        publisher._client.attention_request = lambda **_: _Rejected()
        with caplog.at_level(logging.ERROR, logger="orion.mesh.guardian.attention"):
            publisher.publish_transition(service_id="svc", heartbeat_name="stability", event={"context": {"event": "x"}})
        assert "NOT delivered" in caplog.text

    def test_delivered_card_is_quiet(self, caplog) -> None:
        publisher = AttentionPublisher(Settings())

        class _Accepted:
            ok = True

        publisher._client.attention_request = lambda **_: _Accepted()
        with caplog.at_level(logging.ERROR, logger="orion.mesh.guardian.attention"):
            publisher.publish_transition(service_id="svc", heartbeat_name="stability", event={"context": {"event": "x"}})
        assert "NOT delivered" not in caplog.text


class TestCollectorParsing:
    def test_restart_counts_keyed_by_name_and_id(self) -> None:
        text = (
            "e653466f3601aaaa /orion-athena-bus-mirror 581\n"
            "1234567890abcdef /orion-athena-hub 0\n"
            "garbage line\n"
        )
        assert _parse_restart_counts(text) == {
            "orion-athena-bus-mirror@e653466f3601": 581,
            "orion-athena-hub@1234567890ab": 0,
        }

    def test_recreated_container_is_a_new_history(self) -> None:
        tracker = CrashLoopTracker()
        tracker.observe({"svc@old": 10}, T0)
        # Same name, new id, count already at the old level: not 3 new restarts.
        assert tracker.observe({"svc@new": 12}, T0 + 60) == []

    def test_crash_loop_card_names_the_container_not_the_id(self) -> None:
        tracker = CrashLoopTracker()
        tracker.observe({"svc@abc": 0}, T0)
        alerts = tracker.observe({"svc@abc": 3}, T0 + 600)
        assert alerts[0].subject == "svc"
        assert alerts[0].key == "crash_loop:svc"

    def test_save_config_accepts_str_and_bytes(self) -> None:
        class _Cfg:
            def __init__(self, value):
                self.value = value

            async def config_get(self, key):
                return self.value

        assert asyncio.run(_save_config(_Cfg({"save": "3600 1"}))) == "3600 1"
        assert asyncio.run(_save_config(_Cfg({b"save": b"3600 1"}))) == "3600 1"
        assert asyncio.run(_save_config(_Cfg({"save": ""}))) == ""


def test_docker_restart_counts_reads_the_engine_api() -> None:
    import httpx

    containers = {
        "aaaaaaaaaaaa1111": {"Name": "/orion-athena-bus-mirror", "RestartCount": 581},
        "bbbbbbbbbbbb2222": {"Name": "/orion-athena-hub", "RestartCount": 0},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/containers/json":
            assert request.url.params["all"] == "true"
            return httpx.Response(200, json=[{"Id": cid} for cid in [*containers, "gone"]])
        cid = request.url.path.split("/")[2]
        if cid not in containers:
            return httpx.Response(404, json={"message": "No such container"})
        return httpx.Response(200, json={"Id": cid, **containers[cid]})

    counts = asyncio.run(_docker_restart_counts(transport=httpx.MockTransport(handler)))

    assert counts == {"orion-athena-bus-mirror@aaaaaaaaaaaa": 581, "orion-athena-hub@bbbbbbbbbbbb": 0}
