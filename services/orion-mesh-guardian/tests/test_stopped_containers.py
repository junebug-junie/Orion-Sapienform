from __future__ import annotations

import asyncio

import httpx

from app import service as service_module
from app.service import MeshGuardianService, _docker_container_states, _parse_docker_time
from app.settings import Settings
from app.stability import AlertGate, ContainerState, StoppedContainerTracker, stopped_containers

T0 = 1_793_000_000.0
GRACE = 600.0


def _c(name, status, policy="unless-stopped", since=T0, svc="orion-x/x", cid=None, workdir="orion-x"):
    return ContainerState(
        id=cid or (name.encode().hex() + "0" * 64)[:64],
        name=name,
        status=status,
        restart_policy=policy,
        since=since,
        compose_service=svc,
        compose_dir=workdir,
    )


# 2026-10-29: the post-pull rebuild died mid `up -d --build` of orion-harness-governor. compose
# had stopped the old container (exited 0) and left its replacement "Created" under a temp name.
GOVERNOR_SVC = "orion-athena/harness-governor"
INCIDENT = [
    _c("orion-athena-hub", "running", svc="orion-athena/hub", since=None),
    _c("orion-athena-harness-governor", "exited", since=T0, svc=GOVERNOR_SVC, workdir="orion-harness-governor"),
    _c("ab1a1dd30eb5_orion-athena-harness-governor", "created", since=T0 - 900, svc=GOVERNOR_SVC,
       workdir="orion-harness-governor"),
]


class TestIncidentReplay:
    def test_cards_once_and_never_again_for_the_whole_outage(self) -> None:
        # Real order: replacement created ~07:21:40 (T0 - 900) while the old container drained;
        # the old one exited 07:36 (T0). Before T0 the service was still being served.
        draining = [INCIDENT[0], _c("orion-athena-harness-governor", "running", svc=GOVERNOR_SVC, since=None), INCIDENT[2]]
        tracker = StoppedContainerTracker()
        for t in range(-900, 0, 60):
            assert tracker.observe(draining, T0 + t, grace_sec=GRACE) == []
        # The temp container has been "Created" past the grace, and nothing serves the service now.
        alerts = tracker.observe(INCIDENT, T0 + 60, grace_sec=GRACE)
        assert [(a.kind, a.subject) for a in alerts] == [("stopped_container", GOVERNOR_SVC)]
        assert alerts[0].context["containers"] == ["ab1a1dd30eb5_orion-athena-harness-governor"]
        assert "scripts/safe_docker_build.sh orion-harness-governor up -d harness-governor" in alerts[0].message
        # The old container passing its own grace, then 9.5 hours of the same state: no second card.
        for minute in range(2, 570):
            assert tracker.observe(INCIDENT, T0 + minute * 60, grace_sec=GRACE) == []

    def test_both_containers_past_the_grace_make_one_card(self) -> None:
        alerts = StoppedContainerTracker().observe(INCIDENT, T0 + GRACE, grace_sec=GRACE)
        assert len(alerts) == 1
        assert alerts[0].context["containers"] == [
            "orion-athena-harness-governor",
            "ab1a1dd30eb5_orion-athena-harness-governor",
        ]

    def test_the_temp_container_alone_is_flagged_even_without_a_restart_policy(self) -> None:
        temp = _c("ab1a1dd30eb5_orion-athena-harness-governor", "created", policy="no", svc=None)
        assert stopped_containers([temp], T0 + GRACE, grace_sec=GRACE) == {
            "orion-athena-harness-governor": [(temp, "half-finished compose recreate")]
        }


class TestRules:
    def test_healthy_host_is_silent(self) -> None:
        live = [_c(f"svc{i}", "running", svc=f"p/s{i}", since=None) for i in range(50)]
        assert StoppedContainerTracker().observe(live, T0 + 10**6, grace_sec=GRACE) == []

    def test_restart_policy_no_is_never_flagged(self) -> None:
        # e.g. orion-analytics-lightdash-object-storage-init-1: a one-shot init job.
        job = _c("init-1", "exited", policy="no")
        assert stopped_containers([job], T0 + 10**6, grace_sec=GRACE) == {}

    def test_always_policy_is_flagged(self) -> None:
        assert stopped_containers([_c("a", "exited", policy="always")], T0 + GRACE, grace_sec=GRACE)

    def test_leftover_beside_a_running_replacement_is_not_an_outage(self) -> None:
        old = _c("orion-athena-x", "exited", svc="p/x")
        new = _c("orion-athena-x-2", "running", svc="p/x", since=None)
        assert stopped_containers([old, new], T0 + 10**6, grace_sec=GRACE) == {}

    def test_no_known_stop_time_is_not_flagged(self) -> None:
        assert stopped_containers([_c("a", "exited", since=None)], T0 + 10**6, grace_sec=GRACE) == {}

    def test_a_new_stop_of_the_same_container_is_a_new_card(self) -> None:
        tracker = StoppedContainerTracker()
        first = _c("a", "exited", since=T0)
        assert len(tracker.observe([first], T0 + GRACE, grace_sec=GRACE)) == 1
        tracker.observe([_c("a", "running", since=None)], T0 + 2 * GRACE, grace_sec=GRACE)
        again = _c("a", "exited", since=T0 + 3 * GRACE)
        alerts = tracker.observe([again], T0 + 4 * GRACE, grace_sec=GRACE)
        assert len(alerts) == 1
        # The AlertGate must not swallow it as a repeat of the first card's key.
        assert AlertGate().admit(alerts, T0 + 4 * GRACE) == alerts


class TestDockerCollector:
    def test_parse_docker_time(self) -> None:
        assert _parse_docker_time("2026-10-15T21:11:12.000465157Z") == 1792098672.000465
        assert _parse_docker_time("0001-01-01T00:00:00Z") is None
        assert _parse_docker_time("") is None
        assert _parse_docker_time("garbage") is None

    def test_inspects_only_the_non_running_containers(self) -> None:
        listing = [
            {"Id": "a" * 64, "Names": ["/orion-athena-hub"], "State": "running", "Created": 1,
             "Labels": {"com.docker.compose.project": "orion-athena", "com.docker.compose.service": "hub"}},
            {"Id": "b" * 64, "Names": ["/orion-athena-harness-governor"], "State": "exited", "Created": 1,
             "Labels": {"com.docker.compose.project": "orion-athena",
                        "com.docker.compose.service": "harness-governor",
                        "com.docker.compose.project.working_dir": "/repo/services/orion-harness-governor"}},
            {"Id": "c" * 64, "Names": ["/ab1a1dd30eb5_orion-athena-harness-governor"], "State": "created",
             "Created": 1793000000, "Labels": {}},
            {"Id": "d" * 64, "Names": ["/gone"], "State": "exited", "Created": 1, "Labels": {}},
        ]
        inspected = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/containers/json":
                return httpx.Response(200, json=listing)
            cid = request.url.path.split("/")[2]
            inspected.append(cid[0])
            if cid.startswith("d"):
                return httpx.Response(404, json={})
            return httpx.Response(200, json={
                "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}},
                "State": {"FinishedAt": "2026-10-29T07:36:00.5Z" if cid.startswith("b") else "0001-01-01T00:00:00Z"},
            })

        states = asyncio.run(_docker_container_states(transport=httpx.MockTransport(handler)))

        assert inspected == ["b", "c", "d"]
        by_name = {s.name: s for s in states}
        assert set(by_name) == {"orion-athena-hub", "orion-athena-harness-governor",
                                "ab1a1dd30eb5_orion-athena-harness-governor"}
        gov = by_name["orion-athena-harness-governor"]
        assert (gov.restart_policy, gov.since, gov.compose_service, gov.compose_dir) == (
            "unless-stopped", 1793259360.5, "orion-athena/harness-governor", "orion-harness-governor")
        assert by_name["ab1a1dd30eb5_orion-athena-harness-governor"].since == 1793000000.0


class _Attention:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish_transition(self, *, service_id, heartbeat_name, event) -> None:
        self.events.append({"service_id": service_id, "heartbeat_name": heartbeat_name, **event})


def test_the_cycle_turns_the_incident_into_one_hub_card(monkeypatch) -> None:
    """End to end through run_stability_checks -> _publish_alerts -> attention (Hub card)."""
    guardian = MeshGuardianService(Settings())
    attention = _Attention()
    guardian.attention = attention

    async def states():
        return INCIDENT

    async def boom(_now):
        raise RuntimeError("not under test")

    monkeypatch.setattr(service_module, "_docker_container_states", states)
    guardian._check_crash_loops = boom
    guardian._check_bus_redis = boom
    guardian._check_falkordb = boom

    asyncio.run(guardian.run_stability_checks(T0 + GRACE))
    asyncio.run(guardian.run_stability_checks(T0 + GRACE + 60))

    assert [(e["service_id"], e["context"]["event"], e["heartbeat_name"]) for e in attention.events] == [
        (GOVERNOR_SVC, "stopped_container", "stability")
    ]


def test_disabled_watch_reads_nothing(monkeypatch) -> None:
    guardian = MeshGuardianService(Settings(MESH_GUARDIAN_STOPPED_CONTAINER_WATCH_ENABLED=False))

    async def states():
        raise AssertionError("must not read docker")

    monkeypatch.setattr(service_module, "_docker_container_states", states)
    assert asyncio.run(guardian._check_stopped_containers(T0)) == ([], "disabled")
