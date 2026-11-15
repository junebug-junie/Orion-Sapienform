"""scripts/move_engines_behind_doors.py: which circe engines one deploy moves to loopback, and the command."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "move_engines_behind_doors.py"
spec = importlib.util.spec_from_file_location("move_engines_behind_doors", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _container(service, bindings, running=True):
    return {"Config": {"Labels": {"com.docker.compose.service": service,
                                  "com.docker.compose.project": "orion-llamacpp-host"}},
            "State": {"Running": running},
            "HostConfig": {"PortBindings": {"8080/tcp": bindings}}}


PUBLIC = [{"HostIp": "", "HostPort": "8011"}]
LOOP = [{"HostIp": "127.0.0.1", "HostPort": "18011"}]


class FakeDocker:
    def __init__(self, containers, compose_ports, move_rc=0):
        self.containers, self.compose_ports, self.move_rc = containers, compose_ports, move_rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return 0, "\n".join(f"id{i}" for i in range(len(self.containers)))
        if cmd[:2] == ["docker", "inspect"]:
            return 0, json.dumps(self.containers)
        if cmd[:2] == ["docker", "compose"] and "config" in cmd:
            return 0, json.dumps({"services": {s: {"ports": p} for s, p in self.compose_ports.items()}})
        if cmd[0] == "scripts/safe_docker_build.sh":
            return self.move_rc, "moved"
        raise AssertionError(cmd)


LOOP_CFG = [{"mode": "ingress", "host_ip": "127.0.0.1", "target": 8080, "published": "18011", "protocol": "tcp"}]
PUBLIC_CFG = [{"mode": "ingress", "target": 8080, "published": "8011", "protocol": "tcp"}]


def test_first_deploy_moves_every_public_static_engine_in_one_command():
    docker = FakeDocker(
        [_container("atlas-chat", PUBLIC), _container("atlas-agent", [{"HostIp": "", "HostPort": "8015"}]),
         _container("atlas-fast", [{"HostIp": "0.0.0.0", "HostPort": "8013"}]),
         _container("atlas-chat-announce", []),
         _container("atlas-agent-burst", [{"HostIp": "", "HostPort": "8016"}])],
        {"atlas-chat": LOOP_CFG, "atlas-agent": LOOP_CFG, "atlas-fast": LOOP_CFG})
    assert mod.main([], runner=docker, host="circe") == 0
    (move,) = [c for c in docker.calls if c[0] == "scripts/safe_docker_build.sh"]
    assert move == ["scripts/safe_docker_build.sh", "orion-llamacpp-host", "--compose-file",
                    "docker-compose.atlas-workers.yml", "--profile", "agent-split",
                    "up", "-d", "--no-build", "--no-deps", "atlas-agent", "atlas-chat", "atlas-fast"]
    assert "atlas-agent-burst" not in move           # the GPU pool's: its controller moves it


def test_every_later_deploy_is_a_no_op():
    docker = FakeDocker([_container("atlas-chat", LOOP), _container("atlas-fast", [{"HostIp": "127.0.0.1",
                                                                                     "HostPort": "18013"}])], {})
    assert mod.main([], runner=docker, host="circe") == 0
    assert not any(c[0] == "scripts/safe_docker_build.sh" or "config" in c for c in docker.calls)


def test_an_engine_pinned_to_its_lane_port_is_left_alone(capsys):
    docker = FakeDocker([_container("atlas-chat", PUBLIC)], {"atlas-chat": PUBLIC_CFG})
    assert mod.main([], runner=docker, host="circe") == 0
    assert not any(c[0] == "scripts/safe_docker_build.sh" for c in docker.calls)
    assert "pinned" in capsys.readouterr().out


def test_stopped_engines_and_other_hosts_are_never_touched():
    docker = FakeDocker([_container("atlas-chat", PUBLIC, running=False)], {"atlas-chat": LOOP_CFG})
    assert mod.main([], runner=docker, host="circe") == 0
    assert not any(c[0] == "scripts/safe_docker_build.sh" for c in docker.calls)
    other = FakeDocker([_container("atlas-chat", PUBLIC)], {"atlas-chat": LOOP_CFG})
    assert mod.main([], runner=other, host="athena") == 0 and other.calls == []


def test_the_rollback_llamacpp_chat_engine_moves_with_its_profile():
    docker = FakeDocker([_container("atlas-chat-llamacpp", PUBLIC)], {"atlas-chat-llamacpp": LOOP_CFG})
    assert mod.main([], runner=docker, host="circe") == 0
    (move,) = [c for c in docker.calls if c[0] == "scripts/safe_docker_build.sh"]
    assert "--profile" in move and "chat-llamacpp" in move and move[-1] == "atlas-chat-llamacpp"


def test_dry_run_recreates_nothing_and_a_failed_move_fails_the_hook():
    docker = FakeDocker([_container("atlas-fast", [{"HostIp": "", "HostPort": "8013"}])], {"atlas-fast": LOOP_CFG})
    assert mod.main(["--dry-run"], runner=docker, host="circe") == 0
    assert not any(c[0] == "scripts/safe_docker_build.sh" for c in docker.calls)
    failing = FakeDocker([_container("atlas-fast", [{"HostIp": "", "HostPort": "8013"}])], {"atlas-fast": LOOP_CFG},
                         move_rc=1)
    assert mod.main([], runner=failing, host="circe") == 1


def test_the_door_upstreams_match_the_engine_loopback_defaults():
    """The door's upstream defaults and the engines' compose defaults must name the same loopback ports."""
    import re

    repo = Path(__file__).resolve().parents[3]
    door = (repo / "services/orion-lane-door/docker-compose.yml").read_text()
    workers = (repo / "services/orion-llamacpp-host/docker-compose.atlas-workers.yml").read_text()
    deep = (repo / "services/orion-vllm-host/docker-compose.deep.yml").read_text()
    upstream = dict(re.findall(r"LANE_DOOR_(\w+)_UPSTREAM:-127\.0\.0\.1:(\d+)", door))
    assert upstream == {"CHAT": "18011", "AGENT": "18015", "AGENT_GPU2": "18016", "FAST": "18013",
                        "AGENT_DEEP": "18021"}
    for key, port in {"ATLAS_CHAT": "18011", "ATLAS_AGENT": "18015", "ATLAS_AGENT_BURST": "18016",
                      "ATLAS_FAST": "18013"}.items():
        found = re.findall(rf"\$\{{{key}_ENGINE_PUBLISH:-([^}}]+)\}}", workers)
        assert found and set(found) == {f"127.0.0.1:{port}"}, (key, found)
    assert "${VLLM_ENGINE_PUBLISH:-127.0.0.1:18021}" in deep
    example = (repo / "services/orion-lane-door/.env_example").read_text()
    for lane, port in upstream.items():
        assert f"LANE_DOOR_{lane}_UPSTREAM=127.0.0.1:{port}" in example
