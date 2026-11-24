from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.harness import docker_door_canary as canary

WIN = "2026-11-26T15:00Z/2026-11-26T19:00Z"
INSIDE = datetime(2026, 11, 26, 16, 0, tzinfo=timezone.utc)
OUTSIDE = datetime(2026, 11, 26, 19, 0, tzinfo=timezone.utc)  # end is exclusive


def test_parse_windows() -> None:
    wins = canary.parse_windows(f"{WIN}, 2026-11-27T01:00Z/2026-11-27T02:00Z")
    assert len(wins) == 2 and canary.in_window(INSIDE, wins) and not canary.in_window(OUTSIDE, wins)


@pytest.mark.parametrize("bad", ["2026-11-26T15:00Z", "2026-11-26T19:00Z/2026-11-26T15:00Z",
                                 "2026-11-26T15:00/2026-11-26T19:00", "soon/later"])
def test_parse_windows_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        canary.parse_windows(bad)


def env(**extra: str) -> dict[str, str]:
    return {canary.WINDOWS_ENV: WIN, canary.URL_ENV: "tcp://door:2375", **extra}


def test_door_inside_window_when_it_answers() -> None:
    assert canary.door_docker_host(environ=env(), now=INSIDE, answers=lambda _u: True) == "tcp://door:2375"


def test_socket_outside_window_never_probes() -> None:
    probed: list[str] = []
    assert canary.door_docker_host(environ=env(), now=OUTSIDE, answers=lambda u: probed.append(u) or True) is None
    assert probed == []


def test_socket_kept_when_door_is_down() -> None:
    assert canary.door_docker_host(environ=env(), now=INSIDE, answers=lambda _u: False) is None


def test_no_window_and_bad_window_keep_socket() -> None:
    assert canary.door_docker_host(environ={}, now=INSIDE, answers=lambda _u: True) is None
    assert canary.door_docker_host(environ={canary.WINDOWS_ENV: "garbage"}, now=INSIDE,
                                   answers=lambda _u: True) is None


def test_turn_docker_host_falls_back_to_container_value() -> None:
    assert canary.turn_docker_host(environ=env(), now=OUTSIDE) == ""
    assert canary.turn_docker_host(environ=env(DOCKER_HOST="unix:///x.sock"), now=OUTSIDE) == "unix:///x.sock"
    assert canary.turn_docker_host(environ=env(), now=INSIDE, answers=lambda _u: True) == "tcp://door:2375"


def test_build_subprocess_env_uses_door_only_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.harness import fcc_motor

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(canary, "door_answers", lambda _u, timeout=0.5: True)
    monkeypatch.setenv(canary.WINDOWS_ENV, "2000-01-01T00:00Z/2100-01-01T00:00Z")
    monkeypatch.setenv(canary.URL_ENV, "tcp://door:2375")
    on = fcc_motor._build_subprocess_env(fcc_server_url="http://fcc", auth_token="t")
    assert on["DOCKER_HOST"] == "tcp://door:2375"
    warm = fcc_motor._build_subprocess_env(fcc_server_url="http://fcc", auth_token="t", docker_door=False)
    assert "DOCKER_HOST" not in warm
    monkeypatch.setenv(canary.WINDOWS_ENV, "2000-01-01T00:00Z/2000-01-02T00:00Z")
    off = fcc_motor._build_subprocess_env(fcc_server_url="http://fcc", auth_token="t")
    assert "DOCKER_HOST" not in off


def test_github_mcp_stays_on_the_real_socket() -> None:
    template = Path(__file__).resolve().parents[2] / "fcc" / "fcc_claude_mcp.template.json"
    gh = json.loads(template.read_text())["mcpServers"]["github"]
    assert gh["env"]["DOCKER_HOST"] == "unix:///var/run/docker.sock"
