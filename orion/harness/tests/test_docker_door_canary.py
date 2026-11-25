from __future__ import annotations

import asyncio
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


def test_build_subprocess_env_takes_the_door_from_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.harness import fcc_motor

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    # The env builder never probes (finding 6): the caller passes the door URL, or nothing.
    monkeypatch.setattr(canary, "door_answers", lambda *_a, **_k: pytest.fail("probe inside env build"))
    on = fcc_motor._build_subprocess_env(fcc_server_url="http://fcc", auth_token="t", docker_host="tcp://door:2375")
    assert on["DOCKER_HOST"] == "tcp://door:2375"
    off = fcc_motor._build_subprocess_env(fcc_server_url="http://fcc", auth_token="t")
    assert "DOCKER_HOST" not in off


def test_async_probe_does_not_block_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    def slow(_url: str, timeout: float = 0.5) -> bool:
        time.sleep(0.3)
        return True

    monkeypatch.setattr(canary, "door_answers", slow)

    async def main() -> tuple[str, float]:
        ticks: list[float] = []

        async def ticker() -> None:
            for _ in range(6):
                ticks.append(time.monotonic())
                await asyncio.sleep(0.05)

        tick = asyncio.ensure_future(ticker())
        host = await canary.turn_docker_host_async(environ=env(), now=INSIDE)
        await tick
        return host, max(b - a for a, b in zip(ticks, ticks[1:]))

    host, worst_gap = asyncio.run(main())
    assert host == "tcp://door:2375" and worst_gap < 0.2


def test_async_probe_gives_up_and_keeps_the_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(canary, "PROBE_WAIT_SEC", 0.1)
    monkeypatch.setattr(canary, "door_answers", lambda _u, timeout=0.5: time.sleep(0.4) or True)
    host = asyncio.run(canary.turn_docker_host_async(environ=env(DOCKER_HOST="unix:///s.sock"), now=INSIDE))
    assert host == "unix:///s.sock"


def test_async_probe_outside_window_returns_container_value() -> None:
    assert asyncio.run(canary.turn_docker_host_async(environ=env(), now=OUTSIDE)) == ""


def test_github_mcp_stays_on_the_real_socket() -> None:
    template = Path(__file__).resolve().parents[2] / "fcc" / "fcc_claude_mcp.template.json"
    gh = json.loads(template.read_text())["mcpServers"]["github"]
    assert gh["env"]["DOCKER_HOST"] == "unix:///var/run/docker.sock"
