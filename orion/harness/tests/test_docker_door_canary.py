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


def yes(_url: str):
    async def answer() -> bool:
        return True
    return answer()


def no(_url: str):
    async def answer() -> bool:
        return False
    return answer()


def host(**kwargs):
    return asyncio.run(canary.door_docker_host(**kwargs))


def test_door_inside_window_when_it_answers() -> None:
    assert host(environ=env(), now=INSIDE, answers=yes) == "tcp://door:2375"


def test_socket_outside_window_never_probes() -> None:
    probed: list[str] = []

    def spy(url: str):
        probed.append(url)
        return yes(url)
    assert host(environ=env(), now=OUTSIDE, answers=spy) is None
    assert probed == []


def test_socket_kept_when_door_is_down() -> None:
    assert host(environ=env(), now=INSIDE, answers=no) is None


def test_no_window_and_bad_window_keep_socket() -> None:
    assert host(environ={}, now=INSIDE, answers=yes) is None
    assert host(environ={canary.WINDOWS_ENV: "garbage"}, now=INSIDE, answers=yes) is None


def test_turn_docker_host_falls_back_to_container_value() -> None:
    run = lambda **kw: asyncio.run(canary.turn_docker_host(**kw))  # noqa: E731
    assert run(environ=env(), now=OUTSIDE) == ""
    assert run(environ=env(DOCKER_HOST="unix:///x.sock"), now=OUTSIDE) == "unix:///x.sock"
    assert run(environ=env(), now=INSIDE, answers=yes) == "tcp://door:2375"
    assert run(environ=env(DOCKER_HOST="unix:///x.sock"), now=INSIDE, answers=no) == "unix:///x.sock"


def _serve(handler):
    async def start():
        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        return server, f"tcp://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    return start


def test_door_answers_needs_a_real_ping() -> None:
    async def main() -> tuple[bool, bool, bool]:
        async def ok(r, w):
            await r.readuntil(b"\r\n\r\n")
            w.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
            w.close()

        async def bad_gateway(r, w):
            await r.readuntil(b"\r\n\r\n")
            w.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            w.close()

        async def silent(r, w):
            await asyncio.sleep(5)

        results = []
        for handler in (ok, bad_gateway, silent):
            server, url = await _serve(handler)()
            results.append(await canary.door_answers(url, timeout=0.3))
            server.close()
        return tuple(results)
    assert asyncio.run(main()) == (True, False, False)


def test_door_answers_rejects_non_tcp() -> None:
    assert asyncio.run(canary.door_answers("unix:///var/run/docker.sock")) is False


def test_probe_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake(url: str, timeout: float = 0) -> bool:
        calls.append(url)
        return True
    monkeypatch.setattr(canary, "door_answers", fake)
    monkeypatch.setattr(canary, "_probe_cache", {})

    async def main() -> list:
        return [await canary.door_docker_host(environ=env(), now=INSIDE) for _ in range(5)]
    assert asyncio.run(main()) == ["tcp://door:2375"] * 5 and calls == ["tcp://door:2375"]


def test_build_subprocess_env_takes_the_door_from_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.harness import fcc_motor

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    # The env builder never probes (first review, finding 6): the caller passes the door URL.
    on = fcc_motor._build_subprocess_env(fcc_server_url="http://fcc", auth_token="t", docker_host="tcp://door:2375")
    assert on["DOCKER_HOST"] == "tcp://door:2375"
    off = fcc_motor._build_subprocess_env(fcc_server_url="http://fcc", auth_token="t")
    assert "DOCKER_HOST" not in off


def test_slow_door_does_not_block_the_event_loop() -> None:
    import time

    async def main() -> tuple[object, float]:
        async def silent(r, w):
            await asyncio.sleep(5)
        server, url = await _serve(silent)()
        ticks: list[float] = []

        async def ticker() -> None:
            for _ in range(6):
                ticks.append(time.monotonic())
                await asyncio.sleep(0.05)

        tick = asyncio.ensure_future(ticker())
        ok = await canary.door_answers(url, timeout=0.3)
        await tick
        server.close()
        return ok, max(b - a for a, b in zip(ticks, ticks[1:]))

    ok, worst_gap = asyncio.run(main())
    assert ok is False and worst_gap < 0.2


def test_github_mcp_stays_on_the_real_socket() -> None:
    template = Path(__file__).resolve().parents[2] / "fcc" / "fcc_claude_mcp.template.json"
    gh = json.loads(template.read_text())["mcpServers"]["github"]
    assert gh["env"]["DOCKER_HOST"] == "unix:///var/run/docker.sock"
