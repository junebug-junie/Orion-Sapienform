from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.door import (  # noqa: E402
    Door,
    Record,
    Route,
    classify,
    host_reach,
    normalize_path,
    parse_head,
    rebuild_head,
)
from app.report import read_entries, summarize, parse_ts  # noqa: E402


# ---- pure rules ---------------------------------------------------------------------------

def test_normalize_strips_version_and_query() -> None:
    assert normalize_path("/v1.43/containers/json?all=1") == "/containers/json"
    assert normalize_path("/_ping") == "/_ping"


@pytest.mark.parametrize("method,path,kind", [
    ("GET", "/containers/json", "look"),
    ("GET", "/containers/abc/logs", "look"),
    ("HEAD", "/_ping", "look"),
    ("POST", "/containers/abc/exec", "exec_create"),
    ("POST", "/exec/e1/start", "exec_run"),
    ("POST", "/exec/e1/resize", "exec_run"),
    ("POST", "/containers/abc/restart", "change"),
    ("POST", "/containers/create", "change"),
    ("POST", "/containers/abc/start", "change"),
    ("POST", "/containers/abc/stop", "change"),
    ("DELETE", "/containers/abc", "change"),
    ("POST", "/images/create", "change"),
    ("POST", "/build", "change"),
    ("POST", "/containers/prune", "change"),
    ("PUT", "/containers/abc/archive", "change"),
    ("POST", "/containers/abc/attach", "change"),
    ("GET", "/containers/abc/attach/ws", "change"),
])
def test_classify(method: str, path: str, kind: str) -> None:
    assert classify(method, path).kind == kind


def test_host_reach() -> None:
    assert host_reach({"HostConfig": {"Privileged": True}}) == "privileged"
    assert host_reach({"HostConfig": {"PidMode": "host"}}) == "pid_host"
    assert host_reach({"Mounts": [{"Source": "/var/run/docker.sock", "Destination": "/var/run/docker.sock"}]})
    assert host_reach({"Mounts": [{"Source": "/", "Destination": "/hostfs"}]}) == "mount:/"
    assert host_reach({"Mounts": [{"Source": "/srv/x", "Destination": "/run/docker.sock"}]})
    assert host_reach({"HostConfig": {}, "Mounts": [{"Source": "/mnt/data", "Destination": "/data"}]}) is None


def test_rebuild_head_forces_close_but_keeps_upgrade() -> None:
    req = parse_head(b"POST /exec/e1/start HTTP/1.1\r\nHost: docker\r\nConnection: Upgrade\r\nUpgrade: tcp\r\n\r\n")
    closed = rebuild_head(req, keep_upgrade=False).decode()
    assert "Connection: close" in closed and "Upgrade" in closed.split("Connection: close")[0]
    assert "Connection: Upgrade" not in closed
    kept = rebuild_head(req, keep_upgrade=True).decode()
    assert "Connection: Upgrade" in kept and "Connection: close" not in kept


# ---- end to end against a fake daemon -----------------------------------------------------

INSPECT = {
    "orion-athena-thought": {"Name": "/orion-athena-thought", "HostConfig": {}, "Mounts": []},
    "orion-athena-hub": {"Name": "/orion-athena-hub", "HostConfig": {},
                         "Mounts": [{"Source": "/var/run/docker.sock", "Destination": "/var/run/docker.sock"}]},
    "orion-athena-biometrics": {"Name": "/orion-athena-biometrics", "HostConfig": {"Privileged": True}, "Mounts": []},
}


async def fake_daemon(path: str, seen: list[bytes]) -> asyncio.AbstractServer:
    async def handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        head = await r.readuntil(b"\r\n\r\n")
        seen.append(head)
        line = head.split(b"\r\n", 1)[0].decode()
        body = json.dumps({"ok": line}).encode()
        w.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        await w.drain()
        w.close()
    return await asyncio.start_unix_server(handle, path)


async def fake_get_json(_sock: str, path: str):
    if path.startswith("/containers/"):
        return INSPECT.get(path.split("/")[2])
    if path.startswith("/exec/"):
        return {"ContainerID": path.split("/")[2].removeprefix("exec-of-")}
    return None


async def call(port: int, raw: bytes) -> bytes:
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(raw)
    await w.drain()
    out = await asyncio.wait_for(r.read(), 5)
    w.close()
    return out


def run_door(tmp_path: Path, script):
    async def main():
        seen: list[bytes] = []
        sock = str(tmp_path / "docker.sock")
        daemon = await fake_daemon(sock, seen)
        door = Door(socket_path=sock, record=Record(tmp_path / "record"), allowed_callers=(),
                    get_json=fake_get_json)
        server = await asyncio.start_server(door.handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await script(port, seen, door)
        finally:
            server.close()
            daemon.close()
    return asyncio.run(main())


def records(tmp_path: Path) -> list[dict]:
    files = sorted((tmp_path / "record").glob("docker-*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


def test_look_passes_with_connection_close(tmp_path: Path) -> None:
    async def script(port, seen, _door):
        return await call(port, b"GET /v1.43/containers/json HTTP/1.1\r\nHost: docker\r\nConnection: keep-alive\r\n\r\n")
    out = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 200")
    rec = records(tmp_path)
    assert rec[0]["decision"] == "allow" and rec[0]["path"] == "/containers/json"


def test_second_request_on_same_connection_is_not_forwarded(tmp_path: Path) -> None:
    async def script(port, seen, _door):
        out = await call(port, b"GET /containers/json HTTP/1.1\r\nHost: d\r\n\r\n"
                               b"POST /containers/x/restart HTTP/1.1\r\nHost: d\r\n\r\n")
        return out, seen
    out, seen = run_door(tmp_path, script)
    assert len(seen) == 1 and b"Connection: close" in seen[0]
    assert b"restart" not in b"".join(seen)


def test_change_is_refused_and_recorded(tmp_path: Path) -> None:
    async def script(port, seen, _door):
        return await call(port, b"POST /v1.43/containers/orion-athena-thought/restart HTTP/1.1\r\nHost: d\r\n\r\n"), seen
    out, seen = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 403") and b"changing containers is off" in out
    assert seen == []
    assert records(tmp_path)[0]["decision"] == "refuse"


def test_exec_into_organ_passes_and_logs_command(tmp_path: Path) -> None:
    body = json.dumps({"Cmd": ["python", "-c", "print(1)"], "User": ""}).encode()
    raw = (b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\nContent-Type: application/json\r\n"
           + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)

    async def script(port, seen, _door):
        return await call(port, raw), seen
    out, seen = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 200") and len(seen) == 1
    rec = records(tmp_path)[0]
    assert rec["decision"] == "allow" and rec["container"] == "orion-athena-thought"
    assert rec["exec_cmd"] == ["python", "-c", "print(1)"]


@pytest.mark.parametrize("target,reason", [("orion-athena-hub", "mount:/var/run/docker.sock"),
                                           ("orion-athena-biometrics", "privileged"),
                                           ("no-such-container", "inspect_failed")])
def test_exec_into_host_reaching_container_is_refused(tmp_path: Path, target: str, reason: str) -> None:
    raw = f"POST /containers/{target}/exec HTTP/1.1\r\nHost: d\r\nContent-Length: 2\r\n\r\n{{}}".encode()

    async def script(port, seen, _door):
        return await call(port, raw), seen
    out, seen = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 403") and seen == []
    assert records(tmp_path)[0]["reason"] == reason


def test_exec_start_checks_the_exec_target(tmp_path: Path) -> None:
    async def script(port, seen, _door):
        ok = await call(port, b"POST /exec/exec-of-orion-athena-thought/start HTTP/1.1\r\nHost: d\r\n"
                              b"Connection: Upgrade\r\nUpgrade: tcp\r\nContent-Length: 0\r\n\r\n")
        bad = await call(port, b"POST /exec/exec-of-orion-athena-hub/start HTTP/1.1\r\nHost: d\r\nContent-Length: 0\r\n\r\n")
        return ok, bad, seen
    ok, bad, seen = run_door(tmp_path, script)
    assert ok.startswith(b"HTTP/1.1 200") and bad.startswith(b"HTTP/1.1 403")
    assert len(seen) == 1 and b"Upgrade: tcp" in seen[0] and b"Connection: Upgrade" in seen[0]


def test_unknown_caller_is_refused(tmp_path: Path) -> None:
    async def script(port, seen, door):
        door.allowed_callers = ("name-that-does-not-resolve.invalid",)
        return await call(port, b"GET /containers/json HTTP/1.1\r\nHost: d\r\n\r\n"), seen
    out, seen = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 403") and seen == []
    assert records(tmp_path)[0]["reason"] == "caller"


def test_report_lists_refusals(tmp_path: Path) -> None:
    async def script(port, seen, _door):
        await call(port, b"GET /containers/json HTTP/1.1\r\nHost: d\r\n\r\n")
        await call(port, b"POST /containers/x/restart HTTP/1.1\r\nHost: d\r\n\r\n")
    run_door(tmp_path, script)
    text = summarize(read_entries(tmp_path / "record", parse_ts("2000-01-01T00:00Z"), parse_ts("2100-01-01T00:00Z")))
    assert "2 (1 passed, 1 refused)" in text and "/containers/x/restart" in text


def test_main_refuses_to_serve_without_callers(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import door as door_mod
    monkeypatch.setenv("DOCKER_DOOR_ALLOWED_CALLERS", "")
    with pytest.raises(SystemExit):
        door_mod.main()


def test_route_dataclass_is_hashable() -> None:
    assert Route("look") == Route("look")
