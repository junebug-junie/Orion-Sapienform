from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.door import (  # noqa: E402
    BadRequest,
    Door,
    Record,
    Route,
    body_length,
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
    assert normalize_path("/") == "/"


def test_normalize_decodes_once_before_classify() -> None:
    # Finding 7: %61ttach is "attach" to Docker, so it must be "attach" to the door.
    path = normalize_path("/v1.43/containers/abc/%61ttach")
    assert path == "/containers/abc/attach" and classify("POST", path).kind == "change"
    assert classify("GET", normalize_path("/containers/abc/%61ttach/ws")).kind == "change"


@pytest.mark.parametrize("target", [
    "/containers/abc/%2561ttach",          # double encoding
    "/containers/abc/../xyz/kill",          # dot segments
    "/containers/%2e%2e/xyz/kill",          # encoded dot segments
    "//containers/json",                    # empty segment
    "/containers/abc%00/exec",              # NUL
    "/containers/%ff/json",                 # not UTF-8
])
def test_normalize_refuses_tricky_paths(target: str) -> None:
    with pytest.raises(BadRequest):
        normalize_path(target)


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
    ("POST", "/v1.43/containers/abc/exec", "change"),  # double version prefix after one strip
])
def test_classify(method: str, path: str, kind: str) -> None:
    assert classify(method, path).kind == kind


def test_classify_names_the_exec_action() -> None:
    assert classify("POST", "/exec/e1/start") == Route("exec_run", "e1", "start")
    assert classify("POST", "/exec/e1/resize").action == "resize"


@pytest.mark.parametrize("inspect,reason", [
    ({"HostConfig": {"Privileged": True}}, "privileged"),
    ({"HostConfig": {"PidMode": "host"}}, "pid_host"),
    ({"HostConfig": {"NetworkMode": "host"}}, "network_host"),
    ({"HostConfig": {"IpcMode": "host"}}, "ipc_host"),
    ({"HostConfig": {"UsernsMode": "host"}}, "userns_host"),
    ({"HostConfig": {"CapAdd": ["SYS_ADMIN"]}}, "cap_add:SYS_ADMIN"),
    ({"HostConfig": {"Devices": [{"PathOnHost": "/dev/video0"}]}}, "devices"),
    ({"Mounts": [{"Source": "/var/run/docker.sock", "Destination": "/var/run/docker.sock"}]},
     "socket:/var/run/docker.sock"),
    ({"Mounts": [{"Source": "/", "Destination": "/hostfs"}]}, "mount:/"),
    ({"Mounts": [{"Source": "/srv/x", "Destination": "/run/docker.sock"}]}, "socket:/srv/x"),
    # Matched by source, whatever the destination is called.
    ({"Mounts": [{"Source": "/var/run/docker.sock", "Destination": "/tmp/d"}]}, "socket:/var/run/docker.sock"),
    ({"Mounts": [{"Source": "/run/containerd/containerd.sock", "Destination": "/c"}]},
     "socket:/run/containerd/containerd.sock"),
    ({"Mounts": [{"Source": "/srv/containerd.sock", "Destination": "/c"}]}, "socket:/srv/containerd.sock"),
    ({"Mounts": [{"Source": "/var/run/tailscale/tailscaled.sock", "Destination": "/t"}]},
     "socket:/var/run/tailscale/tailscaled.sock"),
    ({"Mounts": [{"Source": "/proc", "Destination": "/host_proc"}]}, "mount:/proc"),
    ({"Mounts": [{"Source": "/sys/fs/cgroup", "Destination": "/c"}]}, "mount:/sys/fs/cgroup"),
    ({"Mounts": [{"Source": "/dev", "Destination": "/dev"}]}, "mount:/dev"),
    ({"Mounts": [{"Source": "/etc", "Destination": "/hostetc"}]}, "mount:/etc"),
    ({"Mounts": [{"Source": "/root/.ssh/key", "Destination": "/k"}]}, "mount:/root/.ssh/key"),
    ({"Mounts": [{"Source": "/var/../proc", "Destination": "/p"}]}, "mount:/proc"),
])
def test_host_reach_refuses(inspect: dict, reason: str) -> None:
    assert host_reach(inspect) == reason


@pytest.mark.parametrize("inspect", [
    {"HostConfig": {}, "Mounts": [{"Source": "/mnt/data", "Destination": "/data"}]},
    {"HostConfig": {"NetworkMode": "app-net", "CapAdd": None, "Devices": []}},
    # Named volumes live under /var/lib/docker/volumes; only /var/lib/docker itself is the host.
    {"Mounts": [{"Type": "volume", "Source": "/var/lib/docker/volumes/pg/_data", "Destination": "/var/lib/pg"}]},
    # A plain folder under /run (audio, sensors) is not a socket.
    {"Mounts": [{"Source": "/run/orion-audio", "Destination": "/run/orion-audio"}]},
    {"Mounts": [{"Type": "tmpfs", "Destination": "/tmp"}]},
])
def test_host_reach_passes_ordinary_organs(inspect: dict) -> None:
    assert host_reach(inspect) is None


def test_rebuild_head_always_closes_and_keeps_upgrade() -> None:
    # Finding 2: an Upgrade header never switches off the forced close.
    req = parse_head(b"POST /exec/e1/start HTTP/1.1\r\nHost: docker\r\nConnection: Upgrade\r\nUpgrade: tcp\r\n\r\n")
    out = rebuild_head(req).decode()
    assert out.endswith("Connection: close\r\n\r\n") and "Upgrade: tcp" in out
    assert "Connection: Upgrade" not in out and out.count("Connection:") == 1


@pytest.mark.parametrize("head", [
    b"GET /containers/json HTTP/1.1\r\nHost: d\nX: y\r\n\r\n",             # bare LF
    b"GET /containers/json HTTP/1.1\nHost: d\r\n\r\n",                      # bare LF after request line
    b"GET /containers/json HTTP/1.1\r\nHost: d\rX: y\r\n\r\n",              # stray CR
    b"GET /containers/json HTTP/1.1\r\nHost: d\r\n folded\r\n\r\n",        # folded header
    b"GET /containers/json HTTP/1.1\r\nHost: d\r\n\tfolded\r\n\r\n",
    b"GET /containers/json HTTP/2.0\r\nHost: d\r\n\r\n",                    # version
    b"GET /containers/json HTTP/1.1 \r\nHost: d\r\n\r\n",
    b"GET  /containers/json HTTP/1.1\r\nHost: d\r\n\r\n",                   # double space
    b"get /containers/json HTTP/1.1\r\nHost: d\r\n\r\n",                    # not a plain upper token
    b"G@T /containers/json HTTP/1.1\r\nHost: d\r\n\r\n",
    b"GET containers/json HTTP/1.1\r\nHost: d\r\n\r\n",                     # target not a path
    b"GET /containers/json HTTP/1.1\r\nContent-Length : 5\r\n\r\n",         # space before colon
    b"GET /containers/json HTTP/1.1\r\nNoColon\r\n\r\n",
    b"GET /containers/json HTTP/1.1\r\nHost: d\x00\r\n\r\n",                # NUL
])
def test_parse_head_is_strict(head: bytes) -> None:
    with pytest.raises(BadRequest):
        parse_head(head)


@pytest.mark.parametrize("headers", [
    b"Transfer-Encoding: chunked\r\n",
    b"Content-Length: 5\r\nTransfer-Encoding: chunked\r\n",
    b"Content-Length: 12a\r\n",
    b"Content-Length: -1\r\n",
    b"Content-Length: 1, 1\r\n",
    b"Content-Length: 1\r\nContent-Length: 1\r\n",
    b"Content-Length: 1048577\r\n",
    b"Content-Length: 2\r\nExpect: 100-continue\r\n",
])
def test_body_length_refuses_what_it_cannot_read(headers: bytes) -> None:
    req = parse_head(b"POST /containers/x/exec HTTP/1.1\r\nHost: d\r\n" + headers + b"\r\n")
    with pytest.raises(BadRequest):
        body_length(req)


def test_body_length_reads_plain_length() -> None:
    assert body_length(parse_head(b"POST /a HTTP/1.1\r\nContent-Length: 42\r\n\r\n")) == 42
    assert body_length(parse_head(b"GET /a HTTP/1.1\r\nHost: d\r\n\r\n")) is None


# ---- end to end against a keep-alive fake daemon --------------------------------------------

INSPECT = {
    "orion-athena-thought": {"Name": "/orion-athena-thought", "HostConfig": {}, "Mounts": []},
    "orion-athena-hub": {"Name": "/orion-athena-hub", "HostConfig": {},
                         "Mounts": [{"Source": "/var/run/docker.sock", "Destination": "/var/run/docker.sock"}]},
    "orion-athena-biometrics": {"Name": "/orion-athena-biometrics", "HostConfig": {"Privileged": True}, "Mounts": []},
}
KILL = b"POST /containers/orion-athena-thought/kill HTTP/1.1\r\nHost: d\r\nContent-Length: 0\r\n\r\n"


class FakeDaemon:
    """Behaves like dockerd where it matters for smuggling.

    It keeps the connection open and reads request after request until it sees
    ``Connection: close``. It accepts a bare LF as a line end. A non-detached exec start hijacks
    the connection: every later byte is stdin, echoed back.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, str], bytes]] = []
        self.stdin: list[bytes] = []

    def lines(self) -> list[str]:
        return [line for line, _h, _b in self.seen]

    async def handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while True:
                request_line = (await r.readline()).rstrip(b"\r\n")
                if not request_line:
                    return
                headers: dict[str, str] = {}
                while True:
                    raw = await r.readline()
                    if not raw:
                        return
                    raw = raw.rstrip(b"\r\n")
                    if not raw:
                        break
                    key, _, value = raw.decode("latin-1").partition(":")
                    headers[key.strip().lower()] = value.strip()
                length = int(headers.get("content-length") or 0)
                body = await r.readexactly(length) if length else b""
                line = request_line.decode("latin-1")
                self.seen.append((line, headers, body))
                target = line.split(" ")[1].split("?")[0]
                if target.endswith("/start") and "/exec/" in target and not json.loads(body or b"{}").get("Detach"):
                    w.write(b"HTTP/1.1 101 UPGRADED\r\nContent-Type: application/vnd.docker.raw-stream\r\n"
                            b"Connection: Upgrade\r\nUpgrade: tcp\r\n\r\n")
                    data = await r.read()
                    self.stdin.append(data)
                    w.write(b"echo:" + data)
                    await w.drain()
                    return
                close = headers.get("connection", "").lower() == "close"
                out = json.dumps({"ok": line}).encode()
                w.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(out)}\r\n".encode()
                        + (b"Connection: close\r\n" if close else b"") + b"\r\n" + out)
                await w.drain()
                if close:
                    return
        except (ConnectionError, asyncio.IncompleteReadError):
            return
        finally:
            w.close()


async def fake_get_json(_sock: str, path: str):
    if path.startswith("/containers/"):
        return INSPECT.get(path.split("/")[2])
    if path.startswith("/exec/"):
        exec_id = path.split("/")[2]
        if exec_id == "exec-priv":
            return {"ContainerID": "orion-athena-thought", "ProcessConfig": {"privileged": True}}
        return {"ContainerID": exec_id.removeprefix("exec-of-"), "ProcessConfig": {"privileged": False}}
    return None


async def call(port: int, raw: bytes, *, eof: bool = False, timeout: float = 5) -> bytes:
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(raw)
    await w.drain()
    if eof:
        w.write_eof()
    out = await asyncio.wait_for(r.read(), timeout)
    w.close()
    return out


def run_door(tmp_path: Path, script, **door_kwargs):
    async def main():
        fake = FakeDaemon()
        sock = str(tmp_path / "docker.sock")
        daemon = await asyncio.start_unix_server(fake.handle, sock)
        door = Door(socket_path=sock, record=Record(tmp_path / "record"), allowed_callers=(),
                    get_json=fake_get_json, **door_kwargs)
        server = await asyncio.start_server(door.handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await script(port, fake, door, sock)
        finally:
            server.close()
            daemon.close()
    return asyncio.run(main())


def records(tmp_path: Path) -> list[dict]:
    files = sorted((tmp_path / "record").glob("docker-*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


async def direct(sock: str, raw: bytes) -> bytes:
    r, w = await asyncio.open_unix_connection(sock)
    w.write(raw)
    await w.drain()
    w.write_eof()
    out = await asyncio.wait_for(r.read(), 5)
    w.close()
    return out


# The fake must be able to fall for the tricks, or the bypass tests below prove nothing.

def test_fake_daemon_reads_pipelined_requests(tmp_path: Path) -> None:
    async def script(_port, fake, _door, sock):
        await direct(sock, b"GET /containers/json HTTP/1.1\r\nHost: d\r\n\r\n" + KILL)
        return fake
    assert len(run_door(tmp_path, script).seen) == 2


def test_fake_daemon_reads_bare_lf_smuggle(tmp_path: Path) -> None:
    async def script(_port, fake, _door, sock):
        await direct(sock, SMUGGLE_LF)
        return fake
    assert [line.split(" ")[1] for line in run_door(tmp_path, script).lines()] == [
        "/containers/json", "/containers/orion-athena-thought/kill"]


SMUGGLE_LF = (b"GET /containers/json HTTP/1.1\r\nHost: d\nContent-Length: 0\n\n"
              b"POST /containers/orion-athena-thought/kill HTTP/1.1\nHost: d\r\n\r\n")


def test_bare_lf_smuggle_is_refused(tmp_path: Path) -> None:
    # Finding 1.
    async def script(port, fake, _door, _sock):
        return await call(port, SMUGGLE_LF), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 400") and fake.seen == []
    assert records(tmp_path)[0]["reason"] == "bad_request:control_byte"


def test_second_request_on_same_connection_is_not_forwarded(tmp_path: Path) -> None:
    async def script(port, fake, _door, _sock):
        out = await call(port, b"GET /containers/json HTTP/1.1\r\nHost: d\r\nConnection: keep-alive\r\n\r\n" + KILL)
        return out, fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 200")
    assert len(fake.seen) == 1 and fake.seen[0][1]["connection"] == "close"


@pytest.mark.parametrize("first", [
    # Finding 2: resize never upgrades.
    b"POST /exec/exec-of-orion-athena-thought/resize?h=10&w=10 HTTP/1.1\r\nHost: d\r\n"
    b"Connection: Upgrade\r\nUpgrade: tcp\r\nContent-Length: 0\r\n\r\n",
    # Finding 2: a detached start never upgrades.
    b"POST /exec/exec-of-orion-athena-thought/start HTTP/1.1\r\nHost: d\r\nConnection: Upgrade\r\n"
    b"Upgrade: tcp\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"Detach\":true}",
    # A body shorter than the bytes sent: the rest must not become a request.
    b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\nContent-Length: 2\r\n\r\n{}",
])
def test_upgrade_or_body_cannot_carry_a_second_request(tmp_path: Path, first: bytes) -> None:
    async def script(port, fake, _door, _sock):
        return await call(port, first + KILL, eof=True), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 200")
    assert len(fake.seen) == 1 and fake.seen[0][1]["connection"] == "close"
    assert not any("kill" in line for line in fake.lines())


def test_exec_start_streams_stdin_and_never_parses_it(tmp_path: Path) -> None:
    async def script(port, fake, _door, _sock):
        head = (b"POST /exec/exec-of-orion-athena-thought/start HTTP/1.1\r\nHost: d\r\nConnection: Upgrade\r\n"
                b"Upgrade: tcp\r\nContent-Type: application/json\r\nContent-Length: 16\r\n\r\n{\"Detach\":false}")
        return await call(port, head + b"select 1;\n" + KILL, eof=True), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 101") and out.endswith(b"echo:select 1;\n" + KILL)
    assert len(fake.seen) == 1 and fake.stdin == [b"select 1;\n" + KILL]
    assert fake.seen[0][1]["upgrade"] == "tcp" and fake.seen[0][1]["connection"] == "close"


def test_change_is_refused_and_recorded(tmp_path: Path) -> None:
    async def script(port, fake, _door, _sock):
        return await call(port, b"POST /v1.43/containers/orion-athena-thought/restart HTTP/1.1\r\nHost: d\r\n\r\n"), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 403") and b"changing containers is off" in out
    assert fake.seen == []
    assert records(tmp_path)[0]["decision"] == "refuse"


def test_encoded_attach_is_refused(tmp_path: Path) -> None:
    # Finding 7.
    async def script(port, fake, _door, _sock):
        return await call(port, b"GET /v1.43/containers/orion-athena-thought/%61ttach/ws HTTP/1.1\r\nHost: d\r\n\r\n"), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 403") and fake.seen == []


def exec_create(target: str, body: bytes, extra: bytes = b"") -> bytes:
    return (f"POST /containers/{target}/exec HTTP/1.1\r\nHost: d\r\nContent-Type: application/json\r\n".encode()
            + extra + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)


def test_exec_into_organ_passes_and_logs_command(tmp_path: Path) -> None:
    body = json.dumps({"Cmd": ["python", "-c", "print(1)"], "User": ""}).encode()

    async def script(port, fake, _door, _sock):
        return await call(port, exec_create("orion-athena-thought", body)), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 200") and len(fake.seen) == 1 and fake.seen[0][2] == body
    rec = records(tmp_path)[0]
    assert rec["decision"] == "allow" and rec["container"] == "orion-athena-thought"
    assert rec["exec_cmd"] == ["python", "-c", "print(1)"]


@pytest.mark.parametrize("raw,status,reason", [
    # Finding 3: an exec cannot ask for privileges.
    (exec_create("orion-athena-thought", b'{"Cmd":["sh"],"Privileged":true}'), b"403", "exec_privileged"),
    # Finding 3: an exec create the door cannot read is refused.
    (b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\nTransfer-Encoding: chunked\r\n\r\n"
     b"5\r\n{\"a\":\r\n0\r\n\r\n", b"400", "bad_request:transfer_encoding"),
    (b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\nContent-Length: 2000000\r\n\r\n{}",
     b"400", "bad_request:body_too_large"),
    (b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\n\r\n", b"403", "exec_body_unreadable"),
    (b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\nContent-Length: 2x\r\n\r\n{}",
     b"400", "bad_request:content_length"),
    (exec_create("orion-athena-thought", b"[1]"), b"403", "exec_body_unreadable"),
    (exec_create("orion-athena-thought", b"{nope"), b"403", "exec_body_unreadable"),
    (exec_create("orion-athena-thought%3Fx", b"{}"), b"403", "bad_ref"),
    (exec_create("orion-athena-hub", b"{}"), b"403", "socket:/var/run/docker.sock"),
    (exec_create("orion-athena-biometrics", b"{}"), b"403", "privileged"),
    (exec_create("no-such-container", b"{}"), b"403", "inspect_failed"),
])
def test_exec_create_refusals(tmp_path: Path, raw: bytes, status: bytes, reason: str) -> None:
    async def script(port, fake, _door, _sock):
        return await call(port, raw, eof=True), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 " + status) and fake.seen == []
    assert records(tmp_path)[0]["reason"] == reason


def test_exec_start_checks_the_exec_target(tmp_path: Path) -> None:
    async def script(port, fake, _door, _sock):
        ok = await call(port, b"POST /exec/exec-of-orion-athena-thought/start HTTP/1.1\r\nHost: d\r\n"
                              b"Connection: Upgrade\r\nUpgrade: tcp\r\nContent-Length: 0\r\n\r\n", eof=True)
        bad = await call(port, b"POST /exec/exec-of-orion-athena-hub/start HTTP/1.1\r\nHost: d\r\nContent-Length: 0\r\n\r\n")
        priv = await call(port, b"POST /exec/exec-priv/start HTTP/1.1\r\nHost: d\r\nContent-Length: 0\r\n\r\n")
        return ok, bad, priv, fake
    ok, bad, priv, fake = run_door(tmp_path, script)
    assert ok.startswith(b"HTTP/1.1 101") and bad.startswith(b"HTTP/1.1 403") and priv.startswith(b"HTTP/1.1 403")
    assert len(fake.seen) == 1
    assert [r["reason"] for r in records(tmp_path)] == ["exec", "socket:/var/run/docker.sock", "exec_privileged"]


def test_unknown_caller_is_refused_before_it_sends_anything(tmp_path: Path) -> None:
    # Finding 5: the caller check runs before the head is read.
    async def script(port, fake, door, _sock):
        door.allowed_callers = ("name-that-does-not-resolve.invalid",)
        r, w = await asyncio.open_connection("127.0.0.1", port)
        out = await asyncio.wait_for(r.read(), 10)
        w.close()
        return out, fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 403") and fake.seen == []
    assert records(tmp_path)[0]["reason"] == "caller"


def test_slow_head_is_dropped(tmp_path: Path) -> None:
    async def script(port, fake, _door, _sock):
        return await call(port, b"GET /containers/json HTTP/1.1\r\nHost: d\r\n", timeout=3), fake
    out, fake = run_door(tmp_path, script, head_timeout=0.2)
    assert out == b"" and fake.seen == []


def test_short_body_is_dropped(tmp_path: Path) -> None:
    async def script(port, fake, _door, _sock):
        return await call(port, b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\n"
                                b"Content-Length: 100\r\n\r\n{\"Cmd\":", eof=True, timeout=3), fake
    out, fake = run_door(tmp_path, script)
    assert out == b"" and fake.seen == []


def test_slow_body_is_dropped(tmp_path: Path) -> None:
    async def script(port, fake, _door, _sock):
        return await call(port, b"POST /containers/orion-athena-thought/exec HTTP/1.1\r\nHost: d\r\n"
                                b"Content-Length: 100\r\n\r\n{\"Cmd\":", timeout=3), fake
    out, fake = run_door(tmp_path, script, body_timeout=0.2)
    assert out == b"" and fake.seen == []


def test_failed_record_write_refuses(tmp_path: Path) -> None:
    def broken(_entry: dict) -> None:
        raise OSError("disk full")

    async def script(port, fake, door, _sock):
        door.record.write = broken
        return await call(port, b"GET /containers/json HTTP/1.1\r\nHost: d\r\n\r\n"), fake
    out, fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 503") and fake.seen == []


def test_daemon_down_answers_and_closes(tmp_path: Path) -> None:
    async def script(port, fake, door, _sock):
        door.socket_path = str(tmp_path / "missing.sock")
        return await call(port, b"GET /containers/json HTTP/1.1\r\nHost: d\r\n\r\n"), fake
    out, _fake = run_door(tmp_path, script)
    assert out.startswith(b"HTTP/1.1 502")


def test_report_lists_refusals(tmp_path: Path) -> None:
    async def script(port, _fake, _door, _sock):
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
