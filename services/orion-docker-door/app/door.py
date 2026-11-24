"""Orion's Docker door: a filtering proxy in front of the real Docker socket.

Orion's turns may look (every GET/HEAD) and may ``docker exec`` into an organ, but may not change
containers (create, start, stop, restart, remove, build, pull, ...), and may not exec into a
container that leads back to the host (one that holds the Docker socket, runs privileged, shares
the host PID namespace, or bind-mounts a host root path). Every request is appended to a JSONL
record on the host. Design: docs/superpowers/specs/2026-11-24-orion-safe-autonomy-in-place-design.md.

Stdlib only. One request per client connection: the door forwards the request with
``Connection: close`` (the daemon then closes after its response), so a client cannot slip a second,
unchecked request down the same connection. An exec start that upgrades to a raw stream keeps its
``Upgrade`` headers; after the upgrade the connection carries only that exec's stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

MAX_HEADER_BYTES = 64 * 1024
MAX_EXEC_BODY_BYTES = 1024 * 1024
HOST_ROOT_SOURCES = frozenset({"/", "/run", "/var/run", "/var/lib/docker",
                               "/var/run/docker.sock", "/run/docker.sock"})
REFUSE_CHANGE = "Orion's Docker door: changing containers is off (Juniper, 2026-11-24)."
REFUSE_HOST = "Orion's Docker door: exec into a container that leads back to the host is off."
REFUSE_CALLER = "Orion's Docker door: this caller is not allowed."

_VERSION_PREFIX = re.compile(r"^/v\d+(?:\.\d+)*(/.*)$")
_EXEC_CREATE = re.compile(r"^/containers/([^/]+)/exec$")
_EXEC_RUN = re.compile(r"^/exec/([^/]+)/(start|resize)$")


def normalize_path(target: str) -> str:
    """Request target -> API path without the query string and the /vX.YZ prefix."""
    path = target.split("?", 1)[0]
    m = _VERSION_PREFIX.match(path)
    return m.group(1) if m else path


@dataclass(frozen=True)
class Route:
    """What a request asks for, before any container lookup."""

    kind: str  # "look" | "exec_create" | "exec_run" | "change"
    ref: str = ""  # container ref (exec_create) or exec id (exec_run)


def classify(method: str, path: str) -> Route:
    method = method.upper()
    if "/attach" in path:  # attach (incl. GET .../attach/ws) writes to a container's main process
        return Route("change")
    if method in ("GET", "HEAD"):
        return Route("look")
    if method == "POST":
        m = _EXEC_CREATE.match(path)
        if m:
            return Route("exec_create", m.group(1))
        m = _EXEC_RUN.match(path)
        if m:
            return Route("exec_run", m.group(1))
    return Route("change")


def host_reach(inspect: dict[str, Any]) -> Optional[str]:
    """Why this container leads back to the host, or None. Input: GET /containers/{id}/json."""
    host = inspect.get("HostConfig") or {}
    if host.get("Privileged"):
        return "privileged"
    if str(host.get("PidMode") or "") == "host":
        return "pid_host"
    for mount in inspect.get("Mounts") or []:
        src = str(mount.get("Source") or "").rstrip("/") or "/"
        dst = str(mount.get("Destination") or "")
        if src in HOST_ROOT_SOURCES or dst.endswith("/docker.sock"):
            return f"mount:{src}"
    return None


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: list[tuple[str, str]]
    raw_head: bytes

    def header(self, name: str) -> str:
        name = name.lower()
        for key, value in self.headers:
            if key.lower() == name:
                return value
        return ""


def parse_head(head: bytes) -> Request:
    text = head.decode("latin-1")
    lines = text.split("\r\n")
    method, target, version = lines[0].split(" ", 2)
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        key, _, value = line.partition(":")
        headers.append((key.strip(), value.strip()))
    return Request(method, target, version, headers, head)


def rebuild_head(req: Request, *, keep_upgrade: bool) -> bytes:
    """Same request line and headers, with Connection: close unless this is an upgrade."""
    out = [f"{req.method} {req.target} {req.version}"]
    for key, value in req.headers:
        if key.lower() in ("connection", "keep-alive", "proxy-connection") and not keep_upgrade:
            continue
        out.append(f"{key}: {value}")
    if not keep_upgrade:
        out.append("Connection: close")
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1")


def refusal(message: str) -> bytes:
    body = json.dumps({"message": message}).encode()
    return (b"HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)


class Record:
    """Append-only JSONL, one file per UTC day. A line is written and flushed per request."""

    def __init__(self, folder: Path, prefix: str = "docker") -> None:
        self.folder, self.prefix = folder, prefix
        self.folder.mkdir(parents=True, exist_ok=True)

    def path_for(self, when: datetime) -> Path:
        return self.folder / f"{self.prefix}-{when:%Y-%m-%d}.jsonl"

    def write(self, entry: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        line = json.dumps({"ts": now.isoformat(), **entry}, sort_keys=True) + "\n"
        with self.path_for(now).open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


async def docker_get_json(socket_path: str, path: str, timeout: float = 5.0) -> Optional[dict[str, Any]]:
    """GET a JSON object from the real daemon (HTTP/1.0, so the body is plain and the socket closes)."""
    reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(socket_path), timeout)
    try:
        writer.write(f"GET {path} HTTP/1.0\r\nHost: docker\r\n\r\n".encode())
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout)
    finally:
        writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    status = head.split(b" ", 2)[1:2]
    if status != [b"200"]:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


@dataclass
class Door:
    socket_path: str
    record: Record
    allowed_callers: tuple[str, ...]  # DNS names; empty = any caller (tests only)
    _caller_ips: set[str] = field(default_factory=set)
    _caller_ips_at: float = 0.0
    get_json: Callable[[str, str], Awaitable[Optional[dict[str, Any]]]] = docker_get_json

    async def caller_allowed(self, ip: str) -> bool:
        if not self.allowed_callers:
            return True
        if time.monotonic() - self._caller_ips_at > 30 or ip not in self._caller_ips:
            loop = asyncio.get_running_loop()
            ips: set[str] = set()
            for name in self.allowed_callers:
                try:
                    for info in await loop.getaddrinfo(name, None, family=socket.AF_INET):
                        ips.add(info[4][0])
                except OSError:
                    continue
            self._caller_ips, self._caller_ips_at = ips, time.monotonic()
        return ip in self._caller_ips

    async def target_reach(self, container_ref: str) -> tuple[str, Optional[str]]:
        """(container name, reason it leads to the host or None). Unknown container -> refuse."""
        data = await self.get_json(self.socket_path, f"/containers/{container_ref}/json")
        if data is None:
            return container_ref, "inspect_failed"
        return str(data.get("Name") or container_ref).lstrip("/"), host_reach(data)

    async def decide(self, route: Route, body: bytes) -> tuple[bool, str, dict[str, Any]]:
        """(allow, reason, extra record fields)."""
        if route.kind == "look":
            return True, "look", {}
        if route.kind == "exec_create":
            name, why = await self.target_reach(route.ref)
            extra: dict[str, Any] = {"container": name}
            try:
                spec = json.loads(body or b"{}")
                extra["exec_cmd"] = spec.get("Cmd")
                extra["exec_user"] = spec.get("User") or ""
            except ValueError:
                extra["exec_cmd"] = None
            return (why is None), (why or "exec"), extra
        if route.kind == "exec_run":
            info = await self.get_json(self.socket_path, f"/exec/{route.ref}/json")
            if info is None:
                return False, "exec_inspect_failed", {}
            name, why = await self.target_reach(str(info.get("ContainerID") or ""))
            return (why is None), (why or "exec"), {"container": name}
        return False, "change", {}

    async def handle(self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        peer = client_w.get_extra_info("peername") or ("?", 0)
        entry: dict[str, Any] = {"caller_ip": peer[0]}
        try:
            head = await client_r.readuntil(b"\r\n\r\n")
            if len(head) > MAX_HEADER_BYTES:
                raise ValueError("header too large")
            req = parse_head(head)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            client_w.close()
            return
        path = normalize_path(req.target)
        entry.update(method=req.method, path=path)
        if not await self.caller_allowed(peer[0]):
            self.record.write({**entry, "decision": "refuse", "reason": "caller"})
            client_w.write(refusal(REFUSE_CALLER))
            await client_w.drain()
            client_w.close()
            return
        route = classify(req.method, path)
        body = b""
        if route.kind == "exec_create":
            length = int(req.header("content-length") or 0)
            if length > MAX_EXEC_BODY_BYTES:
                length = 0
            if length:
                body = await client_r.readexactly(length)
        try:
            allow, reason, extra = await self.decide(route, body)
        except (OSError, asyncio.TimeoutError) as exc:
            allow, reason, extra = False, f"daemon_error:{type(exc).__name__}", {}
        entry.update(extra)
        self.record.write({**entry, "decision": "allow" if allow else "refuse", "reason": reason})
        if not allow:
            message = REFUSE_HOST if route.kind in ("exec_create", "exec_run") else REFUSE_CHANGE
            if reason.startswith("daemon_error") or reason.endswith("_failed"):
                message = f"Orion's Docker door: could not check the target ({reason})."
            client_w.write(refusal(message))
            await client_w.drain()
            client_w.close()
            return
        keep_upgrade = bool(req.header("upgrade")) and route.kind == "exec_run"
        try:
            up_r, up_w = await asyncio.open_unix_connection(self.socket_path)
        except OSError:
            client_w.close()
            return
        up_w.write(rebuild_head(req, keep_upgrade=keep_upgrade) + body)
        await up_w.drain()
        await asyncio.gather(_pipe(client_r, up_w), _pipe(up_r, client_w))


async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await src.read(65536)
            if not chunk:
                break
            dst.write(chunk)
            await dst.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            if dst.can_write_eof():
                dst.write_eof()
            else:
                dst.close()
        except (OSError, RuntimeError):
            dst.close()


async def serve(door: Door, host: str, port: int) -> None:
    server = await asyncio.start_server(door.handle, host, port, limit=MAX_HEADER_BYTES)
    async with server:
        await server.serve_forever()


def main() -> None:
    callers = tuple(c.strip() for c in os.environ.get("DOCKER_DOOR_ALLOWED_CALLERS", "").split(",") if c.strip())
    if not callers:
        raise SystemExit("DOCKER_DOOR_ALLOWED_CALLERS is empty: refusing to serve every container on the network")
    door = Door(
        socket_path=os.environ.get("DOCKER_DOOR_SOCKET", "/var/run/docker.sock"),
        record=Record(Path(os.environ.get("DOCKER_DOOR_RECORD_DIR", "/record"))),
        allowed_callers=callers,
    )
    asyncio.run(serve(door, "0.0.0.0", int(os.environ.get("DOCKER_DOOR_PORT", "2375"))))


if __name__ == "__main__":
    main()
