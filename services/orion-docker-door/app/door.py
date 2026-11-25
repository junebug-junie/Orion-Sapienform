"""Orion's Docker door: a filtering proxy in front of the real Docker socket.

Orion's turns may look (every GET/HEAD) and may ``docker exec`` into an organ, but may not change
containers (create, start, stop, restart, remove, build, pull, ...), and may not exec into a
container that leads back to the host (see ``host_reach``). Every request is appended to a JSONL
record on the host. Design: docs/superpowers/specs/2026-11-24-orion-safe-autonomy-in-place-design.md.

Stdlib only. The safety rule, from the 2026-11-24 code review: parse strictly, read the whole
request before deciding, one request per connection, always close, and refuse anything the door
cannot fully read.

- The head must use CRLF only (no bare LF, no stray CR, no folded lines), HTTP/1.0 or 1.1, a plain
  upper-case method and token header names. Docker's own parser accepts a bare LF, so a lenient
  parse here would let one checked request hide a second, unchecked one.
- The body is read in full before forwarding (``Content-Length`` only, at most 1 MB; no
  ``Transfer-Encoding``, no ``Expect``). The door forwards exactly head + body, always with
  ``Connection: close``. Nothing else the client sends reaches the daemon, except after a
  non-detached exec start, where Docker hijacks the connection and later bytes are that exec's stdin.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import posixpath
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote, unquote

logger = logging.getLogger("orion.docker_door")

MAX_HEADER_BYTES = 64 * 1024
MAX_BODY_BYTES = 1024 * 1024
HEAD_TIMEOUT_SEC = 10.0
BODY_TIMEOUT_SEC = 30.0
DAEMON_TIMEOUT_SEC = 10.0
ALLOWED_VERSIONS = frozenset({"HTTP/1.0", "HTTP/1.1"})
# Exact sources that are the host itself. Docker's own data root (DockerRootDir, read from the
# daemon) is handled separately: everything under it except named volumes' data.
HOST_ROOT_SOURCES = frozenset({"/", "/run", "/var/run"})
# Any bind of these, or of anything under them, reaches the host.
HOST_PATH_PREFIXES = ("/proc", "/sys", "/dev", "/etc", "/root")
# A writable bind under these lets an exec plant code or state the host runs or trusts (the
# primary checkout's orion/ package is imported by host scripts after every git pull).
HOST_WRITE_PREFIXES = ("/mnt/scripts", "/home", "/usr", "/boot", "/opt", "/var", "/lib", "/lib64",
                       "/bin", "/sbin", "/srv", "/mnt/storage-warm/orion-record")
# Plain data folders under /run that organs share read-only; any other bind under /run or
# /var/run may be a socket by another name (D-Bus, containerd's folder) and is refused.
RUN_ALLOWED = ("/run/orion-audio", "/run/orion-sensors")
DEFAULT_DOCKER_ROOT = "/var/lib/docker"
NAMESPACE_KEYS = ("NetworkMode", "PidMode", "IpcMode")
RUNTIME_SOCKET_NAMES = frozenset({"docker.sock", "containerd.sock"})
REFUSE_CHANGE = "Orion's Docker door: changing containers is off (Juniper, 2026-11-24)."
REFUSE_HOST = "Orion's Docker door: exec into a container that leads back to the host is off."
REFUSE_COPY = "Orion's Docker door: copying files out of a container that leads back to the host is off."
REFUSE_CALLER = "Orion's Docker door: this caller is not allowed."
REFUSE_RECORD = "Orion's Docker door: could not write the record, so nothing passes."

_METHOD = re.compile(r"^[A-Z]{1,16}$")
_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_TARGET = re.compile(r"^/[\x21-\x7e]*$")
_LENGTH = re.compile(r"^[0-9]{1,9}$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_VERSION_PREFIX = re.compile(r"^/v\d+(?:\.\d+)*(/.*)$")
_EXEC_CREATE = re.compile(r"^/containers/([^/]+)/exec$")
_EXEC_RUN = re.compile(r"^/exec/([^/]+)/(start|resize)$")
# Reads a container's files, which includes its binds: host files. Checked like an exec.
_COPY_OUT = re.compile(r"^/containers/([^/]+)/(archive|export)$")
_IMAGE_SAVE = re.compile(r"^/images/(?:[^/]+/)*get$")


class BadRequest(ValueError):
    """The door cannot read this request exactly, so it refuses it."""


def normalize_path(target: str) -> str:
    """Request target -> decoded API path, without the query string and the /vX.YZ prefix.

    Docker decodes the path once before routing, so the door decodes once too. A path that still
    holds ``%`` after that (double encoding), a control byte, an empty segment or a ``.``/``..``
    segment is refused: Docker's CLI never sends one.
    """
    raw = target.split("?", 1)[0]
    try:
        path = unquote(raw, errors="strict")
    except UnicodeDecodeError as exc:
        raise BadRequest("path_encoding") from exc
    if "%" in path or any(ord(ch) < 0x21 or ord(ch) == 0x7F for ch in path):
        raise BadRequest("path_encoding")
    if path != "/" and any(seg in ("", ".", "..") for seg in path.split("/")[1:]):
        raise BadRequest("path_segments")
    m = _VERSION_PREFIX.match(path)
    return m.group(1) if m else path


@dataclass(frozen=True)
class Route:
    """What a request asks for, before any container lookup."""

    kind: str  # "look" | "copy_out" | "exec_create" | "exec_run" | "change"
    ref: str = ""  # container ref (copy_out, exec_create) or exec id (exec_run)
    action: str = ""  # exec_run only: "start" | "resize"


def classify(method: str, path: str) -> Route:
    if "attach" in path.split("/"):  # attach (incl. GET .../attach/ws) writes to a main process
        return Route("change")
    if method in ("GET", "HEAD"):
        m = _COPY_OUT.match(path)
        if m:
            return Route("copy_out", m.group(1))
        if _IMAGE_SAVE.match(path):  # docker save: whole image filesystems
            return Route("change")
        return Route("look")
    if method == "POST":
        m = _EXEC_CREATE.match(path)
        if m:
            return Route("exec_create", m.group(1))
        m = _EXEC_RUN.match(path)
        if m:
            return Route("exec_run", m.group(1), m.group(2))
    return Route("change")


def _runtime_socket(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    if name in RUNTIME_SOCKET_NAMES:
        return True
    return name.endswith(".sock") and any(path.startswith(p + "/") for p in ("/run", "/var/run"))


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def host_reach(inspect: dict[str, Any], docker_root: str = DEFAULT_DOCKER_ROOT) -> Optional[str]:
    """Why this container leads back to the host, or None. Input: GET /containers/{id}/json.

    ``container:<x>`` namespace modes are checked by ``Door.target_reach`` (it inspects x).
    """
    host = inspect.get("HostConfig") or {}
    if host.get("Privileged"):
        return "privileged"
    for key, label in (("PidMode", "pid_host"), ("NetworkMode", "network_host"),
                       ("IpcMode", "ipc_host"), ("UsernsMode", "userns_host"),
                       ("UTSMode", "uts_host"), ("CgroupnsMode", "cgroupns_host")):
        if str(host.get(key) or "") == "host":
            return label
    if host.get("CapAdd"):
        return "cap_add:" + ",".join(str(c) for c in host["CapAdd"])
    if host.get("Devices"):
        return "devices"
    if host.get("DeviceCgroupRules"):
        return "device_cgroup_rules"
    for opt in host.get("SecurityOpt") or []:
        if "unconfined" in str(opt):
            return f"security_opt:{opt}"
    root = posixpath.normpath(docker_root or DEFAULT_DOCKER_ROOT)
    for mount in inspect.get("Mounts") or []:
        src = posixpath.normpath(str(mount["Source"])) if mount.get("Source") else ""
        dst = posixpath.normpath(str(mount.get("Destination") or "/"))
        if dst.rsplit("/", 1)[-1] in RUNTIME_SOCKET_NAMES:
            return f"socket:{src or dst}"
        if not src:
            continue
        if src in HOST_ROOT_SOURCES:
            return f"mount:{src}"
        if any(_under(src, p) for p in HOST_PATH_PREFIXES):
            return f"mount:{src}"
        if _under(src, root) and not _under(src, root + "/volumes"):
            return f"docker_root:{src}"
        if _runtime_socket(src):
            return f"socket:{src}"
        if any(_under(src, p) for p in ("/run", "/var/run")) and not any(_under(src, p) for p in RUN_ALLOWED):
            return f"run:{src}"
        if (mount.get("RW") and not _under(src, root + "/volumes")
                and any(_under(src, p) for p in HOST_WRITE_PREFIXES)):
            return f"writable:{src}"
    return None


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: list[tuple[str, str]]

    def values(self, name: str) -> list[str]:
        name = name.lower()
        return [value for key, value in self.headers if key.lower() == name]

    def header(self, name: str) -> str:
        found = self.values(name)
        return found[0] if found else ""


def parse_head(head: bytes) -> Request:
    """Strict parse of a head that ends with CRLF CRLF. Raises BadRequest on anything loose."""
    if not head.endswith(b"\r\n\r\n"):
        raise BadRequest("head_end")
    lines = head[:-4].split(b"\r\n")
    for line in lines:
        if any(b < 0x20 and b != 0x09 or b == 0x7F for b in line):  # bare LF, stray CR, NUL, ...
            raise BadRequest("control_byte")
    parts = lines[0].decode("latin-1").split(" ")
    if len(parts) != 3:
        raise BadRequest("request_line")
    method, target, version = parts
    if not _METHOD.match(method) or not _TARGET.match(target) or version not in ALLOWED_VERSIONS:
        raise BadRequest("request_line")
    headers: list[tuple[str, str]] = []
    for raw in lines[1:]:
        line = raw.decode("latin-1")
        if not line or line[0] in " \t":
            raise BadRequest("folded_header")
        key, sep, value = line.partition(":")
        if not sep or not _TOKEN.match(key):
            raise BadRequest("header_name")
        headers.append((key, value.strip(" \t")))
    return Request(method, target, version, headers)


def body_length(req: Request) -> Optional[int]:
    """The body length the door will read, or None when there is no Content-Length."""
    if req.values("transfer-encoding"):
        raise BadRequest("transfer_encoding")
    if req.values("expect"):
        raise BadRequest("expect")
    found = req.values("content-length")
    if not found:
        return None
    if len(found) > 1 or not _LENGTH.match(found[0]):
        raise BadRequest("content_length")
    length = int(found[0])
    if length > MAX_BODY_BYTES:
        raise BadRequest("body_too_large")
    return length


def rebuild_head(req: Request) -> bytes:
    """Same request line and headers, always with Connection: close.

    ``Upgrade`` is kept: Docker decides on a hijack from the exec start body and the ``Upgrade``
    header alone, and a hijacked stream no longer reads HTTP, so ``close`` never breaks it.
    """
    out = [f"{req.method} {req.target} {req.version}"]
    for key, value in req.headers:
        if key.lower() in ("connection", "keep-alive", "proxy-connection"):
            continue
        out.append(f"{key}: {value}")
    out.append("Connection: close")
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1")


def refusal(message: str, status: str = "403 Forbidden") -> bytes:
    body = json.dumps({"message": message}).encode()
    return (f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n".encode()
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


def _exec_spec(body: bytes) -> Optional[dict[str, Any]]:
    """The exec body with keys lower-cased, as Go's JSON decoder matches them.

    Go matches field names without case and the last match wins, so ``"privileged": true`` is
    privileged to Docker. Two keys that differ only in case are refused as unreadable.
    """
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            low = key.lower()
            if low in out:
                raise ValueError(f"duplicate key {key!r}")
            out[low] = value
        return out

    try:
        spec = json.loads(body, object_pairs_hook=pairs)
    except (ValueError, RecursionError):
        return None
    return spec if isinstance(spec, dict) else None


def _streams_stdin(route: Route, body: bytes) -> bool:
    """Only a non-detached exec start hijacks the connection; its later bytes are the exec's stdin."""
    if route.kind != "exec_run" or route.action != "start":
        return False
    spec = _exec_spec(body) if body else {}
    return not (spec or {}).get("detach")


@dataclass
class Door:
    socket_path: str
    record: Record
    allowed_callers: tuple[str, ...]  # DNS names; empty = any caller (tests only)
    _caller_ips: set[str] = field(default_factory=set)
    _caller_ips_at: float = 0.0
    get_json: Callable[[str, str], Awaitable[Optional[dict[str, Any]]]] = docker_get_json
    head_timeout: float = HEAD_TIMEOUT_SEC
    body_timeout: float = BODY_TIMEOUT_SEC
    _docker_root: Optional[str] = None

    async def caller_allowed(self, ip: str) -> bool:
        if not self.allowed_callers:
            return True
        if time.monotonic() - self._caller_ips_at > 30 or ip not in self._caller_ips:
            loop = asyncio.get_running_loop()
            ips: set[str] = set()
            for name in self.allowed_callers:
                try:
                    infos = await asyncio.wait_for(loop.getaddrinfo(name, None, family=socket.AF_INET), 5)
                except (OSError, asyncio.TimeoutError):
                    continue
                ips.update(info[4][0] for info in infos)
            self._caller_ips, self._caller_ips_at = ips, time.monotonic()
        return ip in self._caller_ips

    async def docker_root(self) -> Optional[str]:
        """The daemon's DockerRootDir (here /mnt/docker, not /var/lib/docker), read once."""
        if self._docker_root is None:
            info = await self.get_json(self.socket_path, "/info")
            root = str((info or {}).get("DockerRootDir") or "")
            if not root:
                return None
            self._docker_root = root
        return self._docker_root

    async def target_reach(self, container_ref: str, depth: int = 3) -> tuple[str, Optional[str]]:
        """(container name, reason it leads to the host or None). Unknown container -> refuse."""
        if not _REF.match(container_ref):
            return container_ref, "bad_ref"
        data = await self.get_json(self.socket_path, f"/containers/{quote(container_ref)}/json")
        if data is None:
            return container_ref, "inspect_failed"
        name = str(data.get("Name") or container_ref).lstrip("/")
        root = await self.docker_root()
        if root is None:
            return name, "info_failed"
        mounts = []
        for mount in data.get("Mounts") or []:
            # A local volume made with o=bind,device=<host path> is a bind by another name.
            if mount.get("Type") == "volume" and mount.get("Name"):
                vol = await self.get_json(self.socket_path, f"/volumes/{quote(str(mount['Name']))}")
                if vol is None:
                    return name, "volume_inspect_failed"
                opts = vol.get("Options") or {}
                if "bind" in str(opts.get("o") or "").split(",") and opts.get("device"):
                    mount = {**mount, "Source": str(opts["device"])}
            mounts.append(mount)
        why = host_reach({**data, "Mounts": mounts}, root)
        if why is not None:
            return name, why
        host = data.get("HostConfig") or {}
        for key in NAMESPACE_KEYS:
            mode = str(host.get(key) or "")
            if mode.startswith("container:"):
                if depth <= 0:
                    return name, f"shares_{key}:too_deep"
                _other, other_why = await self.target_reach(mode.split(":", 1)[1], depth - 1)
                if other_why is not None:
                    return name, f"shares_{key}:{other_why}"
        return name, None

    async def decide(self, route: Route, body: bytes, has_length: bool) -> tuple[bool, str, dict[str, Any]]:
        """(allow, reason, extra record fields)."""
        if route.kind == "look":
            return True, "look", {}
        if route.kind == "copy_out":
            name, why = await self.target_reach(route.ref)
            return (why is None), (why or "copy_out"), {"container": name}
        if route.kind == "exec_create":
            spec = _exec_spec(body) if has_length else None
            if spec is None:
                return False, "exec_body_unreadable", {}
            extra: dict[str, Any] = {"exec_cmd": spec.get("cmd"), "exec_user": spec.get("user") or ""}
            if spec.get("privileged"):
                return False, "exec_privileged", extra
            name, why = await self.target_reach(route.ref)
            extra["container"] = name
            return (why is None), (why or "exec"), extra
        if route.kind == "exec_run":
            if not _REF.match(route.ref):
                return False, "bad_ref", {}
            info = await self.get_json(self.socket_path, f"/exec/{quote(route.ref)}/json")
            if info is None:
                return False, "exec_inspect_failed", {}
            if (info.get("ProcessConfig") or {}).get("privileged"):
                return False, "exec_privileged", {}
            container = str(info.get("ContainerID") or "")
            if not container:
                return False, "exec_inspect_failed", {}
            name, why = await self.target_reach(container)
            return (why is None), (why or "exec"), {"container": name}
        return False, "change", {}

    async def _write_record(self, entry: dict[str, Any]) -> bool:
        try:
            await asyncio.to_thread(self.record.write, entry)  # fsync stays off the event loop
            return True
        except (OSError, ValueError, TypeError) as exc:
            logger.error("DOCKER_DOOR record_write_failed error=%r entry=%r", exc, entry)
            return False

    async def _send(self, client_w: asyncio.StreamWriter, data: bytes) -> None:
        client_w.write(data)
        with contextlib.suppress(OSError, asyncio.TimeoutError):
            await asyncio.wait_for(client_w.drain(), self.head_timeout)

    async def _refuse(self, client_w: asyncio.StreamWriter, entry: dict[str, Any], message: str,
                      status: str = "403 Forbidden") -> None:
        await self._write_record({**entry, "decision": "refuse"})
        await self._send(client_w, refusal(message, status))

    async def handle(self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        writers = [client_w]
        try:
            await self._handle(client_r, client_w, writers)
        except Exception:  # noqa: BLE001 -- one bad connection must never stop the door
            logger.exception("DOCKER_DOOR handler_error")
        finally:
            for writer in writers:
                with contextlib.suppress(Exception):
                    writer.close()

    async def _handle(self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter,
                      writers: list[asyncio.StreamWriter]) -> None:
        peer = client_w.get_extra_info("peername") or ("?", 0)
        entry: dict[str, Any] = {"caller_ip": str(peer[0])}
        if not await self.caller_allowed(str(peer[0])):
            await self._refuse(client_w, {**entry, "reason": "caller"}, REFUSE_CALLER)
            return
        try:
            head = await asyncio.wait_for(client_r.readuntil(b"\r\n\r\n"), self.head_timeout)
            if len(head) > MAX_HEADER_BYTES:
                raise BadRequest("head_too_large")
            req = parse_head(head)
            entry.update(method=req.method, path=req.target)
            path = normalize_path(req.target)
            entry["path"] = path
            length = body_length(req)
            body = b""
            if length:
                body = await asyncio.wait_for(client_r.readexactly(length), self.body_timeout)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
            return  # client gone or too slow; nothing was forwarded (a door probe lands here too)
        except asyncio.LimitOverrunError:
            await self._refuse(client_w, {**entry, "reason": "bad_request:head_too_large"},
                               "Orion's Docker door: request head too large.", "400 Bad Request")
            return
        except BadRequest as exc:
            await self._refuse(client_w, {**entry, "reason": f"bad_request:{exc}"},
                               f"Orion's Docker door: cannot read this request exactly ({exc}).",
                               "400 Bad Request")
            return
        route = classify(req.method, path)
        try:
            allow, reason, extra = await asyncio.wait_for(
                self.decide(route, body, length is not None), DAEMON_TIMEOUT_SEC)
        except (OSError, asyncio.TimeoutError) as exc:
            allow, reason, extra = False, f"daemon_error:{type(exc).__name__}", {}
        entry.update(extra)
        entry["reason"] = reason
        if not allow:
            message = {"copy_out": REFUSE_COPY, "exec_create": REFUSE_HOST, "exec_run": REFUSE_HOST}.get(
                route.kind, REFUSE_CHANGE)
            if reason.startswith("daemon_error") or reason.endswith(("_failed", "_unreadable")) or reason == "bad_ref":
                message = f"Orion's Docker door: could not check the target ({reason})."
            elif reason == "exec_privileged":
                message = "Orion's Docker door: a privileged exec is off."
            await self._refuse(client_w, entry, message)
            return
        if not await self._write_record({**entry, "decision": "allow"}):
            await self._send(client_w, refusal(REFUSE_RECORD, "503 Service Unavailable"))
            return
        try:
            up_r, up_w = await asyncio.wait_for(asyncio.open_unix_connection(self.socket_path),
                                                DAEMON_TIMEOUT_SEC)
        except (OSError, asyncio.TimeoutError):
            await self._send(client_w, refusal("Orion's Docker door: the daemon did not answer.",
                                               "502 Bad Gateway"))
            return
        writers.append(up_w)
        up_w.write(rebuild_head(req) + body)
        await up_w.drain()
        down = asyncio.ensure_future(_pipe(up_r, client_w))
        if _streams_stdin(route, body):
            # The exec's output ending ends the session, even if stdin stays open.
            side = asyncio.ensure_future(_pipe(client_r, up_w))
            waits = {down}
        else:
            # Nothing more goes upstream; watch for the client leaving, as dockerd itself does,
            # so an abandoned `logs -f` or `events` does not hold the daemon connection open.
            side = asyncio.ensure_future(_drain_until_eof(client_r))
            waits = {down, side}
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (down, side):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


async def _drain_until_eof(src: asyncio.StreamReader) -> None:
    """Read and drop whatever the client still sends; return when it closes."""
    with contextlib.suppress(ConnectionError, OSError):
        while await src.read(65536):
            pass


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
        with contextlib.suppress(OSError, RuntimeError):
            if dst.can_write_eof():
                dst.write_eof()


async def serve(door: Door, host: str, port: int) -> None:
    server = await asyncio.start_server(door.handle, host, port, limit=MAX_HEADER_BYTES)
    async with server:
        await server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
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
