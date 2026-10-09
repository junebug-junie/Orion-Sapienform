"""The recording sandbox: every tool the replayed model can call, with no path to a production write.

HOW WRITES ARE PREVENTED -- by construction, one lane per kind of state, each tested:

  * Shell (anything that is not redis-cli/psql/curl/docker): `docker exec` into a per-task
    container started with `--network none --read-only --cap-drop ALL`, the repo bind-mounted
    read-only, only tmpfs writable, no docker socket. A write has nothing to land on; a network
    call has no route. (``ShellSandbox``)
  * Graph (redis-cli GRAPH.*): never reaches production FalkorDB. Queries run against a scratch
    FalkorDB holding a copy of the graph taken once, read-only (DUMP), at replay start
    (``graph_scratch.ScratchGraphs``). Write queries land in the copy, so the model sees real
    results and real errors, and the replay can diff what actually changed.
  * SQL (psql): the production DB as `orion_readonly` (SELECT-only role, the same one the real
    turn gets), inside BEGIN READ ONLY with default_transaction_read_only=on. SQL that looks like
    a write is not sent at all: it is recorded as a stubbed write and answered with an error.
  * HTTP (curl, WebFetch): GET only. Any other method or a body (-X POST, -d, -F, --json, -T) is
    recorded as a stubbed write and never sent.
  * docker: ps / logs / inspect / images only; everything else (exec, run, compose, rm...) refused.

Every call -- executed, stubbed or refused -- is appended to ``ToolLog``; the write-claim check
reads the stubbed writes and the scratch-graph diff from there.
"""

from __future__ import annotations

import asyncio
import html
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
from urllib.parse import urlparse

from orion.evals.model_replay.command_split import EXTERNAL_TOOLS, Simple, Unsplittable, mentions_external, split_command

OUTPUT_CAP = 30_000
DEFAULT_SANDBOX_IMAGE = "orion-harness-governor-harness-governor:latest"
DOCKER_READ_VERBS = frozenset({"ps", "logs", "inspect", "images"})

# A Cypher clause that changes the graph. Matched after string literals are blanked.
_CYPHER_WRITE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b|\bCALL\s+db\.idx\.", re.I)
_SQL_WRITE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|UPSERT|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|COPY|VACUUM|REINDEX|"
    r"CLUSTER|REFRESH|LOCK|CALL|DO|COMMENT|SECURITY|LISTEN|NOTIFY|SET\s+ROLE|SET\s+SESSION)\b", re.I)
_STRING_LIT = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.)*\"")


def strip_literals(text: str) -> str:
    return _STRING_LIT.sub("''", text)


def is_cypher_write(query: str) -> bool:
    return bool(_CYPHER_WRITE.search(strip_literals(query)))


def is_sql_write(sql: str) -> bool:
    body = re.sub(r"--[^\n]*", " ", sql)
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
    return bool(_SQL_WRITE.search(strip_literals(body)))


@dataclass
class ToolEvent:
    t: float
    tool: str
    lane: str            # shell | graph | sql | http | docker | file | fetch | search
    action: str          # executed | write_landed | write_stubbed | write_denied | refused | error
    detail: dict[str, Any]
    output_chars: int = 0


@dataclass
class ToolLog:
    events: list[ToolEvent] = field(default_factory=list)

    def add(self, tool: str, lane: str, action: str, output: str = "", **detail: Any) -> None:
        self.events.append(ToolEvent(time.time(), tool, lane, action, detail, len(output or "")))

    def writes(self) -> list[ToolEvent]:
        return [e for e in self.events if e.action.startswith("write_")]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [e.__dict__ for e in self.events]


# --- lanes ------------------------------------------------------------------------------------


class GraphLane(Protocol):
    def query(self, graph: str, verb: str, cypher: str) -> tuple[str, bool]:
        """(redis-cli-like output, ok)."""

    def list_graphs(self) -> list[str]: ...


class SqlLane(Protocol):
    def run(self, sql: str, flags: list[str]) -> tuple[str, int]: ...


class HttpLane(Protocol):
    def get(self, url: str) -> tuple[str, int]: ...


class DockerReadLane(Protocol):
    def run(self, argv: list[str]) -> tuple[str, int]: ...


class ReadOnlyPsql:
    """psql as orion_readonly in the DB container, READ ONLY transaction, no secrets on argv
    (local socket; same trust the extraction script relies on)."""

    def __init__(self, container: str = "orion-athena-sql-db", role: str = "orion_readonly",
                 db: str = "conjourney", timeout_sec: float = 120.0) -> None:
        self.container, self.role, self.db, self.timeout_sec = container, role, db, timeout_sec

    def run(self, sql: str, flags: list[str]) -> tuple[str, int]:
        allowed = [f for f in flags if re.fullmatch(r"-(?:[AtqxH]+|-csv|-html)|-F.|-P.*", f)]
        script = f"BEGIN READ ONLY;\n{sql.rstrip().rstrip(';')};\nROLLBACK;\n"
        try:
            p = subprocess.run(
                ["docker", "exec", "-i", "-e", "PGOPTIONS=-c default_transaction_read_only=on", self.container,
                 "psql", "-U", self.role, "-d", self.db, "-X", "-v", "ON_ERROR_STOP=1", *allowed, "-f", "-"],
                input=script, capture_output=True, text=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired:
            return "psql: canceling statement due to statement timeout\n", 1
        out = "\n".join(line for line in (p.stdout + p.stderr).splitlines() if line not in ("BEGIN", "ROLLBACK"))
        return out + "\n", p.returncode


class HostHttpGet:
    """GET only. `host.docker.internal` (what the prompts name) maps to this host."""

    def __init__(self, host_map: Optional[dict[str, str]] = None, timeout_sec: float = 60.0) -> None:
        self.host_map = {"host.docker.internal": "127.0.0.1", **(host_map or {})}
        self.timeout_sec = timeout_sec

    def get(self, url: str) -> tuple[str, int]:
        import httpx

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"curl: (1) Protocol \"{parsed.scheme}\" not supported\n", 1
        host = parsed.hostname or ""
        if host in self.host_map:
            netloc = self.host_map[host] + (f":{parsed.port}" if parsed.port else "")
            url = parsed._replace(netloc=netloc).geturl()
        try:
            with httpx.Client(timeout=self.timeout_sec, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0 (orion-model-replay)"}) as c:
                r = c.get(url)
            return r.text, 0 if r.status_code < 400 else 22
        except Exception as exc:  # noqa: BLE001
            return f"curl: (7) {type(exc).__name__}: {exc}\n", 7


class DockerReadOnly:
    def run(self, argv: list[str]) -> tuple[str, int]:
        verb = next((a for a in argv[1:] if not a.startswith("-")), "")
        if verb not in DOCKER_READ_VERBS or (verb == "logs" and "-f" in argv) or "--follow" in argv:
            return f"docker: '{verb}' is not available in this environment (read-only: ps, logs, inspect, images)\n", 1
        try:
            p = subprocess.run(["docker", *argv[1:]], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "docker: timed out\n", 1
        return p.stdout + p.stderr, p.returncode


class ShellSandbox:
    """One container per (task, model): no network, read-only root, repo read-only, tmpfs scratch."""

    def __init__(self, *, repo: Path, image: str = DEFAULT_SANDBOX_IMAGE, name: str,
                 env: Optional[dict[str, str]] = None, docker: str = "docker") -> None:
        self.repo, self.image, self.name, self.env, self.docker = repo, image, name, dict(env or {}), docker
        self.started = False

    def run_argv(self) -> list[str]:
        argv = [self.docker, "run", "-d", "--rm", "--name", self.name, "--label", "orion.model_replay=1",
                "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "256", "--memory", "2g",
                "--tmpfs", "/tmp:rw,size=512m", "--tmpfs", "/scratch:rw,size=512m", "--tmpfs", "/root:rw,size=64m",
                "-v", f"{self.repo}:/repo:ro", "-v", f"{self.repo}:{self.repo}:ro", "-w", "/repo",
                "-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory", "-e", "GIT_CONFIG_VALUE_0=*",
                "-e", "HOME=/root"]
        for k, v in sorted(self.env.items()):
            argv += ["-e", f"{k}={v}"]
        return argv + ["--entrypoint", "sleep", self.image, "infinity"]

    def start(self) -> None:
        subprocess.run(self.run_argv(), check=True, capture_output=True, text=True, timeout=120)
        self.started = True

    def stop(self) -> None:
        if self.started:
            subprocess.run([self.docker, "rm", "-f", self.name], capture_output=True, text=True, timeout=60)
            self.started = False

    def run(self, command: str, stdin: Optional[str] = None, timeout_sec: float = 300.0) -> tuple[str, int]:
        try:
            p = subprocess.run([self.docker, "exec", "-i", self.name, "bash", "-c", command],
                               input=stdin or "", capture_output=True, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout_sec:.0f}s\n", 124
        return p.stdout + p.stderr, p.returncode


# --- the tool surface -------------------------------------------------------------------------


@dataclass
class FetchCache:
    """One live fetch per URL per replay, shared by both models (so they read the same page)."""

    http: HttpLane
    pages: dict[str, tuple[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, url: str) -> tuple[str, int]:
        with self._lock:  # both models run in threads; the second waits for the first fetch
            if url not in self.pages:
                self.pages[url] = self.http.get(url)
            return self.pages[url]


def html_to_text(body: str) -> str:
    body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", body)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n\n", html.unescape(body))).strip()


@dataclass
class Toolbox:
    """Dispatch a tool_use block to its lane and record the outcome."""

    log: ToolLog
    shell: Any
    graph: Optional[GraphLane]
    sql: SqlLane
    http: HttpLane
    docker_ro: DockerReadLane
    fetch_cache: FetchCache
    allowed: tuple[str, ...]
    fetch_mode: str = "live"
    fetch_fail_error: str = ""
    sandbox_root: str = "/repo"

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """(text result, is_error)."""
        if name not in self.allowed:
            self.log.add(name, "refused", "refused", reason="tool_not_available")
            return f"Error: No such tool available: {name}", True
        try:
            if name == "Bash":
                return self.bash(str(args.get("command") or ""))
            if name == "Read":
                return self._shell_tool("Read", self._read_cmd(args))
            if name == "Grep":
                return self._shell_tool("Grep", self._grep_cmd(args))
            if name == "Glob":
                return self._shell_tool("Glob", self._glob_cmd(args))
            if name == "WebFetch":
                return self.web_fetch(str(args.get("url") or ""))
            if name == "WebSearch":
                self.log.add("WebSearch", "search", "refused", query=str(args.get("query") or "")[:300])
                return "WebSearch is unavailable: no search provider is configured.", True
        except Exception as exc:  # noqa: BLE001 -- a tool error is a tool result, not a crash
            self.log.add(name, "error", "error", error=f"{type(exc).__name__}: {exc}"[:500])
            return f"Error: {type(exc).__name__}: {exc}", True
        return f"Error: unhandled tool {name}", True

    # file tools run in the sandbox too, so they see exactly what Bash sees
    def _read_cmd(self, args: dict[str, Any]) -> str:
        path = shlex.quote(str(args.get("file_path") or args.get("path") or ""))
        offset = max(1, int(args.get("offset") or 1))
        limit = max(1, min(int(args.get("limit") or 2000), 2000))
        return f"cat -n {path} | sed -n '{offset},{offset + limit - 1}p'"

    def _grep_cmd(self, args: dict[str, Any]) -> str:
        pattern = shlex.quote(str(args.get("pattern") or ""))
        path = shlex.quote(str(args.get("path") or "."))
        flags = "-rn" if args.get("output_mode") == "content" else "-rl"
        if args.get("-i"):
            flags += "i"
        glob = f" --include={shlex.quote(str(args['glob']))}" if args.get("glob") else ""
        return f"grep -E {flags}{glob} -- {pattern} {path} | head -n {int(args.get('head_limit') or 250)}"

    def _glob_cmd(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern") or "*")
        path = shlex.quote(str(args.get("path") or "."))
        return (f"cd {path} && python3 -c 'import glob,sys; print(\"\\n\".join(sorted(glob.glob(sys.argv[1], "
                f"recursive=True))[:500]))' {shlex.quote(pattern)}")

    def _shell_tool(self, tool: str, command: str) -> tuple[str, bool]:
        out, rc = self.shell.run(command)
        self.log.add(tool, "file", "executed", out, command=command[:500], rc=rc)
        return _cap(out), rc != 0

    def web_fetch(self, url: str) -> tuple[str, bool]:
        if self.fetch_mode == "fail":
            self.log.add("WebFetch", "fetch", "executed", self.fetch_fail_error, url=url, replayed_failure=True)
            return self.fetch_fail_error, True
        body, rc = self.fetch_cache.get(url)
        text = html_to_text(body)
        self.log.add("WebFetch", "fetch", "executed", text, url=url, rc=rc, ok=rc == 0 and bool(text.strip()))
        return _cap(text) if text.strip() else "Fetched page was empty.", rc != 0 or not text.strip()

    # --- Bash ---------------------------------------------------------------------------------
    def bash(self, command: str) -> tuple[str, bool]:
        if not command.strip():
            return "Error: empty command", True
        if not mentions_external(command):
            out, rc = self.shell.run(command)
            self.log.add("Bash", "shell", "executed", out, command=command[:2000], rc=rc)
            return _cap(out), rc != 0
        try:
            split = split_command(command)
        except Unsplittable as exc:
            self.log.add("Bash", "refused", "refused", command=command[:2000], reason=f"unsplittable:{exc}")
            return ("This environment cannot run redis-cli/psql/curl/docker inside a $(...), backticks, "
                    "a subshell or a background job. Run that call as its own command."), True
        return self._run_split(split.commands)

    def _run_split(self, cmds: list[Simple]) -> tuple[str, bool]:
        # Group runs of plain shell commands so `cd x && grep ...` keeps its meaning.
        groups: list[list[Simple]] = []
        for c in cmds:
            if c.tool not in EXTERNAL_TOOLS and groups and groups[-1][0].tool not in EXTERNAL_TOOLS:
                groups[-1].append(c)
            else:
                groups.append([c])
        transcript: list[str] = []
        stdin: Optional[str] = None
        rc = 0
        skip = False
        for g in groups:
            op = g[-1].op
            if skip:
                skip = False
                stdin = None
                continue
            if g[0].tool in EXTERNAL_TOOLS:
                out, rc = self._external(g[0], stdin if stdin is not None else g[0].stdin)
            else:
                text = " ".join(c.raw + (f" {c.op}" if c.op and c is not g[-1] else "") for c in g)
                out, rc = self.shell.run(text, stdin=stdin)
                self.log.add("Bash", "shell", "executed", out, command=text[:2000], rc=rc)
            if op == "|":
                stdin = out
                continue
            stdin = None
            transcript.append(out)
            if op == "&&" and rc != 0 or op == "||" and rc == 0:
                skip = True
        return _cap("".join(transcript)), rc != 0

    def _external(self, cmd: Simple, stdin: Optional[str]) -> tuple[str, int]:
        tool = cmd.tool
        argv = cmd.argv[[i for i, t in enumerate(cmd.argv) if t.rsplit("/", 1)[-1] == tool][0]:]
        if tool == "redis-cli":
            return self._redis(argv, stdin)
        if tool == "psql":
            return self._psql(argv, stdin)
        if tool == "curl":
            return self._curl(argv)
        verb = next((a for a in argv[1:] if not a.startswith("-")), "")
        out, rc = self.docker_ro.run(argv)
        self.log.add("Bash", "docker", "executed" if verb in DOCKER_READ_VERBS else "refused", out, argv=argv[:12])
        return out, rc

    def _redis(self, argv: list[str], stdin: Optional[str]) -> tuple[str, int]:
        idx = next((i for i, a in enumerate(argv) if a.upper().startswith("GRAPH.")), None)
        if idx is None:
            verb = next((a for a in argv[1:] if not a.startswith("-") and "://" not in a), "")
            if verb.upper() == "PING":
                return "PONG\n", 0
            self.log.add("Bash", "graph", "refused", argv=argv[:8])
            return f"(error) NOPERM this user has no permissions to run the '{verb.lower()}' command\n", 1
        verb = argv[idx].upper()
        if self.graph is None:
            self.log.add("Bash", "graph", "refused", verb=verb, reason="no_scratch_graph")
            return "Could not connect to Redis: Connection refused\n", 1
        if verb == "GRAPH.LIST":
            return "\n".join(self.graph.list_graphs()) + "\n", 0
        graph = argv[idx + 1] if len(argv) > idx + 1 else ""
        query = argv[idx + 2] if len(argv) > idx + 2 else (stdin or "")
        write = is_cypher_write(query)
        out, ok = self.graph.query(graph, verb, query)
        if write:
            action = "write_landed" if ok else ("write_denied" if "NOPERM" in out or "read-only" in out.lower()
                                                 else "write_failed")
            self.log.add("Bash", "graph", action, out, graph=graph, verb=verb, cypher=query[:4000])
        else:
            self.log.add("Bash", "graph", "executed", out, graph=graph, verb=verb, cypher=query[:2000], ok=ok)
        return out, 0 if ok else 1

    def _psql(self, argv: list[str], stdin: Optional[str]) -> tuple[str, int]:
        sqls: list[str] = []
        flags: list[str] = []
        i = 1
        while i < len(argv):
            a = argv[i]
            if a in ("-c", "--command") and i + 1 < len(argv):
                sqls.append(argv[i + 1])
                i += 2
                continue
            if a.startswith("--command="):
                sqls.append(a.split("=", 1)[1])
            elif a in ("-f", "--file"):
                self.log.add("Bash", "sql", "refused", reason="psql_file")
                return "psql: -f is not available here; pass the SQL with -c or a heredoc\n", 1
            elif a.startswith("-"):
                flags.append(a)
            i += 1
        sql = "\n".join(sqls) if sqls else (stdin or "")
        if not sql.strip():
            return "psql: no SQL given (interactive psql is not available)\n", 1
        if is_sql_write(sql):
            self.log.add("Bash", "sql", "write_stubbed", sql=sql[:4000])
            return "ERROR:  permission denied: this role is read-only (SELECT only)\n", 1
        out, rc = self.sql.run(sql, flags)
        self.log.add("Bash", "sql", "executed", out, sql=sql[:2000], rc=rc)
        return out, rc

    def _curl(self, argv: list[str]) -> tuple[str, int]:
        method = "GET"
        url = ""
        body = False
        i = 1
        while i < len(argv):
            a = argv[i]
            if a in ("-X", "--request") and i + 1 < len(argv):
                method = argv[i + 1].upper()
                i += 2
                continue
            if a in ("-d", "--data", "--data-raw", "--data-binary", "--data-urlencode", "-F", "--form",
                     "--json", "-T", "--upload-file") or a.startswith(("--data", "--json", "--form")):
                body = True
            elif a in ("-H", "--header", "-o", "--output", "-m", "--max-time", "-u", "--user", "-w",
                       "--write-out", "-A", "--user-agent", "--connect-timeout", "-e", "--referer"):
                i += 2
                continue
            elif not a.startswith("-") and not url:
                url = a
            i += 1
        if method in ("HEAD",):
            method = "GET"
        if method != "GET" or body:
            self.log.add("Bash", "http", "write_stubbed", method=method if method != "GET" else "POST", url=url)
            return "curl: (22) The requested URL returned error: 403 (write requests are not permitted)\n", 22
        out, rc = self.http.get(url)
        self.log.add("Bash", "http", "executed", out, url=url, rc=rc)
        return out, rc


def _cap(text: str) -> str:
    return text if len(text) <= OUTPUT_CAP else text[:OUTPUT_CAP] + f"\n... [output truncated, {len(text)} chars total]"


def sandbox_env(*, deadline_epoch: float, budget_sec: float, stall_sec: float) -> dict[str, str]:
    """What a real curiosity turn's environment advertises (ORION_TURN_* clock, a DSN to name).
    The DSN carries no credential: psql never runs inside the sandbox."""
    return {
        "ORION_TURN_BUDGET_SEC": str(int(budget_sec)),
        "ORION_TURN_DEADLINE_EPOCH": str(int(deadline_epoch)),
        "ORION_TURN_STEP_STALL_SEC": str(int(stall_sec)),
        "ORION_CURIOSITY_PG_DSN": "postgresql://orion_readonly@orion-athena-sql-db:5432/conjourney",
        "ORION_CURIOSITY_GRAPH_HOST": "orion-athena-falkordb",
        "ORION_CURIOSITY_GRAPH_PORT": "6379",
        "ORION_CURIOSITY_GRAPH_USER": "orion_curiosity",
        "ORION_CURIOSITY_GRAPH_PASSWORD": "replay",
        "ORION_CURIOSITY_GRAPH_OWN": "orion_worldview",
        "ORION_CURIOSITY_GRAPH_ATLAS": "orion_substrate",
    }


def sandbox_name(run_tag: str, task_id: str, model: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"orion-replay-{run_tag}-{task_id}-{model}")[:120]
    return f"{safe}-{uuid.uuid4().hex[:6]}"


async def to_thread(fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
    return await asyncio.to_thread(fn, *a, **kw)
