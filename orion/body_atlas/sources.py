"""The rest of the body: machines, cameras, stores and processes, each from a source that exists.

The v1 atlas saw only the bus. This module gathers everything else into ``BodyInputs``, and
``build.build_atlas(..., body=...)`` turns it into parts and flows. No list of parts lives here:
every part comes from a declared file in the checkout or a live read, and every source's outcome
is recorded in ``reads`` (``unread`` with the error when it fails, so a partial body says so).

Declared (read from the checkout, so a past export works too):

* hosts: ``config/biometrics/node_catalog.yaml`` ``nodes:`` (biometrics-monitored) and
  ``unmonitored_nodes:`` (machines biometrics never reads; NodeCatalog ignores that section).
* placement: ``mesh-utilities/common/include_services_<host>.txt`` and an unmonitored node's
  ``runs_services``.
* cameras: ``config/vision_frame_router.yaml`` ``streams:``. Readers: the service whose compose
  mounts that file (the frame router reads every stream), and any service whose compose
  ``environment:`` or ``.env_example`` sets a ``*STREAM_ID`` key to a stream's name.
* table writers: orion-sql-writer's ``DEFAULT_ROUTE_MAP`` (message kind -> model -> table, read
  by AST, never imported) joined with ``orion/bus/channels.yaml`` producers of that kind.
* durable workflows: the ``DurableWorkflowV1`` Literal in ``orion/schemas/durable_run.py``.

Live (only with ``live=True``):

* docker on this host (``docker info`` / ``docker ps``): observed placement.
* Postgres as ``orion_readonly`` in a READ ONLY transaction (the model-replay sandbox's
  ``ReadOnlyPsql``): table row counts, biometrics last-seen per host, durable run activity,
  reducer cursors, camera activity.
* Chroma collections and counts; FalkorDB graphs (minus test/scratch graphs) and node counts.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from orion.body_atlas.model import SourceReadV1
# The one Postgres read path for Orion's own tools: orion_readonly, READ ONLY transaction, no
# secrets on argv, psql meta-commands refused. Stdlib only, so importing it here is cheap.
from orion.evals.model_replay.sandbox import ReadOnlyPsql

NODE_CATALOG = Path("config/biometrics/node_catalog.yaml")
INCLUDE_GLOB = "mesh-utilities/common/include_services_*.txt"
FRAME_ROUTER = Path("config/vision_frame_router.yaml")
SQL_WRITER_SETTINGS = Path("services/orion-sql-writer/app/settings.py")
SQL_WRITER_MODELS = Path("services/orion-sql-writer/app/models")
DURABLE_RUN_SCHEMA = Path("orion/schemas/durable_run.py")
CHANNELS = Path("orion/bus/channels.yaml")

DEFAULT_CHROMA_URL = "http://orion-athena-vector-db:8000"
# No machine-readable source names who runs a reducer: orion-substrate-runtime owns
# substrate_reduction_cursor (its REDUCER_SPECS), so it is the runner, labeled assumed_runner.
REDUCER_RUNNER = "orion-substrate-runtime"
PROCESS_WINDOW_DAYS = 30

# What no machine-readable source covers yet. Printed with every whole-body atlas.
KNOWN_GAPS = [
    "mood v4 autoencoder: no declared or live source names it as a running part",
    "personality program: only tables, no process record",
    "AI Town's bus identity: runs on circe, traces only (aitown_chat_history_log, falkor:orion_substrate_aitown)",
    "graphify: its graph lives on the host only, not visible from the harness",
    "Postgres reads: who reads a table is not recorded anywhere machine-readable",
    "docker on other hosts: only this host's containers are observed; circe/hecate placement is declared",
]

Origin = Literal["declared", "live"]


class HostIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str = "unknown"
    hardware: str | None = None
    monitored: bool = True  # in node_catalog ``nodes:`` (biometrics reads it)
    expected_online: bool = True
    last_seen_epoch: float | None = None
    aliases: list[str] = Field(default_factory=list)
    cameras: list[str] = Field(default_factory=list)  # stream ids that are this machine's camera


class PlacementIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str  # raw name as the source wrote it (container, allowlist line)
    host: str
    evidence: Literal["observed_placement", "declared_placement"]


class SensorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # stream id
    readers: list[tuple[str, str]] = Field(default_factory=list)  # (raw service, where it was found)
    events: int | None = None
    last_seen_epoch: float | None = None


class StoreIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    store: Literal["postgres", "chroma", "falkordb"]
    rows: int | None = None  # rows / items / nodes; None = not read
    # (raw service, relation, evidence, via)
    links: list[tuple[str, str, str, str]] = Field(default_factory=list)


class ProcessIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    family: Literal["durable_workflow", "substrate_reducer"]
    runner: str | None
    runner_evidence: Literal["declared_runner", "assumed_runner"] = "declared_runner"
    declared: bool
    runs: int | None = None
    last_run_epoch: float | None = None


class BodyInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hosts: list[HostIn] = Field(default_factory=list)
    placements: list[PlacementIn] = Field(default_factory=list)
    sensors: list[SensorIn] = Field(default_factory=list)
    stores: list[StoreIn] = Field(default_factory=list)
    processes: list[ProcessIn] = Field(default_factory=list)
    reads: list[SourceReadV1] = Field(default_factory=list)
    # channel -> the store writer (orion-sql-writer) that persists it into a table. With the body,
    # that producer -> writer bus flow is replaced by the producer -> table ``writes`` flow: the
    # same fact (this message is kept) counted once, at its real destination, not twice.
    persisted: dict[str, str] = Field(default_factory=dict)
    # env-declared stores that do not exist live: reported, never parts
    missing_declared_stores: list[str] = Field(default_factory=list)
    known_gaps: list[str] = Field(default_factory=list)


class _Reads:
    def __init__(self) -> None:
        self.items: list[SourceReadV1] = []

    def add(self, name: str, origin: Origin, status: str, detail: str = "") -> None:
        self.items.append(SourceReadV1(name=name, origin=origin, status=status, detail=detail[:300]))

    def run(self, name: str, origin: Origin, fn: Callable[[], Any], ok: Callable[[Any], str] = lambda v: "") -> Any:
        """Call ``fn``; any exception is recorded as unread and returns None. Never silent."""
        try:
            value = fn()
        except PartialRead as exc:
            self.add(name, origin, "partial", str(exc))
            return exc.value
        except _Absent as exc:
            self.add(name, origin, "absent", str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - every failure is reported, none hidden
            self.add(name, origin, "unread", f"{type(exc).__name__}: {exc}".strip())
            return None
        self.add(name, origin, "read", ok(value))
        return value


class _Absent(Exception):
    pass


def _read_text(root: Path, rel: Path) -> str:
    path = root / rel
    if not path.exists():
        raise _Absent(f"{rel} not in this checkout")
    return path.read_text(encoding="utf-8")


# --- declared parsers (pure: text in, records out) ------------------------------------------

def parse_node_catalog(data: dict) -> tuple[list[HostIn], list[PlacementIn]]:
    defaults = data.get("defaults") or {}
    hosts, placed = [], []
    for section, monitored in (("nodes", True), ("unmonitored_nodes", False)):
        for node_id, spec in (data.get(section) or {}).items():
            spec = spec or {}
            name = str(node_id).strip().lower()
            hosts.append(HostIn(
                name=name,
                role=str(spec.get("role") or defaults.get("role") or "unknown"),
                hardware=str(spec["hardware"]) if spec.get("hardware") else None,
                monitored=monitored,
                expected_online=bool(spec.get("expected_online", defaults.get("expected_online", True))),
                aliases=sorted({str(a).strip().lower() for a in spec.get("aliases") or []} | {name}),
                cameras=[str(c) for c in spec.get("cameras") or []],
            ))
            for svc in spec.get("runs_services") or []:
                placed.append(PlacementIn(service=str(svc), host=name, evidence="declared_placement"))
    return hosts, placed


def parse_include_services(host: str, text: str) -> list[PlacementIn]:
    out = []
    for line in text.splitlines():
        svc = line.split("#", 1)[0].strip()
        if svc:
            out.append(PlacementIn(service=svc, host=host, evidence="declared_placement"))
    return out


def parse_frame_router_streams(data: dict) -> list[str]:
    """Stream ids under ``streams:`` that are not disabled. The file's ``cameras:`` section is a
    stale MVP list (mock-cam-01, porch_eye, ...) the router no longer reads; it is ignored."""
    streams = data.get("streams") or {}
    return sorted(str(k) for k, v in streams.items() if not (isinstance(v, dict) and v.get("enabled") is False))


_STREAM_KEY = re.compile(r"^([A-Z0-9_]*STREAM_ID)\s*=\s*['\"]?([^'\"#\s]+)['\"]?")
_COMPOSE_DEFAULT = re.compile(r"\$\{[A-Za-z0-9_]+:?-([^}]*)\}")


def _stream_setting(line: str) -> tuple[str, str] | None:
    """(key, stream) for a ``*STREAM_ID=...`` line; a compose ``${VAR:-walkway}`` reads as its
    default, a bare ``${VAR}`` as nothing (the service's .env_example says what it is)."""
    m = _STREAM_KEY.match(_COMPOSE_DEFAULT.sub(lambda d: d.group(1), line.strip()))
    if not m or m.group(2).startswith("$"):
        return None
    return m.group(1), m.group(2)


def _compose_env(service_spec: dict) -> list[str]:
    env = service_spec.get("environment") or []
    if isinstance(env, dict):
        return [f"{k}={v}" for k, v in env.items() if v is not None]
    return [str(e) for e in env]


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that reads compose-only tags (``!override``, ``!reset``) as plain values."""


def _untagged(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _untagged)


class PartialRead(Exception):
    """A source read that worked but skipped some inputs; ``value`` is what was read."""

    def __init__(self, value: Any, problem: str) -> None:
        super().__init__(problem)
        self.value = value


def stream_readers(root: Path, streams: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
    """stream -> [(service, where)]. Rules only, no names: a compose that mounts the frame
    router policy reads every stream; a ``*STREAM_ID=<stream>`` setting reads that stream.
    Raises PartialRead (carrying the result) when a compose file does not parse."""
    wanted = set(streams)
    out: dict[str, set[tuple[str, str]]] = {s: set() for s in wanted}
    services = root / "services"
    if not services.is_dir():
        return {s: [] for s in wanted}
    broken: list[str] = []
    for compose in sorted(services.glob("*/docker-compose*.yml")):
        try:
            data = yaml.load(compose.read_text(encoding="utf-8"), Loader=_ComposeLoader) or {}  # noqa: S506
        except yaml.YAMLError:
            broken.append(str(compose.relative_to(root)))
            continue
        rel = str(compose.relative_to(root))
        for svc, spec in (data.get("services") or {}).items():
            if not isinstance(spec, dict):
                continue
            vols = " ".join(str(v) for v in spec.get("volumes") or [])
            if FRAME_ROUTER.name in vols:
                for s in wanted:
                    out[s].add((str(svc), f"{rel} mounts {FRAME_ROUTER.name}"))
            for line in _compose_env(spec):
                kv = _stream_setting(line)
                if kv and kv[1] in wanted:
                    out[kv[1]].add((str(svc), f"{rel} {kv[0]}"))
    for env in sorted(services.glob("*/.env_example")):
        for line in env.read_text(encoding="utf-8").splitlines():
            kv = _stream_setting(line)
            if kv and kv[1] in wanted:
                out[kv[1]].add((env.parent.name, f"{env.relative_to(root)} {kv[0]}"))
    result = {s: sorted(v) for s, v in out.items()}
    if broken:
        raise PartialRead(result, f"{len(broken)} compose file(s) did not parse: {', '.join(broken[:3])}")
    return result


def parse_route_map(settings_text: str) -> dict[str, str]:
    """``DEFAULT_ROUTE_MAP`` literal (message kind -> model class), by AST: sql-writer's
    settings import its own service deps, so it is never imported here."""
    for node in ast.parse(settings_text).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if getattr(target, "id", None) == "DEFAULT_ROUTE_MAP" and isinstance(node.value, ast.Dict):
            return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)}
    raise ValueError("DEFAULT_ROUTE_MAP literal not found")


def parse_model_tables(texts: Iterable[str]) -> dict[str, str]:
    """model class -> ``__tablename__`` across sql-writer's model files."""
    out: dict[str, str] = {}
    for text in texts:
        for cls in ast.walk(ast.parse(text)):
            if not isinstance(cls, ast.ClassDef):
                continue
            for stmt in cls.body:
                if (isinstance(stmt, ast.Assign) and getattr(stmt.targets[0], "id", None) == "__tablename__"
                        and isinstance(stmt.value, ast.Constant)):
                    out[cls.name] = str(stmt.value.value)
    return out


def table_writers(route_map: dict[str, str], tables: dict[str, str], catalog: Iterable[dict],
                  writer: str) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str]]:
    """(table -> [(producer, channel)], channel -> writer) for the kinds ``writer`` (sql-writer)
    persists. The writer itself is not a table's writer edge: it writes every table and would
    glue all stores to itself; the channels it persists are returned so the build can drop the
    producer -> writer bus flow that the ``writes`` flow now carries."""
    out: dict[str, set[tuple[str, str]]] = {}
    persisted: dict[str, str] = {}
    for entry in catalog:
        kind = entry.get("message_kind")
        table = tables.get(route_map.get(kind, "")) if kind else None
        if not table or writer not in (entry.get("consumer_services") or []):
            continue
        producers = [p for p in entry.get("producer_services") or [] if isinstance(p, str) and p and p != "*"]
        for producer in producers:
            out.setdefault(table, set()).add((producer, str(entry["name"])))
        if producers:
            persisted[str(entry["name"])] = writer
    return {t: sorted(v) for t, v in out.items()}, persisted


def _constant(tree: ast.AST, name: str) -> str | None:
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name
                and isinstance(node.value, ast.Constant)):
            return str(node.value.value)
    return None


def durable_runner(schema_text: str, catalog: Iterable[dict]) -> str | None:
    """Who runs durable workflows: the producer of the channel carrying ``DURABLE_RUN_STATE_KIND``
    (whoever reports a run's state is the one executing it). None when not exactly one."""
    kind = _constant(ast.parse(schema_text), "DURABLE_RUN_STATE_KIND")
    producers = {p for c in catalog if kind and c.get("message_kind") == kind
                 for p in c.get("producer_services") or [] if isinstance(p, str) and p != "*"}
    return producers.pop() if len(producers) == 1 else None


def parse_durable_workflows(schema_text: str) -> list[str]:
    for node in ast.walk(ast.parse(schema_text)):
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if getattr(target, "id", None) == "DurableWorkflowV1" and isinstance(node.value, ast.Subscript):
            elts = node.value.slice.elts if isinstance(node.value.slice, ast.Tuple) else [node.value.slice]
            return [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    raise ValueError("DurableWorkflowV1 Literal not found")


_SCRATCH_TOKEN = re.compile(r"^(tmp|tests?\d*|livetest\d*|smoke\d*|scratch\d*)$")


def is_scratch_graph(name: str) -> bool:
    """Test/scratch FalkorDB graphs (``t_assertion_<hash>``, ``*_livetest2``): a leading ``t_``
    (the assertion harness's prefix) or a ``_`` token that is a test word. A rule, so a new
    scratch graph needs no list edit; the dropped names are listed in the read detail."""
    toks = name.lower().split("_")
    return toks[0] == "t" or any(_SCRATCH_TOKEN.match(tok) for tok in toks)


def env_store_refs(root: Path, names: Iterable[str]) -> dict[str, list[str]]:
    """store name -> services whose .env_example sets some key to exactly that name."""
    wanted = set(names)
    out: dict[str, set[str]] = {n: set() for n in wanted}
    for env in sorted((root / "services").glob("*/.env_example")):
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^[A-Z0-9_]+\s*=\s*['\"]?([^'\"#\s]+)['\"]?\s*(#.*)?$", line.strip())
            if m and m.group(1) in wanted:
                out[m.group(1)].add(env.parent.name)
    return {n: sorted(v) for n, v in out.items()}


def env_declared_collections(root: Path) -> set[str]:
    """Values of ``*COLLECTION*`` keys in any .env_example: what services expect Chroma to hold."""
    found = set()
    for env in sorted((root / "services").glob("*/.env_example")):
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^[A-Z0-9_]*COLLECTION[A-Z0-9_]*\s*=\s*['\"]?([A-Za-z0-9_.-]+)['\"]?\s*(#.*)?$", line.strip())
            if m:
                found.add(m.group(1))
    return found


# --- live readers -----------------------------------------------------------------------------

class SqlRunner(Protocol):
    def run(self, sql: str, flags: list[str]) -> tuple[str, int]: ...


_PSQL_STDERR = re.compile(r"^(psql:|WARNING:|NOTICE:|HINT:|DETAIL:|INFO:)")


def psql_rows(runner: SqlRunner, sql: str) -> list[dict[str, str]]:
    out, code = runner.run(sql, ["--csv", "-q"])
    if code != 0:
        raise RuntimeError(" ".join(out.split())[:240] or f"psql exit {code}")
    # ReadOnlyPsql returns stdout and stderr joined: a NOTICE/WARNING on success is not a row
    lines = [ln for ln in out.splitlines() if ln.strip() and not _PSQL_STDERR.match(ln)]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def parse_ts(text: str | None) -> float | None:
    """Postgres text timestamps (``2026-10-29 20:51:12.87+00``) -> epoch; None when unparseable."""
    if not text:
        return None
    t = text.strip().replace(" ", "T", 1)
    if re.search(r"[+-]\d\d$", t):
        t += ":00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


SQL_TABLE_ROWS = "SELECT relname, n_live_tup FROM pg_stat_user_tables"
# timestamp is VARCHAR: keep only ISO-shaped values so max() is chronological
SQL_HOST_LAST_SEEN = ("SELECT node, max(timestamp) AS last_seen FROM orion_biometrics_summary "
                      "WHERE timestamp ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}[ T]' GROUP BY node")
SQL_DURABLE = (f"SELECT workflow, count(DISTINCT run_id) AS runs, max(created_at) AS last_run "
               f"FROM substrate_durable_run_state WHERE created_at > now() - interval '{PROCESS_WINDOW_DAYS} days' "
               "GROUP BY workflow")
SQL_REDUCERS = "SELECT cursor_name, updated_at FROM substrate_reduction_cursor"
SQL_CAMERAS = (f"SELECT stream_id, count(*) AS events, max(created_at) AS last_seen FROM vision_events "
               f"WHERE stream_id IS NOT NULL AND created_at > now() - interval '{PROCESS_WINDOW_DAYS} days' "
               "GROUP BY stream_id")


def _docker(argv: list[str]) -> str:
    p = subprocess.run(["docker", *argv], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:240] or f"docker exit {p.returncode}")
    return p.stdout


def _http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - fixed internal GET
        return json.loads(resp.read().decode("utf-8"))


class LiveReaders(BaseModel):
    """Every live dependency, injectable so tests never touch a DB."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sql: Any = None  # SqlRunner
    docker: Callable[[list[str]], str] | None = None
    http_json: Callable[[str], Any] | None = None
    graph: Any = None  # callable(cmd, *args) -> reply; the FalkorDB connection's execute_command
    chroma_url: str = DEFAULT_CHROMA_URL


def default_live_readers(falkor_uri: str | None) -> LiveReaders:
    graph = None
    if falkor_uri:
        import redis

        conn = redis.Redis.from_url(falkor_uri, decode_responses=True, socket_timeout=10.0, socket_connect_timeout=3.0)
        graph = conn.execute_command
    return LiveReaders(sql=ReadOnlyPsql(timeout_sec=60.0), docker=_docker, http_json=_http_json, graph=graph)


def _require(value: Any, what: str) -> Any:
    if value is None:
        raise RuntimeError(f"no {what} reader")
    return value


# --- gather -----------------------------------------------------------------------------------

def gather_body(root: Path, catalog: list[dict], *, live: LiveReaders | None) -> BodyInputs:
    """Every declared source from ``root``; every live source when ``live`` is given."""
    reads = _Reads()
    body = BodyInputs(known_gaps=list(KNOWN_GAPS))

    # hosts + declared placement
    cat = reads.run(f"declared:{NODE_CATALOG}", "declared",
                    lambda: parse_node_catalog(yaml.safe_load(_read_text(root, NODE_CATALOG)) or {}),
                    lambda v: f"{len(v[0])} hosts")
    if cat:
        body.hosts, body.placements = cat[0], list(cat[1])
    for path in sorted(root.glob(INCLUDE_GLOB)):
        host = path.stem.removeprefix("include_services_")
        got = reads.run(f"declared:{path.relative_to(root)}", "declared",
                        lambda p=path, h=host: parse_include_services(h, p.read_text(encoding="utf-8")),
                        lambda v: f"{len(v)} services")
        body.placements.extend(got or [])

    # cameras
    streams = reads.run(f"declared:{FRAME_ROUTER}", "declared",
                        lambda: parse_frame_router_streams(yaml.safe_load(_read_text(root, FRAME_ROUTER)) or {}),
                        lambda v: f"{len(v)} streams")
    if streams:
        readers = reads.run("declared:services/*/docker-compose*.yml + .env_example (*STREAM_ID)", "declared",
                            lambda: stream_readers(root, streams),
                            lambda v: f"{sum(len(x) for x in v.values())} camera readers") or {}
        body.sensors = [SensorIn(name=s, readers=readers.get(s, [])) for s in streams]

    # table writers (declared) -> postgres store candidates
    def _writers() -> dict[str, list[tuple[str, str]]]:
        route_map = parse_route_map(_read_text(root, SQL_WRITER_SETTINGS))
        models = root / SQL_WRITER_MODELS
        if not models.is_dir():
            raise _Absent(f"{SQL_WRITER_MODELS} not in this checkout")
        tables = parse_model_tables(p.read_text(encoding="utf-8") for p in sorted(models.rglob("*.py")))
        return table_writers(route_map, tables, catalog, writer=SQL_WRITER_SETTINGS.parts[1])

    routed = reads.run(f"declared:{SQL_WRITER_SETTINGS} DEFAULT_ROUTE_MAP", "declared", _writers,
                       lambda v: f"{len(v[0])} tables with a known writer, {len(v[1])} channels persisted")
    writers, body.persisted = routed if routed else ({}, {})

    # durable workflows (declared)
    def _workflows() -> tuple[list[str], str | None]:
        text = _read_text(root, DURABLE_RUN_SCHEMA)
        return parse_durable_workflows(text), durable_runner(text, catalog)

    wf = reads.run(f"declared:{DURABLE_RUN_SCHEMA} DurableWorkflowV1", "declared", _workflows,
                   lambda v: f"{len(v[0])} workflows, runner {v[1] or 'unknown'}")
    workflows, runner = wf if wf else ([], None)
    procs = {w: ProcessIn(name=w, family="durable_workflow", runner=runner, declared=True) for w in workflows}

    table_rows: dict[str, int] | None = None
    if live is not None:
        # observed placement: this host's docker
        def _ps() -> list[PlacementIn]:
            run = _require(live.docker, "docker")
            host = run(["info", "--format", "{{.Name}}"]).strip().lower()
            names = [n.strip() for n in run(["ps", "--format", "{{.Names}}"]).splitlines() if n.strip()]
            return [PlacementIn(service=n, host=host, evidence="observed_placement") for n in names]

        observed = reads.run("live:docker ps (this host)", "live", _ps,
                             lambda v: f"{len(v)} containers on {v[0].host if v else '?'}")
        body.placements.extend(observed or [])

        sql = live.sql
        rows = reads.run("live:postgres pg_stat_user_tables", "live",
                         lambda: psql_rows(_require(sql, "postgres"), SQL_TABLE_ROWS),
                         lambda v: f"{len(v)} tables")
        if rows is not None:
            table_rows = {r["relname"]: int(float(r["n_live_tup"] or 0)) for r in rows}

        seen = reads.run("live:postgres orion_biometrics_summary", "live",
                         lambda: psql_rows(_require(sql, "postgres"), SQL_HOST_LAST_SEEN),
                         lambda v: f"{len(v)} hosts reporting")
        if seen:
            alias = {a: h.name for h in body.hosts for a in h.aliases}
            for r in seen:
                host = alias.get(str(r["node"]).strip().lower())
                ts = parse_ts(r.get("last_seen"))
                for h in body.hosts:
                    if h.name == host and ts:
                        h.last_seen_epoch = max(h.last_seen_epoch or 0.0, ts)

        durable = reads.run("live:postgres substrate_durable_run_state", "live",
                            lambda: psql_rows(_require(sql, "postgres"), SQL_DURABLE),
                            lambda v: f"{len(v)} workflows ran in {PROCESS_WINDOW_DAYS} d")
        if durable is not None:  # read: a workflow with no row ran 0 times in the window
            for p in procs.values():
                p.runs = 0
        for r in durable or []:
            w = r["workflow"]
            p = procs.setdefault(w, ProcessIn(name=w, family="durable_workflow", runner=runner, declared=False))
            p.runs, p.last_run_epoch = int(float(r["runs"] or 0)), parse_ts(r.get("last_run"))

        cursors = reads.run("live:postgres substrate_reduction_cursor", "live",
                            lambda: psql_rows(_require(sql, "postgres"), SQL_REDUCERS),
                            lambda v: f"{len(v)} reducer cursors")
        for r in cursors or []:
            name = f"reducer:{r['cursor_name']}"
            procs[name] = ProcessIn(name=r["cursor_name"], family="substrate_reducer", runner=REDUCER_RUNNER,
                                    runner_evidence="assumed_runner", declared=False,
                                    last_run_epoch=parse_ts(r.get("updated_at")))

        cams = reads.run("live:postgres vision_events", "live",
                         lambda: psql_rows(_require(sql, "postgres"), SQL_CAMERAS),
                         lambda v: f"{len(v)} streams with events in {PROCESS_WINDOW_DAYS} d")
        by_stream = {s.name: s for s in body.sensors}
        if cams is not None:  # read: a stream with no row sent 0 events in the window
            for s in body.sensors:
                s.events = 0
        for r in cams or []:
            s = by_stream.get(r["stream_id"])
            if s is None:  # a stream the router policy no longer lists: still a camera that sent
                s = by_stream[r["stream_id"]] = SensorIn(name=r["stream_id"])
                body.sensors.append(s)
            s.events, s.last_seen_epoch = int(float(r["events"] or 0)), parse_ts(r.get("last_seen"))

        # Chroma
        def _chroma() -> list[StoreIn]:
            get = _require(live.http_json, "http")
            cols = get(f"{live.chroma_url}/api/v1/collections")
            out = []
            for c in cols:
                count = get(f"{live.chroma_url}/api/v1/collections/{c['id']}/count")
                out.append(StoreIn(name=str(c["name"]), store="chroma", rows=int(count)))
            return out

        chroma = reads.run(f"live:chroma {live.chroma_url}", "live", _chroma, lambda v: f"{len(v)} collections")
        if chroma is not None:
            existing = {c.name for c in chroma}
            body.missing_declared_stores += sorted(f"chroma:{n}" for n in env_declared_collections(root) - existing)
            body.stores.extend(chroma)

        # FalkorDB
        dropped: list[str] = []

        def _falkor() -> list[StoreIn]:
            cmd = _require(live.graph, "falkordb")
            out = []
            for g in sorted(str(x) for x in cmd("GRAPH.LIST")):
                if is_scratch_graph(g):
                    dropped.append(g)
                    continue
                reply = cmd("GRAPH.RO_QUERY", g, "MATCH (n) RETURN count(n)")
                n = int(float(reply[1][0][0])) if len(reply) > 1 and reply[1] else 0
                out.append(StoreIn(name=g, store="falkordb", rows=n))
            return out

        falkor = reads.run("live:falkordb GRAPH.LIST", "live", _falkor,
                           lambda v: f"{len(v)} graphs; {len(dropped)} test/scratch dropped: {', '.join(dropped)}")
        body.stores.extend(falkor or [])

    # env links for Chroma/Falkor stores (only stores that exist live)
    named = [s for s in body.stores if s.store in ("chroma", "falkordb")]
    if named:
        refs = reads.run("declared:services/*/.env_example (store names)", "declared",
                         lambda: env_store_refs(root, {s.name for s in named}),
                         lambda v: f"{sum(len(x) for x in v.values())} service->store references") or {}
        for s in named:
            s.links.extend((svc, "uses", "declared_env", "") for svc in refs.get(s.name, []))

    # Postgres stores: a table is a part only when something links to it (a known writer).
    kept_channels: set[str] = set()
    for table, ws in sorted(writers.items()):
        if table_rows is not None and table not in table_rows:
            continue  # declared by a model but not in this database
        body.stores.append(StoreIn(name=table, store="postgres",
                                   rows=table_rows.get(table) if table_rows is not None else None,
                                   links=[(p, "writes", "declared_route", ch) for p, ch in ws]))
        kept_channels.update(ch for _, ch in ws)
    # only a channel whose table became a part moves its flow to that table; any other keeps
    # its producer -> writer bus flow (never a flow dropped with nothing replacing it)
    body.persisted = {ch: w for ch, w in body.persisted.items() if ch in kept_channels}

    body.processes = sorted(procs.values(), key=lambda p: (p.family, p.name))
    body.reads = reads.items
    return body
