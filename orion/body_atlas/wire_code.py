"""Who calls whom over HTTP, and who writes or reads which Redis key, from the code that says so.

The bus atlas cannot see two wires (design ``2026-10-30-self-now-capacities-vsm-design.md``,
gap 1): a service calling another's HTTP API (the body-economy market, ``POST /physiology/bid``
on orion-durable-runs) and a service writing a Redis key others read (the strain line,
``orion:regulation:latest``). This module finds both by rule, never from a hand-typed list:

**HTTP.** A service names a callee's URL: a string constant in its code that is exactly a URL
(``"http://orion-athena-durable-runs:8121/physiology/bid"``), a settings field its code reads
whose value (``.env_example`` / compose first, the code default second) is a URL, or an env key
its code reads (``os.getenv("KEY")``) whose value is one. The URL's host is resolved to an atlas
part later (``build.match_part``: ``orion-athena-notify`` and ``notify`` are orion-notify). A host
that matches no part (``127.0.0.1``, a tailscale address, an outside site) is reported, never
guessed. Direction is the URL's own: the namer calls the owner.

**Redis keys.** A call ``<redis handle>.<verb>(KEY, ...)``. A handle is a name the code binds
from a Redis constructor (``redis.Redis.from_url(...)``, ``aioredis.from_url``), a pipeline of a
handle, a parameter annotated ``Redis``, or an attribute named for Redis (``self.redis``,
``bus.redis``, ``self._redis_client``; not ``redis_url`` or ``redis_key``). The verb gives the
direction (``set``/``setex``/``hset``/``incr``/``expire``/``delete``... write; ``get``/``hgetall``/
``exists``/``lrange``... read; a pop reads). Stream and pub/sub verbs (``xadd``, ``publish``) are
the bus, which the atlas already reads, and are skipped. ``KEY`` must resolve to an exact string:
a literal, a constant (this module's, or one imported from another module), a settings field, or
an env key. A key built from a shared prefix *constant* (``f"{PREFIX}{id}"``, ``PREFIX + id``) is
one key family, ``PREFIX*``; a key built any other way (a literal-prefix f-string, a parameter) is
**dynamic**: counted per service, never a part.

**Attribution** (as ``store_code.py``): ``services/<svc>/...`` is ``<svc>``. A shared ``orion/``
module is split into top-level symbols; a service gets the acts of the symbols it imports, and of
the shared symbols those use in turn (closure over imports and same-module calls). ``tests/``,
``evals/`` and the atlas itself are skipped.
"""
from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import yaml

from orion.body_atlas.store_code import _SKIP_DIRS, _SKIP_PREFIXES, _YAML_LOADER, _symbols, _tokens

# Redis verbs by direction (redis-py method names). A pop takes a value out: a read for flow
# direction (the popper consumes what another wrote).
REDIS_WRITE = frozenset("""set setex setnx psetex mset msetnx getset getdel hset hmset hsetnx hincrby hincrbyfloat
hdel incr incrby incrbyfloat decr decrby expire pexpire expireat pexpireat persist delete unlink rename renamenx
append setrange lpush rpush lpushx rpushx lset linsert ltrim lrem sadd srem smove zadd zrem zincrby
zremrangebyscore zremrangebyrank zremrangebylex pfadd json_set""".split())
REDIS_READ = frozenset("""get mget hget hgetall hmget hkeys hvals hlen hexists hstrlen exists ttl pttl type
lrange llen lindex lpop rpop blpop brpop brpoplpush lmove smembers scard sismember smismember srandmember spop
zrange zrevrange zrangebyscore zrevrangebyscore zscore zcard zcount zrank zrevrank zpopmin zpopmax bzpopmin
strlen getrange pfcount object""".split())
# a handle's name says what it is; one of these tokens says it is config about Redis, not Redis
_NOT_HANDLE = {"url", "uri", "key", "keys", "prefix", "channel", "channels", "stream", "streams", "host",
               "port", "db", "ttl", "timeout", "enabled", "settings", "config", "dsn", "password", "name"}
_URL_RE = re.compile(r"^https?://[^\s/'\"]+(?:[/?#][^\s'\"]*)?$")
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*['\"]?([^'\"#\s]*)['\"]?\s*(#.*)?$")
_COMPOSE_DEFAULT = re.compile(r"\$\{([A-Z][A-Z0-9_]*):?-([^}]*)\}")
_RESOLVE_DEPTH = 4


@dataclass(frozen=True)
class Act:
    """One thing a module does: ``http`` (value = URL), ``redis`` (value = key or ``PREFIX*``,
    relation writes / reads), ``dynamic`` (a Redis key that is not an exact string; value = the
    code spelling), or ``env`` (an env key read whose value decides: resolved per service)."""

    kind: str  # http / redis / dynamic / env-http / env-redis
    value: str
    relation: str = ""


@dataclass
class WireRef:
    service: str
    value: str  # URL (http) or key (redis)
    relation: str  # calls / writes / reads
    via: str  # module that names it (shared: "orion/x.py:sym (imported by svc)")


@dataclass
class WireRefs:
    http: list[WireRef] = field(default_factory=list)
    redis: dict[str, list[WireRef]] = field(default_factory=dict)  # key -> one ref per (service, relation)
    # service -> modules with a Redis act whose key is not an exact string (never a part)
    dynamic: dict[str, list[str]] = field(default_factory=dict)
    # shared modules with an act that no service reaches (reported, never guessed)
    unattributed: list[str] = field(default_factory=list)
    modules_scanned: int = 0
    skipped: list[str] = field(default_factory=list)


def url_host(url: str) -> str | None:
    """The host of an http(s) URL, a compose ``${VAR}-`` prefix dropped (``${PROJECT}-vector-host``
    is ``vector-host``); None when the value is not a URL."""
    if not _URL_RE.match(url):
        return None
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return None
    host = re.sub(r"^\$\{[^}]*\}-?", "", url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]) \
        if "${" in url else host
    return host.lower() or None


def is_url(value: str) -> bool:
    return url_host(value) is not None


# --- service env --------------------------------------------------------------------------------

def _compose_pairs(text: str) -> Iterable[tuple[str, str]]:
    for key, default in _COMPOSE_DEFAULT.findall(text):
        yield key, default.strip().strip("'\"")
    try:
        data = yaml.load(text, Loader=_YAML_LOADER) or {}
    except yaml.YAMLError:
        return
    for spec in (data.get("services") or {}).values() if isinstance(data, dict) else ():
        env = (spec or {}).get("environment") if isinstance(spec, dict) else None
        items = env.items() if isinstance(env, dict) else (
            (i.split("=", 1) for i in env if isinstance(i, str) and "=" in i) if isinstance(env, list) else ())
        for k, v in items:
            if isinstance(v, str):
                yield str(k).strip(), v.strip()


def service_env(root: Path) -> dict[str, dict[str, str]]:
    """service -> {env key: value}: its ``.env_example``, then its compose files (a key the
    ``.env_example`` already sets keeps that value)."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for svc_dir in sorted(p for p in (root / "services").glob("*") if p.is_dir()):
        env = svc_dir / ".env_example"
        if env.is_file():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _ENV_LINE.match(line)
                if m and m.group(2):
                    out[svc_dir.name][m.group(1)] = m.group(2)
        for comp in sorted(svc_dir.glob("docker-compose*.y*ml")):
            for k, v in _compose_pairs(comp.read_text(encoding="utf-8", errors="replace")):
                if v and "${" not in v[:2]:
                    out[svc_dir.name].setdefault(k, v)
    return dict(out)


# --- one module ---------------------------------------------------------------------------------

def _path(node: ast.AST) -> str | None:
    if isinstance(node, (ast.Name, ast.Attribute)):
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001
            return None
    return None


def _last(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _named_redis(segment: str) -> bool:
    """``redis``, ``_redis``, ``redis_client``, ``aioredis``: named for Redis. ``redis_url``,
    ``redis_key``: config about Redis."""
    toks = _tokens(segment)
    return any(t.endswith("redis") for t in toks) and not toks & _NOT_HANDLE


def _docstrings(nodes: list[ast.AST]) -> set[int]:
    out = set()
    for node in nodes:
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            out.add(id(body[0].value))
    return out


class Resolver:
    """Exact string values for names in one module: its own constants (any scope; a name bound
    to two values is ambiguous and dropped), imported constants (``from m import K``, ``m.K``),
    the service's settings fields (``settings.foo_key``: env value first, code default second) and
    env reads (``os.getenv("K")``: deferred to the importing service's env)."""

    def __init__(self, scan: "_Scan", rel: str, nodes: list[ast.AST], svc: str | None) -> None:
        self.scan, self.rel, self.nodes, self.svc = scan, rel, nodes, svc
        self._local: dict[str, ast.AST] | None = None
        self.imported: dict[str, tuple[str, str]] = {}  # alias -> (module rel path, name)
        self.modules: dict[str, str] = {}  # alias -> module rel path
        for mod_rel, name, alias in scan.imports(rel, nodes):
            if name:
                self.imported[alias] = (mod_rel, name)
            else:
                self.modules[alias] = mod_rel

    @property
    def local(self) -> dict[str, ast.AST]:
        if self._local is None:
            local: dict[str, ast.AST] = {}
            bad: set[str] = set()
            for node in self.nodes:
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        if isinstance(t, ast.Name):
                            if t.id in local and ast.dump(local[t.id]) != ast.dump(node.value):
                                bad.add(t.id)
                            local[t.id] = node.value
            self._local = {k: v for k, v in local.items() if k not in bad}
        return self._local

    def value(self, expr: ast.AST | None, depth: int = 0) -> tuple[str, str] | None:
        """(kind, value): ``("str", s)`` exact, ``("family", prefix)``, ``("env", KEY)``; None."""
        if expr is None or depth > _RESOLVE_DEPTH:
            return None
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return ("str", expr.value)
        if isinstance(expr, ast.Await):
            return self.value(expr.value, depth + 1)
        if isinstance(expr, ast.Name):
            if expr.id in self.local:
                return self.value(self.local[expr.id], depth + 1)
            if expr.id in self.imported:
                mod_rel, name = self.imported[expr.id]
                v = self.scan.module_const(mod_rel, name, depth + 1)
                return ("str", v) if v is not None else None
            return None
        if isinstance(expr, ast.Attribute):
            base = _path(expr.value)
            if base in self.modules:
                v = self.scan.module_const(self.modules[base], expr.attr, depth + 1)
                return ("str", v) if v is not None else None
            if base in ("self", "cls") and expr.attr in self.local:
                return self.value(self.local[expr.attr], depth + 1)
            if self.svc is not None:
                v = self.scan.settings_value(self.svc, expr.attr)
                if v is not None:
                    return ("str", v)
            return None
        if isinstance(expr, ast.Call):
            key = _env_key(expr)
            if key is not None:
                return ("env", key)
            return None
        if isinstance(expr, ast.JoinedStr) and expr.values:
            first = expr.values[0]
            if isinstance(first, ast.FormattedValue) and not isinstance(first.value, ast.Constant):
                v = self.value(first.value, depth + 1)
                if v and v[0] == "str":
                    return ("family", v[1])
            return None
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add) and not isinstance(expr.left, ast.Constant):
            v = self.value(expr.left, depth + 1)
            if v and v[0] == "str":
                return ("family", v[1])
        return None


def _attr_read(node: ast.AST) -> str | None:
    """The attribute name a node reads: ``x.name`` or ``getattr(x, "name"[, default])``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
        return node.args[1].value
    return None


def _env_key(call: ast.Call) -> str | None:
    """``os.getenv("K")`` / ``os.environ.get("K")`` / ``getenv("K")`` -> ``K``."""
    f = call.func
    name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
    is_env = name == "getenv" or (name == "get" and isinstance(f, ast.Attribute) and _path(f.value) == "os.environ")
    if is_env and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


# a method on a handle that returns another handle (not a key act, not a reply value)
_HANDLE_METHODS = {"pipeline"}
# a module-level factory on the redis package: ``redis.from_url``, ``aioredis.from_url``
_CTOR_METHODS = {"from_url", "from_pool", "StrictRedis"}


def _redis_ctor(call: ast.Call) -> bool:
    """A call that makes a Redis client: ``redis.Redis.from_url(...)``, ``aioredis.from_url(...)``,
    ``Redis(...)``, ``redis.asyncio.Redis(...)``."""
    fp = _path(call.func) or ""
    segs = fp.split(".")
    if "Redis" in segs[-1]:  # a class: Redis, StrictRedis, FakeRedis
        return True
    return segs[-1] in _CTOR_METHODS and any(seg.lower().endswith("redis") for seg in segs[:-1])


def _handles(nodes: list[ast.AST]) -> set[str]:
    """Paths bound to a Redis handle anywhere in these nodes (scope-blind: a name is a handle if
    any binding of it is)."""
    out: set[str] = set()

    def is_handle(expr: ast.AST | None) -> bool:
        if expr is None:
            return False
        if isinstance(expr, ast.Await):
            return is_handle(expr.value)
        p = _path(expr)
        if p is not None:
            return p in out or _named_redis(_last(p))
        if isinstance(expr, ast.Call):
            if _redis_ctor(expr):
                return True
            return (isinstance(expr.func, ast.Attribute) and expr.func.attr in _HANDLE_METHODS
                    and is_handle(expr.func.value))
        return False

    for _ in range(4):
        grew = False
        for node in nodes:
            pairs: list[tuple[ast.AST, ast.AST | None]] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                pairs = [(t, node.value) for t in targets]
                if isinstance(node, ast.AnnAssign) and "Redis" in ast.unparse(node.annotation):
                    pairs.append((node.target, ast.Name(id="redis")))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                pairs = [(i.optional_vars, i.context_expr) for i in node.items if i.optional_vars is not None]
            elif isinstance(node, ast.arg) and node.annotation is not None and "Redis" in ast.unparse(node.annotation):
                if node.arg not in out:
                    out.add(node.arg)
                    grew = True
            for target, value in pairs:
                p = _path(target)
                if p and p not in out and is_handle(value):
                    out.add(p)
                    grew = True
        if not grew:
            break
    return out


def _is_handle_expr(expr: ast.AST, handles: set[str]) -> bool:
    p = _path(expr)
    if p is not None:
        return p in handles or _named_redis(_last(p))
    if isinstance(expr, ast.Await):
        return _is_handle_expr(expr.value, handles)
    if isinstance(expr, ast.Call):
        return _redis_ctor(expr) or (isinstance(expr.func, ast.Attribute) and expr.func.attr in _HANDLE_METHODS
                                     and _is_handle_expr(expr.func.value, handles))
    return False


def module_acts(nodes: list[ast.AST], res: Resolver, *, redis_text: bool = True) -> set[Act]:
    """Every HTTP / Redis act in ``nodes`` (a module's or one symbol's, from ``ast.walk``).
    ``redis_text=False``: the source never says "redis", so no Redis act is looked for."""
    acts: set[Act] = set()
    docs = _docstrings(nodes)
    fields = _class_field_urls(nodes)
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs \
                and is_url(node.value):
            if id(node) in fields:  # a class field's default: a call only where code reads the field
                acts.add(Act("field-http", node.value, fields[id(node)]))
            else:
                acts.add(Act("http", node.value))
        elif isinstance(node, ast.Call):
            key = _env_key(node)
            if key is not None:
                acts.add(Act("env-http", key))
            if res.svc is not None and (name := _attr_read(node)):  # getattr(settings, "x_url")
                v = res.scan.settings_value(res.svc, name)
                if v is not None and is_url(v):
                    acts.add(Act("http", v))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and res.svc is not None:
            v = res.scan.settings_value(res.svc, node.attr)
            if v is not None and is_url(v):
                acts.add(Act("http", v))
    if not redis_text:
        return acts
    handles = _handles(nodes)
    for call in nodes:
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.args):
            continue
        verb = call.func.attr
        rel = "writes" if verb in REDIS_WRITE else ("reads" if verb in REDIS_READ else None)
        if rel is None or not _is_handle_expr(call.func.value, handles):
            continue
        arg = call.args[0]
        if verb == "object":  # client.object("idletime", KEY)
            if len(call.args) < 2:
                continue
            arg = call.args[1]
        v = res.value(arg)
        if v is None:
            acts.add(Act("dynamic", _short(arg), rel))
        elif v[0] == "str":
            acts.add(Act("redis", v[1], rel))
        elif v[0] == "family":
            acts.add(Act("redis", v[1] + "*", rel))
        else:
            acts.add(Act("env-redis", v[1], rel))
    return acts


def _class_field_urls(nodes: list[ast.AST]) -> dict[int, str]:
    """id(URL constant) -> field name, for a URL that is a class field's default
    (``thought_url: str = "http://..."``, ``Field("http://...", alias=...)``). A model or settings
    class holding a URL is config: whoever imports the class does not call it; whoever reads the
    field (``policy.thought_url``) may."""
    out: dict[int, str] = {}
    for cls in nodes:
        if not isinstance(cls, ast.ClassDef):
            continue
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name, value = stmt.target.id, stmt.value
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name, value = stmt.targets[0].id, stmt.value
            else:
                continue
            for c in ast.walk(value) if value is not None else ():
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out[id(c)] = name.lower()
    return out


def _short(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:60]
    except Exception:  # noqa: BLE001
        return type(node).__name__


# --- repo scan ----------------------------------------------------------------------------------

def _py_files(root: Path) -> Iterable[Path]:
    for base in (root / "services", root / "orion"):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
            rel_dir = Path(dirpath).relative_to(root).as_posix() + "/"
            if rel_dir.startswith(_SKIP_PREFIXES):
                dirnames[:] = []
                continue
            for f in sorted(filenames):
                if f.endswith(".py") and not f.startswith("test_") and f != "conftest.py":
                    yield Path(dirpath) / f


def _service_of(rel: str) -> str | None:
    parts = rel.split("/")
    return parts[1] if parts[0] == "services" and len(parts) > 2 else None


def _settings_fields(tree: ast.AST) -> dict[str, tuple[str | None, str | None]]:
    """Field name (lower case) -> (env alias, default string) for every class whose base name
    ends in ``Settings`` (pydantic settings)."""
    out: dict[str, tuple[str | None, str | None]] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or not any((_path(b) or "").endswith("Settings") for b in cls.bases):
            continue
        for stmt in cls.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            alias = default = None
            v = stmt.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                default = v.value
            elif isinstance(v, ast.Call) and (_path(v.func) or "").endswith("Field"):
                if v.args and isinstance(v.args[0], ast.Constant) and isinstance(v.args[0].value, str):
                    default = v.args[0].value
                for kw in v.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        default = kw.value.value
                    if kw.arg in ("alias", "validation_alias", "env") and isinstance(kw.value, ast.Constant) \
                            and isinstance(kw.value.value, str):
                        alias = kw.value.value
            out[stmt.target.id.lower()] = (alias, default)
    return out


def _walk_stmts(stmts: list[ast.stmt]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for st in stmts:
        out.extend(ast.walk(st))
    return out


def _relevant(text: str, settings_names: Iterable[str] = ()) -> tuple[bool, bool]:
    """(may hold an HTTP act, may hold a Redis act), from the text alone: a cheap pre-filter.
    ``settings_names``: the service's settings fields whose value is a URL."""
    low = text.lower()
    return ("http" in text or "getenv" in text or "environ" in text or any(n in low for n in settings_names),
            "redis" in low)


# ``from orion.a.b import x, y as z`` (parenthesized or not) and ``import orion.a.b [as m]``: the
# orion imports of a module that needs no parse (it names no URL and no Redis)
_FROM_RE = re.compile(r"^[ \t]*from[ \t]+(orion[\w.]*)[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)", re.M)
_IMPORT_RE = re.compile(r"^[ \t]*import[ \t]+(orion[\w.]*)(?:[ \t]+as[ \t]+(\w+))?", re.M)
# an attribute read: ``x.name``, or ``getattr(x, "name"...)``
_ATTR_RE = re.compile(r"\.([A-Za-z_]\w*)|getattr\(\s*[\w.]+\s*,\s*[\"']([A-Za-z_]\w*)[\"']")
_ANY_FROM_RE = re.compile(r"^[ \t]*from[ \t]+(\.*)([\w.]*)[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)", re.M)


class _Scan:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = service_env(root)
        self.result = WireRefs()
        self._trees: dict[str, ast.Module | None] = {}
        self._nodes: dict[str, list[ast.AST]] = {}
        self._imports: dict[str, list[tuple[str, str, str]]] = {}
        self._consts: dict[str, dict[str, str]] = {}
        self._settings: dict[str, dict[str, str]] = {}
        self._texts: dict[str, str] = {}
        self._files: dict[Path, str | None] = {}
        self.service_modules: dict[str, list[str]] = defaultdict(list)

    # files
    def text(self, rel: str) -> str:
        if rel not in self._texts:
            self._texts[rel] = (self.root / rel).read_text(encoding="utf-8", errors="replace")
        return self._texts[rel]

    def tree(self, rel: str) -> ast.Module | None:
        if rel not in self._trees:
            try:
                self._trees[rel] = ast.parse(self.text(rel))
            except (SyntaxError, OSError, ValueError):
                self._trees[rel] = None
                self.result.skipped.append(rel)
        return self._trees[rel]

    def nodes(self, rel: str) -> list[ast.AST]:
        if rel not in self._nodes:
            tree = self.tree(rel)
            self._nodes[rel] = list(ast.walk(tree)) if tree is not None else []
        return self._nodes[rel]

    def _file_of(self, parts: list[str], base: Path) -> str | None:
        p = base.joinpath(*parts)
        if p not in self._files:
            self._files[p] = None
            for cand in (p.with_suffix(".py"), p / "__init__.py"):
                if cand.is_file():
                    self._files[p] = cand.relative_to(self.root).as_posix()
                    break
        return self._files[p]

    def imports(self, rel: str, nodes: list[ast.AST] | None = None) -> list[tuple[str, str, str]]:
        """(module rel path, imported name or "", local alias) for every import this module makes
        that lands on a file in this checkout: ``orion.*`` from the root, a service's own
        packages (``app.x``) from its directory, relative imports from the file's package."""
        if nodes is None and rel in self._imports:
            return self._imports[rel]
        svc = _service_of(rel)
        here = (self.root / rel).parent
        out = []
        for node in nodes if nodes is not None else self.nodes(rel):
            if isinstance(node, ast.Import):
                for a in node.names:
                    m = self._resolve_module(a.name, 0, here, svc)
                    if m:
                        out.append((m, "", a.asname or a.name))
            elif isinstance(node, ast.ImportFrom):
                base = self._resolve_module(node.module or "", node.level, here, svc)
                for a in node.names:
                    sub = self._resolve_module(f"{node.module}.{a.name}" if node.module else a.name,
                                               node.level, here, svc)
                    if sub:  # from pkg import module
                        out.append((sub, "", a.asname or a.name))
                    elif base:
                        out.append((base, a.name, a.asname or a.name))
        if nodes is None:
            self._imports[rel] = out
        return out

    def light_starts(self, rel: str, text: str) -> list[tuple[str, str]]:
        """(shared module rel path, symbol) this module reaches through its ``orion`` imports,
        read from the text with no parse (the same answer ``imports`` + attributes give)."""
        out: list[tuple[str, str]] = []
        for mod, names in _FROM_RE.findall(text):
            for item in names.strip("()").split(","):
                item = item.split("#", 1)[0].strip()
                if not item or item == "*":
                    continue
                name, _, alias = item.partition(" as ")
                name = name.strip()
                sub = self._resolve_module(f"{mod}.{name}", 0, self.root, None)
                if sub:
                    alias = alias.strip() or name
                    out += [(sub, a) for a in set(re.findall(rf"\b{re.escape(alias)}\.(\w+)", text))]
                    out.append((sub, "*"))
                    continue
                base = self._resolve_module(mod, 0, self.root, None)
                if base:
                    out += [(base, name), (base, "*")]
        for mod, alias in _IMPORT_RE.findall(text):
            m = self._resolve_module(mod, 0, self.root, None)
            if m:
                out += [(m, a) for a in set(re.findall(rf"\b{re.escape(alias or mod)}\.(\w+)", text))]
                out.append((m, "*"))
        return out

    def _resolve_module(self, dotted: str, level: int, here: Path, svc: str | None) -> str | None:
        parts = [p for p in dotted.split(".") if p]
        if level:
            base = here
            for _ in range(level - 1):
                base = base.parent
            return self._file_of(parts, base) if parts else self._file_of(["__init__"], base)
        if not parts:
            return None
        if parts[0] == "orion":
            return self._file_of(parts, self.root)
        if svc is not None:
            return self._file_of(parts, self.root / "services" / svc)
        return None

    def module_const(self, rel: str, name: str, depth: int) -> str | None:
        """A module's top-level string constant (followed through its own imports)."""
        if depth > _RESOLVE_DEPTH:
            return None
        if rel not in self._consts:
            self._consts[rel] = {}
            tree = self.tree(rel)
            if tree is not None:
                top = [s for s in tree.body if isinstance(s, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom))]
                res = Resolver(self, rel, _walk_stmts(top), None)
                for n in list(res.local) + list(res.imported):
                    v = res.value(ast.Name(id=n), depth)
                    if v and v[0] == "str":
                        self._consts[rel][n] = v[1]
        return self._consts[rel].get(name)

    def settings_value(self, svc: str, attr: str) -> str | None:
        return self._settings_of(svc).get(attr.lower())

    def _settings_of(self, svc: str) -> dict[str, str]:
        if svc not in self._settings:
            vals: dict[str, str] = {}
            env = self.env.get(svc, {})
            for rel in self.service_modules.get(svc, []):
                if "Settings" not in self.text(rel):
                    continue
                tree = self.tree(rel)
                if tree is None:
                    continue
                for name, (alias, default) in _settings_fields(tree).items():
                    v = env.get(alias) if alias else None
                    v = v if v is not None else env.get(name.upper(), default)
                    if v:
                        vals[name] = v
            self._settings[svc] = vals
        return self._settings[svc]

    # shared modules: acts and dependencies per top-level symbol, parsed only when reached
    def _shared(self, rel: str) -> dict[str, tuple[set[Act], set[tuple[str, str]], set[str]]]:
        if rel in self._shared_cache:
            return self._shared_cache[rel]
        self._shared_cache[rel] = {}
        tree = self.tree(rel)
        if tree is None:
            return {}
        http_t, redis_t = _relevant(self.text(rel))
        res = Resolver(self, rel, self.nodes(rel), None)
        out: dict[str, tuple[set[Act], set[tuple[str, str]], set[str]]] = {}
        symbols = _symbols(tree)
        for sym, mod in symbols.items():
            nodes = _walk_stmts(mod.body)
            acts = module_acts(nodes, res, redis_text=redis_t) if (http_t or redis_t) else set()
            used = {n.id for n in nodes if isinstance(n, ast.Name)}
            deps: set[tuple[str, str]] = set()
            for alias, (mrel, name) in res.imported.items():
                if alias in used:
                    deps.add((mrel, name))
            for n in nodes:
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in res.modules:
                    deps.add((res.modules[n.value.id], n.attr))
            # a function-local import (``from m import K`` inside a def) is used where it sits
            for mrel, name, _alias in self.imports(rel, nodes):
                deps.add((mrel, name or "*"))
            deps |= {(rel, o) for o in symbols if o != sym and o != "*" and o in used}
            if "*" in symbols and sym != "*":
                deps.add((rel, "*"))
            attrs = {a.lower() for n in nodes if (a := _attr_read(n))}
            out[sym] = (acts, deps, attrs)
        self._shared_cache[rel] = out
        return out

    def _module_deps(self, rel: str) -> set[str]:
        """Every shared module this one imports, from the text (an over-estimate: used only to
        prune modules that cannot reach any act, never to attribute one)."""
        text = self.text(rel)
        here = (self.root / rel).parent
        out: set[str] = set()
        for dots, mod, names in _ANY_FROM_RE.findall(text):
            if not dots and not mod.startswith("orion"):
                continue
            base = self._resolve_module(mod, len(dots), here, None)
            if base:
                out.add(base)
            for item in names.strip("()").split(","):
                name = item.split("#", 1)[0].split(" as ")[0].strip()
                if name and name != "*":
                    sub = self._resolve_module(f"{mod}.{name}" if mod else name, len(dots), here, None)
                    if sub:
                        out.add(sub)
        for mod, _alias in _IMPORT_RE.findall(text):
            m = self._resolve_module(mod, 0, self.root, None)
            if m:
                out.add(m)
        return {m for m in out if _service_of(m) is None}

    def _prune(self, shared: list[str]) -> set[str]:
        """Shared modules that may reach an act: their own text may hold one, or they import
        (transitively) one that may."""
        rdeps: dict[str, set[str]] = defaultdict(set)
        live = set()
        for rel in shared:
            http_t, redis_t = _relevant(self.text(rel))
            if http_t or redis_t:
                live.add(rel)
            for d in self._module_deps(rel):
                rdeps[d].add(rel)
        stack = list(live)
        while stack:
            for up in rdeps.get(stack.pop(), ()):
                if up not in live:
                    live.add(up)
                    stack.append(up)
        return live

    def _closure(self, start: tuple[str, str]) -> tuple[set[tuple[tuple[str, str], Act]], set[str]]:
        """Every act of the shared symbols ``start`` reaches, and the attribute names they read."""
        if start in self._closed:
            return self._closed[start]
        seen, stack, out, attrs = {start}, [start], set(), set()
        while stack:
            rel, sym = stack.pop()
            if _service_of(rel) is not None or rel not in self._may_act:
                continue  # a service's own module (scanned as that service), or one that reaches no act
            entry = self._shared(rel).get(sym)
            if entry is None:
                continue
            acts, deps, read = entry
            out |= {((rel, sym), a) for a in acts}
            attrs |= read
            for nxt in deps:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        self._closed[start] = (out, attrs)
        return self._closed[start]

    # the scan
    def run(self) -> WireRefs:
        self._shared_cache: dict[str, dict[str, tuple[set[Act], set[tuple[str, str]], set[str]]]] = {}
        self._closed: dict[tuple[str, str], tuple[set, set[str]]] = {}
        shared = []
        for p in _py_files(self.root):
            rel = p.relative_to(self.root).as_posix()
            svc = _service_of(rel)
            if svc is not None:
                self.service_modules[svc].append(rel)
            else:
                shared.append(rel)
        self._may_act = self._prune(shared)

        http: dict[tuple[str, str], WireRef] = {}
        redis: dict[str, dict[tuple[str, str], WireRef]] = defaultdict(dict)
        dynamic: dict[str, set[str]] = defaultdict(set)
        reached: set[tuple[str, str]] = set()

        def add(svc: str, act: Act, via: str) -> None:
            env = self.env.get(svc, {})
            kind, value = act.kind, act.value
            if kind == "env-http":
                v = env.get(value)
                if not v or not is_url(v):
                    return
                kind, value = "http", v
            elif kind == "env-redis":
                v = env.get(value)
                if not v:
                    dynamic[svc].add(via)
                    return
                kind, value = "redis", v
            if kind == "http":
                http.setdefault((svc, value), WireRef(svc, value, "calls", via))
            elif kind == "redis":
                redis[value].setdefault((svc, act.relation), WireRef(svc, value, act.relation, via))
            elif kind == "dynamic":
                dynamic[svc].add(via)

        scanned = 0
        for svc, rels in sorted(self.service_modules.items()):
            url_fields = [k for k in self._settings_of(svc) if is_url(self._settings[svc][k])]
            pending: list[tuple[Act, str]] = []
            attrs: set[str] = set()
            for rel in rels:
                text = self.text(rel)
                http_t, redis_t = _relevant(text, url_fields)
                starts: list[tuple[str, str]] = []
                if http_t or redis_t:
                    nodes = self.nodes(rel)
                    if not nodes:
                        continue
                    scanned += 1
                    pending += [(act, rel) for act in module_acts(nodes, Resolver(self, rel, nodes, svc),
                                                                  redis_text=redis_t)]
                    for mrel, name, alias in self.imports(rel):
                        if _service_of(mrel) is not None:
                            continue
                        if name:
                            starts.append((mrel, name))
                        else:  # the module itself: the attributes read off it
                            starts += [(mrel, n.attr) for n in nodes
                                       if isinstance(n, ast.Attribute) and _path(n.value) == alias]
                        starts.append((mrel, "*"))  # import-time code runs for any importer
                elif "orion" in text:
                    starts = self.light_starts(rel, text)
                attrs |= {(a or b).lower() for a, b in _ATTR_RE.findall(text)}
                for start in sorted(set(starts)):
                    acts, read = self._closure(start)
                    attrs |= read
                    for (srel, sym), act in sorted(acts, key=lambda x: (x[0], x[1].kind, x[1].value, x[1].relation)):
                        reached.add((srel, sym))
                        pending.append((act, f"{srel}:{sym} (imported by {svc})"))
            for act, via in pending:
                if act.kind == "field-http":  # a class field's URL: only where some code reads the field
                    if act.relation in attrs:
                        add(svc, Act("http", act.value), via)
                else:
                    add(svc, act, via)
        self.result.modules_scanned = scanned + len(self._shared_cache)

        self.result.http = sorted(http.values(), key=lambda r: (r.service, r.value))
        self.result.redis = {k: sorted(v.values(), key=lambda r: (r.service, r.relation))
                             for k, v in sorted(redis.items())}
        self.result.dynamic = {s: sorted(v) for s, v in sorted(dynamic.items())}
        self.result.unattributed = sorted({rel for rel, syms in self._shared_cache.items()
                                           for sym, (acts, _, _) in syms.items()
                                           if (rel, sym) not in reached
                                           and any(a.kind in ("http", "field-http", "redis") for a in acts)})
        return self.result


def code_wire_refs(root: Path) -> WireRefs:
    return _Scan(root).run()
