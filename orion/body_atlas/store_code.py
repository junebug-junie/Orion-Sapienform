"""Who writes or reads a Chroma collection or FalkorDB graph, from the code that names it.

A store's ``.env_example`` link (``declared_env``) only exists when a service puts the store's
name in an env key. Many stores are named in code instead: ``SOCIAL_ROOM_COLLECTION =
"orion_social_room"``, a pydantic ``Field(default="orion_reading_results")``, a shared module's
constant another service imports. This module finds those by rule, never from a hand-typed
store -> service list:

1. **Naming.** A Python module names a store when its AST holds a string constant exactly equal
   to the store's name (``"orion_kg"`` never matches ``"orion_kg_v2"`` or a docstring that only
   mentions it inside a sentence), or when it reads an env key whose value, in that service's
   ``.env_example`` / ``docker-compose*.yml``, is the store's name (``os.getenv("FOO_GRAPH")``,
   ``settings.FOO_GRAPH``, ``Field(alias="FOO_GRAPH")``).
2. **Service.** ``services/<svc>/...`` is ``<svc>``. A shared ``orion/...`` module is attributed
   to every service whose own code imports it (one hop). A shared module no service imports
   directly is reported (``unattributed``), never guessed. ``tests/`` and ``evals/`` are not
   the running body and are skipped; so is the atlas itself (``orion/body_atlas``, ``orion/dig``).
3. **Direction.** Claimed only from three plain shapes; anything else is ``uses`` (the code
   names the store, the call does not say how). A handle is a name bound from the store itself
   (``col = client.get_collection(NAME)``, ``self._col = col``), never from an expression that
   merely contains it. (a) A method called on a handle: ``col.upsert`` writes, ``col.query``
   reads, and a ``query`` call's first argument decides when it is Cypher (``MERGE`` / ``CREATE``
   / ``SET`` / ``DELETE`` write, ``MATCH`` alone reads). (b) A plain function taking the store
   directly and named for its act: ``_publish_semantic_upsert(collection_name=NAME)`` writes.
   (c) ``read_only=True`` on a call taking the store directly reads. A method of any other
   object (``seen.add(NAME)``, ``os.environ.get(KEY, NAME)``, ``parser.add_argument``) is not
   the store's act, and a module run by hand (``if __name__ == "__main__":``) only names.
"""
from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # the C loader when built in

# The atlas names stores in its own docs and gaps; the observer is not a part of what it sees.
_SKIP_PREFIXES = ("orion/body_atlas/", "orion/dig/")
_SKIP_DIRS = {"tests", "evals", "__pycache__", "node_modules", ".venv", "venv"}

# Verb tokens of a call name (``_publish_semantic_upsert`` -> publish, semantic, upsert).
# "store" is left out: in this repo it is a noun far more than a verb (build_store, get_store).
_WRITE = {"add", "upsert", "insert", "update", "delete", "merge", "create", "write", "put",
          "persist", "index", "save"}
_READ = {"query", "get", "search", "read", "fetch", "count", "peek", "find", "match", "select",
         "lookup", "load", "recall"}
_CYPHER_WRITE = re.compile(r"\b(MERGE|CREATE|SET|DELETE|DETACH|REMOVE)\b")  # upper case: Cypher, not English
_CYPHER_READ = re.compile(r"\bMATCH\b")
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*['\"]?([^'\"#\s]*)['\"]?\s*(#.*)?$")
_COMPOSE_DEFAULT = re.compile(r"\$\{([A-Z][A-Z0-9_]*):?-([^}]*)\}")

# a call naming one of these returns the store object (a handle), it does not act on it
_HANDLE = {"collection", "collections", "graph", "client"}

RANK = {"writes": 2, "reads": 1, "uses": 0}


@dataclass
class StoreRef:
    service: str
    relation: str  # writes / reads / uses
    via: str  # repo-relative module path that names the store (shared: "orion/x.py <- svc/y.py")


@dataclass
class CodeRefs:
    refs: dict[str, list[StoreRef]] = field(default_factory=dict)  # store -> refs, one per service
    # store -> shared modules that name it but that no service imports directly
    unattributed: dict[str, list[str]] = field(default_factory=dict)
    modules_scanned: int = 0
    skipped: list[str] = field(default_factory=list)  # files that did not parse


def _tokens(name: str) -> set[str]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return {t for t in snake.split("_") if t}


def verb_relation(name: str) -> str | None:
    """writes / reads from a call's own name; None when it says neither or both. A CapWords
    name builds an object (``SearchConfig(collection=...)``): a constructor is not an act. A
    name about the store object itself (``get_collection``, ``select_graph``) binds a handle."""
    if name[:1].isupper():
        return None
    if _tokens(name) & _HANDLE:
        return None
    toks = _tokens(name)
    w, r = bool(toks & _WRITE), bool(toks & _READ)
    if w and not r:
        return "writes"
    if r and not w:
        return "reads"
    return None


def cypher_relation(text: str) -> str | None:
    if _CYPHER_WRITE.search(text):
        return "writes"
    if _CYPHER_READ.search(text):
        return "reads"
    return None


# --- env-key indirection ------------------------------------------------------------------------

def _compose_pairs(text: str) -> Iterable[tuple[str, str]]:
    for key, default in _COMPOSE_DEFAULT.findall(text):
        yield key, default.strip().strip("'\"")
    try:
        data = yaml.load(text, Loader=_YAML_LOADER) or {}
    except yaml.YAMLError:
        return
    for spec in (data.get("services") or {}).values() if isinstance(data, dict) else ():
        env = (spec or {}).get("environment") if isinstance(spec, dict) else None
        if isinstance(env, dict):
            for k, v in env.items():
                if isinstance(v, str):
                    yield str(k), v.strip()
        elif isinstance(env, list):
            for item in env:
                if isinstance(item, str) and "=" in item:
                    k, v = item.split("=", 1)
                    yield k.strip(), v.strip()


def service_env_keys(root: Path, names: set[str]) -> dict[str, dict[str, str]]:
    """service -> {env key: store name} from its ``.env_example``, then its compose files (a key
    the ``.env_example`` already sets keeps that value), keeping only keys whose value (or
    compose ``${KEY:-default}``) is exactly a store name."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for svc_dir in sorted(p for p in (root / "services").glob("*") if p.is_dir()):
        env = svc_dir / ".env_example"
        if env.is_file():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _ENV_LINE.match(line)
                if m and m.group(2) in names:
                    out[svc_dir.name][m.group(1)] = m.group(2)
        for comp in sorted(svc_dir.glob("docker-compose*.y*ml")):
            for k, v in _compose_pairs(comp.read_text(encoding="utf-8", errors="replace")):
                if v in names:
                    out[svc_dir.name].setdefault(k, v)
    return dict(out)


# --- one module ---------------------------------------------------------------------------------

def _anchor_label(node: ast.AST) -> str | None:
    """What an anchor can match: a string constant, a bare name, an attribute's last name
    (``settings.FOO_GRAPH`` reads env key ``FOO_GRAPH``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _path(node: ast.AST) -> str | None:
    """A bound handle's full spelling: ``col``, ``self._col``. Never just the last name, so
    ``p.name`` is not ``name``."""
    if isinstance(node, (ast.Name, ast.Attribute)):
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001 - an odd node is just not a handle
            return None
    return None


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for e in target.elts for n in _target_names(e)]
    p = _path(target)
    return [p] if p else []


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _index(tree: ast.AST) -> list[ast.AST]:
    """Every node once, cached on the tree: one walk per module, however often it is scanned."""
    cached = getattr(tree, "_atlas_nodes", None)
    if cached is None:
        cached = tree._atlas_nodes = list(ast.walk(tree))  # type: ignore[attr-defined]
    return cached


def _is_main_guard(stmt: ast.stmt) -> bool:
    t = stmt.test if isinstance(stmt, ast.If) else None
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators))


def is_cli_script(tree: ast.AST) -> bool:
    """A module with a top-level ``if __name__ == "__main__":`` is a script someone runs by hand
    (a migration, a diagnostic), not the running service: it may name a store, never act on it."""
    return any(_is_main_guard(s) for s in getattr(tree, "body", []))


class _Taint:
    """Which expressions stand for which stores, in one module."""

    def __init__(self, anchors: dict[str, str]) -> None:
        self.anchors = anchors  # literal / env key / imported constant -> store
        self.handles: dict[str, set[str]] = defaultdict(set)  # bound path -> stores

    def direct(self, expr: ast.AST | None) -> set[str]:
        """Stores an expression IS (not merely contains): an anchor, a bound handle, or a call
        made on / with one directly (``client.get_collection(NAME)``, ``os.getenv("KEY")``)."""
        if expr is None:
            return set()
        lab = _anchor_label(expr)
        if lab is not None and lab in self.anchors:
            return {self.anchors[lab]}
        path = _path(expr)
        if path is not None and path in self.handles:
            return set(self.handles[path])
        if isinstance(expr, ast.Await):
            return self.direct(expr.value)
        if isinstance(expr, ast.Call) and _passes_store_through(expr):
            out: set[str] = set()
            if isinstance(expr.func, ast.Attribute):
                out |= self.direct(expr.func.value)
            for a in [*expr.args, *(k.value for k in expr.keywords)]:
                out |= self.direct(a)
            return out
        return set()


# calls that return the name they were given (``os.getenv("KEY")``, ``str(x).strip()``)
_VALUE_CALLS = {"getenv", "str", "strip"}


def _passes_store_through(call: ast.Call) -> bool:
    """Does this call's result still stand for the store? Yes for the store object or its name:
    a handle binder (``get_collection``, ``select_graph``, ``client``), a constructor
    (``RedisGraphQueryClient(graph_name=...)``), an env read or a string cleanup. No for an act:
    ``rows = g.query(...)`` and ``result = backend.ingest_episode(graph_name=G)`` are results,
    not the store."""
    name = _call_name(call.func)
    if name[:1].isupper() or _tokens(name) & _HANDLE or name in _VALUE_CALLS:
        return True
    # os.environ.get("KEY", default)
    return name == "get" and isinstance(call.func, ast.Attribute) and _path(call.func.value) == "os.environ"


def _first_text(call: ast.Call, consts: dict[str, str]) -> str:
    """The query text: the first positional argument, a literal or a module-level constant."""
    if not call.args:
        return ""
    a = call.args[0]
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return a.value
    if isinstance(a, ast.JoinedStr):
        return "".join(v.value for v in a.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    if isinstance(a, ast.Name):
        return consts.get(a.id, "")
    return ""


def _str_consts(tree: ast.AST) -> dict[str, str]:
    out = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
    return out


def scan_module(tree: ast.AST, anchors: dict[str, str], text: str | None = None) -> dict[str, str]:
    """``anchors``: label -> store (a literal store name, an env key, an imported constant).
    Returns store -> relation for every store the module names.

    Direction is claimed only from three plain shapes, everything else is ``uses``:
    (a) a method called on a handle bound from the store (``col.upsert(...)``,
    ``g.query("MERGE ...")``); (b) a plain function (not a method of some other object) taking
    the store directly as an argument and named for its act (``_publish_semantic_upsert(
    collection_name=NAME)``); (c) ``read_only=True`` on a call taking the store directly."""
    if text is not None and not any(a in text for a in anchors):
        return {}
    nodes = _index(tree)
    named = {anchors[lab] for n in nodes if (lab := _anchor_label(n)) is not None and lab in anchors}
    if not named:
        return {}
    taint = _Taint(anchors)
    # bind handles to a fixpoint: X = NAME; col = client.get_collection(X); self._col = col
    for _ in range(4):
        grew = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            stores = taint.direct(node.value)
            if not stores:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in (t for tg in targets for t in _target_names(tg)):
                if stores - taint.handles[t]:
                    taint.handles[t] |= stores
                    grew = True
        if not grew:
            break

    rel: dict[str, str] = {s: "uses" for s in named}
    if is_cli_script(tree):
        return rel
    consts = _str_consts(tree)

    def bump(stores: set[str], r: str | None) -> None:
        for s in stores if r else ():
            if RANK[r] > RANK[rel.get(s, "uses")]:
                rel[s] = r

    for call in nodes:
        if not isinstance(call, ast.Call):
            continue
        args = set()
        for a in [*call.args, *(k.value for k in call.keywords)]:
            args |= taint.direct(a)
        if args and any(k.arg == "read_only" and isinstance(k.value, ast.Constant) and k.value.value is True
                        for k in call.keywords):
            bump(args, "reads")  # (c)
            continue
        name = _call_name(call.func)
        if isinstance(call.func, ast.Attribute):
            on = taint.direct(call.func.value)
            if not on:
                continue  # a method of some other object (a dict, asyncio, a parser): not the store's act
            r = verb_relation(name)
            if name.lower().endswith("query"):  # (a) query / ro_query: the Cypher decides
                r = cypher_relation(_first_text(call, consts)) or r
            bump(on, r)
        elif isinstance(call.func, ast.Name) and args and len(_tokens(name)) > 1:
            bump(args, verb_relation(name))  # (b)
    return rel


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


def _module_dotted(rel: str) -> str:
    mod = rel[:-3].replace("/", ".")
    return mod[: -len(".__init__")] if mod.endswith(".__init__") else mod


def _imports(tree: ast.AST) -> list[tuple[str, list[tuple[str, str]]]]:
    """(module, [(imported name, local alias)]) for every import. ``import a.b as m`` ->
    ("a.b", [("", "m")]); ``import a.b`` -> ("a.b", [("", "a.b")])."""
    out = []
    stack = [tree]
    while stack:  # statements only: an import is a statement, so expressions are never walked
        node = stack.pop()
        for fname in ("body", "orelse", "finalbody", "handlers", "cases"):
            stack.extend(c for c in getattr(node, fname, ()) or ()
                         if isinstance(c, (ast.stmt, ast.excepthandler, ast.match_case)))
        if isinstance(node, ast.Import):
            out += [(a.name, [("", a.asname or a.name)]) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.append((node.module, [(a.name, a.asname or a.name) for a in node.names]))
    return out


def _attrs_on(tree: ast.AST, alias: str) -> set[str]:
    """Attribute names read off a module alias: ``fs.build_x()`` -> {"build_x"}. A dotted alias
    (``import orion.a.b``) matches ``orion.a.b.build_x``."""
    out = set()
    for node in _index(tree):
        if isinstance(node, ast.Attribute) and _path(node.value) == alias:
            out.add(node.attr)
    return out


def _symbols(tree: ast.AST) -> dict[str, ast.Module]:
    """Top-level name -> the statements defining it, as a scannable module. A top-level
    ``if`` / ``try`` is opened to find the names it defines. Statements that bind no name
    (``register(...)`` at import time) run for any importer: symbol ``*``. The
    ``if __name__ == "__main__":`` block runs for nobody who imports the module."""
    out: dict[str, list[ast.stmt]] = defaultdict(list)

    def visit(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [stmt.name]
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                names = [n for t in targets for n in _target_names(t)] or ["*"]
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)) or _is_main_guard(stmt):
                continue
            elif isinstance(stmt, (ast.If, ast.Try)):
                visit([*stmt.body, *stmt.orelse, *getattr(stmt, "finalbody", []),
                       *(x for h in getattr(stmt, "handlers", []) for x in h.body)])
                continue
            else:
                names = ["*"]
            for n in names:
                out[n].append(stmt)

    visit(list(getattr(tree, "body", [])))
    return {n: ast.Module(body=stmts, type_ignores=[]) for n, stmts in out.items()}


def _parse(root: Path, p: Path, result: CodeRefs) -> ast.AST | None:
    try:
        return ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        result.skipped.append(p.relative_to(root).as_posix())
        return None


def code_store_refs(root: Path, names: Iterable[str]) -> CodeRefs:
    wanted = {n for n in names if n}
    result = CodeRefs()
    if not wanted:
        return result
    env_keys = service_env_keys(root, wanted)
    all_keys = {k for m in env_keys.values() for k in m}
    files = [(p, p.relative_to(root).as_posix()) for p in _py_files(root)]
    texts = {rel: p.read_text(encoding="utf-8", errors="replace") for p, rel in files}

    def mentions(text: str, needles: Iterable[str]) -> bool:
        return any(n in text for n in needles)

    # phase 1: shared orion/ modules that name a store, by literal or by an env key, split into
    # top-level symbols: an importer gets the stores of the symbols it uses, not the whole file
    literal = {n: n for n in wanted}
    shared: dict[str, tuple[str, dict[str, ast.Module], dict[str, str]]] = {}  # dotted -> (rel, symbols, consts)
    for p, rel in files:
        if _service_of(rel) is not None or not mentions(texts[rel], wanted | all_keys):
            continue
        tree = _parse(root, p, result)
        if tree is None:
            continue
        consts = {
            t.id: node.value.value
            for node in getattr(tree, "body", [])
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Constant)
            and node.value.value in wanted
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]) if isinstance(t, ast.Name)
        }
        shared[_module_dotted(rel)] = (rel, _symbols(tree), consts)
    # ``import orion.a.b`` / ``from orion.a.b import x`` spell the dotted path;
    # ``from orion.a import b`` spells the parent path and the leaf
    spellings = [(d, *d.rsplit(".", 1)) for d in shared]

    def imports_shared(text: str) -> bool:
        return any(d in text or (parent in text and leaf in text) for d, parent, leaf in spellings)

    per: dict[str, dict[str, StoreRef]] = defaultdict(dict)  # store -> service -> strongest ref

    def add(store: str, svc: str, relation: str, via: str) -> None:
        cur = per[store].get(svc)
        if cur is None or RANK[relation] > RANK[cur.relation]:
            per[store][svc] = StoreRef(svc, relation, via)

    # phase 2: service modules; each also says which shared symbols its service uses
    used: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # dotted -> svc -> symbols
    scanned = len(shared)
    for p, rel in files:
        svc = _service_of(rel)
        if svc is None:
            continue
        keys = env_keys.get(svc, {})
        if not (mentions(texts[rel], wanted | set(keys)) or imports_shared(texts[rel])):
            continue
        tree = _parse(root, p, result)
        if tree is None:
            continue
        scanned += 1
        anchors = {**literal, **keys}
        for mod, imported in _imports(tree):
            for name, alias in imported:
                dotted = f"{mod}.{name}" if name and f"{mod}.{name}" in shared else mod
                if dotted not in shared:
                    continue
                _rel, symbols, consts = shared[dotted]
                if dotted == mod and name:  # from orion.a.b import sym
                    picked = {name}
                    if name in consts:
                        anchors[alias] = consts[name]
                else:  # the module itself: the attributes this file reads off it
                    picked = _attrs_on(tree, alias)  # none read: only import-time code ("*") runs
                used[dotted][svc] |= {sym for sym in picked if sym in symbols} | ({"*"} & set(symbols))
        for store, relation in scan_module(tree, anchors, texts[rel]).items():
            add(store, svc, relation, rel)
    result.modules_scanned = scanned

    # phase 3: each used shared symbol, its stores to the importing service; the importer's env
    # resolves an env key the symbol reads. Literal scans are done once per symbol.
    sym_cache: dict[tuple[str, str], dict[str, str]] = {}
    for dotted, (rel, symbols, consts) in shared.items():
        base = {**literal, **consts}
        own = {sym: sym_cache.setdefault((dotted, sym), scan_module(mod, base)) for sym, mod in symbols.items()}
        reached: set[str] = set()
        for svc, picked in sorted(used.get(dotted, {}).items()):
            keys = env_keys.get(svc, {})
            for sym in sorted(picked):
                found = dict(own[sym])
                if keys:
                    for store, relation in scan_module(symbols[sym], {**base, **keys}).items():
                        if RANK[relation] >= RANK[found.get(store, "uses")]:
                            found[store] = relation
                for store, relation in found.items():
                    add(store, svc, relation, f"{rel}:{sym} (imported by {svc})")
                    reached.add(store)
        for found in own.values():
            for store in found:
                if store not in reached and rel not in result.unattributed.setdefault(store, []):
                    result.unattributed[store].append(rel)

    result.refs = {s: sorted(v.values(), key=lambda r: r.service) for s, v in sorted(per.items())}
    # a store some service reaches is attributed, whatever other modules also name it
    result.unattributed = {s: sorted(v) for s, v in sorted(result.unattributed.items()) if s not in result.refs}
    return result
