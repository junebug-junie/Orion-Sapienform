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
3. **Direction.** Inside a module that names the store, names bound from it are followed
   (``col = client.get_or_create_collection(NAME)``; ``self._col = ...``). A call on such a
   handle, or a call taking it as an argument, is a write when its verb is a write verb
   (``upsert``, ``add``, ``_publish_semantic_upsert``), a read when it is a read verb
   (``query``, ``get``, ``search``). A Cypher text decides a FalkorDB ``query``: ``MERGE`` /
   ``CREATE`` / ``SET`` / ``DELETE`` write, ``MATCH`` alone reads. ``read_only=True`` on the
   call reads. Anything else is ``uses``: the code names the store, the call does not say how.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

# The atlas names stores in its own docs and gaps; the observer is not a part of what it sees.
_SKIP_PREFIXES = ("orion/body_atlas/", "orion/dig/")
_SKIP_DIRS = {"tests", "evals", "__pycache__", "node_modules", ".venv", "venv"}

# Verb tokens of a call name (``_publish_semantic_upsert`` -> publish, semantic, upsert).
# "store" is left out: in this repo it is a noun far more than a verb (build_store, get_store).
_WRITE = {"add", "upsert", "insert", "update", "delete", "merge", "create", "write", "put",
          "persist", "index", "save"}
_READ = {"query", "get", "search", "read", "fetch", "count", "peek", "find", "match", "select",
         "lookup", "load", "recall"}
_CYPHER_WRITE = re.compile(r"\b(MERGE|CREATE|SET|DELETE|DETACH|REMOVE)\b", re.I)
_CYPHER_READ = re.compile(r"\bMATCH\b", re.I)
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*['\"]?([^'\"#\s]*)['\"]?\s*(#.*)?$")
_COMPOSE_DEFAULT = re.compile(r"\$\{([A-Z][A-Z0-9_]*):?-([^}]*)\}")

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
    name builds an object (``SearchConfig(collection=...)``): a constructor is not an act."""
    if name[:1].isupper():
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
        data = yaml.safe_load(text) or {}
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
    """service -> {env key: store name} from its ``.env_example`` and compose files, keeping only
    keys whose value (or compose ``${KEY:-default}``) is exactly a store name."""
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

def _label(node: ast.AST) -> str | None:
    """The label a node spells: a string constant, a name, or an attribute's last name."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for e in target.elts for n in _target_names(e)]
    return []


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _strings(call: ast.Call) -> str:
    parts = []
    for a in [*call.args, *(k.value for k in call.keywords)]:
        for n in ast.walk(a):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                parts.append(n.value)
    return " ".join(parts)


def _index(tree: ast.AST) -> tuple[list[ast.AST], dict[int, tuple[ast.AST, str]]]:
    """Every node once, and child id -> (parent, the parent field holding the child). Cached on
    the tree: one walk per module, however many times it is scanned."""
    cached = getattr(tree, "_atlas_index", None)
    if cached is None:
        nodes = list(ast.walk(tree))
        parent: dict[int, tuple[ast.AST, str]] = {}
        for node in nodes:
            for fname, value in ast.iter_fields(node):
                for child in value if isinstance(value, list) else [value]:
                    if isinstance(child, ast.AST):
                        parent[id(child)] = (node, fname)
        cached = tree._atlas_index = (nodes, parent)  # type: ignore[attr-defined]
    return cached


def _ancestors(node: ast.AST, parent: dict[int, tuple[ast.AST, str]]):
    """(ancestor, field of the ancestor the path came through, the child on the path)."""
    while id(node) in parent:
        up, fname = parent[id(node)]
        yield up, fname, node
        node = up


def scan_module(tree: ast.AST, anchors: dict[str, str], text: str | None = None) -> dict[str, str]:
    """``anchors``: label -> store (a literal store name, an env key, an imported constant).
    Returns store -> relation for every store the module names. ``text`` (the source) lets a
    module that spells no anchor skip the walk."""
    if text is not None and not any(a in text for a in anchors):
        return {}
    nodes, parent = _index(tree)
    labelled = [(n, lab) for n in nodes if (lab := _label(n)) is not None]
    named = {anchors[lab] for _, lab in labelled if lab in anchors}
    if not named:
        return {}
    # label -> stores it stands for; follow bindings to a fixpoint:
    # X = <expr naming a store>; self.y = X; z = client.f(y)
    label_stores: dict[str, set[str]] = defaultdict(set)
    for label, store in anchors.items():
        label_stores[label].add(store)
    for _ in range(4):
        grew = False
        for n, lab in labelled:
            if lab not in label_stores:
                continue
            for up, fname, _child in _ancestors(n, parent):
                if isinstance(up, (ast.Assign, ast.AnnAssign)) and fname == "value":
                    targets = up.targets if isinstance(up, ast.Assign) else [up.target]
                    for t in (t for tg in targets for t in _target_names(tg)):
                        if label_stores[lab] - label_stores[t]:
                            label_stores[t] |= label_stores[lab]
                            grew = True
                    break
                if isinstance(up, ast.stmt):
                    break
        if not grew:
            break

    rel: dict[str, str] = {s: "uses" for s in named}
    # call id -> (call, stores reached through its receiver, stores reached through its args)
    calls: dict[int, tuple[ast.Call, set[str], set[str]]] = {}
    for n, lab in labelled:
        if lab not in label_stores:
            continue
        for up, fname, child in _ancestors(n, parent):
            if isinstance(up, ast.stmt):
                break
            if not isinstance(up, ast.Call):
                continue
            entry = calls.setdefault(id(up), (up, set(), set()))
            if fname == "func":
                # x.upsert(...): x is the handle; the callee's own name is not a reference
                if child is n:
                    continue
                entry[1].update(label_stores[lab])
            else:
                entry[2].update(label_stores[lab])

    def bump(stores: set[str], r: str | None) -> None:
        for s in stores if r else ():
            if RANK[r] > RANK[rel.get(s, "uses")]:
                rel[s] = r

    for call, on_handle, in_args in calls.values():
        stores = on_handle | in_args
        if not stores:
            continue
        if any(k.arg == "read_only" and isinstance(k.value, ast.Constant) and k.value.value is True
               for k in call.keywords):
            bump(stores, "reads")
            continue
        name = _call_name(call.func)
        r = verb_relation(name)
        if name.lower().endswith("query"):  # query / ro_query / graph_query: the Cypher decides
            r = cypher_relation(_strings(call)) or r
        bump(stores, r)
    return rel


# --- repo scan ----------------------------------------------------------------------------------

def _py_files(root: Path) -> Iterable[Path]:
    for base in (root / "services", root / "orion"):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if rel.startswith(_SKIP_PREFIXES) or _SKIP_DIRS & set(p.relative_to(root).parts[:-1]):
                continue
            if p.name.startswith("test_") or p.name == "conftest.py":
                continue
            yield p


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
    for node in _index(tree)[0]:
        if isinstance(node, ast.Attribute) and ast.unparse(node.value) == alias:
            out.add(node.attr)
    return out


def _symbols(tree: ast.AST) -> dict[str, ast.Module]:
    """Top-level name -> the statement defining it, as a scannable module. Statements that bind
    no name (``register(...)`` at import time) run for any importer: symbol ``*``."""
    out: dict[str, list[ast.stmt]] = defaultdict(list)
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [stmt.name]
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = [n for t in targets for n in _target_names(t)] or ["*"]
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        else:
            names = ["*"]
        for n in names:
            out[n].append(stmt)
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
                    picked = _attrs_on(tree, alias) or set(symbols)
                used[dotted][svc] |= {sym for sym in picked if sym in symbols} | ({"*"} & set(symbols))
        for store, relation in scan_module(tree, anchors, texts[rel]).items():
            add(store, svc, relation, rel)
    result.modules_scanned = scanned

    # phase 3: each used shared symbol, its stores to the importing service; the importer's env
    # resolves an env key the symbol reads
    for dotted, (rel, symbols, consts) in shared.items():
        reached: set[str] = set()
        for svc, picked in sorted(used.get(dotted, {}).items()):
            for sym in sorted(picked):
                anchors = {**literal, **{c: v for c, v in consts.items()}, **env_keys.get(svc, {})}
                for store, relation in scan_module(symbols[sym], anchors).items():
                    add(store, svc, relation, f"{rel}:{sym} (imported by {svc})")
                    reached.add(store)
        for sym, mod in symbols.items():
            for store in scan_module(mod, {**literal, **consts}):
                if store not in reached:
                    result.unattributed.setdefault(store, [])
                    if rel not in result.unattributed[store]:
                        result.unattributed[store].append(rel)

    result.refs = {s: sorted(v.values(), key=lambda r: r.service) for s, v in sorted(per.items())}
    # a store some service reaches is attributed, whatever other modules also name it
    result.unattributed = {s: sorted(v) for s, v in sorted(result.unattributed.items()) if s not in result.refs}
    return result
