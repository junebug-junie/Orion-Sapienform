"""Find every cross-service strict schema mechanically, and compare the copies
running containers actually hold.

The failure shape (2026-09-20 FieldStateV1 incident): a pydantic model with
``extra="forbid"`` (own or inherited) that one service builds and writes -- a
DB row's JSON or a bus payload -- and a DIFFERENT service validates back. When
the writer ships a new field before the reader is rebuilt, the reader rejects
every row with ``extra_forbidden`` while its container reads "Up".

Discovery is one AST pass over ``orion/**`` and ``services/**`` (tests and
evals excluded):

- every pydantic class in ``orion/`` with its effective ``extra`` setting,
  fields, required fields and nested-model references, bases resolved through
  imports and re-exports;
- every call site that *reads* a model (``X.model_validate``/``_json``,
  ``parse_obj``/``parse_raw``, ``parse_obj_as(X, ..)``, ``TypeAdapter(X)``,
  ``X(**data)``) or *writes* one (``X(field=...)``, ``X.model_construct``);
- call sites in shared ``orion/`` modules are attributed to every service
  whose code imports that module (transitively through the ``orion`` package);
- ``orion/bus/channels.yaml`` producers of a channel whose ``schema_id``
  contains the model are writers too (the bus validates on publish only, so
  consumers are counted from their own validate call sites, not from the YAML).

A model nested inside another is read/written wherever its outer model is.

Output is a set of (schema file, writer service, reader services) triples.
A forbid model somebody reads but nobody mechanically writes is ``unresolved``
and must appear in ``DECLARED_WRITERS`` -- a gate test fails otherwise, so a
new strict schema cannot silently fall out of coverage.

Comparison (``compare_models``): the container's copy of each schema file is
parsed on the host (nothing from the repo executes in a container) and the
writer's fields are compared with the reader's. A writer field the reader's
forbid model lacks is the exact ``extra_forbidden`` break; a field the reader
requires that the writer lacks breaks any model. Non-forbid ("ignore") models
drop unknown fields silently instead -- reported as a lower-severity class.

Stdlib only.
"""

from __future__ import annotations

import ast
import collections
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

Ref = tuple[str, str]  # (module, class name)

VALIDATE_ATTRS = frozenset(
    {"model_validate", "model_validate_json", "model_validate_strings", "parse_obj", "parse_raw", "parse_file"}
)
CONSTRUCT_ATTRS = frozenset({"model_construct", "construct"})

#: ``orion.schemas.registry`` imports every schema for name lookup and ~60
#: services import it; importing it is not a validation path (bus validation
#: is publish-side and covered by channels.yaml producers).
TAINT_EXCLUDED_MODULES = frozenset({"orion.schemas.registry"})

LIB_HOPS = 1

# ---------------------------------------------------------------------------
# Declared facts for what code cannot tell us.
# ---------------------------------------------------------------------------

#: Forbid models some service validates but no service mechanically builds.
#: Key: ``module:Class`` or a repo-relative file (every such class in it).
#: Value: (writing service, why) -- or (None, why) when there is genuinely no
#: cross-service writer (a YAML/file the reader loads itself, LLM output the
#: reader parses, a browser request body). A declared writer is injected as if
#: code had found it, so the pair is checked live like any other.
#: ``test_every_unresolved_candidate_is_declared`` fails on a candidate that is
#: neither resolved nor listed; ``test_declared_writers_are_not_stale`` fails
#: on an entry that no longer matches anything unresolved.
DECLARED_WRITERS: dict[str, tuple[Optional[str], str]] = {
    # Cross-service writers code cannot see (validate-then-persist, HTTP API).
    "orion.memory_graph.dto:SuggestDraftV1": (
        "orion-memory-consolidation",
        "validates the LLM draft, insert_pending_draft() stores it; hub routes validate it back",
    ),
    "orion.schemas.resource_admission:CapacityPermitV1": (
        "orion-durable-runs",
        "built from a SQL row in orion.durable_admission.capacity, served by /capacity/acquire; capacity_client validates it",
    ),
    # No cross-service writer.
    "orion/attention/field_attention/policy.py": (None, "policy YAML the reader loads itself"),
    "orion.autonomy.models:CapabilityPolicyV1": (None, "capability policy YAML the reader loads itself"),
    "orion.autonomy.models:CapabilityPolicyRuleV1": (None, "capability policy YAML the reader loads itself"),
    "orion/consolidation/policy.py": (None, "policy YAML the reader loads itself"),
    "orion/execution_dispatch/policy.py": (None, "policy YAML the reader loads itself"),
    "orion/feedback/policy.py": (None, "policy YAML the reader loads itself"),
    "orion/policy/policy.py": (None, "policy YAML the reader loads itself"),
    "orion/proposals/policy.py": (None, "policy YAML the reader loads itself"),
    "orion/reverie/baseline.py": (None, "policy YAML the reader loads itself"),
    "orion/gpu_pool/config.py": (None, "pool YAML the reader loads itself"),
    "orion/schemas/world_pulse.py": (None, "source registry YAML the reader loads itself"),
    "orion.memory_graph.dto:CardProjectionDefaultsV1": (None, "hub request body / settings, not another service"),
    "orion/core/contracts/memory_cards.py": (None, "LLM output (cortex-orch) and Hub browser request bodies"),
    "orion/core/schemas/drives.py": (None, "orion:memory:drives:state has had no producer since the 2026-07-30 drive-pressure deletion"),
    "orion/schemas/actions/daily.py": (None, "LLM output parsed by orion-actions"),
    "orion/schemas/context_exec.py": (None, "LLM-produced artifacts parsed by orion-context-exec"),
    "orion/schemas/world_pulse_read.py": (None, "LLM output and the hub's own DB round trip"),
    "orion/schemas/telemetry/mood_arc.py": (None, "manifest file written by the offline fit script"),
    "orion/schemas/telemetry/phi_encoder.py": (None, "manifest file written by the offline fit script"),
    "orion/schemas/evidence_index.py": (
        None,
        "channels.yaml producer is '*' (external ingesters); no service in services/ publishes these kinds",
    ),
}


def _is_test_path(rel: Path) -> bool:
    return rel.name.startswith("test_") or rel.name == "conftest.py" or any(
        p in ("tests", "evals", "test", "fixtures") for p in rel.parts
    )


def _module_name(root: Path, path: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Per-file facts
# ---------------------------------------------------------------------------


@dataclass
class ClassInfo:
    module: str
    name: str
    path: str  # repo-relative
    base_exprs: list[str]  # dotted names as written
    own_extra: Optional[str]
    fields: tuple[str, ...]
    required: tuple[str, ...]
    ann_names: tuple[str, ...]  # dotted names referenced in field annotations
    # ``model_validator(mode="before")`` / ``root_validator(pre=True)``: may
    # strip or remap input keys before ``extra="forbid"`` sees them.
    before_validator: bool = False
    excluded: tuple[str, ...] = ()  # Field(exclude=True): never serialized

    @property
    def ref(self) -> Ref:
        return (self.module, self.name)


@dataclass
class FileFacts:
    module: str
    path: str
    package: str  # for relative imports
    # local name -> (module, attr) for ``from m import attr as name``;
    # attr None for ``import m as name``.
    imports: dict[str, tuple[str, Optional[str]]] = field(default_factory=dict)
    imported_modules: set[str] = field(default_factory=set)
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    toplevel: set[str] = field(default_factory=set)  # top-level def/class names
    # (dotted name as written, line, enclosing top-level def/class or None)
    reads: list[tuple[str, int, Optional[str]]] = field(default_factory=list)
    writes: list[tuple[str, int, Optional[str]]] = field(default_factory=list)
    # enclosing top-level def/class (None = module level) -> dotted names it
    # references; what lets a call site be attributed to the callers of the
    # function it sits in rather than to every importer of its module.
    refs: dict[Optional[str], set[str]] = field(default_factory=dict)


def _dotted(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return f"{head}.{node.attr}" if head else None
    return None


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and _dotted(node.value) in ("Extra", "pydantic.Extra"):
        return node.attr
    return None


def _extra_from_config_value(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg == "extra":
                return _const_str(kw.value)
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "extra":
                return _const_str(v)
    return None


def _ann_names(node: Optional[ast.AST]) -> set[str]:
    out: set[str] = set()
    if node is None:
        return out
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            try:
                out |= _ann_names(ast.parse(sub.value, mode="eval"))
            except SyntaxError:
                pass
        elif isinstance(sub, (ast.Name, ast.Attribute)):
            d = _dotted(sub)
            if d:
                out.add(d)
    return out


def _field_call(value: Optional[ast.AST]) -> Optional[ast.Call]:
    if isinstance(value, ast.Call) and (_dotted(value.func) or "").split(".")[-1] == "Field":
        return value
    return None


def _is_ellipsis(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def _field_call_required(call: ast.Call) -> bool:
    if call.args:
        return _is_ellipsis(call.args[0])
    for kw in call.keywords:
        if kw.arg == "default":
            return _is_ellipsis(kw.value)
        if kw.arg == "default_factory":
            return False
    return True


def _annotated_field(annotation: ast.AST) -> Optional[ast.Call]:
    """``Annotated[T, Field(...)]`` -> that Field call."""
    if isinstance(annotation, ast.Subscript) and (_dotted(annotation.value) or "").split(".")[-1] == "Annotated":
        sl = annotation.slice
        for el in sl.elts if isinstance(sl, ast.Tuple) else []:
            if _field_call(el) is not None:
                return el
    return None


def _is_required(value: Optional[ast.AST], annotation: Optional[ast.AST] = None) -> bool:
    if value is None:
        ann_field = _annotated_field(annotation) if annotation is not None else None
        return True if ann_field is None else _field_call_required(ann_field)
    call = _field_call(value)
    if call is not None:
        return _field_call_required(call)
    return _is_ellipsis(value)


def _is_excluded(value: Optional[ast.AST], annotation: Optional[ast.AST]) -> bool:
    call = _field_call(value) or (_annotated_field(annotation) if annotation is not None else None)
    return call is not None and any(
        kw.arg == "exclude" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in call.keywords
    )


def _is_before_validator(fn: ast.AST) -> bool:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        name = (_dotted(dec.func) or "").split(".")[-1]
        for kw in dec.keywords:
            if name == "model_validator" and kw.arg == "mode" and _const_str(kw.value) in ("before", "wrap"):
                return True
            if name == "root_validator" and kw.arg == "pre" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
    return False


def _class_info(node: ast.ClassDef, module: str, path: str) -> ClassInfo:
    extra: Optional[str] = None
    for kw in node.keywords:
        if kw.arg == "extra":
            extra = _const_str(kw.value)
    fields: list[str] = []
    required: list[str] = []
    excluded: list[str] = []
    ann: set[str] = set()
    before = False
    for stmt in node.body:
        if _is_before_validator(stmt):
            before = True
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == "model_config":
                    extra = _extra_from_config_value(stmt.value) or extra
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if name == "model_config":
                if stmt.value is not None:
                    extra = _extra_from_config_value(stmt.value) or extra
                continue
            if name.startswith("_") or "ClassVar" in ast.dump(stmt.annotation):
                continue
            fields.append(name)
            if _is_required(stmt.value, stmt.annotation):
                required.append(name)
            if _is_excluded(stmt.value, stmt.annotation):
                excluded.append(name)
            ann |= _ann_names(stmt.annotation)
        elif isinstance(stmt, ast.ClassDef) and stmt.name == "Config":
            for s in stmt.body:
                if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "extra" for t in s.targets):
                    extra = _const_str(s.value) or extra
    return ClassInfo(
        module=module,
        name=node.name,
        path=path,
        base_exprs=[d for d in (_dotted(b) for b in node.bases) if d],
        own_extra=extra,
        fields=tuple(fields),
        required=tuple(required),
        ann_names=tuple(sorted(ann)),
        before_validator=before,
        excluded=tuple(excluded),
    )


def parse_facts(text: str, module: str, path: str, *, is_package: bool = False) -> Optional[FileFacts]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    package = module if is_package else module.rpartition(".")[0]
    facts = FileFacts(module=module, path=path, package=package)

    def absolutize(mod: Optional[str], level: int) -> Optional[str]:
        if level == 0:
            return mod
        base = package.split(".") if package else []
        if level - 1 > len(base):
            return None
        base = base[: len(base) - (level - 1)]
        return ".".join([*base, mod] if mod else base) or None

    # Only module-level classes (also inside a module-level if/try, e.g. a
    # TYPE_CHECKING or ImportError guard) are importable by name; a nested
    # ``class Config`` must not overwrite a top-level model of the same name.
    guarded: set[int] = set()
    for top in tree.body:
        if isinstance(top, (ast.If, ast.Try)):
            blocks = [top.body, top.orelse, getattr(top, "finalbody", [])]
            blocks += [h.body for h in getattr(top, "handlers", [])]
            guarded.update(id(st) for blk in blocks for st in blk if isinstance(st, ast.ClassDef))

    def top_level_class(node: ast.ClassDef) -> bool:
        return id(node) in guarded

    for top in tree.body:
        enclosing: Optional[str] = None
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            enclosing = top.name
            facts.toplevel.add(top.name)
        refs = facts.refs.setdefault(enclosing, set())
        for node in ast.walk(top):
            if isinstance(node, ast.Import):
                for a in node.names:
                    facts.imported_modules.add(a.name)
                    if a.asname:
                        facts.imports[a.asname] = (a.name, None)
                    else:
                        head = a.name.split(".")[0]
                        facts.imports.setdefault(head, (head, None))
                continue
            if isinstance(node, ast.ImportFrom):
                mod = absolutize(node.module, node.level)
                if not mod:
                    continue
                facts.imported_modules.add(mod)
                for a in node.names:
                    if a.name == "*":
                        continue
                    facts.imported_modules.add(f"{mod}.{a.name}")
                    facts.imports[a.asname or a.name] = (mod, a.name)
                continue
            if isinstance(node, ast.ClassDef):
                if node is top or top_level_class(node):
                    facts.classes[node.name] = _class_info(node, module, path)
            elif isinstance(node, ast.Name):
                refs.add(node.id)
            elif isinstance(node, ast.Attribute):
                d = _dotted(node)
                if d:
                    refs.add(d)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = _dotted(func)
            if isinstance(func, ast.Attribute):
                target = _dotted(func.value)
                if target and func.attr in VALIDATE_ATTRS:
                    facts.reads.append((target, node.lineno, enclosing))
                    continue
                if target and func.attr in CONSTRUCT_ATTRS:
                    facts.writes.append((target, node.lineno, enclosing))
                    continue
                if func.attr in ("validate_python", "validate_json") and isinstance(func.value, ast.Call):
                    inner = func.value
                    if (_dotted(inner.func) or "").split(".")[-1] == "TypeAdapter" and inner.args:
                        for n in _ann_names(inner.args[0]):
                            facts.reads.append((n, node.lineno, enclosing))
                    continue
            if fname and fname.split(".")[-1] == "parse_obj_as" and node.args:
                for n in _ann_names(node.args[0]):
                    facts.reads.append((n, node.lineno, enclosing))
                continue
            if fname:
                only_splat = bool(node.keywords) and not node.args and all(kw.arg is None for kw in node.keywords)
                (facts.reads if only_splat else facts.writes).append((fname, node.lineno, enclosing))
    return facts


# ---------------------------------------------------------------------------
# Repo index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One schema file, one writer service, the other services that read the
    models in that file the writer writes."""

    path: str
    writer: str
    readers: tuple[str, ...]
    models: tuple[str, ...]  # class names in ``path``
    strict: bool  # at least one of ``models`` is extra="forbid"
    evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict, compare=False, hash=False)
    # reader service -> the models in ``path`` that reader actually reads
    reader_models: Mapping[str, tuple[str, ...]] = field(default_factory=dict, compare=False, hash=False)

    @property
    def label(self) -> str:
        return ",".join(self.models)


@dataclass(frozen=True)
class Unresolved:
    key: str  # module:Class
    path: str
    readers: tuple[str, ...]


@dataclass
class Discovery:
    candidates: list[Candidate]
    unresolved: list[Unresolved]
    strict_models: int
    # file -> repo-relative files the model signatures in it depend on
    # (inherited bases), so the live check fetches those too.
    dependency_files: dict[str, tuple[str, ...]]
    declared_used: set[str] = field(default_factory=set)
    index: Optional["RepoIndex"] = field(default=None, repr=False, compare=False)

    def strict(self) -> list[Candidate]:
        return [c for c in self.candidates if c.strict]

    def loose(self) -> list[Candidate]:
        return [c for c in self.candidates if not c.strict]


class RepoIndex:
    def __init__(self, repo_root: Path):
        self.root = Path(repo_root)
        self.orion: dict[str, FileFacts] = {}
        self.services: dict[str, list[FileFacts]] = {}
        for path in sorted((self.root / "orion").rglob("*.py")):
            rel = path.relative_to(self.root)
            if _is_test_path(rel):
                continue
            f = self._parse(path, _module_name(self.root, path), str(rel))
            if f:
                self.orion[f.module] = f
        svc_root = self.root / "services"
        for svc in sorted(p for p in svc_root.iterdir() if p.is_dir()) if svc_root.is_dir() else []:
            files = []
            for path in sorted(svc.rglob("*.py")):
                rel_svc = path.relative_to(svc)
                if _is_test_path(rel_svc) or "node_modules" in rel_svc.parts:
                    continue
                # Service-local module names are only used for relative-import
                # anchoring; they never collide with ``orion.*``.
                f = self._parse(
                    path, "svc." + _module_name(svc, path), str(path.relative_to(self.root)), needs_orion=True
                )
                if f:
                    files.append(f)
            if files:
                self.services[svc.name] = files
        self.classes: dict[Ref, ClassInfo] = {
            ci.ref: ci for f in self.orion.values() for ci in f.classes.values()
        }
        self._resolve_cache: dict[tuple[str, str], Optional[Ref]] = {}
        self._pydantic: dict[Ref, bool] = {}

    @staticmethod
    def _parse(path: Path, module: str, rel: str, *, needs_orion: bool = False) -> Optional[FileFacts]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if needs_orion and "orion" not in text:
            return None  # a service file that never names orion cannot touch its models
        return parse_facts(text, module, rel, is_package=path.name == "__init__.py")

    # ------------------------------------------------------------ resolution

    def resolve_in_module(self, module: str, name: str, depth: int = 0) -> Optional[Ref]:
        key = (module, name)
        if key in self._resolve_cache:
            return self._resolve_cache[key]
        self._resolve_cache[key] = None  # cycle guard
        out: Optional[Ref] = None
        f = self.orion.get(module)
        if f is not None and depth < 8:
            if name in f.classes:
                out = (module, name)
            elif name in f.imports:
                mod, attr = f.imports[name]
                if attr is not None:
                    out = self.resolve_in_module(mod, attr, depth + 1)
        self._resolve_cache[key] = out
        return out

    def resolve_dotted(self, facts: FileFacts, dotted: str) -> Optional[Ref]:
        """``X`` / ``alias.X`` / ``pkg.mod.X`` as written in ``facts`` -> class ref."""
        head, _, rest = dotted.partition(".")
        if not rest:
            if head in facts.classes and facts.module in self.orion:
                return (facts.module, head)
            if head in facts.imports:
                mod, attr = facts.imports[head]
                if attr is not None:
                    return self.resolve_in_module(mod, attr)
            return None
        # alias.X or package.module.X
        parts = dotted.split(".")
        if head in facts.imports:
            mod, attr = facts.imports[head]
            base = f"{mod}.{attr}" if attr else mod
            parts = base.split(".") + parts[1:]
        return self.resolve_in_module(".".join(parts[:-1]), parts[-1])

    def _top_symbol(self, module: str, name: str, depth: int = 0) -> Optional[Ref]:
        f = self.orion.get(module)
        if f is None or depth > 8:
            return None
        if name in f.toplevel:
            return (module, name)
        if name in f.imports:
            mod, attr = f.imports[name]
            if attr is not None:
                return self._top_symbol(mod, attr, depth + 1)
        return None

    def resolve_symbol(self, facts: FileFacts, dotted: str) -> Optional[Ref]:
        """A dotted reference in ``facts`` -> the top-level ``orion`` def/class it
        names, following aliases and re-exports; ``None`` if it is not one."""
        parts = dotted.split(".")
        head = parts[0]
        if facts.module in self.orion and head in facts.toplevel:
            return (facts.module, head)
        if head not in facts.imports:
            return None
        mod, attr = facts.imports[head]
        if attr is not None:
            r = self._top_symbol(mod, attr)
            if r is not None or len(parts) == 1:
                return r
            base = f"{mod}.{attr}".split(".")  # ``from orion.x import mod``; mod.fn
        else:
            base = mod.split(".")
        full = base + parts[1:]
        # longest module prefix that exists, next component is the symbol
        for i in range(len(full) - 1, 0, -1):
            m = ".".join(full[:i])
            if m in self.orion:
                return self._top_symbol(m, full[i])
        return None

    # --------------------------------------------------------------- classes

    def bases_of(self, ci: ClassInfo) -> list[Ref]:
        f = self.orion.get(ci.module)
        out = []
        for b in ci.base_exprs:
            r = self.resolve_dotted(f, b) if f else None
            if r and r in self.classes:
                out.append(r)
        return out

    def is_pydantic(self, ref: Ref, _seen: Optional[set] = None) -> bool:
        if ref in self._pydantic:
            return self._pydantic[ref]
        seen = _seen or set()
        if ref in seen:
            return False
        seen.add(ref)
        ci = self.classes[ref]
        f = self.orion.get(ci.module)
        result = False
        for b in ci.base_exprs:
            last = b.split(".")[-1]
            if last in ("BaseModel", "RootModel") and f is not None:
                src = f.imports.get(b.split(".")[0])
                if src is None or src[0].split(".")[0] == "pydantic" or b.startswith("pydantic."):
                    result = True
                    break
        if not result:
            result = any(self.is_pydantic(r, seen) for r in self.bases_of(ci))
        self._pydantic[ref] = result
        return result

    def effective_extra(self, ref: Ref, _depth: int = 0) -> Optional[str]:
        ci = self.classes[ref]
        if ci.own_extra is not None or _depth > 10:
            return ci.own_extra
        for b in self.bases_of(ci):
            e = self.effective_extra(b, _depth + 1)
            if e is not None:
                return e
        return None

    def base_closure(self, ref: Ref) -> list[Ref]:
        out, stack = [], [ref]
        while stack:
            r = stack.pop()
            if r in out:
                continue
            out.append(r)
            stack.extend(self.bases_of(self.classes[r]))
        return out

    def nested(self, ref: Ref) -> set[Ref]:
        """Pydantic models referenced (transitively) from ``ref``'s fields,
        including inherited ones."""
        out: set[Ref] = set()
        stack = [ref]
        while stack:
            r = stack.pop()
            for b in self.base_closure(r):
                ci = self.classes[b]
                f = self.orion.get(ci.module)
                for n in ci.ann_names:
                    t = self.resolve_dotted(f, n) if f else None
                    if t and t in self.classes and t not in out and t != ref and self.is_pydantic(t):
                        out.add(t)
                        stack.append(t)
        return out

    # ------------------------------------------------------------- services

    def scope_symbols(self, facts: FileFacts) -> dict[Optional[str], set[Ref]]:
        """Per enclosing scope, the ``orion`` top-level symbols it references."""
        out: dict[Optional[str], set[Ref]] = {}
        for scope, refs in facts.refs.items():
            syms = set()
            for d in refs:
                r = self.resolve_symbol(facts, d)
                if r is not None:
                    syms.add(r)
            out[scope] = syms
        return out


def _channel_producers(repo_root: Path) -> dict[str, set[str]]:
    """schema_id -> producer services, from channels.yaml (no yaml dependency:
    the file is a flat list of ``- name:`` blocks)."""
    path = Path(repo_root) / "orion" / "bus" / "channels.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, set[str]] = collections.defaultdict(set)
    for block in re.split(r"\n\s*-\s+name:", text):
        sid = re.search(r"^\s*schema_id:\s*['\"]?([A-Za-z0-9_]+)", block, re.M)
        prod = re.search(r"^\s*producer_services:\s*\[([^\]]*)\]", block, re.M)
        if sid and prod:
            out[sid.group(1)] |= {p.strip().strip("'\"") for p in prod.group(1).split(",") if p.strip()}
    return out


def _registry_names(index: RepoIndex) -> dict[str, Ref]:
    """schema_id -> class, read from orion/schemas/registry.py's
    ``"Name": Model`` / ``"Name": SchemaRegistration(Model, ...)`` entries."""
    f = index.orion.get("orion.schemas.registry")
    if f is None:
        return {}
    text = (index.root / f.path).read_text(encoding="utf-8")
    out: dict[str, Ref] = {}
    for m in re.finditer(r"['\"]([A-Za-z0-9_]+)['\"]\s*:\s*(?:SchemaRegistration\(\s*(?:model\s*=\s*)?)?([A-Za-z_][A-Za-z0-9_]*)", text):
        r = index.resolve_dotted(f, m.group(2))
        if r:
            out[m.group(1)] = r
    return out


def discover(repo_root: Path, *, include_loose: bool = True, lib_hops: int = LIB_HOPS) -> Discovery:
    index = RepoIndex(repo_root)
    models = [r for r in index.classes if index.is_pydantic(r)]
    strict = {r for r in models if index.effective_extra(r) == "forbid"}

    model_set = set(models)
    # service -> evidence ("path:line"), per model and role.
    reads: dict[Ref, dict[str, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    writes: dict[Ref, dict[str, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))

    # Shared-library call sites are "sites": (module, enclosing top-level
    # def/class). A service reads/writes through a site when its code
    # references that def/class -- or, after ``lib_hops`` rounds, an orion
    # def/class that references it. Module-level call sites (no enclosing
    # def) run on import, so any import of the module reaches them.
    lib_sites: dict[tuple[str, Ref], set[tuple[str, Optional[str]]]] = collections.defaultdict(set)
    for f in index.orion.values():
        for role, calls in (("r", f.reads), ("w", f.writes)):
            for dotted, _line, enc in calls:
                r = index.resolve_dotted(f, dotted)
                if r in model_set:
                    lib_sites[(role, r)].add((f.module, enc))

    # Reverse reference index: symbol -> orion scopes / service files using it.
    lib_users: dict[tuple[str, Optional[str]], set[tuple[str, Optional[str]]]] = collections.defaultdict(set)
    for f in index.orion.values():
        for scope, syms in index.scope_symbols(f).items():
            for sym in syms:
                lib_users[sym].add((f.module, scope))
            for m in f.imported_modules:
                if m in index.orion:
                    lib_users[(m, None)].add((f.module, scope))
    svc_users: dict[tuple[str, Optional[str]], dict[str, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for svc, files in index.services.items():
        for f in files:
            syms = set().union(*index.scope_symbols(f).values()) if f.refs else set()
            for sym in syms:
                svc_users[sym][svc].add(f.path)
            for m in f.imported_modules:
                if m in index.orion:
                    svc_users[(m, None)][svc].add(f.path)

    for (role, r), sites in lib_sites.items():
        table = reads if role == "r" else writes
        reached = set(sites)
        frontier = set(sites)
        for _ in range(lib_hops):
            nxt = set()
            for site in frontier:
                for user in lib_users.get(site, ()):
                    if user not in reached and user[0] not in TAINT_EXCLUDED_MODULES:
                        nxt.add(user)
            reached |= nxt
            frontier = nxt
        for site in reached:
            if site[0] in TAINT_EXCLUDED_MODULES:
                continue
            label = f"{site[0]}.{site[1] or '<module>'}"
            for svc, paths in svc_users.get(site, {}).items():
                table[r][svc].update(f"{p} (via {label})" for p in sorted(paths)[:2])

    for svc, files in index.services.items():
        for f in files:
            for role, calls in ((reads, f.reads), (writes, f.writes)):
                for dotted, line, _enc in calls:
                    r = index.resolve_dotted(f, dotted)
                    if r in model_set:
                        role[r][svc].add(f"{f.path}:{line}")

    producers = _channel_producers(index.root)
    for sid, ref in _registry_names(index).items():
        for svc in producers.get(sid, ()):
            if svc not in index.services:
                continue  # "*" wildcards and producers outside services/
            writes[ref][svc].add(f"orion/bus/channels.yaml producer of {sid}")

    def build_enclosing() -> dict[Ref, set[Ref]]:
        # A nested model is read/written wherever an enclosing model is.
        out: dict[Ref, set[Ref]] = collections.defaultdict(set)
        for r in set(reads) | set(writes):
            for n in index.nested(r):
                out[n].add(r)
        return out

    enclosing = build_enclosing()

    def roles(ref: Ref, table) -> dict[str, set[str]]:
        out: dict[str, set[str]] = collections.defaultdict(set)
        for src in {ref} | enclosing.get(ref, set()):
            for svc, ev in table.get(src, {}).items():
                out[svc] |= ev if src == ref else {f"{e} [encloses {ref[1]}]" for e in ev}
        return out

    # Declarations apply only where code found readers and no writer.
    declared_used: set[str] = set()
    no_writer: set[Ref] = set()
    for ref in sorted(strict):
        if not roles(ref, reads) or roles(ref, writes):
            continue
        key = f"{ref[0]}:{ref[1]}"
        path = index.classes[ref].path
        hit = key if key in DECLARED_WRITERS else (path if path in DECLARED_WRITERS else None)
        if hit is None:
            continue
        declared_used.add(hit)
        writer, _reason = DECLARED_WRITERS[hit]
        if writer is None:
            no_writer.add(ref)
        else:
            writes[ref][writer].add(f"declared in DECLARED_WRITERS[{hit!r}]")
    enclosing = build_enclosing()

    per_file: dict[tuple[str, str], dict] = {}
    unresolved: list[Unresolved] = []
    dependency_files: dict[str, tuple[str, ...]] = {}
    for ref in sorted(model_set):
        is_strict = ref in strict
        if not is_strict and not include_loose:
            continue
        rd, wr = roles(ref, reads), roles(ref, writes)
        if not rd:
            continue
        ci = index.classes[ref]
        if not wr:
            if is_strict and ref not in no_writer:
                unresolved.append(Unresolved(f"{ref[0]}:{ref[1]}", ci.path, tuple(sorted(rd))))
            continue
        for w, wev in wr.items():
            others = {s: ev for s, ev in rd.items() if s != w}
            if not others:
                continue
            slot = per_file.setdefault(
                (ci.path, w), {"readers": {}, "models": set(), "strict": False, "writer_ev": set(), "reader_models": {}}
            )
            slot["models"].add(ref[1])
            slot["strict"] |= is_strict
            slot["writer_ev"].update(wev)
            for s, ev in others.items():
                slot["readers"].setdefault(s, set()).update(ev)
                slot["reader_models"].setdefault(s, set()).add(ref[1])
        deps = {index.classes[b].path for b in index.base_closure(ref)} - {ci.path}
        if deps:
            dependency_files[ci.path] = tuple(sorted(set(dependency_files.get(ci.path, ())) | deps))

    candidates = []
    for (path, w), slot in sorted(per_file.items()):
        ev = {w: tuple(sorted(slot["writer_ev"]))[:5]}
        ev.update({s: tuple(sorted(e))[:5] for s, e in slot["readers"].items()})
        candidates.append(
            Candidate(
                path=path,
                writer=w,
                readers=tuple(sorted(slot["readers"])),
                models=tuple(sorted(slot["models"])),
                strict=slot["strict"],
                evidence=ev,
                reader_models={s: tuple(sorted(m)) for s, m in slot["reader_models"].items()},
            )
        )
    return Discovery(candidates, unresolved, len(strict), dependency_files, declared_used, index)


# ---------------------------------------------------------------------------
# Structural comparison of container copies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelShape:
    fields: frozenset[str]
    required: frozenset[str]
    extra: Optional[str]
    before_validator: bool = False
    excluded: frozenset[str] = frozenset()


def shapes_from_sources(sources: Mapping[str, Optional[str]]) -> Optional[dict[str, ModelShape]]:
    """Class name -> effective shape, from a set of file texts (repo-relative
    path -> text) as one container sees them. Inheritance resolves across the
    given files only. Returns ``None`` if any file is missing/unparseable, so
    callers fall back to the byte/time comparison instead of guessing."""
    facts: dict[str, FileFacts] = {}
    for path, text in sources.items():
        if text is None:
            return None
        mod = path[: -len(".py")].replace("/", ".")
        is_pkg = mod.endswith(".__init__")
        mod = mod[: -len(".__init__")] if is_pkg else mod
        f = parse_facts(text, mod, path, is_package=is_pkg)
        if f is None:
            return None
        facts[mod] = f
    classes = {(m, n): ci for m, f in facts.items() for n, ci in f.classes.items()}

    by_name: dict[str, list[Ref]] = collections.defaultdict(list)
    for ref in classes:
        by_name[ref[1]].append(ref)

    def resolve_base(f: FileFacts, b: str) -> Optional[Ref]:
        parts = b.split(".")
        head = parts[0]
        if len(parts) == 1 and b in f.classes:
            return (f.module, b)
        if head not in f.imports:
            return None
        mod, attr = f.imports[head]
        if len(parts) == 1:
            if attr is None:
                return None
            ref = (mod, attr)
            if ref in classes:
                return ref
            # Imported through a re-export (``from orion.schemas import Base``);
            # the defining file is in the fetched set because discovery lists
            # it in ``dependency_files``. Ambiguous -> unresolved.
            hits = by_name.get(attr, [])
            return hits[0] if len(hits) == 1 else None
        # ``alias.Base`` / ``pkg.mod.Base``
        full = (f"{mod}.{attr}" if attr else mod).split(".") + parts[1:]
        ref = (".".join(full[:-1]), full[-1])
        return ref if ref in classes else None

    _TRIVIAL = {"BaseModel", "RootModel", "Generic", "object", "Protocol"}
    memo: dict[Ref, Optional[ModelShape]] = {}

    def shape(ref: Ref, depth: int = 0) -> Optional[ModelShape]:
        """``None`` when any base cannot be resolved among the fetched files:
        the inherited ``extra``/fields would be a guess, so the caller falls
        back to bytes/time instead of reporting a forbid reader as loose."""
        if ref in memo:
            return memo[ref]
        memo[ref] = None  # cycle guard
        ci = classes[ref]
        f = facts[ci.module]
        base_shapes: list[ModelShape] = []
        for b in ci.base_exprs:
            r = resolve_base(f, b)
            if r is None:
                if b.split(".")[-1] in _TRIVIAL:
                    continue
                return None
            s = shape(r, depth + 1) if depth < 10 else None
            if s is None:
                return None
            base_shapes.append(s)
        fields: set[str] = set()
        required: set[str] = set()
        excluded: set[str] = set()
        for s in reversed(base_shapes):  # later bases first, earlier override
            fields |= s.fields
            required = (required - s.fields) | s.required
            excluded = (excluded - s.fields) | s.excluded
        # extra: own setting, else the first base in MRO order that sets it
        extra = ci.own_extra
        if extra is None:
            extra = next((s.extra for s in base_shapes if s.extra is not None), None)
        fields |= set(ci.fields)
        required = (required - set(ci.fields)) | set(ci.required)
        excluded = (excluded - set(ci.fields)) | set(ci.excluded)
        before = ci.before_validator or any(s.before_validator for s in base_shapes)
        out_shape = ModelShape(frozenset(fields), frozenset(required), extra, before, frozenset(excluded))
        memo[ref] = out_shape
        return out_shape

    out: dict[str, ModelShape] = {}
    for (m, n) in classes:
        sh = shape((m, n))
        if sh is not None:
            out[f"{m}:{n}"] = sh
    return out


@dataclass(frozen=True)
class ShapeDiff:
    model: str
    extra_forbidden: tuple[str, ...]  # writer fields a forbid reader rejects
    missing_required: tuple[str, ...]  # reader-required fields the writer lacks
    dropped: tuple[str, ...]  # writer fields a non-forbid reader silently drops
    # writer fields a forbid reader lacks but whose before-validator may strip
    # them (the consumer-first removal pattern): reported, not red.
    maybe_stripped: tuple[str, ...] = ()

    @property
    def breaks(self) -> bool:
        return bool(self.extra_forbidden or self.missing_required)


def compare_models(
    module: str,
    models: Iterable[str],
    writer: Mapping[str, ModelShape],
    reader: Mapping[str, ModelShape],
) -> Optional[list[ShapeDiff]]:
    """Per-model diff for ``models`` in ``module``; ``None`` if either side
    lacks a model (renamed/removed class: fall back to bytes/time)."""
    out = []
    for name in models:
        k = f"{module}:{name}"
        w, r = writer.get(k), reader.get(k)
        if w is None or r is None:
            return None
        new = tuple(sorted(w.fields - w.excluded - r.fields))
        forbid = r.extra == "forbid"
        out.append(
            ShapeDiff(
                model=name,
                extra_forbidden=new if forbid and not r.before_validator else (),
                missing_required=tuple(sorted(r.required - w.fields)),
                dropped=new if not forbid else (),
                maybe_stripped=new if forbid and r.before_validator else (),
            )
        )
    return out
