#!/usr/bin/env python3
"""Resolve which docker-compose services need rebuild after a git diff.

Used by scripts/rebuild_services_from_git_diff.sh (post-merge opt-in hook and
manual runs). Mapping is deterministic: path rules, import index, and
scripts/service_rebuild_paths.yaml — no LLM heuristics.

Usage:
    python scripts/rebuild_affected_services.py --base ORIG_HEAD
    python scripts/rebuild_affected_services.py --paths services/orion-hub/app/main.py
    python scripts/rebuild_affected_services.py --list-only --base HEAD~1
    python scripts/rebuild_affected_services.py --plan --base ORIG_HEAD
    python scripts/rebuild_affected_services.py --running-profile-services orion-vision-retina
    python scripts/rebuild_affected_services.py --pool-held-services orion-diffusion-host
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAPPING_PATH = Path(_SCRIPT_DIR) / "service_rebuild_paths.yaml"
_SERVICES_DIR = "services"
_ORION_DIR = "orion"
_DOCKERFILE_COPY_ORION_RE = re.compile(r"^\s*COPY\s+orion\b", re.MULTILINE | re.IGNORECASE)
_COMPOSE_FILE = "docker-compose.yml"
# Host-side default for ${ORION_HOST_REPO_ROOT:-...} in compose bind mounts.
_HOST_REPO_ROOT_DEFAULT = "/mnt/scripts/Orion-Sapienform"
_REPO_ROOT_VAR_RE = re.compile(r"^\$\{ORION_HOST_REPO_ROOT(?::?-[^}]*)?\}")
# "$" too: an unexpanded ARG/ENV reference cannot name a literal repo path.
_GLOB_CHARS = set("*?[$")
# Dependency sources that never fan out through the baked/mounted rule:
# orion/ is resolved by the module-level import graph instead (every image
# COPYs all of orion/, which says nothing about what it imports), and a
# whole-repo source
# ("" = repo root) would make every path affect that service. A source that is
# a whole top-level folder (e.g. `COPY config ...`) is excluded for the same
# reason: it is not evidence the service reads any particular file under it,
# and it would put those images on every config/ edit.
_DEP_SOURCE_EXCLUDED_TOPS = frozenset({"", _ORION_DIR})

# services/<svc>/<dir>/ trees that are offline harnesses, not runtime code:
# an import from them is not evidence the running image uses that module
# (orion-memory-consolidation's evals/ import orion.situational.context; its
# image never runs them).
_SERVICE_OFFLINE_DIRS = frozenset({"tests", "evals"})

MODE_BUILD = "build"
MODE_RECREATE = "recreate"


@dataclass
class ResolveResult:
    services: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    mesh_host: str = ""
    host_allowlist: list[str] = field(default_factory=list)
    host_filtered_out: list[str] = field(default_factory=list)
    # Subset of ``services`` that must be force-recreated (a changed path is
    # bind-mounted into a container, so an unchanged image would not recreate
    # it and a single-file mount would keep serving the old inode).
    recreate: list[str] = field(default_factory=list)
    # Changed hand-applied SQL migrations (hand_applied_migration_globs). Never
    # rebuilt; the caller reports their live applied/missing state instead.
    migrations: list[str] = field(default_factory=list)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def resolve_base_ref(repo_root: Path, explicit: str | None) -> str:
    if explicit:
        proc = _run_git(["rev-parse", "--verify", explicit], cwd=repo_root)
        if proc.returncode != 0:
            raise ValueError(f"Invalid git ref: {explicit!r} ({proc.stderr.strip()})")
        return explicit

    for candidate in ("ORIG_HEAD", "MERGE_HEAD", "HEAD@{1}"):
        proc = _run_git(["rev-parse", "--verify", candidate], cwd=repo_root)
        if proc.returncode == 0:
            return candidate

    proc = _run_git(["rev-parse", "--verify", "HEAD~1"], cwd=repo_root)
    if proc.returncode == 0:
        return "HEAD~1"
    raise ValueError("Could not resolve a base ref (try --base ORIG_HEAD)")


def _same_python_ast(repo_root: Path, base_ref: str, path: str) -> bool:
    """True when a .py file differs between base and HEAD only in comments/whitespace.

    Any doubt (unreadable blob, syntax error) answers False so the file still
    counts as changed.
    """
    blobs = []
    for rev in (base_ref, "HEAD"):
        # Bytes, not text: ast.parse honours the encoding cookie, and a
        # non-UTF8 blob must not raise out of the planner.
        proc = subprocess.run(
            ["git", "show", f"{rev}:{path}"], cwd=repo_root, capture_output=True, check=False
        )
        if proc.returncode != 0:
            return False
        blobs.append(proc.stdout)
    try:
        return ast.dump(ast.parse(blobs[0])) == ast.dump(ast.parse(blobs[1]))
    except (SyntaxError, ValueError):
        return False


def changed_files(repo_root: Path, base_ref: str) -> list[str]:
    proc = _run_git(["diff", "--name-status", f"{base_ref}..HEAD"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git diff failed for base {base_ref!r}")
    files: list[str] = []
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        path = fields[-1].strip()
        if not path:
            continue
        # A comment-only edit to a widely imported module must not rebuild its
        # importers (2026-10-16: a 4-line comment in orion/schemas/grammar.py
        # planned 68 services).
        if fields[0] == "M" and path.endswith(".py") and _same_python_ast(repo_root, base_ref, path):
            print(
                f"rebuild_affected_services: skip {path}: comment/whitespace-only change",
                file=sys.stderr,
            )
            continue
        files.append(path)
    return files


def discover_services(repo_root: Path) -> set[str]:
    services: set[str] = set()
    root = repo_root / _SERVICES_DIR
    if not root.is_dir():
        return services
    for child in root.iterdir():
        if child.is_dir() and (child / "docker-compose.yml").is_file():
            services.add(child.name)
    return services


@lru_cache(maxsize=1)
def services_copying_orion(repo_root: str) -> frozenset[str]:
    root = Path(repo_root)
    out: set[str] = set()
    for dockerfile in (root / _SERVICES_DIR).glob("*/Dockerfile"):
        text = dockerfile.read_text(encoding="utf-8", errors="replace")
        if _DOCKERFILE_COPY_ORION_RE.search(text):
            out.add(dockerfile.parent.name)
    return frozenset(out)


@dataclass(frozen=True)
class DepSource:
    """A repo path one service's image or containers depend on."""

    prefix: str  # repo-relative, no trailing slash
    service: str  # services/<service> directory name
    mode: str  # MODE_BUILD (baked by COPY/ADD) or MODE_RECREATE (bind-mounted)
    origin: str  # human-readable provenance for the reason string


def _dockerfile_instructions(text: str) -> list[str]:
    """Logical Dockerfile lines (backslash continuations joined, comments dropped)."""
    out: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("#"):
            continue  # comments, including ones inside a continuation
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            out.append(buf.strip())
        buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def _copy_sources(instruction: str) -> list[str]:
    """Source operands of a COPY/ADD instruction; [] for other or --from= stages."""
    split = instruction.split(None, 1)
    if len(split) < 2 or split[0].upper() not in {"COPY", "ADD"}:
        return []
    rest = split[1].strip()
    tokens: list[str] = []
    while rest.startswith("--"):
        flag, _, rest = rest.partition(" ") if " " in rest else (rest, "", "")
        if flag.lower().startswith("--from"):
            return []  # copies from another build stage/image, not the context
        rest = rest.strip()
    if rest.startswith("["):
        try:
            tokens = [str(t) for t in json.loads(rest)]
        except ValueError:
            return []
    else:
        tokens = rest.split()
    if len(tokens) < 2:
        return []
    return [t for t in tokens[:-1] if "://" not in t and not t.startswith("<<")]


def _repo_relative(path: Path, repo_root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(repo_root)
    except ValueError:
        return None
    text = rel.as_posix()
    return "" if text == "." else text


def _strip_glob(source: str) -> str:
    """Longest literal directory prefix of a COPY source (``a/b*.yaml`` -> ``a``)."""
    if not any(ch in _GLOB_CHARS for ch in source):
        return source
    parts: list[str] = []
    for part in source.split("/"):
        if any(ch in _GLOB_CHARS for ch in part):
            break
        parts.append(part)
    return "/".join(parts)


def _mount_host_path(volume: Any) -> str | None:
    if isinstance(volume, dict):
        if volume.get("type") != "bind":
            return None
        source = volume.get("source")
        return str(source) if source else None
    if not isinstance(volume, str):
        return None
    # Short syntax HOST:CONTAINER[:MODE]; split on ':' outside ${...}.
    depth = 0
    for idx, ch in enumerate(volume):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ":" and depth == 0:
            return volume[:idx]
    return None  # anonymous volume (container path only)


def _resolve_mount_source(host: str, compose_dir: Path, repo_root: Path) -> str | None:
    host = host.strip().strip('"').strip("'")
    match = _REPO_ROOT_VAR_RE.match(host)
    if match:
        candidate = repo_root / host[match.end():].lstrip("/")
    elif host.startswith(_HOST_REPO_ROOT_DEFAULT + "/") or host == _HOST_REPO_ROOT_DEFAULT:
        candidate = repo_root / host[len(_HOST_REPO_ROOT_DEFAULT):].lstrip("/")
    elif "$" in host or host.startswith("~"):
        return None  # some other env-driven host path; not a repo path we can name
    elif host.startswith("/"):
        candidate = Path(host)
    elif host.startswith("."):
        candidate = compose_dir / host
    else:
        return None  # named volume
    return _repo_relative(candidate, repo_root)


def _dep_source_allowed(
    prefix: str, service: str, mode: str, repo_root: Path, *, include_own: bool = False
) -> bool:
    top = prefix.split("/", 1)[0]
    if top in _DEP_SOURCE_EXCLUDED_TOPS or "/" not in prefix:
        return False  # repo root, orion/, or a whole top-level folder
    if _path_matches_prefix(prefix, f"{_SERVICES_DIR}/{service}"):
        if include_own:
            # The migration/rebuild-only skips bypass the direct rule, so
            # they need to see every own-dir COPY/mount, file or directory.
            return True
        # Own-dir paths are already rebuilt by the direct services/<svc>/ rule.
        # A directory mount of the service's own code/static is live already;
        # only a single-file own mount keeps a stale inode and needs recreate.
        if mode == MODE_BUILD:
            return False
        return (repo_root / prefix).is_file()
    return True


def _load_compose(path: Path) -> dict[str, Any]:
    try:
        data = _load_yaml(path)
    except Exception:  # malformed compose must not break the whole classifier
        return {}
    return data


@lru_cache(maxsize=8)
def build_dependency_sources(repo_root: str, include_own: bool = False) -> tuple[DepSource, ...]:
    """Repo paths each service bakes into its image or bind-mounts into containers.

    Parsed from services/<svc>/docker-compose.yml (build context + dockerfile,
    bind-mount volumes) and the Dockerfiles it names. Catches the case the
    import graph cannot see: a non-Python file such as config/vision_zones.yaml
    COPY'd into three images (PR #2677).

    ``include_own`` keeps a service's paths inside its own services/<svc>/
    (normally dropped: the direct rule already rebuilds them). The glob skip
    rules use it so they can never skip a file the container does read.
    """
    root = Path(repo_root).resolve()
    out: set[DepSource] = set()
    services_root = root / _SERVICES_DIR
    if not services_root.is_dir():
        return ()
    for compose_path in sorted(services_root.glob(f"*/{_COMPOSE_FILE}")):
        service = compose_path.parent.name
        compose_dir = compose_path.parent
        compose = _load_compose(compose_path)
        compose_services = compose.get("services") or {}
        if not isinstance(compose_services, dict):
            continue
        dockerfiles: set[tuple[Path, Path]] = set()
        for name, spec in compose_services.items():
            if not isinstance(spec, dict):
                continue
            build = spec.get("build")
            if isinstance(build, str):
                build = {"context": build}
            if isinstance(build, dict):
                context = (compose_dir / str(build.get("context") or ".")).resolve()
                dockerfile = context / str(build.get("dockerfile") or "Dockerfile")
                dockerfiles.add((context, dockerfile))
            for volume in spec.get("volumes") or []:
                host = _mount_host_path(volume)
                if not host:
                    continue
                prefix = _resolve_mount_source(host, compose_dir, root)
                if prefix is None or not _dep_source_allowed(prefix, service, MODE_RECREATE, root, include_own=include_own):
                    continue
                out.add(DepSource(prefix, service, MODE_RECREATE, f"bind-mounted by {_COMPOSE_FILE} service {name}"))
        for context, dockerfile in sorted(dockerfiles):
            if not dockerfile.is_file():
                continue
            df_rel = _repo_relative(dockerfile, root) or dockerfile.name
            if df_rel and _dep_source_allowed(df_rel, service, MODE_BUILD, root, include_own=include_own):
                out.add(DepSource(df_rel, service, MODE_BUILD, f"Dockerfile of {_COMPOSE_FILE} build"))
            text = dockerfile.read_text(encoding="utf-8", errors="replace")
            for instruction in _dockerfile_instructions(text):
                for src in _copy_sources(instruction):
                    literal = _strip_glob(src)
                    prefix = _repo_relative(context / literal, root)
                    if prefix is None or not _dep_source_allowed(prefix, service, MODE_BUILD, root, include_own=include_own):
                        continue
                    out.add(DepSource(prefix, service, MODE_BUILD, f"COPY/ADD in {df_rel}"))
    return tuple(sorted(out, key=lambda d: (d.prefix, d.service, d.mode)))


def dependency_hits(path: str, sources: tuple[DepSource, ...]) -> list[DepSource]:
    """Dependency sources a changed path falls under."""
    hits: list[DepSource] = []
    for dep in sources:
        if _path_matches_prefix(path, dep.prefix):
            hits.append(dep)
    return hits


def _is_test_path(parts: tuple[str, ...]) -> bool:
    """tests/ directories and pytest files never ship as runtime imports."""
    if "tests" in parts:
        return True
    name = parts[-1] if parts else ""
    return name == "conftest.py" or (name.startswith("test_") and name.endswith(".py"))


def _orion_module_name(rel_parts: tuple[str, ...]) -> str:
    """('orion','a','b.py') -> 'orion.a.b'; ('orion','a','__init__.py') -> 'orion.a'."""
    parts = list(rel_parts)
    last = parts[-1][:-3]  # strip .py
    if last == "__init__":
        parts = parts[:-1]
    else:
        parts[-1] = last
    return ".".join(parts)


_IMPORT_STMT_RE = re.compile(r"^[ \t]*import[ \t]+([^\n#;]+)", re.MULTILINE)
_FROM_STMT_RE = re.compile(
    r"^[ \t]*from[ \t]+(\.*)([A-Za-z_][\w.]*)?[ \t]+import[ \t]+(\([^)]*\)|[^\n#;]+)",
    re.MULTILINE,
)
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _imported_orion_names(text: str, *, module: str | None, is_package: bool) -> set[str]:
    """Every orion.* name a file's import statements mention.

    ``from orion.a import b`` yields both ``orion.a`` and ``orion.a.b`` (b may
    be a submodule). Relative imports are resolved when ``module`` (the file's
    own dotted name under orion/) is given. Regex, not ast: ast.walk over the
    ~4.6k files cost ~7s per run (regex: ~1.3s); a docstring that happens to start a line with
    an import statement only over-includes a consumer.
    """
    out: set[str] = set()
    for match in _IMPORT_STMT_RE.finditer(text):
        for item in match.group(1).split(","):
            name = item.strip().split()[0] if item.strip() else ""
            if name == "orion" or name.startswith("orion."):
                out.add(name)
    for match in _FROM_STMT_RE.finditer(text):
        dots, base, names = match.group(1), match.group(2) or "", match.group(3)
        if dots:
            if not module:
                continue
            pkg_parts = module.split(".") if is_package else module.split(".")[:-1]
            drop = len(dots) - 1
            if drop > len(pkg_parts):
                continue
            anchor = pkg_parts[: len(pkg_parts) - drop]
            base = ".".join([*anchor, base] if base else anchor)
        if base != "orion" and not base.startswith("orion."):
            continue
        out.add(base)
        cleaned = re.sub(r"#[^\n]*", "", names.strip("()"))
        for item in cleaned.split(","):
            words = item.split()
            if words and words[0] != "*" and _IDENT_RE.fullmatch(words[0]):
                out.add(f"{base}.{words[0]}")
    return out


@dataclass(frozen=True)
class ImportGraph:
    """Module-level orion import graph (test files excluded)."""

    # dotted orion module -> True when it is a package (__init__.py)
    orion_modules: dict[str, bool]
    # orion module -> orion names it imports
    orion_imports: dict[str, frozenset[str]]
    # services/<name> -> orion names its non-test code imports
    service_imports: dict[str, frozenset[str]]


@lru_cache(maxsize=4)
def build_import_graph(repo_root: str) -> ImportGraph:
    root = Path(repo_root)
    orion_modules: dict[str, bool] = {}
    orion_imports: dict[str, frozenset[str]] = {}
    service_imports: dict[str, set[str]] = {}

    for py_file in (root / _ORION_DIR).rglob("*.py"):
        rel = py_file.relative_to(root).parts
        if _is_test_path(rel):
            continue
        module = _orion_module_name(rel)
        is_package = rel[-1] == "__init__.py"
        orion_modules[module] = is_package
        text = py_file.read_text(encoding="utf-8", errors="replace")
        orion_imports[module] = frozenset(_imported_orion_names(text, module=module, is_package=is_package))

    for py_file in (root / _SERVICES_DIR).rglob("*.py"):
        rel = py_file.relative_to(root).parts
        if len(rel) < 3 or _is_test_path(rel) or _SERVICE_OFFLINE_DIRS.intersection(rel[2:-1]):
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        names = _imported_orion_names(text, module=None, is_package=False)
        if names:
            service_imports.setdefault(rel[1], set()).update(names)

    return ImportGraph(
        orion_modules=orion_modules,
        orion_imports=orion_imports,
        service_imports={svc: frozenset(n) for svc, n in service_imports.items()},
    )


def _executed_modules(names: frozenset[str] | set[str], modules: dict[str, bool]) -> set[str]:
    """orion modules whose code runs when ``names`` are imported.

    Importing orion.a.b runs orion/__init__.py and orion/a/__init__.py too, so
    every existing ancestor package counts. Names that are not modules
    (``from orion.a.b import SomeClass`` also yields ``orion.a.b.SomeClass``)
    resolve to their nearest existing module.
    """
    out: set[str] = set()
    for name in names:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix in modules:
                out.add(prefix)
    return out


@lru_cache(maxsize=4)
def _reverse_edges(repo_root: str) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """(module -> orion modules that run it, module -> services that run it)."""
    graph = build_import_graph(repo_root)
    by_module: dict[str, set[str]] = {}
    by_service: dict[str, set[str]] = {}
    for importer, names in graph.orion_imports.items():
        for mod in _executed_modules(names, graph.orion_modules):
            if mod != importer:
                by_module.setdefault(mod, set()).add(importer)
    for svc, names in graph.service_imports.items():
        for mod in _executed_modules(names, graph.orion_modules):
            by_service.setdefault(mod, set()).add(svc)
    return (
        {k: frozenset(v) for k, v in by_module.items()},
        {k: frozenset(v) for k, v in by_service.items()},
    )


def _top_package(module: str) -> str:
    return ".".join(module.split(".")[:2])


def module_service_consumers(
    target: str,
    repo_root: str,
    *,
    whole_package: bool = False,
    hops: int = 1,
    barriers: frozenset[str] = frozenset(),
) -> tuple[set[str], set[str]]:
    """(services, orion modules walked through) whose runtime code runs ``target``.

    Walk, over reverse import edges:
      1. every module of ``target``'s own top-level package that reaches it,
         however deep (a package is one unit: orion.curiosity.worldview <-
         atlas <- write_stamp is still "curiosity");
      2. then ``hops`` hops into other packages (orion.harness.fcc_motor
         imports orion.curiosity.write_stamp), each hop closed the same way
         inside the package it lands in (orion.substrate.activation imports
         orion.core.activation_decay; falkor_store and falkor_anchor_store
         in orion.substrate reach it through activation).
    Services importing any walked module are consumers. A module in
    ``barriers`` is a consumer itself but is never walked through.

    Measured on the last 10 merges to main: without the registry barrier this
    reaches ~60 services (unbounded: 84) for an ordinary orion.situational /
    orion.schemas edit, through orion.schemas.registry ->
    orion.core.bus.async_service.
    ``whole_package`` treats every module under ``target`` as changed (a data
    file read by its package).
    """
    graph = build_import_graph(repo_root)
    by_module, by_service = _reverse_edges(repo_root)
    if whole_package:
        prefix = target + "."
        start = {m for m in graph.orion_modules if m == target or m.startswith(prefix)}
        start.add(target)
    else:
        start = {target}
    pkg = _top_package(target)

    def importers(modules: set[str]) -> set[str]:
        return {imp for mod in modules if mod not in barriers for imp in by_module.get(mod, ())}

    def close_within(new: set[str], packages: set[str]) -> set[str]:
        """Add every module of ``packages`` that (transitively) imports ``new``."""
        added = set(new)
        frontier = set(new)
        while frontier:
            frontier = {m for m in importers(frontier) if _top_package(m) in packages} - seen - added
            added |= frontier
        return added

    seen: set[str] = set()
    seen |= close_within(start, {pkg})  # 1. the changed module's own package
    frontier = set(seen)
    for _ in range(max(0, hops)):  # 2. hop into other packages, closed per package
        crossed = importers(frontier) - seen
        if not crossed:
            break
        frontier = close_within(crossed, {_top_package(m) for m in crossed})
        seen |= frontier
    services = {svc for mod in seen for svc in by_service.get(mod, ())}
    return services, seen - start


def _orion_targets_for_path(
    path: str,
    data_readers: dict[str, list[str]],
) -> tuple[list[tuple[str, bool]], str]:
    """([(dotted target, whole_package)], label) a changed orion/ path maps to.

    A .py file maps to its own module. A non-Python file (verb YAML, prompt
    template) has no import edge, so it maps to the modules that read it:
    an explicit ``orion_data_readers`` entry when one matches, else every
    module of its top-level orion/<pkg>/ (loaders read data from their own
    package tree, e.g. orion/cognition/plan_loader.py -> cognition/prompts/).
    Empty list when the file sits directly under orion/.
    """
    parts = tuple(path.split("/"))
    if path.endswith(".py"):
        module = _orion_module_name(parts)
        kind = "package" if parts[-1] == "__init__.py" else "module"
        return [(module, False)], f"{module} ({kind})"
    best = ""
    for prefix in data_readers:
        if _path_matches_prefix(path, prefix) and len(prefix) > len(best):
            best = prefix
    if best:
        readers = [r for r in data_readers[best] if r == "orion" or r.startswith("orion.")]
        label = f"data file read by {', '.join(data_readers[best])}"
        return [(m, False) for m in readers], label
    if len(parts) < 3:
        return [], ""
    pkg = ".".join(parts[:2])
    return [(pkg, True)], f"data file in {pkg} (whole package)"


def _load_service_list_file(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.is_file():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            names.add(stripped)
    return names


def _normalize_mesh_host(name: str) -> str:
    normalized = name.strip().lower()
    if normalized.startswith("orion-"):
        normalized = normalized[len("orion-") :]
    return normalized.split(".", 1)[0]


def detect_mesh_host(repo_root: Path, explicit: str | None = None) -> str:
    if explicit:
        return _normalize_mesh_host(explicit)

    env_host = os.environ.get("ORION_MESH_HOST", "").strip()
    if env_host:
        return _normalize_mesh_host(env_host)

    env_file = repo_root / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "NODE_NAME" and value.strip():
                return _normalize_mesh_host(value)
            if key == "PROJECT" and value.strip().lower().startswith("orion-"):
                return _normalize_mesh_host(value[len("orion-") :])

    return _normalize_mesh_host(socket.gethostname())


def _include_file_for_host(repo_root: Path, host: str) -> Path | None:
    override = os.environ.get("ORION_INCLUDE_SERVICES_FILE", "").strip()
    if override:
        path = Path(override)
        return path if path.is_file() else None

    mesh_common = repo_root / "mesh-utilities" / "common"
    for candidate in (
        mesh_common / f"include_services_{host}.txt",
        mesh_common / "include_services" / f"{host}.txt",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_host_allowlist(repo_root: Path, host: str) -> set[str] | None:
    include_file = _include_file_for_host(repo_root, host)
    if include_file is None:
        return None
    return _load_service_list_file(include_file)


def _load_excludes(repo_root: Path) -> set[str]:
    path = repo_root / "mesh-utilities" / "common" / "exclude_services.txt"
    return _load_service_list_file(path)


def _service_from_path(path: str) -> str | None:
    if not path.startswith(f"{_SERVICES_DIR}/"):
        return None
    rest = path[len(_SERVICES_DIR) + 1 :]
    if not rest:
        return None
    return rest.split("/", 1)[0]


def _glob_match(path: str, pattern: str) -> bool:
    """fnmatch, but `*` never crosses a `/` (same depth as the pattern)."""
    return path.count("/") == pattern.count("/") and fnmatch.fnmatchcase(path, pattern)


def _path_matches_prefix(path: str, prefix: str) -> bool:
    normalized = prefix.rstrip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def resolve_affected_services(
    paths: list[str],
    repo_root: Path | None = None,
    mesh_host: str | None = None,
) -> ResolveResult:
    root = (repo_root or _REPO_ROOT).resolve()
    if _MAPPING_PATH.exists():
        try:
            import yaml  # type: ignore  # noqa: F401
        except ImportError as exc:
            # Without the mapping every skip/fleet-wide rule vanishes and a
            # channels.yaml edit would rebuild the whole COPY-orion fleet.
            raise RuntimeError(f"PyYAML is required to read {_MAPPING_PATH.name}") from exc
    mapping = _load_yaml(_MAPPING_PATH)
    known_services = discover_services(root)
    orion_copy_services = set(services_copying_orion(str(root)))
    contract_raw = mapping.get("contract_paths")
    contract_prefixes = list(["orion/bus/", "orion/schemas/"] if contract_raw is None else contract_raw)
    fleet_wide_paths: dict[str, str] = mapping.get("fleet_wide_paths") or {}
    hops_raw = mapping.get("orion_import_hops")
    import_hops = 1 if hops_raw is None else int(hops_raw)
    data_readers: dict[str, list[str]] = mapping.get("orion_data_readers") or {}
    import_barriers = frozenset(mapping.get("orion_import_barriers") or [])
    data_no_fallback: list[str] = list(mapping.get("orion_data_no_fallback") or [])
    skip_prefixes = list(mapping.get("skip_prefixes") or [])
    skip_exact = set(mapping.get("skip_exact") or [])
    migration_globs: list[str] = list(mapping.get("hand_applied_migration_globs") or [])
    rebuild_only: dict[str, list[str]] = mapping.get("service_rebuild_only") or {}
    script_services: dict[str, list[str]] = mapping.get("script_services") or {}
    orion_extra: dict[str, list[str]] = mapping.get("orion_package_extra_services") or {}
    dep_sources = build_dependency_sources(str(root))
    own_dep_sources = build_dependency_sources(str(root), include_own=True)

    affected: set[str] = set()
    recreate: set[str] = set()
    skipped: list[str] = []
    reasons: dict[str, str] = {}
    migrations: list[str] = []

    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]

        # Baked (COPY/ADD) or bind-mounted repo paths. Checked before the skip
        # rules: config/ is skipped for the generic rule but is exactly what
        # Dockerfiles COPY (config/vision_zones.yaml, PR #2677).
        hits = [d for d in dependency_hits(path, dep_sources) if d.service in known_services]
        dep_reason = ""
        if hits:
            for dep in hits:
                affected.add(dep.service)
                if dep.mode == MODE_RECREATE:
                    recreate.add(dep.service)
            dep_reason = "; ".join(sorted({f"{d.service} ({d.mode}: {d.origin})" for d in hits}))

        def note(reason: str, *, skip: bool) -> None:
            # A dependency hit means the path is NOT skipped, whatever the
            # generic rules below would say about it.
            if hits:
                reasons[path] = dep_reason if skip else f"{dep_reason}; {reason}"
            else:
                reasons[path] = reason
                if skip:
                    skipped.append(path)

        if path in skip_exact:
            note("repo metadata (no runtime rebuild)", skip=True)
            continue

        if path in fleet_wide_paths:
            note(f"read by every bus service; not auto-rebuilt ({fleet_wide_paths[path]})", skip=True)
            continue

        matched_skip = False
        for prefix in skip_prefixes:
            if _path_matches_prefix(path, prefix):
                note(f"under {prefix.rstrip('/')} (no auto rebuild)", skip=True)
                matched_skip = True
                break
        if not matched_skip and path.startswith(f"{_ORION_DIR}/"):
            if _is_test_path(tuple(path.split("/"))):
                note("orion package tests (no auto rebuild)", skip=True)
                matched_skip = True
        if matched_skip:
            continue

        owner = _service_from_path(path)
        is_migration = any(_glob_match(path, g) for g in migration_globs)
        only = rebuild_only.get(owner or "")
        outside_only = (
            only is not None
            and not is_migration
            and path[len(f"{_SERVICES_DIR}/{owner}/"):] not in set(only)
        )
        # These skips bypass the direct services/<svc>/ rule, so they never
        # skip a path the owning service's own compose mount or Dockerfile
        # COPY brings into a container; such a path falls through to that
        # rule. (Another service's mount is already handled by `hits` above
        # and must not drag the owner along.)
        consumed = (is_migration or outside_only) and any(
            d.service == owner for d in dependency_hits(path, own_dep_sources)
        )
        if is_migration and not consumed:
            migrations.append(path)
            note(
                "hand-applied SQL migration: no container consumes it, so no rebuild "
                "(apply it by hand with docker exec ... psql; never by recreating the DB container)",
                skip=True,
            )
            continue
        if outside_only and not consumed:
            if path.endswith("/.env_example"):
                note(
                    f"env template change for {owner} (rebuild-only service) -- run "
                    "sync_local_env_from_example.py; recreate it by hand only if the change needs it",
                    skip=True,
                )
            else:
                note(
                    f"{owner} only rebuilds on {', '.join(only)} (service_rebuild_only: a recreate is "
                    "an outage); this file is not read by its container",
                    skip=True,
                )
            continue

        service = _service_from_path(path)
        if service:
            if service in known_services:
                affected.add(service)
                note(f"direct change in services/{service}/", skip=False)
            else:
                note(f"unknown service directory {service!r}", skip=True)
            continue

        if path.startswith(f"{_ORION_DIR}/"):
            if path.endswith(".md"):
                note("orion markdown docs (no auto rebuild)", skip=True)
                continue
            path_parts = tuple(path.split("/"))
            if not path.endswith(".py") and (
                _SERVICE_OFFLINE_DIRS.intersection(path_parts[:-1]) or path_parts[-1] == "Makefile"
            ):
                note("orion eval/test fixture or Makefile (no auto rebuild)", skip=True)
                continue
            in_contract = any(_path_matches_prefix(path, p) for p in contract_prefixes)
            targets, label = _orion_targets_for_path(path, data_readers)
            if in_contract and not path.endswith(".py") and not any(
                _path_matches_prefix(path, p) for p in data_readers
            ):
                # The import graph only sees .py modules. A data file under a
                # contract root (orion/bus/, orion/schemas/) has no import edge
                # to its readers, so fall back to every image that bakes orion.
                for svc in orion_copy_services:
                    affected.add(svc)
                message = "non-Python contract file, readers unresolvable → all services with COPY orion"
                print(f"rebuild_affected_services: {path}: {message}", file=sys.stderr)
                note(message, skip=False)
                continue
            data_reader_prefix = max(
                (p for p in data_readers if _path_matches_prefix(path, p)), key=len, default=""
            )
            if not path.endswith(".py") and not data_reader_prefix and any(
                _path_matches_prefix(path, p) for p in data_no_fallback
            ):
                note(
                    "data file with no orion_data_readers entry in a package whose whole-package "
                    "fallback reaches the bus core (rebuild manually, or add orion_data_readers)",
                    skip=True,
                )
                continue
            if not targets and not data_reader_prefix:
                note("file directly under orion/ (rebuild manually if needed)", skip=True)
                continue
            consumers: set[str] = set()
            hop_modules: set[str] = set()
            for module, whole_package in targets:
                found, hops_seen = module_service_consumers(
                    module,
                    str(root),
                    whole_package=whole_package,
                    hops=import_hops,
                    barriers=import_barriers,
                )
                consumers |= found
                hop_modules |= hops_seen
            top_pkg = path.split("/")[1]
            consumers.update(orion_extra.get(top_pkg, []))
            if data_reader_prefix:
                # Non-module entries are services that read the file directly.
                consumers.update(
                    r for r in data_readers[data_reader_prefix] if r != "orion" and not r.startswith("orion.")
                )
            if consumers:
                for svc in consumers:
                    affected.add(svc)
                via = f", via {len(hop_modules)} orion module(s)" if hop_modules else ""
                note(f"{label} → importers (own package + {import_hops} package hop(s)){via}", skip=False)
            else:
                note(
                    f"{label}: no service imports it (own package + {import_hops} package hop(s)) "
                    "(rebuild manually, or add orion_data_readers / orion_package_extra_services)",
                    skip=True,
                )
            continue

        if path.startswith("scripts/"):
            mapped = script_services.get(path)
            if mapped is None:
                note("scripts/ change with no script_services mapping (rebuild manually if needed)", skip=True)
            elif not mapped:
                note("scripts/ meta-tool (explicit empty mapping)", skip=True)
            else:
                for svc in mapped:
                    if svc in known_services:
                        affected.add(svc)
                note(f"script_services mapping → {', '.join(mapped)}", skip=False)
            continue

        if path == ".env" or path.endswith("/.env_example"):
            note(
                "env template change — run sync_local_env_from_example.py and restart affected services manually",
                skip=True,
            )
            continue

        note("no mapping rule (rebuild manually if needed)", skip=True)

    host = detect_mesh_host(root, mesh_host)
    allowlist = load_host_allowlist(root, host)
    known_affected = {svc for svc in affected if svc in known_services}

    if allowlist is not None:
        host_filtered_out = sorted(svc for svc in known_affected if svc not in allowlist)
        filtered = sorted(svc for svc in known_affected if svc in allowlist)
        return ResolveResult(
            services=filtered,
            skipped=sorted(set(skipped)),
            reasons=reasons,
            mesh_host=host,
            host_allowlist=sorted(allowlist),
            host_filtered_out=host_filtered_out,
            recreate=sorted(svc for svc in recreate if svc in filtered),
            # Migrations are reported on the host that runs their database.
            migrations=sorted(
                {m for m in migrations if _service_from_path(m) in allowlist}
            ),
        )

    excludes = _load_excludes(root)
    filtered = sorted(svc for svc in known_affected if svc not in excludes)
    return ResolveResult(
        services=filtered,
        skipped=sorted(set(skipped)),
        reasons=reasons,
        mesh_host=host,
        recreate=sorted(svc for svc in recreate if svc in filtered),
        migrations=sorted(set(migrations)),
    )


# docker ps format used for profile detection: compose project, compose
# service, compose config files (comma-separated absolute paths).
_DOCKER_PS_FORMAT = (
    '{{.Label "com.docker.compose.project"}}\t'
    '{{.Label "com.docker.compose.service"}}\t'
    '{{.Label "com.docker.compose.project.config_files"}}'
)


def read_running_compose_containers(ps_file: Path | None = None) -> list[tuple[str, str, str]] | None:
    """(project, compose service, config_files) for every running container.

    ``ps_file`` replays a saved ``docker ps --format`` dump (tests). Returns
    None when docker cannot be queried -- callers must then start nothing extra.
    """
    if ps_file is not None:
        text = ps_file.read_text(encoding="utf-8")
    else:
        try:
            proc = subprocess.run(
                ["docker", "ps", "--format", _DOCKER_PS_FORMAT],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        text = proc.stdout
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 3:
            parts.append("")
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def running_profile_services(
    repo_root: Path,
    service: str,
    running: list[tuple[str, str, str]],
    skip_profiles: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Profile-gated compose services in services/<service>/ that are running now.

    A plain ``up -d --build`` never touches a service behind ``profiles:``, so
    a live profile container (orion-vision-retina-walkway) kept its old image
    after every pull. Only services already running are returned: this must
    never start something an operator left off. Returns (profile, compose
    service) pairs, sorted.
    """
    compose_path = repo_root / _SERVICES_DIR / service / _COMPOSE_FILE
    compose = _load_compose(compose_path)
    compose_services = compose.get("services") or {}
    if not isinstance(compose_services, dict):
        return []
    suffix = f"{_SERVICES_DIR}/{service}/{_COMPOSE_FILE}"
    own_compose = compose_path.resolve().as_posix()
    running_from: dict[str, list[str]] = {}
    for _project, compose_service, config_files in running:
        files = [f.strip().replace("\\", "/") for f in config_files.split(",") if f.strip()]
        if any(f == suffix or f.endswith("/" + suffix) for f in files):
            running_from[compose_service] = files
    skip = skip_profiles or set()
    out: list[tuple[str, str]] = []
    for name, spec in compose_services.items():
        if not isinstance(spec, dict) or name not in running_from:
            continue
        profiles = [str(p) for p in (spec.get("profiles") or [])]
        if not profiles or any(p in skip for p in profiles):
            continue
        if running_from[name] != [own_compose]:
            # Started from another checkout (a worktree) or with an overlay
            # compose file: an `up` from here would silently take it over.
            print(
                f"rebuild_affected_services: WARNING: running profile service {service}/{name} "
                f"was started from {','.join(running_from[name])!r}, not {own_compose!r} -- leaving it alone",
                file=sys.stderr,
            )
            continue
        out.append((profiles[0], name))
    return sorted(out)


_GPU_POOL_CONFIG = Path("config") / "gpu_pool.yaml"


class PoolConfigError(RuntimeError):
    """config/gpu_pool.yaml could not be read: the caller must not start anything it might own."""


def pool_launch_targets(repo_root: Path, service: str) -> list[str]:
    """Plain (no compose profile) services of services/<service>/docker-compose.yml that the GPU
    pool starts and stops itself (a ``launch:`` block in config/gpu_pool.yaml).

    Raises PoolConfigError when the pool config exists but cannot be read (malformed YAML, no
    PyYAML): an unreadable config must not read as "the pool owns nothing here"."""
    path = repo_root / _GPU_POOL_CONFIG
    if not path.exists():
        return []
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- ImportError and YAML errors alike
        raise PoolConfigError(f"{path}: {type(exc).__name__}: {exc}") from exc
    roles = cfg.get("roles") if isinstance(cfg, dict) else None
    if not isinstance(roles, dict):
        raise PoolConfigError(f"{path}: no roles mapping")
    compose_rel = f"{_SERVICES_DIR}/{service}/{_COMPOSE_FILE}"
    out: set[str] = set()
    for spec in roles.values():
        launch = spec.get("launch") if isinstance(spec, dict) else None
        if not isinstance(launch, dict):
            continue
        if str(launch.get("compose") or "").strip() != compose_rel:
            continue
        # A profile service is never started by a plain `up`; running_profile_services covers it.
        if launch.get("compose_profile"):
            continue
        name = str(launch.get("service") or "").strip()
        if name:
            out.add(name)
    return sorted(out)


def pool_held_services(
    repo_root: Path,
    service: str,
    running: list[tuple[str, str, str]] | None,
    targets: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """(hold, start, running_targets) for a pull-time rebuild of services/<service>/.

    ``hold``: GPU-pool launch targets that are NOT running now. The pool stopped them on purpose
    (an eviction: diffusion-host while agent-gpu2 holds gpu2), so a rebuild builds their image and
    leaves them stopped. A plain ``up -d`` restarted them on a card the pool had given to another
    role; the lane controller then refuses every load and unload (seat_and_evicted_both_running)
    and the card stays in fault. Confirmed live 2026-10-16 04:02 UTC: one circe pull restarted
    diffusion-host under agent-gpu2 and no picture rendered for 15 h. ``running`` None (docker
    unreadable) holds every target: unknown state never starts something on a pool card.

    ``running_targets``: launch targets running now (the rebuild recreates them, as before).
    ``start``: the stack's other plain services. All three are empty when the pool launches
    nothing from this stack, and the caller keeps its normal ``up -d --build``.
    """
    if targets is None:
        targets = pool_launch_targets(repo_root, service)
    if not targets:
        return [], [], []
    suffix = f"{_SERVICES_DIR}/{service}/{_COMPOSE_FILE}"
    running_here: set[str] = set()
    for _project, compose_service, config_files in running or []:
        files = [f.strip().replace("\\", "/") for f in config_files.split(",") if f.strip()]
        if any(f == suffix or f.endswith("/" + suffix) for f in files):
            running_here.add(compose_service)
    hold = [t for t in targets if running is None or t not in running_here]
    up = [t for t in targets if t not in hold]
    compose = _load_compose(repo_root / _SERVICES_DIR / service / _COMPOSE_FILE)
    compose_services = compose.get("services") or {}
    if not isinstance(compose_services, dict):
        compose_services = {}
    start = sorted(
        name
        for name, spec in compose_services.items()
        if name not in targets and not (isinstance(spec, dict) and spec.get("profiles"))
    )
    return hold, start, up


def migration_report(migrations: list[str], repo_root: Path) -> list[str]:
    """Changed migrations that still exist (a deleted one has nothing to apply)."""
    return [rel for rel in migrations if (repo_root / rel).is_file()]


def _emit_host_summary(result: ResolveResult) -> None:
    if result.mesh_host:
        print(f"rebuild_affected_services: mesh host: {result.mesh_host}", file=sys.stderr)
    if result.host_allowlist:
        print(
            f"rebuild_affected_services: host allowlist active "
            f"({len(result.host_allowlist)} services, exclude_services.txt ignored)",
            file=sys.stderr,
        )
    if result.host_filtered_out:
        print(
            "rebuild_affected_services: affected on other hosts, skipped here: "
            + ", ".join(result.host_filtered_out),
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git ref before the merge/pull (default: ORIG_HEAD, MERGE_HEAD, …)")
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Explicit changed paths (skip git diff; useful for tests)",
    )
    parser.add_argument("--list-only", action="store_true", help="Print affected service names, one per line")
    parser.add_argument("--json", action="store_true", help="Print JSON {services, skipped, reasons}")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print '<service>\\t<build|recreate>' per affected service (recreate = bind-mounted change)",
    )
    parser.add_argument(
        "--migrations",
        action="store_true",
        help="Print each changed hand-applied SQL migration still in the tree, one per line",
    )
    parser.add_argument(
        "--running-profile-services",
        metavar="SERVICE",
        help="Print '<profile>\\t<compose service>' for running profile-gated services of SERVICE and exit",
    )
    parser.add_argument(
        "--pool-held-services",
        metavar="SERVICE",
        help="For a stack the GPU pool launches from, print 'hold\\t<name>' (stopped launch target: "
        "build, never start), 'running\\t<name>' (running launch target) and 'start\\t<name>' (other "
        "plain services), then exit; exit 3 when config/gpu_pool.yaml is unreadable",
    )
    parser.add_argument(
        "--skip-profile",
        action="append",
        default=[],
        help="With --running-profile-services: ignore services in this profile (repeatable)",
    )
    parser.add_argument(
        "--docker-ps-file",
        type=Path,
        default=None,
        help="Replay a saved docker ps dump instead of querying docker (tests)",
    )
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT, help="Repo root (for tests)")
    parser.add_argument(
        "--host",
        help="Mesh host name override (default: ORION_MESH_HOST, .env NODE_NAME, hostname)",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()

    if args.pool_held_services:
        # Exit 3 = "cannot tell what the pool owns here": the shell must not plain-`up` the stack.
        try:
            targets = pool_launch_targets(repo_root, args.pool_held_services)
        except PoolConfigError as exc:
            print(f"rebuild_affected_services: ERROR: GPU pool config unreadable: {exc}", file=sys.stderr)
            return 3
        if not targets:
            return 0
        ps_file = args.docker_ps_file
        if ps_file is None and os.environ.get("ORION_REBUILD_DOCKER_PS_FILE", "").strip():
            ps_file = Path(os.environ["ORION_REBUILD_DOCKER_PS_FILE"].strip())
        try:
            running = read_running_compose_containers(ps_file)
        except OSError:
            running = None
        if running is None:
            print(
                "rebuild_affected_services: WARNING: docker ps unavailable -- "
                "holding every GPU-pool launch target stopped",
                file=sys.stderr,
            )
        hold, start, up = pool_held_services(repo_root, args.pool_held_services, running, targets)
        for name in hold:
            print(f"hold\t{name}")
        for name in up:
            print(f"running\t{name}")
        for name in start:
            print(f"start\t{name}")
        return 0

    if args.running_profile_services:
        ps_file = args.docker_ps_file
        if ps_file is None and os.environ.get("ORION_REBUILD_DOCKER_PS_FILE", "").strip():
            ps_file = Path(os.environ["ORION_REBUILD_DOCKER_PS_FILE"].strip())
        running = read_running_compose_containers(ps_file)
        if running is None:
            print(
                "rebuild_affected_services: WARNING: docker ps unavailable -- "
                "not rebuilding any profile-gated services",
                file=sys.stderr,
            )
            return 0
        for profile, name in running_profile_services(
            repo_root, args.running_profile_services, running, set(args.skip_profile)
        ):
            print(f"{profile}\t{name}")
        return 0

    try:
        if args.paths is not None:
            paths = [p.replace("\\", "/") for p in args.paths]
        else:
            base = resolve_base_ref(repo_root, args.base)
            paths = changed_files(repo_root, base)
    except (ValueError, RuntimeError) as exc:
        print(f"rebuild_affected_services: ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        result = resolve_affected_services(paths, repo_root, mesh_host=args.host)
    except RuntimeError as exc:
        print(f"rebuild_affected_services: ERROR: {exc}", file=sys.stderr)
        return 2
    _emit_host_summary(result)

    if args.json:
        payload = {
            "services": result.services,
            "skipped": result.skipped,
            "reasons": result.reasons,
            "mesh_host": result.mesh_host,
            "host_allowlist": result.host_allowlist,
            "host_filtered_out": result.host_filtered_out,
            "recreate": result.recreate,
            "migrations": result.migrations,
        }
        print(json.dumps(payload, indent=2))
    elif args.migrations:
        for rel in migration_report(result.migrations, repo_root):
            print(rel)
    elif args.plan:
        recreate = set(result.recreate)
        for svc in result.services:
            print(f"{svc}\t{MODE_RECREATE if svc in recreate else MODE_BUILD}")
    elif args.list_only:
        for svc in result.services:
            print(svc)
    else:
        if not paths:
            print("rebuild_affected_services: no changed files detected")
        print(f"Affected services ({len(result.services)}):")
        for svc in result.services:
            print(f"  - {svc}")
        if result.skipped:
            print(f"Skipped paths ({len(result.skipped)}):")
            for path in result.skipped:
                print(f"  - {path}: {result.reasons.get(path, '?')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
