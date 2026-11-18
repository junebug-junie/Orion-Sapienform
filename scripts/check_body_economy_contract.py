#!/usr/bin/env python3
"""Body-economy gate: "no unregistered veto" (Phase 2 Gate 1, early slice in Patch A1).

Every module that reads Orion's regulation, arousal, thermal or price state must be listed
in ``config/physiology/contract.yaml``: either under ``readers`` (with a role: spinal,
bidder or display) or under ``sources`` (the modules that define or produce that state).
The point is that no new ``if arousal == "strained": return`` can land without someone
writing down why, so the strained pile-up cannot grow back while Phase 1 retires it.

Detection is structural (AST), never a list of module names. A module "reads the state"
when any of these is true:

1. it imports from a watched module (``WATCHED_MODULE_PREFIXES``), e.g.
   ``from orion.regulation.arousal_view import AROUSAL_STRAINED_REASON``;
2. it references a watched symbol (``WATCHED_NAMES``) as a name or an attribute, e.g.
   ``schemas.RegulationStateV1``;
3. it holds a watched string constant (``WATCHED_STRINGS``) as a literal, e.g. the Redis key
   ``"orion:regulation:latest"`` or a locally re-spelled ``"arousal_strained"`` reason;
4. it compares a value against the arousal level literal ``"strained"`` (``x == "strained"``,
   ``x != "strained"``, ``x in ("engaged", "strained")``) -- the shape of every strained veto,
   including ones whose level arrives through an injected callable.

Limits, stated plainly: a module that gets the state only through a wrapper object passed
in by its caller, with no watched import, name or literal of its own, is not seen. The
wrapper's own module and the caller that wires it are.

The gate fails when:

* a module reads the state but is in neither ``readers`` nor ``sources`` (unregistered);
* a listed module no longer exists or no longer reads the state (stale entry);
* the YAML is malformed (missing table, bad role, duplicate entry, empty reason).

Deterministic, no live data, no imports of repo code. Run from the repo root:

    python3 scripts/check_body_economy_contract.py
    python3 scripts/check_body_economy_contract.py --inventory   # every hit + evidence
"""

from __future__ import annotations

import argparse
import ast
import functools
import multiprocessing
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "config" / "physiology" / "contract.yaml"

ROLES = ("spinal", "bidder", "display")

# Importing anything from these modules (or their submodules) is reading the state.
WATCHED_MODULE_PREFIXES = (
    "orion.regulation.arousal",       # the arousal reducer (strain causes, reflex inputs)
    "orion.regulation.arousal_view",  # level + reason constants a reader acts on
    "orion.regulation.arousal_read",  # one Redis GET of orion:regulation:latest
    "orion.regulation.strain_news",   # strain onset/rest news derived from the reading
    "orion.schemas.regulation",  # RegulationStateV1, REGULATION_STATE_REDIS_KEY
    "orion.autonomy.thermal_gate",
    "orion.autonomy.cabinet_heat",
    "orion.physiology",          # Phase 1 Body/Budget organs (PhysiologyStateV1, prices)
)

# Referencing any of these as a bare name or attribute is reading the state.
WATCHED_NAMES = frozenset({
    "RegulationStateV1",
    "REGULATION_STATE_REDIS_KEY",
    "parse_regulation_state",
    "ArousalView",
    "ArousalReader",
    "read_arousal_sync",
    "read_arousal_async",
    "AROUSAL_STRAINED_REASON",
    "AROUSAL_ENGAGED_REASON",
    "AROUSAL_ENGAGED_COOLDOWN_REASON",
    "ThermalState",
    "ThermalVerdict",
    "CabinetHeatReading",
    "read_cabinet_heat",
    "PhysiologyStateV1",
    "thermal_gate_snapshot",     # orion-thought's thermal verdict for GPU work
    "allows_gpu_work",           # the verdict's go/no-go field, read as an attribute
})

# A relative import whose target module has one of these leaf names is reading the state
# (``from .arousal_view import x`` inside orion/regulation, ``from .arousal_reader import``
# inside the Hub). Absolute imports are matched by WATCHED_MODULE_PREFIXES instead.
WATCHED_RELATIVE_LEAVES = frozenset({
    "arousal", "arousal_view", "arousal_read", "strain_news", "regulation",
    "thermal_gate", "cabinet_heat", "arousal_reader", "physiology",
})

# Holding any of these as a string literal is reading the state.
WATCHED_STRINGS = frozenset({
    "orion:regulation:latest",   # the Redis key durable-runs writes every regulate step
    "arousal_level",             # the level field, read as a dict key off HTTP/JSON payloads
    "arousal_strained",          # the strained-veto reason, re-spelled locally in places
    "arousal_engaged",
    "arousal_engaged_cooldown",
    "cabinet_hot",               # hardware-watch's 34 C reflex reason
    "allows_gpu_work",           # thermal verdict field, read as a dict key
    "thermal_refused",           # visual-chain refusal reason on heat
    "deferred_thermal",          # visual-chain deferral outcome on heat
})

# The arousal level that every strained veto compares against.
STRAINED_LEVEL = "strained"

# Cheap text pre-filter: a file that contains none of these cannot match any rule above
# (every watched module, name and literal contains one of them, case-insensitively).
_PREFILTER = ("regulation", "arousal", "thermal", "cabinet", "physiology", "strained", "allows_gpu_work")

DEFAULT_INCLUDE = ("**/*.py",)
DEFAULT_EXCLUDE = (
    "**/tests/**", "**/evals/**", "**/test_*.py", "**/conftest.py",
    "**/.venv/**", "**/node_modules/**", ".worktrees/**", ".claude/**", "graphify-out/**",
)


@dataclass
class Hit:
    path: str
    evidence: list[str] = field(default_factory=list)


def _watched_module(mod: str) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in WATCHED_MODULE_PREFIXES)


def _is_strained(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == STRAINED_LEVEL


def _compares_to_strained(node: ast.Compare) -> bool:
    """``x == "strained"`` / ``!=`` (either side), or ``x in (<literal set holding "strained">)``.
    ``"strained" in some_text`` is a substring test, not a level check, and is ignored."""
    left = node.left
    for op, right in zip(node.ops, node.comparators):
        if isinstance(op, (ast.Eq, ast.NotEq)) and (_is_strained(left) or _is_strained(right)):
            return True
        if (isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List, ast.Set))
                and any(_is_strained(e) for e in right.elts)):
            return True
        left = right
    return False


def _holds_strained(node: ast.AST) -> bool:
    """A literal collection holding "strained" (``BLOCK = {"strained"}``) -- a veto set."""
    return isinstance(node, (ast.Tuple, ast.List, ast.Set)) and any(_is_strained(e) for e in node.elts)


def _relative_target(module: str | None) -> list[str]:
    """Module path segments of a relative import (``from ..a.b import y`` -> a, b)."""
    return [p for p in (module or "").split(".") if p]


class Unparseable(Exception):
    pass


def scan_source(text: str, rel: str = "") -> list[str]:
    """Evidence lines for one module's source; empty when it does not read the state.
    Raises ``Unparseable`` for a syntax error, so the gate fails closed on it."""
    low = text.lower()
    if not any(tok in low for tok in _PREFILTER):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise Unparseable(f"{exc.msg} (line {exc.lineno})") from exc
    evidence: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            leaves = _relative_target(node.module)
            leaves += [a.name for a in node.names]
            hit = sorted(set(leaves) & WATCHED_RELATIVE_LEAVES)
            if hit:
                dots = "." * node.level
                evidence.append(f"L{node.lineno} from {dots}{node.module or ''} import ... ({','.join(hit)})")
            for a in node.names:
                if a.name in WATCHED_NAMES:
                    evidence.append(f"L{node.lineno} import name {a.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _watched_module(node.module):
                names = ",".join(a.name for a in node.names)
                evidence.append(f"L{node.lineno} from {node.module} import {names}")
            else:
                for a in node.names:
                    if _watched_module(f"{node.module}.{a.name}"):
                        evidence.append(f"L{node.lineno} from {node.module} import {a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _watched_module(a.name):
                    evidence.append(f"L{node.lineno} import {a.name}")
        elif isinstance(node, ast.Name) and node.id in WATCHED_NAMES:
            evidence.append(f"L{node.lineno} name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in WATCHED_NAMES:
            evidence.append(f"L{node.lineno} attribute .{node.attr}")
        elif isinstance(node, ast.alias) and node.name in WATCHED_NAMES:
            evidence.append(f"L{getattr(node, 'lineno', '?')} import name {node.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in WATCHED_STRINGS:
            evidence.append(f"L{node.lineno} literal {node.value!r}")
        elif isinstance(node, ast.Compare) and _compares_to_strained(node):
            evidence.append(f"L{node.lineno} compares against {STRAINED_LEVEL!r}")
        elif isinstance(node, ast.MatchValue) and _is_strained(node.value):
            evidence.append(f"L{node.lineno} matches case {STRAINED_LEVEL!r}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and (
                _is_strained(node.value) or _holds_strained(node.value)):
            evidence.append(f"L{node.lineno} binds a name to {STRAINED_LEVEL!r}")
    # Stable order, no duplicates.
    seen: set[str] = set()
    return [e for e in evidence if not (e in seen or seen.add(e))]


@functools.lru_cache(maxsize=None)
def _glob_re(pattern: str) -> re.Pattern[str]:
    """Path glob -> regex. ``**/`` matches zero or more directories, ``*`` stays in one."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matches(rel: str, globs: Iterable[str]) -> bool:
    return any(_glob_re(g).match(rel) for g in globs)


def iter_candidate_files(root: Path, include: Iterable[str], exclude: Iterable[str]) -> list[str]:
    out: set[str] = set()
    for pattern in include:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if _matches(rel, exclude):
                continue
            out.add(rel)
    return sorted(out)


def _git_prefiltered(root: Path, include: Iterable[str], exclude: Iterable[str]) -> list[str] | None:
    """Fast path: let ``git grep`` (tracked + untracked, .gitignore respected) apply the text
    pre-filter. Returns None when ``root`` is not a git work tree, so callers fall back."""
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
        return None
    cmd = ["git", "-C", str(root), "-c", "core.quotePath=false", "grep", "-l", "-z", "-i",
           "--untracked", "-E", "|".join(_PREFILTER), "--"]
    # ":(glob)" magic: git understands "**/" itself, so custom include globs keep their meaning.
    cmd += [f":(glob){g}" for g in include]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode not in (0, 1):
        return None
    out = []
    for rel in proc.stdout.split("\0"):
        rel = rel.strip()
        if not rel or not _matches(rel, include):
            continue
        if _matches(rel, exclude):
            continue
        out.append(rel)
    return sorted(out)


def _scan_file(job: tuple[str, str]) -> tuple[str, list[str], str | None]:
    """(path, evidence, problem). A file the gate cannot read or parse is a problem, never a
    silent pass: an unreadable module could be hiding a veto."""
    root, rel = job
    try:
        text = (Path(root) / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return rel, [], f"cannot read: {type(exc).__name__}"
    try:
        return rel, scan_source(text, rel), None
    except Unparseable as exc:
        return rel, [], f"cannot parse: {exc}"


def _run_pool(jobs: list[tuple[str, str]], workers: int) -> list[tuple[str, list[str], str | None]]:
    """Fan out over processes. Uses fork where the platform has it (the module may have been
    loaded by file path, so a spawned child could not re-import ``_scan_file``); any pool
    failure falls back to the serial path, which gives the same result, only slower."""
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        return [_scan_file(j) for j in jobs]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            return list(pool.map(_scan_file, jobs, chunksize=8))
    except Exception:  # noqa: BLE001 -- serial fallback is always correct
        return [_scan_file(j) for j in jobs]


def scan_repo(root: Path, include: Iterable[str], exclude: Iterable[str]) -> tuple[dict[str, Hit], list[str]]:
    """(hits, problems). ``problems`` are files the gate could not read or parse."""
    files = _git_prefiltered(root, include, exclude)
    if files is None:
        files = iter_candidate_files(root, include, exclude)
    jobs = [(str(root), rel) for rel in files]
    # Parsing ~250 modules serially takes ~2.5 s; fan out so the gate stays under its 2 s
    # budget. Results are keyed by path and sorted, so the output does not depend on order.
    workers = min(8, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 32:
        results = _run_pool(jobs, workers)
    else:
        results = [_scan_file(j) for j in jobs]
    hits = {rel: Hit(rel, ev) for rel, ev, _ in sorted(results, key=lambda r: r[0]) if ev}
    problems = [f"UNSCANNABLE {rel}: {why}. Fix: make the file valid UTF-8 Python, or exclude it "
                f"via `scan.exclude` in the contract if it is not code."
                for rel, _, why in sorted(results, key=lambda r: r[0]) if why]
    return hits, problems


@dataclass
class Contract:
    readers: dict[str, dict]
    sources: dict[str, dict]
    include: tuple[str, ...]
    exclude: tuple[str, ...]


def load_contract(path: Path) -> tuple[Contract | None, list[str]]:
    errors: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [f"{path}: contract file missing"]
    except yaml.YAMLError as exc:
        return None, [f"{path}: malformed YAML: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{path}: top level must be a mapping"]

    for table in ("protected_paths", "readers", "sources"):
        if not isinstance(data.get(table), list):
            errors.append(f"{path}: `{table}` must be a list")
    if errors:
        return None, errors

    for i, row in enumerate(data["protected_paths"]):
        if not isinstance(row, dict) or not row.get("glob") or not row.get("organ"):
            errors.append(f"{path}: protected_paths[{i}] needs non-empty `glob` and `organ`")

    def _rows(table: str, need_role: bool) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i, row in enumerate(data[table]):
            if not isinstance(row, dict):
                errors.append(f"{path}: {table}[{i}] must be a mapping")
                continue
            mod = row.get("module")
            if not isinstance(mod, str) or not mod.strip():
                errors.append(f"{path}: {table}[{i}] needs a non-empty `module` path")
                continue
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                errors.append(f"{path}: {table} entry {mod} needs a non-empty `reason`")
            if need_role and row.get("role") not in ROLES:
                errors.append(f"{path}: readers entry {mod} has role {row.get('role')!r}; must be one of {ROLES}")
            if mod in out:
                errors.append(f"{path}: {table} lists {mod} twice")
            out[mod] = row
        return out

    readers = _rows("readers", need_role=True)
    sources = _rows("sources", need_role=False)
    for both in sorted(set(readers) & set(sources)):
        errors.append(f"{path}: {both} is in both `readers` and `sources`; pick one")

    scan = data.get("scan") or {}
    include = tuple(scan.get("include") or DEFAULT_INCLUDE)
    exclude = tuple(scan.get("exclude") or DEFAULT_EXCLUDE)
    return Contract(readers, sources, include, exclude), errors


# The gate holds every watched name and literal as data; it is not a reader.
SELF_PATH = "scripts/check_body_economy_contract.py"


def check(root: Path, contract_path: Path) -> tuple[list[str], dict[str, Hit]]:
    contract, errors = load_contract(contract_path)
    if contract is None:
        return errors, {}
    hits, problems = scan_repo(root, contract.include, contract.exclude)
    hits.pop(SELF_PATH, None)
    errors.extend(problems)
    try:
        rel_contract = contract_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel_contract = contract_path

    registered = set(contract.readers) | set(contract.sources)
    for rel in sorted(set(hits) - registered):
        ev = "; ".join(hits[rel].evidence[:3])
        errors.append(
            f"UNREGISTERED READER {rel}: reads regulation/thermal/price state ({ev}). "
            f"Fix: add it to `readers` in {rel_contract} with role spinal|bidder|display and a "
            f"one-line reason -- or, better, do not gate on the state (Phase 1 moves vetoes to bid())."
        )
    for table, rows in (("readers", contract.readers), ("sources", contract.sources)):
        for rel in sorted(rows):
            if not (root / rel).is_file():
                errors.append(f"STALE ENTRY {rel}: listed in `{table}` but the file does not exist. "
                              f"Fix: remove or rename the entry in {rel_contract}.")
            elif rel not in hits:
                errors.append(f"STALE ENTRY {rel}: listed in `{table}` but no longer reads the state "
                              f"(or is outside the scan globs). Fix: remove the entry from {rel_contract}.")
    return errors, hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--contract", type=Path, default=None)
    ap.add_argument("--inventory", action="store_true", help="print every detected module with evidence")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    contract_path = args.contract or (root / "config" / "physiology" / "contract.yaml")

    errors, hits = check(root, contract_path)
    if args.inventory:
        for rel in sorted(hits):
            print(rel)
            for e in hits[rel].evidence[:6]:
                print(f"    {e}")
    if errors:
        print(f"body-economy contract: FAIL ({len(errors)} problem(s))", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"body-economy contract: OK ({len(hits)} state-reading modules, all registered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
