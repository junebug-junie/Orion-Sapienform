#!/usr/bin/env python3
"""Report (and optionally remove) dead keys in live service `.env` files.

A key is DEAD when nothing reads it any more:

  * it is not in the service's `.env_example` (the operator contract), and
  * no non-comment code reads it: the service's own files (settings.py, app/, scripts, compose,
    Dockerfile), the shared `orion/` package every container imports, or `config/`.

`scripts/sync_local_env_from_example.py` only ever ADDS keys, so every key a PR deletes stays in
the live `.env` files forever. Three GPU pool PR reports (4.6, 5.6, 6.3) listed them by hand; this
replaces the hand list (GPU pool stage 6.6, spec 2026-09-30-gpu-pool-stage6-telemetry-reducers-
lockdown.md, "Env keys").

How "reads it" is decided (deliberately conservative -- --apply deletes, so a false "dead" is the
expensive mistake):

  * Python files are parsed with `ast`. Names (pydantic field names, case-insensitive), attribute
    names, keyword names and string constants count; comments and docstrings do not, so a comment
    saying "FOO was removed" does not keep FOO alive.
  * Other files: every identifier on a non-comment line counts.
  * `env_prefix = "X_"` in a settings class makes `X_<field>` read.
  * A string constant ending in `_` (e.g. a `startswith("HUB_")` scan) makes every key under that
    prefix read.

--apply removes only KNOWN_DEAD keys unless --include-heuristic is given, and refuses when the
code tree is on a different commit than the checkout holding the .env files.

`.env.bak.<ts>` backups hold the same secrets as `.env`. `.gitignore` covers them (`.env.*`); a
`COPY . .` docker build is only safe if the service's `.dockerignore` excludes them too.

Keys that are NEVER removed unless listed in KNOWN_DEAD below:
  secret-named keys and NEVER_SYNC_KEYS (both from sync_local_env_from_example.py). They are
  listed as "protected" so a human can decide.

Keys never reported dead at all: PROTECTED_PATTERNS (e.g. LLM_LANE_* until the stage 6.4 lane
census shows no caller sends options.lane).

KNOWN_DEAD: keys the GPU pool PRs deleted on purpose. They are dead even if secret-named. If one
is still read by code, or is back in .env_example, it is reported as a CONFLICT and not removed.

Usage:
    python3 scripts/report_dead_env_keys.py                # --report (default), read-only
    python3 scripts/report_dead_env_keys.py --json
    python3 scripts/report_dead_env_keys.py --apply        # KNOWN_DEAD only; writes .env.bak.<UTC ts> first
    python3 scripts/report_dead_env_keys.py --apply --include-heuristic   # also the code-scan verdicts
    python3 scripts/report_dead_env_keys.py --service orion-hub --service orion-thought
    python3 scripts/report_dead_env_keys.py --env-root /mnt/scripts/Orion-Sapienform

`--root` is the code tree (default: this script's checkout); `--env-root` is where the live
`.env` files are (default: the primary checkout, because a linked worktree has no `.env`). On
circe, run it from circe's own checkout: `ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform &&
python3 scripts/report_dead_env_keys.py'`. Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import fnmatch
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

REPO = pathlib.Path(__file__).resolve().parents[1]

# Deleted on purpose by the GPU pool stages; each verified against code on 2026-10-01 (only
# comment / docstring mentions remain). Pattern -> the PR that deleted it.
KNOWN_DEAD: dict[str, str] = {
    "GPU2_*": "stage 5.6 deleted the gpu2 bridge (GPU2_ENABLED/_DIFFUSION_URL/_AGENT_URL/..., "
              "GPU2_AUTHORITY*, renamed GPU2_POOL_FENCE_STATE_PATH/GPU2_DRAIN_TIMEOUT_SEC)",
    "GPU_LANE_CONTROLLER_TOKEN": "stage 5.6 (controller token)",
    "WM_GPU2_CAPACITY_*": "stage 5.4 (world-model /capacity permit)",
    "ORION_VISUAL_CHAIN_GPU2_CAPACITY_*": "stage 5.4 (visual chain /capacity permit)",
    "ORION_VISUAL_ELASTIC_*": "stage 5.4 (visual elastic status)",
    "GPU_POOL_VISUAL_ACTIVITY_URL": "stage 5.4",
    "GPU_POOL_ACTUATE_ROLES": "stage 5.7",
    "GPU_LANE_MAP_*": "stage 5.5 (hub GPU lane maps)",
    "DIFFUSION_POWER_INTENT_GPU_INDEX": "stage 5.5",
    "DURABLE_RUNS_CAPACITY_ENABLED": "stage 5.6",
    "DURABLE_RUNS_LEASE_SECONDS": "stage 5.6",
    "DURABLE_RUNS_ELASTIC_*": "stage 4.6/5.x (elastic gpu2 borrow moved into config/gpu_pool.yaml)",
    "LLM_GATEWAY_LEASE_*": "stage 4.6 (gateway lease validation)",
    "LLM_GATEWAY_CAPACITY_*": "stage 4.6 (gateway /capacity)",
    "HUB_CURIOSITY_LEASE_VALIDATION_URL": "stage 4.6",
    "HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED": "stage 4.6",
    "HUB_LLM_GATEWAY_URL": "stage 6.3 (its only reader was the /routes read)",
    "CORTEX_EXEC_LLM_GATEWAY_URL": "stage 6.3",
    "CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED": "stage 6.3",
}

# Never reported dead, whatever the scan says. Pattern -> why.
PROTECTED_PATTERNS: dict[str, str] = {
    "LLM_LANE_*": "excluded until the stage 6.4 lane census shows no caller sends options.lane",
    "COMPOSE_*": "read by docker compose itself, not by service code",
    "DOCKER_*": "read by docker / compose tooling",
    # env_file passes the whole .env into the container: these are read by the image, CUDA, the
    # llama.cpp server or third-party libraries, never named in repo code.
    "LLAMA_ARG_*": "read by the llama.cpp server binary",
    "HF_*": "read by huggingface libraries", "HUGGINGFACE_*": "read by huggingface libraries",
    "TRANSFORMERS_*": "read by transformers", "TORCH_*": "read by torch", "PYTORCH_*": "read by torch",
    "CUDA_*": "read by CUDA", "NVIDIA_*": "read by the nvidia container runtime", "NCCL_*": "read by NCCL",
    "OMP_*": "read by OpenMP", "MKL_*": "read by MKL", "TZ": "read by libc", "LANG": "read by libc",
    "LC_*": "read by libc", "PYTHON*": "read by the python interpreter", "PATH": "read by the shell",
    "PG*": "read by libpq", "POSTGRES_*": "read by the postgres image", "GF_*": "read by the grafana image",
    "NEO4J_*": "read by the neo4j image", "REDIS_*": "read by the redis image",
    "UVICORN_*": "read by uvicorn", "GUNICORN_*": "read by gunicorn", "WEB_CONCURRENCY": "read by uvicorn",
    "OLLAMA_*": "read by the ollama image", "VLLM_*": "read by vllm", "XDG_*": "read by libraries",
}

_EXCLUDED_PARTS = {"tests", "evals", "docs", "graphify-out", "venv", ".venv", "node_modules", "__pycache__"}
_TEXT_SUFFIXES = {".yml", ".yaml", ".sh", ".js", ".mjs", ".ts", ".json", ".toml", ".conf", ".ini", ".cfg", ".sql", ".txt"}
_TEXT_NAMES = {"Dockerfile", "Makefile", "entrypoint", ".env_example"}
_MAX_FILE_BYTES = 4 * 1024 * 1024  # bigger files are bundled data, not config readers
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


# ---- secret / never-sync rules: reuse the sync script, fall back to a copy ------------------
def _load_sync_rules(root: pathlib.Path):
    try:
        sys.path.insert(0, str(root / "scripts"))
        import sync_local_env_from_example as sync  # type: ignore

        return frozenset(sync.NEVER_SYNC_KEYS), sync.is_secret_key
    except Exception:  # circe's checkout may be older than this script
        secret_segments = {"TOKEN", "SECRET", "PASSWORD", "PASSWD", "PASS", "PWD", "DSN", "CREDENTIAL", "CREDENTIALS"}
        secret_pairs = {("API", "KEY"), ("ACCESS", "KEY"), ("ADMIN", "KEY"), ("PRIVATE", "KEY"), ("SECRET", "KEY")}

        def is_secret_key(key: str) -> bool:
            parts = key.upper().split("_")
            return any(p in secret_segments for p in parts) or any(pair in secret_pairs for pair in zip(parts, parts[1:]))

        never = frozenset({"PUBLISH_CORTEX_EXEC_GRAMMAR", "ORION_BUS_URL", "REOLINK_URL", "WALKWAY_RTSP_URL",
                           "RECALL_GRAPHITI_IN_CHAT", "RECALL_GRAPHITI_ADAPTER_URL", "ILO_HOST", "ILO_USERNAME",
                           "ILO_PASSWORD", "HUB_CURIOSITY_GRAPH_ORION_PASSWORD", "CURSOR_API_KEY",
                           "ORION_CURIOSITY_GRAPH_PASSWORD", "ENERGY_USAGE_POINT_ID"})
        return never, is_secret_key
    finally:
        if sys.path and sys.path[0] == str(root / "scripts"):
            sys.path.pop(0)


# ---- what code reads ----------------------------------------------------------------------
@dataclass
class ReadSet:
    tokens: set[str] = field(default_factory=set)       # upper-cased identifiers
    prefixes: set[str] = field(default_factory=set)     # upper-cased "FOO_" string constants
    env_prefixes: set[str] = field(default_factory=set)

    def update(self, other: "ReadSet") -> None:
        self.tokens |= other.tokens
        self.prefixes |= other.prefixes
        self.env_prefixes |= other.env_prefixes

    def reads(self, key: str) -> bool:
        k = key.upper()
        if k in self.tokens:
            return True
        if any(k.startswith(p) and k[len(p):] in self.tokens for p in self.env_prefixes):
            return True
        return any(k.startswith(p) for p in self.prefixes)


def _docstring_ids(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _python_reads(text: str) -> ReadSet:
    rs = ReadSet()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _text_reads(text)
    skip = _docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            rs.tokens.add(node.id.upper())
        elif isinstance(node, ast.Attribute):
            rs.tokens.add(node.attr.upper())
        elif isinstance(node, ast.arg):
            rs.tokens.add(node.arg.upper())
        elif isinstance(node, ast.keyword) and node.arg:
            rs.tokens.add(node.arg.upper())
            if node.arg == "env_prefix" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                rs.env_prefixes.add(node.value.value.upper())
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for t in targets:
                if isinstance(t, ast.Name) and t.id == "env_prefix" and isinstance(value, ast.Constant) \
                        and isinstance(value.value, str):
                    rs.env_prefixes.add(value.value.upper())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            s = node.value
            for tok in _IDENT_RE.findall(s):
                rs.tokens.add(tok.upper())
            stripped = s.strip()
            if re.fullmatch(r"[A-Z][A-Z0-9_]*_", stripped) and len(stripped) >= 4:
                rs.prefixes.add(stripped)
    return rs


def _text_reads(text: str, *, sql: bool = False) -> ReadSet:
    rs = ReadSet()
    # `--` starts a comment only in SQL; in compose/shell it is a flag (`--max-rows ${KEY}`).
    comment_marks = ("#", "//", "--") if sql else ("#", "//")
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith(comment_marks):
            continue
        code = s.split(" #", 1)[0]
        for tok in _IDENT_RE.findall(code):
            rs.tokens.add(tok.upper())
    return rs


def _iter_files(base: pathlib.Path, root: pathlib.Path):
    if not base.exists():
        return
    paths = [base] if base.is_file() else base.rglob("*")
    for p in paths:
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        if any(part in _EXCLUDED_PARTS for part in rel.parts):
            continue
        if p.name == ".env" or p.name.startswith(".env.bak"):
            continue
        if p.name == "report_dead_env_keys.py":  # this tool names every KNOWN_DEAD key
            continue
        yield p


_FILE_CACHE: dict[pathlib.Path, ReadSet | None] = {}


def _file_reads(p: pathlib.Path) -> ReadSet | None:
    if p in _FILE_CACHE:
        return _FILE_CACHE[p]
    rs: ReadSet | None = None
    if p.suffix == ".py" or p.suffix in _TEXT_SUFFIXES or p.name in _TEXT_NAMES or p.name.startswith("docker-compose"):
        try:
            if p.stat().st_size > _MAX_FILE_BYTES:  # fixtures / bundled data, not config readers
                raise OSError("too large")
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            text = None
        if text is not None:
            rs = _python_reads(text) if p.suffix == ".py" else _text_reads(text, sql=p.suffix == ".sql")
    _FILE_CACHE[p] = rs
    return rs


def reads_under(base: pathlib.Path, root: pathlib.Path) -> ReadSet:
    """What the code under `base` reads. `.env_example` is not code: it is checked separately."""
    rs = ReadSet()
    for p in _iter_files(base, root):
        if p.name == ".env_example":
            continue
        got = _file_reads(p)
        if got is not None:
            rs.update(got)
    return rs


def _opens_multiline(raw: str) -> str | None:
    """The quote char if this KEY=value line opens a quoted value it does not close."""
    value = raw.split("=", 1)[1].lstrip() if "=" in raw else ""
    if value[:1] in ("'", '"') and value.count(value[0]) % 2 == 1:
        return value[0]
    return None


def parse_env(path: pathlib.Path) -> list[tuple[int, str, bool]]:
    """(line number, key, value spans several lines). Continuation lines are not keys."""
    out: list[tuple[int, str, bool]] = []
    if not path.is_file():
        return out
    open_quote: str | None = None
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if open_quote:
            if raw.count(open_quote) % 2 == 1:
                open_quote = None
            continue
        if raw.lstrip().startswith("#"):
            continue
        m = _ENV_LINE_RE.match(raw)
        if m:
            open_quote = _opens_multiline(raw)
            out.append((i, m.group(1), open_quote is not None))
    return out


def env_keys(path: pathlib.Path) -> list[tuple[int, str]]:
    return [(i, k) for i, k, _ in parse_env(path)]


# ---- per-service verdict ------------------------------------------------------------------
@dataclass
class ServiceReport:
    service: str
    env_path: str
    orphan: bool = False                                     # service dir gone from the code tree
    dead: list[str] = field(default_factory=list)            # removable
    protected: list[str] = field(default_factory=list)       # looks dead, but secret / never-sync
    conflicts: list[str] = field(default_factory=list)       # KNOWN_DEAD but still read / in example
    reasons: dict[str, str] = field(default_factory=dict)


def _match(key: str, patterns: dict[str, str]) -> str | None:
    for pat in patterns:
        if fnmatch.fnmatchcase(key, pat):
            return pat
    return None


def analyse(
    service: str,
    env_path: pathlib.Path,
    example_path: pathlib.Path,
    reads: ReadSet,
    never_sync: frozenset[str],
    is_secret,
) -> ServiceReport:
    rep = ServiceReport(service=service, env_path=str(env_path))
    example = {k for _, k in env_keys(example_path)}
    seen: set[str] = set()
    multiline = {k for _, k, multi in parse_env(env_path) if multi}
    for _, key in env_keys(env_path):
        if key in seen:
            continue
        seen.add(key)
        if _match(key, PROTECTED_PATTERNS):
            continue
        if key in multiline and key not in example and not reads.reads(key):
            rep.protected.append(key)
            rep.reasons[key] = "multi-line value: looks dead, remove by hand"
            continue
        known = _match(key, KNOWN_DEAD)
        in_example = key in example
        read = reads.reads(key)
        if known:
            if in_example or read:
                rep.conflicts.append(key)
                rep.reasons[key] = (
                    f"listed dead ({KNOWN_DEAD[known]}) but still "
                    + ("in .env_example" if in_example else "read by code")
                )
            else:
                rep.dead.append(key)
                rep.reasons[key] = KNOWN_DEAD[known]
            continue
        if in_example or read:
            continue
        if key in never_sync or is_secret(key):
            rep.protected.append(key)
            rep.reasons[key] = "secret-named or NEVER_SYNC: not removed unless listed in KNOWN_DEAD"
        else:
            rep.dead.append(key)
            rep.reasons[key] = "not in .env_example and no code reads it"
    return rep


def default_env_root(root: pathlib.Path) -> pathlib.Path:
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=root, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return root
    cand = pathlib.Path(common).resolve().parent if common else root
    return cand if (cand / "services").is_dir() else root


def build_reports(root: pathlib.Path, env_root: pathlib.Path, services: list[str] | None) -> list[ServiceReport]:
    never_sync, is_secret = _load_sync_rules(root)
    shared = ReadSet()
    for top in ("orion", "config"):
        shared.update(reads_under(root / top, root))
    reports: list[ServiceReport] = []
    names = sorted(p.parent.name for p in env_root.glob("services/*/.env"))
    if services:
        names = [n for n in names if n in services]
    for name in names:
        code_dir = root / "services" / name
        if not code_dir.is_dir() or not any(
            not f.name.startswith(".env") or f.name == ".env_example" for f in code_dir.iterdir()
        ):
            # The service was deleted from the code tree (or this checkout is older than the
            # host's). Every key would read as dead; report the file, never edit it.
            reports.append(ServiceReport(service=name, env_path=str(env_root / "services" / name / ".env"), orphan=True))
            continue
        reads = ReadSet()
        reads.update(shared)
        reads.update(reads_under(root / "services" / name, root))
        reports.append(analyse(
            name, env_root / "services" / name / ".env", root / "services" / name / ".env_example",
            reads, never_sync, is_secret,
        ))
    if not services and (env_root / ".env").is_file():
        # The root .env feeds every compose (`--env-file .env`): anything any service reads keeps it.
        reads = ReadSet()
        reads.update(shared)
        reads.update(reads_under(root / "services", root))
        reads.update(reads_under(root / "scripts", root))
        for p in root.glob("docker-compose*.yml"):
            reads.update(_text_reads(p.read_text(encoding="utf-8")))
        reports.append(analyse("<root>", env_root / ".env", root / ".env_example", reads, never_sync, is_secret))
    return reports


def removable(rep: ServiceReport, known_only: bool) -> list[str]:
    if rep.orphan:
        return []
    if known_only:
        return [k for k in rep.dead if _match(k, KNOWN_DEAD)]
    return list(rep.dead)


def apply_removals(rep: ServiceReport, stamp: str, known_only: bool = False) -> pathlib.Path | None:
    dead = set(removable(rep, known_only))
    if not dead:
        return None
    path = pathlib.Path(rep.env_path)
    backup = path.with_name(f".env.bak.{stamp}")
    shutil.copy2(path, backup)
    kept: list[str] = []
    open_quote: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if open_quote:  # continuation of a multi-line value: never a key line
            if raw.count(open_quote) % 2 == 1:
                open_quote = None
            kept.append(raw)
            continue
        m = _ENV_LINE_RE.match(raw)
        if m and not raw.lstrip().startswith("#"):
            open_quote = _opens_multiline(raw)
            if m.group(1) in dead and open_quote is None:
                continue
        kept.append(raw)
    # Atomic: a crash mid-write must not leave a half-written .env.
    tmp = path.with_name(f".env.tmp.{stamp}")
    tmp.write_text("".join(kept), encoding="utf-8")
    shutil.copymode(path, tmp)
    os.replace(tmp, path)
    return backup


def _head(path: pathlib.Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="list dead keys (default; read-only)")
    mode.add_argument("--apply", action="store_true", help="remove dead keys, after writing .env.bak.<ts>")
    ap.add_argument("--known-only", action="store_true",
                    help="report only KNOWN_DEAD keys (the GPU pool deletions)")
    ap.add_argument("--include-heuristic", action="store_true",
                    help="with --apply: also remove keys found dead by the code scan, not just KNOWN_DEAD")
    ap.add_argument("--allow-tree-mismatch", action="store_true",
                    help="with --apply: allow the code tree (--root) to be on a different commit than the "
                         "checkout holding the .env files")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", type=pathlib.Path, default=REPO, help="code tree (default: this checkout)")
    ap.add_argument("--env-root", type=pathlib.Path, default=None,
                    help="where the live .env files are (default: the primary checkout)")
    ap.add_argument("--service", action="append", default=None, help="limit to these services (repeatable)")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    env_root = (args.env_root or default_env_root(root)).resolve()
    if args.apply:
        # --apply deletes by default only KNOWN_DEAD keys; the heuristic needs an explicit opt-in.
        args.known_only = not args.include_heuristic
        # The verdict is about the code that READS these files. A feature worktree that removed a
        # reader the running deploy still has would delete a live key.
        if root != env_root and not args.allow_tree_mismatch and _head(root) != _head(env_root):
            print(f"refusing --apply: code tree {root} (HEAD {_head(root)}) is not the commit of the checkout "
                  f"holding the .env files {env_root} (HEAD {_head(env_root)}). Run from {env_root}, or pass "
                  f"--allow-tree-mismatch.", file=sys.stderr)
            return 3
    reports = build_reports(root, env_root, args.service)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups: dict[str, str] = {}
    if args.apply:
        for rep in reports:
            b = apply_removals(rep, stamp, args.known_only)
            if b:
                backups[rep.service] = str(b)

    def listed(r: ServiceReport) -> list[str]:
        return removable(r, args.known_only)

    def interesting(r: ServiceReport) -> bool:
        return bool(r.orphan or listed(r) or (not args.known_only and r.protected) or r.conflicts)

    total_dead = sum(len(listed(r)) for r in reports)
    orphans = [r for r in reports if r.orphan]
    if args.json:
        print(json.dumps({
            "env_root": str(env_root), "mode": "apply" if args.apply else "report",
            "known_only": args.known_only, "dead_total": total_dead,
            "services": [{**r.__dict__, "dead": listed(r)} for r in reports if interesting(r)],
            "backups": backups,
        }, indent=2))
        return 0

    verb = "removed" if args.apply else "dead"
    scope = "KNOWN_DEAD only" if args.known_only else "all"
    print(f"dead env keys ({'APPLY' if args.apply else 'report, read-only'}; {scope}) env-root={env_root}")
    for r in reports:
        if r.orphan or not interesting(r):
            continue
        print(f"\n{r.service}  ({r.env_path})")
        for k in listed(r):
            print(f"  {verb:9} {k}  -- {r.reasons[k]}")
        if not args.known_only:
            for k in r.protected:
                print(f"  protected {k}  -- {r.reasons[k]}")
        for k in r.conflicts:
            print(f"  CONFLICT  {k}  -- {r.reasons[k]}")
        if r.service in backups:
            print(f"  backup    {backups[r.service]}")
    if orphans:
        print("\norphan .env files (service directory not in this code tree; never edited -- remove by hand "
              "if the service is really gone):")
        for r in orphans:
            print(f"  {r.env_path}")
    clean = sum(1 for r in reports if not interesting(r))
    print(f"\n{total_dead} {verb} key(s) across {len(reports)} .env file(s); {clean} file(s) clean; "
          f"{len(orphans)} orphan file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
