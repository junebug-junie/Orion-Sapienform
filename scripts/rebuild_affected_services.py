#!/usr/bin/env python3
"""Resolve which docker-compose services need rebuild after a git diff.

Used by scripts/rebuild_services_from_git_diff.sh (post-merge opt-in hook and
manual runs). Mapping is deterministic: path rules, import index, and
scripts/service_rebuild_paths.yaml — no LLM heuristics.

Usage:
    python scripts/rebuild_affected_services.py --base ORIG_HEAD
    python scripts/rebuild_affected_services.py --paths services/orion-hub/app/main.py
    python scripts/rebuild_affected_services.py --list-only --base HEAD~1
"""
from __future__ import annotations

import argparse
import json
import re
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
_EXCLUDE_FILE = _REPO_ROOT / "mesh-utilities" / "common" / "exclude_services.txt"
_SERVICES_DIR = "services"
_ORION_DIR = "orion"
_IMPORT_RE = re.compile(
    r"(?:^\s*(?:from|import)\s+orion\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*))",
    re.MULTILINE,
)
_DOCKERFILE_COPY_ORION_RE = re.compile(r"^\s*COPY\s+orion\b", re.MULTILINE | re.IGNORECASE)


@dataclass
class ResolveResult:
    services: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


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


def changed_files(repo_root: Path, base_ref: str) -> list[str]:
    proc = _run_git(["diff", "--name-only", f"{base_ref}..HEAD"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git diff failed for base {base_ref!r}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


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


@lru_cache(maxsize=1)
def build_import_index(repo_root: str) -> dict[str, set[str]]:
    """Map orion top-level subpackage -> consumers ('service:<name>' or 'orion:<pkg>')."""
    root = Path(repo_root)
    index: dict[str, set[str]] = {}

    def add(imported: str, consumer: str) -> None:
        top = imported.split(".", 1)[0]
        index.setdefault(top, set()).add(consumer)

    for py_file in (root / _SERVICES_DIR).rglob("*.py"):
        parts = py_file.parts
        if _SERVICES_DIR not in parts:
            continue
        svc_idx = parts.index(_SERVICES_DIR) + 1
        if svc_idx >= len(parts):
            continue
        service = parts[svc_idx]
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for match in _IMPORT_RE.finditer(text):
            add(match.group(1), f"service:{service}")

    for py_file in (root / _ORION_DIR).rglob("*.py"):
        parts = py_file.parts
        if _ORION_DIR not in parts:
            continue
        pkg_idx = parts.index(_ORION_DIR) + 1
        if pkg_idx >= len(parts):
            continue
        pkg = parts[pkg_idx]
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for match in _IMPORT_RE.finditer(text):
            add(match.group(1), f"orion:{pkg}")

    return index


def _load_excludes(repo_root: Path) -> set[str]:
    excludes: set[str] = set()
    if not _EXCLUDE_FILE.is_file():
        return excludes
    for line in _EXCLUDE_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            excludes.add(stripped)
    return excludes


def _orion_package_from_path(path: str) -> str | None:
    if not path.startswith(f"{_ORION_DIR}/"):
        return None
    rest = path[len(_ORION_DIR) + 1 :]
    if not rest:
        return None
    return rest.split("/", 1)[0]


def _service_from_path(path: str) -> str | None:
    if not path.startswith(f"{_SERVICES_DIR}/"):
        return None
    rest = path[len(_SERVICES_DIR) + 1 :]
    if not rest:
        return None
    return rest.split("/", 1)[0]


def _direct_and_one_hop_service_consumers(
    pkg: str,
    import_index: dict[str, set[str]],
) -> set[str]:
    """Services importing orion.<pkg>, plus services importing orion modules that import pkg."""
    services: set[str] = set()
    orion_hops: set[str] = set()

    for consumer in import_index.get(pkg, ()):
        if consumer.startswith("service:"):
            services.add(consumer.split(":", 1)[1])
        elif consumer.startswith("orion:"):
            orion_hops.add(consumer.split(":", 1)[1])

    for hop_pkg in orion_hops:
        for consumer in import_index.get(hop_pkg, ()):
            if consumer.startswith("service:"):
                services.add(consumer.split(":", 1)[1])

    return services


def _path_matches_prefix(path: str, prefix: str) -> bool:
    normalized = prefix.rstrip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def resolve_affected_services(
    paths: list[str],
    repo_root: Path | None = None,
) -> ResolveResult:
    root = (repo_root or _REPO_ROOT).resolve()
    mapping = _load_yaml(_MAPPING_PATH)
    known_services = discover_services(root)
    import_index = build_import_index(str(root))
    orion_copy_services = set(services_copying_orion(str(root)))
    contract_prefixes = list(mapping.get("contract_paths") or ["orion/bus/", "orion/schemas/"])
    skip_prefixes = list(mapping.get("skip_prefixes") or [])
    skip_exact = set(mapping.get("skip_exact") or [])
    script_services: dict[str, list[str]] = mapping.get("script_services") or {}
    orion_extra: dict[str, list[str]] = mapping.get("orion_package_extra_services") or {}
    orion_import_packages_raw = mapping.get("orion_import_packages")
    if orion_import_packages_raw is None:
        orion_import_allowlist: set[str] | None = None
    else:
        orion_import_allowlist = set(orion_import_packages_raw)

    affected: set[str] = set()
    skipped: list[str] = []
    reasons: dict[str, str] = {}

    for raw_path in paths:
        path = raw_path.replace("\\", "/").lstrip("./")

        if path in skip_exact:
            skipped.append(path)
            reasons[path] = "repo metadata (no runtime rebuild)"
            continue

        matched_skip = False
        for prefix in skip_prefixes:
            if _path_matches_prefix(path, prefix):
                skipped.append(path)
                reasons[path] = f"under {prefix.rstrip('/')} (no auto rebuild)"
                matched_skip = True
                break
        if not matched_skip and path.startswith(f"{_ORION_DIR}/"):
            parts = path.split("/")
            if len(parts) >= 3 and parts[2] == "tests":
                skipped.append(path)
                reasons[path] = "orion package tests (no auto rebuild)"
                matched_skip = True
        if matched_skip:
            continue

        service = _service_from_path(path)
        if service:
            if service in known_services:
                affected.add(service)
                reasons[path] = f"direct change in services/{service}/"
            else:
                skipped.append(path)
                reasons[path] = f"unknown service directory {service!r}"
            continue

        if contract_prefixes and any(_path_matches_prefix(path, p) for p in contract_prefixes):
            for svc in orion_copy_services:
                affected.add(svc)
            reasons[path] = "contract path → all services with COPY orion"
            continue

        orion_pkg = _orion_package_from_path(path)
        if orion_pkg:
            if orion_import_allowlist is not None and orion_pkg not in orion_import_allowlist:
                skipped.append(path)
                reasons[path] = (
                    f"orion/{orion_pkg}/ not in orion_import_packages (direct services/ changes only)"
                )
                continue
            for svc in _direct_and_one_hop_service_consumers(orion_pkg, import_index):
                affected.add(svc)
            for svc in orion_extra.get(orion_pkg, []):
                affected.add(svc)
            reasons[path] = f"orion/{orion_pkg}/ → direct import + one orion hop"
            continue

        if path.startswith("scripts/"):
            mapped = script_services.get(path)
            if mapped is None:
                skipped.append(path)
                reasons[path] = "scripts/ change with no script_services mapping (rebuild manually if needed)"
            elif not mapped:
                skipped.append(path)
                reasons[path] = "scripts/ meta-tool (explicit empty mapping)"
            else:
                for svc in mapped:
                    if svc in known_services:
                        affected.add(svc)
                reasons[path] = f"script_services mapping → {', '.join(mapped)}"
            continue

        if path == ".env" or path.endswith("/.env_example"):
            skipped.append(path)
            reasons[path] = (
                "env template change — run sync_local_env_from_example.py and restart affected services manually"
            )
            continue

        skipped.append(path)
        reasons[path] = "no mapping rule (rebuild manually if needed)"

    excludes = _load_excludes(root)
    filtered = sorted(svc for svc in affected if svc in known_services and svc not in excludes)
    return ResolveResult(services=filtered, skipped=sorted(set(skipped)), reasons=reasons)


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
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT, help="Repo root (for tests)")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()

    try:
        if args.paths is not None:
            paths = [p.replace("\\", "/") for p in args.paths]
        else:
            base = resolve_base_ref(repo_root, args.base)
            paths = changed_files(repo_root, base)
    except (ValueError, RuntimeError) as exc:
        print(f"rebuild_affected_services: ERROR: {exc}", file=sys.stderr)
        return 2

    result = resolve_affected_services(paths, repo_root)

    if args.json:
        payload = {
            "services": result.services,
            "skipped": result.skipped,
            "reasons": result.reasons,
        }
        print(json.dumps(payload, indent=2))
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
