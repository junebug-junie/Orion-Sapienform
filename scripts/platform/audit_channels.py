from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from scripts.platform._common import (
    find_repo_root,
    iter_files,
    load_channels_catalog,
    read_text,
    relpath,
    service_guess_from_path,
    write_json,
    write_text,
)

CHANNEL_KEY_RE = re.compile(
    r"^\s*(channel|request_channel|reply_channel|reply_channel_prefix|[A-Za-z0-9_]*_channel)\s*:\s*['\"]?([^'\"#\s]+)",
    re.IGNORECASE,
)

CONST_RE = re.compile(r"^\s*([A-Z0-9_]*CHANNEL[A-Z0-9_]*)\s*=\s*['\"]([^'\"]+)['\"]")


def _extract_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_from_python(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    inv: Dict[str, Dict[str, Any]] = {}

    def add(channel: str, usage: str, p: Path, lineno: int) -> None:
        channel = channel.strip()
        if not channel:
            return
        entry = inv.setdefault(
            channel,
            {
                "channel": channel,
                "usage": set(),
                "files": set(),
                "service_guess": set(),
            },
        )
        entry["usage"].add(usage)
        entry["files"].add(f"{relpath(repo_root, p)}:{lineno}")
        entry["service_guess"].add(service_guess_from_path(p))

    for p in iter_files(repo_root, (".py",)):
        src = read_text(p)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # publish/subscribe/psubscribe callsites
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in {"publish", "subscribe", "psubscribe", "unsubscribe", "punsubscribe"}:
                    if not node.args:
                        continue
                    ch = _extract_str(node.args[0])
                    if ch is None:
                        continue
                    usage = "publish" if method == "publish" else ("pattern" if method == "psubscribe" else "subscribe")
                    add(ch, usage, p, getattr(node, "lineno", 0))

        # channel constants
        for i, line in enumerate(src.splitlines(), start=1):
            m = CONST_RE.match(line)
            if m:
                add(m.group(2), "const", p, i)

    return inv


def extract_from_yaml(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    inv: Dict[str, Dict[str, Any]] = {}

    def add(channel: str, usage: str, p: Path, lineno: int) -> None:
        channel = channel.strip()
        if not channel:
            return
        entry = inv.setdefault(
            channel,
            {
                "channel": channel,
                "usage": set(),
                "files": set(),
                "service_guess": set(),
            },
        )
        entry["usage"].add(usage)
        entry["files"].add(f"{relpath(repo_root, p)}:{lineno}")
        entry["service_guess"].add(service_guess_from_path(p))

    for p in iter_files(repo_root, (".yml", ".yaml"), include_dirs=("services", "orion", "scripts", "docs")):
        src = read_text(p)
        for i, line in enumerate(src.splitlines(), start=1):
            m = CHANNEL_KEY_RE.match(line)
            if m:
                add(m.group(2), "yaml", p, i)

    return inv


def merge_inv(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = dict(a)
    for ch, entry in b.items():
        tgt = out.setdefault(ch, {"channel": ch, "usage": set(), "files": set(), "service_guess": set()})
        tgt["usage"].update(entry["usage"])
        tgt["files"].update(entry["files"])
        tgt["service_guess"].update(entry["service_guess"])
    return out


def triage_unknown(ch: str, files: List[str]) -> str:
    # Heuristics only; human still decides.
    if any("/tests/" in f or "smoke" in f or "test" in f for f in files):
        return "test_only"
    if "*" in ch or ch.endswith(":"):
        return "pattern"
    if ch.startswith("orion:effect:"):
        return "canonical_candidate"
    if ch in {"orion:verb:request", "orion:verb:result"}:
        return "canonical"
    return "unknown"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/platform/audit_channels.py <RUN_DIR>")
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    repo_root = find_repo_root(run_dir)

    catalog = load_channels_catalog(repo_root)
    inv = merge_inv(extract_from_python(repo_root), extract_from_yaml(repo_root))

    # Normalize for JSON
    inventory = []
    for ch, e in sorted(inv.items()):
        inventory.append(
            {
                "channel": ch,
                "usage": sorted(list(e["usage"])),
                "files": sorted(list(e["files"])),
                "service_guess": sorted(list(e["service_guess"])),
            }
        )

    found = set(inv.keys())
    catalog_set = set(catalog.keys())

    unknown = sorted(list(found - catalog_set))
    unused = sorted(list(catalog_set - found))

    pattern_subs = []
    for ch in sorted(found):
        if "pattern" in inv[ch]["usage"] or "*" in ch or ch.endswith(":"):
            pattern_subs.append({"pattern": ch, "files": sorted(list(inv[ch]["files"])), "risk_note": "pattern subscription or prefix"})

    missing_roles = []
    for ch in sorted(catalog_set & found):
        entry = catalog.get(ch, {})
        missing = []
        for k in ("kind", "schema_id", "producer_services", "consumer_services"):
            if k not in entry or entry.get(k) in (None, "", []):
                missing.append(k)
        if missing:
            missing_roles.append({"channel": ch, "missing": missing})

    drift = {
        "catalog_entries": sorted(list(catalog_set)),
        "found_channels": sorted(list(found)),
        "unknown_channels": unknown,
        "catalog_unused_channels": unused,
        "pattern_subscriptions": pattern_subs,
        "channels_missing_roles": missing_roles,
        "summary": {
            "found_count": len(found),
            "catalog_count": len(catalog_set),
            "unknown_count": len(unknown),
        },
    }

    triage_lines = ["# Channel triage (heuristic)", "", "| channel | classification | evidence |", "|---|---|---|"]
    for ch in unknown:
        files = next((x["files"] for x in inventory if x["channel"] == ch), [])
        triage = triage_unknown(ch, files)
        triage_lines.append(f"| `{ch}` | `{triage}` | {', '.join(files[:3])}{'…' if len(files)>3 else ''} |")

    write_json(run_dir / "reports" / "channel_inventory.json", inventory)
    write_json(run_dir / "reports" / "channel_drift.json", drift)
    write_text(run_dir / "reports" / "channel_triage.md", "\n".join(triage_lines) + "\n")

    print(f"Wrote channel artifacts to {run_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
