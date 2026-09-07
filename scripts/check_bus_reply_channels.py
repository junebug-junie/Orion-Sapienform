#!/usr/bin/env python3
"""Catch the exact bug class that shipped 2026-09-06: a per-correlation RPC
reply channel (`reply_to=`/`reply_channel = f"...:{corr_id}"`) with no
matching wildcard entry in `orion/bus/channels.yaml`. That gap made every
reply from Hub to `orion-durable-runs` raise `ValueError: Channel not found
in catalog` on any service with `ORION_BUS_ENFORCE_CATALOG=true` -- the
request channel worked, the (expensive) work happened, and only the reply
publish failed, silently as far as the caller was concerned (a timeout, not
an error).

Regex-based, not AST -- this is a "smallest useful gate" that catches the
common, dominant shape in this codebase (`f"<prefix>:{corr_id}"`, optionally
with the prefix itself coming from a single `{CONST}` interpolation that
resolves to a literal elsewhere), not a general dynamic-channel prover. A
prefix this script cannot confidently resolve to a literal is skipped, not
flagged -- silence here means "not checked," never "checked and clean." Run
it, don't trust it as exhaustive.

Exit 0 = every reply-channel prefix this script could resolve has a matching
catalog entry (or none were found). Exit 1 = at least one resolved prefix has
no exact or wildcard catalog entry.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "orion" / "bus" / "channels.yaml"

SCAN_DIRS = [ROOT / "orion", ROOT / "services"]
SCAN_GLOBS = ["**/*.py"]
SKIP_PARTS = {"tests", "test", "__pycache__", ".venv", "venv"}

# `reply_channel = f"..."` or `reply_to=f"..."` (inline kwarg), single line.
ASSIGN_RE = re.compile(r'reply_channel\s*=\s*f"([^"]+)"')
KWARG_RE = re.compile(r'reply_to\s*=\s*f"([^"]+)"')
# A bare-identifier interpolation at the START of the f-string: f"{NAME}:...".
LEADING_CONST_RE = re.compile(r'^\{([A-Za-z_][A-Za-z0-9_]*)\}')
CONST_DEF_RE_TMPL = r'^{name}\s*=\s*"([^"]+)"'


def _load_catalog_names() -> list[str]:
    if not CATALOG_PATH.is_file():
        return []
    import yaml

    raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    entries = raw.get("channels", raw) if isinstance(raw, dict) else raw
    names: list[str] = []
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and isinstance(e.get("name"), str):
                names.append(e["name"])
    return names


def _covers(catalog_names: list[str], channel_prefix: str) -> bool:
    """A prefix is covered if the catalog has that exact name, or a
    `<same-or-shorter-prefix>*` wildcard entry whose non-star part is a
    prefix of `channel_prefix + ':'` (so `orion:foo:*` covers the prefix
    `orion:foo:bar`)."""
    probe = channel_prefix + ":"
    for name in catalog_names:
        if name == channel_prefix or name == channel_prefix + ":*":
            return True
        if name.endswith("*") and probe.startswith(name[:-1]):
            return True
    return False


def _resolve_leading_const(name: str, py_files: list[Path]) -> str | None:
    pattern = re.compile(CONST_DEF_RE_TMPL.format(name=re.escape(name)), re.MULTILINE)
    for f in py_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def _candidate_prefix(template: str, py_files: list[Path]) -> str | None:
    """From an f-string template (without the `f"` / `"`), return the literal
    channel prefix it builds, or None if this script cannot confidently
    resolve it (multiple unresolved interpolations, no trailing `{...}`,
    etc.)."""
    if not template.endswith("}") or "{" not in template:
        return None
    # Strip the final `{...}` (the correlation id / uuid piece).
    last_open = template.rfind("{")
    body, tail = template[:last_open], template[last_open:]
    if not tail.endswith("}"):
        return None
    body = body.rstrip(":")
    if "{" not in body:
        return body or None
    m = LEADING_CONST_RE.match(body)
    if not m or "{" in body[m.end():]:
        return None  # more than one unresolved interpolation -- skip, don't guess
    resolved = _resolve_leading_const(m.group(1), py_files)
    if resolved is None:
        return None
    return (resolved + body[m.end():]).rstrip(":")


def main() -> int:
    py_files = [
        p
        for d in SCAN_DIRS
        for pat in SCAN_GLOBS
        for p in d.glob(pat)
        if not (SKIP_PARTS & set(p.parts))
    ]
    catalog_names = _load_catalog_names()
    if not catalog_names:
        print("check_bus_reply_channels: catalog empty or unreadable -- skipping (nothing to check against)")
        return 0

    found: dict[str, str] = {}  # prefix -> one example file:line
    for f in py_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rx in (ASSIGN_RE, KWARG_RE):
                m = rx.search(line)
                if not m:
                    continue
                prefix = _candidate_prefix(m.group(1), py_files)
                if prefix and prefix not in found:
                    found[prefix] = f"{f.relative_to(ROOT)}:{lineno}"

    missing = {p: loc for p, loc in found.items() if not _covers(catalog_names, p)}
    print(f"check_bus_reply_channels: {len(found)} reply-channel prefix(es) resolved, {len(missing)} uncovered")
    if not missing:
        return 0
    for prefix, loc in sorted(missing.items()):
        print(f"  UNCOVERED  {prefix}  (first seen {loc}) -- add \"{prefix}:*\" to orion/bus/channels.yaml")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
