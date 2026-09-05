#!/usr/bin/env python3
"""Refuse to deploy a service whose live .env is behind its .env_example.

**Why this cannot be a CI gate.** `.env` is gitignored and machine-local, so no
GitHub workflow can ever see it. That is precisely why it rots: every merge
updates the committed `.env_example`, nothing updates the live `.env`, and the
deploy reports success either way. AGENTS.md section 17 has named a
`check_env_template_parity.py` for a long time; this is it, wired where it can
actually fire -- at deploy time, in `scripts/safe_docker_build.sh`, the single
chokepoint every bring-up already goes through.

**What it blocks, and what it deliberately does not.** The severity split is
measured, not assumed: run across all 84 services on 2026-09-05, 30 were behind
their template. A gate that blocks a third of all deploys gets bypassed on day
one, and a bypassed gate is worse than no gate -- so the two failure modes are
graded by whether existing tooling can already see them.

1. STRUCTURED DRIFT -> BLOCK. A key present in BOTH files, both values parsing
   as a JSON list or object, live copy MISSING MEMBERS the contract has. This is
   the case that motivated the script and the one every existing tool is blind
   to: `sync_local_env_from_example.py` adds whole missing KEYS and cannot see
   inside a value, so nothing in this repo could detect it. On 2026-09-05
   `orion-sql-writer`'s `.env` had `SQL_WRITER_SUBSCRIBE_CHANNELS` present but
   four channels short; `self_concept_history` and `self_knowledge_items` sat at
   0 rows while two merged PRs reported clean deploys. The key was there. The
   subscription was not. Exactly one service was in this state -- so blocking is
   precise, not noisy.

2. MISSING KEYS -> WARN, do not block. 29 services are missing at least one key,
   mostly infrastructure defaults (`HEARTBEAT_INTERVAL_SEC`, `NODE_NAME`,
   `ORION_BUS_URL`) that have code fallbacks. These are real and worth fixing,
   but `sync_local_env_from_example.py` already adds them automatically, so they
   are a backlog with an existing remedy rather than a reason to stop a deploy.
   Warning keeps them visible without training everyone to reach for the escape
   hatch.

3. A scalar value that simply DIFFERS -> allowed, silently. Local overrides are
   legitimate and routine: secrets, host-specific URLs, a deliberately tuned
   threshold. Blocking on those would make the gate pure noise.

Values are never printed. Structured drift reports only the missing member
names, and a missing key reports only its name -- so this is safe to run in a
log even when a template carries placeholder secrets.

Escape hatch, set consciously per command, never as a habit:

    ORION_ALLOW_ENV_DRIFT=1 scripts/safe_docker_build.sh <service> up -d

Exit 0 = parity holds (or nothing to compare). Exit 1 = drift that would ship a
service configured differently from its own contract.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_local_env_from_example import main_worktree_root, parse_kv  # noqa: E402

MAX_REPORTED = 8


def _structured(raw: str):
    """Parse a value as a JSON list/object, or return None.

    Tolerates the single-quote wrapping the templates document for shell safety.
    """
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    if not s or s[0] not in "[{":
        return None
    try:
        val = json.loads(s)
    except (ValueError, TypeError):
        return None
    return val if isinstance(val, (list, dict)) else None


def _missing_members(example_raw: str, local_raw: str) -> list[str]:
    ex, lo = _structured(example_raw), _structured(local_raw)
    if ex is None or lo is None or type(ex) is not type(lo):
        return []
    if isinstance(ex, list):
        return [str(m) for m in ex if m not in lo]
    return [str(k) for k in ex if k not in lo]


def check_service(service_dir: Path) -> tuple[list[str], list[str]]:
    """Return (blocking, warnings) for one service. Both empty = full parity."""
    example, local = service_dir / ".env_example", service_dir / ".env"
    if not example.is_file() or not local.is_file():
        return [], []
    ex_kv, lo_kv = parse_kv(example), parse_kv(local)

    blocking: list[str] = []
    warnings: list[str] = []

    missing = sorted(k for k in ex_kv if k not in lo_kv)
    if missing:
        shown = ", ".join(missing[:MAX_REPORTED])
        more = f" (+{len(missing) - MAX_REPORTED} more)" if len(missing) > MAX_REPORTED else ""
        warnings.append(f"{len(missing)} key(s) missing from .env: {shown}{more}")

    for key in sorted(set(ex_kv) & set(lo_kv)):
        gone = _missing_members(ex_kv[key], lo_kv[key])
        if gone:
            shown = ", ".join(gone[:MAX_REPORTED])
            more = f" (+{len(gone) - MAX_REPORTED} more)" if len(gone) > MAX_REPORTED else ""
            blocking.append(f"{key} is missing {len(gone)} entr(ies) the contract has: {shown}{more}")
    return blocking, warnings


def main() -> int:
    root = main_worktree_root()
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if argv:
        dirs = [root / "services" / a for a in argv]
        for d in dirs:
            if not d.is_dir():
                print(f"env parity: no such service {d.name}", file=sys.stderr)
                return 1
    else:
        dirs = sorted(p for p in (root / "services").iterdir() if p.is_dir())

    blocked: dict[str, list[str]] = {}
    warned: dict[str, list[str]] = {}
    compared = 0
    for d in dirs:
        if not (d / ".env").is_file() or not (d / ".env_example").is_file():
            continue
        compared += 1
        blocking, warnings = check_service(d)
        if blocking:
            blocked[d.name] = blocking
        if warnings:
            warned[d.name] = warnings

    for name, items in sorted(warned.items()):
        for w in items:
            print(f"env parity WARN  {name}: {w}")
    if warned:
        print(f"env parity: {len(warned)} service(s) missing whole keys -- not blocking; "
              f"`python scripts/sync_local_env_from_example.py` adds these.")

    if not blocked:
        print(f"env template parity: PASS ({compared} service(s) compared)")
        return 0

    allowed = os.environ.get("ORION_ALLOW_ENV_DRIFT") == "1"
    verdict = "ALLOWED by ORION_ALLOW_ENV_DRIFT=1" if allowed else "FAIL"
    print(f"env template parity: {verdict} -- {len(blocked)} service(s) have a "
          f"structured value short of its contract")
    for name, items in sorted(blocked.items()):
        for b in items:
            print(f"  {name}: {b}")
    if allowed:
        return 0
    print()
    print("A key that is PRESENT but short an entry deploys cleanly and then does nothing.")
    print("`sync_local_env_from_example.py` cannot fix this -- it adds whole keys and does")
    print("not look inside a value. Edit the .env by hand, then re-run.")
    print("Deliberate exception: ORION_ALLOW_ENV_DRIFT=1")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
