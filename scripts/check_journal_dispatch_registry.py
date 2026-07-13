#!/usr/bin/env python3
"""Completeness gate for orion/journaler/dispatch_registry.py.

Context (2026-07-13): `orion.journaler.dispatch_registry.JOURNAL_DISPATCH_REGISTRY`
replaced two ad-hoc, dedupe-key-mismatched email codepaths in
services/orion-actions/app/main.py with a single, fail-closed table keyed by
`trigger_kind` (see that module's docstring for the double-email bug this fixed).
`resolve_policy()` already fails closed for any *unknown* trigger_kind at runtime
(returns an all-disabled policy) -- but that only protects behavior, not the registry
itself silently going stale as new trigger kinds are added to
`orion.journaler.worker._TRIGGER_TO_MODE` (the canonical list of trigger kinds the
journaler actually understands) without a matching, deliberate registry row.

This script is the deterministic gate for that gap: it fails if any trigger_kind in
`_TRIGGER_TO_MODE` has no corresponding row in `JOURNAL_DISPATCH_REGISTRY`. A silent
runtime fail-closed default is invisible until someone notices a trigger_kind never
notifies anyone; this gate makes the gap loud and CI-catchable instead.

It also (report-only, not a failure) flags the reverse direction: a
JOURNAL_DISPATCH_REGISTRY row whose trigger_kind is no longer in `_TRIGGER_TO_MODE` at
all -- this can't happen from normal trigger_kind additions, but would indicate the
registry still references a retired/renamed trigger_kind and is worth a human look.

Usage:
    python scripts/check_journal_dispatch_registry.py
    python scripts/check_journal_dispatch_registry.py --json

Exit codes: 0 = every _TRIGGER_TO_MODE trigger_kind has a JOURNAL_DISPATCH_REGISTRY row.
            1 = at least one trigger_kind is unregistered -- FAIL.
            2 = could not run the check (import failure).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_journal_dispatch_registry.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_service_env_compose_parity.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load() -> tuple[dict, dict]:
    from orion.journaler.dispatch_registry import JOURNAL_DISPATCH_REGISTRY
    from orion.journaler.worker import _TRIGGER_TO_MODE

    return dict(_TRIGGER_TO_MODE), dict(JOURNAL_DISPATCH_REGISTRY)


def find_unregistered_trigger_kinds(trigger_to_mode: dict, registry: dict) -> list[str]:
    """trigger_kinds the journaler actually understands but the dispatch registry
    does not -- these fall through to resolve_policy()'s fail-closed default at
    runtime (silently sends nothing) instead of a deliberate, reviewed policy."""
    return sorted(k for k in trigger_to_mode if k not in registry)


def find_orphaned_registry_entries(trigger_to_mode: dict, registry: dict) -> list[str]:
    """Registry rows for trigger_kinds no longer recognized by the journaler at all
    -- report-only, does not fail the gate (could be a deliberately-retained row for
    a recently-retired trigger_kind, e.g. during a migration window)."""
    return sorted(k for k in registry if k not in trigger_to_mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    try:
        trigger_to_mode, registry = _load()
    except Exception as exc:  # pragma: no cover - import failure path
        print(f"check_journal_dispatch_registry: failed to import registry/worker: {exc}", file=sys.stderr)
        return 2

    unregistered = find_unregistered_trigger_kinds(trigger_to_mode, registry)
    orphaned = find_orphaned_registry_entries(trigger_to_mode, registry)

    if args.json:
        print(json.dumps({
            "trigger_kind_count": len(trigger_to_mode),
            "registry_entry_count": len(registry),
            "unregistered_trigger_kinds": unregistered,
            "orphaned_registry_entries": orphaned,
        }))
    else:
        if unregistered:
            print(
                f"check_journal_dispatch_registry: {len(unregistered)} of {len(trigger_to_mode)} "
                f"trigger_kind(s) in orion.journaler.worker._TRIGGER_TO_MODE have no "
                f"JOURNAL_DISPATCH_REGISTRY row:"
            )
            for kind in unregistered:
                print(f"    {kind}")
        else:
            print(
                f"check_journal_dispatch_registry: OK -- all {len(trigger_to_mode)} trigger_kind(s) "
                f"in _TRIGGER_TO_MODE have a JOURNAL_DISPATCH_REGISTRY row."
            )
        if orphaned:
            print(
                f"check_journal_dispatch_registry: NOTE (non-failing) -- "
                f"{len(orphaned)} JOURNAL_DISPATCH_REGISTRY entr{'y' if len(orphaned) == 1 else 'ies'} "
                f"reference trigger_kind(s) no longer in _TRIGGER_TO_MODE: {', '.join(orphaned)}"
            )

    if unregistered:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
