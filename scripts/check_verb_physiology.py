#!/usr/bin/env python3
"""Body-economy gate: "every verb declares its physiology" (Phase 2 Gate 1 row, Patch C2).

Every ``orion/cognition/verbs/*.yaml`` (except the ``active.yaml`` activation manifest) must
carry a valid ``physiology`` block: tier 0/1/2, a family slug, lanes that are GPU pool roles or
classes in ``config/gpu_pool.yaml``, and ``deferrable_window_s >= 0``. Tier 0 must use family
``turn`` or ``autonomic`` (it never bids), and those two families are tier 0 only.

Spec: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md ("Execution seam").
Structural only: YAML parse plus validation, no live data. Run from the repo root:

    python3 scripts/check_verb_physiology.py
    python3 scripts/check_verb_physiology.py --summary   # counts by tier and family
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.cognition.verb_physiology import (  # noqa: E402
    VerbPhysiologyV1,
    load_pool_lanes,
    parse_verb_physiology,
)

DEFAULT_VERBS_DIR = REPO_ROOT / "orion" / "cognition" / "verbs"
DEFAULT_POOL = REPO_ROOT / "config" / "gpu_pool.yaml"
#: Not verbs: the activation manifest (same exclusion as orion/cognition/planner/loader.py).
NON_VERB_FILES = frozenset({"active.yaml"})


def check(verbs_dir: Path, pool_path: Path) -> tuple[list[str], dict[str, VerbPhysiologyV1]]:
    """(errors, parsed blocks by verb file stem)."""
    errors: list[str] = []
    parsed: dict[str, VerbPhysiologyV1] = {}
    lanes = load_pool_lanes(pool_path)
    paths = sorted(p for p in verbs_dir.glob("*.yaml") if p.name not in NON_VERB_FILES)
    if not paths:
        return [f"{verbs_dir}: no verb YAML found"], parsed
    for path in paths:
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(f"{rel}: YAML does not parse: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{rel}: not a YAML mapping")
            continue
        try:
            block = parse_verb_physiology(data, known_lanes=lanes)
        except ValueError as exc:
            errors.append(f"{rel}: invalid physiology block: {' '.join(str(exc).split())}")
            continue
        if block is None:
            errors.append(f"{rel}: missing physiology block (tier, family, lanes, deferrable_window_s)")
            continue
        parsed[path.stem] = block
    return errors, parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verbs-dir", type=Path, default=DEFAULT_VERBS_DIR)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--summary", action="store_true", help="print counts by tier and family")
    args = parser.parse_args(argv)

    errors, parsed = check(args.verbs_dir, args.pool)
    if args.summary and parsed:
        tiers = Counter(b.tier for b in parsed.values())
        families = Counter((b.tier, b.family) for b in parsed.values())
        print("tier counts: " + ", ".join(f"tier {t}={tiers[t]}" for t in sorted(tiers)))
        for (tier, family), n in sorted(families.items()):
            print(f"  tier {tier}  {family:<32} {n}")
    if errors:
        print(f"check_verb_physiology: FAIL ({len(errors)} problem(s))", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        print("Fix: add or correct the verb's `physiology:` block. See orion/cognition/verb_physiology.py.",
              file=sys.stderr)
        return 1
    print(f"check_verb_physiology: OK ({len(parsed)} verbs declare a valid physiology block)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
