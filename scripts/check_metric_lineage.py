#!/usr/bin/env python3
"""Metric semantic layer CLI: resolve lineage, discover blast radius.

Phases 1+2 of docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md.
Read-only. Reports; does not enforce (the CI gate is phase 4).

Usage:
    python scripts/check_metric_lineage.py                 # summary
    python scripts/check_metric_lineage.py --json          # full joined graph
    python scripts/check_metric_lineage.py --metric cpu_pressure   # lineage card
    python scripts/check_metric_lineage.py --drift         # declared vs discovered consumers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# scripts/ on sys.path[0] shadows stdlib `platform` via scripts/platform/ and
# breaks pydantic -- same fix as check_inner_state_registry.py.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.metrics.consumers import HIGH_CONFIDENCE_KINDS, scan_repo  # noqa: E402
from orion.metrics.lineage import build_graph, to_dict  # noqa: E402


def cmd_summary(graph, scan) -> int:
    counts = graph.counts()
    print("Metric semantic layer -- resolved from existing registries only\n")
    for surface in sorted(counts):
        print(f"  {surface:<16} {counts[surface]:>5}")
    print(f"  {'TOTAL':<16} {len(graph.nodes):>5} URNs")
    print(f"\n  scan tokens      {len(graph.scan_tokens()):>5}")
    print(f"  files scanned    {scan.files_scanned:>5}")
    print(f"  consumer hits    {len(scan.hits):>5}")
    if scan.unparsed:
        print(f"  UNPARSED files   {len(scan.unparsed):>5} (reported, not skipped silently)")
        for rel in scan.unparsed[:10]:
            print(f"      {rel}")

    by_token = scan.by_token()
    orphans = [t for t in graph.scan_tokens() if not by_token.get(t)]
    print(f"\n  tokens with zero consumers anywhere: {len(orphans)}")
    for tok in sorted(orphans)[:15]:
        print(f"      {tok}")
    if len(orphans) > 15:
        print(f"      ... and {len(orphans) - 15} more")
    return 0


def cmd_metric(graph, scan, token: str) -> int:
    nodes = graph.by_token(token)
    if not nodes:
        print(f"UNREGISTERED: {token!r} resolves to no URN in any registry.")
        print("That is the finding -- do not hand-author a URN for it.")
        return 1

    print(f"=== lineage card: {token} ===\n")
    for node in nodes:
        print(f"  URN            {node.urn}")
        print(f"  surface        {node.surface}")
        print(f"  producer       {node.producer_service}")
        print(f"  registry       {node.registry_source}")
        if node.schema_id:
            print(f"  schema         {node.schema_id}")
        if node.meaning:
            print(f"  meaning        {node.meaning}")
        if node.notes:
            print(f"  notes          {node.notes}")
        if node.upstream:
            print(f"  upstream       {', '.join(node.upstream)}")
        if node.declared_consumers:
            print(f"  declared       {', '.join(node.declared_consumers)}")
        print()

    prod = scan.consumers_for(token, high_confidence_only=True, include_tests=False)
    tests = [
        h
        for h in scan.hits
        if h.token == token and h.is_test and h.kind in HIGH_CONFIDENCE_KINDS
    ]
    print(f"  BLAST RADIUS (discovered, non-test, high-confidence): {len(prod)}")
    for hit in prod:
        print(f"      {hit.path}:{hit.line}  [{hit.kind}]")
    print(f"\n  test/eval references: {len(tests)}")
    print("\n  Liveness verdict: NOT COMPUTED (phase 5).")
    print("  Do not treat absence of a verdict as 'this metric is fine'.")
    return 0


def cmd_drift(graph, scan) -> int:
    """Hand-maintained consumer lists vs mechanically discovered ones."""
    print("Declared-vs-discovered consumer drift\n")
    rows = 0
    for node in graph.nodes.values():
        if not node.declared_consumers:
            continue
        if node.surface not in ("inner_state",):
            continue
        discovered = scan.consumers_for(node.name, high_confidence_only=True)
        disc_files = {h.path for h in discovered}
        print(f"  {node.urn}")
        print(f"      declared   ({len(node.declared_consumers)}): {list(node.declared_consumers)}")
        print(f"      discovered ({len(disc_files)}): {sorted(disc_files)[:6]}")
        rows += 1
    if not rows:
        print("  (no inner-state entries carry declared consumers)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="dump the joined graph as JSON")
    ap.add_argument("--metric", help="print a lineage card for one metric token")
    ap.add_argument("--drift", action="store_true", help="declared vs discovered consumers")
    args = ap.parse_args()

    graph = build_graph()

    if args.json and not args.metric:
        print(json.dumps([to_dict(n) for n in graph.nodes.values()], indent=2))
        return 0

    scan = scan_repo(graph.scan_tokens().keys())

    if args.metric:
        return cmd_metric(graph, scan, args.metric)
    if args.drift:
        return cmd_drift(graph, scan)
    return cmd_summary(graph, scan)


if __name__ == "__main__":
    raise SystemExit(main())
