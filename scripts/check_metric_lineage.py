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

    # Orphans must be judged on real consumers only. Counting every hit made
    # this structurally blind to the two largest surfaces: every field channel
    # appears in the glossary and every bus channel in channels.yaml, both
    # inside SCAN_ROOTS, so 299 of 386 tokens could never be reported orphaned.
    orphans: dict[str, list[str]] = {}
    for tok, nodes in graph.scan_tokens().items():
        if not scan.consumers_for(tok, exclude_paths=graph.registry_sources_for(tok)):
            orphans.setdefault(nodes[0].surface, []).append(tok)

    total = sum(len(v) for v in orphans.values())
    print(f"\n  tokens with no discoverable code consumer: {total}")
    print("  (registry-of-origin excluded; NOT a liveness verdict)\n")

    for surface in sorted(orphans):
        toks = sorted(orphans[surface])
        note = _ORPHAN_RELIABILITY.get(surface, "")
        print(f"    {surface:<14} {len(toks):>4}   {note}")
        for tok in toks[:5]:
            print(f"        {tok}")
        if len(toks) > 5:
            print(f"        ... and {len(toks) - 5} more")
    return 0


# How much a "no code consumer" result actually means, per surface. A string
# -literal scan can only see metrics that are READ as string keys in code.
_ORPHAN_RELIABILITY = {
    "field_channel": "STRONG signal -- these are read as dict keys",
    "organ_signal": "STRONG signal -- signal_kinds are read as dict keys",
    "inner_state": "MIXED -- scalar fields are strong, `*.v1` signal ids are weak",
    "bus_channel": "WEAK -- channels are subscribed via config, not string reads; "
    "trust channels.yaml consumer_services instead",
}


def cmd_metric(graph, scan, token: str, as_json: bool = False) -> int:
    nodes = graph.by_token(token)
    if not nodes:
        if as_json:
            print(json.dumps({"token": token, "registered": False}, indent=2))
        else:
            print(f"UNREGISTERED: {token!r} resolves to no URN in any registry.")
            print("That is the finding -- do not hand-author a URN for it.")
        return 1

    sources = graph.registry_sources_for(token)
    if as_json:
        print(
            json.dumps(
                {
                    "token": token,
                    "registered": True,
                    "nodes": [to_dict(n) for n in nodes],
                    "consumers": [
                        h.to_dict()
                        for h in scan.consumers_for(token, exclude_paths=sources)
                    ],
                    "liveness": None,
                    "liveness_note": "NOT COMPUTED (phase 5)",
                },
                indent=2,
            )
        )
        return 0

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
        if node.all_producers and len(node.all_producers) > 1:
            print(f"  all producers  {', '.join(node.all_producers)}")
        if node.upstream:
            print(f"  upstream       {', '.join(node.upstream)}")
        if node.upstream_organs:
            print(f"  parent organs  {', '.join(node.upstream_organs)}")
        if node.declared_consumers:
            print(f"  declared svcs  {', '.join(node.declared_consumers)}")
        if node.feeds_dimensions:
            print(f"  feeds dims     {', '.join(node.feeds_dimensions)}")
        print()

    prod = scan.consumers_for(
        token, high_confidence_only=True, include_tests=False, exclude_paths=sources
    )
    tests = [
        h
        for h in scan.hits
        if h.token == token
        and h.is_test
        and h.kind in HIGH_CONFIDENCE_KINDS
        and h.path not in sources
    ]
    print(f"  declared in    {', '.join(sorted(sources))} (excluded from blast radius)")
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
        discovered = scan.consumers_for(
            node.name,
            high_confidence_only=True,
            exclude_paths=graph.registry_sources_for(node.name),
        )
        disc_files = {h.path for h in discovered}
        print(f"  {node.urn}")
        print(f"      declared   ({len(node.declared_consumers)}): {list(node.declared_consumers)}")
        print(f"      discovered ({len(disc_files)}): {sorted(disc_files)[:6]}")
        rows += 1
    if not rows:
        print("  (no inner-state entries carry declared consumers)")
    return 0


def cmd_gate(update_baseline: bool = False) -> int:
    from orion.metrics.gate import (
        REPO_ROOT as GATE_ROOT,
        missing_declared_consumers,
        orphan_counts,
        run_gate,
        write_baseline,
    )

    if update_baseline:
        from orion.metrics.consumers import scan_repo

        graph = build_graph()
        scan = scan_repo(graph.scan_tokens().keys())
        counts = orphan_counts(graph, scan)
        missing = missing_declared_consumers(graph, GATE_ROOT)
        path = write_baseline(counts, missing)
        print(f"baseline written: {path}")
        print(f"  orphans by surface: {dict(sorted(counts.items()))}")
        print(f"  known-missing declared consumers: {len(missing)}")
        for consumer, source in sorted(missing.items()):
            print(f"      {consumer}  (declared in {source})")
        return 0

    result = run_gate()
    for note in result.notes:
        print(f"  {note}")
    if result.ok:
        print("\nmetric lineage gate: PASS")
        return 0
    print(f"\nmetric lineage gate: FAIL ({len(result.failures)})\n")
    for failure in result.failures:
        print(f"  - {failure}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="dump the joined graph as JSON")
    ap.add_argument("--metric", help="print a lineage card for one metric token")
    ap.add_argument("--drift", action="store_true", help="declared vs discovered consumers")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="CI gate: registry integrity, declared-consumer existence, orphan ratchet",
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the orphan ratchet baseline (deliberate; run when a decrease is real)",
    )
    args = ap.parse_args()

    if args.gate or args.update_baseline:
        return cmd_gate(update_baseline=args.update_baseline)

    graph = build_graph()

    if args.json and not args.metric:
        print(json.dumps([to_dict(n) for n in graph.nodes.values()], indent=2))
        return 0

    scan = scan_repo(graph.scan_tokens().keys())

    if args.metric:
        return cmd_metric(graph, scan, args.metric, as_json=args.json)
    if args.drift:
        return cmd_drift(graph, scan)
    return cmd_summary(graph, scan)


if __name__ == "__main__":
    raise SystemExit(main())
