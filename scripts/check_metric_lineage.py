#!/usr/bin/env python3
"""Metric semantic layer CLI: resolve lineage, discover blast radius.

Phases 1+2 of docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md.
Read-only. Reports; does not enforce (the CI gate is phase 4).

Usage:
    python scripts/check_metric_lineage.py                 # summary
    python scripts/check_metric_lineage.py --json          # full joined graph
    python scripts/check_metric_lineage.py --metric cpu_pressure   # lineage card
    python scripts/check_metric_lineage.py --drift         # declared vs discovered consumers
    python scripts/check_metric_lineage.py --generic-consumers  # whole-vector readers
    python scripts/check_metric_lineage.py --unwritten     # declared but never written
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
from orion.metrics.generic_consumers import (  # noqa: E402
    CONFIRMED,
    LIKELY,
    VECTOR_SURFACES,
    scan_generic_consumers,
)
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

    # Same definition the gate ratchets on, so report and gate cannot disagree:
    # a metric feeds nothing only when it has no discoverable code consumer AND
    # no surviving declared consumer, counted per node on its own surface.
    from orion.metrics.gate import REPO_ROOT as GATE_ROOT, orphan_nodes

    orphans: dict[str, list[str]] = {}
    for node in orphan_nodes(graph, scan, GATE_ROOT):
        orphans.setdefault(node.surface, []).append(node.scan_token)

    total = sum(len(v) for v in orphans.values())
    print(f"\n  registered metrics that feed nothing: {total}")
    print("  (no code consumer AND no surviving declared consumer;")
    print("   registry-of-origin excluded; NOT a liveness verdict)\n")

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
    writers = scan.producers_for(
        token,
        include_tests=False,
        exclude_paths=sources,
        schema_ids={n.schema_id for n in nodes if n.schema_id},
    )
    print(f"  declared in    {', '.join(sorted(sources))} (excluded from blast radius)")
    print(f"  WRITTEN BY (discovered, non-test): {len(writers)}")
    for hit in writers:
        print(f"      {hit.path}:{hit.line}  [{hit.kind}]")
    if not writers:
        print("      none found -- declared but never written, or written via a")
        print("      path this scan cannot see (see --generic-consumers).")
    print(f"  BLAST RADIUS (discovered, non-test, high-confidence): {len(prod)}")
    for hit in prod:
        print(f"      {hit.path}:{hit.line}  [{hit.kind}]")
    print(f"\n  test/eval references: {len(tests)}")

    # A field channel lives inside a node/capability vector, so any site that
    # enumerates a whole vector reads it WITHOUT naming it -- invisible to the
    # string-literal scan above. Printing the blast radius without this line
    # is how `field_coherence_warning` read as "zero consumers" on 2026-08-14
    # while attention's `_current_pressure_proxy()` max()'d over it every tick.
    generic = _generic_consumers_cached()
    # `any(n...)`, NOT `node.surface` -- `node` here would be whatever the
    # per-node print loop above left bound, i.e. the LAST node for this token.
    # Three tokens span surfaces (`confidence`, `memory_pressure`,
    # `repair_pressure` -- named in gate.py's own orphan_nodes docstring), and
    # for all three the last node is organ_signal/inner_state, so the field
    # channel's floor warning was silently suppressed on exactly the tokens
    # most likely to be misread. Caught in review; no test covered cmd_metric.
    if any(n.surface in VECTOR_SURFACES for n in nodes) and generic:
        confirmed = [c for c in generic if c.confidence == CONFIRMED]
        print(
            f"\n  PLUS {len(generic)} generic whole-vector consumer(s) "
            f"({len(confirmed)} confirmed) that read this channel without naming it."
        )
        print("  Run --generic-consumers to list them. Blast radius above is a")
        print("  FLOOR, not a total -- do not retire this channel on it alone.")

    print("\n  Liveness verdict: NOT COMPUTED (phase 5).")
    print("  Do not treat absence of a verdict as 'this metric is fine'.")
    return 0


_GENERIC_CACHE: list | None = None


def _generic_consumers_cached() -> list:
    global _GENERIC_CACHE
    if _GENERIC_CACHE is None:
        _GENERIC_CACHE = scan_generic_consumers(REPO_ROOT)
    return _GENERIC_CACHE


def cmd_generic_consumers() -> int:
    """Sites that consume whole channel vectors without naming any channel."""
    found = _generic_consumers_cached()
    print("Generic whole-vector consumers\n")
    print("  These read every channel in a node/capability vector at once, so")
    print("  no field channel is safely retirable without reading them. The")
    print("  string-literal blast radius cannot see any of this.\n")
    for tier, label in ((CONFIRMED, "confirmed"), (LIKELY, "likely")):
        rows = [c for c in found if c.confidence == tier]
        print(f"  {label} ({len(rows)})")
        for c in rows:
            print(f"      {c.path}:{c.line}  {c.function}")
            print(f"          {c.evidence}")
        print()
    print("  `likely` is a prompt to go read the call site, not proof: the")
    print("  detector keys on a `dict[str, float]` parameter in a module that")
    print("  also touches a vector, which is a shape plus a neighbourhood, not")
    print("  a dataflow proof. An UNANNOTATED vector parameter is missed")
    print("  entirely, so an empty result means 'none found', never 'none exist'.")
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
        counts = orphan_counts(graph, scan, GATE_ROOT)
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


def cmd_unwritten(graph, scan) -> int:
    """Metrics with a declaration and no discovered write site.

    The exact mirror of the orphan list, and NOT derivable from it: a metric
    can have real consumers and no reachable producer, which is what
    `expected_offline_suppression` is. It has a live consumer
    (`suppression.py`), a real writer (`state_deltas.py`), and that writer is
    gated on `expected_online is False` -- true for zero of the five nodes in
    `config/biometrics/node_catalog.yaml`. Zero of 126,983 stored ticks carry
    a nonzero value. Finding that took a manual trace; the writer half of it
    is now mechanical.

    What this CANNOT tell you is whether a writer's guard is ever satisfiable.
    A listed writer means "something can set this", not "something does".
    """
    print("Declared metrics with no discovered write site\n")
    print("  A missing writer means one of: never implemented, retired without")
    print("  removing the declaration, or written through a path this scan")
    print("  cannot see (config-driven, cross-language, or a whole-dict copy).")
    print("  It is evidence to go look, not a verdict.\n")

    by_surface: dict[str, list[str]] = {}
    checked: dict[str, bool] = {}
    for node in graph.nodes.values():
        token = node.scan_token
        if token not in checked:
            checked[token] = bool(
                scan.producers_for(
                    token,
                    exclude_paths=graph.registry_sources_for(token),
                    schema_ids={
                        n.schema_id for n in graph.by_token(token) if n.schema_id
                    },
                )
            )
        if not checked[token]:
            by_surface.setdefault(node.surface, []).append(token)

    assessable = {s: v for s, v in by_surface.items() if s not in _WRITER_BLIND_SURFACES}
    total = sum(len(set(v)) for v in assessable.values())
    # Tokens, not URNs: 595 URNs collapse to 397 scan tokens, and a write is
    # discovered per token. Saying "URNs" here would overstate by the
    # multi-node factor.
    print(f"  {total} unwritten scan tokens on surfaces this detector can assess\n")
    for surface in sorted(assessable):
        toks = sorted(set(assessable[surface]))
        print(f"    {surface:<14} {len(toks):>4}   {_WRITER_RELIABILITY.get(surface, '')}")
        for tok in toks[:6]:
            print(f"        {tok}")
        if len(toks) > 6:
            print(f"        ... and {len(toks) - 6} more")

    # Reported, never silently dropped -- but kept out of the headline, because
    # a surface where the detector finds 260 of 260 is measuring itself, not
    # the repo. Same discipline as _ORPHAN_RELIABILITY's "WEAK" labels.
    for surface in sorted(_WRITER_BLIND_SURFACES):
        n = len(set(by_surface.get(surface, [])))
        if n:
            print(f"\n    {surface:<14} {n:>4}   NOT ASSESSABLE -- {_WRITER_RELIABILITY[surface]}")

    print("\n  Write kinds this detects: `vec[\"metric\"] = ...` (Store context),")
    print("  `F(channel=\"metric\", ...)`, and `Model(metric=0.5)`. A metric")
    print("  written by copying a whole dict, or from config, is invisible.")
    return 0


# Surfaces where "no writer found" says nothing about the repo, because the
# write idiom is one this detector structurally cannot see. bus_channel came
# back 260 of 260 on the first run -- channels are published positionally
# (`publish(name, payload)`) or resolved from config, never as `x["name"] = `.
_WRITER_BLIND_SURFACES = frozenset({"bus_channel"})

_WRITER_RELIABILITY = {
    # Checked rather than assumed: `cpu_pressure` IS caught (a literal
    # `Perturbation(channel="cpu_pressure")` in state_deltas.py), while
    # `cortex_exec_step_load` has no literal write anywhere in Python despite
    # being declared, decayed, merged, and read. So this surface is mixed, not
    # strong -- an entry here means "go find the writer", not "there is none".
    "field_channel": "MIXED -- biometric channels are literal perturbations; "
    "execution/harness channels may be written dynamically",
    "organ_signal": "MIXED -- emitted via adapters, some by dict construction",
    "inner_state": "MIXED -- pydantic kwargs are caught, dict-splat writes are not",
    "bus_channel": "channels are published positionally or from config, "
    "never as a named write; use channels.yaml producer_services instead",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="dump the joined graph as JSON")
    ap.add_argument("--metric", help="print a lineage card for one metric token")
    ap.add_argument("--drift", action="store_true", help="declared vs discovered consumers")
    ap.add_argument(
        "--generic-consumers",
        action="store_true",
        help="sites that read whole channel vectors without naming any channel",
    )
    ap.add_argument(
        "--unwritten",
        action="store_true",
        help="metrics with no discovered write site (mirror of the orphan list)",
    )
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

    if args.generic_consumers:
        return cmd_generic_consumers()

    graph = build_graph()

    if args.json and not args.metric:
        print(json.dumps([to_dict(n) for n in graph.nodes.values()], indent=2))
        return 0

    scan = scan_repo(graph.scan_tokens().keys())

    if args.metric:
        return cmd_metric(graph, scan, args.metric, as_json=args.json)
    if args.drift:
        return cmd_drift(graph, scan)
    if args.unwritten:
        return cmd_unwritten(graph, scan)
    return cmd_summary(graph, scan)


if __name__ == "__main__":
    raise SystemExit(main())
