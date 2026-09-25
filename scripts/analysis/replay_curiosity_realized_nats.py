"""Read-only gate for the curiosity spend log: is realized belief change a
usable measurement, and how many runs does the arm comparison need?

Acceptance check 1 of docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md.
`HUB_CURIOSITY_VALUE_ORDER_ENABLED` stays false until this reports a
non-degenerate metric and a sample size.

Two sources, either or both:

  --graph   Orion's own graph (read-only): every `:PriorRevision` Orion ever
            wrote, and every prior's `times_tested`. Revisions exist only for
            tests that MOVED a confidence, so this reconstructs the moved tests
            exactly and the unmoved ones only as a count
            (sum(times_tested) - revisions). The 2026-09-14 base rate -- 21
            revisions across 94 journaled runs -- predicts most tests move
            nothing.
  --pg      The spend log itself (`curiosity_run_outcomes` joined to
            `curiosity_offer_decisions`) once Hub has written rows: the real
            per-run distribution, and value-vs-uncertainty arm means.

Degenerate means: essentially never non-zero, or a sample size per arm no
realistic number of runs can reach (at 7 runs a day, --max-days, default 60).

    python scripts/analysis/replay_curiosity_realized_nats.py --pg
    python scripts/analysis/replay_curiosity_realized_nats.py --graph --graph-host 127.0.0.1 --graph-port 6380

Writes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pg_readonly import open_readonly_connection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.curiosity.value import (  # noqa: E402
    ARM_UNCERTAINTY_ORDER,
    ARM_VALUE_ORDER,
    kl_nats,
    valid_confidence,
)

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
RUNS_PER_DAY = 7
Z_ALPHA = 1.959964  # two-sided 0.05
Z_POWER = 0.841621  # 80% power


@dataclass
class Distribution:
    n: int = 0
    n_null: int = 0
    n_zero: int = 0
    mean: Optional[float] = None
    sd: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None

    @property
    def fraction_zero(self) -> Optional[float]:
        known = self.n - self.n_null
        return self.n_zero / known if known else None


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, int(math.ceil(q * len(sorted_values))) - 1))
    return sorted_values[idx]


def distribution(values: Iterable[Optional[float]]) -> Distribution:
    values = list(values)
    known = sorted(v for v in values if v is not None)
    d = Distribution(n=len(values), n_null=len(values) - len(known))
    if not known:
        return d
    d.n_zero = sum(1 for v in known if v == 0.0)
    d.mean = mean(known)
    d.sd = pstdev(known)
    d.p50 = _quantile(known, 0.5)
    d.p90 = _quantile(known, 0.9)
    return d


def runs_per_arm_to_detect_doubling(mean_nats: Optional[float], sd_nats: Optional[float]) -> Optional[int]:
    """Two-arm normal approximation: n per arm to detect a mean of 2*mu
    against mu at alpha 0.05, 80% power. Rough for zero-inflated data -- it is
    a planning number, stated as such, not a guarantee."""
    if not mean_nats or mean_nats <= 0.0 or sd_nats is None:
        return None
    return int(math.ceil(2.0 * ((Z_ALPHA + Z_POWER) * sd_nats / mean_nats) ** 2))


@dataclass
class GraphSummary:
    priors: int = 0
    total_tests: int = 0
    revisions: int = 0
    unmoved_tests_estimate: int = 0
    revision_nats: Optional[Distribution] = None
    distinct_confidences: int = 0
    on_005_grid: Optional[float] = None


def summarize_graph(prior_rows: Sequence[dict[str, Any]], revision_rows: Sequence[dict[str, Any]]) -> GraphSummary:
    tests = 0
    confidences: list[float] = []
    for row in prior_rows:
        try:
            tests += max(0, int(row.get("times_tested") or 0))
        except (TypeError, ValueError):
            pass
        c = valid_confidence(row.get("confidence"))
        if c is not None:
            confidences.append(c)
    nats: list[Optional[float]] = []
    for row in revision_rows:
        before = valid_confidence(row.get("from_confidence"))
        after = valid_confidence(row.get("to_confidence"))
        nats.append(kl_nats(after, before) if before is not None and after is not None else None)
        for c in (before, after):
            if c is not None:
                confidences.append(c)
    on_grid = (
        sum(1 for c in confidences if abs(c * 20 - round(c * 20)) < 1e-9) / len(confidences)
        if confidences
        else None
    )
    return GraphSummary(
        priors=len(prior_rows),
        total_tests=tests,
        revisions=len(revision_rows),
        unmoved_tests_estimate=max(0, tests - len(revision_rows)),
        revision_nats=distribution(nats),
        distinct_confidences=len({round(c, 6) for c in confidences}),
        on_005_grid=on_grid,
    )


@dataclass
class ArmComparison:
    value: Distribution
    uncertainty: Distribution
    difference: Optional[float] = None
    ci95: Optional[tuple[float, float]] = None


def compare_arms(rows: Sequence[dict[str, Any]]) -> ArmComparison:
    """Welch difference of mean realized nats, value minus uncertainty."""
    by_arm: dict[str, list[Optional[float]]] = {ARM_VALUE_ORDER: [], ARM_UNCERTAINTY_ORDER: []}
    for row in rows:
        arm = row.get("arm")
        if arm in by_arm:
            v = row.get("realized_nats")
            by_arm[arm].append(None if v is None else float(v))
    value = distribution(by_arm[ARM_VALUE_ORDER])
    uncertainty = distribution(by_arm[ARM_UNCERTAINTY_ORDER])
    out = ArmComparison(value=value, uncertainty=uncertainty)
    nv, nu = value.n - value.n_null, uncertainty.n - uncertainty.n_null
    if nv >= 2 and nu >= 2 and value.mean is not None and uncertainty.mean is not None:
        out.difference = value.mean - uncertainty.mean
        se = math.sqrt((value.sd or 0.0) ** 2 / nv + (uncertainty.sd or 0.0) ** 2 / nu)
        out.ci95 = (out.difference - Z_ALPHA * se, out.difference + Z_ALPHA * se)
    return out


def verdict(dist: Distribution, *, max_days: int) -> tuple[str, Optional[int]]:
    known = dist.n - dist.n_null
    if known == 0:
        return "NO DATA", None
    if dist.n_zero == known:
        return "DEGENERATE: every known run reads exactly 0", None
    need = runs_per_arm_to_detect_doubling(dist.mean, dist.sd)
    budget = RUNS_PER_DAY * max_days // 2
    if need is None:
        return "DEGENERATE: no positive mean", None
    if need > budget:
        return f"DEGENERATE at this cadence: needs {need} runs per arm, {max_days} days gives {budget}", need
    return f"USABLE: ~{need} runs per arm (~{math.ceil(2 * need / RUNS_PER_DAY)} days at {RUNS_PER_DAY}/day)", need


# --- I/O ------------------------------------------------------------------------

PG_SQL = """
SELECT o.realized_nats, d.arm
FROM curiosity_run_outcomes o
LEFT JOIN curiosity_offer_decisions d USING (run_id)
ORDER BY o.completed_at ASC
"""


def read_pg(dsn: str) -> Optional[list[dict[str, Any]]]:
    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(PG_SQL)
            return [{"realized_nats": r[0], "arm": r[1]} for r in cur.fetchall()]
    finally:
        conn.close()


def read_graph(host: str, port: int, graph: str) -> Optional[tuple[list, list]]:
    from orion.curiosity.atlas import ATLAS_PRIORS_CYPHER
    from orion.curiosity.worldview import WorldviewReader, WorldviewUnavailable

    reader = WorldviewReader(host=host, port=port, graph_name=graph)
    revisions_cypher = (
        "MATCH (n:PriorRevision) RETURN n.run_id AS run_id, n.prior_id AS prior_id, "
        "n.from_confidence AS from_confidence, n.to_confidence AS to_confidence LIMIT 20000"
    )
    try:
        return reader.query(ATLAS_PRIORS_CYPHER), reader.query(revisions_cypher)
    except WorldviewUnavailable as exc:
        print(f"UNKNOWN: graph unreadable: {exc}", file=sys.stderr)
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pg", action="store_true", help="read the spend log tables")
    parser.add_argument("--graph", action="store_true", help="read Orion's graph (revisions)")
    parser.add_argument("--graph-host", default=os.environ.get("HUB_CURIOSITY_GRAPH_HOST", "127.0.0.1"))
    parser.add_argument("--graph-port", type=int, default=int(os.environ.get("HUB_CURIOSITY_GRAPH_PORT", "6379")))
    parser.add_argument("--graph-name", default=os.environ.get("HUB_CURIOSITY_GRAPH_OWN", "orion_worldview"))
    parser.add_argument("--max-days", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not (args.pg or args.graph):
        parser.error("choose --pg, --graph, or both")

    report: dict[str, Any] = {}
    status = 0
    if args.graph:
        got = read_graph(args.graph_host, args.graph_port, args.graph_name)
        if got is None:
            status = 2
        else:
            summary = summarize_graph(*got)
            report["graph"] = asdict(summary)
            report["graph"]["verdict"] = (
                "revisions only cover MOVED tests; the unmoved share is "
                f"{summary.unmoved_tests_estimate}/{summary.total_tests} tests"
                if summary.total_tests
                else "NO DATA"
            )
    if args.pg:
        rows = read_pg(os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI))
        if rows is None:
            print("UNKNOWN: could not open a read-only Postgres session", file=sys.stderr)
            status = 2
        else:
            dist = distribution(r["realized_nats"] for r in rows)
            text, need = verdict(dist, max_days=args.max_days)
            arms = compare_arms(rows)
            report["spend_log"] = {
                "runs": asdict(dist),
                "fraction_zero": dist.fraction_zero,
                "runs_per_arm_needed": need,
                "verdict": text,
                "arms": asdict(arms),
            }
    print(json.dumps(report, indent=2, sort_keys=True, default=str) if args.json else _render(report))
    return status


def _render(report: dict[str, Any]) -> str:
    lines = []
    if "graph" in report:
        g = report["graph"]
        lines += [
            f"graph: {g['priors']} priors, {g['total_tests']} tests, {g['revisions']} revisions",
            f"  {g['verdict']}",
            f"  distinct confidences {g['distinct_confidences']}, on a 0.05 grid: {g['on_005_grid']}",
            f"  moved-test nats: {g['revision_nats']}",
        ]
    if "spend_log" in report:
        s = report["spend_log"]
        lines += [
            f"spend log: {s['runs']}",
            f"  fraction exactly 0: {s['fraction_zero']}",
            f"  {s['verdict']}",
            f"  arms: {s['arms']}",
        ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
