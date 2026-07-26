#!/usr/bin/env python3
"""Repeatable single-domain audit for `orion/substrate/prediction_error.py`'s Active-Inference
domains (or any other node carrying a `prediction_error` metadata field in
`substrate_field_state`).

Operationalizes CLAUDE.md's metric-quality-gate step 4 ("live-data sanity check") after two real,
distinct failures were found the same day (2026-07-26) by hand, not by process:

1. `bus_synaptic_prediction_error()` had a mathematically permanent ~0.27 floor --
   `mean(|z|)` for a calm, standard-normal-distributed z-score population has expected value
   `sqrt(2/pi)`, not 0, so the domain could never read "genuinely calm" no matter how quiet the
   real mesh was. Caught by recovering the underlying pre-aggregation statistic from the
   already-persisted value and checking its theoretical rest point by hand (PR #1391).
2. `node:substrate.route`'s `prediction_error` was silently decaying to zero via a generic
   field-digester staleness-decay loop that should never have applied to it (`prediction_error`
   is designed elsewhere to be an undecayed raw snapshot). Caught by pulling the raw history and
   noticing an exact geometric ratio (0.92) between every successive value for 48+ hours straight
   (PR #1393).

Both directions of failure -- a metric that can never look calm, and a metric that looks calm for
a fake reason -- were only found by recovering real numbers from what was already persisted, not
by re-deriving from code alone. This script mechanizes the two checks that judgment call relies
on so the next domain doesn't need the same hand-archaeology:

1. **Distributional sanity** -- does it read degenerate (always-0, always-1, zero variance)?
2. **Decay-artifact detection** -- is there a suspiciously exact geometric ratio between
   consecutive non-zero values sustained over many ticks? (The exact signature the route bug had.)

This does not replace judgment or a live theory-anchor check (CLAUDE.md step 3) -- it only
mechanizes the parts that are mechanizable. A clean report here is necessary, not sufficient,
before trusting a domain.

Run:
    python scripts/analysis/audit_prediction_error_domain.py bus_synaptic
    python scripts/analysis/audit_prediction_error_domain.py route --window-hours 48
    python scripts/analysis/audit_prediction_error_domain.py --node-id node:athena --window-hours 6
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.prediction_error_domain_audit")

DEFAULT_WINDOW_HOURS: float = 24.0
MAX_ROWS: int = 200_000
DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# Decay-artifact detection: a run of consecutive ratios this close together (relative
# tolerance) is treated as suspicious. 1e-9 is tight enough that real, independent
# fresh measurements essentially never land this close by chance, but loose enough to
# survive float round-tripping through jsonb.
_RATIO_TOLERANCE = 1e-9
# A run this long crosses from "could be coincidence" to "mechanically certain" -- the
# route bug ran for 48+ hours (~86,000+ ticks at 2s cadence); 20 is a conservative floor
# that still catches a decay bug early, long before it becomes a multi-day incident.
_MIN_SUSPICIOUS_RUN_LENGTH = 20
# Below this many samples, a fraction-based flag (e.g. "100% exactly 0.0") is
# trivially true of almost any tiny window and shouldn't be reported with the
# same confidence as a real finding over a representative sample.
_MIN_SAMPLES_FOR_DISTRIBUTION_FLAGS = 10


@dataclass
class DomainSample:
    observed_at: datetime
    value: float


@dataclass
class DistributionReport:
    n: int
    mean: float
    median: float
    min_value: float
    max_value: float
    stdev: float
    frac_exact_zero: float
    frac_exact_one: float
    distinct_values: int


@dataclass
class DecayArtifactReport:
    longest_suspicious_run: int
    suspicious_ratio: Optional[float]
    run_start: Optional[datetime]
    run_end: Optional[datetime]


# ===========================================================================
# Pure layer -- no I/O. Exercised directly by unit tests with synthetic data.
# ===========================================================================


def compute_distribution(samples: list[DomainSample]) -> Optional[DistributionReport]:
    if not samples:
        return None
    values = [s.value for s in samples]
    return DistributionReport(
        n=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        min_value=min(values),
        max_value=max(values),
        stdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
        frac_exact_zero=sum(1 for v in values if v == 0.0) / len(values),
        frac_exact_one=sum(1 for v in values if v == 1.0) / len(values),
        distinct_values=len(set(values)),
    )


def detect_decay_artifact(samples: list[DomainSample]) -> DecayArtifactReport:
    """Walk consecutive (prev, curr) pairs where both are nonzero and look for a long
    run of near-identical ratios -- the exact signature a per-tick multiplicative decay
    leaves behind (route's real bug: ratio == 0.92 every single tick, unbroken, for
    48+ hours). A handful of matching ratios can happen by coincidence; a long
    unbroken run cannot -- independent real measurements don't repeat a ratio to 9
    decimal places dozens of times in a row.

    A ratio of exactly 1.0 is deliberately excluded and never counted toward a run:
    that means the value is genuinely held flat (unchanged), the CORRECT behavior for
    a hold-until-stale channel post the 2026-07-26 decay.py fix -- not a decay bug.
    Found live via this script's own first real smoke test: a 249-tick run of ratio
    1.000000 on a genuinely calm bus_synaptic stretch was a false positive before this
    exclusion existed.
    """
    best_run = 0
    best_ratio: Optional[float] = None
    best_start: Optional[datetime] = None
    best_end: Optional[datetime] = None

    current_run = 0
    current_ratio: Optional[float] = None
    current_start: Optional[datetime] = None

    for prev, curr in zip(samples, samples[1:]):
        if prev.value == 0.0 or curr.value == 0.0:
            current_run, current_ratio, current_start = 0, None, None
            continue
        ratio = curr.value / prev.value
        # ratio == 1.0 means the value is genuinely held flat (unchanged) --
        # the CORRECT behavior for a hold-until-stale channel, not decay.
        # Skip this transition WITHOUT resetting current_run/current_ratio:
        # review caught that resetting here creates a false-negative -- a
        # single held-flat tick landing in the middle of an otherwise-
        # continuous real decay run would fragment it into two shorter
        # sub-runs, potentially hiding a genuine bug below the detection
        # threshold. A flat tick should be invisible to run continuity, not
        # break it.
        if abs(ratio - 1.0) <= _RATIO_TOLERANCE:
            continue
        if current_ratio is not None and abs(ratio - current_ratio) <= _RATIO_TOLERANCE:
            current_run += 1
        else:
            current_run = 1
            current_ratio = ratio
            current_start = prev.observed_at
        if current_run > best_run:
            best_run = current_run
            best_ratio = current_ratio
            best_start = current_start
            best_end = curr.observed_at

    return DecayArtifactReport(
        longest_suspicious_run=best_run,
        suspicious_ratio=best_ratio,
        run_start=best_start,
        run_end=best_end,
    )


def render_report(domain_label: str, node_id: str, dist: Optional[DistributionReport],
                   decay: DecayArtifactReport, window_hours: float) -> str:
    lines = [
        f"# prediction_error domain audit: {domain_label} ({node_id})",
        f"window: last {window_hours}h",
        "",
    ]
    if dist is None:
        lines.append("NO DATA in this window -- cannot audit. Widen --window-hours or check node_id.")
        return "\n".join(lines)

    lines.append(
        f"n={dist.n} mean={dist.mean:.5f} median={dist.median:.5f} "
        f"min={dist.min_value:.5g} max={dist.max_value:.5g} stdev={dist.stdev:.5f}"
    )
    lines.append(
        f"frac exactly 0.0={dist.frac_exact_zero*100:.1f}%  "
        f"frac exactly 1.0={dist.frac_exact_one*100:.1f}%  "
        f"distinct values={dist.distinct_values}"
    )
    lines.append("")

    if dist.n < _MIN_SAMPLES_FOR_DISTRIBUTION_FLAGS:
        lines.append(
            f"NOTE: only {dist.n} sample(s) in this window (below "
            f"{_MIN_SAMPLES_FOR_DISTRIBUTION_FLAGS}) -- distribution flags below are unreliable "
            f"with this little data (e.g. a single 0.0 tick trivially reads 'ALWAYS ZERO'). "
            f"Widen --window-hours before trusting them."
        )
    degenerate_flags = []
    if dist.stdev == 0.0:
        degenerate_flags.append("ZERO VARIANCE -- every value in this window is identical.")
    if dist.frac_exact_zero > 0.99:
        degenerate_flags.append("ALWAYS ZERO -- >99% of ticks read exactly 0.0.")
    if dist.frac_exact_one > 0.5:
        degenerate_flags.append("MOSTLY SATURATED -- >50% of ticks read exactly 1.0.")
    if degenerate_flags:
        lines.append("DISTRIBUTION FLAGS:")
        lines.extend(f"  - {f}" for f in degenerate_flags)
    else:
        lines.append("Distribution: not obviously degenerate by these checks.")
    lines.append("")

    if decay.longest_suspicious_run >= _MIN_SUSPICIOUS_RUN_LENGTH:
        lines.append(
            f"DECAY ARTIFACT SUSPECTED: {decay.longest_suspicious_run} consecutive ticks held an "
            f"identical ratio of {decay.suspicious_ratio:.6f} between {decay.run_start} and "
            f"{decay.run_end}. This is the exact signature of a mechanical per-tick "
            f"multiplicative decay (route's real 2026-07-26 bug: ratio 0.92, unbroken, for "
            f"48+ hours) -- a value this consistent is not a real, independently-arriving "
            f"measurement. Check field-digester's NODE_DECAY_CHANNELS for this channel before "
            f"trusting it."
        )
    else:
        lines.append(
            f"No decay artifact detected (longest matching-ratio run: "
            f"{decay.longest_suspicious_run}, threshold: {_MIN_SUSPICIOUS_RUN_LENGTH})."
        )

    return "\n".join(lines)


# ===========================================================================
# I/O layer.
# ===========================================================================


def open_readonly_connection(dsn: str):
    try:
        import psycopg2
    except Exception:  # pragma: no cover
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None
    try:
        conn = psycopg2.connect(dsn)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            cur.execute("SHOW default_transaction_read_only;")
            value = cur.fetchone()
        if not value or str(value[0]).lower() != "on":
            logger.error("refusing to run: session is not read-only (got %r)", value)
            conn.close()
            return None
    except Exception:
        logger.error("failed to enforce read-only session", exc_info=True)
        conn.close()
        return None
    return conn


def fetch_samples(conn, node_id: str, since: datetime) -> list[DomainSample]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT generated_at, field_json->'node_vectors'->%s->>'prediction_error'
            FROM substrate_field_state
            WHERE generated_at > %s
            ORDER BY generated_at ASC
            LIMIT %s
            """,
            (node_id, since, MAX_ROWS),
        )
        rows = cur.fetchall()
    samples: list[DomainSample] = []
    for observed_at, raw in rows:
        if raw is None:
            continue
        try:
            samples.append(DomainSample(observed_at=observed_at, value=float(raw)))
        except (TypeError, ValueError):
            continue
    return samples


def resolve_node_id(domain: Optional[str], node_id: Optional[str]) -> tuple[str, str]:
    if node_id:
        return node_id, node_id
    if not domain:
        raise SystemExit("must pass either a domain name or --node-id")
    # scripts/analysis/ has no __init__.py -- load by file path, same pattern
    # scripts/analysis/tests/test_measure_ast_hot_reducer.py already uses,
    # rather than a package-style import that isn't reliably importable here.
    try:
        module_path = Path(__file__).resolve().parent / "measure_ast_hot_reducer.py"
        spec = importlib.util.spec_from_file_location("measure_ast_hot_reducer", module_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules.setdefault("measure_ast_hot_reducer", mod)
        spec.loader.exec_module(mod)
        PREDICTION_ERROR_DOMAIN_NODES = mod.PREDICTION_ERROR_DOMAIN_NODES
    except Exception:
        logger.warning("could not load PREDICTION_ERROR_DOMAIN_NODES", exc_info=True)
        PREDICTION_ERROR_DOMAIN_NODES = {}
    resolved = PREDICTION_ERROR_DOMAIN_NODES.get(domain)
    if resolved is None:
        known = ", ".join(sorted(PREDICTION_ERROR_DOMAIN_NODES)) or "(none importable)"
        raise SystemExit(
            f"unknown domain {domain!r} -- not in PREDICTION_ERROR_DOMAIN_NODES "
            f"(scripts/analysis/measure_ast_hot_reducer.py). Known: {known}. "
            f"Pass --node-id directly for anything else."
        )
    return domain, resolved


def run(domain: Optional[str], node_id_arg: Optional[str], window_hours: float, postgres_uri: str) -> int:
    domain_label, node_id = resolve_node_id(domain, node_id_arg)
    conn = open_readonly_connection(postgres_uri)
    if conn is None:
        return 1
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        samples = fetch_samples(conn, node_id, since)
    finally:
        conn.close()

    dist = compute_distribution(samples)
    decay = detect_decay_artifact(samples)
    print(render_report(domain_label, node_id, dist, decay, window_hours))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeatable live-data sanity + decay-artifact audit for one prediction_error domain."
    )
    parser.add_argument(
        "domain", nargs="?", default=None,
        help="domain name from PREDICTION_ERROR_DOMAIN_NODES (e.g. bus_synaptic, route)",
    )
    parser.add_argument(
        "--node-id", type=str, default=None,
        help="raw node_id to audit directly (e.g. node:athena), bypassing the domain lookup",
    )
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"how far back to look in substrate_field_state (default {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--postgres-uri", type=str, default=DEFAULT_POSTGRES_URI,
        help="read-only Postgres DSN (default: in-cluster orion-athena-sql-db)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.domain, args.node_id, args.window_hours, args.postgres_uri)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
