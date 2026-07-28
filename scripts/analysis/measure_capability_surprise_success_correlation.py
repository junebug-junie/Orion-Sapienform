#!/usr/bin/env python3
"""Read-only correlation probe: real bus_synaptic surprise vs. real success.

Acceptance check #2 of docs/superpowers/specs/2026-07-24-efe-capability-gate-design.md
(2026-07-28 re-scope). For `web.fetch.readonly`/`recall.query.readonly` -- the two
capabilities whose `action_outcomes` rows now carry a genuine `bus_synaptic_prediction_
error()` value (PR #1403), not the old binary success/fail proxy -- checks whether that
real surprise value at the moment a real capability was exercised correlates with whether
it succeeded. No `substrate_field_state` timestamp-join needed: the real value is already
on the same row as `success`.

This performs NO writes, emits NO events, flips NO flags.

Run:
    python scripts/analysis/measure_capability_surprise_success_correlation.py
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.capability_surprise_success_correlation")

MAX_ROWS: int = 200_000
DEGENERATE_VARIANCE_EPS: float = 1e-12

OUTPUT_DIR = Path("/tmp/capability-surprise-success-correlation")
REPORT_PATH = OUTPUT_DIR / "report.md"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# The two capabilities whose action_outcomes.surprise is confirmed real as of PR #1403 --
# NOT journal.compose.episode, which still has no direct signal (see the design doc's
# 2026-07-28 re-scope).
COVERED_KINDS = ("web.fetch.readonly", "recall.query.readonly")


# ===========================================================================
# Pure layer -- no I/O. Exercised directly by unit tests with synthetic data.
# ===========================================================================


@dataclass
class CorrelationResult:
    kind: str
    n: int
    surprise_variance: float
    success_variance: float
    correlation: Optional[float]
    verdict: str


def pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    """Same shape as measure_transport_biometrics_prediction_error_correlation.py's own
    implementation -- returns None (not 0.0 or NaN) when either series has no real
    variance, so a degenerate input can never masquerade as "zero correlation" (a real,
    meaningful claim) rather than "undefined" (the honest one)."""
    n = len(x)
    if n < 2 or n != len(y):
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x < DEGENERATE_VARIANCE_EPS or var_y < DEGENERATE_VARIANCE_EPS:
        return None
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def compute_correlation(kind: str, surprise: list[float], success: list[float]) -> CorrelationResult:
    n = min(len(surprise), len(success))
    surprise = surprise[:n]
    success = success[:n]
    var_s = _variance(surprise)
    var_o = _variance(success)
    r = pearson_correlation(surprise, success)

    if n < 2:
        verdict = "INSUFFICIENT DATA: fewer than 2 real rows for this kind."
    elif r is None:
        degenerate = []
        if var_s < DEGENERATE_VARIANCE_EPS:
            degenerate.append("surprise")
        if var_o < DEGENERATE_VARIANCE_EPS:
            degenerate.append("success")
        verdict = (
            "NO CORRELATION COMPUTABLE: insufficient real variance in "
            f"{' and '.join(degenerate)}'s series over this window."
        )
    elif abs(r) >= 0.3:
        verdict = f"Real, non-trivial correlation found (|r|={abs(r):.4f} >= 0.3)."
    else:
        verdict = f"No meaningful correlation found (|r|={abs(r):.4f} < 0.3)."

    return CorrelationResult(
        kind=kind, n=n, surprise_variance=var_s, success_variance=var_o, correlation=r, verdict=verdict
    )


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
        try:
            conn.close()
        except Exception:
            pass
        return None
    return conn


def fetch_surprise_success_series(
    conn, kind: str, max_rows: int = MAX_ROWS
) -> tuple[list[float], list[float], bool]:
    """Real per-row `surprise`/`success` values for one capability `kind`, straight off
    `action_outcomes` -- no timestamp join, the real value is already on the same row."""
    if conn is None:
        return [], [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT surprise, success
                FROM action_outcomes
                WHERE kind = %s
                  AND surprise IS NOT NULL
                  AND success IS NOT NULL
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (kind, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch action_outcomes surprise/success series", exc_info=True)
        return [], [], False

    surprise: list[float] = []
    success: list[float] = []
    for s_val, o_val in rows:
        if s_val is None or o_val is None:
            continue
        surprise.append(float(s_val))
        success.append(1.0 if o_val else 0.0)
    return surprise, success, len(rows) >= max_rows


# ===========================================================================
# Report rendering + orchestration.
# ===========================================================================


class ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._start = time.monotonic()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")
        except Exception:
            self._fh = None

    def emit(self, title: str, *, percent: float, processed: int, total: int) -> None:
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate = processed / elapsed
        line = (
            f"{datetime.now(timezone.utc).isoformat()} | {title} | "
            f"{percent:5.1f}% | rows={processed}/{total} | rate={rate:.1f}/s"
        )
        logger.info(line)
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def render_report(*, results: list[CorrelationResult], truncated: dict[str, bool]) -> str:
    lines = [
        "# Capability Surprise <-> Success Correlation Probe",
        "",
        "Read-only. No writes, no config changes. See "
        "`docs/superpowers/specs/2026-07-24-efe-capability-gate-design.md` Acceptance check #2.",
        "",
    ]
    for result in results:
        lines.append(f"## `{result.kind}`")
        lines.append("")
        lines.append(f"- Real rows (surprise+success both present): {result.n}")
        if truncated.get(result.kind):
            lines.append(f"- **TRUNCATED at MAX_ROWS={MAX_ROWS}** -- more real rows exist than were fetched.")
        lines.append(f"- `surprise` variance: {result.surprise_variance:.10f}")
        lines.append(f"- `success` variance: {result.success_variance:.10f}")
        lines.append("")
        lines.append(f"**Verdict**: {result.verdict}")
        if result.correlation is not None:
            lines.append(f"\nPearson r = {result.correlation:.4f}")
        lines.append("")
    return "\n".join(lines)


def run(postgres_uri: str = DEFAULT_POSTGRES_URI) -> int:
    progress = ProgressLog(PROGRESS_PATH)
    progress.emit("correlation probe started", percent=0.0, processed=0, total=0)

    conn = open_readonly_connection(postgres_uri)
    results: list[CorrelationResult] = []
    truncated: dict[str, bool] = {}
    total_rows = 0
    for kind in COVERED_KINDS:
        surprise, success, was_truncated = fetch_surprise_success_series(conn, kind)
        truncated[kind] = was_truncated
        total_rows += len(surprise)
        results.append(compute_correlation(kind, surprise, success))
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

    progress.emit("series fetched", percent=70.0, processed=total_rows, total=total_rows)
    report = render_report(results=results, truncated=truncated)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("report written", percent=100.0, processed=total_rows, total=total_rows)
    progress.close()

    print(report)
    print(f"\nFull report: {REPORT_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only correlation probe between action_outcomes.surprise and .success."
    )
    parser.add_argument(
        "--postgres-uri", type=str, default=DEFAULT_POSTGRES_URI,
        help="read-only Postgres DSN (default: in-cluster orion-athena-sql-db)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.postgres_uri)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
