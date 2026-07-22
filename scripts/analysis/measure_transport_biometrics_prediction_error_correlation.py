#!/usr/bin/env python3
"""Read-only correlation probe: transport vs. biometrics prediction-error.

Item 4 of docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-
design.md. Before recalibrating or composing anything new for transport's currently-flat
`prediction_error` channel (`node:substrate.transport`, written by
`services/orion-substrate-runtime/app/worker.py::_write_prediction_error_node`), this
checks whether the instrument can carry information at all -- does transport's (tiny, real)
variance ever move together with a domain already confirmed live and non-degenerate
(`node:substrate.biometrics`: 81% of 48h rows nonzero, max 0.627, per the design spec's
own live-data table). Reuses the correlation methodology already established in
`scripts/analysis/measure_emergent_clustering_probe.py` (Pearson over aligned per-tick
series), applied to two `FieldStateV1.node_vectors` channels instead of attention targets.

This performs NO writes, emits NO events, flips NO flags.

Run:
    python scripts/analysis/measure_transport_biometrics_prediction_error_correlation.py --window-hours 48
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.transport_biometrics_prediction_error_correlation")

DEFAULT_WINDOW_HOURS: float = 48.0
MAX_ROWS: int = 200_000
DEGENERATE_VARIANCE_EPS: float = 1e-12

OUTPUT_DIR = Path("/tmp/transport-biometrics-prediction-error-correlation")
REPORT_PATH = OUTPUT_DIR / "report.md"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

TRANSPORT_NODE = "node:substrate.transport"
BIOMETRICS_NODE = "node:substrate.biometrics"


# ===========================================================================
# Pure layer -- no I/O. Exercised directly by unit tests with synthetic data.
# ===========================================================================


@dataclass
class CorrelationResult:
    n: int
    transport_variance: float
    biometrics_variance: float
    correlation: Optional[float]
    verdict: str


def pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    """Same shape as measure_emergent_clustering_probe.py's own implementation --
    returns None (not 0.0 or NaN) when either series has no real variance, so a
    degenerate input can never masquerade as "zero correlation" (a real, meaningful
    claim) rather than "undefined" (the honest one)."""
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


def compute_correlation(transport: list[float], biometrics: list[float]) -> CorrelationResult:
    n = min(len(transport), len(biometrics))
    transport = transport[:n]
    biometrics = biometrics[:n]
    var_t = _variance(transport)
    var_b = _variance(biometrics)
    r = pearson_correlation(transport, biometrics)

    if n < 2:
        verdict = "INSUFFICIENT DATA: fewer than 2 aligned rows."
    elif r is None:
        degenerate = []
        if var_t < DEGENERATE_VARIANCE_EPS:
            degenerate.append("transport")
        if var_b < DEGENERATE_VARIANCE_EPS:
            degenerate.append("biometrics")
        verdict = (
            "NO CORRELATION COMPUTABLE: insufficient real variance in "
            f"{' and '.join(degenerate)}'s series over this window."
        )
    elif abs(r) >= 0.5:
        verdict = f"Real, non-trivial correlation found (|r|={abs(r):.4f} >= 0.5)."
    else:
        verdict = f"No meaningful correlation found (|r|={abs(r):.4f} < 0.5)."

    return CorrelationResult(
        n=n, transport_variance=var_t, biometrics_variance=var_b, correlation=r, verdict=verdict
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


def fetch_aligned_prediction_error_series(
    conn, since: datetime, max_rows: int = MAX_ROWS
) -> tuple[list[float], list[float], bool]:
    """Real per-row `prediction_error` values for both nodes from the same
    `substrate_field_state` rows (so "aligned" means "same tick," not a join on
    timestamp proximity). Returns (transport_series, biometrics_series, truncated)."""
    if conn is None:
        return [], [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (field_json->'node_vectors'->%s->>'prediction_error')::float,
                    (field_json->'node_vectors'->%s->>'prediction_error')::float
                FROM substrate_field_state
                WHERE generated_at >= %s
                  AND field_json->'node_vectors' ? %s
                  AND field_json->'node_vectors' ? %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (TRANSPORT_NODE, BIOMETRICS_NODE, since, TRANSPORT_NODE, BIOMETRICS_NODE, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_field_state prediction_error series", exc_info=True)
        return [], [], False

    transport: list[float] = []
    biometrics: list[float] = []
    for t_val, b_val in rows:
        if t_val is None or b_val is None:
            continue
        transport.append(float(t_val))
        biometrics.append(float(b_val))
    return transport, biometrics, len(rows) >= max_rows


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


def render_report(*, window_hours: float, result: CorrelationResult, truncated: bool) -> str:
    lines = [
        "# Transport <-> Biometrics Prediction-Error Correlation Probe",
        "",
        "Read-only. No writes, no config changes. See "
        "`docs/superpowers/specs/2026-07-22-transport-bus-signal-quality-measurement-design.md` "
        "item 4.",
        "",
        f"- Requested window: {window_hours:g}h",
        f"- Aligned rows (both nodes present, same tick): {result.n}",
    ]
    if truncated:
        lines.append(f"- **TRUNCATED at MAX_ROWS={MAX_ROWS}** -- more real rows exist than were fetched.")
    lines.extend(
        [
            f"- `{TRANSPORT_NODE}` variance this window: {result.transport_variance:.10f}",
            f"- `{BIOMETRICS_NODE}` variance this window: {result.biometrics_variance:.10f}",
            "",
            "## Verdict",
            "",
            result.verdict,
        ]
    )
    if result.correlation is not None:
        lines.append(f"\nPearson r = {result.correlation:.4f}")
    return "\n".join(lines)


def run(window_hours: float, postgres_uri: str = DEFAULT_POSTGRES_URI) -> int:
    progress = ProgressLog(PROGRESS_PATH)
    progress.emit("correlation probe started", percent=0.0, processed=0, total=0)

    conn = open_readonly_connection(postgres_uri)
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    transport, biometrics, truncated = fetch_aligned_prediction_error_series(conn, since)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

    progress.emit("series fetched", percent=70.0, processed=len(transport), total=len(transport))
    result = compute_correlation(transport, biometrics)
    report = render_report(window_hours=window_hours, result=result, truncated=truncated)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("report written", percent=100.0, processed=len(transport), total=len(transport))
    progress.close()

    print(report)
    print(f"\nFull report: {REPORT_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only correlation probe between transport and biometrics prediction-error channels."
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
    return run(args.window_hours, args.postgres_uri)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
