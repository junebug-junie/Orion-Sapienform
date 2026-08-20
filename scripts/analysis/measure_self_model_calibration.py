#!/usr/bin/env python3
"""Read-only L6 item 5 (HOT) measurement: does the self-model's own confidence
in a prediction track its actual reliability?

Item 5 is the L6 design's higher-order piece
(`docs/superpowers/specs/2026-07-23-predicted-shift-reversion-finding.md`'s
"Implication for item 5" section) -- a genuine second-order signal per
Rosenthal-style Higher-Order Theory: a representation of whether a
lower-order representation (`predicted_shift`) is accurate. Deferred until
item 4's own trend formula was validated above chance (shipped, PR #1304,
reversion sign) and TEST-confirmed (shipped, PR #1747, window=2) -- building
a confidence-tracks-reliability check on top of a formula that was itself
worse than a coin flip would have answered a shallow question ("am I usually
wrong"), not this one.

Unlike `measure_ast_hot_reducer.py` (which replays the pure reducer over raw
upstream tables because no persisted self-model history existed at Phase 1),
this script reads the REAL, already-persisted live reducer output directly
from `substrate_attention_self_model` -- both `predicted_shift` (the
lower-order prediction) and `prediction_error_confidence` (the unconditional
Active-Inference confidence, `orion/substrate/attention_self_model.py::
_unconditional_prediction_error_confidence`) are already real production
values on every row, no replay required.

**Method.** For each row with a `predicted_shift` naming domain D and
direction (rising/falling), and where D is present in that row's own
`prediction_error_by_domain`: look `--horizon-ticks` rows ahead (default 2,
matching item 3/4's own already-validated horizon on this exact table --
`docs/superpowers/pr-reports/2026-08-19-l6-item34-hit-it-all-pr.md`) for D's
actual value. `correct` = the actual direction of change matches the
predicted direction. A zero actual delta is honestly excluded (ambiguous,
not fabricated as either outcome), not counted either way.

Then, on a held-out TEST split (chronological 70/30, no shuffle -- same
leakage-avoidance convention as item 4's own TEST validation): bin samples by
`prediction_error_confidence` using bin edges computed from TRAIN only (no
peeking), and check whether high-confidence bins are actually more often
correct than low-confidence bins. Reports both the per-bin table and a
two-proportion z-test comparing the top and bottom confidence bins, plus a
plain Pearson correlation between confidence and correctness across all TEST
samples.

This performs NO writes, emits NO events, flips NO flags, changes NO schema.

Run:
    python scripts/analysis/measure_self_model_calibration.py --window-hours 168
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.self_model_calibration")

DEFAULT_WINDOW_HOURS: float = 168.0  # matches attention_self_model_log_retention_hours default (7d)
MAX_ROWS: int = 200_000
DEFAULT_HORIZON_TICKS: int = 2  # matches item 3/4's own validated horizon on this table
DEFAULT_N_BINS: int = 4
DEFAULT_TRAIN_FRACTION: float = 0.7

OUTPUT_DIR = Path("/tmp/self-model-calibration")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "samples.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

_PREDICTED_SHIFT_RE = re.compile(r"^(\w+) prediction-error (rising|falling)")


# ===========================================================================
# Pure layer -- no I/O. Exercised directly by
# scripts/analysis/tests/test_measure_self_model_calibration.py with
# synthetic fixtures, no DB involved.
# ===========================================================================


@dataclass(frozen=True)
class SelfModelRow:
    generated_at: datetime
    predicted_shift: Optional[str]
    prediction_error_confidence: Optional[float]
    prediction_error_by_domain: dict[str, float]


@dataclass(frozen=True)
class CalibrationSample:
    generated_at: datetime
    domain: str
    predicted_direction: int  # +1 rising, -1 falling
    confidence: float
    correct: bool


def parse_predicted_shift(text: Optional[str]) -> Optional[tuple[str, int]]:
    """`"biometrics prediction-error rising (trend=+0.1234 ...)"` -> `("biometrics", +1)`.
    `None` for anything that doesn't match the real reducer's own format
    string (`reduce_attention_self_model`'s `predicted_shift` assignment) --
    never guesses at a malformed value.
    """
    if not text:
        return None
    m = _PREDICTED_SHIFT_RE.match(text)
    if not m:
        return None
    domain, direction = m.group(1), m.group(2)
    return domain, (1 if direction == "rising" else -1)


def build_calibration_samples(
    rows: list[SelfModelRow], *, horizon_ticks: int
) -> tuple[list[CalibrationSample], int, int, int]:
    """Ordered (oldest-to-newest) `rows` -> calibration samples, plus counts
    of (no_predicted_shift, domain_missing_from_window, ambiguous_zero_delta)
    skips, for honest coverage reporting -- every skip is accounted for, none
    silently dropped.
    """
    samples: list[CalibrationSample] = []
    n_no_shift = 0
    n_domain_missing = 0
    n_ambiguous = 0

    for i, row in enumerate(rows):
        parsed = parse_predicted_shift(row.predicted_shift)
        if parsed is None:
            n_no_shift += 1
            continue
        if row.prediction_error_confidence is None:
            n_no_shift += 1
            continue
        domain, predicted_direction = parsed
        j = i + horizon_ticks
        if j >= len(rows):
            n_domain_missing += 1
            continue
        current = row.prediction_error_by_domain.get(domain)
        future = rows[j].prediction_error_by_domain.get(domain)
        if current is None or future is None:
            n_domain_missing += 1
            continue
        delta = future - current
        if delta == 0.0:
            n_ambiguous += 1
            continue
        actual_direction = 1 if delta > 0 else -1
        samples.append(
            CalibrationSample(
                generated_at=row.generated_at,
                domain=domain,
                predicted_direction=predicted_direction,
                confidence=row.prediction_error_confidence,
                correct=(actual_direction == predicted_direction),
            )
        )
    return samples, n_no_shift, n_domain_missing, n_ambiguous


def chronological_split(
    samples: list[CalibrationSample], *, train_fraction: float
) -> tuple[list[CalibrationSample], list[CalibrationSample]]:
    """No shuffle -- avoids leakage, same convention as item 4's own
    TRAIN/TEST validation (`docs/superpowers/pr-reports/
    2026-08-19-l6-item34-hit-it-all-pr.md`)."""
    cut = int(len(samples) * train_fraction)
    return samples[:cut], samples[cut:]


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def train_bin_edges(train_confidences: list[float], *, n_bins: int) -> list[float]:
    """Bin edges from TRAIN confidence quantiles only -- TEST binning below
    applies these without ever looking at TEST values, so bin boundaries
    can't leak TEST information into the report."""
    values = sorted(train_confidences)
    if not values:
        return []
    return [_quantile(values, i / n_bins) for i in range(1, n_bins)]


def assign_bin(confidence: float, edges: list[float]) -> int:
    b = 0
    for edge in edges:
        if confidence >= edge:
            b += 1
        else:
            break
    return b


@dataclass(frozen=True)
class BinStat:
    bin_index: int
    lo: Optional[float]
    hi: Optional[float]
    n: int
    n_correct: int

    @property
    def accuracy(self) -> Optional[float]:
        return None if self.n == 0 else self.n_correct / self.n


def bin_accuracy(
    test_samples: list[CalibrationSample], *, edges: list[float], n_bins: int
) -> list[BinStat]:
    counts_n = [0] * n_bins
    counts_correct = [0] * n_bins
    for s in test_samples:
        b = assign_bin(s.confidence, edges)
        counts_n[b] += 1
        if s.correct:
            counts_correct[b] += 1
    stats: list[BinStat] = []
    bounds = [None, *edges, None]
    for b in range(n_bins):
        stats.append(
            BinStat(
                bin_index=b,
                lo=bounds[b],
                hi=bounds[b + 1],
                n=counts_n[b],
                n_correct=counts_correct[b],
            )
        )
    return stats


def pearson_correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    """Plain-Python Pearson r -- no numpy dependency, same "keep this
    portable across whatever venv runs it" spirit as the rest of
    scripts/analysis. `None` if undefined (fewer than 2 points, or either
    series has zero variance)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def two_proportion_z(n1: int, x1: int, n2: int, x2: int) -> Optional[float]:
    """z-score for comparing two accuracy proportions -- same reporting
    convention this whole L6 arc has used (`z=+16.6` etc in
    docs/superpowers/pr-reports/2026-08-19-l6-item34-hit-it-all-pr.md).
    `None` if undefined (either group empty, or pooled proportion at 0/1)."""
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    denom = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if denom <= 0.0:
        return None
    return (p1 - p2) / math.sqrt(denom)


# ===========================================================================
# I/O layer -- psycopg2 read-only. Mirrors measure_ast_hot_reducer.py's
# connection contract exactly (refuses a non-read-only session).
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


def fetch_self_model_rows(conn, since: datetime) -> tuple[list[SelfModelRow], bool]:
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT self_model_json, generated_at
                FROM substrate_attention_self_model
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_self_model", exc_info=True)
        return [], False

    out: list[SelfModelRow] = []
    for payload, ts in rows:
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        pe_by_domain = payload.get("prediction_error_by_domain") or {}
        if not isinstance(pe_by_domain, dict):
            pe_by_domain = {}
        out.append(
            SelfModelRow(
                generated_at=ts,
                predicted_shift=payload.get("predicted_shift"),
                prediction_error_confidence=payload.get("prediction_error_confidence"),
                prediction_error_by_domain={
                    k: float(v) for k, v in pe_by_domain.items() if isinstance(v, (int, float))
                },
            )
        )
    return out, len(rows) >= MAX_ROWS


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


def write_samples_csv(path: Path, samples: list[CalibrationSample]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generated_at", "domain", "predicted_direction", "confidence", "correct"])
        for s in samples:
            writer.writerow(
                [s.generated_at.isoformat(), s.domain, s.predicted_direction, f"{s.confidence:.4f}", s.correct]
            )


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    horizon_ticks: int,
    n_bins: int,
    rows_truncated: bool,
    n_rows: int,
    n_no_shift: int,
    n_domain_missing: int,
    n_ambiguous: int,
    train: list[CalibrationSample],
    test: list[CalibrationSample],
    bin_stats: list[BinStat],
    correlation: Optional[float],
    z_top_vs_bottom: Optional[float],
    domain_counts_train: dict[str, int],
    domain_counts_test: dict[str, int],
) -> str:
    n_train, n_test = len(train), len(test)
    train_acc = (sum(1 for s in train if s.correct) / n_train) if n_train else None
    test_acc = (sum(1 for s in test if s.correct) / n_test) if n_test else None

    confidences = [s.confidence for s in train + test]
    conf_min = min(confidences) if confidences else None
    conf_max = max(confidences) if confidences else None
    conf_degenerate = (
        conf_min is not None and conf_max is not None and (conf_max - conf_min) < 1e-6
    )

    lines = [
        "# L6 item 5: self-model confidence-vs-reliability calibration",
        "",
        "Read-only. No writes, no events, no schema/flag changes. Reads the REAL, "
        "already-persisted `substrate_attention_self_model` reducer output directly "
        "(`predicted_shift`, `prediction_error_confidence`, `prediction_error_by_domain`) "
        "-- no replay, this table already holds live production values.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- Rows in window: {n_rows} (truncated at MAX_ROWS={MAX_ROWS}: {rows_truncated})",
        f"- Horizon: {horizon_ticks} row(s) ahead "
        "(matches item 3/4's own validated horizon on this table)",
        f"- Skipped -- no predicted_shift/confidence this row: {n_no_shift}",
        f"- Skipped -- domain data missing at current or +{horizon_ticks} row: {n_domain_missing}",
        f"- Skipped -- ambiguous (actual delta == 0.0, excluded not fabricated): {n_ambiguous}",
        "",
        "## Metric-quality-gate: is `prediction_error_confidence` degenerate?",
        "",
        f"- min={_fmt(conf_min)} max={_fmt(conf_max)}",
        "- **DEGENERATE (flat within 1e-6)**" if conf_degenerate else "- real variance observed, not flat",
        "",
        "## TRAIN / TEST split (chronological 70/30, no shuffle)",
        "",
        f"- TRAIN: n={n_train}, raw reversion accuracy={_fmt(train_acc, 3) if train_acc is not None else 'n/a'} "
        f"(this is item 4's own predicted_shift accuracy re-measured live on real, not synthetic, wins -- "
        "not this script's actual question, reported for context)",
        f"- TEST: n={n_test}, raw reversion accuracy={_fmt(test_acc, 3) if test_acc is not None else 'n/a'}",
        f"- TRAIN predicted_shift domain distribution: "
        + (", ".join(f"{d}={c}" for d, c in sorted(domain_counts_train.items(), key=lambda kv: -kv[1])) or "none"),
        f"- TEST predicted_shift domain distribution: "
        + (", ".join(f"{d}={c}" for d, c in sorted(domain_counts_test.items(), key=lambda kv: -kv[1])) or "none"),
        "",
        "## Does confidence track reliability? (the actual item 5 question)",
        "",
        f"- Pearson r(confidence, correct) on TEST: {_fmt(correlation, 4) if correlation is not None else 'n/a (undefined -- check n/variance above)'}",
        "",
        f"### Per-bin accuracy on TEST (bin edges computed from TRAIN quantiles only -- no leakage, {n_bins} bins)",
        "",
        "| bin | confidence range | n | correct | accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for b in bin_stats:
        lo = "-inf" if b.lo is None else f"{b.lo:.4f}"
        hi = "+inf" if b.hi is None else f"{b.hi:.4f}"
        acc = "n/a" if b.accuracy is None else f"{100 * b.accuracy:.1f}%"
        lines.append(f"| {b.bin_index} | [{lo}, {hi}) | {b.n} | {b.n_correct} | {acc} |")

    lines.append("")
    if bin_stats and bin_stats[0].n > 0 and bin_stats[-1].n > 0:
        lines.append(
            f"- Top bin ({bin_stats[-1].n} samples, {100*bin_stats[-1].accuracy:.1f}% accurate) vs. "
            f"bottom bin ({bin_stats[0].n} samples, {100*bin_stats[0].accuracy:.1f}% accurate): "
            f"z={_fmt(z_top_vs_bottom, 2) if z_top_vs_bottom is not None else 'n/a'}"
        )
        if z_top_vs_bottom is not None:
            if z_top_vs_bottom > 1.96:
                lines.append(
                    "  - **Confidence tracks reliability**: top-confidence predictions are "
                    "significantly more often correct than bottom-confidence ones (p<0.05, "
                    "two-tailed against z=1.96)."
                )
            elif z_top_vs_bottom < -1.96:
                lines.append(
                    "  - **Confidence is INVERTED**: top-confidence predictions are "
                    "significantly LESS often correct than bottom-confidence ones -- do not "
                    "wire this in as-is if this holds up on a re-run."
                )
            else:
                lines.append(
                    "  - **No significant calibration signal** at this sample size -- "
                    "confidence and reliability are not distinguishable from independent here."
                )
    else:
        lines.append("- Top/bottom bin comparison unavailable (one or both bins empty).")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- `prediction_error_confidence` and `predicted_shift` are computed by the SAME "
            "underlying `prediction_error_by_domain` snapshot each tick (both are Active-"
            "Inference-domain aggregates) -- a real correlation here does not prove an "
            "independent second-order sense, only that the two are coherently related. This is "
            "the honest limit of what a same-tick offline replay can show; a genuine causal "
            "separation would need confidence computed from a different information source than "
            "the prediction it's rating.",
            "- The `SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS` default changed 10->2 and "
            "`chat_prediction_error`'s EWMA fix shipped together on 2026-08-19 -- most of this "
            "window's history predates that restart. The confidence-vs-reliability question this "
            "script asks is formula-agnostic (it doesn't care which window size produced a given "
            "prediction, only whether confidence tracked that prediction's own correctness), but a "
            "future re-run restricted to post-restart history only would isolate any window=2-era "
            "shift in this result.",
            "- Domain distribution above is not uniform (execution/biometrics currently dominate "
            "live `predicted_shift` wins) -- this measures aggregate calibration across whichever "
            "domains actually won the argmax during the window, not a per-domain-controlled result.",
            "",
        ]
    )
    return "\n".join(lines)


def run(window: timedelta, window_label: str, *, horizon_ticks: int, n_bins: int, train_fraction: float) -> int:
    now = datetime.now(timezone.utc)
    window_start = now - window
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(PROGRESS_PATH)

    progress.emit("connect", percent=0.0, processed=0, total=0)
    conn = open_readonly_connection(dsn)
    if conn is None:
        progress.close()
        print("ERROR: could not open a read-only postgres connection; see log")
        return 2

    rows, truncated = fetch_self_model_rows(conn, window_start)
    progress.emit("self_model rows loaded", percent=40.0, processed=len(rows), total=len(rows))
    try:
        conn.close()
    except Exception:
        pass

    if not rows:
        progress.close()
        print("ERROR: no substrate_attention_self_model rows in window; nothing measured")
        return 2

    samples, n_no_shift, n_domain_missing, n_ambiguous = build_calibration_samples(
        rows, horizon_ticks=horizon_ticks
    )
    progress.emit("samples built", percent=70.0, processed=len(samples), total=len(rows))

    if len(samples) < 20:
        progress.close()
        print(
            f"ERROR: only {len(samples)} usable calibration samples in window "
            f"(no_shift={n_no_shift}, domain_missing={n_domain_missing}, ambiguous={n_ambiguous}); "
            "too few for a meaningful split"
        )
        return 2

    train, test = chronological_split(samples, train_fraction=train_fraction)
    edges = train_bin_edges([s.confidence for s in train], n_bins=n_bins)
    bin_stats = bin_accuracy(test, edges=edges, n_bins=n_bins)
    correlation = pearson_correlation([s.confidence for s in test], [1.0 if s.correct else 0.0 for s in test])

    z_top_vs_bottom = None
    if bin_stats and bin_stats[0].n > 0 and bin_stats[-1].n > 0:
        # top bin first -> positive z means "top more accurate" directly, no sign flip needed
        # (code review, 2026-08-20: the previous bottom-first-then-negate form was correct but
        # one step more roundabout than calling it with top args first).
        z_top_vs_bottom = two_proportion_z(
            bin_stats[-1].n, bin_stats[-1].n_correct, bin_stats[0].n, bin_stats[0].n_correct
        )

    def domain_counts(samples_: list[CalibrationSample]) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in samples_:
            out[s.domain] = out.get(s.domain, 0) + 1
        return out

    write_samples_csv(CSV_PATH, samples)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        horizon_ticks=horizon_ticks,
        n_bins=n_bins,
        rows_truncated=truncated,
        n_rows=len(rows),
        n_no_shift=n_no_shift,
        n_domain_missing=n_domain_missing,
        n_ambiguous=n_ambiguous,
        train=train,
        test=test,
        bin_stats=bin_stats,
        correlation=correlation,
        z_top_vs_bottom=z_top_vs_bottom,
        domain_counts_train=domain_counts(train),
        domain_counts_test=domain_counts(test),
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(samples), total=len(rows))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only L6 item 5 self-model confidence-vs-reliability calibration measurement."
    )
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"analysis window in hours (default {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--horizon-ticks", type=int, default=DEFAULT_HORIZON_TICKS,
        help=f"rows ahead to check actual direction against (default {DEFAULT_HORIZON_TICKS})",
    )
    parser.add_argument(
        "--n-bins", type=int, default=DEFAULT_N_BINS,
        help=f"number of TRAIN-quantile confidence bins (default {DEFAULT_N_BINS})",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION,
        help=f"chronological TRAIN fraction (default {DEFAULT_TRAIN_FRACTION})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    window = timedelta(hours=args.window_hours)
    window_label = f"{args.window_hours:g}h"
    return run(
        window,
        window_label,
        horizon_ticks=args.horizon_ticks,
        n_bins=args.n_bins,
        train_fraction=args.train_fraction,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
