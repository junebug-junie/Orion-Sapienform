#!/usr/bin/env python3
"""Read-only confidence-recovery baseline over real `substrate_attention_self_model` history.

Missing Question 2 of `docs/superpowers/specs/2026-07-28-collapse-mirror-generative-
triggers-design.md`: what is the real historical shape of confidence recovery /
prediction-error drops? Does a sharp rise in `prediction_error_confidence` (or an
equivalent drop in mean prediction-error) happen as discrete, meaningful events, or is
it smooth/continuous with no natural crossing to trigger a gate on?

That doc's Acceptance Check 1: a measurement script against real historical
`AttentionSelfModelV1` data must show confidence-recovery events are discrete and
non-degenerate (not smooth noise, not a permanent floor/ceiling artifact) *before* any
live "insight" gate ships. This is that script.

Reads `substrate_attention_self_model` -- the durable table
`docs/superpowers/specs/2026-07-29-ast-hot-reducer-live-ticking-design.md`'s live tick
started writing to (PR #1459). That table only accumulates forward from its own deploy
time (2026-07-29) -- it cannot be backfilled, so a run against a short window after
deploy will honestly show "insufficient history," not a false confident verdict. Applies
CLAUDE.md's metric-quality-gate step 4 explicitly: checks for the *opposite* failure mode
from `bus_synaptic`'s calm-floor bug -- a metric that reads "recovered" because it's
permanently stuck near the ceiling, not because anything real resolved.

Run:
    python scripts/analysis/measure_attention_self_model_confidence_baseline.py --since-hours 6
    python scripts/analysis/measure_attention_self_model_confidence_baseline.py --since-hours 6 \
        --low-threshold 0.5 --high-threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.attention_self_model_confidence_baseline")

DEFAULT_SINCE_HOURS: float = 48.0
MAX_ROWS: int = 200_000

# A recovery event only counts once real inter-service history has had a chance to
# accumulate -- this repo's own precedent (bus_synaptic, execution_prediction_error)
# is that a single-digit-minutes-old table produces misleading first-glance numbers.
# Not a hard gate, just what flips the report's own verdict language from a real
# characterization to an explicit "insufficient history" caveat.
MIN_TICKS_FOR_TRUSTED_VERDICT: int = 200
MIN_HOURS_FOR_TRUSTED_VERDICT: float = 2.0

# Not independently calibrated -- starting anchors on the [0,1] prediction_error_confidence
# range, matching this repo's own convention of disclosing an unvalidated first-pass
# constant rather than hiding it (see PREDICTION_ERROR_TREND_WINDOW_TICKS's own docstring
# in orion/substrate/prediction_error_trend.py for the same posture). CLI-overridable.
DEFAULT_LOW_THRESHOLD: float = 0.5
DEFAULT_HIGH_THRESHOLD: float = 0.8

OUTPUT_DIR = Path("/tmp/attention-self-model-confidence-baseline")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O. Exercised directly by unit tests with synthetic rows.
# ===========================================================================


@dataclass
class ConfidenceTick:
    generated_at: datetime
    confidence: Optional[float]
    prediction_error_confidence: Optional[float]
    attention_reason: str
    predicted_shift: Optional[str]


@dataclass
class RecoveryEvent:
    """One low->high crossing of `prediction_error_confidence` (or `confidence`,
    whichever field the caller selected). `ticks_to_cross` is the number of
    consecutive real ticks between first dropping into the low band and first
    reaching the high band -- a large value here (many ticks, i.e. minutes) is
    evidence of a *smooth* transition; a small value (1-2 ticks) is evidence of
    a *discrete* one.
    """

    low_at: datetime
    high_at: datetime
    low_value: float
    high_value: float
    ticks_to_cross: int


def _finite_float_or_none(value: object) -> Optional[float]:
    """`isinstance(value, (int, float))` alone is not enough: `bool` is a
    subclass of `int` in Python (a stray `True`/`False` would silently become
    1.0/0.0), and `json.loads` accepts non-standard `NaN`/`Infinity` tokens by
    default -- a NaN `confidence` would pass the isinstance check and then
    silently corrupt `min()`/`max()`/`statistics.stdev()` downstream without
    raising. Review finding 2026-07-29: this module's own docstring already
    invokes the CLAUDE.md metric-quality-gate precedent this guards against.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def parse_self_model_row(payload: dict, generated_at: datetime) -> Optional[ConfidenceTick]:
    if not isinstance(payload, dict):
        return None
    return ConfidenceTick(
        generated_at=generated_at,
        confidence=_finite_float_or_none(payload.get("confidence")),
        prediction_error_confidence=_finite_float_or_none(payload.get("prediction_error_confidence")),
        attention_reason=str(payload.get("attention_reason") or "no_data"),
        predicted_shift=payload.get("predicted_shift"),
    )


def degenerate_check(values: list[float]) -> dict:
    """CLAUDE.md metric-quality-gate step 4, explicit: confirm the field is not
    flat, always-null, always-saturated -- and specifically check whether it can
    ever read a genuine low value, not just whether it varies at all (a metric
    can show real tick-to-tick variance and still be structurally incapable of
    ever reading "in trouble" -- the exact bus_synaptic calm-floor failure mode,
    checked here for the opposite direction: permanently-recovered-looking).
    """
    if not values:
        return {
            "count": 0, "min": None, "max": None, "mean": None, "stdev": None,
            "frac_near_ceiling": None, "frac_near_floor": None, "verdict": "NO_DATA",
        }
    n = len(values)
    frac_ceiling = sum(1 for v in values if v >= 0.99) / n
    frac_floor = sum(1 for v in values if v <= 0.01) / n
    stdev = statistics.stdev(values) if n >= 2 else 0.0
    verdict = "OK"
    if frac_ceiling >= 0.95:
        verdict = "SUSPICIOUS: >=95% of ticks read >=0.99 -- likely a permanent-ceiling artifact, not genuine sustained confidence"
    elif frac_floor >= 0.95:
        verdict = "SUSPICIOUS: >=95% of ticks read <=0.01 -- likely a permanent-floor artifact, not genuine sustained low confidence"
    elif stdev < 1e-6:
        verdict = "SUSPICIOUS: zero variance -- flat/degenerate signal"
    return {
        "count": n,
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": stdev,
        "frac_near_ceiling": frac_ceiling,
        "frac_near_floor": frac_floor,
        "verdict": verdict,
    }


def delta_stats(ticks: list[ConfidenceTick], *, field: str) -> dict:
    """Tick-to-tick delta distribution for `field`. Large, sparse jumps amid
    mostly-near-zero deltas is the signature of discrete events; a broad,
    continuous spread of deltas with no dominant near-zero mode is the
    signature of smooth/continuous drift with no natural crossing to gate on.
    """
    values = [getattr(t, field) for t in ticks if getattr(t, field) is not None]
    if len(values) < 2:
        return {"count": 0, "mean_abs_delta": None, "max_abs_delta": None, "frac_near_zero": None}
    deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    frac_near_zero = sum(1 for d in deltas if d < 0.02) / len(deltas)
    return {
        "count": len(deltas),
        "mean_abs_delta": statistics.fmean(deltas),
        "max_abs_delta": max(deltas),
        "frac_near_zero": frac_near_zero,
    }


def detect_recovery_events(
    ticks: list[ConfidenceTick],
    *,
    field: str = "prediction_error_confidence",
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> list[RecoveryEvent]:
    """Simple armed/fired state machine over an ordered (oldest-to-newest) tick
    list: "armed" the first time `field` drops to/below `low_threshold`, "fires"
    a RecoveryEvent the next time it rises to/above `high_threshold` while
    armed. Re-arms only after dropping back to/below `low_threshold` again --
    this deliberately does not fire on every small wiggle above the high
    threshold, only on a genuine low->high transition.
    """
    events: list[RecoveryEvent] = []
    armed_at: Optional[tuple[datetime, float]] = None
    armed_tick_idx: Optional[int] = None
    for i, tick in enumerate(ticks):
        value = getattr(tick, field)
        if value is None:
            continue
        if value <= low_threshold:
            if armed_at is None:
                armed_at = (tick.generated_at, value)
                armed_tick_idx = i
        elif value >= high_threshold and armed_at is not None:
            events.append(
                RecoveryEvent(
                    low_at=armed_at[0],
                    high_at=tick.generated_at,
                    low_value=armed_at[1],
                    high_value=value,
                    # `armed_tick_idx or i` would be wrong here -- index 0
                    # (the very first tick) is falsy, so `or` would silently
                    # fall back to `i` and always report ticks_to_cross=0 for
                    # any event that armed on tick 0. Caught by
                    # test_detect_recovery_events_finds_discrete_single_tick_jump.
                    ticks_to_cross=i - (armed_tick_idx if armed_tick_idx is not None else i),
                )
            )
            armed_at = None
            armed_tick_idx = None
    return events


# ===========================================================================
# I/O layer -- psycopg2, read-only. Mirrors measure_ast_hot_reducer.py's
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


def fetch_self_model_rows(conn, since: datetime) -> tuple[list[ConfidenceTick], bool]:
    """Returns `(ticks, truncated)`. `truncated=True` means the window had
    `>= MAX_ROWS` rows and this only returns the oldest `MAX_ROWS` of them
    (the `ORDER BY generated_at ASC LIMIT` above) -- the caller must not
    silently report `span_hours`/`ticks found` as if they covered the full
    requested window in that case. Parity fix, review finding 2026-07-29:
    the sibling `measure_ast_hot_reducer.py` already surfaces this same
    signal (`_rows_to_payload_list`'s `len(rows) >= MAX_ROWS`); this script's
    first draft silently dropped it.
    """
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
    truncated = len(rows) >= MAX_ROWS
    out: list[ConfidenceTick] = []
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
        tick = parse_self_model_row(payload, ts)
        if tick is not None:
            out.append(tick)
    return out, truncated


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


def write_ticks_csv(path: Path, ticks: list[ConfidenceTick]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["generated_at", "attention_reason", "confidence", "prediction_error_confidence", "predicted_shift"]
        )
        for t in ticks:
            writer.writerow(
                [
                    t.generated_at.isoformat(),
                    t.attention_reason,
                    "" if t.confidence is None else f"{t.confidence:.4f}",
                    "" if t.prediction_error_confidence is None else f"{t.prediction_error_confidence:.4f}",
                    t.predicted_shift or "",
                ]
            )


def render_report(
    *,
    since_hours: float,
    ticks: list[ConfidenceTick],
    low_threshold: float,
    high_threshold: float,
    truncated: bool = False,
) -> str:
    lines: list[str] = []
    lines.append("# AST/HOT Self-Model Confidence-Recovery Baseline")
    lines.append("")
    lines.append(
        "Read-only. No writes, no events, no flag/config changes. Measures "
        "`substrate_attention_self_model` (docs/superpowers/specs/2026-07-29-ast-hot-reducer-"
        "live-ticking-design.md's live tick, PR #1459) against CollapseMirror generative-"
        "triggers design's Missing Question 2 and Acceptance Check 1: are confidence-recovery "
        "events discrete and non-degenerate, or smooth noise with no natural crossing?"
    )
    lines.append("")

    span_hours = 0.0
    if len(ticks) >= 2:
        span_hours = (ticks[-1].generated_at - ticks[0].generated_at).total_seconds() / 3600.0

    lines.append(f"- Window requested: last {since_hours:g}h")
    lines.append(f"- Ticks found: {len(ticks)}")
    lines.append(f"- Real span covered by those ticks: {span_hours:.2f}h")
    if truncated:
        lines.append(
            f"- **TRUNCATED**: the window had >= {MAX_ROWS} rows; this run only covers the "
            f"*oldest* {MAX_ROWS} of them (`ORDER BY generated_at ASC LIMIT`). `span_hours` above "
            "understates the real requested window -- re-run with a shorter `--since-hours` to get "
            "full coverage of a sub-window instead."
        )
    lines.append("")

    if not ticks:
        lines.append(
            "**INSUFFICIENT HISTORY.** No rows found in `substrate_attention_self_model` for this "
            "window. This table only accumulates forward from its own deploy time (2026-07-29, PR "
            "#1459) -- it cannot be backfilled. Re-run once the live tick "
            "(`SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED=true`) has been running for a while."
        )
        return "\n".join(lines)

    lines.append("## Live-data sanity check (CLAUDE.md metric-quality-gate step 4)")
    lines.append("")
    for field_name in ("confidence", "prediction_error_confidence"):
        values = [getattr(t, field_name) for t in ticks if getattr(t, field_name) is not None]
        stats = degenerate_check(values)
        coverage_pct = 100.0 * stats["count"] / len(ticks) if ticks else 0.0
        lines.append(f"### `{field_name}`")
        lines.append(f"- Coverage: {stats['count']}/{len(ticks)} ({coverage_pct:.1f}%)")
        if stats["count"]:
            lines.append(
                f"- min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f} "
                f"stdev={stats['stdev']:.4f}"
            )
            lines.append(
                f"- fraction >=0.99: {stats['frac_near_ceiling']:.1%} | "
                f"fraction <=0.01: {stats['frac_near_floor']:.1%}"
            )
        lines.append(f"- **{stats['verdict']}**")
        lines.append("")

    lines.append("## Delta distribution (discrete vs. smooth signature)")
    lines.append("")
    for field_name in ("confidence", "prediction_error_confidence"):
        dstats = delta_stats(ticks, field=field_name)
        lines.append(f"### `{field_name}`")
        if dstats["count"]:
            lines.append(
                f"- mean |delta|={dstats['mean_abs_delta']:.4f}, max |delta|={dstats['max_abs_delta']:.4f}, "
                f"fraction of ticks with |delta| < 0.02: {dstats['frac_near_zero']:.1%}"
            )
            lines.append(
                "  (a high near-zero fraction with occasional large spikes is the signature of "
                "discrete events; a broad, continuous spread is the signature of smooth drift)"
            )
        else:
            lines.append("- insufficient data")
        lines.append("")

    lines.append(
        f"## Recovery events (low<={low_threshold:g} -> high>={high_threshold:g}, "
        f"`prediction_error_confidence`)"
    )
    lines.append("")
    events = detect_recovery_events(
        ticks, field="prediction_error_confidence", low_threshold=low_threshold, high_threshold=high_threshold
    )
    lines.append(f"- Events found: {len(events)}")
    if events:
        crossing_ticks = [e.ticks_to_cross for e in events]
        lines.append(
            f"- ticks-to-cross: min={min(crossing_ticks)}, max={max(crossing_ticks)}, "
            f"mean={statistics.fmean(crossing_ticks):.1f}"
        )
        lines.append("")
        lines.append("| low_at | high_at | low_value | high_value | ticks_to_cross |")
        lines.append("| --- | --- | --- | --- | --- |")
        for e in events[:20]:
            lines.append(
                f"| {e.low_at.isoformat()} | {e.high_at.isoformat()} | {e.low_value:.4f} | "
                f"{e.high_value:.4f} | {e.ticks_to_cross} |"
            )
        if len(events) > 20:
            lines.append(f"| ... {len(events) - 20} more ... |  |  |  |  |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if len(ticks) < MIN_TICKS_FOR_TRUSTED_VERDICT or span_hours < MIN_HOURS_FOR_TRUSTED_VERDICT:
        lines.append(
            f"**NOT YET TRUSTABLE.** Only {len(ticks)} ticks over {span_hours:.2f}h -- below this "
            f"script's own {MIN_TICKS_FOR_TRUSTED_VERDICT}-tick / {MIN_HOURS_FOR_TRUSTED_VERDICT:g}h "
            "floor for treating a characterization as real rather than an early-window artifact. "
            "The numbers above are honestly computed but should not be used to pick a gate "
            "threshold yet -- re-run once more history has accumulated."
        )
    elif not events:
        lines.append(
            "No low->high recovery events observed in this window at the configured thresholds. "
            "Either genuine recoveries are rarer than this window's span, or the thresholds need "
            "adjusting (see --low-threshold/--high-threshold) -- re-run with a longer window or "
            "different thresholds before concluding the signal doesn't produce discrete events at all."
        )
    else:
        crossing_ticks = [e.ticks_to_cross for e in events]
        median_cross = statistics.median(crossing_ticks)
        if median_cross <= 2:
            lines.append(
                f"**Evidence of discrete events**: median crossing took {median_cross:g} tick(s) "
                f"({len(events)} event(s) observed over {span_hours:.2f}h). Consistent with a real, "
                "gateable signal rather than smooth continuous drift."
            )
        else:
            lines.append(
                f"**Evidence leans smooth, not discrete**: median crossing took {median_cross:g} "
                f"ticks ({len(events)} event(s) observed over {span_hours:.2f}h) -- recoveries "
                "unfold gradually rather than as a sharp jump. A gate keyed on this crossing may "
                "need a sustained-duration condition, not a single-tick threshold crossing."
            )

    lines.append("")
    lines.append("## Coverage caveats")
    lines.append("")
    lines.append(
        "- `substrate_attention_self_model` only accumulates forward from its own deploy time "
        "(2026-07-29) -- no backfill possible."
    )
    lines.append(
        "- Recovery-event thresholds (--low-threshold/--high-threshold) are starting anchors, not "
        "independently calibrated -- see this script's own module docstring."
    )

    return "\n".join(lines)


def run(since: datetime, since_hours: float, *, low_threshold: float, high_threshold: float) -> int:
    progress = ProgressLog(PROGRESS_PATH)
    progress.emit("connect", percent=0.0, processed=0, total=0)

    import os

    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)
    conn = open_readonly_connection(dsn)
    if conn is None:
        print("ERROR: could not open a read-only postgres connection; see log")
        progress.close()
        return 2

    ticks, truncated = fetch_self_model_rows(conn, since)
    progress.emit("self_model loaded", percent=60.0, processed=len(ticks), total=len(ticks))

    write_ticks_csv(CSV_PATH, ticks)
    progress.emit("csv written", percent=90.0, processed=len(ticks), total=len(ticks))

    report = render_report(
        since_hours=since_hours,
        ticks=ticks,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        truncated=truncated,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(ticks), total=len(ticks))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only AST/HOT self-model confidence-recovery baseline."
    )
    parser.add_argument(
        "--since-hours", type=float, default=DEFAULT_SINCE_HOURS,
        help=f"analysis window in hours (default {DEFAULT_SINCE_HOURS})",
    )
    parser.add_argument(
        "--low-threshold", type=float, default=DEFAULT_LOW_THRESHOLD,
        help=f"prediction_error_confidence value considered 'low' (default {DEFAULT_LOW_THRESHOLD})",
    )
    parser.add_argument(
        "--high-threshold", type=float, default=DEFAULT_HIGH_THRESHOLD,
        help=f"prediction_error_confidence value considered 'recovered' (default {DEFAULT_HIGH_THRESHOLD})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
    return run(
        since, args.since_hours, low_threshold=args.low_threshold, high_threshold=args.high_threshold
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
