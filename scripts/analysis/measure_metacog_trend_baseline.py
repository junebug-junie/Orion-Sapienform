#!/usr/bin/env python3
"""Read-only measurement: does either candidate series for the metacog trend
reducer (hop 0 of the stream-of-consciousness design) have a genuine rest
state -- not smooth noise, not a permanent floor/ceiling artifact -- before
the reducer ships live?

Session context (2026-07-28/30, Juniper conversation): `docs/superpowers/
specs/2026-07-28-metacog-turn-scoped-trend-reducer-design.md`'s Missing
Questions 1-2 asked whether `turn_effect`/`repair_pressure` has durable
queryable history at all. Resolved live 2026-07-30: `repair_pressure_
appraisal_log` (dedicated Postgres table, `services/orion-sql-writer/app/
models/repair_pressure_appraisal.py`) has real rows; `chat_history_log.
spark_meta->'turn_effect'` has real rows too. That doc's own acceptance
check 1 is the next unmet step: "a measurement script ... shows the chosen
signal has genuine discrete elevated periods and a genuine rest state --
not smooth noise, not a permanent floor/ceiling artifact -- before any
reducer ships live." This script is that instrument, for both candidate
series.

## Disclosed scoping decisions

1. **Two candidate series, not one.** `repair_pressure_appraisal_log.level`
   (dedicated table, gated-and-ungated appraisals both persisted) and
   `chat_history_log.spark_meta->'turn_effect'->'turn'->'novelty'` (the one
   scalar consistently present across sampled real rows; JSONB path, not a
   dedicated column). Measuring both because the metacog doc's Missing
   Question 1 left "which series" open, and this run's job is to inform
   that choice with real numbers, not assume one.

2. **Small-N is disclosed, not hidden.** Real row counts here (tens, not
   hundreds of thousands like the Layer-5 attention probes) are genuinely
   small. `MIN_TOTAL_ROWS = 20` is set low enough to run at all on real
   current data, and every report explicitly flags when a verdict rests on
   a small sample rather than implying Layer-5-probe-grade statistical
   power.

3. **Confidence-gating is measured as its own first-class question for
   `repair_pressure`, not folded silently into the level series.** Live
   data found (this run, re-verify rather than trust this docstring)
   `repair_pressure_appraisal_log` rows cluster heavily at one repeated
   exact `level` value with `confidence=0.0` -- but neither live appraiser
   (`orion/substrate/appraisal/repair_pressure.py::appraise_repair_pressure`,
   `orion/substrate/appraisal/paradigms/repair_pressure_v2.py::run`) emits
   that value for its own documented "no evidence" path (both return
   `level=0.0` exactly, not the observed nonzero constant). This script
   does NOT root-cause where that nonzero floor comes from (a smoothing or
   gating step between appraisal and publish that a short search did not
   locate) -- that is a separate, disclosed open question, named as its
   own finding below rather than silently absorbed into "genuine rest
   state: yes." What this script DOES measure: whether treating `level`
   naively (ungated) vs. gated on `confidence > 0` changes the answer to
   "is there a real rest state," since `confidence=0.0` is the appraiser's
   own explicit "don't trust this reading" signal.

4. **Rest-state classification is a floor-clustering check, not a
   z-score/monoculture check.** Layer 5's sibling scripts (`measure_
   attention_salience_normalization.py`, `measure_emergent_clustering_
   probe.py`) ask "does one target always win a competition." This is a
   different question -- "does this one series, read alone, look like real
   discrete elevated periods against a real calm baseline, or does it look
   like a constant/near-constant floor with occasional noise." Modal-value
   share (the most common single exact value's fraction of all rows) is
   the primary statistic, since CLAUDE.md's own worked example (bus_
   synaptic_prediction_error's sqrt(2/pi) floor, node:substrate.route's
   decay-to-zero artifact) shows that raw variance/stddev alone can miss
   both a permanently-elevated-above-true-zero floor and a
   decaying-to-exactly-zero artifact -- checking what fraction of mass
   sits at one exact recurring value catches both shapes that a bare
   stddev would not flag as obviously as it should.

5. **This script is strictly read-only.** Read-only Postgres session (same
   refuse-if-not-read-only contract as every sibling `measure_*` script).
   Writes only to `/tmp/measure-metacog-trend-baseline/`. Does not modify
   `orion/substrate/appraisal/*`, `services/orion-sql-writer/*`, any other
   live code path, or any database table. Does not build the trend reducer
   itself -- that is `orion/metacog/trend_reducer.py`, a separate, larger
   patch.

Run:
    python scripts/analysis/measure_metacog_trend_baseline.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.metacog_trend_baseline")

DEGENERATE_VARIANCE_EPS: float = 1e-12
MIN_TOTAL_ROWS: int = 20
FLOOR_SHARE_THRESHOLD: float = 0.5
MEANINGFUL_ELEVATED_SHARE: float = 0.10
MULTI_FLOOR_MAX_DISTINCT: int = 3
MULTI_FLOOR_TOP3_SHARE_THRESHOLD: float = 0.85

OUTPUT_DIR = Path("/tmp/measure-metacog-trend-baseline")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "rows.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O.
# ===========================================================================


def channel_mean_std(values: list[float]) -> tuple[float, float, bool]:
    """(mean, stddev, is_degenerate). Population stddev, same convention as
    measure_attention_salience_normalization.py::channel_mean_std."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, True
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance <= DEGENERATE_VARIANCE_EPS:
        return mean, 0.0, True
    return mean, math.sqrt(variance), False


def modal_value(values: list[float], *, round_to: int = 9) -> tuple[Optional[float], int, int]:
    """(modal_value, count, total). Rounds to `round_to` decimals before
    counting so float-repr noise (e.g. 0.08706577244027125 vs an
    almost-identical neighbor from a different code path) doesn't silently
    split what is really one recurring value into two buckets -- while
    still being tight enough not to merge genuinely distinct readings."""
    if not values:
        return None, 0, 0
    counter = Counter(round(v, round_to) for v in values)
    value, count = counter.most_common(1)[0]
    return value, count, len(values)


def distinct_value_stats(values: list[float], *, round_to: int = 9) -> tuple[int, float]:
    """(n_distinct, top3_share). `n_distinct` is the count of distinct
    rounded values; `top3_share` is the combined share of rows held by the
    (up to) 3 most common distinct values. Real continuous data (independent
    float readings) should have n_distinct close to n and a low top3_share;
    a series built from a handful of repeated discrete constants -- even if
    no single one crosses FLOOR_SHARE_THRESHOLD alone -- will have a small
    n_distinct and/or a high top3_share. Added after code review caught that
    a series of three exact repeated floors (e.g. [0.0]*34 + [0.5]*33 +
    [1.0]*33, modal_share=0.34) previously passed as GENUINE_VARIATION
    purely because no single value cleared 50% -- multi-floor clustering is
    exactly the artifact class this script exists to catch, not a case to
    silently wave through."""
    if not values:
        return 0, 0.0
    counter = Counter(round(v, round_to) for v in values)
    n_distinct = len(counter)
    top3_count = sum(c for _, c in counter.most_common(3))
    return n_distinct, top3_count / len(values)


def classify_rest_state(
    n: int,
    mean: float,
    std: float,
    degenerate: bool,
    modal_share: float,
    n_distinct: int,
    top3_share: float,
) -> str:
    """Explicit classification bands for "does this series have a genuine
    rest state, not smooth noise, not a floor/ceiling artifact."

    - `INSUFFICIENT_DATA`: fewer than MIN_TOTAL_ROWS real rows.
    - `DEGENERATE`: near-zero full-history variance. Not merely
      concentrated -- mathematically flat. No signal at all.
    - `FLOOR_DOMINATED`: one exact value (rounded) accounts for >=
      FLOOR_SHARE_THRESHOLD (0.5) of all rows, OR (multi-floor variant) the
      series is dominated by a small handful of discrete repeated values
      even though no single one crosses that threshold alone --
      `n_distinct <= MULTI_FLOOR_MAX_DISTINCT` or the top 3 distinct values
      together hold >= `MULTI_FLOOR_TOP3_SHARE_THRESHOLD` of all rows. Real
      variance may exist in the remainder, but the series is not showing
      genuine continuous spread -- worth checking (see the confidence-gated
      table) whether the dominant point(s) are genuine "checked and found
      calm" readings or a default/fallback value, before trusting this as
      organic rest state.
    - `GENUINE_VARIATION`: no single value dominates, the series is not
      multi-floor-dominated either, AND at least `MEANINGFUL_ELEVATED_SHARE`
      (0.10) of rows fall outside the modal cluster -- i.e. real, non-floor,
      non-discrete-cluster spread.
    - `AMBIGUOUS`: doesn't cleanly fall in the above bands.
    """
    if n < MIN_TOTAL_ROWS:
        return "INSUFFICIENT_DATA"
    if degenerate:
        return "DEGENERATE"
    if modal_share >= FLOOR_SHARE_THRESHOLD:
        return "FLOOR_DOMINATED"
    if n_distinct <= MULTI_FLOOR_MAX_DISTINCT or top3_share >= MULTI_FLOOR_TOP3_SHARE_THRESHOLD:
        return "FLOOR_DOMINATED"
    if (1.0 - modal_share) >= MEANINGFUL_ELEVATED_SHARE:
        return "GENUINE_VARIATION"
    return "AMBIGUOUS"


# ===========================================================================
# I/O layer -- psycopg2 read-only. Same refuse-if-not-read-only contract as
# measure_attention_salience_normalization.py::open_readonly_connection.
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


def fetch_repair_pressure_rows(conn) -> list[tuple[str, float, str, float]]:
    """Real (created_at, level, level_label, confidence) rows, ASC by
    created_at. `created_at` is stored as text (application-generated ISO
    string, not a Postgres timestamp column) -- read as-is, not cast."""
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, level, level_label, confidence
                FROM repair_pressure_appraisal_log
                WHERE level IS NOT NULL
                ORDER BY created_at ASC
                """
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch repair_pressure_appraisal_log rows", exc_info=True)
        return []
    return [(str(ts), float(level), str(label), float(conf)) for ts, level, label, conf in rows]


def fetch_turn_effect_novelty_rows(conn, days: int = 30) -> list[tuple[datetime, float]]:
    """Real (created_at, novelty) rows from chat_history_log.spark_meta ->
    'turn_effect' -> 'turn' -> 'novelty', ASC by created_at, over the last
    `days` days (this table's retention/volume makes a bounded window the
    honest choice -- unbounded would risk MAX_ROWS-style silent truncation
    at the oldest end for a fast-growing chat table)."""
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, spark_meta -> 'turn_effect' -> 'turn' ->> 'novelty'
                FROM chat_history_log
                WHERE spark_meta -> 'turn_effect' -> 'turn' ->> 'novelty' IS NOT NULL
                  AND created_at > now() - (%s || ' days')::interval
                ORDER BY created_at ASC
                """,
                (days,),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch chat_history_log turn_effect novelty rows", exc_info=True)
        return []
    out: list[tuple[datetime, float]] = []
    for ts, novelty_raw in rows:
        if ts is None or novelty_raw is None:
            continue
        try:
            novelty = float(novelty_raw)
        except (TypeError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append((ts, novelty))
    return out


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


def render_series_section(
    title: str,
    *,
    n: int,
    mean: float,
    std: float,
    degenerate: bool,
    modal_val: Optional[float],
    modal_count: int,
    modal_share: float,
    classification: str,
    extra_lines: list[str],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- Real rows: {n}",
        f"- Mean: {mean:.6f}  |  Stddev: {std:.6f}  |  Degenerate (near-zero variance): {degenerate}",
        f"- Modal value (rounded to 9dp): {modal_val if modal_val is not None else 'n/a'} "
        f"({modal_count}/{n} rows = {modal_share*100:.1f}%)",
        f"- Classification: **{classification}**",
    ]
    lines.extend(extra_lines)
    lines.append("")
    return lines


def run() -> int:
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(PROGRESS_PATH)
    caveats: list[str] = []

    progress.emit("connect", percent=0.0, processed=0, total=0)
    conn = open_readonly_connection(dsn)
    if conn is None:
        progress.close()
        print("ERROR: could not open a read-only postgres connection; see log")
        return 2

    rp_rows = fetch_repair_pressure_rows(conn)
    progress.emit("repair_pressure rows fetched", percent=30.0, processed=len(rp_rows), total=len(rp_rows))

    te_rows = fetch_turn_effect_novelty_rows(conn, days=30)
    progress.emit("turn_effect rows fetched", percent=60.0, processed=len(te_rows), total=len(te_rows))

    try:
        conn.close()
    except Exception:
        pass

    lines = [
        "# Metacog Trend Baseline Probe -- Does Either Candidate Series Have a Genuine Rest State?",
        "",
        "Read-only. No writes, no events, no flag/config changes, no edits to any appraisal or "
        "sql-writer code path. Answers the metacog trend reducer design doc's acceptance check 1: "
        "does the chosen signal show genuine discrete elevated periods and a genuine rest state -- "
        "not smooth noise, not a permanent floor/ceiling artifact -- before the reducer ships live.",
        "",
    ]

    # --- repair_pressure_appraisal_log.level, ungated ---
    rp_levels_all = [r[1] for r in rp_rows]
    rp_n = len(rp_levels_all)
    rp_mean, rp_std, rp_degenerate = channel_mean_std(rp_levels_all)
    rp_modal_val, rp_modal_count, _ = modal_value(rp_levels_all)
    rp_modal_share = (rp_modal_count / rp_n) if rp_n else 0.0
    rp_n_distinct, rp_top3_share = distinct_value_stats(rp_levels_all)
    rp_classification = classify_rest_state(
        rp_n, rp_mean, rp_std, rp_degenerate, rp_modal_share, rp_n_distinct, rp_top3_share
    )

    # confidence split: how much of the series is confidence==0.0 (the
    # appraiser's own "no evidence, don't trust this" signal)?
    zero_conf_rows = [r for r in rp_rows if r[3] == 0.0]
    n_zero_conf = len(zero_conf_rows)
    zero_conf_share = (n_zero_conf / rp_n) if rp_n else 0.0
    zero_conf_at_modal = sum(
        1 for r in zero_conf_rows if rp_modal_val is not None and round(r[1], 9) == rp_modal_val
    )

    # gated series: confidence > 0 only (the appraiser's "trust this" subset)
    rp_gated = [r[1] for r in rp_rows if r[3] > 0.0]
    rp_gated_n = len(rp_gated)
    rp_gated_mean, rp_gated_std, rp_gated_degenerate = channel_mean_std(rp_gated)
    rp_gated_modal_val, rp_gated_modal_count, _ = modal_value(rp_gated)
    rp_gated_modal_share = (rp_gated_modal_count / rp_gated_n) if rp_gated_n else 0.0
    rp_gated_n_distinct, rp_gated_top3_share = distinct_value_stats(rp_gated)
    rp_gated_classification = classify_rest_state(
        rp_gated_n, rp_gated_mean, rp_gated_std, rp_gated_degenerate, rp_gated_modal_share,
        rp_gated_n_distinct, rp_gated_top3_share,
    )

    level_label_counts = Counter(r[2] for r in rp_rows)

    rp_extra = [
        f"- `confidence == 0.0` rows (appraiser's own \"no evidence, don't trust this\" signal): "
        f"{n_zero_conf}/{rp_n} ({zero_conf_share*100:.1f}%)",
        f"- Of the modal-value cluster, {zero_conf_at_modal}/{rp_modal_count} rows also have "
        f"`confidence == 0.0` -- {'this modal value is essentially synonymous with the zero-confidence default, not a genuine repeated calm *reading*' if rp_modal_count and zero_conf_at_modal / rp_modal_count > 0.9 else 'not fully synonymous; mixed'}.",
        f"- `level_label` distribution: {dict(level_label_counts)}",
        "- **Unresolved finding, disclosed rather than silently absorbed:** neither live appraiser "
        "(`orion/substrate/appraisal/repair_pressure.py::appraise_repair_pressure`, `orion/substrate/"
        "appraisal/paradigms/repair_pressure_v2.py::run`) emits this modal value for its own "
        "documented \"no evidence\" path -- both return `level=0.0` exactly, not this nonzero "
        "constant. A short search did not locate the smoothing/gating step between appraisal and "
        "`repair_pressure_appraisal_log` persistence that produces it. This script does not "
        "root-cause it; whoever builds hop 0 should treat the exact source of this floor as an open "
        "question, not assume it is organic.",
        "",
        f"### Confidence-gated series (`confidence > 0` only, n={rp_gated_n})",
        "",
        f"- Mean: {rp_gated_mean:.6f}  |  Stddev: {rp_gated_std:.6f}  |  Degenerate: {rp_gated_degenerate}",
        f"- Modal value: {rp_gated_modal_val if rp_gated_modal_val is not None else 'n/a'} "
        f"({rp_gated_modal_count}/{rp_gated_n} = {rp_gated_modal_share*100:.1f}%)",
        f"- Classification (gated): **{rp_gated_classification}**",
    ]

    lines.extend(
        render_series_section(
            "Series 1: `repair_pressure_appraisal_log.level` (ungated, all rows)",
            n=rp_n, mean=rp_mean, std=rp_std, degenerate=rp_degenerate,
            modal_val=rp_modal_val, modal_count=rp_modal_count, modal_share=rp_modal_share,
            classification=rp_classification, extra_lines=rp_extra,
        )
    )

    # --- chat_history_log turn_effect novelty ---
    te_novelty = [v for _, v in te_rows]
    te_n = len(te_novelty)
    te_mean, te_std, te_degenerate = channel_mean_std(te_novelty)
    te_modal_val, te_modal_count, _ = modal_value(te_novelty, round_to=4)
    te_modal_share = (te_modal_count / te_n) if te_n else 0.0
    te_n_distinct, te_top3_share = distinct_value_stats(te_novelty, round_to=4)
    te_classification = classify_rest_state(
        te_n, te_mean, te_std, te_degenerate, te_modal_share, te_n_distinct, te_top3_share
    )
    n_near_one = sum(1 for v in te_novelty if v >= 0.99)
    n_near_zero = sum(1 for v in te_novelty if v <= 0.01)
    te_extra = [
        f"- Rows with novelty >= 0.99 (near-ceiling): {n_near_one}/{te_n} "
        f"({(n_near_one/te_n*100.0) if te_n else 0.0:.1f}%)",
        f"- Rows with novelty <= 0.01 (near-floor): {n_near_zero}/{te_n} "
        f"({(n_near_zero/te_n*100.0) if te_n else 0.0:.1f}%)",
        "- Sourced from `chat_history_log.spark_meta -> 'turn_effect' -> 'turn' -> 'novelty'` over "
        "the last 30 days -- a bounded window, not full history, since this table grows fast and an "
        "unbounded fetch risked silent truncation at the oldest end.",
    ]
    lines.extend(
        render_series_section(
            "Series 2: `chat_history_log.spark_meta->'turn_effect'->'turn'->'novelty'` (last 30 days)",
            n=te_n, mean=te_mean, std=te_std, degenerate=te_degenerate,
            modal_val=te_modal_val, modal_count=te_modal_count, modal_share=te_modal_share,
            classification=te_classification, extra_lines=te_extra,
        )
    )

    lines.extend(
        [
            "## Recommendation",
            "",
        ]
    )
    if rp_gated_classification == "GENUINE_VARIATION":
        lines.append(
            f"- **`repair_pressure_appraisal_log.level`, gated on `confidence > 0`, is the better "
            f"candidate series** ({rp_gated_classification}, n={rp_gated_n}). The ungated series "
            f"reads {rp_classification} only because {zero_conf_share*100:.0f}% of rows are the "
            f"appraiser's own zero-confidence \"no evidence\" default, not real elevated/calm "
            f"readings -- a naive trend reducer over the ungated series would mostly be detecting "
            f"noise in when evidence happens to fire, not real repair-pressure dynamics. Gate on "
            f"confidence before computing any trend."
        )
    else:
        lines.append(
            f"- `repair_pressure_appraisal_log.level` gated on confidence reads "
            f"{rp_gated_classification} (n={rp_gated_n}) -- "
            + (
                "small-N result, worth re-running once more history accumulates before treating as final."
                if rp_gated_n < 100 else "treat per the classification above."
            )
        )
    if te_classification in ("GENUINE_VARIATION", "AMBIGUOUS"):
        lines.append(
            f"- `turn_effect` novelty reads {te_classification} (n={te_n}) -- "
            f"{n_near_one}/{te_n} rows near-ceiling suggests this channel may skew toward "
            f"'everything is novel' rather than a calm/elevated split; worth checking if that's real "
            f"conversational novelty or another instrument that rarely reads calm."
        )
    else:
        lines.append(f"- `turn_effect` novelty reads {te_classification} (n={te_n}).")

    lines.extend(
        [
            "",
            "## Coverage caveats",
            "",
        ]
    )
    all_caveats = list(caveats)
    if rp_n < 100:
        all_caveats.append(
            f"repair_pressure_appraisal_log has only {rp_n} total real rows -- small-N relative to "
            f"the 127k+-row Layer-5 attention probes in this same investigation; re-run this "
            f"measurement once more history accumulates before treating its classification as final."
        )
    if te_n < 100:
        all_caveats.append(
            f"turn_effect novelty (30-day window) has only {te_n} real rows -- same small-N caveat."
        )
    if all_caveats:
        lines.extend(f"- {c}" for c in all_caveats)
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(
        [
            "## Explicit scope boundary",
            "",
            "This script is a measurement only. It did not modify any appraisal, sql-writer, or "
            "reducer code path, and did not build `orion/metacog/trend_reducer.py`. That remains a "
            "separate patch, following the `ReducerSpec` pattern in `orion-substrate-runtime/app/"
            "worker.py`, registered as a flag-gated candidate producer per the stream-of-consciousness "
            "hop-chain design doc.",
            "",
        ]
    )

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")

    with CSV_PATH.open("w", encoding="utf-8") as fh:
        fh.write("series,created_at,value,extra\n")
        for ts, level, label, conf in rp_rows:
            fh.write(f"repair_pressure,{ts},{level},{label}:{conf}\n")
        for ts, v in te_rows:
            fh.write(f"turn_effect_novelty,{ts.isoformat()},{v},\n")

    progress.emit("done", percent=100.0, processed=1, total=1)
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only probe: does turn_effect/repair_pressure have a genuine rest state "
        "before the metacog trend reducer (hop 0) ships live?"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_arg_parser().parse_args(argv)
    return run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
