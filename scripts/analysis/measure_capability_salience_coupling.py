#!/usr/bin/env python3
"""Read-only capability-budget <-> field-attention-salience coupling measurement.

`orion/autonomy/capability_policy.py`'s `evaluate_capability()` currently
gates autonomous-action capability budget with a flat `budget_per_cycle`
integer per rule in `config/autonomy/capability_policy.v1.yaml` (plus
categorical checks: goal status, drive_origin membership, signal kinds).
`orion/sentience_striving_program/README.md` (not yet merged, but describes
real, agreed direction) wants that budget eventually coupled to Orion's real
internal state instead of a flat allowance -- specifically to the
ALREADY-LIVE field-attention salience system
(`orion/attention/field_attention/{scoring,selectors}.py`, running
continuously via `orion-attention-runtime`, writing `FieldAttentionFrameV1`
rows to Postgres `substrate_attention_frames` every ~2 seconds).

The natural integration point has a real naming mismatch:
`capability_policy.v1.yaml`'s rules use action-permission IDs
(`web.fetch.readonly`, `journal.compose.episode`, etc.), while the live
field-attention system's capability targets
(`FieldAttentionFrameV1.capability_targets`, each a `FieldAttentionTargetV1`
with a `target_id` and a `salience_score` in [0, 1]) use different,
infrastructure-capacity IDs -- confirmed live in Postgres:
`capability:transport`, `capability:llm_inference`, and likely others.
There is no automatic 1:1 match; any live wiring needs an explicit,
human-legible mapping, measured against real historical data BEFORE
anything gets wired live, not guessed and shipped. This script is that
measurement. It is explicitly NOT the live wiring -- it does not modify
`orion/autonomy/capability_policy.py`, `config/autonomy/capability_policy.v1.yaml`,
or any field-attention code.

The mapping used here (already decided, not re-derived by this script):

    web.fetch.readonly      -> capability:transport
    web.fetch.write         -> capability:transport
    world_pulse.run         -> capability:transport
    recall.query.readonly   -> capability:transport
    journal.compose.episode -> capability:llm_inference

Rationale: the first four are all outbound network/RPC-bound actions (web
fetch, world-pulse run, recall service RPC) -- `capability:transport` fits
all of them. `journal.compose.episode` is the one rule that requires
generating real text via an LLM call, matching `capability:llm_inference`
specifically.

This performs NO writes, emits NO events, flips NO flags, and proposes NO
config change. For each of the 5 capability rules above, it replays the
mapped target's historical `salience_score` series from
`substrate_attention_frames` over the measurement window and reports:

  1. The salience distribution (median / p90 / max / count).
  2. A threshold sweep at (0.10, 0.25, 0.45, 0.70) -- matching
     `config/attention/field_attention_policy.v1.yaml`'s own
     `thresholds.min_salience=0.10` / `thresholds.high_salience=0.70`, plus
     two intermediate points -- showing what fraction of ticks would clear
     each candidate salience floor.

A tick where the mapped target_id is absent from that tick's
`capability_targets` list means it wasn't salient enough to appear at all
(below `min_salience` in the live policy) -- this is treated as
`salience_score = 0.0`, not a missing/skipped observation.

Run:
    python scripts/analysis/measure_capability_salience_coupling.py --window-hours 48
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.capability_salience_coupling")

DEFAULT_WINDOW_HOURS: float = 48.0
MAX_ROWS: int = 200_000

# The decided mapping from capability_policy.v1.yaml rule IDs to
# field-attention capability_targets target_id. See module docstring for
# rationale. Not re-derived here -- this script only measures against it.
CAPABILITY_TARGET_MAP: dict[str, str] = {
    "web.fetch.readonly": "capability:transport",
    "web.fetch.write": "capability:transport",
    "world_pulse.run": "capability:transport",
    "recall.query.readonly": "capability:transport",
    "journal.compose.episode": "capability:llm_inference",
}

# Matches config/attention/field_attention_policy.v1.yaml's
# thresholds.min_salience=0.10 / thresholds.high_salience=0.70, plus two
# intermediate points. Not arbitrary -- do not change without re-reading
# that policy file.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.10, 0.25, 0.45, 0.70)

OUTPUT_DIR = Path("/tmp/capability-salience-coupling")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure extraction + summary layer. No I/O. Unit-testable directly on
# synthetic capability_targets JSON blobs and synthetic (datetime, float)
# tick lists -- no engine replay, no DB.
# ===========================================================================


def extract_salience_for_target(targets_raw: Any, target_id: str) -> float:
    """Given the raw `frame_json->'capability_targets'` value for one tick
    (a JSON array of FieldAttentionTargetV1-shaped objects, possibly still a
    JSON string, possibly malformed, possibly None), return the
    `salience_score` for the entry whose `target_id` matches, or 0.0 if the
    target is absent from this tick's list (it wasn't salient enough to
    appear at all -- below `min_salience` in the live policy, not a missing
    observation) or if the payload is malformed in any way. Never raises.
    """
    try:
        if targets_raw is None:
            return 0.0
        if isinstance(targets_raw, str):
            targets_raw = json.loads(targets_raw)
        if not isinstance(targets_raw, list):
            return 0.0
        for entry in targets_raw:
            if not isinstance(entry, dict):
                continue
            if entry.get("target_id") == target_id:
                try:
                    return float(entry.get("salience_score", 0.0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0
    except Exception:
        return 0.0


@dataclass
class Distribution:
    count: int = 0
    median: Optional[float] = None
    p90: Optional[float] = None
    max: Optional[float] = None


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def summarize_salience(values: list[float]) -> Distribution:
    if not values:
        return Distribution()
    return Distribution(
        count=len(values),
        median=float(statistics.median(values)),
        p90=_percentile(values, 0.9),
        max=float(max(values)),
    )


def frac_ge(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v >= threshold) / len(values)


def threshold_sweep(
    values: list[float], thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
) -> dict[float, float]:
    """For each threshold, what fraction of ticks had salience >= threshold."""
    return {th: frac_ge(values, th) for th in thresholds}


# ===========================================================================
# I/O layer -- psycopg2 read-only. Function body is reused verbatim from
# measure_origination_gate.py's open_readonly_connection() (refuses a
# non-read-only session), not reinvented here.
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


def fetch_capability_target_rows(
    conn, since: datetime, target_id: str, max_rows: int = MAX_ROWS
) -> tuple[list[tuple[datetime, dict]], bool]:
    """Fetch (generated_at, {"target_id", "salience_score"}) ordered ASC for
    one capability target_id, over substrate_attention_frames rows with
    generated_at >= since. Returns (rows, truncated). Never raises -- a
    malformed row is skipped (extract_salience_for_target itself never
    raises, but the surrounding row-tuple handling is still guarded).
    """
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT generated_at, frame_json->'capability_targets' AS targets
                FROM substrate_attention_frames
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_frames", exc_info=True)
        return [], False
    out: list[tuple[datetime, dict]] = []
    for generated_at, targets_raw in rows:
        try:
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            score = extract_salience_for_target(targets_raw, target_id)
            out.append((generated_at, {"target_id": target_id, "salience_score": score}))
        except Exception:
            logger.warning("skipping malformed attention-frame row ts=%s", generated_at, exc_info=True)
            continue
    return out, len(rows) >= max_rows


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


def write_ticks_csv(path: Path, rule_rows: dict[str, list[tuple[datetime, dict]]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rule_id", "target_id", "generated_at", "salience_score"])
        for rule_id, rows in rule_rows.items():
            for generated_at, rec in rows:
                writer.writerow([rule_id, rec["target_id"], generated_at.isoformat(), f"{rec['salience_score']:.4f}"])


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    rule_results: dict[str, dict[str, Any]],
    thresholds: tuple[float, ...],
    caveats: list[str],
) -> str:
    lines = [
        "# Capability Budget <-> Field-Attention Salience Coupling Measurement",
        "",
        "Read-only. No writes, no events, no flag/config changes. Measures the "
        "decided rule -> capability_target mapping (see script docstring) against "
        "real historical `substrate_attention_frames` rows. This is NOT the live "
        "wiring -- `orion/autonomy/capability_policy.py` and "
        "`config/autonomy/capability_policy.v1.yaml` are untouched.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        "",
        "## Mapping under measurement",
        "",
        "| capability rule | field-attention target_id |",
        "| --- | --- |",
    ]
    for rule_id, target_id in CAPABILITY_TARGET_MAP.items():
        lines.append(f"| `{rule_id}` | `{target_id}` |")
    lines.extend(
        [
            "",
            "## Salience distribution and threshold sweep, per rule",
            "",
            "A tick where the target is absent from that tick's `capability_targets` "
            "list counts as `salience_score = 0.0` (below the live policy's "
            "`min_salience=0.10`, not a missing observation).",
            "",
            "| rule | target | n ticks | median | p90 | max | "
            + " | ".join(f">= {th:g}" for th in thresholds)
            + " |",
            "| --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in thresholds) + " |",
        ]
    )
    for rule_id, result in rule_results.items():
        dist: Distribution = result["dist"]
        sweep: dict[float, float] = result["sweep"]
        sweep_cells = " | ".join(_fmt(sweep.get(th)) for th in thresholds)
        lines.append(
            f"| `{rule_id}` | `{result['target_id']}` | {dist.count} | "
            f"{_fmt(dist.median)} | {_fmt(dist.p90)} | {_fmt(dist.max)} | {sweep_cells} |"
        )
    lines.extend(
        [
            "",
            "## Reading this",
            "",
            "- A rule whose fraction at `>= 0.10` is near 1.0 means its mapped target is "
            "almost always at least minimally salient -- coupling budget to salience "
            "there would rarely starve the rule below its current flat allowance.",
            "- A rule whose fraction at `>= 0.70` is near 0.0 means its mapped target "
            "essentially never reaches `high_salience` -- a budget formula that scales "
            "up sharply only past that threshold would rarely engage for this rule.",
            "- Rules sharing a target_id (all four `capability:transport` rules here) "
            "will show identical distributions by construction -- that is expected, not "
            "a bug: the live field-attention system scores per infrastructure capacity, "
            "not per action-permission rule, so any coupling necessarily shares signal "
            "across rules mapped to the same target.",
            "",
        ]
    )
    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run(window: timedelta, window_label: str, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> int:
    now = datetime.now(timezone.utc)
    window_start = now - window
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(PROGRESS_PATH)
    caveats: list[str] = []

    progress.emit("connect", percent=0.0, processed=0, total=0)
    conn = open_readonly_connection(dsn)
    if conn is None:
        caveats.append("postgres unavailable or not read-only; nothing measured")
        progress.close()
        print("ERROR: could not open a read-only postgres connection; see log")
        return 2

    unique_targets = sorted(set(CAPABILITY_TARGET_MAP.values()))
    target_rows: dict[str, list[tuple[datetime, dict]]] = {}
    for idx, target_id in enumerate(unique_targets):
        rows, truncated = fetch_capability_target_rows(conn, window_start, target_id, MAX_ROWS)
        target_rows[target_id] = rows
        if truncated:
            caveats.append(f"rows truncated at MAX_ROWS={MAX_ROWS} for target {target_id}")
        progress.emit(
            f"fetched {target_id}",
            percent=10.0 + 60.0 * (idx + 1) / max(len(unique_targets), 1),
            processed=len(rows),
            total=len(rows),
        )

    try:
        conn.close()
    except Exception:
        pass

    rule_results: dict[str, dict[str, Any]] = {}
    rule_rows: dict[str, list[tuple[datetime, dict]]] = {}
    for rule_id, target_id in CAPABILITY_TARGET_MAP.items():
        rows = target_rows.get(target_id, [])
        rule_rows[rule_id] = rows
        values = [rec["salience_score"] for _, rec in rows]
        dist = summarize_salience(values)
        sweep = threshold_sweep(values, thresholds)
        rule_results[rule_id] = {"target_id": target_id, "dist": dist, "sweep": sweep}
        if not rows:
            caveats.append(f"no attention-frame rows found in window for target {target_id} (rule {rule_id})")

    write_ticks_csv(CSV_PATH, rule_rows)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        rule_results=rule_results,
        thresholds=thresholds,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(rule_rows), total=len(rule_rows))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only capability-budget <-> field-attention-salience coupling measurement."
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=DEFAULT_WINDOW_HOURS,
        help=f"analysis window in hours (default {DEFAULT_WINDOW_HOURS})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    window = timedelta(hours=args.window_hours)
    window_label = f"{args.window_hours:g}h"
    return run(window, window_label)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
