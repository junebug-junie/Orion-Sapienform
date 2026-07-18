#!/usr/bin/env python3
"""Read-only endogenous-origination gate measurement.

Answers a specific empirical question, replacing a guess with data: live
`drive_audits` shows `tension.endogenous.v1` (the tension `OriginationEngine`
mints when Orion's own internal dynamics cross an origination band with no
exogenous input) has fired ZERO times in the last 48h despite the flag being
on. Is that because Orion's internal signal (D/W/A -> P) never actually
crosses the P >= ORIGINATION_THRESHOLD bar, or because Gate 1
(exogenous_tension_count <= ORIGINATION_EXOGENOUS_FLOOR, i.e. the tick must
be *completely* silent) is structurally unsatisfiable given current tension
volume (~887 tensions/hr, matching the self-state tick rate almost 1:1 per
orion/autonomy/drives_and_autonomy_retrospective.md sec5b)? Those have
different fixes and this script's whole job is to say which one, with
numbers, rather than picking between them by argument.

Method: replay the REAL production code -- `extract_tensions_from_self_state`
(orion/spark/concept_induction/tensions.py) and `OriginationEngine`
(orion/autonomy/endogenous_origination.py) -- over historical
`substrate_self_state` rows in chronological order, exactly as
`ConceptWorker._tensions_from_self_state` calls them live
(orion/spark/concept_induction/bus_worker.py:714-752). No math is
reimplemented here; drift between "what this script measures" and "what the
live worker actually does" is the exact failure class this repo's own
CLAUDE.md warns about for this subsystem, so the production functions are
imported and called directly.

This performs NO writes, emits NO events, flips NO flags, and proposes NO
config change. It reports:

  1. The live-config replay: does the real gate (floor=0, threshold=0.55,
     cooldown=900s) fire at all over the measured window, replayed
     independently of drive_audits (a second, code-level check against the
     zero-fires-in-48h Postgres finding).
  2. P's distribution -- is the signal itself ever above threshold, gate
     aside?
  3. exogenous_tension_count's distribution -- how close to 0 does a tick
     ever get?
  4. A floor sweep (holding threshold/cooldown fixed) and a threshold sweep
     (holding floor/cooldown fixed) -- which knob, if any, is actually
     binding, and by how much.

Run:
    python scripts/analysis/measure_origination_gate.py --window-hours 48
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("orion.analysis.origination_gate")

DEFAULT_WINDOW_HOURS: float = 48.0
MAX_ROWS: int = 200_000

# Live defaults (services/orion-spark-concept-induction/.env, verified
# 2026-07-17) -- kept as script-local constants, NOT re-read from that
# service's env file, so this measurement is reproducible independent of
# whatever the live service happens to be configured to right now.
LIVE_WINDOW = 8
LIVE_THRESHOLD = 0.55
LIVE_COOLDOWN_SEC = 900.0
LIVE_MAG_CAP = 0.5
LIVE_W_DRIFT = 0.4
LIVE_W_DWELL = 0.35
LIVE_W_AGENCY = 0.25
LIVE_EXOGENOUS_FLOOR = 0

FLOOR_SWEEP = (0, 1, 2, 3, 5, 10, 20)
THRESHOLD_SWEEP = (0.55, 0.45, 0.35, 0.25, 0.15)

OUTPUT_DIR = Path("/tmp/origination-gate")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# In-memory record + replay layer.
#
# This calls into the REAL production functions (extract_tensions_from_
# self_state, OriginationEngine) but those functions are themselves pure
# (no I/O -- verified by reading both modules), so this whole layer has no
# I/O of its own and is exercised directly by the unit tests with synthetic
# SelfStateV1 fixtures, no DB involved.
# ===========================================================================


@dataclass
class ReplayTick:
    generated_at: datetime
    exogenous_tension_count: int
    drift: float
    dwell: float
    agency: float
    P: float
    fired_live_config: bool


def replay_origination(
    rows: list[tuple[datetime, dict]],
) -> list[ReplayTick]:
    """Replay the real extraction + origination-engine code over ordered
    (generated_at, self_state_json) rows. Returns one ReplayTick per row that
    parses successfully; malformed rows are skipped (never raises), matching
    the live worker's own defensive posture.
    """
    from orion.autonomy.endogenous_origination import OriginationConfig, OriginationEngine
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.self_state import SelfStateV1
    from orion.spark.concept_induction.tensions import extract_tensions_from_self_state

    engine = OriginationEngine(
        OriginationConfig(
            window=LIVE_WINDOW,
            threshold=LIVE_THRESHOLD,
            cooldown_sec=LIVE_COOLDOWN_SEC,
            mag_cap=LIVE_MAG_CAP,
            w_drift=LIVE_W_DRIFT,
            w_dwell=LIVE_W_DWELL,
            w_agency=LIVE_W_AGENCY,
            exogenous_floor=LIVE_EXOGENOUS_FLOOR,
        )
    )
    source = ServiceRef(name="measure-origination-gate", version="0.1.0", node="analysis")

    out: list[ReplayTick] = []
    previous_state: Optional[SelfStateV1] = None
    for generated_at, payload in rows:
        try:
            state = SelfStateV1.model_validate(payload)
        except Exception:
            continue
        env = BaseEnvelope(
            id=uuid4(),
            kind="substrate.self_state.v1",
            correlation_id=uuid4(),
            created_at=generated_at,
            source=source,
            payload=payload,
        )
        try:
            tensions = extract_tensions_from_self_state(
                envelope=env,
                intake_channel="substrate.self_state.v1",
                self_state=state,
                previous_self_state=previous_state,
            )
        except Exception:
            logger.warning("extract_tensions_from_self_state_failed ts=%s", generated_at, exc_info=True)
            previous_state = state
            continue
        exo_count = len(tensions)

        engine.observe(state)
        fired_tension = engine.maybe_originate(exogenous_tension_count=exo_count, now=generated_at)
        signal = engine.last_signal

        out.append(
            ReplayTick(
                generated_at=generated_at,
                exogenous_tension_count=exo_count,
                drift=float(signal.get("drift", 0.0)),
                dwell=float(signal.get("dwell", 0.0)),
                agency=float(signal.get("agency", 0.0)),
                P=float(signal.get("P", 0.0)),
                fired_live_config=fired_tension is not None,
            )
        )
        previous_state = state
    return out


# ===========================================================================
# Pure summary/sweep layer. Unit-testable on synthetic ReplayTick lists,
# no engine replay or I/O involved.
# ===========================================================================


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


def summarize(values: list[float]) -> Distribution:
    if not values:
        return Distribution()
    return Distribution(
        count=len(values),
        median=float(statistics.median(values)),
        p90=_percentile(values, 0.9),
        max=float(max(values)),
    )


def exogenous_count_histogram(ticks: list[ReplayTick], *, cap: int = 10) -> dict[int, int]:
    """{count: ticks} for exogenous_tension_count in [0, cap]; cap+ bucketed as cap+1."""
    hist: dict[int, int] = {}
    for t in ticks:
        key = t.exogenous_tension_count if t.exogenous_tension_count <= cap else cap + 1
        hist[key] = hist.get(key, 0) + 1
    return hist


def frac_le(ticks: list[ReplayTick], n: int) -> float:
    if not ticks:
        return 0.0
    return sum(1 for t in ticks if t.exogenous_tension_count <= n) / len(ticks)


def frac_p_ge(ticks: list[ReplayTick], threshold: float) -> float:
    if not ticks:
        return 0.0
    return sum(1 for t in ticks if t.P >= threshold) / len(ticks)


def sweep_gate(
    ticks: list[ReplayTick],
    *,
    floors: tuple[int, ...] = FLOOR_SWEEP,
    thresholds: tuple[float, ...] = THRESHOLD_SWEEP,
    cooldown_sec: float = LIVE_COOLDOWN_SEC,
) -> tuple[dict[int, int], dict[float, int]]:
    """Replay just the gate/cooldown decision (P and exogenous_tension_count
    already computed) for each floor value (threshold held at LIVE_THRESHOLD)
    and each threshold value (floor held at LIVE_EXOGENOUS_FLOOR). Sequential,
    respects cooldown exactly like the real engine -- this is the same gate
    logic as OriginationEngine._maybe_originate's Gates 1-3, applied to the
    already-computed per-tick signal so the expensive replay only has to run
    once regardless of how many sweep points are measured.

    Returns (floor -> fire_count, threshold -> fire_count).
    """
    ordered = sorted(ticks, key=lambda t: t.generated_at)

    def _fire_count(*, floor: int, threshold: float) -> int:
        last_fire: Optional[datetime] = None
        fires = 0
        for t in ordered:
            exo_ok = t.exogenous_tension_count <= floor
            cooldown_ok = last_fire is None or (t.generated_at - last_fire).total_seconds() >= cooldown_sec
            threshold_ok = t.P >= threshold
            if exo_ok and cooldown_ok and threshold_ok:
                fires += 1
                last_fire = t.generated_at
        return fires

    floor_results = {f: _fire_count(floor=f, threshold=LIVE_THRESHOLD) for f in floors}
    threshold_results = {th: _fire_count(floor=LIVE_EXOGENOUS_FLOOR, threshold=th) for th in thresholds}
    return floor_results, threshold_results


# ===========================================================================
# I/O layer -- psycopg2 read-only. Mirrors measure_autonomy_gate.py's
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


def fetch_self_state_rows(conn, since: datetime, max_rows: int = MAX_ROWS) -> tuple[list[tuple[datetime, dict]], bool]:
    """Fetch (generated_at, self_state_json) ordered ASC. Returns (rows, truncated)."""
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT generated_at, self_state_json
                FROM substrate_self_state
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_self_state", exc_info=True)
        return [], False
    out: list[tuple[datetime, dict]] = []
    for generated_at, raw_json in rows:
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        payload = raw_json
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        out.append((generated_at, payload))
    return out, len(rows) >= max_rows


def fetch_endogenous_fire_count_postgres(conn, since: datetime) -> Optional[int]:
    """Cross-check against the actually-published record: how many
    drive_audits rows in the window carry tension.endogenous.v1 in
    tension_kinds. None on any failure (table absent, query error) -- this is
    a cross-check, not the primary measurement, so it degrades quietly.
    """
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)
                FROM drive_audits, jsonb_array_elements_text(tension_kinds) AS tk
                WHERE tk = 'tension.endogenous.v1'
                  AND observed_at >= %s
                """,
                (since,),
            )
            row = cur.fetchone()
    except Exception:
        logger.warning("drive_audits endogenous cross-check query failed", exc_info=True)
        return None
    return int(row[0]) if row else None


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


def write_ticks_csv(path: Path, ticks: list[ReplayTick]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generated_at", "exogenous_tension_count", "drift", "dwell", "agency", "P", "fired_live_config"])
        for t in ticks:
            writer.writerow([t.generated_at.isoformat(), t.exogenous_tension_count, f"{t.drift:.4f}", f"{t.dwell:.4f}", f"{t.agency:.4f}", f"{t.P:.4f}", t.fired_live_config])


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    ticks: list[ReplayTick],
    exo_dist: Distribution,
    p_dist: Distribution,
    exo_hist: dict[int, int],
    floor_sweep: dict[int, int],
    threshold_sweep: dict[float, int],
    postgres_fire_count: Optional[int],
    caveats: list[str],
) -> str:
    replay_fires = sum(1 for t in ticks if t.fired_live_config)
    hist_line = ", ".join(
        f"{k if k <= 10 else '10+'}:{v}" for k, v in sorted(exo_hist.items())
    ) or "(no ticks)"
    floor_line = "\n".join(f"| {f} | {n} |" for f, n in sorted(floor_sweep.items()))
    threshold_line = "\n".join(f"| {th} | {n} |" for th, n in sorted(threshold_sweep.items(), reverse=True))
    pg_line = "n/a (cross-check unavailable)" if postgres_fire_count is None else str(postgres_fire_count)

    lines = [
        "# Endogenous Origination Gate Measurement",
        "",
        "Read-only. No writes, no events, no flag/config changes. Replays the real "
        "extract_tensions_from_self_state + OriginationEngine production code over "
        "historical substrate_self_state rows.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- Self-state rows replayed: {len(ticks)}",
        "",
        "## Headline",
        "",
        f"- Replayed fires at LIVE config (floor={LIVE_EXOGENOUS_FLOOR}, "
        f"threshold={LIVE_THRESHOLD}, cooldown={LIVE_COOLDOWN_SEC:.0f}s): **{replay_fires}**",
        f"- Actual published tension.endogenous.v1 count in drive_audits over the same window: **{pg_line}**",
        "",
        "## Is the signal (P) ever above threshold?",
        "",
        f"- P distribution: median={_fmt(p_dist.median)} p90={_fmt(p_dist.p90)} max={_fmt(p_dist.max)} (n={p_dist.count})",
        f"- Fraction of ticks with P >= {LIVE_THRESHOLD} (gate aside): {_fmt(frac_p_ge(ticks, LIVE_THRESHOLD))}",
        "",
        "## Is the tick ever exogenously silent?",
        "",
        f"- exogenous_tension_count distribution: median={_fmt(exo_dist.median)} p90={_fmt(exo_dist.p90)} max={_fmt(exo_dist.max)} (n={exo_dist.count})",
        f"- histogram (count:ticks): {hist_line}",
        f"- fraction of ticks with exogenous_tension_count <= 0 (current floor): {_fmt(frac_le(ticks, 0))}",
        f"- fraction of ticks with exogenous_tension_count <= 2: {_fmt(frac_le(ticks, 2))}",
        f"- fraction of ticks with exogenous_tension_count <= 5: {_fmt(frac_le(ticks, 5))}",
        "",
        "## Floor sweep (threshold + cooldown held at live values)",
        "",
        "| exogenous_floor | fires over window |",
        "| --- | --- |",
        floor_line or "| (no data) | |",
        "",
        "## Threshold sweep (floor + cooldown held at live values)",
        "",
        "| P threshold | fires over window |",
        "| --- | --- |",
        threshold_line or "| (no data) | |",
        "",
        "## Reading this",
        "",
        "- If the floor sweep barely moves off zero even at a generous floor, the binding "
        "constraint is P never reaching threshold, not exogenous silence -- loosening "
        "ORIGINATION_EXOGENOUS_FLOOR would not help.",
        "- If the threshold sweep produces many fires even at low thresholds while floor=0 "
        "produces almost none, exogenous silence (Gate 1) is the actual bottleneck -- the "
        "floor value is the lever, not the D/W/A weights or threshold.",
        "- If neither sweep moves much, P itself needs re-deriving (the D/W/A weights or "
        "their inputs), not just a gate parameter -- a materially different, larger patch.",
        "",
    ]
    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run(window: timedelta, window_label: str) -> int:
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

    rows, truncated = fetch_self_state_rows(conn, window_start)
    if truncated:
        caveats.append(f"self-state rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("self_state loaded", percent=30.0, processed=len(rows), total=len(rows))

    postgres_fire_count = fetch_endogenous_fire_count_postgres(conn, window_start)

    try:
        conn.close()
    except Exception:
        pass

    if not rows:
        caveats.append("no substrate_self_state rows in window; nothing to replay")
        progress.close()
        report = render_report(
            window_label=window_label, window_start=window_start, window_end=now,
            ticks=[], exo_dist=Distribution(), p_dist=Distribution(), exo_hist={},
            floor_sweep={f: 0 for f in FLOOR_SWEEP}, threshold_sweep={t: 0 for t in THRESHOLD_SWEEP},
            postgres_fire_count=postgres_fire_count, caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    progress.emit("replaying", percent=40.0, processed=0, total=len(rows))
    ticks = replay_origination(rows)
    progress.emit("replay done", percent=90.0, processed=len(ticks), total=len(rows))

    exo_dist = summarize([float(t.exogenous_tension_count) for t in ticks])
    p_dist = summarize([t.P for t in ticks])
    exo_hist = exogenous_count_histogram(ticks)
    floor_sweep, threshold_sweep = sweep_gate(ticks)

    write_ticks_csv(CSV_PATH, ticks)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        ticks=ticks,
        exo_dist=exo_dist,
        p_dist=p_dist,
        exo_hist=exo_hist,
        floor_sweep=floor_sweep,
        threshold_sweep=threshold_sweep,
        postgres_fire_count=postgres_fire_count,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(ticks), total=len(rows))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only endogenous-origination gate measurement.")
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
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
