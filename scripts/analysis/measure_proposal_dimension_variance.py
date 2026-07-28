#!/usr/bin/env python3
"""Read-only measurement: does `orion/proposals/scoring.py::dimension_confidence()`
have real, non-degenerate historical variance to precision-weight against?

Session context (2026-07-28, Juniper conversation, "HIT IT" authorization for the
"Recommended next patch" of `docs/superpowers/specs/2026-07-28-precision-weighted-
proposal-scoring-design.md`): tonight's investigation found the same
"fixed-constant-instead-of-adaptive-baseline" disease in four places in this
codebase. Three were already fixed live tonight (`bus_synaptic_prediction_error`
2026-07-26, `recent_perturbations`/Layer 5 attention PR #1433, `execution_
prediction_error` PR #1434), all via `orion.bus.ewma.compute_ewma_update` --
a live EWMA baseline compared against, instead of a frozen constant. The fourth
is `dimension_confidence()`: currently `1.0 if dimension_id in field_pressures
else 0.0` -- a binary data-presence flag wearing a name that implies real
epistemic confidence.

This script answers the design doc's Missing Questions 1 and 2 -- BEFORE any
EWMA code gets written, per the doc's own "Recommended next patch" section and
CLAUDE.md's metric-quality-gate step 4 (real data first, no guessed formula):

  1. What real historical range/variance does each of the four
     `orion.proposals.scoring.PRESSURE_DIMENSIONS` (execution_pressure,
     resource_pressure, reasoning_pressure, reliability_pressure) actually have?
  2. Does `field_pressures()`'s source (`FieldStateV1` / `substrate_field_state`,
     via `orion.field.pressure.field_pressures()`) have durable, queryable
     historical rows to compute a real EWMA baseline against?

## Disclosed scoping decisions

1. **Reuses the real production code, not a reimplementation.** Every historical
   row is parsed into a real `FieldStateV1` (`orion.schemas.field_state`) and run
   through the real, unmodified `orion.field.pressure.field_pressures()` --
   exactly the function `orion/proposals/builder.py::build_proposal_frame()`
   calls in production (confirmed by import, not by re-deriving the merge/map
   logic here). This script does not edit, import as executable-and-mutate, or
   otherwise touch any live scoring path.

2. **Data source, confirmed live, not assumed:** `services/orion-proposal-
   runtime/app/store.py::ProposalRuntimeStore.load_latest_field()` reads
   `SELECT field_json FROM substrate_field_state ORDER BY generated_at DESC
   LIMIT 1` -- this script queries that exact same table. Row count, time
   range, and per-row byte size are all queried fresh at the top of every run
   (see "Data source" section of the report), not cited from a prior write-up.

3. **Bounded recent window, not full table history.** `substrate_field_state`
   is an append-only, unbounded-growth table (one row per ~2s digestion tick,
   never pruned) -- live-confirmed this run at ~25KB/row average and ~1.2GB
   total for ~72h of real history (127k+ rows). Loading the full table would
   mean multi-GB of JSON parsing for a variance estimate that does not need
   that much data. Defaults to the most recent `--window-hours` (6.0) of real
   history, matching `measure_ast_hot_reducer.py`'s own `--window-hours`
   convention -- overridable, and the real row count actually used is reported,
   not silently truncated without disclosure. Rows are streamed via a
   server-side (named) psycopg2 cursor so client memory stays bounded by the
   window, not the whole table.

4. **Missing dimension in a tick's `field_pressures()` output is reported as
   ABSENT, never fabricated as 0.0.** `field_pressures()` only includes a
   dimension key when at least one real channel mapped to it this tick (see
   `orion/field/pressure.py::map_channels_to_dimensions`) -- an absent
   dimension means "no data this tick", which is a different fact from "data
   present, reading zero". Per-dimension coverage (ticks present / ticks in
   window) is its own reported number.

5. **Degenerate-variance threshold reused verbatim from this directory's own
   convention**: `DEGENERATE_VARIANCE_EPS = 1e-12`
   (`measure_attention_salience_normalization.py`,
   `measure_emergent_clustering_probe.py::pearson_correlation`).

6. **Decay-artifact detector, not just a variance number.** CLAUDE.md's
   metric-quality-gate step 4 requires checking BOTH failure directions, not
   just "does it vary": a metric that never varies (useless for precision
   weighting) AND a metric that reads a suspiciously clean near-zero that
   is actually a decayed-to-zero artifact rather than genuine calm --
   documented live 2026-07-26 for `node:substrate.route`'s `prediction_error`
   (silently multiplied by `NODE_DECAY_CHANNELS`' 0.92/tick staleness decay
   for 48+ hours with no fresh perturbation, going subnormal). This is a
   real, live risk for THIS measurement specifically:
   `services/orion-field-digester/app/digestion/decay.py::NODE_DECAY_CHANNELS`
   and `CAPABILITY_DECAY_CHANNELS` both list `reliability_pressure` by name
   -- confirmed by direct read of that module before writing this script, not
   assumed. `detect_decay_artifact()` below looks for a long run of
   consecutive strictly-decreasing values with a near-constant ratio (the
   exact signature of unopposed geometric decay) in each dimension's real
   series, and reports the ratio found so it can be compared by hand against
   `BIOMETRICS_FIELD_DECAY_RATE`'s live default (0.92,
   `services/orion-field-digester/app/settings.py`).

7. **This script is strictly read-only.** Refuses to proceed if the Postgres
   session cannot be forced read-only (same contract as every sibling
   `measure_*` script). Writes only to `/tmp/measure-proposal-dimension-
   variance/`. Does not modify `orion/proposals/scoring.py`,
   `orion/field/pressure.py`, `orion/schemas/field_state.py`, or any database
   table. If this measurement supports precision-weighting, implementing it is
   a separate, explicit patch (Step 2 of this task) -- not a side effect of
   running this script.

Run:
    python scripts/analysis/measure_proposal_dimension_variance.py --window-hours 6
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from orion.field.pressure import field_pressures
from orion.proposals.scoring import PRESSURE_DIMENSIONS
from orion.schemas.field_state import FieldStateV1

logger = logging.getLogger("orion.analysis.proposal_dimension_variance")

DEGENERATE_VARIANCE_EPS: float = 1e-12
MIN_TOTAL_ROWS: int = 200
DEFAULT_WINDOW_HOURS: float = 6.0
MAX_ROWS: int = 30_000
SERVER_CURSOR_ITERSIZE: int = 500

# Reused verbatim from services/orion-field-digester/app/digestion/decay.py
# (read directly before writing this script, not assumed) -- the live default
# per-tick staleness-decay ratio. A dimension whose real series shows a long,
# near-constant-ratio decreasing run close to this value is a strong signal of
# the node:substrate.route class of bug (2026-07-26): geometric decay toward
# zero from a stalled producer, not genuine calm.
KNOWN_DECAY_RATE: float = 0.92
DECAY_ARTIFACT_MIN_STREAK: int = 20
DECAY_ARTIFACT_RATIO_TOLERANCE: float = 0.03  # +/- around a candidate ratio

# Disqualification is by REAL-EVENT COUNT (zero anywhere in the window == the
# actual node:substrate.route signature, see LivenessFinding's docstring),
# not by recency-since-last-event against a flat cutoff -- a flat-threshold
# version of this check was tried first and mislabeled a real but sparsely-
# updating dimension as "abandoned" (see classify_dimension's docstring and
# the Step 1 verdict narrative for the concrete numbers). Recency and mean
# inter-event gap are still reported per-dimension for human judgment.
UPWARD_TRANSITION_EPS: float = 1e-9

DIMENSIONS: tuple[str, ...] = tuple(sorted(PRESSURE_DIMENSIONS))

OUTPUT_DIR = Path("/tmp/measure-proposal-dimension-variance")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O.
# ===========================================================================


@dataclass
class DimensionStats:
    n_present: int
    n_window: int
    mean: float
    variance: float
    stddev: float
    min_value: float
    max_value: float
    degenerate: bool
    decay_artifact: "DecayArtifactFinding"
    liveness: "LivenessFinding"


@dataclass
class DecayArtifactFinding:
    suspected: bool
    streak_len: int = 0
    ratio_mean: Optional[float] = None
    ratio_stddev: Optional[float] = None
    matches_known_decay_rate: bool = False
    streak_start_index: Optional[int] = None


@dataclass
class LivenessFinding:
    """Distinguishes "this channel legitimately decays toward a quiet floor
    between real, periodically-arriving perturbations" (expected, by-design
    behavior of `services/orion-field-digester/app/digestion/decay.py`'s
    hold-then-decay mechanic) from "this channel is structurally abandoned
    and only ever decays" -- the actual `node:substrate.route` disease
    (2026-07-26: ZERO real events for 48+ hours straight, decaying unopposed
    toward subnormal). A long near-constant-ratio decreasing streak
    (`DecayArtifactFinding.suspected`) is NOT itself disqualifying if the
    channel is still being genuinely refreshed by real upward (perturbation-
    driven) jumps at all during the window.

    Deliberately does NOT disqualify on recency-since-last-event alone: a
    first attempt at this script used a flat 1-hour "abandoned" cutoff, then
    that was live-checked against a 16h window and found to mislabel
    `reasoning_pressure` (13 real events/16h, ~74min average gap -- an event
    89min in the past is well within this dimension's own normal rhythm, not
    evidence it died) as a false-positive "abandoned producer". The real
    node:substrate.route signature is ZERO events across the ENTIRE observed
    history, not "currently mid-quiet-stretch relative to an arbitrary global
    threshold" -- `no_real_events_in_window` tests exactly that, and
    `mean_inter_event_gap_sec` / `seconds_since_last_upward_transition` are
    reported alongside it so a human can judge whether "quiet right now" is
    within this dimension's own normal cadence.
    """

    n_upward_transitions: int
    last_upward_transition_ts: Optional[str]
    seconds_since_last_upward_transition: Optional[float]
    mean_inter_event_gap_sec: Optional[float]
    no_real_events_in_window: bool


def population_mean_variance(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance


def detect_decay_artifact(values: list[float]) -> DecayArtifactFinding:
    """Longest run of consecutive, strictly-decreasing, near-constant-ratio
    values in `values` (in tick order, already restricted to present-only
    observations for one dimension). A long streak with a near-constant ratio
    is the exact signature of unopposed per-tick geometric decay (see module
    docstring #6) -- NOT evidence on its own that this happened (a real signal
    can also decrease smoothly for other reasons), but flagged explicitly so a
    human decides rather than a silent "reads near zero == calm" assumption.
    """
    n = len(values)
    if n < DECAY_ARTIFACT_MIN_STREAK + 1:
        return DecayArtifactFinding(suspected=False)

    best_len = 0
    best_start = 0
    best_ratios: list[float] = []

    cur_start = 0
    cur_ratios: list[float] = []
    for i in range(1, n):
        prev_v, cur_v = values[i - 1], values[i]
        if prev_v > 0.0 and cur_v > 0.0 and cur_v < prev_v:
            cur_ratios.append(cur_v / prev_v)
        else:
            streak_len = i - cur_start
            if streak_len > best_len:
                best_len = streak_len
                best_start = cur_start
                best_ratios = cur_ratios
            cur_start = i
            cur_ratios = []
    streak_len = n - cur_start
    if streak_len > best_len:
        best_len = streak_len
        best_start = cur_start
        best_ratios = cur_ratios

    if best_len < DECAY_ARTIFACT_MIN_STREAK or not best_ratios:
        return DecayArtifactFinding(suspected=False, streak_len=best_len)

    ratio_mean, ratio_variance = population_mean_variance(best_ratios)
    ratio_stddev = math.sqrt(ratio_variance)
    # "Near-constant ratio" -- coefficient of variation small relative to the
    # mean ratio. A genuinely noisy real signal that merely trends down would
    # not hold a tight ratio across 20+ consecutive ticks.
    near_constant = ratio_mean > 0.0 and (ratio_stddev / ratio_mean) < 0.05
    matches_known = (
        near_constant
        and abs(ratio_mean - KNOWN_DECAY_RATE) <= DECAY_ARTIFACT_RATIO_TOLERANCE
    )
    return DecayArtifactFinding(
        suspected=near_constant,
        streak_len=best_len,
        ratio_mean=ratio_mean,
        ratio_stddev=ratio_stddev,
        matches_known_decay_rate=matches_known,
        streak_start_index=best_start,
    )


def detect_producer_liveness(
    present_timestamps: list[datetime], present_values: list[float], window_end: datetime
) -> LivenessFinding:
    """Counts real upward (perturbation-driven) transitions -- decay only ever
    moves a value DOWN (`vec[ch] = vec[ch] * decay_rate`,
    `services/orion-field-digester/app/digestion/decay.py`), so any strict
    increase can only come from a real `apply_perturbations()` write. Reports
    how long ago the most recent one was, relative to the window's end --
    the actual test for "is this the node:substrate.route disease" (module
    docstring #6), separate from whether a long decay-ratio streak exists at
    all (which, on its own, is expected, by-design behavior for any
    `NODE_DECAY_CHANNELS`/`CAPABILITY_DECAY_CHANNELS` entry during a real
    quiet stretch between events).
    """
    n = len(present_values)
    event_indices: list[int] = []
    for i in range(1, n):
        if present_values[i] > present_values[i - 1] + UPWARD_TRANSITION_EPS:
            event_indices.append(i)
    if not event_indices:
        return LivenessFinding(
            n_upward_transitions=0,
            last_upward_transition_ts=None,
            seconds_since_last_upward_transition=None,
            mean_inter_event_gap_sec=None,
            no_real_events_in_window=True,
        )
    last_idx = event_indices[-1]
    last_ts = present_timestamps[last_idx]
    seconds_since = (window_end - last_ts).total_seconds()
    mean_gap: Optional[float] = None
    if len(event_indices) >= 2:
        event_ts = [present_timestamps[i] for i in event_indices]
        gaps = [
            (event_ts[i] - event_ts[i - 1]).total_seconds() for i in range(1, len(event_ts))
        ]
        mean_gap = sum(gaps) / len(gaps)
    return LivenessFinding(
        n_upward_transitions=len(event_indices),
        last_upward_transition_ts=last_ts.isoformat(),
        seconds_since_last_upward_transition=seconds_since,
        mean_inter_event_gap_sec=mean_gap,
        no_real_events_in_window=False,
    )


def compute_dimension_stats(
    present_timestamps: list[datetime],
    present_values: list[float],
    n_window: int,
    window_end: datetime,
) -> DimensionStats:
    mean, variance = population_mean_variance(present_values)
    stddev = math.sqrt(variance)
    degenerate = variance <= DEGENERATE_VARIANCE_EPS
    decay = detect_decay_artifact(present_values)
    liveness = detect_producer_liveness(present_timestamps, present_values, window_end)
    return DimensionStats(
        n_present=len(present_values),
        n_window=n_window,
        mean=mean,
        variance=variance,
        stddev=stddev,
        min_value=min(present_values) if present_values else 0.0,
        max_value=max(present_values) if present_values else 0.0,
        degenerate=degenerate,
        decay_artifact=decay,
        liveness=liveness,
    )


def classify_dimension(stats: DimensionStats) -> str:
    """Explicit classification band per dimension, disclosed here rather than
    hidden in an if/else (same discipline as `measure_attention_salience_
    normalization.py::classify_normalization_effect`).

    - `INSUFFICIENT_DATA`: fewer than a usable minimum of present
      observations in the window.
    - `DEGENERATE_FLAT`: real observations exist but variance is at/below
      `DEGENERATE_VARIANCE_EPS` -- no real spread to precision-weight
      against.
    - `NO_REAL_EVENTS_IN_WINDOW`: the actual `node:substrate.route` disease --
      ZERO real upward (perturbation-driven) transitions anywhere in the
      window. Whatever variance this dimension shows is stale decay heading
      to zero, not a live signal -- disqualifying regardless of whether a
      decay-ratio streak was separately detected.
    - `ACTIVE_WITH_DECAY_MECHANIC`: a long, near-constant-ratio decreasing
      run WAS found (confirms the hold-then-decay mechanic is genuinely
      active for part of the window), but the channel is also being
      recently, really refreshed by real upward transitions -- this is
      expected, by-design behavior (quiet stretch between real events), not
      the abandoned-producer disease. Usable.
    - `NON_DEGENERATE_USABLE`: real, non-flat variance, no dominant decay-
      artifact streak found at all.
    """
    if stats.n_present < DECAY_ARTIFACT_MIN_STREAK:
        return "INSUFFICIENT_DATA"
    if stats.degenerate:
        return "DEGENERATE_FLAT"
    if stats.liveness.no_real_events_in_window:
        return "NO_REAL_EVENTS_IN_WINDOW"
    if stats.decay_artifact.suspected:
        return "ACTIVE_WITH_DECAY_MECHANIC"
    return "NON_DEGENERATE_USABLE"


# ===========================================================================
# I/O layer -- psycopg2 read-only, server-side cursor to bound client memory
# against substrate_field_state's real per-row size (live-confirmed ~25KB
# average, see report's "Data source" section).
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


def fetch_field_state_summary(conn) -> tuple[Optional[datetime], Optional[datetime], int, float]:
    """(min_generated_at, max_generated_at, total_row_count, avg_row_bytes)
    over the WHOLE table -- confirms Missing Question 2 (durable history
    exists) and justifies the bounded-window decision (module docstring #3),
    queried fresh, not assumed."""
    if conn is None:
        return None, None, 0, 0.0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT min(generated_at), max(generated_at), count(*), "
                "avg(octet_length(field_json::text)) FROM substrate_field_state"
            )
            row = cur.fetchone()
    except Exception:
        logger.error("failed to fetch substrate_field_state summary", exc_info=True)
        return None, None, 0, 0.0
    if not row:
        return None, None, 0, 0.0
    min_ts, max_ts, count, avg_bytes = row
    if min_ts is not None and min_ts.tzinfo is None:
        min_ts = min_ts.replace(tzinfo=timezone.utc)
    if max_ts is not None and max_ts.tzinfo is None:
        max_ts = max_ts.replace(tzinfo=timezone.utc)
    return min_ts, max_ts, int(count or 0), float(avg_bytes or 0.0)


def stream_field_state_rows(
    conn, start: datetime, end: datetime, max_rows: int, progress: "ProgressLog"
) -> tuple[list[tuple[datetime, dict[str, float]]], int, bool]:
    """Streams real rows in [start, end], ASC by generated_at, via a named
    (server-side) cursor so the client never buffers more than
    SERVER_CURSOR_ITERSIZE raw field_json payloads at once. Each row is
    reduced to (generated_at, dimension_dict) via the REAL
    `orion.field.pressure.field_pressures()` immediately, then the raw
    payload is discarded -- module docstring #1/#3.

    Returns (reduced_rows, n_schema_errors, truncated).
    """
    if conn is None:
        return [], 0, False
    reduced: list[tuple[datetime, dict[str, float]]] = []
    n_errors = 0
    n_seen = 0
    # A named (server-side) cursor requires a real transaction -- autocommit
    # mode (set by open_readonly_connection for the read-only-session check)
    # is incompatible with it. Session-level read-only is already enforced by
    # `SET default_transaction_read_only = on`, so this transaction can only
    # ever SELECT regardless of autocommit; explicitly rolled back below since
    # nothing is ever written.
    conn.autocommit = False
    try:
        with conn.cursor(name="proposal_dimension_variance_cursor") as cur:
            cur.itersize = SERVER_CURSOR_ITERSIZE
            cur.execute(
                """
                SELECT generated_at, field_json
                FROM substrate_field_state
                WHERE generated_at >= %s AND generated_at <= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (start, end, max_rows),
            )
            for generated_at, payload in cur:
                n_seen += 1
                if generated_at.tzinfo is None:
                    generated_at = generated_at.replace(tzinfo=timezone.utc)
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        n_errors += 1
                        continue
                try:
                    state = FieldStateV1.model_validate(payload)
                    dims = field_pressures(state)
                except Exception:
                    n_errors += 1
                    continue
                reduced.append((generated_at, dims))
                if n_seen % 2000 == 0:
                    progress.emit(
                        "streaming substrate_field_state",
                        percent=min(95.0, 20.0 + 70.0 * (n_seen / max(max_rows, 1))),
                        processed=n_seen,
                        total=max_rows,
                    )
    except Exception:
        logger.error("failed to stream substrate_field_state rows", exc_info=True)
        return reduced, n_errors, False
    finally:
        try:
            conn.rollback()  # read-only: nothing written, nothing to commit
        except Exception:
            pass
        conn.autocommit = True
    return reduced, n_errors, n_seen >= max_rows


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


def write_ticks_csv(
    path: Path,
    rows: list[tuple[datetime, dict[str, float]]],
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generated_at", *DIMENSIONS])
        for ts, dims in rows:
            row = [ts.isoformat()]
            for d in DIMENSIONS:
                v = dims.get(d)
                row.append("" if v is None else f"{v:.6f}")
            writer.writerow(row)


def render_report(
    *,
    table_summary_label: str,
    total_table_rows: int,
    avg_row_bytes: float,
    window_label: str,
    n_window_rows: int,
    n_schema_errors: int,
    truncated: bool,
    dim_series: dict[str, list[float]],
    dim_stats: dict[str, DimensionStats],
    dim_classification: dict[str, str],
    caveats: list[str],
) -> str:
    lines = [
        "# Proposal Scoring `dimension_confidence()` -- Real Historical Variance Probe",
        "",
        "Read-only. No writes, no events, no flag/config changes, no edits to "
        "`orion/proposals/scoring.py`, `orion/field/pressure.py`, "
        "`orion/schemas/field_state.py`, or any database table. Answers Missing "
        "Questions 1-2 of `docs/superpowers/specs/2026-07-28-precision-weighted-"
        "proposal-scoring-design.md` by replaying the REAL, unmodified "
        "`orion.field.pressure.field_pressures()` over real "
        "`substrate_field_state` history.",
        "",
        "## Data source (Missing Question 2)",
        "",
        f"- Table: `substrate_field_state` (confirmed live as the exact table "
        f"`services/orion-proposal-runtime/app/store.py::ProposalRuntimeStore."
        f"load_latest_field()` reads via `SELECT field_json ... ORDER BY "
        f"generated_at DESC LIMIT 1`)",
        f"- Full-table real history span: {table_summary_label}",
        f"- Full-table real row count: {total_table_rows}",
        f"- Average `field_json` payload size: {avg_row_bytes:,.0f} bytes/row",
        f"- **Durable, queryable history: YES** -- {total_table_rows} real rows "
        f"confirmed present, not fabricated or assumed.",
        "",
        "## Measurement window (Missing Question 1)",
        "",
        f"- Window used: {window_label}",
        f"- Real rows in window: {n_window_rows}",
        f"- Rows that failed to parse as `FieldStateV1` (schema errors, excluded): "
        f"{n_schema_errors}",
    ]
    if truncated:
        lines.append(
            f"- **Window truncated at MAX_ROWS={MAX_ROWS}** -- fetch is ORDER BY "
            f"generated_at ASC, so truncation keeps the OLDEST rows in the window "
            f"and drops the newest. Narrow `--window-hours` to avoid this."
        )
    lines.extend(["", "## Per-dimension real historical stats", ""])
    lines.append(
        "| dimension | present/window | coverage | mean | stddev | variance | min | max | classification |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for d in DIMENSIONS:
        st = dim_stats[d]
        coverage = st.n_present / st.n_window if st.n_window else 0.0
        lines.append(
            f"| `{d}` | {st.n_present}/{st.n_window} | {coverage*100:.1f}% | "
            f"{st.mean:.6g} | {st.stddev:.6g} | {st.variance:.6g} | "
            f"{st.min_value:.6g} | {st.max_value:.6g} | **{dim_classification[d]}** |"
        )

    lines.extend(["", "## Per-dimension honest findings", ""])
    for d in DIMENSIONS:
        st = dim_stats[d]
        cls = dim_classification[d]
        lines.append(f"### `{d}` -- {cls}")
        lines.append("")
        if cls == "INSUFFICIENT_DATA":
            lines.append(
                f"- Only {st.n_present} present observations in the window "
                f"(need >= {DECAY_ARTIFACT_MIN_STREAK}). Cannot judge variance or "
                f"decay-artifact risk on this little real data."
            )
        elif cls == "DEGENERATE_FLAT":
            lines.append(
                f"- Real variance is at/below the degenerate threshold "
                f"({DEGENERATE_VARIANCE_EPS:g}): this dimension reads essentially "
                f"the SAME value ({st.mean:.6g}) every tick in the window. No real "
                f"spread exists to precision-weight against."
            )
        else:
            lines.append(
                f"- Real, non-flat variance across {st.n_present} present ticks: "
                f"mean={st.mean:.6g}, stddev={st.stddev:.6g}, range "
                f"[{st.min_value:.6g}, {st.max_value:.6g}]."
            )
        decay = st.decay_artifact
        if decay.suspected:
            match_note = (
                f" -- ratio {decay.ratio_mean:.4f} is within "
                f"{DECAY_ARTIFACT_RATIO_TOLERANCE} of the live "
                f"`BIOMETRICS_FIELD_DECAY_RATE` default ({KNOWN_DECAY_RATE}), "
                f"strongly consistent with unopposed per-tick staleness decay, "
                f"NOT genuine live signal."
                if decay.matches_known_decay_rate
                else f" -- ratio {decay.ratio_mean:.4f} does not closely match "
                f"the known {KNOWN_DECAY_RATE} decay rate, but a near-constant "
                f"ratio over {decay.streak_len} consecutive ticks is still "
                f"suspicious and worth checking by hand against real perturbation "
                f"timestamps for this window."
            )
            lines.append(
                f"- **Decay-artifact streak detected**: {decay.streak_len} "
                f"consecutive strictly-decreasing ticks with near-constant ratio "
                f"(stddev/mean = {(decay.ratio_stddev / decay.ratio_mean):.4f})"
                f"{match_note}"
            )
        else:
            lines.append(
                f"- No decay-artifact streak found (longest near-constant-ratio "
                f"decreasing run: {decay.streak_len} ticks, below the "
                f"{DECAY_ARTIFACT_MIN_STREAK}-tick suspicion threshold)."
            )
        live = st.liveness
        if live.n_upward_transitions == 0:
            lines.append(
                "- **No real upward (perturbation-driven) transition anywhere in the "
                "window.** Every value in this window is decay or a flat hold from "
                "whatever value existed before the window started -- the actual "
                "`node:substrate.route` signature."
            )
        else:
            gap_note = (
                f", average gap between real events {live.mean_inter_event_gap_sec:.0f}s"
                if live.mean_inter_event_gap_sec is not None
                else ""
            )
            lines.append(
                f"- {live.n_upward_transitions} real upward (perturbation-driven) "
                f"transitions in the window{gap_note}; most recent at "
                f"`{live.last_upward_transition_ts}` "
                f"({live.seconds_since_last_upward_transition:.0f}s before window end)."
            )
        lines.append("")

    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")

    n_usable = sum(
        1
        for c in dim_classification.values()
        if c in ("NON_DEGENERATE_USABLE", "ACTIVE_WITH_DECAY_MECHANIC")
    )
    n_dead = sum(1 for c in dim_classification.values() if c == "NO_REAL_EVENTS_IN_WINDOW")
    n_bad = sum(
        1 for c in dim_classification.values() if c in ("DEGENERATE_FLAT", "INSUFFICIENT_DATA")
    )
    lines.extend(
        [
            "## Step 1 verdict",
            "",
            f"- Usable (`NON_DEGENERATE_USABLE` + `ACTIVE_WITH_DECAY_MECHANIC`): "
            f"{n_usable}/{len(DIMENSIONS)}",
            f"- `NO_REAL_EVENTS_IN_WINDOW` (real disqualifier -- the "
            f"node:substrate.route disease): {n_dead}/{len(DIMENSIONS)}",
            f"- `DEGENERATE_FLAT` or `INSUFFICIENT_DATA`: {n_bad}/{len(DIMENSIONS)}",
            "",
            "Per the task scope: Step 2 (the EWMA precision-weighted replacement) "
            "proceeds only for dimensions the real data actually supports. "
            "`ACTIVE_WITH_DECAY_MECHANIC` is NOT automatically excluded -- a real "
            "channel that also decays toward a quiet floor between events is "
            "expected, by-design behavior "
            "(`services/orion-field-digester/app/digestion/decay.py`), not the "
            "node:substrate.route disease. A dimension's real event cadence can be "
            "much sparser than another's (see each dimension's "
            "`mean_inter_event_gap_sec` above) without that alone meaning it is "
            "dead -- the disqualifying test is ZERO real events anywhere in the "
            "window, not merely a comparatively quiet stretch relative to another "
            "dimension's cadence. (An earlier version of this script used a flat "
            "1-hour recency cutoff and mislabeled `reasoning_pressure` -- 13 real "
            "events/16h, ~74min average gap -- as abandoned; corrected after "
            "checking a wider window by hand.)",
            "",
        ]
    )
    lines.extend(
        [
            "## Explicit scope boundary",
            "",
            "This script is a measurement only. It did NOT modify "
            "`orion/proposals/scoring.py`, `orion/field/pressure.py`, "
            "`orion/schemas/field_state.py`, any live scoring/producer code path, "
            "or any database table.",
            "",
        ]
    )
    return "\n".join(lines)


def run(window_hours: float) -> int:
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

    min_ts, max_ts, total_rows, avg_bytes = fetch_field_state_summary(conn)
    progress.emit("table summary fetched", percent=5.0, processed=total_rows, total=total_rows)

    if min_ts is None or max_ts is None or total_rows < MIN_TOTAL_ROWS:
        caveats.append(
            f"insufficient real substrate_field_state history (rows={total_rows}, "
            f"min_required={MIN_TOTAL_ROWS}); cannot run a meaningful probe -- "
            f"STOPPING, not proceeding to any EWMA implementation on assumed data."
        )
        try:
            conn.close()
        except Exception:
            pass
        progress.close()
        report = (
            "# Proposal Scoring `dimension_confidence()` -- Real Historical Variance Probe\n\n"
            "## Data source (Missing Question 2)\n\n"
            "**Durable, queryable history: NO / INSUFFICIENT.**\n\n"
            + "\n".join(f"- {c}" for c in caveats)
            + "\n"
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    table_label = (
        f"{min_ts.isoformat()} -> {max_ts.isoformat()} "
        f"({(max_ts - min_ts).total_seconds() / 3600.0:.1f}h)"
    )

    window_end = max_ts
    window_start = max(min_ts, max_ts - timedelta(hours=window_hours))
    window_label = f"{window_start.isoformat()} -> {window_end.isoformat()} ({window_hours:g}h requested)"

    rows, n_schema_errors, truncated = stream_field_state_rows(
        conn, window_start, window_end, MAX_ROWS, progress
    )
    if truncated:
        caveats.append(
            f"window truncated at MAX_ROWS={MAX_ROWS} rows (oldest kept, newest dropped); "
            f"narrow --window-hours for a non-truncated read"
        )
    if n_schema_errors:
        caveats.append(
            f"{n_schema_errors} row(s) in the window failed FieldStateV1 schema validation "
            f"and were excluded (not counted as present-but-absent, not counted as a reading)"
        )

    try:
        conn.close()
    except Exception:
        pass

    progress.emit("rows reduced", percent=95.0, processed=len(rows), total=len(rows))

    n_window = len(rows)
    if n_window < MIN_TOTAL_ROWS:
        caveats.append(
            f"insufficient real rows in the requested window ({n_window} < {MIN_TOTAL_ROWS}); "
            f"widen --window-hours"
        )

    dim_series: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    dim_timestamps: dict[str, list[datetime]] = {d: [] for d in DIMENSIONS}
    for ts, dims in rows:
        for d in DIMENSIONS:
            v = dims.get(d)
            if v is not None:
                dim_series[d].append(float(v))
                dim_timestamps[d].append(ts)

    dim_stats: dict[str, DimensionStats] = {
        d: compute_dimension_stats(dim_timestamps[d], dim_series[d], n_window, window_end)
        for d in DIMENSIONS
    }
    dim_classification: dict[str, str] = {d: classify_dimension(dim_stats[d]) for d in DIMENSIONS}

    write_ticks_csv(CSV_PATH, rows)
    report = render_report(
        table_summary_label=table_label,
        total_table_rows=total_rows,
        avg_row_bytes=avg_bytes,
        window_label=window_label,
        n_window_rows=n_window,
        n_schema_errors=n_schema_errors,
        truncated=truncated,
        dim_series=dim_series,
        dim_stats=dim_stats,
        dim_classification=dim_classification,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=1, total=1)
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only probe: real historical variance of the four "
        "orion.proposals.scoring.PRESSURE_DIMENSIONS, answering Missing "
        "Questions 1-2 of the precision-weighted-proposal-scoring design doc."
    )
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"Most recent N hours of real substrate_field_state history to "
        f"replay (default: {DEFAULT_WINDOW_HOURS}).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.window_hours)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
