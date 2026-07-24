#!/usr/bin/env python3
"""Read-only Phase 3 shadow-measure comparison: biometrics prediction-error
vs. the halted drives system's bucket-vote output, over real historical data.

`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-
roadmap-design.md` Phase 3: "Route one real producer domain onto real field
channels in observe-only mode, alongside the still-live bucket-vote system,
and compare... whether field-native routing produces a recognizably
different (and better) picture... than the old drive-bucket output did for
the same real ticks." Producer instrumentation for all five domains already
shipped 2026-07-21 (see the charter's own status note, §6 item 3) -- this
script is the comparison itself, the one piece of Phase 3 not yet built.

Biometrics is the domain compared here because it already has a real,
shipped shadow-measure slice (`biometrics_prediction_error()`,
`orion/substrate/prediction_error.py`) and because the OLD system has a
real, named mapping for it: `config/autonomy/signal_drive_map.yaml`'s
`biometrics_state` entry routes deviations into the `capability`
(weight 0.4/0.3) and `continuity` (weight 0.3/0.2) drives via
`SignalTensionSource` -- confirmed live 2026-07-24, this is the ONLY drives
path biometrics feeds; there is no dedicated "biometrics" bucket/tension
kind (`TensionEventV1.kind` for this path is the generic
`"tension.signal.v1"`, `orion/autonomy/signal_tension.py`, not a
biometrics-specific label -- so `dominant_drive`/per-drive `drive_pressures`
is the only fair comparison axis, not `tension_kinds`).

**No reliance on SelfStateV1, either side, confirmed live 2026-07-24:**
`biometrics_prediction_error()` has zero self_state references
(`orion/substrate/prediction_error.py`); `DriveEngine`/bucket-voting's real
tension source (`SignalTensionSource` + `signal_drive_map.yaml`) is
explicitly documented as untouched by, and independent of, the 2026-07-22
SelfStateV1 burn (`orion/spark/concept_induction/bus_worker.py:158-169`'s
own comment).

Data sources (both confirmed live 2026-07-24 to hold real, dense history
since the 2026-07-23 Postgres rebuild):

- `substrate_field_state` (`field_json.node_vectors["node:substrate.
  biometrics"].prediction_error`) -- the new, field-native signal.
  32,454/32,481 rows in an ~18h window carried this node.
- `drive_audits` (`drive_pressures`, a per-event JSON snapshot of ALL drive
  pressures, plus `dominant_drive`) -- the old bucket-vote signal. 42,755
  rows in the same window. This is a real per-event artifact
  (`build_drive_audit()`, `orion/spark/concept_induction/audit.py`),
  NOT the local-JSON current-snapshot-only `LocalProfileStore.save_drive_
  state()` (which has no history at all -- checked and ruled out before
  building this).

This performs NO writes, emits NO events, flips NO flags, changes NO
consumer. It reports two real, falsifiable numbers, not a vibes-based
"which picture is nicer":

  1. Pearson correlation between `biometrics_prediction_error` and
     max(capability_pressure, continuity_pressure) at matched real moments
     -- are these actually related signals at all.
  2. Split comparison: mean `biometrics_prediction_error` at moments the OLD
     system's `dominant_drive` was capability/continuity, vs. moments it
     was some other drive -- does the new signal spike specifically when
     the old system attributes pressure to the two drives biometrics feeds.

Explicitly NOT decided here: whether field-native routing is "better" in
any normative sense, whether to migrate biometrics live (Phase 4), or
anything about the bucket-vote layer's retirement (Phase 5). This is
Phase 3's own scope: prove related-or-not, on real data, before Phase 4
decides to act on it.

Run:
    python scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py --window-hours 18
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.phase3_biometrics_drive_shadow_comparison")

DEFAULT_WINDOW_HOURS: float = 18.0
MAX_ROWS: int = 200_000

# How stale a field_state row may be and still count as "concurrent" with a
# drive_audit event. field_state ticks at ~2s cadence (confirmed live
# 2026-07-18/23) -- 30s is a generous multiple of that, not a tight bound,
# chosen so a genuine gap in field_state coverage is reported as "no
# concurrent data" rather than silently joined to a stale, unrelated reading.
MAX_JOIN_STALENESS_SEC: float = 30.0

# Below this many paired observations, the isolated-subset correlation delta
# is not given a directional reading in the report -- a small sample can
# swing a correlation coefficient wildly, and a confident-sounding verdict
# off a handful of points would be exactly the kind of unearned certainty
# CLAUDE.md's metric-quality-gate discipline exists to prevent. Not
# calibrated against a formal power analysis -- a conservative round number,
# same spirit as this script's other starting-anchor constants.
MIN_ISOLATED_N_FOR_INTERPRETATION: int = 30

BIOMETRICS_NODE_ID = "node:substrate.biometrics"

# The only two drives `biometrics_state` deviations route into
# (`config/autonomy/signal_drive_map.yaml`), confirmed live 2026-07-24.
BIOMETRICS_FED_DRIVES: frozenset[str] = frozenset({"capability", "continuity"})

OUTPUT_DIR = Path("/tmp/phase3-biometrics-drive-shadow-comparison")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# The exact `tension_kinds` singleton used to PARTIALLY isolate a drive-audit
# event's capability/continuity impulse -- the disentangling filter named in
# `docs/notes/2026-07-24-phase3-biometrics-drive-shadow-comparison-finding.md`'s
# "recommended next step". Traced every real kind that can carry a
# `capability`/`continuity` drive_impact (`orion/spark/concept_induction/
# tensions.py`, `orion/autonomy/signal_tension.py`,
# `config/autonomy/signal_drive_map.yaml`):
#   - "tension.failure.v1"       (failure_event -> capability 0.5)
#   - "tension.chat_evidence.v1" (chat_social_hazard -> capability 0.2/0.3)
#   - "tension.cognitive_load.v1" (turn_effect energy delta -> capability 0.9)
#   - "tension.distress.v1"      (turn_effect valence delta OR action-outcome
#     -> continuity 0.55)
#   - "tension.identity_drift.v1" (turn_effect novelty delta -> continuity 0.8)
#   - "tension.satisfaction.v1"  (action-outcome, dynamic per-outcome weights
#     that can include continuity -- conservatively treated as polluting)
# "tension.drive_competition.v1" carries EMPTY drive_impacts
# (`derive_pressure_competition_tensions`, a pure competition marker) and
# "tension.contradiction.v1" only touches coherence/predictive -- neither
# pollutes capability/continuity, so their co-presence does not disqualify
# a tick.
#
# **KNOWN, UNRESOLVED GAP (found in review 2026-07-24): this filter does NOT
# fully isolate biometrics.** `mesh_health` deviations do NOT produce
# `"tension.health.v1"` in practice -- `SignalTensionSource.from_equilibrium`
# (the only code that would emit that kind) has zero live callers, confirmed
# dead code. The real live path (`orion/signals/adapters/equilibrium.py` ->
# `"orion:signals:equilibrium"` channel -> `bus_worker.py`'s generic "signal"
# rail -> `signal_to_tension()`) emits the SAME generic `"tension.signal.v1"`
# kind biometrics_state uses (`orion/autonomy/signal_tension.py`) -- so a
# drive-audit event whose `tension_kinds == {"tension.signal.v1"}` may be
# biometrics-sourced, mesh_health-sourced, or both, and `drive_audits` (no
# `signal_kind`/`evidence_text`/`related_nodes` column, only a generic
# `summary` like "pressure concentrates on capability") cannot disambiguate
# them after the fact. The `"orion:equilibrium:snapshot"` channel that could
# independently confirm mesh_health's real firing rate is not durably logged
# to Postgres (consumer is `orion-cortex-orch` only) -- checked and ruled
# out 2026-07-24, no way to bound the contamination rate with real data.
# **Net effect: `BIOMETRICS_ISOLATED_TENSION_KINDS` correctly excludes chat-
# hazard/turn-effect/action-outcome pollution, but does NOT prove the
# resulting subset is biometrics-only.** Any correlation computed on this
# subset should be read as "capability/continuity movement attributable to
# either biometrics_state or mesh_health, nothing else" -- weaker evidence
# than a clean biometrics-only isolation, not a false result, but not what
# an earlier version of this docstring claimed either.
BIOMETRICS_ISOLATED_TENSION_KINDS: frozenset[str] = frozenset({"tension.signal.v1"})

# Every OTHER real capability/continuity-touching kind -- confirmed via the
# same trace above. A tick is "isolated" only if its tension_kinds is
# EXACTLY BIOMETRICS_ISOLATED_TENSION_KINDS (a strict subset match would
# still let e.g. tension.distress.v1 slip in alongside tension.signal.v1).
# Deliberately excludes "tension.health.v1" -- confirmed dead code above, so
# it is not a real co-occurrence risk in live data (unlike the constant name
# might suggest to a future reader who doesn't also read the caveat above).
_KNOWN_POLLUTING_TENSION_KINDS: frozenset[str] = frozenset(
    {
        "tension.failure.v1", "tension.chat_evidence.v1",
        "tension.cognitive_load.v1", "tension.distress.v1",
        "tension.identity_drift.v1", "tension.satisfaction.v1",
    }
)

assert not (BIOMETRICS_ISOLATED_TENSION_KINDS & _KNOWN_POLLUTING_TENSION_KINDS), (
    "BIOMETRICS_ISOLATED_TENSION_KINDS and _KNOWN_POLLUTING_TENSION_KINDS must "
    "not overlap -- a kind cannot simultaneously define the isolated set and "
    "disqualify membership in it."
)


# ===========================================================================
# Pure layer -- no I/O. Deterministic, unit-testable without a DB.
# ===========================================================================


@dataclass
class AlignedTick:
    observed_at: datetime
    biometrics_prediction_error: Optional[float]
    capability_pressure: Optional[float]
    continuity_pressure: Optional[float]
    dominant_drive: Optional[str]
    field_state_age_sec: Optional[float]
    is_biometrics_isolated: bool = False


def is_biometrics_isolated_event(tension_kinds: object) -> bool:
    """True iff `tension_kinds` (the raw `drive_audits.tension_kinds` jsonb
    value for one event) is EXACTLY `BIOMETRICS_ISOLATED_TENSION_KINDS` --
    i.e. this event's capability/continuity movement, if any, can only have
    come from biometrics_state, with no other real drive_impacts-bearing
    kind folded into the same tick. Malformed/non-list input is honestly
    `False` (not isolated), never assumed.
    """
    if not isinstance(tension_kinds, list):
        return False
    return set(tension_kinds) == BIOMETRICS_ISOLATED_TENSION_KINDS


def extract_biometrics_prediction_error(field_state_payload: dict) -> Optional[float]:
    """Pull the raw `prediction_error` value for the biometrics node out of
    one `FieldStateV1.node_vectors` payload. Missing, not defaulted to 0.0,
    so an absent tick doesn't masquerade as "confirmed calm."
    """
    node_vectors = field_state_payload.get("node_vectors") or {}
    vector = node_vectors.get(BIOMETRICS_NODE_ID)
    if not isinstance(vector, dict) or "prediction_error" not in vector:
        return None
    try:
        return float(vector["prediction_error"])
    except (TypeError, ValueError):
        return None


def align_drive_audits_to_field_state(
    drive_audit_rows: list[tuple[datetime, dict]],
    field_state_rows: list[tuple[datetime, dict]],
) -> list[AlignedTick]:
    """Two-pointer nearest-preceding (as-of) join, driven by `drive_audit_
    rows` cadence -- one output row per real drive-audit event, since that
    is the comparison's unit of analysis, regardless of which side happens
    to be denser (confirmed live 2026-07-24: `drive_audits` is actually the
    DENSER side in real data, 41,919 rows vs. 31,793 `substrate_field_state`
    rows in the same 18h window -- the opposite density relationship from
    `measure_ast_hot_reducer.py`'s `replay_reducer`, which drives its own
    loop off the densest signal instead). Same two-pointer mechanics as that
    script either way: both inputs are ASC-sorted, `fs_idx` only advances,
    so no row is skipped. For each drive-audit event, find the latest
    `field_state` row at or before that timestamp, within
    `MAX_JOIN_STALENESS_SEC`; beyond that, `biometrics_prediction_error`
    (and its age) is honestly `None` rather than joined to a stale reading.
    Because `drive_audits` is denser here, many consecutive drive-audit
    events legitimately share the same nearest-preceding field_state
    reading -- not double-counting, correct as-of-join semantics, but see
    `compute_correlation`'s docstring for what this means for the paired
    sample size.

    Both input lists must be ASC-sorted by timestamp (as returned by the
    fetch functions' `ORDER BY ... ASC`).
    """
    field_states: list[tuple[datetime, Optional[float]]] = [
        (ts, extract_biometrics_prediction_error(payload)) for ts, payload in field_state_rows
    ]

    ticks: list[AlignedTick] = []
    fs_idx = 0
    current_pe: Optional[float] = None
    current_pe_ts: Optional[datetime] = None

    for ts, payload in drive_audit_rows:
        while fs_idx < len(field_states) and field_states[fs_idx][0] <= ts:
            current_pe = field_states[fs_idx][1]
            current_pe_ts = field_states[fs_idx][0]
            fs_idx += 1

        age_sec: Optional[float] = None
        pe_value = current_pe
        if current_pe_ts is not None:
            age_sec = (ts - current_pe_ts).total_seconds()
            if age_sec > MAX_JOIN_STALENESS_SEC:
                pe_value = None

        pressures = payload.get("drive_pressures") or {}
        cap = pressures.get("capability")
        cont = pressures.get("continuity")
        ticks.append(
            AlignedTick(
                observed_at=ts,
                biometrics_prediction_error=pe_value,
                capability_pressure=float(cap) if isinstance(cap, (int, float)) else None,
                continuity_pressure=float(cont) if isinstance(cont, (int, float)) else None,
                dominant_drive=payload.get("dominant_drive"),
                field_state_age_sec=age_sec if pe_value is not None else None,
                is_biometrics_isolated=is_biometrics_isolated_event(payload.get("tension_kinds")),
            )
        )
    return ticks


def pearson_correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    """Standard Pearson correlation coefficient. `None` if fewer than 2
    paired points, or either series has zero variance (undefined, not 0.0 --
    a flat series isn't "uncorrelated," it's degenerate).
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def compute_correlation(ticks: list[AlignedTick]) -> tuple[Optional[float], int]:
    """Correlation between `biometrics_prediction_error` and
    max(capability_pressure, continuity_pressure) across ticks where all
    three values are present. Returns (coefficient_or_none, n_paired).

    **`n_paired` counts drive-audit events, not independent readings.**
    Because `drive_audits` is the denser side in real data (see
    `align_drive_audits_to_field_state`'s docstring), many consecutive
    events legitimately share the same nearest-preceding
    `biometrics_prediction_error` value -- the true number of distinct `x`
    values is bounded above by the `field_state` row count, not `n_paired`.
    The Pearson formula itself is computed correctly on the data as given;
    this only means `n_paired` overstates independent statistical power at
    a glance and should not be read as "N truly independent draws."
    """
    xs: list[float] = []
    ys: list[float] = []
    for t in ticks:
        if (
            t.biometrics_prediction_error is None
            or t.capability_pressure is None
            or t.continuity_pressure is None
        ):
            continue
        xs.append(t.biometrics_prediction_error)
        ys.append(max(t.capability_pressure, t.continuity_pressure))
    return pearson_correlation(xs, ys), len(xs)


def filter_isolated(ticks: list[AlignedTick]) -> list[AlignedTick]:
    """The disentangling subset named in `docs/notes/2026-07-24-phase3-
    biometrics-drive-shadow-comparison-finding.md`'s recommended next step:
    ticks where `is_biometrics_isolated` is True, i.e. this event's
    capability/continuity movement (if any) cannot have come from
    `mesh_health`/`failure_event`/turn-effect tensions/etc. -- only
    `biometrics_state`, or nothing. Re-running `compute_correlation` on
    this subset answers the open question: does correlation improve once
    the OTHER real contributors to the same two drives are excluded, or
    does it stay near zero even isolated.
    """
    return [t for t in ticks if t.is_biometrics_isolated]


@dataclass
class GroupStats:
    n: int
    mean: Optional[float]


def split_comparison(ticks: list[AlignedTick]) -> tuple[GroupStats, GroupStats]:
    """Mean `biometrics_prediction_error` when the OLD system's
    `dominant_drive` is one of `BIOMETRICS_FED_DRIVES` vs. when it's some
    other drive -- only over ticks where both `dominant_drive` and
    `biometrics_prediction_error` are present. Returns
    (fed_drive_group, other_drive_group).
    """
    fed_values: list[float] = []
    other_values: list[float] = []
    for t in ticks:
        if t.dominant_drive is None or t.biometrics_prediction_error is None:
            continue
        if t.dominant_drive in BIOMETRICS_FED_DRIVES:
            fed_values.append(t.biometrics_prediction_error)
        else:
            other_values.append(t.biometrics_prediction_error)

    def _stats(values: list[float]) -> GroupStats:
        return GroupStats(n=len(values), mean=(sum(values) / len(values)) if values else None)

    return _stats(fed_values), _stats(other_values)


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


def _rows_to_payload_list(rows, *, ts_col_idx: int = 1, payload_col_idx: int = 0) -> tuple[list[tuple[datetime, dict]], bool]:
    out: list[tuple[datetime, dict]] = []
    for row in rows:
        payload, ts = row[payload_col_idx], row[ts_col_idx]
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
        out.append((ts, payload))
    return out, len(rows) >= MAX_ROWS


def fetch_field_state_rows(conn, since: datetime) -> tuple[list[tuple[datetime, dict]], bool]:
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT field_json, generated_at
                FROM substrate_field_state
                WHERE generated_at >= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_field_state", exc_info=True)
        return [], False
    return _rows_to_payload_list(rows)


def fetch_drive_audit_rows(conn, since: datetime) -> tuple[list[tuple[datetime, dict]], bool]:
    """Real per-event drive-pressure snapshots, NOT the current-snapshot-only
    `LocalProfileStore` (checked and ruled out -- that store overwrites in
    place with no history). `observed_at` falls back to `created_at` when
    null (confirmed both columns exist; `observed_at` is the semantic event
    time when present).
    """
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    jsonb_build_object(
                        'drive_pressures', drive_pressures,
                        'dominant_drive', dominant_drive,
                        'active_drives', active_drives,
                        'tension_kinds', tension_kinds
                    ),
                    COALESCE(observed_at, created_at)
                FROM drive_audits
                WHERE COALESCE(observed_at, created_at) >= %s
                ORDER BY COALESCE(observed_at, created_at) ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch drive_audits", exc_info=True)
        return [], False
    return _rows_to_payload_list(rows)


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


def write_ticks_csv(path: Path, ticks: list[AlignedTick]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "observed_at", "biometrics_prediction_error", "capability_pressure",
                "continuity_pressure", "dominant_drive", "field_state_age_sec",
                "is_biometrics_isolated",
            ]
        )
        for t in ticks:
            writer.writerow(
                [
                    t.observed_at.isoformat(),
                    "" if t.biometrics_prediction_error is None else f"{t.biometrics_prediction_error:.4f}",
                    "" if t.capability_pressure is None else f"{t.capability_pressure:.4f}",
                    "" if t.continuity_pressure is None else f"{t.continuity_pressure:.4f}",
                    t.dominant_drive or "",
                    "" if t.field_state_age_sec is None else f"{t.field_state_age_sec:.2f}",
                    t.is_biometrics_isolated,
                ]
            )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    ticks: list[AlignedTick],
    field_state_rows_count: int,
    drive_audit_rows_count: int,
    field_state_truncated: bool,
    drive_audit_truncated: bool,
    correlation: Optional[float],
    correlation_n: int,
    fed_group: GroupStats,
    other_group: GroupStats,
    isolated_correlation: Optional[float],
    isolated_correlation_n: int,
    isolated_ticks_count: int,
    caveats: list[str],
) -> str:
    n = len(ticks)
    n_with_pe = sum(1 for t in ticks if t.biometrics_prediction_error is not None)

    pe_values = [t.biometrics_prediction_error for t in ticks if t.biometrics_prediction_error is not None]
    pe_degenerate = (
        len(pe_values) > 0 and (max(pe_values) - min(pe_values)) < 1e-6
    )
    pressure_values = [
        v for t in ticks for v in (t.capability_pressure, t.continuity_pressure) if v is not None
    ]
    pressure_degenerate = (
        len(pressure_values) > 0 and (max(pressure_values) - min(pressure_values)) < 1e-6
    )

    lines = [
        "# Phase 3 shadow-measure: biometrics prediction-error vs. drives bucket-vote",
        "",
        "Read-only. No writes, no events, no flag/config changes. Joins real "
        "`substrate_field_state` (`node:substrate.biometrics.prediction_error`, the new "
        "field-native signal) to real `drive_audits` (`drive_pressures`, the old "
        "bucket-vote signal) by nearest-preceding timestamp, at each real drive-audit "
        "event.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- `drive_audits` rows in window: {drive_audit_rows_count}"
        + (" (truncated at MAX_ROWS)" if drive_audit_truncated else ""),
        f"- `substrate_field_state` rows in window: {field_state_rows_count}"
        + (" (truncated at MAX_ROWS)" if field_state_truncated else ""),
        f"- Aligned ticks (one per drive-audit event): {n}",
        f"- Aligned ticks with a concurrent (within {MAX_JOIN_STALENESS_SEC:.0f}s) "
        f"biometrics reading: {n_with_pe} / {n} "
        f"({'n/a' if n == 0 else f'{100.0 * n_with_pe / n:.1f}%'})",
        "",
        "## Live-data sanity check (CLAUDE.md metric-quality-gate)",
        "",
        f"- biometrics_prediction_error: n={len(pe_values)}, "
        f"min={_fmt(min(pe_values) if pe_values else None)}, "
        f"max={_fmt(max(pe_values) if pe_values else None)}"
        + ("  **DEGENERATE (flat within 1e-6)**" if pe_degenerate else "  -- real variance"),
        f"- capability/continuity pressure (pooled): n={len(pressure_values)}, "
        f"min={_fmt(min(pressure_values) if pressure_values else None)}, "
        f"max={_fmt(max(pressure_values) if pressure_values else None)}"
        + ("  **DEGENERATE (flat within 1e-6)**" if pressure_degenerate else "  -- real variance"),
        "",
        "## 1. Correlation: biometrics_prediction_error vs. max(capability, continuity) pressure",
        "",
        f"- n paired observations: {correlation_n}",
        f"- Pearson r: {_fmt(correlation)}"
        + ("" if correlation is not None else " (undefined -- too few points or zero variance in one series)"),
        f"- Note: `n` counts drive-audit events, not independent readings -- `drive_audits` "
        f"is the denser side in real data, so many consecutive events legitimately share "
        f"the same nearest-preceding biometrics reading (true distinct-x count is bounded "
        f"above by the `substrate_field_state` row count, {field_state_rows_count}, not "
        f"`n` above). The formula itself is correct on the data as given; this just means "
        f"`n` overstates independent statistical power at a glance.",
        "",
        "## 2. Split comparison: does the new signal spike when the OLD system blames "
        "capability/continuity?",
        "",
        f"- Ticks where `dominant_drive` in {{capability, continuity}}: n={fed_group.n}, "
        f"mean biometrics_prediction_error={_fmt(fed_group.mean)}",
        f"- Ticks where `dominant_drive` is some OTHER drive: n={other_group.n}, "
        f"mean biometrics_prediction_error={_fmt(other_group.mean)}",
        "",
    ]

    if fed_group.mean is not None and other_group.mean is not None:
        diff = fed_group.mean - other_group.mean
        lines.append(
            f"- Difference (fed-drive mean - other-drive mean): {diff:+.4f} -- "
            + (
                "positive means the new signal reads higher precisely when the old "
                "system attributes pressure to the drives biometrics feeds (the "
                "expected direction if these are the same real signal)."
                if diff > 0 else
                "non-positive -- the new signal does NOT read higher when the old "
                "system attributes pressure to capability/continuity; this does not "
                "confirm the two are measuring the same real phenomenon."
            )
        )
    else:
        lines.append(
            "- Insufficient data in one or both groups to compute a difference."
        )

    lines.extend(
        [
            "",
            "## 3. Isolated comparison: correlation restricted to biometrics-or-mesh_health-"
            "only events",
            "",
            "The disentangling step named in `docs/notes/2026-07-24-phase3-biometrics-"
            "drive-shadow-comparison-finding.md`'s recommended next step: repeat the "
            "correlation, restricted to `drive_audits` events whose `tension_kinds` is "
            "EXACTLY `{\"tension.signal.v1\"}` -- excludes `failure_event`/chat-hazard/"
            "turn-effect/action-outcome pollution. **Does NOT prove biometrics-only**: "
            "`mesh_health` deviations also emit this same generic kind (confirmed in "
            "review 2026-07-24 -- see `BIOMETRICS_ISOLATED_TENSION_KINDS`'s docstring for "
            "the full trace and why this can't be tightened further with data this script "
            "has access to). Read this subset as \"capability/continuity movement "
            "attributable to biometrics_state or mesh_health, nothing else\" -- narrower "
            "evidence than a clean biometrics-only isolation would give.",
            "",
            f"- Isolated ticks in window: {isolated_ticks_count} / {n}",
            f"- n paired observations (isolated subset): {isolated_correlation_n}",
            f"- Pearson r (isolated subset): {_fmt(isolated_correlation)}"
            + ("" if isolated_correlation is not None else " (undefined -- too few points or zero variance)"),
        ]
    )
    if isolated_correlation_n < MIN_ISOLATED_N_FOR_INTERPRETATION:
        lines.append(
            f"- Only {isolated_correlation_n} paired observation(s) in the isolated subset "
            f"(below MIN_ISOLATED_N_FOR_INTERPRETATION={MIN_ISOLATED_N_FOR_INTERPRETATION}) "
            f"-- no directional reading given. A correlation coefficient computed on this "
            f"few points can swing wildly and would be a false confidence, not a real "
            f"comparison."
        )
    elif correlation is not None and isolated_correlation is not None:
        delta = isolated_correlation - correlation
        lines.append(
            f"- Change vs. full-dataset correlation ({_fmt(correlation)}): {delta:+.4f} -- "
            + (
                "correlation improved once other real contributors were excluded -- "
                "weakly supports the old bucket-dilution explanation over \"genuinely "
                "unrelated\" (weakly, since this subset still can't rule out mesh_health "
                "specifically -- see the caveat above)."
                if delta > 0.05 else
                "correlation did NOT meaningfully improve when isolated -- does not support "
                "the bucket-dilution explanation as strongly as a clean isolation would; "
                "still worth a second look at biometrics_prediction_error's own formula "
                "before trusting it as this domain's field-native replacement, though "
                "residual mesh_health contamination in this subset means that conclusion "
                "isn't airtight either."
                if abs(delta) <= 0.05 else
                "correlation got WORSE when isolated -- unexpected, worth independent "
                "re-checking before drawing a conclusion either way."
            )
        )
    else:
        lines.append(
            "- Cannot compare: one or both correlations are undefined (insufficient data)."
        )

    lines.extend(
        [
            "",
            "## Reading this",
            "",
            "- This measures whether the two signals are RELATED on real data -- it does "
            "not decide whether field-native routing is normatively \"better,\" and does "
            "not itself authorize any live migration (Phase 4) or bucket-vote retirement "
            "(Phase 5).",
            "- `dominant_drive` is a single winner-take-all label per event, not a "
            "proportional attribution -- a real capability/continuity contribution can "
            "be present even when a different drive technically dominates that tick. A "
            "weak or absent split difference does not by itself rule out a real "
            "relationship; the correlation number above is the more sensitive of the two.",
            "- `tension_kinds` is used ONLY for the isolated-subset filter above (section "
            "3), not as the primary split-comparison axis (section 2) -- biometrics-"
            "derived tensions share the generic `\"tension.signal.v1\"` kind with no "
            "other real signal_kind that touches capability/continuity, confirmed live "
            "2026-07-24, which is exactly what makes it usable as an isolation filter "
            "even though it can't distinguish biometrics from itself as a raw label.",
            "- **Real confound, not a bug**: `capability` is not a clean biometrics proxy "
            "on the old side either -- `signal_drive_map.yaml` also routes `mesh_health` "
            "(weight 0.5) and `failure_event` (weight 0.5) into the same drive. A weak or "
            "near-zero correlation/split-difference here can mean the two signals are "
            "genuinely unrelated, OR it can mean the old bucket blends biometrics pressure "
            "with mesh/failure pressure so thoroughly that no biometrics-specific baseline "
            "survives to compare against -- which would itself be a concrete illustration "
            "of the many-to-one bucket-vote problem this whole roadmap exists to fix, not "
            "evidence against the new signal. This measurement cannot distinguish those two "
            "explanations on its own; it reports the numbers, not a verdict.",
            "",
            "## Coverage caveats",
            "",
        ]
    )
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

    drive_audit_rows, drive_audit_truncated = fetch_drive_audit_rows(conn, window_start)
    if drive_audit_truncated:
        caveats.append(f"drive_audits rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("drive_audits loaded", percent=30.0, processed=len(drive_audit_rows), total=len(drive_audit_rows))

    field_state_rows, field_state_truncated = fetch_field_state_rows(conn, window_start)
    if field_state_truncated:
        caveats.append(f"field-state rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("field_state loaded", percent=60.0, processed=len(field_state_rows), total=len(field_state_rows))

    try:
        conn.close()
    except Exception:
        pass

    if not drive_audit_rows:
        caveats.append("no drive_audits rows in window; nothing to compare")
        progress.close()
        report = render_report(
            window_label=window_label, window_start=window_start, window_end=now,
            ticks=[], field_state_rows_count=len(field_state_rows),
            drive_audit_rows_count=0, field_state_truncated=field_state_truncated,
            drive_audit_truncated=drive_audit_truncated, correlation=None,
            correlation_n=0, fed_group=GroupStats(0, None), other_group=GroupStats(0, None),
            isolated_correlation=None, isolated_correlation_n=0, isolated_ticks_count=0,
            caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    progress.emit("aligning", percent=70.0, processed=0, total=len(drive_audit_rows))
    ticks = align_drive_audits_to_field_state(drive_audit_rows, field_state_rows)
    progress.emit("aligned", percent=90.0, processed=len(ticks), total=len(drive_audit_rows))

    correlation, correlation_n = compute_correlation(ticks)
    fed_group, other_group = split_comparison(ticks)
    isolated_ticks = filter_isolated(ticks)
    isolated_correlation, isolated_correlation_n = compute_correlation(isolated_ticks)

    write_ticks_csv(CSV_PATH, ticks)
    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        ticks=ticks,
        field_state_rows_count=len(field_state_rows),
        drive_audit_rows_count=len(drive_audit_rows),
        field_state_truncated=field_state_truncated,
        drive_audit_truncated=drive_audit_truncated,
        correlation=correlation,
        correlation_n=correlation_n,
        fed_group=fed_group,
        other_group=other_group,
        isolated_correlation=isolated_correlation,
        isolated_correlation_n=isolated_correlation_n,
        isolated_ticks_count=len(isolated_ticks),
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=len(ticks), total=len(drive_audit_rows))
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Phase 3 shadow-measure comparison: biometrics prediction-error vs. drives bucket-vote."
    )
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
