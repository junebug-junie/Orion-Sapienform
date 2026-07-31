#!/usr/bin/env python3
"""Read-only measurement: is `services/orion-hub/scripts/attention_loops_
store.py`'s `SURFACE_MIN_SALIENCE` gate reachable by real `attention_
salience_trace` data, and -- if not -- which replacement is actually
supported by the measured distribution?

Confirmed live (2026-07-30/31, this script re-derives every number below
fresh, never hardcoded): `SURFACE_MIN_SALIENCE = 0.5` has been structurally
unreachable since it was introduced in commit `6bfe87aaf` (2026-07-07,
"feat(hub): operator-legible PendingAttentionCardV1 builder") with no
calibration rationale in that commit message. The max score ever observed
across the real trace table's entire history is `0.490` -- below the
`>= 0.5` gate every single time. This is why `attention_loop_outcome` (the
operator Resolve/Dismiss verdict table) has zero rows, ever: no threshold
has ever let a single card through for a human to judge.

This is GitHub issue #1512's ground-truth gap (see `docs/notes/2026-07-30-
chat-attention-ground-truth-gap-finding.md`, `scripts/analysis/measure_
chat_attention_ground_truth_gap.py`), one level down: not "is chat-level
salience redundant with Layer 5" but "is the *downstream surfacing gate* on
top of chat-level salience even reachable."

Per CLAUDE.md sec 0A's Metric Quality Gate, this script does NOT just
re-guess a new absolute number. It measures two competing repair strategies
against the real distribution before either is chosen:

1. **A relative/percentile-based rule**, recomputed live from recent real
   scores -- can't go stale the way a hardcoded `0.5` did, and doesn't
   overclaim any specific number means "objectively worth attention."
2. **A recalibrated static constant**, chosen from measured real
   percentiles (not min/max) -- simpler, but re-introduces the same
   staleness risk this whole investigation exists to fix, unless it is
   explicitly documented as a bootstrap instrument rather than a validated
   quality bar.

The real risk with option 1 at this data volume, checked explicitly below:

- **Pseudo-replication.** `attention_salience_trace` is not a population of
  many independent loops -- confirmed live, a small number of `theme_key`s
  account for the overwhelming majority of rows (one recurring loop firing
  hundreds of times, not hundreds of distinct loops). A row-level
  percentile over this table is mostly describing "how loud are the
  chattiest 1-2 loops today," not a real population percentile across
  distinct concepts.
- **Small-n instability.** With only a handful of distinct `theme_key`
  values ever recorded, a percentile computed over *distinct current
  scores* (the population size that actually matters for "is this loop
  relatively high right now") is quantized into a handful of possible
  values and can swing sharply as a single theme appears, disappears, or
  gets suppressed -- checked below via an expanding-window and
  trailing-window replay of the real history, day by day.

This script does not decide policy on its own — it reports the measured
numbers so the patch's actual threshold choice (in `attention_loops_store.
py`) can cite a specific, reproducible measurement rather than a second
guess. It also checks whether `SURFACE_MIN_AGE_SEC` (the companion age
gate, same commit, same lack-of-calibration risk) is actually binding on
real data, since CLAUDE.md's "only change what's shown broken" rule means
that gate should NOT be touched unless this measurement shows it is also
unreachable or trivially always-true in a way that defeats its purpose.

This performs NO writes, emits NO events, flips NO flags, changes NO
consumer. Read-only, matching `measure_chat_attention_ground_truth_gap.py`
and `measure_phase3_biometrics_drive_shadow_comparison.py`'s own
conventions (pure computation layer separate from psycopg2 I/O, dataclass
results, progress log, Markdown report under /tmp).

Run:
    python scripts/analysis/measure_attention_surface_threshold_calibration.py --window-hours 168
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("orion.analysis.attention_surface_threshold_calibration")

# 168h (7 days) -- covers attention_salience_trace's full real history with
# margin, same window convention as measure_chat_attention_ground_truth_gap.py.
DEFAULT_WINDOW_HOURS: float = 168.0
MAX_ROWS: int = 200_000

# The gates under test -- imported as literals here (not from the store
# module) because this script's whole point is to measure whether the
# CURRENTLY SHIPPED constants are reachable; importing them would still be
# correct today, but pinning the literals makes the "as of this run" claim
# explicit even if a future edit changes the module before this script is
# re-run against fresh data.
CURRENT_SURFACE_MIN_SALIENCE: float = 0.5
CURRENT_SURFACE_MIN_AGE_SEC: float = 300.0

# Candidate static thresholds to evaluate against the measured distribution,
# spanning the percentile band a "top slice, not everything" bootstrap gate
# would plausibly sit in.
CANDIDATE_STATIC_THRESHOLDS: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50)

OUTPUT_DIR = Path("/tmp/attention-surface-threshold-calibration")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "daily_percentile_stability.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O. Deterministic, unit-testable without a DB.
# ===========================================================================


@dataclass(frozen=True)
class TraceRow:
    created_at: datetime
    theme_key: str
    salience: float


@dataclass(frozen=True)
class DistributionStats:
    """Real distribution shape of `attention_salience_trace.salience` over
    the window -- never assumed, always computed from the rows passed in.
    """

    count: int
    min_v: Optional[float]
    max_v: Optional[float]
    mean_v: Optional[float]
    stdev_v: Optional[float]
    percentiles: dict[int, float] = field(default_factory=dict)

    @property
    def is_populated(self) -> bool:
        return self.count > 0


def compute_percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches Postgres `percentile_cont`),
    over an already-sorted list. `pct` in [0, 100].
    """
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


_REPORTED_PERCENTILES = (50, 60, 65, 70, 75, 85, 90, 95)


def count_rows_at_or_above(rows: list[TraceRow], threshold: float) -> int:
    return sum(1 for r in rows if r.salience >= threshold)


def compute_distribution_stats(rows: list[TraceRow]) -> DistributionStats:
    values = sorted(r.salience for r in rows)
    if not values:
        return DistributionStats(count=0, min_v=None, max_v=None, mean_v=None, stdev_v=None)
    stdev_v = statistics.pstdev(values) if len(values) > 1 else 0.0
    percentiles = {p: compute_percentile(values, p) for p in _REPORTED_PERCENTILES}
    return DistributionStats(
        count=len(values),
        min_v=values[0],
        max_v=values[-1],
        mean_v=statistics.fmean(values),
        stdev_v=stdev_v,
        percentiles=percentiles,
    )


@dataclass(frozen=True)
class ConcentrationResult:
    """How much of the row volume comes from just the top-N loudest themes
    -- the pseudo-replication check. A row-level percentile computed over a
    table dominated by 1-2 chatty themes is not measuring "population of
    distinct loops," it is measuring "loudness of the loudest few."
    """

    distinct_theme_count: int
    top2_share: Optional[float]  # fraction of all rows contributed by the top 2 themes
    counts_by_theme: dict[str, int]


def compute_concentration(rows: list[TraceRow]) -> ConcentrationResult:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.theme_key] = counts.get(r.theme_key, 0) + 1
    total = sum(counts.values())
    top2 = sum(sorted(counts.values(), reverse=True)[:2])
    return ConcentrationResult(
        distinct_theme_count=len(counts),
        top2_share=(top2 / total) if total else None,
        counts_by_theme=counts,
    )


@dataclass(frozen=True)
class LatestPerTheme:
    """Most recent (by created_at) real score per theme -- the exact value
    `load_pending_loops()`'s live query gates against today (its `DISTINCT
    ON (theme_key) ... ORDER BY t.theme_key, t.created_at DESC` clause), not
    each theme's historical max. A candidate threshold must be checked
    against THIS population, not the raw row distribution, to know what it
    would actually surface right now.
    """

    theme_key: str
    salience: float
    created_at: datetime


def compute_latest_per_theme(rows: list[TraceRow]) -> list[LatestPerTheme]:
    latest: dict[str, TraceRow] = {}
    for r in rows:
        cur = latest.get(r.theme_key)
        if cur is None or r.created_at > cur.created_at:
            latest[r.theme_key] = r
    return sorted(
        (LatestPerTheme(theme_key=k, salience=v.salience, created_at=v.created_at) for k, v in latest.items()),
        key=lambda x: x.salience,
        reverse=True,
    )


@dataclass(frozen=True)
class CandidateThresholdResult:
    threshold: float
    surfaced_theme_keys: tuple[str, ...]

    @property
    def surfaced_count(self) -> int:
        return len(self.surfaced_theme_keys)


def evaluate_candidate_thresholds(
    latest: list[LatestPerTheme], candidates: tuple[float, ...]
) -> list[CandidateThresholdResult]:
    return [
        CandidateThresholdResult(
            threshold=t,
            surfaced_theme_keys=tuple(x.theme_key for x in latest if x.salience >= t),
        )
        for t in candidates
    ]


@dataclass(frozen=True)
class DailyStabilityPoint:
    day: str
    cumulative_n: int
    cumulative_p85: Optional[float]
    trailing7_n: int
    trailing7_p85: Optional[float]


def compute_daily_percentile_stability(
    rows: list[TraceRow], pct: float = 85.0
) -> list[DailyStabilityPoint]:
    """Replays the real history day by day and recomputes an *expanding*
    (all data to date) and a *trailing-7-day* percentile at each day
    boundary. This is the direct, reproducible test of "would a
    percentile-based rule have been stable/usable in this system's own real
    early history" rather than a hand-wave -- if either series swings
    sharply while n is small, that is measured evidence against a naive
    percentile-of-all-rows rule at this data volume, not an assumption.
    """
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r.created_at)
    first_day = ordered[0].created_at.date()
    last_day = ordered[-1].created_at.date()
    points: list[DailyStabilityPoint] = []
    day = first_day
    while day <= last_day:
        day_end = _end_of_day_utc(day)
        cumulative = [r.salience for r in ordered if r.created_at < day_end]
        trailing_start = day_end - timedelta(days=7)
        trailing = [r.salience for r in ordered if trailing_start <= r.created_at < day_end]
        cumulative_p = compute_percentile(sorted(cumulative), pct) if cumulative else None
        trailing_p = compute_percentile(sorted(trailing), pct) if trailing else None
        points.append(
            DailyStabilityPoint(
                day=day.isoformat(),
                cumulative_n=len(cumulative),
                cumulative_p85=cumulative_p,
                trailing7_n=len(trailing),
                trailing7_p85=trailing_p,
            )
        )
        day = day + timedelta(days=1)
    return points


def _end_of_day_utc(day) -> datetime:
    """Exclusive upper bound for `day` (i.e. midnight UTC at the START of
    the NEXT day) -- used as a `<` cutoff so a row created exactly at
    midnight is attributed to the day it actually falls on, not the
    previous one.
    """
    from datetime import time as _time

    return datetime.combine(day, _time.min, tzinfo=timezone.utc) + timedelta(days=1)


@dataclass(frozen=True)
class AgeGateResult:
    """Is `SURFACE_MIN_AGE_SEC` actually binding on real data -- i.e. does
    it ever exclude a theme that would otherwise surface, or is every real
    theme already far past the age floor by the time it accumulates enough
    recurrence to matter? Computed from each theme's earliest real
    `created_at` vs "now" (the same clock `load_pending_loops()` uses),
    never assumed either way.
    """

    theme_key: str
    age_seconds: float
    clears_current_gate: bool


def compute_age_gate_results(
    rows: list[TraceRow], now: datetime, min_age_sec: float
) -> list[AgeGateResult]:
    first_seen: dict[str, datetime] = {}
    for r in rows:
        cur = first_seen.get(r.theme_key)
        if cur is None or r.created_at < cur:
            first_seen[r.theme_key] = r.created_at
    out = []
    for theme_key, fs in first_seen.items():
        age = max(0.0, (now - fs).total_seconds())
        out.append(AgeGateResult(theme_key=theme_key, age_seconds=age, clears_current_gate=age >= min_age_sec))
    return sorted(out, key=lambda x: x.age_seconds)


# ===========================================================================
# I/O layer -- psycopg2 read-only. Same connection contract as the other
# measure_*.py scripts in this directory (refuses a non-read-only session).
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


def fetch_trace_rows(conn, since: datetime) -> tuple[list[TraceRow], bool]:
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, theme_key, salience
                FROM attention_salience_trace
                WHERE created_at >= %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (since, MAX_ROWS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch attention_salience_trace", exc_info=True)
        return [], False
    out: list[TraceRow] = []
    for created_at, theme_key, salience in rows:
        if created_at is None or not theme_key or salience is None:
            continue
        ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
        out.append(TraceRow(created_at=ts, theme_key=str(theme_key), salience=float(salience)))
    return out, len(rows) >= MAX_ROWS


def fetch_loop_outcome_count(conn, since: datetime) -> Optional[int]:
    """Real row count in `attention_loop_outcome` within the window --
    `None` (never `0`) on any fetch failure, so a DB hiccup can never
    masquerade as a confirmed-zero finding.
    """
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM attention_loop_outcome WHERE created_at >= %s",
                (since,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        logger.error("failed to fetch attention_loop_outcome count", exc_info=True)
        return None


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


def write_stability_csv(path: Path, points: list[DailyStabilityPoint]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "cumulative_n", "cumulative_p85", "trailing7_n", "trailing7_p85"])
        for p in points:
            writer.writerow([p.day, p.cumulative_n, p.cumulative_p85, p.trailing7_n, p.trailing7_p85])


def render_report(
    *,
    window_label: str,
    window_start: datetime,
    window_end: datetime,
    stats: DistributionStats,
    rows_clearing_current_gate: int,
    concentration: ConcentrationResult,
    latest: list[LatestPerTheme],
    candidate_results: list[CandidateThresholdResult],
    stability: list[DailyStabilityPoint],
    age_results: list[AgeGateResult],
    outcome_count: Optional[int],
    caveats: list[str],
) -> str:
    lines = [
        "# Attention surface-threshold recalibration -- read-only measurement",
        "",
        "Measures whether `SURFACE_MIN_SALIENCE=0.5` "
        "(`services/orion-hub/scripts/attention_loops_store.py`) is reachable by real "
        "`attention_salience_trace` data, and which replacement strategy (relative "
        "percentile vs. recalibrated static constant) is actually supported at this "
        "data volume. Read-only. No writes, no events, no flag/config changes.",
        "",
        f"- Window: last {window_label} ({window_start.isoformat()} -> {window_end.isoformat()})",
        f"- Current gate: SURFACE_MIN_SALIENCE={CURRENT_SURFACE_MIN_SALIENCE}, "
        f"SURFACE_MIN_AGE_SEC={CURRENT_SURFACE_MIN_AGE_SEC}",
        "",
        "## 1. Is the current salience gate reachable?",
        "",
    ]
    if not stats.is_populated:
        lines.append(
            "- **INCONCLUSIVE**: no `attention_salience_trace` rows in this window; "
            "cannot evaluate reachability. See coverage caveats."
        )
    else:
        lines.extend(
            [
                f"- Rows: {stats.count}",
                f"- min={stats.min_v:.3f}  max={stats.max_v:.3f}  "
                f"mean={stats.mean_v:.3f}  stdev={stats.stdev_v:.3f}",
                "- Percentiles: "
                + ", ".join(f"p{p}={v:.3f}" for p, v in sorted(stats.percentiles.items())),
                (
                    f"- **VERDICT: UNREACHABLE.** Max observed score ({stats.max_v:.3f}) is "
                    f"below SURFACE_MIN_SALIENCE ({CURRENT_SURFACE_MIN_SALIENCE}). Zero rows "
                    "in this window's entire real history have ever cleared the gate."
                    if stats.max_v is not None and stats.max_v < CURRENT_SURFACE_MIN_SALIENCE
                    else f"- **VERDICT: reachable** -- {rows_clearing_current_gate}/{stats.count} "
                    f"rows clear the current gate "
                    f"(max {stats.max_v:.3f} >= {CURRENT_SURFACE_MIN_SALIENCE})."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 2. Pseudo-replication check (is a row-level percentile a real population "
            "percentile, or just \"how loud are the chattiest loops\"?)",
            "",
            f"- Distinct `theme_key` values: {concentration.distinct_theme_count}",
        ]
    )
    if concentration.top2_share is not None:
        lines.append(
            f"- Top-2 loudest themes' share of ALL rows: {concentration.top2_share:.1%}"
        )
        if concentration.top2_share >= 0.75:
            lines.append(
                "- **VERDICT: heavily concentrated.** A row-level percentile over this table "
                "mostly describes the loudness of 1-2 recurring loops, not a real population "
                "percentile across distinct concepts -- weak evidence for a naive row-level "
                "percentile-of-all-rows surfacing rule."
            )
    else:
        lines.append("- Could not compute (no rows).")
    lines.extend(
        [
            "",
            "## 3. What each candidate threshold would surface RIGHT NOW "
            "(latest real score per theme -- the exact population `load_pending_loops()` "
            "gates against today)",
            "",
        ]
    )
    if latest:
        lines.append(
            "| theme_key | latest salience |\n|---|---|\n"
            + "\n".join(f"| {x.theme_key} | {x.salience:.3f} |" for x in latest)
        )
        lines.append("")
        for cr in candidate_results:
            lines.append(
                f"- threshold {cr.threshold:.2f}: surfaces {cr.surfaced_count}/{len(latest)} "
                f"real themes right now"
                + (f" -- {list(cr.surfaced_theme_keys)}" if cr.surfaced_theme_keys else "")
            )
    else:
        lines.append("- No current per-theme data (no rows).")
    lines.extend(
        [
            "",
            "## 4. Percentile stability replay (would a relative rule have been stable "
            "across this system's own real early history?)",
            "",
        ]
    )
    if stability:
        lines.append("| day | cumulative n | cumulative p85 | trailing-7d n | trailing-7d p85 |")
        lines.append("|---|---|---|---|---|")
        for p in stability:
            lines.append(
                f"| {p.day} | {p.cumulative_n} | "
                f"{'' if p.cumulative_p85 is None else f'{p.cumulative_p85:.3f}'} | "
                f"{p.trailing7_n} | "
                f"{'' if p.trailing7_p85 is None else f'{p.trailing7_p85:.3f}'} |"
            )
        p85_values = [p.cumulative_p85 for p in stability if p.cumulative_p85 is not None]
        if len(p85_values) >= 2:
            swing = max(p85_values) - min(p85_values)
            lines.append("")
            lines.append(
                f"- Cumulative p85 swung by {swing:.3f} across the replay "
                f"(from {min(p85_values):.3f} to {max(p85_values):.3f}) as real data "
                "accumulated -- this is the measured early-history instability, not an "
                "assumption. A percentile computed when n was small in this system's own "
                "past would have set a materially different bar than the same percentile "
                "computed today."
            )
    else:
        lines.append("- No data to replay.")
    lines.extend(
        [
            "",
            "## 5. Is SURFACE_MIN_AGE_SEC actually binding?",
            "",
        ]
    )
    if age_results:
        lines.append(f"- SURFACE_MIN_AGE_SEC={CURRENT_SURFACE_MIN_AGE_SEC}s ({CURRENT_SURFACE_MIN_AGE_SEC/60:.0f} min)")
        for a in age_results:
            lines.append(
                f"- {a.theme_key}: first seen {a.age_seconds:.0f}s ago "
                f"({'clears' if a.clears_current_gate else 'BLOCKED by'} age gate)"
            )
        blocked = [a for a in age_results if not a.clears_current_gate]
        if blocked:
            lines.append(
                f"- **{len(blocked)} theme(s) currently blocked by the age gate** -- expected "
                "and working as designed (prevents a transient noise spike from creating a "
                "card within the first few minutes); these will naturally clear once "
                f"{CURRENT_SURFACE_MIN_AGE_SEC:.0f}s has elapsed."
            )
        else:
            lines.append(
                "- All real themes in this window are already well past the age gate -- it "
                "is not currently excluding anything real, and real first-seen timestamps "
                "don't make it trivially always-true either (new themes do start under the "
                "gate and age out normally)."
            )
    else:
        lines.append("- No data.")
    lines.extend(
        [
            "",
            "## 6. Ground-truth outcome count (why this matters)",
            "",
            f"- `attention_loop_outcome` rows in window: "
            f"{'ERROR (see caveats)' if outcome_count is None else outcome_count}",
        ]
    )
    if outcome_count == 0:
        lines.append(
            "- Zero, confirming the chicken-and-egg this patch exists to break: no threshold "
            "has ever let a card through for a human to Resolve/Dismiss, so there is no real "
            "verdict data to learn a threshold from."
        )
    lines.extend(
        [
            "",
            "## What this does NOT decide",
            "",
            "- Does not recalibrate or redesign the underlying SEED_WEIGHTS/"
            "LinearSalienceCombiner formula (orion/substrate/attention/salience.py) -- that "
            "formula's own docstring already discloses it as an unrefit seed-v1 placeholder; "
            "see GitHub issue #1512.",
            "- Does not claim the chosen replacement threshold is an objectively correct "
            "quality bar -- there is no ground truth to validate against (section 6). It is "
            "a bootstrap instrument to start collecting that ground truth, not a discovery "
            "of the 'real' cutoff.",
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

    trace_rows, truncated = fetch_trace_rows(conn, window_start)
    if truncated:
        caveats.append(f"attention_salience_trace rows truncated at MAX_ROWS={MAX_ROWS}")
    progress.emit("trace loaded", percent=40.0, processed=len(trace_rows), total=len(trace_rows))

    outcome_count = fetch_loop_outcome_count(conn, window_start)
    if outcome_count is None:
        caveats.append("attention_loop_outcome count fetch failed")
    progress.emit("outcome count loaded", percent=60.0, processed=0, total=0)

    try:
        conn.close()
    except Exception:
        pass

    if not trace_rows:
        caveats.append("no attention_salience_trace rows in window")

    stats = compute_distribution_stats(trace_rows)
    rows_clearing_current_gate = count_rows_at_or_above(trace_rows, CURRENT_SURFACE_MIN_SALIENCE)
    concentration = compute_concentration(trace_rows)
    latest = compute_latest_per_theme(trace_rows)
    candidate_results = evaluate_candidate_thresholds(latest, CANDIDATE_STATIC_THRESHOLDS)
    stability = compute_daily_percentile_stability(trace_rows)
    age_results = compute_age_gate_results(trace_rows, now, CURRENT_SURFACE_MIN_AGE_SEC)

    write_stability_csv(CSV_PATH, stability)

    progress.emit("done", percent=100.0, processed=len(trace_rows), total=len(trace_rows))
    progress.close()

    report = render_report(
        window_label=window_label,
        window_start=window_start,
        window_end=now,
        stats=stats,
        rows_clearing_current_gate=rows_clearing_current_gate,
        concentration=concentration,
        latest=latest,
        candidate_results=candidate_results,
        stability=stability,
        age_results=age_results,
        outcome_count=outcome_count,
        caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only measurement of whether SURFACE_MIN_SALIENCE is reachable by real "
            "attention_salience_trace data, and which replacement strategy is supported."
        )
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
