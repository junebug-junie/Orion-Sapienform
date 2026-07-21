#!/usr/bin/env python3
"""Read-only emergent-clustering probe over real `FieldAttentionFrameV1` history.

Sentience Striving Program (`orion/sentience_striving_program/README.md`) §6
item 5, toward outcome O4: *"the 'what are Orion's drives' question is
answered empirically, continuously, not asserted once by a human design
chat."* This is the "Recommended next patch" from
`docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-
design.md`: pull real, historical `FieldAttentionFrameV1` rows already
produced live by `orion-attention-runtime`, and check with real numbers
whether `dominant_targets`/`capability_targets` produce (a) recognizably
similar, non-identical, non-random groupings across two historical windows
via correlation-based grouping (the baseline design's literal acceptance
check), and (b) a differentiation profile meaningfully different from the
drives system's own documented 96%-`dominant_drive`-monoculture pathology.

**Verified table (not assumed):** `orion/attention/field_attention/
{scoring,selectors,builder}.py` computes `FieldAttentionFrameV1` in-process;
`services/orion-attention-runtime/app/store.py::save_attention_frame()` is
the actual write path, persisting to Postgres `substrate_attention_frames`
(`frame_json` jsonb column). Confirmed live via `\\d substrate_attention_
frames` and a row-count query (2026-07-21): 127,953+ rows spanning
~2026-07-18T19:34Z to ~2026-07-21T19:35Z (~72h retention, matching the
design doc's own "~43k rows/day retained 72h" characterization).

This performs NO writes, emits NO events, flips NO flags, and imports
NOTHING from `orion.spark.concept_induction` (per this patch's own hard
constraint) -- the drives-system comparison numbers below are transcribed,
with citations, from `orion/autonomy/drives_and_autonomy_retrospective.md`
as plain float constants, not imported code.

## Disclosed scoping decisions (baseline design under-specifies these)

The baseline design names "correlation-based grouping" and "recognizably
similar (not identical, not random)" but does not specify a concrete
similarity metric, window size, or classification bands. Decided here,
stated plainly rather than silently assumed:

1. **Target universe is small by construction.** Live data shows only
   ~9 distinct `target_id`s ever appear (3 nodes, up to 5 capabilities,
   1 system target) -- `FieldAttentionFrameV1.dominant_targets` is a
   *capped union* of `node_targets + capability_targets + system_targets`
   (confirmed by reading `orion/attention/field_attention/builder.py`), not
   an independently-sampled top-N out of a large pool. This caps the
   correlation matrix at C(9,2)=36 possible pairs. That is real, not a
   defect of this script -- reported honestly, including how many of those
   36 pairs turn out non-degenerate.
2. **Absence = 0.0 salience**, same convention as the sibling script
   `measure_capability_salience_coupling.py::extract_salience_for_target`
   (a target absent from a tick's list means it fell below the live
   policy's `min_salience=0.10`, not a missing observation).
3. **Correlation signal: per-target `salience_score` time series, not
   binary presence.** Binary co-occurrence is close to degenerate here
   (several targets are present on >99% of ticks by construction -- see
   finding below), so the magnitude series carries the real information
   about which targets rise and fall together.
4. **Similarity metric, two independent measures, both reported:**
   - *Correlation-of-correlations*: flatten each window's pairwise
     correlation matrix (restricted to pairs with a real, non-degenerate
     value in BOTH windows) into a vector and take the Pearson correlation
     between the two windows' vectors. This is the standard way to ask "do
     two correlation structures agree," used the same way a Mantel test or
     RV-coefficient compares two correlation/distance matrices. Requires
     >= 3 common non-degenerate pairs to be meaningful (n=1 or n=2 makes
     "correlation of correlations" undefined in any useful sense -- reported
     as `INCONCLUSIVE_INSUFFICIENT_PAIRS`, not forced to a number).
   - *Jaccard overlap of "co-winning" pairs*: the set of pairs whose
     correlation clears `CORR_CLUSTER_THRESHOLD` (0.5) in each window,
     compared by Jaccard index. This is the literal "top co-occurring
     pairs" metric named as an option in the task brief.
5. **Classification bands** (`classify_similarity`): explicit, documented
   numeric thresholds -- see that function's docstring for the exact bands
   and the reasoning for each. Not asserted as universally correct, just
   disclosed rather than hidden in an if/else.
6. **Window sizing**: chosen from the real observed data span, not a fixed
   guess. Default 24h windows with a 12h gap between them (`choose_windows`)
   -- checked live to fit inside the real ~72h retention window with room
   to spare (24 + 12 + 24 = 60h <= ~72h available as of 2026-07-21).
   `choose_windows` degrades gracefully (shrinks the window, drops the gap)
   if run against a shorter real span, and returns `None` (reported as
   insufficient data, not a fabricated split) if even that isn't possible.

Run:
    python scripts/analysis/measure_emergent_clustering_probe.py --window-hours 24 --gap-hours 12
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("orion.analysis.emergent_clustering_probe")

DEFAULT_WINDOW_HOURS: float = 24.0
DEFAULT_GAP_HOURS: float = 12.0
MIN_WINDOW_HOURS: float = 4.0
CORR_CLUSTER_THRESHOLD: float = 0.5
DEGENERATE_VARIANCE_EPS: float = 1e-12
MIN_COMMON_PAIRS_FOR_CORR_OF_CORR: int = 3
MIN_TOTAL_ROWS: int = 200
MIN_WINDOW_ROWS: int = 50
MAX_ROWS: int = 200_000

# From orion/autonomy/drives_and_autonomy_retrospective.md, transcribed as
# plain float constants (NOT imported code -- this script imports nothing
# from orion.spark.concept_induction, per this patch's hard constraint).
# Line ~177: "producing dominant_drive=relational in 96% of ticks" (the
# pre-fix pathology this task explicitly names). Line ~267: post O1/O2/O3
# fix, "top dominant-drive share dropped from the 96% relational monoculture
# ... down to 31.65%". Line ~487: a later measurement held at "top share
# 32.05%". Both post-fix numbers are close; 31.65% is used as the primary
# post-fix reference since it is the first, most-cited post-fix figure.
DRIVES_DOMINANT_SHARE_PRE_FIX: float = 0.96
DRIVES_DOMINANT_SHARE_POST_FIX: float = 0.3165

OUTPUT_DIR = Path("/tmp/emergent-clustering-probe")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "ticks.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O. Parses raw frame_json target lists, builds aligned
# series, computes correlation matrices, clusters, and similarity metrics.
# Exercised directly by unit tests with synthetic fixtures, no DB.
# ===========================================================================


def _parse_target_list(raw: Any) -> list[dict]:
    """Parse a raw `frame_json->'dominant_targets'` or `->'capability_targets'`
    value (possibly a JSON string, possibly malformed, possibly None) into a
    list of dict entries. Never raises."""
    try:
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            return []
        return [e for e in raw if isinstance(e, dict)]
    except Exception:
        return []


def extract_target_salience_map(dominant_raw: Any, capability_raw: Any) -> dict[str, float]:
    """Merge one tick's `dominant_targets` and `capability_targets` raw JSON
    into a single {target_id: salience_score} map. `dominant_targets` is
    authoritative (it is the capped union computed by
    `orion/attention/field_attention/builder.py::build_attention_frame`);
    `capability_targets` only fills in a target_id that, for some reason,
    didn't make it into `dominant_targets` (defensive, not expected on real
    data -- see module docstring finding #1). Never raises."""
    result: dict[str, float] = {}
    for entry in _parse_target_list(dominant_raw):
        tid = entry.get("target_id")
        if not isinstance(tid, str):
            continue
        try:
            score = float(entry.get("salience_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        result[tid] = score
    for entry in _parse_target_list(capability_raw):
        tid = entry.get("target_id")
        if not isinstance(tid, str) or tid in result:
            continue
        try:
            score = float(entry.get("salience_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        result[tid] = score
    return result


def build_target_universe(raw_maps: list[dict[str, float]]) -> list[str]:
    """Sorted, deduped target_ids observed anywhere in `raw_maps`."""
    seen: set[str] = set()
    for m in raw_maps:
        seen.update(m.keys())
    return sorted(seen)


def align_series(raw_maps: list[dict[str, float]], target_ids: list[str]) -> dict[str, list[float]]:
    """Per-target salience_score series aligned to `raw_maps`' tick order.
    Absence in a tick's map -> 0.0 (see module docstring, disclosed decision
    #2)."""
    series: dict[str, list[float]] = {t: [] for t in target_ids}
    for m in raw_maps:
        for t in target_ids:
            series[t].append(float(m.get(t, 0.0)))
    return series


def pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    """Plain-Python Pearson correlation, no numpy/scipy dependency (matches
    this patch's constraint against pulling in a clustering/ML library for
    day one). Returns None if the series lengths mismatch, are too short,
    contain a non-finite value (NaN/inf -- a malformed upstream salience
    value, not real signal), or either series is degenerate (near-zero
    variance -- flat/constant, not real signal; see the metric-quality-gate
    "not degenerate" check).

    Deliberate NaN handling: Python's `min`/`max` do NOT reliably reject NaN
    (`min(1.0, float("nan"))` returns `1.0`, since NaN comparisons are always
    False), so a NaN-poisoned series would otherwise silently clamp to a
    false "perfect correlation" instead of being caught by the variance
    check above (variance involving NaN is itself NaN, which also fails to
    trip the `<=` comparison). Checked explicitly, up front, before any
    arithmetic that could propagate it."""
    if len(x) != len(y) or len(x) < 2:
        return None
    if any(not math.isfinite(v) for v in x) or any(not math.isfinite(v) for v in y):
        return None
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x <= DEGENERATE_VARIANCE_EPS or var_y <= DEGENERATE_VARIANCE_EPS:
        return None
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    r = cov / denom
    return max(-1.0, min(1.0, r))


CorrMatrix = dict[tuple[str, str], Optional[float]]


def compute_correlation_matrix(target_ids: list[str], series: dict[str, list[float]]) -> CorrMatrix:
    """Pairwise Pearson correlation over the upper triangle, keyed by
    (target_a, target_b) with target_a < target_b lexicographically for a
    deterministic key regardless of `target_ids` order."""
    matrix: CorrMatrix = {}
    ordered = sorted(target_ids)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            matrix[(a, b)] = pearson_correlation(series[a], series[b])
    return matrix


def cluster_by_correlation(
    target_ids: list[str], corr_matrix: CorrMatrix, threshold: float = CORR_CLUSTER_THRESHOLD
) -> list[list[str]]:
    """Union-find grouping: two targets join the same cluster iff their
    salience correlation is real (non-None) and >= threshold. Unconnected
    or degenerate targets end up as singleton clusters -- this is the
    correlation-based grouping the baseline design names as the day-one
    starting point (not a from-scratch ML clustering pipeline, per its own
    non-goal)."""
    parent = {t: t for t in target_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b), r in corr_matrix.items():
        if r is not None and r >= threshold:
            union(a, b)

    groups: dict[str, list[str]] = {}
    for t in target_ids:
        groups.setdefault(find(t), []).append(t)
    clusters = [sorted(g) for g in groups.values()]
    clusters.sort(key=lambda g: (-len(g), g))
    return clusters


def edge_set_at_threshold(corr_matrix: CorrMatrix, threshold: float = CORR_CLUSTER_THRESHOLD) -> set[tuple[str, str]]:
    return {pair for pair, r in corr_matrix.items() if r is not None and r >= threshold}


def jaccard_similarity(a: set, b: set) -> Optional[float]:
    """None only when both sets are empty (undefined, not zero -- zero would
    falsely read as 'maximally dissimilar' when really there was nothing to
    compare)."""
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def correlation_of_correlations(corr_a: CorrMatrix, corr_b: CorrMatrix) -> tuple[Optional[float], int]:
    """Pearson correlation between two windows' flattened correlation
    matrices, restricted to pairs with a real (non-None) value in BOTH.
    Returns (value, n_common_pairs); value is None if n_common_pairs < 3
    (too few degrees of freedom for this to mean anything -- see module
    docstring disclosed decision #4)."""
    common_pairs = sorted(set(corr_a) & set(corr_b))
    xs: list[float] = []
    ys: list[float] = []
    for p in common_pairs:
        va, vb = corr_a[p], corr_b[p]
        if va is None or vb is None:
            continue
        xs.append(va)
        ys.append(vb)
    n = len(xs)
    if n < MIN_COMMON_PAIRS_FOR_CORR_OF_CORR:
        return None, n
    return pearson_correlation(xs, ys), n


def classify_similarity(
    corr_of_corr: Optional[float], jaccard: Optional[float], n_common_pairs: int
) -> str:
    """Explicit classification bands for the baseline design's acceptance
    check ("recognizably similar, not identical, not random"). These exact
    numbers are a disclosed scoping decision (see module docstring #5), not
    handed down by the design doc:

    - `INCONCLUSIVE_INSUFFICIENT_PAIRS`: fewer than
      MIN_COMMON_PAIRS_FOR_CORR_OF_CORR=3 non-degenerate common pairs, or
      corr_of_corr could not be computed. Too little real structure to
      judge either way.
    - `IDENTICAL_TRIVIAL`: corr_of_corr >= 0.999 AND jaccard == 1.0 -- the
      two windows agree completely. Given the small (~9-target, <=36-pair)
      universe here (module docstring #1), this is flagged as possibly
      trivial rather than celebrated -- a tiny, mostly-fixed target universe
      can produce perfect agreement without that being evidence of genuine
      *emergent* recurring structure, since there may be little room for it
      to differ in the first place.
    - `RANDOM`: corr_of_corr < 0.3, or jaccard is not None and < 0.15 --
      the two windows' co-movement structure does not recognizably agree.
    - `RECOGNIZABLE_SIMILARITY`: corr_of_corr >= 0.5 AND jaccard is not None
      and 0.15 <= jaccard < 1.0 -- meaningful, non-trivial, non-identical
      agreement. This is the literal MET case for the acceptance check.
    - `AMBIGUOUS`: falls in neither band above (e.g. moderate corr_of_corr
      with an extreme jaccard, or vice versa) -- reported plainly as
      ambiguous rather than forced into MET or NOT MET.
    """
    if n_common_pairs < MIN_COMMON_PAIRS_FOR_CORR_OF_CORR or corr_of_corr is None:
        return "INCONCLUSIVE_INSUFFICIENT_PAIRS"
    if corr_of_corr >= 0.999 and jaccard == 1.0:
        return "IDENTICAL_TRIVIAL"
    if corr_of_corr < 0.3 or (jaccard is not None and jaccard < 0.15):
        return "RANDOM"
    if corr_of_corr >= 0.5 and jaccard is not None and 0.15 <= jaccard < 1.0:
        return "RECOGNIZABLE_SIMILARITY"
    return "AMBIGUOUS"


def top1_winner(target_map: dict[str, float]) -> Optional[str]:
    """The single highest-salience target_id for one tick -- the closest
    real analog to the drives system's `dominant_drive`. Deterministic tie-
    break: highest score, then alphabetical target_id."""
    if not target_map:
        return None
    return sorted(target_map.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def top1_winner_distribution(raw_maps: list[dict[str, float]]) -> Counter:
    counter: Counter = Counter()
    for m in raw_maps:
        winner = top1_winner(m)
        if winner is not None:
            counter[winner] += 1
    return counter


def membership_frequency(raw_maps: list[dict[str, float]], target_ids: list[str]) -> dict[str, float]:
    """Fraction of ticks in which each target_id appears at all in the
    tick's target map (i.e. was salient enough to clear min_salience)."""
    n = len(raw_maps)
    if n == 0:
        return {t: 0.0 for t in target_ids}
    counts = {t: 0 for t in target_ids}
    for m in raw_maps:
        for t in m:
            if t in counts:
                counts[t] += 1
    return {t: counts[t] / n for t in target_ids}


@dataclass
class WindowSpec:
    start: datetime
    end: datetime
    label: str


def choose_windows(
    min_ts: datetime,
    max_ts: datetime,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    gap_hours: float = DEFAULT_GAP_HOURS,
    min_window_hours: float = MIN_WINDOW_HOURS,
) -> Optional[tuple[WindowSpec, WindowSpec]]:
    """Anchor window A at the start of the real available span and window B
    at the end, with `gap_hours` of untouched history between them (real
    non-overlapping windows, not just non-identical timestamps). Degrades
    gracefully if the real span is shorter than requested: shrinks the
    window size (never below `min_window_hours`) and drops the gap before
    giving up. Returns None if even a back-to-back split can't produce two
    windows of at least `min_window_hours` each -- i.e. genuinely
    insufficient real historical data, not a number to fabricate a verdict
    from."""
    if min_ts is None or max_ts is None or max_ts <= min_ts:
        return None
    total_hours = (max_ts - min_ts).total_seconds() / 3600.0

    if total_hours >= 2 * window_hours + gap_hours:
        a_start, a_end = min_ts, min_ts + timedelta(hours=window_hours)
        b_end = max_ts
        b_start = b_end - timedelta(hours=window_hours)
        return (
            WindowSpec(a_start, a_end, f"{window_hours:g}h (window A, start-anchored)"),
            WindowSpec(b_start, b_end, f"{window_hours:g}h (window B, end-anchored)"),
        )

    if total_hours >= 2 * min_window_hours:
        half_hours = total_hours / 2.0
        mid = min_ts + timedelta(hours=half_hours)
        return (
            WindowSpec(min_ts, mid, f"{half_hours:g}h (window A, back-to-back split, no gap)"),
            WindowSpec(mid, max_ts, f"{half_hours:g}h (window B, back-to-back split, no gap)"),
        )

    return None


def partition_by_window(
    timestamps: list[datetime],
    raw_maps: list[dict[str, float]],
    win_a: WindowSpec,
    win_b: WindowSpec,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Split `raw_maps` into (window A, window B) by their aligned
    `timestamps`, guaranteeing a real, non-overlapping partition even when
    `win_a.end == win_b.start` exactly -- `choose_windows`' degraded
    back-to-back path produces exactly that shared boundary instant when the
    real data span is too short for a real gap. Half-open on window A's
    upper bound (`win_a.start <= ts < win_a.end`), inclusive on window B's
    both ends (`win_b.start <= ts <= win_b.end`, so the very last real row,
    exactly at `max_ts`, is still counted) -- a row landing exactly on the
    shared boundary goes to B only, never both. Makes no difference in the
    ample-span case, where `win_a.end < win_b.start` with real daylight
    between them."""
    maps_a: list[dict[str, float]] = []
    maps_b: list[dict[str, float]] = []
    for ts, m in zip(timestamps, raw_maps):
        if win_a.start <= ts < win_a.end:
            maps_a.append(m)
        elif win_b.start <= ts <= win_b.end:
            maps_b.append(m)
    return maps_a, maps_b


# ===========================================================================
# I/O layer -- psycopg2 read-only. Connection contract mirrors
# measure_ast_hot_reducer.py / measure_capability_salience_coupling.py
# exactly (refuses a non-read-only session).
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


def fetch_frame_time_range(conn) -> tuple[Optional[datetime], Optional[datetime], int]:
    if conn is None:
        return None, None, 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT min(generated_at), max(generated_at), count(*) FROM substrate_attention_frames")
            row = cur.fetchone()
    except Exception:
        logger.error("failed to fetch substrate_attention_frames time range", exc_info=True)
        return None, None, 0
    if not row:
        return None, None, 0
    min_ts, max_ts, count = row
    if min_ts is not None and min_ts.tzinfo is None:
        min_ts = min_ts.replace(tzinfo=timezone.utc)
    if max_ts is not None and max_ts.tzinfo is None:
        max_ts = max_ts.replace(tzinfo=timezone.utc)
    return min_ts, max_ts, int(count or 0)


def fetch_target_rows(
    conn, start: datetime, end: datetime, max_rows: int = MAX_ROWS
) -> tuple[list[tuple[datetime, Any, Any]], bool]:
    """Real historical rows in [start, end], ASC by generated_at. Each row:
    (generated_at, dominant_targets_raw, capability_targets_raw)."""
    if conn is None:
        return [], False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT generated_at, frame_json->'dominant_targets', frame_json->'capability_targets'
                FROM substrate_attention_frames
                WHERE generated_at >= %s AND generated_at <= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (start, end, max_rows),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch substrate_attention_frames rows", exc_info=True)
        return [], False
    out: list[tuple[datetime, Any, Any]] = []
    for ts, dom_raw, cap_raw in rows:
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append((ts, dom_raw, cap_raw))
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


def write_ticks_csv(path: Path, raw_maps: list[dict[str, float]], timestamps: list[datetime], target_ids: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generated_at"] + target_ids)
        for ts, m in zip(timestamps, raw_maps):
            writer.writerow([ts.isoformat()] + [f"{m.get(t, 0.0):.4f}" for t in target_ids])


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _corr_matrix_table(target_ids: list[str], corr_matrix: CorrMatrix) -> list[str]:
    lines = ["| target A | target B | correlation |", "| --- | --- | --- |"]
    for a in target_ids:
        for b in target_ids:
            if a >= b:
                continue
            r = corr_matrix.get((a, b))
            lines.append(f"| `{a}` | `{b}` | {_fmt(r)} |")
    return lines


def render_report(
    *,
    global_window_label: str,
    total_rows: int,
    windows: Optional[tuple[WindowSpec, WindowSpec]],
    window_a_n: int,
    window_b_n: int,
    target_ids: list[str],
    full_target_ids: list[str],
    corr_a: CorrMatrix,
    corr_b: CorrMatrix,
    clusters_a: list[list[str]],
    clusters_b: list[list[str]],
    edges_a: set[tuple[str, str]],
    edges_b: set[tuple[str, str]],
    jaccard: Optional[float],
    corr_of_corr: Optional[float],
    n_common_pairs: int,
    classification: str,
    full_history_top1: Counter,
    full_history_n: int,
    full_history_membership: dict[str, float],
    caveats: list[str],
) -> str:
    lines = [
        "# Emergent-Clustering Probe (O4) -- Field-Attention Coalition History",
        "",
        "Read-only. No writes, no events, no flag/config changes. Runs correlation-based "
        "grouping over real `substrate_attention_frames` history (real "
        "`FieldAttentionFrameV1.dominant_targets`/`capability_targets` rows), per the "
        "baseline field-native motivational substrate design's 'Recommended next patch' "
        "and Sentience Striving Program §6 item 5. Imports nothing from "
        "`orion.spark.concept_induction`.",
        "",
        f"- Real data span: {global_window_label}",
        f"- Total `substrate_attention_frames` rows in span: {total_rows}",
        f"- Target universe observed (dominant_targets ∪ capability_targets): {len(target_ids)} "
        f"({', '.join(f'`{t}`' for t in target_ids)})",
        "",
    ]

    if windows is None:
        lines.extend(
            [
                "## Windows: INSUFFICIENT REAL DATA",
                "",
                f"The real available span ({global_window_label}) is too short to carve out "
                f"two non-overlapping windows of at least {MIN_WINDOW_HOURS:g}h each. No "
                "clustering-similarity check was run. This is reported honestly rather than "
                "forcing a verdict from too little data.",
                "",
            ]
        )
    else:
        win_a, win_b = windows
        lines.extend(
            [
                "## Windows selected (real data, see `choose_windows` for the anchoring rule)",
                "",
                f"- Window A: {win_a.label} -- {win_a.start.isoformat()} -> {win_a.end.isoformat()} "
                f"({window_a_n} rows)",
                f"- Window B: {win_b.label} -- {win_b.start.isoformat()} -> {win_b.end.isoformat()} "
                f"({window_b_n} rows)",
                "",
            ]
        )

        if window_a_n < MIN_WINDOW_ROWS or window_b_n < MIN_WINDOW_ROWS:
            lines.extend(
                [
                    "**INSUFFICIENT ROWS IN AT LEAST ONE WINDOW** -- below "
                    f"MIN_WINDOW_ROWS={MIN_WINDOW_ROWS}. Correlation matrices below may be "
                    "unreliable; treat the acceptance-check verdict as low-confidence.",
                    "",
                ]
            )

        lines.extend(["### Window A correlation matrix", ""])
        lines.extend(_corr_matrix_table(target_ids, corr_a))
        lines.extend(["", "### Window B correlation matrix", ""])
        lines.extend(_corr_matrix_table(target_ids, corr_b))
        lines.extend(
            [
                "",
                f"### Clusters at threshold >= {CORR_CLUSTER_THRESHOLD:g}",
                "",
                f"- Window A: {clusters_a}",
                f"- Window B: {clusters_b}",
                "",
                "### Similarity metrics",
                "",
                f"- Edge set A (`corr >= {CORR_CLUSTER_THRESHOLD:g}`): {sorted(edges_a)}",
                f"- Edge set B (`corr >= {CORR_CLUSTER_THRESHOLD:g}`): {sorted(edges_b)}",
                f"- Jaccard(edges_A, edges_B): {_fmt(jaccard)}",
                f"- Correlation-of-correlations (over {n_common_pairs} common non-degenerate "
                f"pairs): {_fmt(corr_of_corr)}",
                "",
                f"### Classification: **{classification}**",
                "",
            ]
        )
        lines.extend(
            {
                "IDENTICAL_TRIVIAL": [
                    "Windows A and B agree completely (corr-of-corr >= 0.999, edge-set "
                    "Jaccard == 1.0). Given the small (~9-target) universe here, this is "
                    "flagged as possibly trivial rather than treated as strong evidence of "
                    "genuine emergent recurring structure -- see module docstring "
                    "disclosed decision #5.",
                ],
                "RANDOM": [
                    "Windows A and B do not recognizably agree (low corr-of-corr and/or low "
                    "edge-set Jaccard). This does not clear the baseline design's acceptance "
                    "check.",
                ],
                "RECOGNIZABLE_SIMILARITY": [
                    "Windows A and B show meaningful, non-identical, non-random agreement in "
                    "which targets co-vary in salience. This is the literal MET case for the "
                    "baseline design's acceptance check: \"The emergent clustering step, run "
                    "on two different historical windows, produces recognizably similar (not "
                    "identical, not random) groupings.\"",
                ],
                "INCONCLUSIVE_INSUFFICIENT_PAIRS": [
                    f"Fewer than {MIN_COMMON_PAIRS_FOR_CORR_OF_CORR} common non-degenerate "
                    "correlation pairs were available between the two windows -- too little "
                    "real structure (most targets' salience series were flat/near-constant, "
                    "or too few targets co-occurred with real variance in both windows) to "
                    "judge similarity either way.",
                ],
                "AMBIGUOUS": [
                    "The two windows' agreement falls between the defined RANDOM and "
                    "RECOGNIZABLE_SIMILARITY bands (see `classify_similarity` docstring for "
                    "the exact numeric thresholds) -- reported plainly rather than forced "
                    "into either verdict.",
                ],
            }.get(classification, ["(no narrative for this classification)"])
        )
        lines.append("")

    lines.extend(
        [
            "## Item 5: differentiation vs. the drives system's known monoculture",
            "",
            "The closest real analog to the drives system's `dominant_drive` is the single "
            "highest-`salience_score` target per tick (deterministic tie-break: highest "
            "score, then alphabetical target_id), computed over the FULL available real "
            "history (not just the two probe windows above).",
            "",
            f"- Full-history ticks: {full_history_n}",
            "",
            "| target_id | ticks won (top-1) | share |",
            "| --- | --- | --- |",
        ]
    )
    for target_id, count in full_history_top1.most_common():
        share = count / full_history_n if full_history_n else 0.0
        lines.append(f"| `{target_id}` | {count} | {share * 100:.2f}% |")
    top_share = (full_history_top1.most_common(1)[0][1] / full_history_n) if full_history_n and full_history_top1 else 0.0

    lines.extend(
        [
            "",
            "| target_id | fraction of ticks present in dominant_targets/capability_targets |",
            "| --- | --- |",
        ]
    )
    # Deliberately `full_target_ids`, NOT the window-scoped `target_ids` used
    # by the correlation matrices above -- this table's own text (just above)
    # claims full-history scope, so a target seen in full history but never
    # inside either probe window must still appear here.
    for target_id in full_target_ids:
        pct = full_history_membership.get(target_id, 0.0) * 100
        lines.append(f"| `{target_id}` | {pct:.2f}% |")

    lines.extend(
        [
            "",
            "### Comparison to the drives system's documented monoculture",
            "",
            f"- Drives system, pre-fix (documented, `orion/autonomy/drives_and_autonomy_"
            f"retrospective.md` ~line 177): `dominant_drive=relational` in "
            f"**{DRIVES_DOMINANT_SHARE_PRE_FIX * 100:.0f}%** of ticks.",
            f"- Drives system, post O1/O2/O3 fix (same file, ~line 267): top dominant-drive "
            f"share **{DRIVES_DOMINANT_SHARE_POST_FIX * 100:.2f}%**.",
            f"- Field-attention top-1-winner share over full real history here: "
            f"**{top_share * 100:.2f}%** (`{full_history_top1.most_common(1)[0][0] if full_history_top1 else 'n/a'}`).",
            "",
        ]
    )
    if full_history_top1 and top_share >= DRIVES_DOMINANT_SHARE_PRE_FIX:
        lines.append(
            "**Honest finding: this is NOT meaningfully different from the drives system's "
            "known failure pattern -- if anything it is worse.** The single highest-salience "
            "target wins essentially every tick, at or above the drives system's *pre-fix* "
            "96% monoculture share, and well above its *post-fix* ~32% share. This is exactly "
            "the pathology the baseline design's own Missing Question 1 named as the risk to "
            "check for (\"'resource_pressure always wins because it's noisiest,' the exact "
            "96%-dominant-drive monoculture pathology already found once\") -- and it is "
            "present here too, just for a different target (a near-constant-perturbation "
            "system counter and/or the primary host node/outbound-capability channel, "
            "depending on which target universe is inspected -- see the per-target table "
            "above)."
        )
    elif full_history_top1 and top_share >= DRIVES_DOMINANT_SHARE_POST_FIX:
        lines.append(
            "This is roughly in the same range as the drives system's *post-fix* concentration "
            "-- not clearly better, not the pre-fix pathology either."
        )
    elif full_history_top1:
        lines.append(
            "This is meaningfully less concentrated than either drives-system figure -- some "
            "real evidence of differentiation in which target wins top salience over time."
        )
    else:
        lines.append("No full-history data available to compute this comparison.")
    lines.append("")

    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run(
    window_hours: float = DEFAULT_WINDOW_HOURS,
    gap_hours: float = DEFAULT_GAP_HOURS,
    corr_threshold: float = CORR_CLUSTER_THRESHOLD,
) -> int:
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

    min_ts, max_ts, total_rows = fetch_frame_time_range(conn)
    progress.emit("time range fetched", percent=5.0, processed=total_rows, total=total_rows)

    if min_ts is None or max_ts is None or total_rows < MIN_TOTAL_ROWS:
        caveats.append(
            f"insufficient real substrate_attention_frames history (rows={total_rows}, "
            f"min_required={MIN_TOTAL_ROWS}); cannot run a meaningful probe"
        )
        try:
            conn.close()
        except Exception:
            pass
        progress.close()
        report = render_report(
            global_window_label="n/a (insufficient data)",
            total_rows=total_rows, windows=None, window_a_n=0, window_b_n=0, target_ids=[],
            full_target_ids=[],
            corr_a={}, corr_b={}, clusters_a=[], clusters_b=[], edges_a=set(), edges_b=set(),
            jaccard=None, corr_of_corr=None, n_common_pairs=0,
            classification="INCONCLUSIVE_INSUFFICIENT_PAIRS", full_history_top1=Counter(),
            full_history_n=0, full_history_membership={}, caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    global_label = f"{min_ts.isoformat()} -> {max_ts.isoformat()} ({(max_ts - min_ts).total_seconds() / 3600.0:.1f}h)"

    all_rows, truncated = fetch_target_rows(conn, min_ts, max_ts, MAX_ROWS)
    if truncated:
        caveats.append(
            f"full-history rows truncated at MAX_ROWS={MAX_ROWS} -- the fetch is ORDER BY "
            "generated_at ASC, so truncation keeps the OLDEST rows and drops the newest, "
            "which would disproportionately starve window B (the end-anchored/'newest' "
            "window) rather than shrinking both windows evenly. Not triggered at current "
            "real data volume (well under MAX_ROWS as of this run), but treat window B's "
            "numbers as suspect if this caveat is ever present."
        )
    progress.emit("full history fetched", percent=25.0, processed=len(all_rows), total=len(all_rows))

    try:
        conn.close()
    except Exception:
        pass

    all_timestamps = [ts for ts, _, _ in all_rows]
    all_raw_maps = [extract_target_salience_map(d, c) for _, d, c in all_rows]
    full_target_ids = build_target_universe(all_raw_maps)
    full_history_top1 = top1_winner_distribution(all_raw_maps)
    full_history_membership = membership_frequency(all_raw_maps, full_target_ids)
    progress.emit("full-history stats computed", percent=40.0, processed=len(all_raw_maps), total=len(all_raw_maps))

    windows = choose_windows(min_ts, max_ts, window_hours, gap_hours)

    if windows is None:
        caveats.append(
            f"real data span ({(max_ts - min_ts).total_seconds() / 3600.0:.1f}h) too short "
            f"for two non-overlapping windows >= {MIN_WINDOW_HOURS:g}h each"
        )
        write_ticks_csv(CSV_PATH, all_raw_maps, all_timestamps, full_target_ids)
        report = render_report(
            global_window_label=global_label,
            total_rows=total_rows, windows=None, window_a_n=0, window_b_n=0,
            target_ids=full_target_ids, full_target_ids=full_target_ids,
            corr_a={}, corr_b={}, clusters_a=[], clusters_b=[],
            edges_a=set(), edges_b=set(), jaccard=None, corr_of_corr=None, n_common_pairs=0,
            classification="INCONCLUSIVE_INSUFFICIENT_PAIRS", full_history_top1=full_history_top1,
            full_history_n=len(all_raw_maps), full_history_membership=full_history_membership,
            caveats=caveats,
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        progress.close()
        print(report)
        return 2

    win_a, win_b = windows
    maps_a, maps_b = partition_by_window(all_timestamps, all_raw_maps, win_a, win_b)
    progress.emit("windows partitioned", percent=55.0, processed=len(maps_a) + len(maps_b), total=len(all_raw_maps))

    if len(maps_a) < MIN_WINDOW_ROWS:
        caveats.append(f"window A has only {len(maps_a)} rows (< MIN_WINDOW_ROWS={MIN_WINDOW_ROWS})")
    if len(maps_b) < MIN_WINDOW_ROWS:
        caveats.append(f"window B has only {len(maps_b)} rows (< MIN_WINDOW_ROWS={MIN_WINDOW_ROWS})")

    shared_target_ids = build_target_universe(maps_a + maps_b)

    series_a = align_series(maps_a, shared_target_ids)
    series_b = align_series(maps_b, shared_target_ids)
    corr_a = compute_correlation_matrix(shared_target_ids, series_a)
    corr_b = compute_correlation_matrix(shared_target_ids, series_b)
    progress.emit("correlation matrices computed", percent=75.0, processed=len(corr_a), total=len(corr_a))

    clusters_a = cluster_by_correlation(shared_target_ids, corr_a, corr_threshold)
    clusters_b = cluster_by_correlation(shared_target_ids, corr_b, corr_threshold)
    edges_a = edge_set_at_threshold(corr_a, corr_threshold)
    edges_b = edge_set_at_threshold(corr_b, corr_threshold)
    jaccard = jaccard_similarity(edges_a, edges_b)
    corr_of_corr, n_common_pairs = correlation_of_correlations(corr_a, corr_b)
    classification = classify_similarity(corr_of_corr, jaccard, n_common_pairs)
    progress.emit("similarity computed", percent=90.0, processed=1, total=1)

    write_ticks_csv(CSV_PATH, all_raw_maps, all_timestamps, full_target_ids)
    report = render_report(
        global_window_label=global_label,
        total_rows=total_rows, windows=windows, window_a_n=len(maps_a), window_b_n=len(maps_b),
        target_ids=shared_target_ids, full_target_ids=full_target_ids,
        corr_a=corr_a, corr_b=corr_b, clusters_a=clusters_a,
        clusters_b=clusters_b, edges_a=edges_a, edges_b=edges_b, jaccard=jaccard,
        corr_of_corr=corr_of_corr, n_common_pairs=n_common_pairs, classification=classification,
        full_history_top1=full_history_top1, full_history_n=len(all_raw_maps),
        full_history_membership=full_history_membership, caveats=caveats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=1, total=1)
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only emergent-clustering probe over real field-attention history.")
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"size of each of the two comparison windows in hours (default {DEFAULT_WINDOW_HOURS})",
    )
    parser.add_argument(
        "--gap-hours", type=float, default=DEFAULT_GAP_HOURS,
        help=f"untouched real history between the two windows, in hours (default {DEFAULT_GAP_HOURS})",
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=CORR_CLUSTER_THRESHOLD,
        help=f"correlation threshold for clustering an edge/pair together (default {CORR_CLUSTER_THRESHOLD})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.window_hours, args.gap_hours, args.corr_threshold)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
