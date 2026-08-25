"""ONE analysis, FOUR inputs -> a self-study journal entry.

`skills.self_study.analyze.v1` reads a window of one already-stored telemetry
source, contrasts it against the window immediately before it, and writes a
journal entry ONLY when a disclosed notability rule fires. See
`orion/schemas/self_study_analysis.py` for why this is one verb rather than
four.

WHAT THIS IS NOT. Not a new metric, not a new signal, not a new cognition
input. Every number below is a read-only summary of rows some other producer
already wrote, and nothing here feeds field pressure, proposal scoring, or any
model. The only output is an append-only journal entry on the existing
`journal.entry.write.v1` channel, persisted by orion-sql-writer into
`journal_entries` the same way the other 34k entries are.

THE ANTI-SPAM GATE IS THE POINT. `journal_entries` already holds 32,991
`metacog` digests. An action that writes an entry every time it runs would be
digest #4, not cognition. So:

  * `_evaluate_rules` must fire, or nothing is written (`skipped_not_notable`).
  * An identical finding-set for the same source inside the cooldown is not
    written twice (`skipped_recently_journaled`), keyed on the journal's own
    indexed `source_ref` column rather than a new table.

THE RULES ARE SHARED, NOT PER-SOURCE. `_evaluate_rules` operates on two
`SourceWindow`s and knows nothing about vision or affect or crystallizations.
That is what makes "four inputs" true rather than decorative: adding a fifth
source is a `SourceSpec` entry, not a fifth copy of the analysis.

SQL SAFETY. Table and column names are interpolated into the SELECT, but every
one of them is a literal in `SOURCE_SPECS` below -- `source` is validated
against that dict's keys before any string is built, and the window bounds are
bound parameters. There is no path from caller input to an identifier.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import create_engine, text

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler.schemas import JournalEntryWriteV1
from orion.schemas.self_study import SelfWritebackStatusV1
from orion.schemas.self_study_analysis import (
    ANALYSIS_SOURCES,
    AnalysisFindingV1,
    AnalysisMetricV1,
    SelfStudyAnalysisResultV1,
)

logger = logging.getLogger("orion.cortex_exec.self_study_analysis")

# The same already-registered channel every other journal producer uses
# (orion/bus/channels.yaml:1940). No new channel: orion-sql-writer already
# maps `journal.entry.write.v1` -> JournalEntrySQL -> `journal_entries`.
JOURNAL_WRITE_CHANNEL = "orion:journal:write"
_AUTHOR = "orion"
_SOURCE_KIND = "self_study"

# Per-connection statement timeout. These are four small windowed SELECTs over
# tables whose largest is ~38k rows; the same belt-and-suspenders shape
# metacog_trend_reader.py uses, set higher because this runs on the dispatch
# path (own RPC budget) rather than inside a chat turn's latency budget.
_QUERY_STATEMENT_TIMEOUT_MS = 4000

# Hard row ceiling per window. A runaway producer must not turn one journal
# action into an unbounded fetch. Reaching it is itself reported, so a
# truncated window is never silently read as the whole window.
_MAX_ROWS_PER_WINDOW = 20000

DEFAULT_WINDOW_HOURS = 6.0
MIN_WINDOW_HOURS = 0.5
MAX_WINDOW_HOURS = 168.0

# --- Notability bars -------------------------------------------------------
#
# DISCLOSED, UNCALIBRATED STARTING VALUES. Not fitted to anything; chosen to be
# obviously-crossable-but-not-always, and every one of them is reported next to
# the number that did or did not cross it, so a reader can see the bar rather
# than trust it. The failure mode these guard is the one already recorded in
# this repo's own history: a short window makes distribution statistics into
# artifacts, so every distribution rule below refuses to fire until the
# BASELINE window has at least `MIN_BASELINE_ROWS` rows to compute against.
MIN_BASELINE_ROWS = 5
# Ratio band outside which recent volume counts as a shift. 2x / half is a
# coarse bar on purpose -- it exists to catch a producer changing regime, not
# to detect small drift, which this action has no power to distinguish from
# noise at these row counts.
VOLUME_RATIO_LOW = 0.5
VOLUME_RATIO_HIGH = 2.0
# A category (event_type, domain, crystallization kind, ...) has to reach this
# many rows before its appearance or disappearance is worth a sentence. Below
# it, one stray row would fire the rule every window.
MIN_CATEGORY_ROWS = 3
# Mean shift is measured in BASELINE standard deviations, so the bar adapts to
# how noisy the quantity already is instead of being an absolute number that
# means different things per column. 1.0 sigma is deliberately weak evidence
# for a single comparison -- it is a "worth writing down", not a test result,
# and the journal body says so.
MEAN_SHIFT_SIGMAS = 1.0
# Longest stretch with no observation inside the recent window before it counts
# as a gap. Anchored on a real incident class rather than taste: the 2026-08-21
# vision outage ran 21 hours with a healthy container and green logs
# (project_vision_liveness_alert_shipped_2026-08-21), and any of these four
# producers can fail the same silent way.
GAP_MINUTES = 120.0
# ...and it must ALSO be this many times the producer's own baseline-window
# largest gap. See the rule body for the live-smoke finding that added this.
GAP_RELATIVE_MULTIPLE = 2.0

ALL_RULES: tuple[str, ...] = (
    "producer_stalled",
    "observation_gap",
    "volume_shift",
    "new_category",
    "lost_category",
    "mean_shift",
)


@dataclass(frozen=True)
class SourceSpec:
    """Everything that differs between the four inputs. Note there is no
    per-source analysis code -- only which rows to read and what to call
    them."""

    label: str
    table: str
    time_column: str
    # True when `time_column` is when the thing HAPPENED, False when it is when
    # the row was WRITTEN. Recorded and rendered because those are different
    # clocks and a reader cannot re-derive which one a window was cut on.
    time_is_occurrence: bool
    numeric_columns: tuple[str, ...]
    category_columns: tuple[str, ...]
    # Plain-language statement of what the rows actually measure. Written to
    # the journal verbatim so an entry never overclaims its own source -- the
    # affect log is swear/message frequency, not emotion, and says so.
    what_it_measures: str


SOURCE_SPECS: Mapping[str, SourceSpec] = {
    "concept_induction": SourceSpec(
        label="concept induction",
        table="memory_crystallizations",
        time_column="created_at",
        time_is_occurrence=False,
        numeric_columns=("salience",),
        category_columns=("kind", "status"),
        what_it_measures=(
            "Crystallizations induced from memory -- one row per concept the "
            "induction lane proposed, with the kind it was filed under and "
            "whether it was accepted, is still proposed, or was rejected."
        ),
    ),
    "vision_events": SourceSpec(
        label="vision events",
        table="vision_events",
        time_column="created_at",
        time_is_occurrence=False,
        numeric_columns=("confidence", "salience"),
        category_columns=("event_type",),
        what_it_measures=(
            "Events the vision pipeline emitted -- one row per recognised "
            "event, with the model's own confidence and the pipeline's "
            "salience score. Cut on write time; the table has no separate "
            "occurrence-time column."
        ),
    ),
    "affective_state": SourceSpec(
        label="affective state",
        table="juniper_affective_state_log",
        time_column="observed_at",
        time_is_occurrence=True,
        numeric_columns=("swear_frequency", "message_count", "word_count"),
        category_columns=("cold_start",),
        what_it_measures=(
            "A coarse agitation proxy over Juniper's recent messages: message "
            "volume, word count, and swear frequency per rolling window. This "
            "is NOT an emotion read and must not be described as one -- the "
            "facial/vocal affect lane (orion:affectgpt:assessment) keeps no "
            "history to analyse, only a 1h Redis key."
        ),
    ),
    "cocreation_signals": SourceSpec(
        label="co-creation signals",
        table="substrate_codebase_delta_log",
        time_column="observed_at",
        time_is_occurrence=True,
        numeric_columns=("score",),
        category_columns=("domain",),
        what_it_measures=(
            "Codebase-delta signals from orion-cocreation-signals -- one row "
            "per observed change event (PR lifecycle, git delta, graph "
            "delta), with the producer's own score for it."
        ),
    ),
}


@dataclass
class SourceWindow:
    """One window's worth of rows, reduced to exactly what the shared rules
    need. Deliberately source-agnostic."""

    rows: int = 0
    truncated: bool = False
    numeric: dict[str, list[float]] = field(default_factory=dict)
    categories: dict[str, dict[str, int]] = field(default_factory=dict)
    timestamps: list[datetime] = field(default_factory=list)


# --- statistics ------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _stdev(values: Sequence[float]) -> float | None:
    """Sample standard deviation, or None when it cannot be computed.

    Returns None (not 0.0) for n < 2: "cannot measure spread" and "measured
    zero spread" are different, and `mean_shift` must not fire on the first."""
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _largest_gap_minutes(timestamps: Sequence[datetime], *, since: datetime, until: datetime) -> float | None:
    """Longest observation-free stretch inside [since, until].

    The window EDGES count: a producer that died 3h before the window closed
    leaves its gap at the end, where a naive consecutive-pairs scan would miss
    it entirely. Returns None when there is nothing to measure."""
    if until <= since:
        return None
    if not timestamps:
        # An empty window is one gap the width of the whole window. That is a
        # real reading, and `producer_stalled` is the rule that owns it.
        return (until - since).total_seconds() / 60.0
    ordered = sorted(timestamps)
    boundaries = [since, *ordered, until]
    return max(
        (boundaries[i + 1] - boundaries[i]).total_seconds() / 60.0
        for i in range(len(boundaries) - 1)
    )


# --- the shared rules ------------------------------------------------------


def evaluate_rules(
    *,
    recent: SourceWindow,
    baseline: SourceWindow,
    recent_since: datetime,
    recent_until: datetime,
) -> tuple[list[AnalysisFindingV1], list[str]]:
    """Run every rule in `ALL_RULES` and return (fired, not_fired).

    Pure. Knows nothing about which source produced the windows -- that is the
    whole point of `SourceWindow` being the boundary."""
    findings: list[AnalysisFindingV1] = []
    fired: set[str] = set()

    def fire(finding: AnalysisFindingV1) -> None:
        findings.append(finding)
        fired.add(finding.rule)

    # 1. producer_stalled -- the loudest and cheapest failure to catch.
    if recent.rows == 0 and baseline.rows > 0:
        fire(
            AnalysisFindingV1(
                rule="producer_stalled",
                detail=(
                    f"No rows at all in the recent window, against {baseline.rows} "
                    "in the window before it. The producer stopped."
                ),
                metric="rows",
                recent=0.0,
                baseline=float(baseline.rows),
            )
        )

    # 2. observation_gap -- a stall INSIDE the window, which a row count alone
    # cannot see (the 2026-08-21 vision outage's failure shape).
    #
    # The bar is RELATIVE to the producer's own normal, not just absolute.
    # Caught by the first live smoke: concept_induction legitimately emits ~2
    # rows/day, so an absolute 120-min bar fired on it every single window
    # forever -- true, useless, and exactly the "cognition-shaped output with
    # no cognitive substance" this action is supposed to refuse. A gap is only
    # news if this producer does not normally go quiet that long, so it must
    # clear BOTH the absolute floor and 2x the baseline window's own largest
    # gap, and the baseline has to be substantial enough to define a normal.
    gap = _largest_gap_minutes(recent.timestamps, since=recent_since, until=recent_until)
    baseline_gap = _largest_gap_minutes(
        baseline.timestamps, since=recent_since - (recent_until - recent_since), until=recent_since
    )
    if (
        gap is not None
        and recent.rows > 0
        and baseline.rows >= MIN_BASELINE_ROWS
        and baseline_gap is not None
    ):
        relative_bar = max(GAP_MINUTES, GAP_RELATIVE_MULTIPLE * baseline_gap)
        if gap >= relative_bar:
            fire(
                AnalysisFindingV1(
                    rule="observation_gap",
                    detail=(
                        f"Longest stretch with no observation was {gap:.0f} min, "
                        f"against a bar of {relative_bar:.0f} min "
                        f"(max of the {GAP_MINUTES:.0f} min floor and "
                        f"{GAP_RELATIVE_MULTIPLE:g}x this producer's own "
                        f"{baseline_gap:.0f} min baseline gap). The window did "
                        f"otherwise carry {recent.rows} rows."
                    ),
                    metric="gap_minutes",
                    recent=round(gap, 1),
                    baseline=round(baseline_gap, 1),
                )
            )

    # 3. volume_shift -- regime change in how much the producer emits.
    if baseline.rows >= MIN_BASELINE_ROWS and recent.rows > 0:
        ratio = recent.rows / baseline.rows
        if ratio <= VOLUME_RATIO_LOW or ratio >= VOLUME_RATIO_HIGH:
            fire(
                AnalysisFindingV1(
                    rule="volume_shift",
                    detail=(
                        f"{recent.rows} rows against {baseline.rows} the window "
                        f"before ({ratio:.2f}x; bar: outside "
                        f"{VOLUME_RATIO_LOW:g}x-{VOLUME_RATIO_HIGH:g}x)."
                    ),
                    metric="rows",
                    recent=float(recent.rows),
                    baseline=float(baseline.rows),
                )
            )

    # 4/5. new_category / lost_category.
    #
    # Both require a baseline substantial enough for "absent" to mean
    # something. Also caught by the first live smoke: against an EMPTY
    # baseline window every category in the recent window is trivially "new",
    # which fired two findings that carried no information at all.
    categorical_baseline_ok = baseline.rows >= MIN_BASELINE_ROWS
    for dimension in sorted(set(recent.categories) | set(baseline.categories)):
        if not categorical_baseline_ok:
            break
        recent_counts = recent.categories.get(dimension, {})
        baseline_counts = baseline.categories.get(dimension, {})
        for label in sorted(recent_counts):
            count = recent_counts[label]
            if count >= MIN_CATEGORY_ROWS and label not in baseline_counts:
                fire(
                    AnalysisFindingV1(
                        rule="new_category",
                        detail=(
                            f"{dimension}={label!r} appears {count}x in the "
                            "recent window and not at all in the one before "
                            f"(bar: {MIN_CATEGORY_ROWS} rows)."
                        ),
                        metric=f"{dimension}:{label}",
                        recent=float(count),
                        baseline=0.0,
                    )
                )
        for label in sorted(baseline_counts):
            count = baseline_counts[label]
            # A category cannot be "lost" in a window that saw nothing at all;
            # that is `producer_stalled`, which already owns the reading.
            if recent.rows < MIN_BASELINE_ROWS:
                continue
            if count >= MIN_CATEGORY_ROWS and label not in recent_counts:
                fire(
                    AnalysisFindingV1(
                        rule="lost_category",
                        detail=(
                            f"{dimension}={label!r} was {count}x in the window "
                            "before and is absent from the recent one."
                        ),
                        metric=f"{dimension}:{label}",
                        recent=0.0,
                        baseline=float(count),
                    )
                )

    # 6. mean_shift, in BASELINE sigmas so the bar adapts per column.
    for column in sorted(set(recent.numeric) | set(baseline.numeric)):
        recent_values = recent.numeric.get(column, [])
        baseline_values = baseline.numeric.get(column, [])
        if len(baseline_values) < MIN_BASELINE_ROWS or len(recent_values) < MIN_BASELINE_ROWS:
            continue
        recent_mean = _mean(recent_values)
        baseline_mean = _mean(baseline_values)
        sigma = _stdev(baseline_values)
        if recent_mean is None or baseline_mean is None or not sigma:
            continue
        delta_sigmas = abs(recent_mean - baseline_mean) / sigma
        if delta_sigmas >= MEAN_SHIFT_SIGMAS:
            fire(
                AnalysisFindingV1(
                    rule="mean_shift",
                    detail=(
                        f"mean {column} moved {recent_mean:.4g} vs "
                        f"{baseline_mean:.4g} = {delta_sigmas:.2f} baseline "
                        f"sigma (bar: {MEAN_SHIFT_SIGMAS:g}; baseline sigma "
                        f"{sigma:.4g} over n={len(baseline_values)})."
                    ),
                    metric=column,
                    recent=round(recent_mean, 6),
                    baseline=round(baseline_mean, 6),
                )
            )

    not_fired = [rule for rule in ALL_RULES if rule not in fired]
    return findings, not_fired


def finding_digest(source: str, findings: Sequence[AnalysisFindingV1]) -> str:
    """Stable digest of WHICH rules fired on WHICH metrics -- not of their
    values. Two consecutive windows reporting the same gap on the same producer
    are the same news; the cooldown should suppress the second even though the
    minute counts differ."""
    parts = sorted(f"{f.rule}|{f.metric or ''}" for f in findings)
    raw = "\n".join([source, *parts])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# --- database --------------------------------------------------------------

_ENGINE: Any = None
_ENGINE_URL: str | None = None


def _envelope_correlation_id(raw: str) -> str:
    """BaseEnvelope wants a UUID-shaped correlation id; a dispatch correlation
    id is not always one. Same coercion self_study.py already applies."""
    try:
        return str(UUID(str(raw)))
    except Exception:  # noqa: BLE001
        return str(uuid5(NAMESPACE_URL, str(raw)))


def _dsn() -> str:
    # Same conjourney instance and same fallback chain every other cortex-exec
    # Postgres reader in this service uses. No fifth DB-URL key for one DB.
    # No new env key on purpose. All three below are already set in every
    # cortex-exec container (verified live 2026-08-25 with `docker exec ... env`),
    # so a fourth would be one more surface to drift. SUBSTRATE_FELT_STATE first
    # because POSTGRES_URI points at `orion-sql-db`, a hostname this service's
    # own .env_example records as historically unresolvable here.
    return (
        os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL", "").strip()
        or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", "").strip()
        or os.getenv("POSTGRES_URI", "").strip()
    )


def _get_engine() -> Any:
    global _ENGINE, _ENGINE_URL
    url = _dsn()
    if not url:
        return None
    if _ENGINE is None or _ENGINE_URL != url:
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"options": f"-c statement_timeout={_QUERY_STATEMENT_TIMEOUT_MS}"},
        )
        _ENGINE_URL = url
    return _ENGINE


def reset_engine_for_tests() -> None:
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_datetime(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def fetch_window(engine: Any, spec: SourceSpec, *, since: datetime, until: datetime) -> SourceWindow:
    """Read one window and reduce it to a `SourceWindow`.

    Identifiers come from `spec`, which is a frozen literal in `SOURCE_SPECS`;
    only the bounds are parameters."""
    columns = [spec.time_column, *spec.numeric_columns]
    selected = ", ".join(columns) + "".join(
        f", CAST({c} AS text) AS {c}" for c in spec.category_columns
    )
    sql = text(
        f"SELECT {selected} FROM {spec.table} "
        f"WHERE {spec.time_column} >= :since AND {spec.time_column} < :until "
        f"ORDER BY {spec.time_column} LIMIT :limit"
    )
    window = SourceWindow()
    with engine.connect() as conn:
        result = conn.execute(
            sql, {"since": since, "until": until, "limit": _MAX_ROWS_PER_WINDOW}
        )
        for row in result.mappings():
            window.rows += 1
            stamp = _as_datetime(row.get(spec.time_column))
            if stamp is not None:
                window.timestamps.append(stamp)
            for column in spec.numeric_columns:
                value = _as_float(row.get(column))
                if value is not None:
                    window.numeric.setdefault(column, []).append(value)
            for column in spec.category_columns:
                raw = row.get(column)
                label = "null" if raw is None else str(raw)
                bucket = window.categories.setdefault(column, {})
                bucket[label] = bucket.get(label, 0) + 1
    window.truncated = window.rows >= _MAX_ROWS_PER_WINDOW
    return window


def select_least_recently_analysed(engine: Any, *, lookback_days: int = 30) -> str:
    """Pick which of the four inputs to study this run.

    THE DEFAULT, not a fallback. The dispatch route deliberately does NOT pin a
    source: there is one action ("study yourself"), and which lens it uses is a
    scheduling question, not a proposal-arena question. Four near-identical
    templates competing for the same five dispatch slots would have starved
    four existing templates to say the same thing four ways -- the
    keyword-cathedral shape in template form.

    "Most overdue" rather than round-robin or a hash: it is just as
    deterministic, it self-heals after any source is unreachable for a while,
    and it spends the action on the lens Orion has looked at least recently.
    Ties (including the all-never-analysed cold start) break on
    `ANALYSIS_SOURCES` order, so the first run of a fresh deployment is
    predictable rather than arbitrary.
    """
    sql = text(
        "SELECT split_part(source_ref, ':', 1) AS src, MAX(created_at) AS last_at "
        "FROM journal_entries "
        "WHERE source_kind = :source_kind AND created_at >= :since "
        "GROUP BY 1"
    )
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    last_by_source: dict[str, datetime] = {}
    with engine.connect() as conn:
        for row in conn.execute(sql, {"source_kind": _SOURCE_KIND, "since": since}).mappings():
            stamp = _as_datetime(row.get("last_at"))
            src = str(row.get("src") or "")
            if src in SOURCE_SPECS and stamp is not None:
                last_by_source[src] = stamp
    # None sorts first: never-analysed beats any real timestamp.
    return min(
        ANALYSIS_SOURCES,
        key=lambda name: (
            last_by_source.get(name) is not None,
            last_by_source.get(name) or since,
            ANALYSIS_SOURCES.index(name),
        ),
    )


def recently_journaled(engine: Any, *, source_ref: str, since: datetime) -> bool:
    """True when an entry with this exact source_ref already exists inside the
    cooldown. Uses `journal_entries`' own columns -- no new table, and the
    dedup key is visible in the artifact it dedups."""
    sql = text(
        "SELECT 1 FROM journal_entries "
        "WHERE source_kind = :source_kind AND source_ref = :source_ref "
        "AND created_at >= :since LIMIT 1"
    )
    with engine.connect() as conn:
        row = conn.execute(
            sql, {"source_kind": _SOURCE_KIND, "source_ref": source_ref, "since": since}
        ).first()
    return row is not None


# --- metrics + journal body ------------------------------------------------


def build_metrics(spec: SourceSpec, recent: SourceWindow, baseline: SourceWindow) -> list[AnalysisMetricV1]:
    metrics = [
        AnalysisMetricV1(
            name="rows",
            recent=float(recent.rows),
            baseline=float(baseline.rows),
            unit="count",
        )
    ]
    for column in spec.numeric_columns:
        metrics.append(
            AnalysisMetricV1(
                name=f"{column}_mean",
                recent=_mean(recent.numeric.get(column, [])),
                baseline=_mean(baseline.numeric.get(column, [])),
                unit="mean",
            )
        )
    for column in spec.category_columns:
        metrics.append(
            AnalysisMetricV1(
                name=f"{column}_distinct",
                recent=float(len(recent.categories.get(column, {}))),
                baseline=float(len(baseline.categories.get(column, {}))),
                unit="count",
            )
        )
    return metrics


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def build_analysis_journal_entry(
    *,
    spec: SourceSpec,
    result: SelfStudyAnalysisResultV1,
    recent: SourceWindow,
    created_at: datetime | None = None,
) -> JournalEntryWriteV1:
    ts = created_at or datetime.now(timezone.utc)
    clock = "occurrence time" if spec.time_is_occurrence else "write time"
    lines = [
        f"Self-study analysis of {spec.label}: the last {result.window_hours:g}h "
        f"against the {result.window_hours:g}h before it.",
        "",
        f"Source: {spec.table}.{spec.time_column} ({clock}). "
        f"{result.recent_rows} rows recent, {result.baseline_rows} baseline.",
        f"What these rows are: {spec.what_it_measures}",
    ]
    if recent.truncated:
        lines.append(
            f"NOTE: the recent window hit the {_MAX_ROWS_PER_WINDOW}-row read "
            "ceiling, so every number below is over a truncated window."
        )
    lines += ["", "What fired:"]
    for finding in result.findings:
        lines.append(f"- {finding.rule}: {finding.detail}")
    lines += ["", "Measured:"]
    for metric in result.metrics:
        lines.append(
            f"- {metric.name}: {_fmt(metric.recent)} (baseline {_fmt(metric.baseline)})"
        )
    if result.rules_not_fired:
        lines += [
            "",
            "Checked and did not fire: " + ", ".join(result.rules_not_fired) + ".",
        ]
    lines += [
        "",
        "This is a read-only window contrast against disclosed bars, not a "
        "test result. It says something changed enough to be worth writing "
        "down; it does not say why, and nothing here feeds back into any "
        "score.",
    ]
    return JournalEntryWriteV1(
        created_at=ts,
        author=_AUTHOR,
        mode="manual",
        title=f"Self-study analysis: {spec.label}",
        body="\n".join(lines),
        source_kind=_SOURCE_KIND,
        source_ref=f"{result.source}:{result.finding_digest}",
        correlation_id=None,
    )


# --- runner ----------------------------------------------------------------


def _clamp_window_hours(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_WINDOW_HOURS
    if not math.isfinite(value):
        return DEFAULT_WINDOW_HOURS
    return max(MIN_WINDOW_HOURS, min(MAX_WINDOW_HOURS, value))


def _unavailable(
    *, source: str, reason: str, run_id: str, window_hours: float, now: datetime, time_column: str
) -> SelfStudyAnalysisResultV1:
    delta = timedelta(hours=window_hours)
    return SelfStudyAnalysisResultV1(
        run_id=run_id,
        source=source,  # type: ignore[arg-type]
        status="unavailable",
        window_hours=window_hours,
        time_column=time_column,
        recent_since=now - delta,
        recent_until=now,
        baseline_since=now - (2 * delta),
        unavailable_reason=reason,
    )


async def run_self_study_analysis(
    *,
    bus: Any | None,
    source_ref: ServiceRef,
    # None (the live default) means "pick the most overdue source" --
    # see `select_least_recently_analysed`.
    source: str | None = None,
    window_hours: Any = None,
    correlation_id: str,
    engine: Any | None = None,
    now: datetime | None = None,
) -> SelfStudyAnalysisResultV1:
    run_id = str(uuid4())
    hours = _clamp_window_hours(window_hours if window_hours is not None else DEFAULT_WINDOW_HOURS)
    moment = now or datetime.now(timezone.utc)

    db = engine if engine is not None else _get_engine()
    if db is None:
        return _unavailable(
            source="concept_induction",
            reason="database_url_unset",
            run_id=run_id,
            window_hours=hours,
            now=moment,
            time_column="",
        )

    if source is None or not str(source).strip():
        try:
            source = select_least_recently_analysed(db)
        except Exception as exc:  # noqa: BLE001
            # Selection failing is not a reason to skip the run entirely; fall
            # back to the first source rather than doing nothing, and say so.
            logger.warning(
                "self_study_analysis_source_selection_failed corr=%s err=%s", correlation_id, exc
            )
            source = ANALYSIS_SOURCES[0]

    spec = SOURCE_SPECS.get(str(source))
    if spec is None:
        return _unavailable(
            # `source` is not a valid literal here by definition, so the result
            # is reported under a fixed placeholder rather than echoed back.
            source="concept_induction",
            reason=f"unknown_source:{source!r} (known: {', '.join(ANALYSIS_SOURCES)})",
            run_id=run_id,
            window_hours=hours,
            now=moment,
            time_column="",
        )

    delta = timedelta(hours=hours)
    recent_since = moment - delta
    baseline_since = moment - (2 * delta)
    try:
        recent = fetch_window(db, spec, since=recent_since, until=moment)
        baseline = fetch_window(db, spec, since=baseline_since, until=recent_since)
    except Exception as exc:  # noqa: BLE001 -- fail-open, reported not raised
        logger.warning(
            "self_study_analysis_query_failed source=%s corr=%s err=%s",
            source,
            correlation_id,
            exc,
        )
        return _unavailable(
            source=source,
            reason=f"query_failed:{type(exc).__name__}",
            run_id=run_id,
            window_hours=hours,
            now=moment,
            time_column=spec.time_column,
        )

    findings, not_fired = evaluate_rules(
        recent=recent, baseline=baseline, recent_since=recent_since, recent_until=moment
    )
    result = SelfStudyAnalysisResultV1(
        run_id=run_id,
        source=source,  # type: ignore[arg-type]
        status="skipped_not_notable",
        window_hours=hours,
        time_column=spec.time_column,
        recent_since=recent_since,
        recent_until=moment,
        baseline_since=baseline_since,
        recent_rows=recent.rows,
        baseline_rows=baseline.rows,
        metrics=build_metrics(spec, recent, baseline),
        findings=findings,
        rules_not_fired=not_fired,
        finding_digest=finding_digest(source, findings) if findings else None,
    )
    if not findings:
        return result

    entry_source_ref = f"{result.source}:{result.finding_digest}"
    try:
        if recently_journaled(db, source_ref=entry_source_ref, since=moment - delta):
            result.status = "skipped_recently_journaled"
            return result
    except Exception as exc:  # noqa: BLE001
        # A dedup lookup that cannot run must not silently become "write it".
        # Skipping a real entry is recoverable; a spam loop against a broken
        # journal table is not.
        logger.warning(
            "self_study_analysis_dedup_failed source=%s corr=%s err=%s",
            source,
            correlation_id,
            exc,
        )
        result.status = "unavailable"
        result.unavailable_reason = f"dedup_failed:{type(exc).__name__}"
        return result

    entry = build_analysis_journal_entry(spec=spec, result=result, recent=recent, created_at=moment)
    entry.correlation_id = correlation_id
    result.journal_entry = entry

    if bus is None:
        result.status = "journal_failed"
        result.journal_write = SelfWritebackStatusV1(
            target="journal",
            status="skipped",
            authoritative=False,
            channel=JOURNAL_WRITE_CHANNEL,
            idempotency_key=entry.entry_id,
            append_only=True,
            detail="missing_bus",
        )
        return result

    envelope = BaseEnvelope(
        kind="journal.entry.write.v1",
        source=source_ref,
        correlation_id=_envelope_correlation_id(correlation_id),
        payload=entry.model_dump(mode="json"),
    )
    try:
        await bus.publish(JOURNAL_WRITE_CHANNEL, envelope)
    except Exception as exc:  # noqa: BLE001
        result.status = "journal_failed"
        result.journal_write = SelfWritebackStatusV1(
            target="journal",
            status="failed",
            authoritative=False,
            channel=JOURNAL_WRITE_CHANNEL,
            idempotency_key=entry.entry_id,
            append_only=True,
            detail=str(exc),
        )
        return result

    result.status = "journaled"
    result.journal_write = SelfWritebackStatusV1(
        target="journal",
        status="written",
        authoritative=False,
        channel=JOURNAL_WRITE_CHANNEL,
        idempotency_key=entry.entry_id,
        append_only=True,
        detail="append_only_by_design",
    )
    return result
