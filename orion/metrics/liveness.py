"""Phase 5 of docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md:
computed liveness for metrics beyond field channels.

Deliberately narrow. The design doc scoped phase 5 as a per-surface decision
("where does the live sample come from") too big to solve generically in one
pass, and left it deferred. This resolves exactly two candidates.

**Correction 2026-08-20** (code review): an earlier version of this
docstring claimed "25 of 48 [inner_state URNs] have retired producers ... see
docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md's R6 section for
the full walk". Neither half of that checked out on re-verification: that
doc's R6 is an unrelated, still-open question ("can a metric express rest"),
with no such walk recorded in it; and re-running
`check_metric_lineage.py --json` live against current `main` (2026-08-20)
found 48 `inner_state` nodes is correct (15 signals + 33 scalar fields), but
only 3 signals (8 nodes total) have a retired/non-viable producer
(`self_state.v1`, `drive_state.v1`, `autonomy_state_v2`) -- not 25. The
real, verified accounting for all 15 signals:

- **Retired producer, out of scope (3):** `self_state.v1`, `drive_state.v1`,
  `autonomy_state_v2`.
- **Ruled out with a specific reason (4):** `mood_arc_corpus.v1` and
  `field_channel_corpus.v1` (both config-gated off by default -- see
  `orion/inner_state_registry.py`'s notes on each); `chat_stance_disposition`
  (categorical, not a numeric series); `biometrics_cluster.v1` (registry
  flags it `DUPLICATE_OF field_state.v1`).
- **Built here (2):** `attention_self_model.v1`, `l7_l11_ladder` -- see below
  for why these two specifically.
- **Not investigated in this pass (6):** `field_state.v1`,
  `field_attention_frame.v1`, `attention_broadcast_projection.v1`,
  `mood_arc_encoder.v1`, `phi_heuristic.valence`, `phi_intrinsic_reward.v1`.
  Named explicitly rather than left implicit -- absence of investigation is
  not evidence either way for these.

Why these two:

- `attention_self_model.v1` -- five scalar fields read straight off
  `substrate_attention_self_model.self_model_json`, fed into the existing
  `classify_channel_series()` unmodified. Cadence confirmed live 2026-08-19:
  ~30s/tick, 119 rows in the trailing hour.
- `l7_l11_ladder` -- five backing tables (ProposalFrameV1 -> ConsolidationV1)
  with no shared scalar to sample -- it is a pipeline, not a signal.
  Reframed as THROUGHPUT liveness: rows-per-bucket over a real window, same
  classifier. Elevated priority over other phase-5 candidates: the ladder
  carries a live mutating route (`skills.runtime.builder_prune.v1`) whose
  terminal effect is deleting host data, independent of whether anything
  downstream consumes it cognitively (see the R6 section above -- this was
  found only after a direct challenge to an earlier "no cognition consumer
  means nothing at stake" misreading).

Cadences confirmed live 2026-08-19 (500 most-recent rows, avg inter-row gap):
    substrate_proposal_frames         2.12s
    substrate_policy_decision_frames  2.12s
    substrate_feedback_frames         2.06s
    substrate_attention_frames        2.13s
    substrate_consolidation_frames    5761.49s (~96min)

The classifier math is never reimplemented here -- only the data source is
new. See `orion.field.channel_glossary.classify_channel_series` for the
verdict vocabulary (never_produced / dead / ratchet_suspect / quiet / live)
and `orion.attention.tension.liveness` for the sibling producer-liveness
classifier this deliberately does not replace or duplicate (different
question: "is this telling me anything" vs "is anyone still writing this").

`classify_channel_series`'s thresholds (`LIVE_VARIANCE_THRESHOLD=0.05`,
`SUBNORMAL_CUTOFF`) are calibrated for `[0,1]`-bounded field-channel
salience/pressure values -- see that module's own header. The ladder's raw
signal is integer row counts (tens per bucket), a different domain those
thresholds were never tuned for -- a repeat of this repo's own documented
"borrowed calibrated constant doesn't transfer across domains" failure mode,
this time caught before shipping rather than after: a live sanity check
(a regression test, not eyeballing) found feeding raw counts in trips
`ratchet_suspect` on any ordinary busy-but-healthy burst, AND that scale
normalization alone does not fix it (a monotonic climb, rescaled to its own
max of 1.0, clears `LIVE_VARIANCE_THRESHOLD` almost every time regardless of
original scale). `_classify_unbounded_series()` is the actual fix: it
normalizes for the quiet/live spread boundary (`_normalize_to_unit_scale()`)
AND downgrades `ratchet_suspect` to `live`, because that verdict's real
meaning -- "a mode=add channel that should decay but never does" -- has no
equivalent for pipeline throughput, where a monotonic run across a short or
coarsely-bucketed window (the 16-bucket consolidation stage) is unremarkable.
`dead`/`never_produced` (the genuine problems) are untouched by either
change.

Read-only, real-DB, off the request path: this runs only from
`scripts/check_metric_lineage.py --metric <name>` (an operator/agent CLI
invocation), never from a service tick. Connection contract is
`orion.db_readonly.open_readonly_connection` -- the canonical version of
`scripts/analysis/_pg_readonly.py`'s read-only-session-enforcing helper,
moved to `orion/` so both `scripts/` and `orion/metrics/` can depend on it
without inverting this repo's layering (found live 2026-08-19: an earlier
draft of this module reimplemented that helper from scratch instead of
importing it, the exact duplication `_pg_readonly.py` already existed to
prevent, one layer down). Every DB failure -- connect, session-readonly
check, a query hanging after connect (bounded by `STATEMENT_TIMEOUT_MS`,
added 2026-08-20 after `CONNECT_TIMEOUT_SECONDS` alone was found to only
cover the connect phase), or a query failing mid-fetch -- degrades to
"liveness unknown", never to a fabricated verdict; query-time failures are
caught by `scripts/check_metric_lineage.py`'s `_liveness_for_nodes`, the
same place connection failures are already handled, so there is one place
callers check for "did this actually work." `resolved_host()` exposes which
host was actually tried (never credentials) so an UNKNOWN verdict is
debuggable rather than a silent mystery.

`host=localhost port=55432` is the confirmed-live default for host-run
scripts (verified 2026-08-19); the docker-network hostnames used by
in-container readers (`orion-athena-sql-db`, `orion-sql-db`) do not resolve
from the host. Env override chain is `POSTGRES_URI` / `DATABASE_URL` /
`ORION_SQL_URL`, same vocabulary as `scripts/print_recent_turn_effects.py`
and `scripts/trace_unified_turn.py` -- but not the same *priority order* as
either (they disagree with each other: the former checks `ORION_SQL_URL`
first, the latter checks `POSTGRES_URI` before `ORION_SQL_URL`). This module
checks `POSTGRES_URI` first because that is `.env_example`'s canonical,
operator-set key; `ORION_SQL_URL` is a narrower override not every script
even agrees on the precedence of.

**Known limitation, not fixed here** (review finding 2026-08-20):
`.env_example`'s own `POSTGRES_URI` is set to the docker-network hostname
`orion-sql-db`, which does not resolve from the host. If an operator/agent
shell has sourced a `.env` carrying that value (CLAUDE.md's env-sync rules
say local `.env` should track `.env_example`), `resolve_dsn()` would pick it
up before ever reaching the localhost default, and every phase-5 check would
degrade to `db_unreachable` -- honestly, not a crash or a fabricated
verdict, but silently pointed at the wrong host with no obvious reason why
(`resolved_host()` at least makes that debuggable, see above). Not currently
live-triggered: today's checked-out root `.env` has no `POSTGRES_URI` key at
all. This module shares the exact same exposure as
`scripts/print_recent_turn_effects.py` and `scripts/trace_unified_turn.py`
(both check `POSTGRES_URI` before any localhost fallback too) -- fixing it
for real needs a distinct host-reachable-Postgres env convention applied
repo-wide, out of scope for this patch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from orion.db_readonly import open_readonly_connection as _open_readonly_connection
from orion.field.channel_glossary import CLEAN_VERDICTS, classify_channel_series
from orion.metrics.lineage import MetricNode

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"
_ENV_KEYS = ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_URL")

MAX_ROWS = 50_000

# psycopg2.connect() has no default timeout -- an unreachable (not merely
# refused) host hangs the whole `--metric` CLI call indefinitely, breaking
# the "any DB failure degrades to an honest UNKNOWN verdict" promise this
# module makes everywhere else. Found live 2026-08-19 testing the failure
# path itself: a connect to a black-holed port hung past a 15s timeout.
CONNECT_TIMEOUT_SECONDS = 5

# Worst-of rollup order for a multi-series verdict (the ladder's 5 stages).
# Reuses channel_glossary.CLEAN_VERDICTS rather than inventing a fresh
# severity scale -- `live` and `quiet` are BOTH clean states there (a
# low-cadence stage reading `quiet` is not "worse" than a fast stage reading
# `live`, it is just slower). Found live 2026-08-19: an earlier version of
# this rollup ranked quiet above live and reported the ladder's overall
# verdict as QUIET when 4 of 5 stages were LIVE and only the ~96min-cadence
# consolidation stage was quiet by design -- an accurate-sounding but
# misleading "worst-of" that would have made every healthy tick look
# concerning. Only an actually unclean verdict (dead / never_produced /
# ratchet_suspect) should ever win a rollup.
_UNCLEAN_SEVERITY = {
    "never_produced": 3,
    "dead": 2,
    "ratchet_suspect": 1,
}


def _worst_of(verdicts: list[str]) -> str:
    """Rolls up N per-stage verdicts into one. If anything is genuinely
    unclean, report the worst of those. Otherwise every stage is live or
    quiet -- report `live` if any stage is, `quiet` only if all are.
    """
    unclean = [v for v in verdicts if v not in CLEAN_VERDICTS]
    if unclean:
        return max(unclean, key=lambda v: _UNCLEAN_SEVERITY.get(v, 0))
    return "live" if "live" in verdicts else "quiet"


def _normalize_to_unit_scale(series: list[float]) -> list[float]:
    """Rescale a series by its own window max so classify_channel_series's
    `[0,1]`-calibrated spread thresholds (the quiet/live boundary) mean
    something for a domain (row counts) they were never tuned for.
    All-zero/empty series pass through unchanged -- scaling by a positive
    max preserves zero exactly, so `dead`/`never_produced` detection is
    untouched.

    Does NOT fix `ratchet_suspect` -- see `_classify_unbounded_series()`,
    which is why callers use that instead of calling this directly into
    `classify_channel_series()`. Normalizing does not change whether a
    series is monotonically non-decreasing (climbed = last-first is
    guaranteed to exceed LIVE_VARIANCE_THRESHOLD for almost any monotonic
    series once rescaled to a max of 1.0), so a live sanity check caught
    that scale alone does not fix the false-positive this exists to
    prevent -- see the module docstring's ratchet_suspect note.
    """
    if not series:
        return series
    peak = max(series)
    if peak <= 0:
        return series
    return [v / peak for v in series]


def _classify_unbounded_series(series: list[float]) -> str:
    """classify_channel_series(), scoped to what actually transfers to a
    domain that isn't `[0,1]`-bounded the way field-channel salience/
    pressure values are -- currently the ladder's bucketed row counts and
    `attention_self_model.v1#broadcast_lane_age_sec` (a plain float age in
    seconds, not `Field(ge=0.0, le=1.0)` like its four sibling scalars).

    `ratchet_suspect` means "a mode=add channel that should decay but
    never does" (channel_glossary.py's own docstring) -- a concept with no
    equivalent for an unbounded magnitude that legitimately climbs (a
    growing bucket count, a broadcast lane that hasn't refreshed in a
    while). A monotonically non-decreasing run across a short or
    coarsely-bucketed window (the ladder's consolidation stage: 16 buckets)
    is common and unremarkable, not evidence of a stuck accumulator.
    Live-tested 2026-08-19: normalizing by max (`_normalize_to_unit_scale`)
    does NOT prevent this on its own -- a climb from any value up to the
    series' own max, rescaled to max=1.0, clears
    `LIVE_VARIANCE_THRESHOLD=0.05` almost every time a series is monotonic
    at all, regardless of scale. So `ratchet_suspect` is downgraded to
    `live` here rather than silently mis-flagging a healthy busy/ramping
    series as a suspected stuck accumulator -- `dead` and `never_produced`
    (genuine problems) are untouched.
    """
    verdict = classify_channel_series(_normalize_to_unit_scale(series))
    return "live" if verdict == "ratchet_suspect" else verdict


# Fields on AttentionSelfModelV1 that are NOT `[0,1]`-bounded by schema
# (`orion/schemas/attention_self_model.py`) -- these go through
# `_classify_unbounded_series` instead of `classify_channel_series` directly.
# Found by review 2026-08-20: an earlier version fixed this exact borrowed-
# threshold problem for the ladder's row counts but missed that
# `broadcast_lane_age_sec` (a plain float age in seconds) has the identical
# exposure among attention_self_model.v1's five scalar fields -- the other
# four (`confidence`, `prediction_error_confidence`, `field_overall_salience`,
# `heartbeat_mean_ratio`) all carry `Field(ge=0.0, le=1.0)`, so
# `classify_channel_series`'s thresholds are the right, unmodified tool for
# them and stay that way.
_UNBOUNDED_ATTENTION_SELF_MODEL_FIELDS = frozenset({"broadcast_lane_age_sec"})


def resolve_dsn() -> str:
    for key in _ENV_KEYS:
        val = os.environ.get(key)
        if val:
            return val
    return DEFAULT_POSTGRES_URI


def resolved_host() -> str:
    """The host:port `resolve_dsn()` would connect to, for error messages --
    never credentials. Added 2026-08-20 (review finding): if an operator's
    shell has `POSTGRES_URI` exported to a docker-network hostname that
    doesn't resolve from the host (a real risk -- `.env_example`'s own
    `POSTGRES_URI` is exactly that), every phase-5 check silently degrades
    to `db_unreachable` with no way to tell it tried the wrong host at all.
    This does not fix that ambiguity -- fixing it for real needs a distinct
    host-reachable-Postgres env convention repo-wide, out of scope for this
    patch -- but it does make the failure legible instead of a silent
    mystery.
    """
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(resolve_dsn())
        return parsed.hostname and f"{parsed.hostname}:{parsed.port or '?'}" or "?"
    except Exception:
        return "?"


# Bounded so a query hanging AFTER connect (lock contention, an unindexed
# scan) cannot reproduce the same "CLI call hangs indefinitely" failure
# CONNECT_TIMEOUT_SECONDS exists to prevent for the connect phase itself
# (review finding 2026-08-20). These queries are small and bounded by
# construction (MAX_ROWS, fixed bucket counts); 10s is generous headroom,
# not a tuned budget.
STATEMENT_TIMEOUT_MS = 10_000


def open_readonly_connection(dsn: str | None = None):
    """Thin wrapper over `orion.db_readonly.open_readonly_connection`: adds
    this module's DSN-resolution chain, connect timeout, and statement
    timeout. Returns `None` on any failure -- callers must treat that as
    "liveness unknown", never as a "dead" verdict.
    """
    return _open_readonly_connection(
        dsn or resolve_dsn(),
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        statement_timeout_ms=STATEMENT_TIMEOUT_MS,
    )


@dataclass(frozen=True)
class LivenessOutcome:
    """One computed liveness verdict, with the receipts to trust it.

    `sample_count` is always a real observation count in the same unit for
    both source kinds: rows read for a scalar field, total rows summed
    across all buckets (not "buckets with any rows") for a throughput
    source -- comparable across both, not "buckets touched".

    `truncated=True` means the window hit `MAX_ROWS` and this verdict does
    NOT cover the full requested window -- only the most recent `MAX_ROWS`
    rows within it (see `ScalarFieldSource.fetch`'s DESC+LIMIT). Added
    2026-08-20 (review finding): existing readers of the same table
    (`measure_attention_self_model_confidence_baseline.py`'s
    `fetch_self_model_rows`) already carry this signal specifically because
    a prior review caught a sibling script silently hiding it; a liveness
    verdict computed from a silently-truncated window would otherwise look
    identical to one from a complete window. Unreachable at current scale
    (~120 rows/hour vs `MAX_ROWS=50,000`) but not asserted away.
    """

    verdict: str  # classify_channel_series() vocabulary (worst-of if multi-series)
    sample_count: int
    detail: str  # human-readable receipt, e.g. per-stage breakdown for the ladder
    truncated: bool = False


@dataclass(frozen=True)
class ScalarFieldSource:
    """One JSONB scalar field, sampled ordered by a timestamp column."""

    table: str
    json_column: str
    ts_column: str
    window_hours: float

    def fetch(self, conn, field: str) -> tuple[list[float], bool]:
        """Returns `(values, truncated)`. `truncated=True` means the window
        had >= MAX_ROWS matching rows and this only covers the most recent
        MAX_ROWS of them (the DESC+LIMIT below, reversed to ASC) -- see
        `LivenessOutcome.truncated`'s docstring for why that distinction
        matters and matches the precedent this mirrors
        (`measure_attention_self_model_confidence_baseline.py`'s
        `fetch_self_model_rows`).
        """
        # DESC + LIMIT, then reverse to ASC in Python -- if a window ever
        # yields more than MAX_ROWS, this keeps the MOST RECENT rows (the
        # ones that actually matter for "did this just go dead"), not the
        # oldest. An ASC-ordered LIMIT would silently do the opposite.
        query = f"""
            SELECT ({self.json_column} ->> %s)::float8 AS v
            FROM {self.table}
            WHERE {self.ts_column} >= now() - (%s * interval '1 hour')
              AND ({self.json_column} ->> %s) IS NOT NULL
            ORDER BY {self.ts_column} DESC
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(query, (field, self.window_hours, field, MAX_ROWS))
            rows = cur.fetchall()
        truncated = len(rows) >= MAX_ROWS
        values = [r[0] for r in rows if r[0] is not None]
        values.reverse()
        return values, truncated


@dataclass(frozen=True)
class ThroughputSource:
    """Rows-per-bucket over a real window -- liveness for a pipeline stage
    with no shared scalar to sample, not a signal value. Bucket size must be
    picked relative to the table's real cadence (see module docstring) --
    too fine and a healthy slow producer reads as mostly-empty buckets
    ("dead" false positive); too coarse and a real stall hides inside one
    still-nonzero bucket.
    """

    table: str
    ts_column: str
    window_hours: float
    bucket_hours: float

    def fetch(self, conn) -> list[float]:
        query = f"""
            SELECT count(t.{self.ts_column})::float8
            FROM generate_series(
                now() - (%s * interval '1 hour'),
                now(),
                (%s * interval '1 hour')
            ) AS bucket_start
            LEFT JOIN {self.table} t
              ON t.{self.ts_column} >= bucket_start
             AND t.{self.ts_column} < bucket_start + (%s * interval '1 hour')
            GROUP BY bucket_start
            ORDER BY bucket_start ASC
        """
        with conn.cursor() as cur:
            cur.execute(
                query, (self.window_hours, self.bucket_hours, self.bucket_hours)
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]


# --------------------------------------------------------------------------
# attention_self_model.v1 -- scalar fields
# --------------------------------------------------------------------------

_ATTENTION_SELF_MODEL_SOURCE = ScalarFieldSource(
    table="substrate_attention_self_model",
    json_column="self_model_json",
    ts_column="generated_at",
    window_hours=1.0,  # ~30s cadence -> ~120 samples/hour when healthy
)

_ATTENTION_SELF_MODEL_SCHEMA_ID = "AttentionSelfModelV1"


def _attention_self_model_liveness(conn, field: str) -> LivenessOutcome:
    values, truncated = _ATTENTION_SELF_MODEL_SOURCE.fetch(conn, field)
    if field in _UNBOUNDED_ATTENTION_SELF_MODEL_FIELDS:
        verdict = _classify_unbounded_series(values)
    else:
        verdict = classify_channel_series(values)
    detail = (
        f"n={len(values)} over "
        f"{_ATTENTION_SELF_MODEL_SOURCE.window_hours:g}h from "
        f"{_ATTENTION_SELF_MODEL_SOURCE.table}.{_ATTENTION_SELF_MODEL_SOURCE.json_column}->>'{field}'"
    )
    if truncated:
        detail += " [TRUNCATED -- window exceeded MAX_ROWS, verdict covers only the most recent rows]"
    return LivenessOutcome(
        verdict=verdict, sample_count=len(values), detail=detail, truncated=truncated
    )


# --------------------------------------------------------------------------
# l7_l11_ladder -- throughput across 5 pipeline stages
# --------------------------------------------------------------------------

_LADDER_STAGES: dict[str, ThroughputSource] = {
    "substrate_proposal_frames": ThroughputSource(
        table="substrate_proposal_frames",
        ts_column="generated_at",
        window_hours=1.0,
        bucket_hours=1.0 / 60,  # 1min buckets, ~2.1s cadence -> ~28 rows/bucket healthy
    ),
    "substrate_policy_decision_frames": ThroughputSource(
        table="substrate_policy_decision_frames",
        ts_column="generated_at",
        window_hours=1.0,
        bucket_hours=1.0 / 60,
    ),
    "substrate_feedback_frames": ThroughputSource(
        table="substrate_feedback_frames",
        ts_column="generated_at",
        window_hours=1.0,
        bucket_hours=1.0 / 60,
    ),
    "substrate_attention_frames": ThroughputSource(
        table="substrate_attention_frames",
        ts_column="generated_at",
        window_hours=1.0,
        bucket_hours=1.0 / 60,
    ),
    "substrate_consolidation_frames": ThroughputSource(
        table="substrate_consolidation_frames",
        ts_column="generated_at",
        window_hours=48.0,  # matches this repo's own measure_attention_self_model
        # precedent (DEFAULT_SINCE_HOURS=48) for a slow-cadence table
        bucket_hours=3.0,  # ~96min cadence -> ~1.9 rows/bucket healthy; a
        # 1h bucket would read ~0.6 rows/bucket on a HEALTHY table -- false
        # "dead" territory. 3h clears that with margin.
    ),
}


def _ladder_liveness(conn) -> LivenessOutcome:
    per_stage: dict[str, str] = {}
    total_rows = 0
    for table, source in _LADDER_STAGES.items():
        series = source.fetch(conn)
        total_rows += int(sum(series))
        per_stage[table] = _classify_unbounded_series(series)
    worst = _worst_of(list(per_stage.values()))
    detail = ", ".join(f"{t}={v}" for t, v in per_stage.items())
    return LivenessOutcome(verdict=worst, sample_count=total_rows, detail=detail)


# --------------------------------------------------------------------------
# public dispatch
# --------------------------------------------------------------------------


_KIND_ATTENTION_SELF_MODEL = "attention_self_model"
_KIND_LADDER = "ladder"


def _resolve_source_kind(node: MetricNode) -> str | None:
    """Single source of truth for "does this node have a registered phase-5
    liveness source, and which one". `has_registered_source()` and
    `liveness_for_node()` both call this rather than each carrying their own
    copy of the routing conditionals -- found by review 2026-08-20: two
    independent copies drift silently the moment a third candidate is added
    to only one of them (no error, no test catches it; the cheap pre-check
    in `has_registered_source` would just skip a node that IS registered).
    """
    if node.schema_id == _ATTENTION_SELF_MODEL_SCHEMA_ID and node.metric_field:
        return _KIND_ATTENTION_SELF_MODEL
    if node.name == "l7_l11_ladder" and node.metric_field is None:
        return _KIND_LADDER
    return None


def liveness_for_node(node: MetricNode, conn) -> LivenessOutcome | None:
    """Computed liveness for one lineage node, or `None` if no data source
    is registered for it -- the honest "not computed" case. Never guess.

    Can raise on a query-time DB failure (network drop, table renamed,
    statement timeout) -- unlike `open_readonly_connection`, this does not
    swallow that itself. `scripts/check_metric_lineage.py`'s
    `_liveness_for_nodes` is the one place that catches it and degrades to
    "unknown", the same place connection failures are already handled.
    """
    kind = _resolve_source_kind(node)
    if kind == _KIND_ATTENTION_SELF_MODEL:
        return _attention_self_model_liveness(conn, node.metric_field)
    if kind == _KIND_LADDER:
        return _ladder_liveness(conn)
    return None


def has_registered_source(node: MetricNode) -> bool:
    """Cheap pre-check so callers can skip opening a DB connection entirely
    when nothing in a lookup's node set has a registered liveness source.
    """
    return _resolve_source_kind(node) is not None
