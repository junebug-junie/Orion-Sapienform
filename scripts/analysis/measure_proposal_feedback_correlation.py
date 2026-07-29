#!/usr/bin/env python3
"""Read-only measurement: does real historical `FeedbackFrameV1` data support
calibrating `orion/proposals/scoring.py::proposal_priority()`'s fixed
`0.4*match_score + 0.2*urgency + 0.1*confidence` coefficients (or
`config/proposals/proposal_policy.v1.yaml`'s `dimension_weights`/
`base_priority`/`base_risk`) against real recorded outcomes?

Session context (2026-07-28/29, deliberately-deferred second half of
`docs/superpowers/specs/2026-07-28-precision-weighted-proposal-scoring-design.md`,
following PR #1442's `dimension_confidence()` EWMA fix): the design doc's own
Missing Questions 3 and 4 are the real fork this script answers with real
numbers, not guessed:

  3. Per-template or per-dimension calibration? (real design fork -- decide
     from real sample size / correlation signal, not by default)
  4. Periodic offline refit, not live online update (doc's own strong prior;
     this script IS that offline measurement pass, matching this repo's
     "measure before minting" convention, `scripts/analysis/measure_*.py`).

## Disclosed scoping decisions

1. **Reuses the real production schema, not a reimplementation.** Every row
   is parsed into a real `FeedbackFrameV1` (`orion.schemas.feedback_frame`).
   `orion.proposals.policy.load_proposal_policy` loads the real, live
   `config/proposals/proposal_policy.v1.yaml` to build the real
   template -> dominant-dimension map (max-weight dimension per template),
   not a hand-copied guess.

2. **Chain-completeness check first.** `FeedbackFrameV1.source_proposal_frame_id`
   / `source_policy_frame_id` / `source_execution_dispatch_frame_id` are
   confirmed to resolve against real rows in `substrate_proposal_frames`,
   `substrate_policy_decision_frames`, `substrate_execution_dispatch_frames`
   respectively (a real join, not assumed from the schema alone) before any
   correlation is computed.

3. **Template attribution via the real, already-shipped ID contract, not a
   new join.** `orion/proposals/builder.py::stable_proposal_id()` embeds the
   template key directly: `f"proposal:{template_key}:{field_tick_id}:
   {attention_frame_id}"`. Every downstream ID that descends from a proposal
   candidate (`ExecutionDispatchCandidateV1.dispatch_id`, its cortex result's
   `result_id`) embeds this same substring verbatim (confirmed live by direct
   inspection of real rows before writing this script, not assumed from
   schema reading alone -- see `extract_template_key()`'s docstring for the
   real observed formats). `extract_template_key()` recovers the template key
   with a single regex anchored on the literal `"proposal:"` marker
   (`template_key` values are plain config dict keys, never containing `:`),
   so this works directly on `OutcomeObservationV1.source_id` without needing
   to separately load and join `substrate_proposal_frames` or
   `substrate_execution_dispatch_frames` payloads at all.

4. **Score is NOT the real per-candidate signal -- checked, not assumed.**
   `FeedbackFrameV1.observations[].score` is a fixed constant per
   `outcome_kind` (`orion/feedback/builder.py::_score_for_outcome_kind`,
   backed by `FeedbackScoringV1`'s 8 hand-set constants) -- real stddev within
   any single (template, source_kind, outcome_kind) bucket is float noise
   (~1e-13), confirmed live against the full real table before writing this
   report section, not assumed from reading the mapping function. The real,
   graduated signal is the *outcome_kind distribution itself* -- specifically
   `cortex_result`'s `completed` vs `failed` rate, the only source_kind in
   this pipeline that reflects a real dispatched action's real result rather
   than a deterministic function of policy-gate/dispatch-mode config.

5. **This script is strictly read-only.** Refuses to proceed if the Postgres
   session cannot be forced read-only (same contract as every sibling
   `measure_*` script). Writes only to
   `/tmp/measure-proposal-feedback-correlation/`. Does NOT modify
   `orion/proposals/scoring.py`, `config/proposals/proposal_policy.v1.yaml`,
   or any database table, and does NOT build or persist any calibration
   artifact -- that is an explicit, separate Step 2, gated on this script's
   own verdict.

Run:
    python scripts/analysis/measure_proposal_feedback_correlation.py --window-hours 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from orion.proposals.policy import ProposalPolicyV1, load_proposal_policy
from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1

logger = logging.getLogger("orion.analysis.proposal_feedback_correlation")

DEGENERATE_VARIANCE_EPS: float = 1e-12
MIN_TOTAL_ROWS: int = 200
DEFAULT_WINDOW_HOURS: float = 200.0
MAX_ROWS: int = 300_000
SERVER_CURSOR_ITERSIZE: int = 500

# A modest bar for a Bernoulli completion-rate estimate to be worth reading at
# all -- not a claim that n=30 is "enough" for a live calibration, just the
# floor below which a rate is not worth reporting as a candidate signal.
MIN_SAMPLES_FOR_RATE_SIGNAL: int = 30

# Only these source_kinds carry an outcome_kind distribution that is not
# collapsed to a single tick-level (not per-candidate) observation --
# field_delta and absence are per-tick, not per-template, so they cannot be
# attributed to a single template's real ID chain.
TEMPLATE_ATTRIBUTABLE_SOURCE_KINDS: tuple[str, ...] = ("dispatch_candidate", "cortex_result", "policy_decision")

# orion/proposals/builder.py::stable_proposal_id() -- template_key is a plain
# config dict key (see config/proposals/proposal_policy.v1.yaml's
# proposal_templates mapping), never containing ':', so this anchors cleanly
# regardless of what follows (field_tick_id, attention_frame_id -- the latter
# itself contains further ':'-separated segments).
_TEMPLATE_KEY_RE = re.compile(r"proposal:([^:]+):")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "proposals" / "proposal_policy.v1.yaml"

OUTPUT_DIR = Path("/tmp/measure-proposal-feedback-correlation")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "template_outcome_counts.csv"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O.
# ===========================================================================


def extract_template_key(source_id: str) -> Optional[str]:
    """Recovers the proposal template key from any ID descending from
    `orion/proposals/builder.py::stable_proposal_id()` (dispatch candidate
    dispatch_id, cortex result result_id, policy decision proposal_id) --
    all embed `proposal:{template_key}:` verbatim. Returns None (never a
    guessed fallback) only when the `proposal:` marker itself is absent.

    IMPORTANT (code review, 2026-07-29): `orion/reverie/proposal.py` builds
    reverie-sourced (Phase B spontaneous-thought) proposal candidates as
    `proposal_id=f"proposal:reverie:{thought.thought_id}"` -- structurally
    IDENTICAL in shape to a real config template's ID
    (`f"proposal:{template_key}:..."`, `orion/proposals/builder.py::
    stable_proposal_id()`). This function therefore extracts `"reverie"` for
    those IDs, NOT None -- an earlier version of this docstring and comment
    incorrectly claimed reverie IDs "use a different ID shape entirely" and
    return None; that was never true and was caught only by tracing the real
    `orion/reverie/proposal.py` source, not by this function's own tests
    (which used a synthetic ID that didn't match the real shape). Correctness
    here relies on `known_templates` filtering (see `is_known_template()`)
    happening at the call site, not on this function itself distinguishing
    "reverie" from a real config template key -- it cannot, and does not try
    to.
    """
    m = _TEMPLATE_KEY_RE.search(source_id)
    return m.group(1) if m else None


def is_known_template(template: Optional[str], known_templates: set[str]) -> bool:
    """True only if `template` is a real key in the live
    `config/proposals/proposal_policy.v1.yaml` (`known_templates`, built from
    `ProposalPolicyV1.proposal_templates`). Excludes both `None` (no
    `proposal:` marker found at all) and any real-but-unrecognized key --
    most importantly `"reverie"` (see `extract_template_key()`'s docstring):
    `extract_template_key()` cannot distinguish a reverie-sourced ID from a
    real template's ID by shape alone, so this is the one place that filter
    actually happens. Applied uniformly at bucketing time so an unrecognized
    template can never appear in one report table (e.g. the score-degeneracy
    table) while being silently absent from another (e.g. the cortex
    completion-rate table) -- a single, consistent filter, not two
    independently-drifting ones.
    """
    return template is not None and template in known_templates


def population_mean_variance(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance


@dataclass
class BucketStats:
    """Real historical stats for one (template, source_kind) bucket."""

    template: str
    source_kind: str
    n: int
    outcome_kind_counts: Counter
    score_mean: float
    score_variance: float
    score_stddev: float

    @property
    def score_degenerate(self) -> bool:
        return self.score_variance <= DEGENERATE_VARIANCE_EPS


@dataclass
class CortexCompletionSignal:
    """Real completed/failed rate for one template's `cortex_result`
    observations -- the only source_kind in this pipeline reflecting a real
    dispatched action's real result rather than a deterministic function of
    policy-gate/dispatch-mode config (see module docstring #4)."""

    template: str
    n_completed: int
    n_failed: int
    n_other: int

    @property
    def n_total(self) -> int:
        return self.n_completed + self.n_failed + self.n_other

    @property
    def completion_rate(self) -> Optional[float]:
        denom = self.n_completed + self.n_failed
        if denom == 0:
            return None
        return self.n_completed / denom

    @property
    def meets_min_samples(self) -> bool:
        return self.n_total >= MIN_SAMPLES_FOR_RATE_SIGNAL


def build_bucket_stats(
    observations: dict[tuple[str, str], list[OutcomeObservationV1]],
) -> dict[tuple[str, str], BucketStats]:
    out: dict[tuple[str, str], BucketStats] = {}
    for (template, source_kind), obs_list in observations.items():
        scores = [o.score for o in obs_list]
        mean, variance = population_mean_variance(scores)
        out[(template, source_kind)] = BucketStats(
            template=template,
            source_kind=source_kind,
            n=len(obs_list),
            outcome_kind_counts=Counter(o.outcome_kind for o in obs_list),
            score_mean=mean,
            score_variance=variance,
            score_stddev=math.sqrt(variance),
        )
    return out


def build_cortex_completion_signals(
    observations: dict[tuple[str, str], list[OutcomeObservationV1]],
    known_templates: list[str],
) -> dict[str, CortexCompletionSignal]:
    signals: dict[str, CortexCompletionSignal] = {}
    for template in known_templates:
        obs_list = observations.get((template, "cortex_result"), [])
        counts = Counter(o.outcome_kind for o in obs_list)
        signals[template] = CortexCompletionSignal(
            template=template,
            n_completed=counts.get("completed", 0),
            n_failed=counts.get("failed", 0),
            n_other=sum(v for k, v in counts.items() if k not in ("completed", "failed")),
        )
    return signals


def template_dominant_dimension(policy: ProposalPolicyV1) -> dict[str, str]:
    """Real, config-derived map: each template's highest-weighted scored
    dimension (`ProposalTemplateV1.dimensions`), used only to report whether
    a per-dimension calibration would see meaningfully more distinct
    dimension buckets than templates already provide (module docstring's
    Missing Question 3) -- not a guess, read directly from the live
    `config/proposals/proposal_policy.v1.yaml`."""
    out: dict[str, str] = {}
    for key, template in policy.proposal_templates.items():
        if not template.dimensions:
            continue
        out[key] = max(template.dimensions.items(), key=lambda kv: kv[1])[0]
    return out


@dataclass
class Verdict:
    templates_with_any_cortex_signal: int
    templates_with_sufficient_cortex_signal: int
    total_templates: int
    distinct_dominant_dimensions_with_signal: int
    proceed_to_calibration: bool
    reasons: list[str] = field(default_factory=list)


def compute_verdict(
    cortex_signals: dict[str, CortexCompletionSignal],
    dominant_dimension: dict[str, str],
    total_templates: int,
) -> Verdict:
    with_any = [t for t, s in cortex_signals.items() if s.n_total > 0]
    with_sufficient = [t for t, s in cortex_signals.items() if s.meets_min_samples]
    dims_with_signal = {dominant_dimension[t] for t in with_sufficient if t in dominant_dimension}

    reasons: list[str] = []
    proceed = True

    coverage_frac = len(with_sufficient) / total_templates if total_templates else 0.0
    if coverage_frac < 0.5:
        proceed = False
        reasons.append(
            f"only {len(with_sufficient)}/{total_templates} templates "
            f"({coverage_frac*100:.0f}%) have >= {MIN_SAMPLES_FOR_RATE_SIGNAL} real "
            f"cortex_result observations to estimate a completion rate from -- "
            f"the rest have never been dispatched far enough to produce a real "
            f"outcome at all (blocked by policy-gate/dispatch-mode config, not "
            f"by their own quality), so any weight fit now would only ever be "
            f"informed by a minority of templates and would silently treat "
            f"'never given the chance to run' as equivalent to 'ran and failed' "
            f"for the rest unless explicitly excluded"
        )
    if with_any and min(s.n_total for s in cortex_signals.values() if s.n_total > 0) < MIN_SAMPLES_FOR_RATE_SIGNAL:
        thin = [
            f"{t} (n={s.n_total})"
            for t, s in cortex_signals.items()
            if 0 < s.n_total < MIN_SAMPLES_FOR_RATE_SIGNAL
        ]
        if thin:
            reasons.append(
                f"template(s) with real but statistically thin cortex_result "
                f"samples (below n={MIN_SAMPLES_FOR_RATE_SIGNAL}): {', '.join(thin)} "
                f"-- a completion rate estimated from single-digit samples is "
                f"not a real signal, it is noise dressed as one"
            )
    if len(dims_with_signal) < 2:
        # Code review (2026-07-29): this used to be reported as a disqualifying
        # reason without actually setting proceed=False -- if coverage ever
        # independently clears the 50% bar above while dimension diversity
        # stays collapsed, the verdict would say PROCEED while its own reason
        # list simultaneously argued the data can't support the distinction.
        # A real, non-degenerate per-template signal with only one distinct
        # dominant dimension IS still enough to fit per-TEMPLATE weights (the
        # dimension question is separate from whether calibration can proceed
        # at all) -- so this alone does not block `proceed`. What it blocks is
        # ever claiming a per-dimension answer is resolvable from this data;
        # `distinct_dominant_dimensions_with_signal` is reported explicitly so
        # a reader can see that limit for themselves, without the boolean
        # verdict overclaiming a distinction the data can't make.
        reasons.append(
            f"only {len(dims_with_signal)} distinct dominant dimension(s) "
            f"({sorted(dims_with_signal)}) appear among templates with "
            f"sufficient real signal -- per-DIMENSION calibration cannot be "
            f"distinguished from per-TEMPLATE calibration on this data (not a "
            f"blocker for per-template calibration itself, which this does not "
            f"disqualify on its own)"
        )

    return Verdict(
        templates_with_any_cortex_signal=len(with_any),
        templates_with_sufficient_cortex_signal=len(with_sufficient),
        total_templates=total_templates,
        distinct_dominant_dimensions_with_signal=len(dims_with_signal),
        proceed_to_calibration=proceed,
        reasons=reasons,
    )


# ===========================================================================
# I/O layer -- psycopg2 read-only, server-side cursor.
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


def fetch_chain_completeness(conn) -> dict[str, int]:
    """Real join, not assumed from schema alone: for each feedback frame,
    do its source_proposal_frame_id / source_policy_frame_id /
    source_execution_dispatch_frame_id actually resolve to real rows in the
    upstream tables?"""
    if conn is None:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  count(*) AS total_feedback,
                  count(*) FILTER (WHERE f.source_proposal_frame_id IS NOT NULL) AS has_proposal_ref,
                  count(*) FILTER (WHERE p.frame_id IS NOT NULL) AS proposal_ref_resolves,
                  count(*) FILTER (WHERE f.source_policy_frame_id IS NOT NULL) AS has_policy_ref,
                  count(*) FILTER (WHERE pol.frame_id IS NOT NULL) AS policy_ref_resolves,
                  count(*) FILTER (WHERE d.frame_id IS NOT NULL) AS dispatch_ref_resolves
                FROM substrate_feedback_frames f
                LEFT JOIN substrate_proposal_frames p ON p.frame_id = f.source_proposal_frame_id
                LEFT JOIN substrate_policy_decision_frames pol ON pol.frame_id = f.source_policy_frame_id
                LEFT JOIN substrate_execution_dispatch_frames d ON d.frame_id = f.source_execution_dispatch_frame_id
                """
            )
            row = cur.fetchone()
    except Exception:
        logger.error("failed to compute chain completeness", exc_info=True)
        return {}
    if not row:
        return {}
    cols = [
        "total_feedback",
        "has_proposal_ref",
        "proposal_ref_resolves",
        "has_policy_ref",
        "policy_ref_resolves",
        "dispatch_ref_resolves",
    ]
    return dict(zip(cols, (int(v or 0) for v in row)))


def fetch_feedback_summary(conn) -> tuple[Optional[datetime], Optional[datetime], int]:
    if conn is None:
        return None, None, 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT min(generated_at), max(generated_at), count(*) FROM substrate_feedback_frames")
            row = cur.fetchone()
    except Exception:
        logger.error("failed to fetch substrate_feedback_frames summary", exc_info=True)
        return None, None, 0
    if not row:
        return None, None, 0
    min_ts, max_ts, count = row
    if min_ts is not None and min_ts.tzinfo is None:
        min_ts = min_ts.replace(tzinfo=timezone.utc)
    if max_ts is not None and max_ts.tzinfo is None:
        max_ts = max_ts.replace(tzinfo=timezone.utc)
    return min_ts, max_ts, int(count or 0)


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


@dataclass
class StreamResult:
    buckets: dict[tuple[str, str], list[OutcomeObservationV1]]
    n_frames: int
    n_schema_errors: int
    n_excluded_unknown_template: int
    truncated: bool
    hard_failure: bool


def stream_feedback_observations(
    conn,
    start: datetime,
    end: datetime,
    max_rows: int,
    known_templates: set[str],
    progress: ProgressLog,
) -> StreamResult:
    """Streams real substrate_feedback_frames rows in [start, end], parses
    each into a real FeedbackFrameV1, and buckets every observation by
    (template_key, source_kind) -- template_key recovered from the
    observation's own source_id (module docstring #3), not from a separate
    join.

    Buckets ONLY observations whose extracted template_key is a real key in
    `known_templates` (`is_known_template()`), applied uniformly here rather
    than per-report-section -- code review (2026-07-29) found reverie-sourced
    (Phase B) observations are structurally indistinguishable from a real
    template's ID by `extract_template_key()` alone (both match
    `proposal:{key}:`), so without this filter a `"reverie"` bucket would
    silently appear in some report tables and not others. Excluded-count is
    tracked and reported, not silently dropped.

    `hard_failure=True` means the query/stream itself errored (network,
    schema drift in a way `ValidationError` doesn't catch, etc.) -- distinct
    from a clean "the window legitimately contained zero matching rows".
    `run()` must not print a decisive STOP/PROCEED verdict when this is set
    and no rows were processed; that would misrepresent an infrastructure
    failure as a real finding about the data.
    """
    buckets: dict[tuple[str, str], list[OutcomeObservationV1]] = defaultdict(list)
    n_frames = 0
    n_errors = 0
    n_excluded = 0
    n_seen = 0
    if conn is None:
        return StreamResult(buckets, n_frames, n_errors, n_excluded, False, True)
    conn.autocommit = False
    try:
        with conn.cursor(name="proposal_feedback_correlation_cursor") as cur:
            cur.itersize = SERVER_CURSOR_ITERSIZE
            cur.execute(
                """
                SELECT generated_at, feedback_frame_json
                FROM substrate_feedback_frames
                WHERE generated_at >= %s AND generated_at <= %s
                ORDER BY generated_at ASC
                LIMIT %s
                """,
                (start, end, max_rows),
            )
            for _generated_at, payload in cur:
                n_seen += 1
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        n_errors += 1
                        continue
                try:
                    frame = FeedbackFrameV1.model_validate(payload)
                except ValidationError:
                    n_errors += 1
                    continue
                n_frames += 1
                for obs in frame.observations:
                    if obs.source_kind not in TEMPLATE_ATTRIBUTABLE_SOURCE_KINDS:
                        continue
                    template = extract_template_key(obs.source_id)
                    if not is_known_template(template, known_templates):
                        n_excluded += 1
                        continue
                    buckets[(template, obs.source_kind)].append(obs)
                if n_seen % 5000 == 0:
                    progress.emit(
                        "streaming substrate_feedback_frames",
                        percent=min(95.0, 20.0 + 70.0 * (n_seen / max(max_rows, 1))),
                        processed=n_seen,
                        total=max_rows,
                    )
    except Exception:
        logger.error("failed to stream substrate_feedback_frames rows", exc_info=True)
        return StreamResult(buckets, n_frames, n_errors, n_excluded, False, True)
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.autocommit = True
    return StreamResult(buckets, n_frames, n_errors, n_excluded, n_seen >= max_rows, False)


# ===========================================================================
# Report rendering + orchestration.
# ===========================================================================


def write_csv(path: Path, cortex_signals: dict[str, CortexCompletionSignal], dominant_dimension: dict[str, str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["template", "dominant_dimension", "n_completed", "n_failed", "n_other", "n_total", "completion_rate"]
        )
        for t in sorted(cortex_signals):
            s = cortex_signals[t]
            writer.writerow(
                [
                    t,
                    dominant_dimension.get(t, ""),
                    s.n_completed,
                    s.n_failed,
                    s.n_other,
                    s.n_total,
                    f"{s.completion_rate:.4f}" if s.completion_rate is not None else "",
                ]
            )


def render_report(
    *,
    chain: dict[str, int],
    table_span_label: str,
    total_table_rows: int,
    window_label: str,
    n_window_frames: int,
    n_schema_errors: int,
    n_excluded_unknown_template: int,
    truncated: bool,
    hard_failure: bool,
    bucket_stats: dict[tuple[str, str], BucketStats],
    cortex_signals: dict[str, CortexCompletionSignal],
    dominant_dimension: dict[str, str],
    known_templates: list[str],
    verdict: Verdict,
) -> str:
    lines = [
        "# Proposal Priority Calibration -- Real Historical Correlation Probe",
        "",
        "Read-only. No writes, no events, no flag/config changes, no edits to "
        "`orion/proposals/scoring.py`, `config/proposals/proposal_policy.v1.yaml`, "
        "or any database table. Answers Missing Questions 3-4 of "
        "`docs/superpowers/specs/2026-07-28-precision-weighted-proposal-scoring-"
        "design.md` (the priority/weight-calibration half deferred by PR #1442) "
        "using real historical `FeedbackFrameV1` data.",
        "",
        "## Chain completeness",
        "",
    ]
    if chain:
        lines.extend(
            [
                f"- Total `substrate_feedback_frames` rows: {chain.get('total_feedback', 0)}",
                f"- Rows with a non-null `source_proposal_frame_id`: "
                f"{chain.get('has_proposal_ref', 0)}, of which "
                f"{chain.get('proposal_ref_resolves', 0)} resolve to a real "
                f"`substrate_proposal_frames` row",
                f"- Rows with a non-null `source_policy_frame_id`: "
                f"{chain.get('has_policy_ref', 0)}, of which "
                f"{chain.get('policy_ref_resolves', 0)} resolve to a real "
                f"`substrate_policy_decision_frames` row",
                f"- `source_execution_dispatch_frame_id` resolves to a real "
                f"`substrate_execution_dispatch_frames` row for "
                f"{chain.get('dispatch_ref_resolves', 0)} rows",
            ]
        )
        complete = (
            chain.get("total_feedback", 0) > 0
            and chain.get("proposal_ref_resolves", 0) == chain.get("has_proposal_ref", 0)
            and chain.get("policy_ref_resolves", 0) == chain.get("has_policy_ref", 0)
            and chain.get("dispatch_ref_resolves", 0) == chain.get("total_feedback", 0)
        )
        lines.append(
            f"- **Chain completeness: {'100% -- every reference resolves' if complete else 'INCOMPLETE, see counts above'}.**"
        )
    else:
        lines.append("- Could not compute (query failed, see log).")
    lines.extend(
        [
            "",
            "## Data source",
            "",
            f"- Table: `substrate_feedback_frames` (real production writer: "
            f"`services/orion-feedback-runtime/app/store.py::"
            f"FeedbackRuntimeStore.save_feedback_frame()`)",
            f"- Full-table real history span: {table_span_label}",
            f"- Full-table real row count: {total_table_rows}",
            f"- Measurement window used: {window_label}",
            f"- Real feedback frames parsed in window: {n_window_frames}",
            f"- Rows that failed `FeedbackFrameV1` schema validation (excluded): {n_schema_errors}",
            f"- Observations excluded as unrecognized template (not a real "
            f"`config/proposals/proposal_policy.v1.yaml` key -- includes "
            f"reverie-sourced `proposal:reverie:...` observations, which are "
            f"structurally indistinguishable from a real template's ID by "
            f"marker alone; see `is_known_template()`): {n_excluded_unknown_template}",
        ]
    )
    if truncated:
        lines.append(
            f"- **Window truncated at MAX_ROWS={MAX_ROWS}** -- fetch is ORDER BY "
            f"generated_at ASC, so truncation keeps the OLDEST rows and drops the "
            f"newest. Narrow `--window-hours` to avoid this."
        )
    if hard_failure:
        lines.append(
            "- **A stream error occurred partway through this run** (see "
            "progress log) -- rows processed before the failure are still "
            "reported below, but this window's real coverage may be "
            "incomplete relative to what `--window-hours` requested."
        )
    lines.extend(
        [
            "",
            "## Per-candidate `score` degeneracy check (module docstring #4)",
            "",
            "`FeedbackFrameV1.observations[].score` is a fixed constant per "
            "`outcome_kind` by construction (`orion/feedback/builder.py::"
            "_score_for_outcome_kind`). Checked against real data, not assumed:",
            "",
            "| template | source_kind | n | outcome_kind distribution | score mean | score stddev | degenerate? |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for (template, source_kind), st in sorted(bucket_stats.items(), key=lambda kv: (-kv[1].n, kv[0])):
        kinds = ", ".join(f"{k}:{v}" for k, v in sorted(st.outcome_kind_counts.items(), key=lambda kv: -kv[1]))
        lines.append(
            f"| `{template}` | `{source_kind}` | {st.n} | {kinds} | "
            f"{st.score_mean:.4g} | {st.score_stddev:.3g} | "
            f"{'YES (real signal degenerate)' if st.score_degenerate else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Real per-template `cortex_result` completion-rate signal",
            "",
            "The only source_kind reflecting a real dispatched action's real "
            "result, not a deterministic function of policy-gate/dispatch-mode "
            f"config. `MIN_SAMPLES_FOR_RATE_SIGNAL = {MIN_SAMPLES_FOR_RATE_SIGNAL}` "
            "is a floor for a rate to be worth reading at all, not a claim it is "
            "sufficient to calibrate from.",
            "",
            "| template | dominant dimension | n_completed | n_failed | n_other | n_total | completion_rate | usable? |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for t in sorted(known_templates, key=lambda t: -cortex_signals[t].n_total if t in cortex_signals else 0):
        s = cortex_signals.get(t)
        if s is None:
            continue
        rate = f"{s.completion_rate:.4f}" if s.completion_rate is not None else "n/a"
        lines.append(
            f"| `{t}` | `{dominant_dimension.get(t, 'n/a')}` | {s.n_completed} | "
            f"{s.n_failed} | {s.n_other} | {s.n_total} | {rate} | "
            f"{'yes' if s.meets_min_samples else 'NO -- below sample floor'} |"
        )
    lines.extend(
        [
            "",
            "## Missing Question 3/4 verdict",
            "",
            f"- Templates configured (`config/proposals/proposal_policy.v1.yaml`): "
            f"{verdict.total_templates}",
            f"- Templates with ANY real `cortex_result` observation ever: "
            f"{verdict.templates_with_any_cortex_signal}",
            f"- Templates with >= {MIN_SAMPLES_FOR_RATE_SIGNAL} real "
            f"`cortex_result` observations: "
            f"{verdict.templates_with_sufficient_cortex_signal}",
            f"- Distinct dominant dimensions represented among templates with "
            f"sufficient signal: {verdict.distinct_dominant_dimensions_with_signal}",
            "",
            f"**VERDICT: {'PROCEED to calibration (Step 2)' if verdict.proceed_to_calibration else 'STOP -- do NOT build a calibration mechanism from this data yet'}.**",
            "",
        ]
    )
    if verdict.reasons:
        lines.append("Reasons:")
        lines.append("")
        for r in verdict.reasons:
            lines.append(f"- {r}")
        lines.append("")
    lines.extend(
        [
            "## Explicit scope boundary",
            "",
            "This script is a measurement only. It did NOT modify "
            "`orion/proposals/scoring.py`, `config/proposals/"
            "proposal_policy.v1.yaml`, any live scoring/producer code path, or "
            "any database table, and did NOT persist any calibration artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def run(window_hours: float, policy_path: Path) -> int:
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressLog(PROGRESS_PATH)

    policy = load_proposal_policy(policy_path)
    known_templates = list(policy.proposal_templates.keys())
    dominant_dimension = template_dominant_dimension(policy)

    progress.emit("connect", percent=0.0, processed=0, total=0)
    conn = open_readonly_connection(dsn)
    if conn is None:
        progress.close()
        print("ERROR: could not open a read-only postgres connection; see log")
        return 2

    chain = fetch_chain_completeness(conn)
    min_ts, max_ts, total_rows = fetch_feedback_summary(conn)
    progress.emit("summary fetched", percent=5.0, processed=total_rows, total=total_rows)

    if min_ts is None or max_ts is None or total_rows < MIN_TOTAL_ROWS:
        try:
            conn.close()
        except Exception:
            pass
        progress.close()
        print(
            f"insufficient real substrate_feedback_frames history "
            f"(rows={total_rows}, min_required={MIN_TOTAL_ROWS}); stopping"
        )
        return 2

    table_span_label = (
        f"{min_ts.isoformat()} -> {max_ts.isoformat()} "
        f"({(max_ts - min_ts).total_seconds() / 3600.0:.1f}h)"
    )
    window_end = max_ts
    window_start = max(min_ts, max_ts - timedelta(hours=window_hours))
    window_label = f"{window_start.isoformat()} -> {window_end.isoformat()} ({window_hours:g}h requested)"

    result = stream_feedback_observations(
        conn, window_start, window_end, MAX_ROWS, set(known_templates), progress
    )
    try:
        conn.close()
    except Exception:
        pass
    progress.emit("observations bucketed", percent=95.0, processed=result.n_frames, total=result.n_frames)

    # A hard failure (query/stream error, or no connection) with zero real
    # frames processed is an infrastructure failure, not a real finding about
    # the data -- must not be reported as a decisive STOP/PROCEED verdict
    # (code review, 2026-07-29: an earlier version of this script would have
    # silently printed a confident STOP here, indistinguishable from a real
    # "the data genuinely doesn't support calibration" result).
    if result.hard_failure and result.n_frames == 0:
        progress.close()
        msg = (
            "MEASUREMENT FAILED (not a real STOP verdict): streaming "
            "substrate_feedback_frames errored before any real frame was "
            "processed -- see the progress log for the underlying exception. "
            "Re-run once the underlying failure is fixed; do not treat this "
            "as evidence the data is insufficient."
        )
        print(msg)
        return 3

    bucket_stats = build_bucket_stats(result.buckets)
    cortex_signals = build_cortex_completion_signals(result.buckets, known_templates)
    verdict = compute_verdict(cortex_signals, dominant_dimension, len(known_templates))

    write_csv(CSV_PATH, cortex_signals, dominant_dimension)
    report = render_report(
        chain=chain,
        table_span_label=table_span_label,
        total_table_rows=total_rows,
        window_label=window_label,
        n_window_frames=result.n_frames,
        n_schema_errors=result.n_schema_errors,
        n_excluded_unknown_template=result.n_excluded_unknown_template,
        truncated=result.truncated,
        hard_failure=result.hard_failure,
        bucket_stats=bucket_stats,
        cortex_signals=cortex_signals,
        dominant_dimension=dominant_dimension,
        known_templates=known_templates,
        verdict=verdict,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("done", percent=100.0, processed=1, total=1)
    progress.close()

    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}, {PROGRESS_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only probe: does real historical FeedbackFrameV1 data "
        "support calibrating proposal_priority()'s coefficients or "
        "proposal_policy.v1.yaml's dimension_weights/base_priority against real "
        "recorded outcomes? Answers Missing Questions 3-4 of the precision-"
        "weighted-proposal-scoring design doc."
    )
    parser.add_argument(
        "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"Most recent N hours of real substrate_feedback_frames history to "
        f"replay (default: {DEFAULT_WINDOW_HOURS}).",
    )
    parser.add_argument(
        "--policy-path", type=Path, default=DEFAULT_POLICY_PATH,
        help=f"Path to the real proposal policy YAML (default: {DEFAULT_POLICY_PATH}).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args.window_hours, args.policy_path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
