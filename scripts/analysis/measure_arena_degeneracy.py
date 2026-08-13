#!/usr/bin/env python3
"""Read-only measurement: does Orion's proposal arena actually rank anything?

## Why this exists

`docs/superpowers/specs/2026-08-12-metric-commensurability-handover-spec.md` §7
names one measurement as the next patch, ahead of any design work:

    "Measure how many consecutive policy frames are semantically identical."

That question is answered here (§C below), but running it surfaced a strictly
larger defect that the spec's §5 diagnosis names in the abstract and never
measured: the arena's scalar has almost no resolution across its own
competitors, so the "starvation" the previous arc (PR #1599) fixed with aging
and a reserved lane was a symptom of a ranking function that is arithmetically
incapable of ranking. This script measures that too (§A, §B).

## What it measures, and why each number matters

Four sections, each independently interpretable, each a pasted number rather
than a derivation (spec §6.2):

  A. **Urgency resolution.** How many DISTINCT `urgency_score` values exist
     among the candidates of a single proposal frame, and how many candidates
     are tied at that frame's maximum. If most candidates share one value, the
     tiebreak -- and therefore the whole arena -- is decided by whatever
     constant is added downstream, not by any measurement of the world.

  B. **Cross-template correlation.** The Pearson correlation of `urgency_score`
     and `priority_score` between pairs of templates within the same frame. A
     correlation of exactly 1.0 means two "competing" signals are literally the
     same number; the arena has one degree of freedom wearing N names. This is
     the quantitative form of the spec's failure mode A (incomparable scales)
     and D (argmax starvation).

  C. **Decision redundancy.** Runs of consecutive `substrate_policy_decision_
     frames` carrying an identical set of (template, decision) pairs, plus the
     count of frames carrying zero decisions at all. This is spec §7 verbatim.
     It decides between "reduce production" and "raise throughput" for the
     91%-stale-discard blocker in spec §2.1.

  D. **Dispatch reachability.** Per template: proposed count, dispatched count,
     and blocked count over the window. Answers "which candidates can win at
     all", which is the outcome the other three sections predict.

## Disclosed scoping decisions

1. **Read-only, enforced at the session level.** Uses the same
   `SET default_transaction_read_only = on` + read-back verification pattern as
   `measure_proposal_dimension_variance.py`, and refuses to run if the session
   does not come back read-only. This script never writes.

2. **Reads persisted frames, not live code.** Every number comes from what the
   production path actually stored (`substrate_proposal_frames`,
   `substrate_policy_decision_frames`, `substrate_execution_dispatch_frames`),
   not from re-running the scoring functions here. That is deliberate: the
   point is to measure what shipped, not what the source implies should have
   shipped. A reimplementation could agree with a bug.

3. **Template key is parsed from `proposal_id`, not from config.** Live ids have
   the shape `proposal:<template_key>:<tick_id>:...`, so field 2 of a
   colon-split is the template key. The same parse is already load-bearing in
   `services/orion-execution-dispatch-runtime/app/worker.py`'s starvation
   keying. If the id format changes, this script reports `UNPARSEABLE` for the
   affected rows and counts them, rather than silently attributing them to a
   wrong template.

4. **Semantic identity for §C is (template, decision) pairs, sorted, ignoring
   scores and ids.** Scores are continuous and would make every frame trivially
   "distinct" while carrying no new information for dispatch; ids embed the tick
   and would do the same. What dispatch actually consumes is which actions were
   approved, so that is what identity is defined over. Frames with zero
   decisions are fingerprinted as the sentinel `EMPTY` and included in the run
   analysis, since an unbroken run of empty frames is exactly as redundant as
   an unbroken run of identical non-empty ones.

5. **Correlation pairs are chosen from the data, not hardcoded.** §B reports the
   full pairwise matrix over the N most frequently co-occurring templates, so a
   config change that renames or replaces templates does not silently reduce
   this section to a no-op the way a hardcoded pair would.

6. **Bounded window.** Defaults to `--window-hours 24`, matching the cadence at
   which these tables actually turn over (~34k policy frames/day live). The real
   row count used is reported every run, never silently truncated.

## Usage

    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_arena_degeneracy.py --window-hours 24

Exit codes: 0 = measurement completed (regardless of what it found; this is an
instrument, not a gate), 1 = could not measure (no DB, no rows, bad window).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Iterable, Sequence

logger = logging.getLogger("measure_arena_degeneracy")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"

# Sentinel fingerprint for a policy frame that carried no decisions at all.
# Deliberately not the empty string: an empty string is what a bug that fails to
# build a fingerprint would also produce, and those two cases must stay
# distinguishable in the output.
EMPTY_FINGERPRINT = "<EMPTY:no-decisions>"

# Sentinel for a proposal_id that does not have the documented
# `proposal:<template_key>:...` shape. Counted and reported rather than dropped,
# so an id-format change shows up as a number instead of as silence.
UNPARSEABLE_TEMPLATE = "<UNPARSEABLE>"


# ===========================================================================
# Parsing helpers
# ===========================================================================


def template_key_from_proposal_id(proposal_id: Any) -> str:
    """Recover the template key from a live proposal or dispatch id.

    Two live shapes, both verified against stored frames (2026-08-12) rather
    than assumed from the schema:

        proposal:inspect_bus_channel_catalog:tick_5b0822b7faf3:attention.frame:...
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^ field 2

        dispatch:proposal:inspect_bus_channel_catalog:tick_24caa35e0805:...
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ field 3

    A `policy.decision:` prefix wraps the proposal id the same way, so the rule
    generalises to "the field after the last `proposal` marker". Anything that
    does not match returns `UNPARSEABLE_TEMPLATE` rather than a guess, so §D's
    per-template table cannot silently misattribute rows -- and unparseable rows
    are counted in the report instead of being dropped.
    """
    if not isinstance(proposal_id, str):
        return UNPARSEABLE_TEMPLATE
    parts = proposal_id.split(":")
    try:
        marker = len(parts) - 1 - parts[::-1].index("proposal")
    except ValueError:
        return UNPARSEABLE_TEMPLATE
    if marker + 1 >= len(parts) or not parts[marker + 1]:
        return UNPARSEABLE_TEMPLATE
    return parts[marker + 1]


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or None when it is undefined.

    Returns None (never 0.0) when either series has zero variance -- a constant
    series has an undefined correlation, and reporting 0.0 for it would read as
    "measured, unrelated" when the truth is "one of these never moved". That
    distinction is the entire point of this script.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    var_x = sum(d * d for d in dx)
    var_y = sum(d * d for d in dy)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(var_x * var_y)


def run_lengths(fingerprints: Iterable[str]) -> list[int]:
    """Lengths of maximal runs of consecutive equal fingerprints."""
    runs: list[int] = []
    current: str | None = None
    count = 0
    for fingerprint in fingerprints:
        if fingerprint == current:
            count += 1
            continue
        if current is not None:
            runs.append(count)
        current = fingerprint
        count = 1
    if current is not None:
        runs.append(count)
    return runs


def _run_values(fingerprints: Sequence[str]) -> list[str]:
    """The fingerprint of each maximal run, aligned 1:1 with `run_lengths`."""
    values: list[str] = []
    for fingerprint in fingerprints:
        if not values or values[-1] != fingerprint:
            values.append(fingerprint)
    return values


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile over an already-sorted sequence."""
    if not sorted_values:
        return None
    index = max(0, min(len(sorted_values) - 1, int(math.ceil(q * len(sorted_values))) - 1))
    return float(sorted_values[index])


# ===========================================================================
# Measurement sections
# ===========================================================================


@dataclass
class UrgencyResolution:
    frames: int = 0
    candidates: int = 0
    distinct_urgency_histogram: Counter = dataclass_field(default_factory=Counter)
    tied_at_max: list[int] = dataclass_field(default_factory=list)
    candidates_per_frame: list[int] = dataclass_field(default_factory=list)
    frames_missing_urgency: int = 0


def measure_urgency_resolution(frames: Sequence[Sequence[dict[str, Any]]]) -> UrgencyResolution:
    """§A -- how many distinct urgency values does a frame's field actually hold?

    A frame with 10 candidates and 2 distinct urgency values has, at most, 2
    levels of discrimination available to it no matter how the downstream
    formula combines things. Everything below that resolution is decided by a
    constant.
    """
    result = UrgencyResolution()
    for candidates in frames:
        urgencies = [_as_float(c.get("urgency_score")) for c in candidates]
        present = [u for u in urgencies if u is not None]
        if not present:
            result.frames_missing_urgency += 1
            continue
        result.frames += 1
        # Counts candidates with a READABLE urgency, not all candidates.
        # `tied_at_max` below is computed over the same population, so the
        # ratio the report prints has one consistent denominator. These are
        # currently identical because `urgency_score` is required on
        # ProposalCandidateV1 -- correct by accident of the schema rather than
        # by construction, which is not a property worth relying on.
        result.candidates += len(present)
        result.candidates_per_frame.append(len(present))
        # Round to 9dp before counting distinctness: float round-trips through
        # JSON can differ in the last bit for values that were computed
        # identically, and counting those as "distinct" would understate the
        # degeneracy this section exists to find.
        rounded = [round(u, 9) for u in present]
        result.distinct_urgency_histogram[len(set(rounded))] += 1
        top = max(rounded)
        result.tied_at_max.append(sum(1 for u in rounded if u == top))
    return result


@dataclass
class PairCorrelation:
    template_a: str
    template_b: str
    n: int
    corr_urgency: float | None
    corr_priority: float | None


def measure_cross_template_correlation(
    frames: Sequence[Sequence[dict[str, Any]]],
    *,
    top_n: int,
    min_pairs: int,
) -> tuple[list[PairCorrelation], list[str]]:
    """§B -- are two "competing" templates carrying the same number?

    Pairs are drawn from the `top_n` most frequently appearing templates in the
    window rather than hardcoded, so this section keeps working across config
    changes. Only pairs co-occurring in at least `min_pairs` frames are
    reported; a correlation over a handful of samples is an artifact, which this
    repo has been burned by before (see the short-window-distribution-stats
    lesson in CLAUDE.md's lineage).
    """
    by_template_urgency: dict[str, dict[int, float]] = defaultdict(dict)
    by_template_priority: dict[str, dict[int, float]] = defaultdict(dict)
    appearances: Counter = Counter()

    for frame_index, candidates in enumerate(frames):
        for candidate in candidates:
            template = template_key_from_proposal_id(candidate.get("proposal_id"))
            if template == UNPARSEABLE_TEMPLATE:
                continue
            urgency = _as_float(candidate.get("urgency_score"))
            priority = _as_float(candidate.get("priority_score"))
            if urgency is None and priority is None:
                continue
            appearances[template] += 1
            # A template appearing twice in one frame (different targets) would
            # otherwise silently overwrite; keep the first and let §D's counts
            # expose the duplication if it exists.
            if urgency is not None:
                by_template_urgency[template].setdefault(frame_index, urgency)
            if priority is not None:
                by_template_priority[template].setdefault(frame_index, priority)

    templates = [name for name, _ in appearances.most_common(top_n)]
    pairs: list[PairCorrelation] = []
    for i, a in enumerate(templates):
        for b in templates[i + 1 :]:
            shared = sorted(set(by_template_urgency[a]) & set(by_template_urgency[b]))
            if len(shared) < min_pairs:
                continue
            corr_u = pearson(
                [by_template_urgency[a][k] for k in shared],
                [by_template_urgency[b][k] for k in shared],
            )
            shared_p = sorted(set(by_template_priority[a]) & set(by_template_priority[b]))
            corr_p = (
                pearson(
                    [by_template_priority[a][k] for k in shared_p],
                    [by_template_priority[b][k] for k in shared_p],
                )
                if len(shared_p) >= min_pairs
                else None
            )
            pairs.append(PairCorrelation(a, b, len(shared), corr_u, corr_p))
    pairs.sort(key=lambda p: (-(p.corr_urgency if p.corr_urgency is not None else -2.0), p.template_a))
    return pairs, templates


@dataclass
class DecisionRedundancy:
    total_frames: int = 0
    empty_frames: int = 0
    distinct_fingerprints: int = 0
    runs: list[int] = dataclass_field(default_factory=list)
    empty_runs: list[int] = dataclass_field(default_factory=list)
    identical_runs: list[int] = dataclass_field(default_factory=list)


def measure_decision_redundancy(
    decision_sets: Sequence[Sequence[dict[str, Any]]],
) -> DecisionRedundancy:
    """§C -- spec §7 verbatim: how many consecutive policy frames say the same thing?

    Input is expected in ascending `generated_at` order; runs are only
    meaningful over a time-ordered sequence.
    """
    result = DecisionRedundancy(total_frames=len(decision_sets))
    fingerprints: list[str] = []
    for decisions in decision_sets:
        if not decisions:
            result.empty_frames += 1
            fingerprints.append(EMPTY_FINGERPRINT)
            continue
        parts = sorted(
            f"{template_key_from_proposal_id(d.get('proposal_id'))}={d.get('decision')}"
            for d in decisions
        )
        fingerprints.append("|".join(parts))
    result.distinct_fingerprints = len(set(fingerprints))
    result.runs = run_lengths(fingerprints)
    # Split, because the two populations imply DIFFERENT remedies and the
    # combined headline hides which one dominates: a run of empty frames means
    # "Orion produced nothing and published it anyway", while a run of
    # identical non-empty frames means "Orion re-published the same conclusion".
    # The first is fixed by not emitting; the second by change-detection. Spec
    # section 7 asks which of those the 91% discard actually is.
    result.empty_runs = [n for fp, n in zip(_run_values(fingerprints), result.runs) if fp == EMPTY_FINGERPRINT]
    result.identical_runs = [
        n for fp, n in zip(_run_values(fingerprints), result.runs) if fp != EMPTY_FINGERPRINT
    ]
    return result


# Block reasons that mean "policy said no", NOT "lost the capacity contest".
#
# A template gated to `operator_review` or `deferred` is BLOCKED BY DESIGN --
# `config/execution_dispatch/execution_dispatch_policy.v1.yaml` lists both under
# `blocked_policy_decisions`, and `proposal_kind_to_cortex` has no route for
# `defer` or `request_policy_review` at all. Counting those as starvation was a
# false headline: the first live 24h run reported three "starved" templates
# (defer_due_to_low_readiness, request_policy_review_for_action, reverie) and
# ALL THREE were policy-blocked on 100% of their blocked rows. True starvation
# count was zero.
#
# Matched as a prefix so a reason gains detail without silently escaping the
# classification.
POLICY_BLOCK_REASON_PREFIXES = ("policy_decision:",)


def is_policy_block_reason(reason: str) -> bool:
    return any(reason.startswith(prefix) for prefix in POLICY_BLOCK_REASON_PREFIXES)


@dataclass
class TemplateReach:
    proposed: int = 0
    dispatched: int = 0
    blocked: int = 0
    blocked_reasons: Counter = dataclass_field(default_factory=Counter)

    @property
    def policy_blocked(self) -> int:
        return sum(n for reason, n in self.blocked_reasons.items() if is_policy_block_reason(reason))

    @property
    def contested_blocked(self) -> int:
        """Blocked for a reason that is NOT a policy refusal -- i.e. it entered
        the capacity contest and lost."""
        return self.blocked - self.policy_blocked

    @property
    def is_starved(self) -> bool:
        """Proposed, never dispatched, and not merely refused by policy.

        A template that policy declines to route is not starved; it is doing
        exactly what it was configured to do. Only a template that was allowed
        to compete and never won is starved.
        """
        if self.proposed <= 0 or self.dispatched > 0:
            return False
        return self.blocked == 0 or self.contested_blocked > 0


def _candidate_id(candidate: dict[str, Any]) -> Any:
    """The best available id on a dispatch candidate.

    Dispatch frames do not carry `proposal_id` on their candidates -- they carry
    `dispatch_id` (`dispatch:proposal:<template>:...`), with `result_ref` as a
    same-shaped fallback. Checked in that order, `proposal_id` last so this also
    works if a future frame version adds it.
    """
    for key in ("dispatch_id", "result_ref", "proposal_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def measure_dispatch_reachability(
    proposal_frames: Sequence[Sequence[dict[str, Any]]],
    dispatched: Sequence[dict[str, Any]],
    blocked: Sequence[dict[str, Any]],
) -> dict[str, TemplateReach]:
    """§D -- which templates can actually reach a dispatch slot at all?

    The outcome the other three sections predict. A template with a nonzero
    `proposed` and a zero `dispatched` over a full window is starved regardless
    of what its score says.

    `dispatched` and `blocked` arrive as separate lists because that is how
    `ExecutionDispatchFrameV1` actually stores them (`dispatched_candidates` /
    `blocked_candidates`) -- there is no single `candidates` list carrying a
    `status` discriminator, and reading one produced a table of all-zeroes that
    looked exactly like total starvation.
    """
    reach: dict[str, TemplateReach] = defaultdict(TemplateReach)
    for candidates in proposal_frames:
        for candidate in candidates:
            reach[template_key_from_proposal_id(candidate.get("proposal_id"))].proposed += 1
    for candidate in dispatched:
        reach[template_key_from_proposal_id(_candidate_id(candidate))].dispatched += 1
    for candidate in blocked:
        entry = reach[template_key_from_proposal_id(_candidate_id(candidate))]
        entry.blocked += 1
        reasons = candidate.get("reasons")
        if isinstance(reasons, list) and reasons:
            entry.blocked_reasons[str(reasons[0])] += 1
        else:
            entry.blocked_reasons["<no-reason>"] += 1
    return dict(reach)


# ===========================================================================
# I/O layer -- read-only psycopg2, server-side cursors to bound client memory
# ===========================================================================


def open_readonly_connection(dsn: str):
    try:
        import psycopg2  # noqa: F401
    except Exception:
        logger.error("psycopg2 unavailable; cannot open DB session")
        return None
    import psycopg2

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
        # Named (server-side) cursors below require an open transaction, which
        # autocommit suppresses. The read-only guarantee above is a SESSION-level
        # setting, so it should survive this flip -- but "should" is exactly the
        # kind of assumption this repo keeps getting burned by, so read it back
        # from inside a real transaction and refuse to proceed if it did not.
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only;")
            in_tx = cur.fetchone()
        conn.rollback()
        if not in_tx or str(in_tx[0]).lower() != "on":
            logger.error(
                "refusing to run: transaction is not read-only after leaving autocommit (got %r)",
                in_tx,
            )
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


def _stream(conn, name: str, sql: str, params: Sequence[Any]):
    with conn.cursor(name=name) as cur:
        cur.itersize = 500
        cur.execute(sql, params)
        for row in cur:
            yield row


def _json_field(raw: Any, key: str) -> list[dict[str, Any]]:
    """Pull a list-of-objects field out of a stored frame, defensively.

    psycopg2 returns a `jsonb` column as an already-decoded dict; a `text`
    column comes back as a string. Both are handled so this script does not
    depend on which type a given table happens to use.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, dict):
        return []
    value = raw.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def load_proposal_frames(conn, window_hours: float) -> list[list[dict[str, Any]]]:
    sql = (
        "SELECT proposal_frame_json FROM substrate_proposal_frames "
        "WHERE generated_at > now() - (%s || ' hours')::interval "
        "ORDER BY generated_at ASC"
    )
    return [
        _json_field(row[0], "candidates")
        for row in _stream(conn, "arena_proposals", sql, (str(window_hours),))
    ]


def load_policy_decisions(conn, window_hours: float) -> list[list[dict[str, Any]]]:
    sql = (
        "SELECT policy_decision_frame_json FROM substrate_policy_decision_frames "
        "WHERE generated_at > now() - (%s || ' hours')::interval "
        "ORDER BY generated_at ASC"
    )
    return [
        _json_field(row[0], "decisions")
        for row in _stream(conn, "arena_policy", sql, (str(window_hours),))
    ]


def load_dispatch_candidates(
    conn, window_hours: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Returns (dispatched, blocked, frames_seen).

    `frames_seen` is reported so a section-D table of zeroes can be told apart
    from "there were no dispatch frames at all" -- two very different findings
    that an unqualified zero would conflate.
    """
    # `generated_at`, not `created_at`, for two independent reasons:
    #
    # 1. Correctness. Proposal frames are windowed on `generated_at`, and a
    #    dispatch frame is written well after the proposal frame it descends
    #    from -- measured live, median lag 287s, max 918s. Windowing the two
    #    populations on different clocks skews section D's proposed-vs-
    #    dispatched comparison. At the 24h default that skew is under 1%, but
    #    `--window-hours` accepts any positive value, and at 0.1h it reported
    #    five "starved" templates that had dispatched fine over a longer window.
    # 2. Speed. `created_at` is unindexed on this table; `generated_at` has
    #    idx_substrate_execution_dispatch_frames_generated_at.
    sql = (
        "SELECT dispatch_frame_json FROM substrate_execution_dispatch_frames "
        "WHERE generated_at > now() - (%s || ' hours')::interval "
        "ORDER BY generated_at ASC"
    )
    dispatched: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    frames = 0
    for row in _stream(conn, "arena_dispatch", sql, (str(window_hours),)):
        frames += 1
        dispatched.extend(_json_field(row[0], "dispatched_candidates"))
        blocked.extend(_json_field(row[0], "blocked_candidates"))
    return dispatched, blocked, frames


# ===========================================================================
# Report
# ===========================================================================


def _fmt_corr(value: float | None) -> str:
    return "  UNDEFINED" if value is None else f"{value:10.6f}"


def render_report(
    *,
    window_hours: float,
    urgency: UrgencyResolution,
    pairs: Sequence[PairCorrelation],
    top_templates: Sequence[str],
    redundancy: DecisionRedundancy,
    reach: dict[str, TemplateReach],
    dispatch_frames_seen: int,
) -> str:
    out: list[str] = []
    add = out.append

    add("=" * 78)
    add(f"ARENA DEGENERACY MEASUREMENT -- window = {window_hours}h")
    add("=" * 78)

    add("")
    add("## A. Urgency resolution (how many distinct values does a frame hold?)")
    add("")
    if urgency.frames == 0:
        add("  NO FRAMES WITH URGENCY DATA IN WINDOW")
    else:
        add(f"  frames measured                 {urgency.frames}")
        add(f"  candidates measured             {urgency.candidates}")
        avg_candidates = urgency.candidates / urgency.frames
        add(f"  avg candidates per frame        {avg_candidates:.2f}")
        # Live, this counter is dominated by frames carrying no candidates at
        # all -- an empty proposal frame, not a frame whose candidates lack the
        # field. Named for what it literally counts (no readable urgency
        # anywhere in the frame) so it stays true if that mix ever changes.
        add(f"  frames w/ no readable urgency   {urgency.frames_missing_urgency}")
        add("")
        add("  distinct urgency values per frame:")
        for distinct in sorted(urgency.distinct_urgency_histogram):
            count = urgency.distinct_urgency_histogram[distinct]
            pct = 100.0 * count / urgency.frames
            add(f"    {distinct:3d} distinct  ->  {count:7d} frames ({pct:5.1f}%)")
        avg_tied = sum(urgency.tied_at_max) / len(urgency.tied_at_max)
        add("")
        add(f"  avg candidates TIED at frame max  {avg_tied:.2f} of {avg_candidates:.2f}")
        add(
            f"  -> {100.0 * (1.0 - avg_tied / avg_candidates):.1f}% of candidates are NOT tied at their frame's max urgency"
        )

    add("")
    add("## B. Cross-template correlation (are competitors the same number?)")
    add("")
    if not pairs:
        add("  NO CO-OCCURRING TEMPLATE PAIRS MET THE MINIMUM SAMPLE THRESHOLD")
    else:
        add(f"  templates considered: {', '.join(top_templates)}")
        add("")
        add(f"  {'template A':36s} {'template B':36s} {'n':>7s} {'corr(urg)':>10s} {'corr(prio)':>10s}")
        for pair in pairs:
            add(
                f"  {pair.template_a:36s} {pair.template_b:36s} {pair.n:7d} "
                f"{_fmt_corr(pair.corr_urgency)} {_fmt_corr(pair.corr_priority)}"
            )
        perfect = [p for p in pairs if p.corr_urgency is not None and p.corr_urgency >= 0.999999]
        add("")
        add(f"  pairs with corr(urgency) >= 0.999999 (identical signal): {len(perfect)} of {len(pairs)}")

    add("")
    add("## C. Decision redundancy (spec section 7)")
    add("")
    if redundancy.total_frames == 0:
        add("  NO POLICY FRAMES IN WINDOW")
    else:
        add(f"  policy frames                   {redundancy.total_frames}")
        empty_pct = 100.0 * redundancy.empty_frames / redundancy.total_frames
        add(f"  frames with zero decisions      {redundancy.empty_frames} ({empty_pct:.1f}%)")
        add(f"  distinct decision sets          {redundancy.distinct_fingerprints}")
        n_runs = len(redundancy.runs)
        add(f"  runs of consecutive-identical   {n_runs}")
        if n_runs:
            redundant_pct = 100.0 * (redundancy.total_frames - n_runs) / redundancy.total_frames
            add(f"  frames adding no new decision   {redundancy.total_frames - n_runs} ({redundant_pct:.1f}%)")
            add("")
            # Split by population: the two imply different remedies, and the
            # combined number cannot tell them apart. See
            # measure_decision_redundancy's own comment.
            for label, runs in (
                ("all runs        ", redundancy.runs),
                ("EMPTY runs      ", redundancy.empty_runs),
                ("identical runs  ", redundancy.identical_runs),
            ):
                if not runs:
                    add(f"  {label}  (none)")
                    continue
                ordered = sorted(runs)
                add(
                    f"  {label}  n={len(runs):5d}  frames={sum(runs):6d}  "
                    f"mean/median/p95/max = {sum(ordered) / len(ordered):.2f} / "
                    f"{percentile(ordered, 0.5):.0f} / {percentile(ordered, 0.95):.0f} / {max(ordered)}"
                )

    add("")
    add("## D. Dispatch reachability (which templates ever win a slot?)")
    add("")
    add(f"  dispatch frames read in window  {dispatch_frames_seen}")
    add("")
    if not reach:
        add("  NO CANDIDATES IN WINDOW")
    else:
        add(
            f"  {'template':38s} {'proposed':>9s} {'dispatch':>9s} "
            f"{'blk:policy':>11s} {'blk:lost':>9s} {'win rate':>9s}"
        )
        for template, entry in sorted(reach.items(), key=lambda kv: -kv[1].proposed):
            # Guarded rather than defaulted to 0.0: `proposed == 0` with real
            # dispatches is reachable when the two populations are windowed at
            # a boundary, and printing "0.00%" for it would read as a starved
            # template rather than as an unmeasurable one.
            rate = f"{100.0 * entry.dispatched / entry.proposed:8.2f}%" if entry.proposed else "     n/a"
            add(
                f"  {template:38s} {entry.proposed:9d} {entry.dispatched:9d} "
                f"{entry.policy_blocked:11d} {entry.contested_blocked:9d} {rate}"
            )

        add("")
        add("  block reasons (why a candidate did not get a slot):")
        aggregate_reasons: Counter = Counter()
        for entry in reach.values():
            aggregate_reasons.update(entry.blocked_reasons)
        if not aggregate_reasons:
            add("    (none)")
        for reason, count in aggregate_reasons.most_common(10):
            label = "policy" if is_policy_block_reason(reason) else "contested"
            add(f"    {count:8d}  [{label:9s}]  {reason}")

        starved = [t for t, e in reach.items() if e.is_starved]
        policy_only = [
            t
            for t, e in reach.items()
            if e.proposed > 0 and e.dispatched == 0 and not e.is_starved
        ]
        add("")
        add(f"  STARVED (competed, never won): {len(starved)}")
        for template in sorted(starved):
            entry = reach[template]
            add(
                f"    - {template} ({entry.proposed} proposals, 0 dispatches, "
                f"{entry.contested_blocked} contested blocks)"
            )
        add(f"  blocked by policy, NOT starved: {len(policy_only)}")
        for template in sorted(policy_only):
            add(f"    - {template} ({reach[template].policy_blocked} policy blocks)")

    add("")
    add("=" * 78)
    return "\n".join(out)


# ===========================================================================
# Entry point
# ===========================================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument(
        "--top-templates",
        type=int,
        default=8,
        help="how many of the most-frequent templates to include in the section B matrix",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=100,
        help="minimum co-occurring frames before a correlation is reported at all",
    )
    parser.add_argument("--json-out", type=str, default=None, help="also write raw numbers here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.window_hours <= 0:
        logger.error("--window-hours must be positive (got %r)", args.window_hours)
        return 1

    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)
    conn = open_readonly_connection(dsn)
    if conn is None:
        return 1

    try:
        proposal_frames = load_proposal_frames(conn, args.window_hours)
        policy_decisions = load_policy_decisions(conn, args.window_hours)
        dispatched, blocked, dispatch_frames_seen = load_dispatch_candidates(
            conn, args.window_hours
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not proposal_frames and not policy_decisions:
        logger.error("no rows in the requested window; nothing to measure")
        return 1

    urgency = measure_urgency_resolution(proposal_frames)
    pairs, top_templates = measure_cross_template_correlation(
        proposal_frames, top_n=args.top_templates, min_pairs=args.min_pairs
    )
    redundancy = measure_decision_redundancy(policy_decisions)
    reach = measure_dispatch_reachability(proposal_frames, dispatched, blocked)

    report = render_report(
        window_hours=args.window_hours,
        urgency=urgency,
        pairs=pairs,
        top_templates=top_templates,
        redundancy=redundancy,
        reach=reach,
        dispatch_frames_seen=dispatch_frames_seen,
    )
    print(report)

    if args.json_out:
        payload = {
            "window_hours": args.window_hours,
            "urgency_resolution": {
                "frames": urgency.frames,
                "candidates": urgency.candidates,
                "distinct_urgency_histogram": dict(urgency.distinct_urgency_histogram),
                "avg_tied_at_max": (
                    sum(urgency.tied_at_max) / len(urgency.tied_at_max)
                    if urgency.tied_at_max
                    else None
                ),
            },
            "cross_template_correlation": [
                {
                    "a": p.template_a,
                    "b": p.template_b,
                    "n": p.n,
                    "corr_urgency": p.corr_urgency,
                    "corr_priority": p.corr_priority,
                }
                for p in pairs
            ],
            "decision_redundancy": {
                "total_frames": redundancy.total_frames,
                "empty_frames": redundancy.empty_frames,
                "distinct_fingerprints": redundancy.distinct_fingerprints,
                "n_runs": len(redundancy.runs),
                "max_run": max(redundancy.runs) if redundancy.runs else 0,
            },
            "dispatch_reachability": {
                template: {
                    "proposed": entry.proposed,
                    "dispatched": entry.dispatched,
                    "blocked": entry.blocked,
                    "blocked_reasons": dict(entry.blocked_reasons),
                }
                for template, entry in reach.items()
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        logger.info("wrote raw numbers to %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
