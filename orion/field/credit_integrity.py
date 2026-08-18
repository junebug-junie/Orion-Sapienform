"""Can a feedback-credited channel currently be fooled by a producer outage?

THE TRAP THIS WATCHES FOR
-------------------------
`config/feedback/feedback_policy.v1.yaml` lists `resource_pressure: decrease`
under `positive_delta_channels`, so a DROP is credited to the in-flight action.
If the drop happens because a producer went quiet rather than because the
action helped, the loop learns from an outage. This module is the scheduled
watch for that precondition -- report-only, no gating of the feedback loop
itself (see WHAT IT DELIBERATELY DOES NOT DO below).

REBUILT 2026-08-16 -- THE FIRST VERSION MEASURED THE WRONG THING
------------------------------------------------------------------
The first version of this module (merged as #1686, `421b69936`) reused
`orion.attention.tension.liveness.classify_producer_liveness()` verbatim,
on the premise that a decay-only stretch is a monotone geometric run and any
real write breaks monotonicity. Post-merge review falsified that on live
data: over 6,000 ticks / 3 credited dimensions, 0 of 454 flagged windows
contained a single actual 0.92 decay step. What actually tripped the
classifier was a flat plateau followed by one discontinuous step-down --
`execution_pressure` and `resource_pressure` are 3-5 distinct values in
6,000 ticks, `reliability_pressure` is 2 (0.0 / 0.9), so an ordinary
quantized state transition (including a REAL fix: 0.9 -> 0.0 is exactly the
true-positive this module exists to credit) satisfied the classifier's own
"decreased and never increased" premise without decaying at all. Separately,
75% of `resource_pressure`'s findings had the merge WINNER changing source
mid-window -- a max() envelope over ~5 producers can fall monotonically
while every producer writes every tick, which is a second, independent way
to fool a shape-based classifier that has nothing to do with the digester's
decay loop. The premise was false for these three channels specifically,
because `services/orion-field-digester/app/digestion/decay.py::apply_decay`
was rewritten (2026-07-17) to hold-then-decay-on-staleness rather than decay
unconditionally every tick, and all three are `CAPABILITY_DECAY_CHANNELS`
whose stored value is recomputed memorylessly by `apply_diffusion` every
tick regardless of decay.py's own loop. Full findings:
docs/superpowers/pr-reports/2026-08-16-credit-integrity-rebuild-pr.md.

THE REPLACEMENT: TWO WRITE-EVIDENCE SIGNALS, ROUTED BY HOW A WINNER WON
-------------------------------------------------------------------------
No shape, no monotonicity, no resolution floor. For each tick, this module
asks which of two structurally different things actually happened to
produce the credited dimension's WINNING channel this tick, and checks the
kind of evidence that mechanism can honestly produce:

1. **Node-vector-sourced** (the channel is a raw entry in the winning
   node's OWN `node_vectors[node_id]`, e.g. `reliability_pressure` on
   `node:athena`/`node:rpc_timeout`, confirmed 100% node-sourced and 100%
   `node_vector_updated_at`-stamped over 3,000 live ticks, 2026-08-16):
   reuse `orion.field.regime._refresh_from_timestamps` verbatim -- the same
   tested mechanism R2 shipped for the Hub glossary panel, comparing the
   winner's real write timestamp against the window. No new heuristic; a
   second consumer of an already-proven one.

2. **Capability-routed** (the channel lives in `capability_vectors`,
   resolved through `capability_provenance` -- `execution_pressure` and
   `resource_pressure`'s `pressure` channel are BOTH 100% capability-routed
   live, and confirmed to have ZERO resolvable `node_vector_updated_at`
   entries -- 0 of 3,000 -- so the timestamp mechanism above has nothing to
   read for them). `services/orion-field-digester/app/digestion/
   diffusion.py::apply_diffusion` is a fully memoryless per-tick recompute
   (verified by reading it, not assumed): `capability_provenance[cap][ch]`
   is set ONLY when a real (>0) or measured-zero contribution lands THIS
   tick, and is explicitly POPPED -- not left stale -- the instant nobody
   contributes ("clear stale provenance too... so the two never disagree
   about what's currently true", diffusion.py). So whether the winning
   channel's provenance resolves to a real node THIS tick (as opposed to
   the capability's own id, diffusion's documented "nobody contributed"
   default, or another capability one hop from a real sensor) IS the
   freshness signal -- not a stamp, but a genuine per-tick fact, checked
   fresh every 2s tick with no resolution floor to wait out.

Each dimension picks whichever mechanism its winning channel actually used
on a majority of sampled ticks (`report.source_kind[dim]` records the
split), and each finding records which one fired (`WindowFinding.evidence`)
so a reader never mistakes one signal for the other.

**Signal 1's tautology, found by the same review, fixed by being signal 1
instead of a member-of-`node_vectors` check.** The PREVIOUS version's
"backed" boolean was `provenance.get(ch) in state.node_vectors` applied
uniformly to every winner -- for a node-vector winner that is trivially
always true (a node's own vector entry is definitionally a member of its
own vector), proven live with a synthetic 40-tick pure-decay series with
zero producer writes reading `backed=True` on all 40 ticks. Signal 1 does
not ask "is the winner a node" -- it asks "what does that node's own
`node_vector_updated_at` say", which can and does say "stale".

Signal 2's version of that same check (does the winner resolve to a real
node) is NOT the same tautology, because -- unlike a node-vector winner --
a capability-routed winner's provenance is recomputed and cleared every
tick by `apply_diffusion`, not held from whenever it first appeared. The
prior version's provenance check was directionally correct for THIS case;
its bug was applying it uniformly to signal 1's case too (F3 in the
review), plus a separate vacuous-truth bug already fixed and released as
PR #1694 before this rebuild (`all()` over an empty generator when a
multi-channel dimension's contributor was entirely absent from the merge).

WHAT THIS MODULE STILL DOES NOT DO (R5a, the watch)
-----------------------------------------------------
Everything above this line is report-only, swept over a BATCH of ticks after
the fact -- it cannot gate a single feedback decision because it needs a
whole window to find a silent RUN, and by the time a run is long enough to
report, the credit for every tick in it has already been assigned.

R5B, THE GATE -- `channel_write_backed()` BELOW
--------------------------------------------------
2026-08-18. CLAUDE.md 0A requires proposal mode before changing a learning
loop unless Juniper directly asks to implement; Juniper did
("build it", after the R5a rebuild above and its own two review passes
landed and the roadmap doc recorded the open decision as resolved -- "offline
suppression is binary when shit is variable... live with it": design the
guard against the CURRENT loop, not a hypothetically-fixed one).

`channel_write_backed()` is the single-tick sibling of the batch watch above
-- same two mechanisms (`_classify_tick`, `_refresh_from_timestamps`), same
vocabulary, evaluated at exactly the one tick a feedback decision is about to
credit, not swept over history. It answers "should THIS credit decision trust
THIS tick's value" rather than "was there ever a silent run" -- a real-time
question the batch watch cannot answer because it needs a whole window's
worth of future ticks before it can conclude anything.

Consumed by `orion/feedback/builder.py::build_feedback_frame` to withhold
positive/negative pressure-delta evidence and the `reliability_pressure`
improved/worsened observation when the credited AFTER tick has no fresh
write evidence -- the exact trap this doc names above ("a DROP is credited
... If the drop happens because a producer went quiet ... the loop learns
from an outage"). Withheld evidence is recorded (`FeedbackFrameV1
.withheld_evidence`), never silently dropped, and the whole gate is a policy
flag (`FeedbackPolicyV1.write_evidence_guard_enabled`, default `true`) so it
is a one-line rollback, not a code revert, if it needs to come out.

Deliberately NOT reused for this gate: `analyse_credit_integrity()`'s
window-based signals. A live feedback tick has exactly two `FieldStateV1`
snapshots on hand (`field_before`, `field_after`) -- there is no store method
that returns the intervening window (confirmed: `FeedbackRuntimeStore` only
exposes `load_field_for_tick` and `load_latest_field_after`, both single-tick
loaders). Building one to reuse the window-based signals here would be a
second, heavier mechanism for a question the single-tick classifier already
answers correctly. If that changes, revisit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from orion.field.pressure import (
    CHANNEL_DIMENSION_MAP,
    collect_field_channel_pressures,
    map_channels_to_dimensions_with_provenance,
    winning_write_time,
)
from orion.field.regime import _refresh_from_timestamps
from orion.schemas.field_state import FieldStateV1

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "config" / "feedback" / "feedback_policy.v1.yaml"

# The verdict that means "no write evidence for the credited dimension's
# winning channel across this whole window" -- named to match
# orion.field.regime's own vocabulary for the same underlying question
# (`refresh_state == "no_write_in_window"`), not the old module's
# decay-flavoured `silent_producer`.
SILENT = "no_write_in_window"

_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class ChannelSample:
    at: datetime
    # "node" | "capability" | "absent" -- which of the two write-evidence
    # mechanisms this tick's winning channel actually used, or "absent" if
    # the channel did not appear in the merge at all this tick.
    source_kind: str
    # Meaningful only when source_kind == "node": that node's own
    # node_vector_updated_at entry for this channel (may be None -- no stamp
    # recorded yet -- distinct from "stale", which is a real, old stamp).
    node_stamp: datetime | None
    # Meaningful only when source_kind == "capability": did this tick's
    # diffusion recompute resolve to a real originating node (True), or to
    # the capability's own id / another capability / nothing (False).
    capability_fresh: bool | None


@dataclass(frozen=True)
class WindowFinding:
    """A span in which a credited channel could have misled the loop."""

    channel: str
    kind: str  # SILENT | "unmappable_dimension"
    # "timestamp" (signal 1) | "contribution" (signal 2) | "" for findings
    # that are not a write-evidence verdict at all (the other two kinds).
    evidence: str
    started_at: datetime
    span_seconds: float
    tick_count: int

    def describe(self) -> str:
        suffix = f" [{self.evidence}]" if self.evidence else ""
        return (
            f"{self.channel}: {self.kind} for {self.span_seconds:.0f}s "
            f"({self.tick_count} ticks) from {self.started_at.isoformat()}{suffix}"
        )


@dataclass
class CreditIntegrityReport:
    window_seconds: float
    channels: dict[str, int] = field(default_factory=dict)  # channel -> samples
    # Per dimension: {"node": n, "capability": n} -- which mechanism actually
    # had evidence to check, reported even when clean, so a reader can see
    # resolution rather than assume it. The two always sum to `channels[dim]`
    # -- see the "absent ticks never reach samples" note in
    # analyse_credit_integrity for why there is no third bucket here.
    source_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    findings: list[WindowFinding] = field(default_factory=list)
    # Longest observed run of capability_fresh=False, in ticks -- ONLY
    # populated for capability-routed dimensions (signal 2 has no timestamp
    # to report a seconds-based trend from; signal 1's trend is the finding
    # itself, since one stale window is already the whole verdict).
    longest_unbacked_run: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.findings)


def load_credited_dimensions(
    policy_path: Path | None = None,
) -> tuple[dict[str, list[str]], float]:
    """DIMENSION -> contributing channels, and the window the loop samples over.

    The policy names DIMENSIONS (`resource_pressure`), and
    `collect_field_channel_pressures()` returns CHANNELS -- not the same
    namespace: `resource_pressure` is fed by the channel `pressure`. A
    dimension that maps to no channel is returned with an empty list rather
    than dropped, so `analyse_credit_integrity` can raise it as a finding
    instead of silently reporting zero samples as zero problems.
    """
    from orion.feedback.policy import load_feedback_policy

    policy = load_feedback_policy(policy_path or POLICY_PATH)
    mapping = {
        dim: sorted(ch for ch, d in CHANNEL_DIMENSION_MAP.items() if d == dim)
        for dim in sorted(policy.positive_delta_channels)
    }
    return mapping, float(policy.windows.field_after_window_sec)


def _classify_tick(
    state: FieldStateV1,
    merged: dict[str, float],
    provenance: dict[str, str],
    channel: str | None,
) -> tuple[str, datetime | None, bool | None]:
    """Which write-evidence mechanism this tick's winning channel used.

    Node-sourced: the channel is a literal entry in the winner's OWN
    `node_vectors[source_id]` -- signal 1, checked via `node_stamp`.

    Capability-routed: everything else that still won the merge. This
    covers both a genuine capability_vectors winner and an edge-resolved
    node acting as a diffusion SOURCE for a channel it does not itself
    carry raw (`winning_write_time`'s own documented case) -- both read
    signal 2 correctly, since `capability_fresh` asks "did provenance
    resolve to a real node this tick", not "does the channel exist under
    that node's raw vector".
    """
    if not channel or channel not in merged:
        return "absent", None, None
    source_id = provenance.get(channel)
    node_vector = state.node_vectors.get(source_id) if source_id else None
    if node_vector is not None and channel in node_vector:
        return "node", winning_write_time(state, source_id, channel), None
    fresh = bool(source_id) and source_id in state.node_vectors
    return "capability", None, fresh


def channel_write_backed(
    state: FieldStateV1,
    dimension: str,
    *,
    max_staleness_seconds: float,
) -> bool | None:
    """R5b's gate primitive: is THIS tick's credited dimension backed by a
    real, fresh write -- checked at exactly the tick a feedback decision is
    about to use, not swept over a window (see the module docstring's R5B
    section for why this is a separate, single-tick sibling of
    `analyse_credit_integrity`, not a caller of it).

    Reuses the same two mechanisms `_classify_tick`/`_find_silent_*` use,
    nothing new: `_classify_tick` for which mechanism the winner actually
    used, `_refresh_from_timestamps` (regime.py, R2) for the node-sourced
    staleness comparison, given a synthetic one-sample "window" whose start
    is `max_staleness_seconds` before this tick -- the single-sample case
    degenerates cleanly (one non-None stamp is 100% coverage, so the
    coverage floor that exists for a real multi-sample window is a no-op
    here; a None stamp returns "unknown" directly on regime.py's first
    line). No new staleness heuristic.

    Returns:
      True  -- real write evidence, inside `max_staleness_seconds` (a node
               stamp, or a capability winner whose provenance resolved to a
               real node THIS tick).
      False -- a winner exists but its evidence says stale (a node stamp
               older than `max_staleness_seconds`) or not-fresh-this-tick (a
               capability winner whose provenance did not resolve to a real
               node this tick) -- the exact "reads calm during a producer
               outage" trap this rung exists to close.
      None  -- the dimension did not win a value from the merge at all this
               tick, or a node winner has never been stamped. Distinct from
               False on purpose, same unknown/silent distinction R5a's own
               tests regress (`test_low_stamp_coverage_reads_unknown_not_
               silent`) -- callers that need a single yes/no should treat
               `is not True` as "do not credit", not conflate this with
               False.
    """
    merged, provenance = collect_field_channel_pressures(state)
    dims, detail = map_channels_to_dimensions_with_provenance(merged, provenance)
    if dimension not in dims:
        return None
    winning_channel = detail[dimension].winning_channel if dimension in detail else None
    kind, node_stamp, cap_fresh = _classify_tick(state, merged, provenance, winning_channel)
    if kind == "absent":
        return None
    if kind == "capability":
        return bool(cap_fresh)
    verdict = _refresh_from_timestamps(
        [node_stamp],
        window_start=state.generated_at - timedelta(seconds=max_staleness_seconds),
    )
    if verdict == "unknown":
        return None
    return verdict == "producer_written"


def _samples_for(
    states: list[FieldStateV1], dimensions: dict[str, list[str]]
) -> dict[str, list[ChannelSample]]:
    """Value = the DIMENSION the loop actually credits; evidence = whichever
    write-evidence mechanism this tick's WINNING channel used.

    Sampling the dimension rather than a channel matters: `extractors.py`
    calls `field_pressures()` and diffs dimensions, so a per-channel series
    would be measuring something the loop never sees. The WINNER specifically
    (not "all contributing channels", the prior version's rule) matters
    because only the winning channel determines the dimension's value this
    tick -- a losing channel's staleness cannot make the credited value wrong
    if it never drove it.
    """
    out: dict[str, list[ChannelSample]] = {dim: [] for dim in dimensions}
    for state in states:
        merged, provenance = collect_field_channel_pressures(state)
        dims, detail = map_channels_to_dimensions_with_provenance(merged, provenance)
        for dim in dimensions:
            if dim not in dims:
                continue
            winning_channel = detail[dim].winning_channel if dim in detail else None
            kind, node_stamp, cap_fresh = _classify_tick(
                state, merged, provenance, winning_channel
            )
            out[dim].append(
                ChannelSample(
                    at=state.generated_at,
                    source_kind=kind,
                    node_stamp=node_stamp,
                    capability_fresh=cap_fresh,
                )
            )
    return out


def _rolling_windows(samples: list[ChannelSample], window_seconds: float):
    """Yield (start_index, end_index_exclusive) spans of >= window_seconds.

    Verified against brute-force minimal-window enumeration on both regular
    and irregular cadence during the review of the prior version -- the
    pointer is correctly monotone (the minimal right edge for `left+1` is
    never less than for `left`, since the window only shrinks as the left
    edge advances) and the final window DOES reach the last sample; a
    still-ongoing outage at the very end of a series is covered. The only
    thing this cannot see is an outage confined to the final
    `< window_seconds` of the series -- a real resolution floor, not a bug.
    """
    n = len(samples)
    right = 0
    for left in range(n):
        if right < left:
            right = left
        while right < n and (samples[right].at - samples[left].at).total_seconds() < window_seconds:
            right += 1
        if right >= n:
            break
        yield left, right + 1


def _find_silent_window_timestamp(
    channel: str, samples: list[ChannelSample], window_seconds: float
) -> WindowFinding | None:
    """Signal 1: reuse regime.py's `_refresh_from_timestamps` verbatim, at
    the REAL policy window -- no resolution floor, since a real write
    timestamp answers "was there a write in this window" directly rather
    than inferring it from shape."""
    for left, right in _rolling_windows(samples, window_seconds):
        window = samples[left:right]
        verdict = _refresh_from_timestamps(
            [s.node_stamp for s in window], window_start=window[0].at
        )
        if verdict == SILENT:
            return WindowFinding(
                channel=channel,
                kind=SILENT,
                evidence="timestamp",
                started_at=window[0].at,
                span_seconds=(window[-1].at - window[0].at).total_seconds(),
                tick_count=len(window),
            )
    return None


def _find_silent_run_contribution(
    channel: str, samples: list[ChannelSample], window_seconds: float
) -> tuple[WindowFinding | None, int]:
    """Signal 2: the longest contiguous run of `capability_fresh is False`.
    Returns the finding (if the run reaches window_seconds) and the longest
    run length in ticks regardless, so a trend toward the window is visible
    before it crosses."""
    run = 0
    run_start = 0
    longest = 0
    longest_start = 0
    for i, sample in enumerate(samples):
        if sample.capability_fresh:
            run = 0
            continue
        if run == 0:
            run_start = i
        run += 1
        if run > longest:
            longest, longest_start = run, run_start
    if not longest:
        return None, 0
    end = longest_start + longest - 1
    span = (samples[end].at - samples[longest_start].at).total_seconds()
    if span < window_seconds:
        return None, longest
    return (
        WindowFinding(
            channel=channel,
            kind=SILENT,
            evidence="contribution",
            started_at=samples[longest_start].at,
            span_seconds=span,
            tick_count=longest,
        ),
        longest,
    )


def analyse_credit_integrity(
    states: list[FieldStateV1],
    dimensions: dict[str, list[str]],
    window_seconds: float,
) -> CreditIntegrityReport:
    report = CreditIntegrityReport(window_seconds=window_seconds)
    by_channel = _samples_for(states, dimensions)

    for channel, samples in by_channel.items():
        report.channels[channel] = len(samples)
        if not dimensions.get(channel):
            report.findings.append(
                WindowFinding(
                    channel=channel,
                    kind="unmappable_dimension",
                    evidence="",
                    started_at=states[0].generated_at if states else _EPOCH,
                    span_seconds=0.0,
                    tick_count=0,
                )
            )
            continue
        if not samples:
            continue

        node_samples = [s for s in samples if s.source_kind == "node"]
        cap_samples = [s for s in samples if s.source_kind == "capability"]
        # "absent" ticks (the channel missing from the merge entirely) never
        # reach `samples` at all -- `_samples_for` only records a tick when
        # the credited DIMENSION won a value from `dims`, and the winning
        # channel `_classify_tick` receives is by construction always a key
        # of `merged` (`map_channels_to_dimensions_with_provenance` only
        # ever picks a winner from `channel_pressures.items()`). So
        # node_samples + cap_samples == samples always.
        report.source_kind[channel] = {
            "node": len(node_samples),
            "capability": len(cap_samples),
        }

        # BOTH signals run, independently, whenever they have any evidence to
        # check -- never a majority vote picking ONE for the whole channel.
        # An earlier draft of this function picked whichever mechanism had
        # more ticks across the WHOLE analysed batch and only ever scanned
        # that one, which review found reintroduces the exact defect class
        # this rebuild exists to fix, one level up: a channel whose winner
        # type is mostly capability with a real 30s NODE-sourced outage
        # buried in it had that entire stretch excluded from analysis by
        # BOTH signals, not merely misrouted to the wrong one. Confirmed via
        # a live-style repro (25 healthy capability ticks + 15 node ticks
        # containing a genuine, real 30s node write-outage -> 0 findings
        # under the majority-vote version). Running both independently means
        # a channel's minority-type stretch is still fully checked by its own
        # mechanism regardless of how few ticks it has, at the cost of
        # potentially two findings for one channel if both stretches are
        # separately bad -- strictly more informative than picking one.
        #
        # Residual, honestly stated: an outage that straddles the exact tick
        # where the winner type switches, with neither half alone reaching
        # window_seconds, is still invisible to both signals. Same shape as
        # `_rolling_windows`'s own documented resolution floor (an outage
        # confined to the final `< window_seconds` of a series) -- a real
        # limit, not silently hidden.
        if node_samples:
            finding = _find_silent_window_timestamp(channel, node_samples, window_seconds)
            if finding is not None:
                report.findings.append(finding)
        if cap_samples:
            finding, longest = _find_silent_run_contribution(channel, cap_samples, window_seconds)
            report.longest_unbacked_run[channel] = longest
            if finding is not None:
                report.findings.append(finding)
    return report
