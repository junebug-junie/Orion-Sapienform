"""What a curiosity run bought, and what the next one is expected to buy.

Pure functions, no I/O. P1 of
docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md.

THE UNIT IS NATS OF BELIEF CHANGE. Orion holds each prior at a confidence p
it wrote itself. When a run tests a prior and the confidence moves p -> q,
the run changed Orion's mind by KL(Bern(q) || Bern(p)) nats: Bayesian
surprise, the same "posterior against prior" quantity
`orion.autonomy.prediction.bayesian_surprise_nats` scores for dispatched
actions (Itti & Baldi 2009). A test that moved nothing scores exactly 0.0,
and that 0.0 is a result, not an absence.

MEASURED FROM HUB'S OWN SNAPSHOTS, NOT FROM :PriorRevision. Orion writes a
revision node only when a confidence moves; an inconclusive test bumps
`times_tested` and leaves no revision (`kickoff_prompt.py`, the TESTING
block). Scoring revisions alone would make "tested, did not move"
indistinguishable from "not recorded" -- 21 revisions across 94 journaled
runs by 2026-09-14. So Hub snapshots every prior's confidence and tested
count when a turn starts and again when it ends, and diffs.

ONLY CHANGES THIS RUN STAMPED COUNT. Durable runs can sit queued for hours and
other lines' turns can overlap, so "confidence changed between the two
snapshots" is not "this run changed it". Orion stamps `last_run_id` on a prior
it tests and `run_id` on a prior it forms; a change carrying another run's
stamp (or none) is counted as unattributed and not scored.

VALUE IS PROGRESS, NOT SURPRISE. The expected value of offering a prior is
its entropy H(p) -- the most one test can teach in expectation, for a
Bayes-consistent update -- times its learning yield: net belief change per
unit of uncertainty offered, over its last few tests. Net, because a prior
the LLM flips 0.3 <-> 0.7 is surprising on every test and learns nothing;
summing surprises would make it the most valuable prior forever (the noisy-TV
trap). Its net change over the window is ~0 -- and because a flip-flop
nets a full swing over any odd-length window, net change is also discounted
by the window's straightness (net displacement over path length) -- so its
value falls. Only progress a recorded test MADE and that was still there at
the end of the window is credited: a prior's scored tests are not a
continuous chain whenever something moved it between two of them. Yield is
pulled toward the pool average while a prior has few tests (empirical-Bayes
shrinkage), and with no history at all the pool yield is 1.0, which makes
value order identical to today's most-uncertain-first order.

`realized_nats` of None is always paired with an `unknown_reason`, so an
unknown run says why it is unknown.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

# Confidences are clamped away from 0 and 1 before any log: Orion does write
# 0.0 / 1.0 on refutation, and KL against an exact 0 or 1 is infinite. The
# clamp bounds a full reversal at ~4.5 nats and H at >= 0.056 nats.
CONFIDENCE_FLOOR = 0.01
CONFIDENCE_CEIL = 0.99

# With no scored history, every prior's yield is this, so value order equals
# entropy order equals the existing |p - 0.5| order. A cold start changes
# nothing about what Orion is shown.
COLD_POOL_YIELD = 1.0

ARM_VALUE_ORDER = "value_order"
ARM_UNCERTAINTY_ORDER = "uncertainty_order"

KIND_TESTED = "tested"
KIND_FORMED = "formed"
KIND_MOVED_UNTESTED = "moved_untested"

# Why a run's realized_nats is unknown. Recorded with the outcome, so the
# replay can tell a flaky graph from a protocol problem.
UNKNOWN_NO_START_SNAPSHOT = "no_start_snapshot"  # unreadable, over the row cap, or expired
UNKNOWN_NO_END_SNAPSHOT = "no_end_snapshot"
UNKNOWN_NO_RUN_ID = "no_run_id"
UNKNOWN_START_STAMPED = "start_stamped_by_this_run"  # an earlier attempt wrote first
UNKNOWN_NO_SCORABLE_TEST = "no_scorable_test"  # every tested prior had an unusable confidence


def valid_confidence(value: Any) -> Optional[float]:
    """A usable confidence, or None. Out-of-range or non-finite is None, not
    clamped: a belief Orion wrote as 1.7 is a protocol error to report, not a
    number to quietly repair."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        return None
    return v


def _clamp(p: float) -> float:
    return min(CONFIDENCE_CEIL, max(CONFIDENCE_FLOOR, p))


def entropy_nats(p: Optional[float]) -> float:
    """Binary entropy in nats. A prior with no usable confidence -- none, or
    one `valid_confidence` rejects -- reads as maximally uncertain (ln 2),
    the same honest reading `Prior.uncertainty` gives "Orion never said how
    sure it was". Validated here, not by callers, so a 1.7 can never read as
    near-certain through a caller that forgot."""
    p = valid_confidence(p)
    if p is None:
        return math.log(2.0)
    q = _clamp(p)
    return -(q * math.log(q) + (1.0 - q) * math.log(1.0 - q))


def kl_nats(after: float, before: float) -> float:
    """KL(Bern(after) || Bern(before)) in nats. Exactly 0.0 when the clamped
    values are equal; never negative."""
    q, p = _clamp(after), _clamp(before)
    if q == p:
        return 0.0
    kl = q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
    return max(kl, 0.0)


@dataclass(frozen=True)
class PriorState:
    """One prior as a snapshot saw it."""

    prior_id: str
    confidence: Optional[float]
    times_tested: int
    status: str = ""
    last_run_id: str = ""
    run_id: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "confidence": self.confidence,
            "times_tested": self.times_tested,
            "status": self.status,
            "last_run_id": self.last_run_id,
            "run_id": self.run_id,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Optional["PriorState"]:
        prior_id = str(row.get("prior_id") or "").strip()
        if not prior_id:
            return None
        try:
            # int(float(...)), same as worldview._as_int: FalkorDB hands numbers
            # back as strings and Orion sometimes writes "2.0".
            tested = int(float(row.get("times_tested") or 0))
        except (TypeError, ValueError, OverflowError):
            tested = 0
        return cls(
            prior_id=prior_id,
            confidence=valid_confidence(row.get("confidence")),
            times_tested=tested,
            status=str(row.get("status") or ""),
            last_run_id=str(row.get("last_run_id") or ""),
            run_id=str(row.get("run_id") or ""),
        )


def _fork_rank(state: PriorState) -> tuple:
    return (
        state.times_tested,
        -1.0 if state.confidence is None else state.confidence,
        state.last_run_id,
        state.run_id,
    )


def index_states(states: Iterable[PriorState]) -> dict[str, PriorState]:
    """By prior_id, most-tested copy of a fork first, ties broken on the
    copy's own values, never on read order. Graph snapshots are collapsed
    before this, with the offer's own rule (`read_prior_states` in Hub), so
    the copy scored is the copy Orion was shown; this is for already-unique
    snapshots and for callers without the raw rows."""
    out: dict[str, PriorState] = {}
    for state in states:
        seen = out.get(state.prior_id)
        if seen is None or _fork_rank(state) > _fork_rank(seen):
            out[state.prior_id] = state
    return out


@dataclass(frozen=True)
class PriorOutcome:
    prior_id: str
    kind: str
    before: Optional[float]
    after: Optional[float]
    tested_delta: int
    nats: Optional[float]

    def as_json(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "tested_delta": self.tested_delta,
            "nats": self.nats,
        }


@dataclass(frozen=True)
class RunOutcome:
    """What one run changed. `realized_nats` is None when it cannot be known,
    with `unknown_reason` saying why -- unknown is never reported as zero."""

    realized_nats: Optional[float]
    unknown_reason: Optional[str] = None
    per_prior: tuple[PriorOutcome, ...] = ()
    n_tested: int = 0
    n_moved: int = 0
    n_formed: int = 0
    n_moved_untested: int = 0
    n_unattributed: int = 0
    n_invalid_confidence: int = 0

    @property
    def moved(self) -> dict[str, tuple[float, float]]:
        """prior_id -> (before, after) for tested priors whose confidence moved."""
        return {
            o.prior_id: (o.before, o.after)
            for o in self.per_prior
            if o.kind == KIND_TESTED
            and o.before is not None
            and o.after is not None
            and o.before != o.after
        }


def diff_snapshots(
    before: Optional[Mapping[str, PriorState]],
    after: Optional[Mapping[str, PriorState]],
    *,
    run_id: str,
) -> RunOutcome:
    """Score one run from the snapshot taken when its turn started and the one
    taken when it ended. See the module docstring for the attribution rule.

    A start snapshot that already carries this run's own stamps is not a
    start: an earlier attempt of the same run wrote to the graph before it was
    taken (that attempt's own start was never recorded). Scoring from there
    would report part of the run as all of it, so the answer is unknown."""
    if before is None:
        return RunOutcome(realized_nats=None, unknown_reason=UNKNOWN_NO_START_SNAPSHOT)
    if after is None:
        return RunOutcome(realized_nats=None, unknown_reason=UNKNOWN_NO_END_SNAPSHOT)
    if not run_id:
        return RunOutcome(realized_nats=None, unknown_reason=UNKNOWN_NO_RUN_ID)
    if any(s.last_run_id == run_id or s.run_id == run_id for s in before.values()):
        return RunOutcome(realized_nats=None, unknown_reason=UNKNOWN_START_STAMPED)
    outcomes: list[PriorOutcome] = []
    total = 0.0
    n_scored = 0
    n_tested = n_moved = n_formed = n_moved_untested = n_unattributed = n_invalid = 0
    for prior_id in sorted(after):
        now = after[prior_id]
        was = before.get(prior_id)
        if was is None:
            if now.run_id == run_id:
                n_formed += 1
                outcomes.append(
                    PriorOutcome(prior_id, KIND_FORMED, None, now.confidence, now.times_tested, None)
                )
            else:
                n_unattributed += 1
            continue
        delta = now.times_tested - was.times_tested
        changed = now.confidence != was.confidence
        if delta <= 0 and not changed:
            continue
        if now.last_run_id != run_id:
            n_unattributed += 1
            continue
        if was.confidence is None or now.confidence is None:
            nats = None
            n_invalid += 1
        else:
            nats = kl_nats(now.confidence, was.confidence)
        if delta > 0:
            n_tested += 1
            if changed:
                n_moved += 1
            if nats is not None:
                total += nats
                n_scored += 1
            outcomes.append(
                PriorOutcome(prior_id, KIND_TESTED, was.confidence, now.confidence, delta, nats)
            )
        else:
            # Moved without a test being counted: the prompt asks for
            # `times_tested + 1` on every test, so this is protocol drift.
            # Reported with its nats for audit, not summed.
            n_moved_untested += 1
            outcomes.append(
                PriorOutcome(prior_id, KIND_MOVED_UNTESTED, was.confidence, now.confidence, 0, nats)
            )
    # Tested priors, none of them scorable (a confidence unusable before or
    # after every test): what the run bought is unknown, not zero.
    unscorable = bool(n_tested) and not n_scored
    return RunOutcome(
        realized_nats=None if unscorable else total,
        unknown_reason=UNKNOWN_NO_SCORABLE_TEST if unscorable else None,
        per_prior=tuple(outcomes),
        n_tested=n_tested,
        n_moved=n_moved,
        n_formed=n_formed,
        n_moved_untested=n_moved_untested,
        n_unattributed=n_unattributed,
        n_invalid_confidence=n_invalid,
    )


def revision_agreement(
    outcome: RunOutcome, revisions: Mapping[str, tuple[Optional[float], Optional[float]]]
) -> Optional[float]:
    """Share of the moves Hub measured that Orion also recorded as a
    `:PriorRevision` with the same from/to. None when Hub measured no moves.
    Below 1.0 is informative, not an error: a revision is written by hand."""
    moved = outcome.moved
    if not moved:
        return None
    agree = 0
    for prior_id, (before, after) in moved.items():
        rev = revisions.get(prior_id)
        if rev is None or rev[0] is None or rev[1] is None:
            continue
        if abs(rev[0] - before) <= 1e-6 and abs(rev[1] - after) <= 1e-6:
            agree += 1
    return agree / len(moved)


@dataclass(frozen=True)
class PriorTestRecord:
    """One run's net effect on one prior it tested."""

    prior_id: str
    before: float
    after: float


def prior_tests_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[PriorTestRecord]:
    """Tested priors with two usable confidences, in the order given."""
    out: list[PriorTestRecord] = []
    for row in rows:
        if row.get("kind") != KIND_TESTED:
            continue
        before = valid_confidence(row.get("before"))
        after = valid_confidence(row.get("after"))
        prior_id = str(row.get("prior_id") or "").strip()
        if prior_id and before is not None and after is not None:
            out.append(PriorTestRecord(prior_id, before, after))
    return out


def _credited_net(tests: Sequence[PriorTestRecord]) -> float:
    """The part of the window's net belief change that the recorded tests
    made AND that was still there at the end.

    A prior's scored tests are not a continuous chain whenever something
    moved it BETWEEN two of them -- a self-inquiry turn (never measured), an
    unstamped edit, a run scored unknown. Two nets can then disagree:

    - observed, `last.after - first.before`: where the belief actually went.
      A jump between tests is not the tests' doing, so this alone credits a
      forward jump as progress ([0.5->0.5], gap, [0.9->0.9] "learned" 0.4).
    - recorded, the sum of the tests' own moves. A jump that UNDID a test
      means that progress did not stick, so this alone credits a belief the
      tests keep pushing up and something keeps knocking back down.

    Credit the smaller, when both point the same way; otherwise nothing. The
    credited end point then always lies between the first tested value and
    the last observed one."""
    recorded = sum(t.after - t.before for t in tests)
    observed = tests[-1].after - tests[0].before
    if recorded * observed <= 0.0:
        return 0.0
    return math.copysign(min(abs(recorded), abs(observed)), recorded)


def straightness(tests: Sequence[PriorTestRecord]) -> float:
    """Net displacement over path length of the belief's whole trajectory in
    the window -- the tests' moves AND any jumps between them -- in confidence
    units: 1.0 for a belief that moved one way, 1/3 for one that flipped
    0.7 -> 0.3 -> 0.7 -> 0.3, 1.0 for one that never moved (no wasted motion
    to discount; its net KL is already 0). Always in [0, 1]. The straightness
    index of movement ecology (Batschelet 1981; Benhamou 2004), used here
    because net KL alone is parity-blind: a period-2 flip-flop nets one full
    swing over any odd window, which would read as real progress."""
    path = 0.0
    for i, t in enumerate(tests):
        path += abs(t.after - t.before)
        if i + 1 < len(tests):
            path += abs(tests[i + 1].before - t.after)
    if path <= 0.0:
        return 1.0
    return min(1.0, abs(tests[-1].after - tests[0].before) / path)


def _window_progress_and_offered(tests: Sequence[PriorTestRecord]) -> tuple[float, float]:
    """(credited net KL x straightness, uncertainty offered) over one window.
    Uncertainty is charged per test, at the confidence it was offered at."""
    start = tests[0].before
    progress = kl_nats(start + _credited_net(tests), start) * straightness(tests)
    offered = sum(entropy_nats(t.before) for t in tests)
    return progress, offered


def raw_yield(tests: Sequence[PriorTestRecord]) -> Optional[float]:
    """Directed net belief change per unit of uncertainty offered, over
    `tests` (oldest first). Clipped at 1: in expectation one Bayes-consistent
    test teaches at most H(p), but a single realized reversal can exceed it."""
    if not tests:
        return None
    progress, offered = _window_progress_and_offered(tests)
    if offered <= 0.0:
        return None
    return min(1.0, progress / offered)


@dataclass(frozen=True)
class YieldModel:
    """Per-prior learning yield from scored history. Build with
    `build_yield_model`; an empty model is the cold start."""

    window: int
    pseudo_tests: float
    pool_yield: float
    history_tests: int = 0
    raw_by_prior: Mapping[str, tuple[float, int]] = field(default_factory=dict)

    def yield_for(self, prior_id: str) -> float:
        raw_n = self.raw_by_prior.get(prior_id)
        if raw_n is None:
            return self.pool_yield
        raw, n = raw_n
        k = self.pseudo_tests
        return (n * raw + k * self.pool_yield) / (n + k) if (n + k) > 0 else self.pool_yield

    def expected_nats(self, prior_id: str, confidence: Optional[float]) -> float:
        return entropy_nats(confidence) * self.yield_for(prior_id)

    def constants(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "pseudo_tests": self.pseudo_tests,
            "pool_yield": self.pool_yield,
            "history_tests": self.history_tests,
            "confidence_floor": CONFIDENCE_FLOOR,
            "confidence_ceil": CONFIDENCE_CEIL,
            "cold_pool_yield": COLD_POOL_YIELD,
        }


def build_yield_model(
    history: Iterable[PriorTestRecord], *, window: int, pseudo_tests: float
) -> YieldModel:
    """`history` oldest first. Each prior's last `window` tests give its raw
    yield; the pool yield is directed progress over offered, summed across
    every prior's window (a ratio of sums, so a prior with one lucky test
    cannot dominate)."""
    window = max(1, int(window))
    by_prior: dict[str, list[PriorTestRecord]] = {}
    count = 0
    for record in history:
        by_prior.setdefault(record.prior_id, []).append(record)
        count += 1
    raw_by_prior: dict[str, tuple[float, int]] = {}
    pool_net = pool_offered = 0.0
    for prior_id, tests in by_prior.items():
        recent = tests[-window:]
        progress, offered = _window_progress_and_offered(recent)
        if offered <= 0.0:
            continue
        raw_by_prior[prior_id] = (min(1.0, progress / offered), len(recent))
        pool_net += progress
        pool_offered += offered
    pool = min(1.0, pool_net / pool_offered) if pool_offered > 0.0 else COLD_POOL_YIELD
    return YieldModel(
        window=window,
        pseudo_tests=max(0.0, float(pseudo_tests)),
        pool_yield=pool,
        history_tests=count,
        raw_by_prior=raw_by_prior,
    )


def offer_arm(run_id: str, *, enabled: bool, propensity: float) -> tuple[str, float]:
    """(arm, P(value arm)) for one run. Deterministic in run_id so any run's
    assignment can be re-derived; run ids are random, so this is a fair coin.
    Disabled means every run is the uncertainty arm with P(value) = 0."""
    p = min(1.0, max(0.0, float(propensity))) if enabled else 0.0
    if p <= 0.0:
        return ARM_UNCERTAINTY_ORDER, 0.0
    digest = hashlib.sha256(f"curiosity-offer-arm:{run_id}".encode("utf-8")).hexdigest()
    draw = int(digest[:8], 16) / float(2**32)  # [0, 1): propensity 1.0 always assigns value
    return (ARM_VALUE_ORDER if draw < p else ARM_UNCERTAINTY_ORDER), p
