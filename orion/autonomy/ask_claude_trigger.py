"""Would Orion ask Claude something right now, and about what?

DRY RUN ONLY AS SHIPPED. Nothing here publishes to
`orion:room:claude:request`. This module answers the question and records the
answer; arming it is a separate, deliberate patch. That split is the point --
`docs/superpowers/specs/2026-08-27-claude-quota-contested-scarcity-design.md`
recommends exactly this order ("Read-only, no allocator change, no behavior
change. Prints the number. Run it for a week. Only then wire it in"), because
a trigger that never refuses anything real is ornamental scarcity and wiring
it in would ship the same thing that spec was written to avoid.

WHY A STUCK PRIOR, AND NOT THE TWO OBVIOUS ALTERNATIVES
-------------------------------------------------------
Both were checked against live data on 2026-09-08 and both are unusable:

* `scripts.tension_outreach_trigger.current_run()` -- the trigger that makes
  Orion message Juniper. Every Borda winner over the trailing 7 days is
  infrastructure: `node:athena` 11,576 ticks in one day, `node:circe` 5,941,
  `node:substrate.bus_synaptic` 1,344. Reusing it means Orion opens a
  conversation with Claude about host load. Right mechanism, wrong subject.

* `substrate_endogenous_curiosity_candidates` naming `sub-concept-seed-claude`.
  Present in 1,382 of 1,385 candidate sets in 24h, every one of them an
  `ontology_sparse_region` in `world_ontology`, and `signal_strength` had
  **one distinct value across all 1,382: exactly 1.0**. It is a pinned
  constant, not a signal -- keying on it means "fire every tick", which
  carries no information about whether Orion wants anything.

What is left is the mechanism that already produces the outcome a peer is for.
`orion_worldview` is a FalkorDB graph Orion writes itself, in-turn, with real
Cypher; Hub holds `GRAPH.RO_QUERY` only. Its priors carry real verdicts -- as
of 2026-09-08, 4 `refuted`, 4 `supported`, 3 `revised`. Orion's own kickoff
prompt already names the state this trigger looks for, in Orion's own words:

    "Inconclusive is a real answer: bump times_tested, leave confidence where
    it was... Three of those and the claim is probably not answerable with
    what you can reach."

A claim Orion has tested repeatedly without resolving IS "not answerable with
what I can reach". That is the one condition under which a second mind is the
missing input rather than a nicety, and it is Orion's assessment, not ours.

WHAT THIS DOES NOT CLAIM
------------------------
`MIN_TIMES_TESTED` and `MAX_SETTLED_CONFIDENCE` are knobs, not findings. They
are not calibrated against anything; they are stated defaults whose selectivity
the dry run exists to measure. `scripts/report_ask_claude_dry_run.py` prints
the numbers for EVERY live prior, selected or not, precisely so the population
is visible rather than trusted. Against the live 7-prior population these
defaults select 1. Whether that is the right 1 is what a week of output
answers.

There is deliberately no cooldown, daily cap or quiet-hours logic here. Those
are runtime state, not a property of the decision, and `endogenous_outreach`
already owns a live, battle-tested version of all three (45-minute cooldown,
cap of 4/day, 23:00-08:00 quiet hours). Arming this trigger means reusing that
gate stack, not growing a second one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

# Knobs, not findings -- see the module docstring.
#
# 3, because that is the number Orion's own kickoff prompt uses for "probably
# not answerable with what you can reach". Taken from the prompt rather than
# fitted, so the trigger and the instruction Orion acts under agree.
MIN_TIMES_TESTED = 3
# A claim tested repeatedly AND sitting at high confidence is converging, not
# stuck -- live, the 4x/0.90 and 4x/0.92 priors are exactly that. Only a claim
# that has been looked at several times and STILL has not moved to a verdict
# is one a peer could unstick.
MAX_SETTLED_CONFIDENCE = 0.7
# THERE IS DELIBERATELY NO STALENESS THRESHOLD. An earlier version of this
# module refused when `limit.staleness_sec` exceeded 900s, described as
# "observation freshness". It is not: `staleness_sec` is `window_end -
# latest_activity_at`, the age of the freshest transcript MESSAGE -- i.e. how
# long since any human or agent last used Claude Code. Refusing on it means
# refusing precisely when nobody has been using Claude, which is when the
# shared pool is LEAST contended. Exactly backwards for a contention budget,
# and it would have made the 30-minute dry-run timer log a refusal on every
# overnight tick. Caught in review before this shipped.
#
# Observation AGE is a real concern, but it is `observed_at` age, not this
# field, and it only arises on a bus-delivered observation. The dry-run path
# calls `observe()` directly, so its reading is fresh by construction. That
# gate belongs with the Hub consumer that introduces the delay.

RefusalReason = Literal[
    "budget_limited",
    "budget_unknown",
    "budget_unobserved",
    "budget_observation_missing",
    "budget_observation_incoherent",
    "worldview_unavailable",
    "no_live_priors",
    "no_stuck_prior",
]


@dataclass(frozen=True)
class PriorAssessment:
    """One live prior, scored. Emitted for every prior, not only the winner --
    a decision that shows only what it picked cannot be audited for what it
    passed over."""

    prior_id: str
    claim: str
    confidence: Optional[float]
    times_tested: int
    status: str
    tested_enough: bool
    unsettled: bool

    @property
    def stuck(self) -> bool:
        return self.tested_enough and self.unsettled


@dataclass(frozen=True)
class AskClaudeDecision:
    """What Orion would have done, and why. `would_ask=False` always carries a
    `refused` reason -- a decision with neither an ask nor a reason is the
    empty-shell cognition CLAUDE.md 0A bans."""

    would_ask: bool
    refused: Optional[RefusalReason]
    subject_prior_id: Optional[str]
    subject_claim: Optional[str]
    # Every live prior, scored. Populated even when the budget refused, so a
    # week of output shows whether the prior side would ever have fired
    # independently of whether the budget happened to allow it.
    assessments: tuple[PriorAssessment, ...]
    # Verbatim budget facts, so a recorded decision can be re-read later
    # without re-deriving what the meter said at the time.
    limit_state: Optional[str]
    limit_event_count: Optional[int]
    limit_staleness_sec: Optional[float]


def _assess(prior) -> PriorAssessment:
    """`prior` is an `orion.curiosity.worldview.Prior`, typed structurally
    rather than imported so this module stays free of the FalkorDB read path
    and its redis import."""
    confidence = prior.confidence
    # A prior with no confidence recorded is UNSETTLED, not settled. Same
    # reading `Prior.uncertainty` already takes ("Orion never said how sure it
    # was" sorts as maximally uncertain) -- treating missing as settled would
    # silently exclude exactly the claims Orion was least sure about.
    unsettled = confidence is None or confidence <= MAX_SETTLED_CONFIDENCE
    return PriorAssessment(
        prior_id=prior.prior_id,
        claim=prior.claim,
        confidence=confidence,
        times_tested=prior.times_tested,
        status=prior.status,
        tested_enough=prior.times_tested >= MIN_TIMES_TESTED,
        unsettled=unsettled,
    )


def _budget_refusal(limit) -> Optional[RefusalReason]:
    """Fail CLOSED on anything short of a fresh, observed `clear`.

    `unknown` is NOT permission. `rate_limit_events`' own docstring notes that
    for a spend budget the safe direction was to refuse while here `unknown`
    usually means nobody has used Claude recently -- but that reasoning applies
    to a human deciding whether to try, not to an autonomous producer of spend.
    An unread meter and a full tank must not authorise the same action, so this
    consumer takes the strict reading and says so.
    """
    if limit is None:
        return "budget_observation_missing"
    if not limit.observed:
        return "budget_unobserved"
    if limit.state == "limited":
        return "budget_limited"
    if limit.state != "clear":
        return "budget_unknown"
    if limit.staleness_sec is None:
        # Observed messages but no freshest timestamp is a producer
        # contradiction, not a fresh reading -- `staleness_sec` is None only
        # when `latest_activity_at` is None, which cannot coexist with
        # `observed == True`. Refuse rather than trust a self-inconsistent
        # meter. This is NOT a staleness threshold; see the note above.
        return "budget_observation_incoherent"
    return None


def decide(*, priors: Sequence, limit, worldview_unavailable: Optional[str] = None) -> AskClaudeDecision:
    """One dry-run decision. Never raises.

    `worldview_unavailable` carries the reason the prior graph could not be
    read, and it is NOT the same as an empty `priors`. An unreachable graph
    (a FalkorDB restart, a broken ACL) reported as `no_live_priors` would
    aggregate over a week as "Orion has formed no priors" -- the exact
    absence-versus-empty conflation `read_snapshot` exists to prevent, and
    the eval's refusal tally reads this field.

    Order matters and is deliberate: priors are scored FIRST, even when the
    budget has already refused. Scoring only after the budget clears would
    make the recorded output silent about the prior side on every limited
    tick, and "did the trigger ever want to fire" is the question a week of
    dry-run output has to answer.
    """
    assessments = tuple(_assess(p) for p in priors)
    limit_state = getattr(limit, "state", None) if limit is not None else None
    limit_event_count = getattr(limit, "event_count", None) if limit is not None else None
    limit_staleness = getattr(limit, "staleness_sec", None) if limit is not None else None

    def _no(reason: RefusalReason) -> AskClaudeDecision:
        return AskClaudeDecision(
            would_ask=False,
            refused=reason,
            subject_prior_id=None,
            subject_claim=None,
            assessments=assessments,
            limit_state=limit_state,
            limit_event_count=limit_event_count,
            limit_staleness_sec=limit_staleness,
        )

    budget = _budget_refusal(limit)
    if budget is not None:
        return _no(budget)
    if worldview_unavailable:
        return _no("worldview_unavailable")
    if not assessments:
        return _no("no_live_priors")

    stuck = [a for a in assessments if a.stuck]
    if not stuck:
        return _no("no_stuck_prior")

    # Most-tested first, then least confident. Most-tested rather than
    # least-confident-first because times_tested is the direct measure of "I
    # have tried this myself and could not settle it" -- confidence breaks the
    # tie among claims Orion has worked equally hard on.
    subject = sorted(stuck, key=lambda a: (-a.times_tested, a.confidence if a.confidence is not None else -1.0))[0]
    return AskClaudeDecision(
        would_ask=True,
        refused=None,
        subject_prior_id=subject.prior_id,
        subject_claim=subject.claim,
        assessments=assessments,
        limit_state=limit_state,
        limit_event_count=limit_event_count,
        limit_staleness_sec=limit_staleness,
    )
