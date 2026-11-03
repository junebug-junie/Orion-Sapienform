"""The self-monitor organ's contracts: Orion's settled bets and per-faculty track record.

Spec: docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md (rung 6, metacognition).

A *bet* is anything Orion said about the future before the world answered: an act's predicted
chance of working, a forecast's 80% range, stance's guess about its own turn. ``SettledBetV1`` is
one shape for all of them; one pure adapter per source makes it (``orion/self_monitor/adapters.py``).
``FacultyRecordV1`` is what the organ knows about one faculty, reduced from its bets
(``orion/self_monitor/record.py``). ``SelfMonitorStateV1`` is the whole picture at one moment.

Two rules from the 2026-10-20 live audit, enforced by the reducer:

* ``failed_to_execute`` is not ``refuted``: an abandoned painting run says "my pipeline doesn't
  run", not "I predicted badly". Execution reliability is tracked apart from reliability.
* A bet settled by Orion's own re-judgment (``settled_by="self"``) never counts as calibration
  evidence.

A faculty with no usable evidence is not hidden: it is ``ungradeable`` with an enumerated reason
and the fix that would unlock it.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

SETTLED_BET_KIND = "self_monitor.settled_bet.v1"
FACULTY_RECORD_KIND = "self_monitor.faculty_record.v1"
SELF_MONITOR_STATE_KIND = "self_monitor.state.v1"

# success:  p that this act of mine works (the event is my success).
# event:    p that something happens (asking a question, a subject showing up) -- not a success.
# interval: a range stated at a coverage level (stated = the coverage, e.g. 0.8).
ClaimKind = Literal["success", "event", "interval"]
BetOutcome = Literal["upheld", "refuted", "failed_to_execute", "no_change", "expired_unsettled"]
SettledBy = Literal["world", "juniper", "self", "timer"]
Gradeability = Literal[
    "graded",
    "ungradeable:no_stated_confidence",
    "ungradeable:never_settled",
    "ungradeable:self_graded",
    "ungradeable:too_few",
    "ungradeable:source_bug",
]
FacultyStatus = Literal["reliable", "unreliable", "broken", "miscalibrated", "unknown"]


class SettledBetV1(BaseModel):
    """One bet Orion made, and how it came out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bet_id: str = Field(..., min_length=1, description="Source-qualified id, stable: never counted twice.")
    faculty: str = Field(..., min_length=1, description="e.g. act.make.journal_trigger.walkway_grade")
    source: str = Field(..., min_length=1, description="Adapter that made it, e.g. grammar_acts")
    made_at: AwareDatetime
    settled_at: AwareDatetime
    claim_kind: ClaimKind
    stated: Optional[float] = Field(
        None,
        description=(
            "Confidence recorded BEFORE the outcome: p for success/event bets, the stated "
            "coverage for interval bets. None when nothing was recorded -- never reconstructed."
        ),
    )
    belief_key: Optional[str] = Field(
        None,
        description=(
            "The key the stated confidence was formed at, when coarser than the faculty (the act "
            "ledger's posterior is per verb|actor|signal, shared by every trigger name). Calibration "
            "is only judged where one faculty owns its belief."
        ),
    )
    outcome: BetOutcome = Field(
        ..., description="upheld = the claimed event happened (inside the range, for intervals)."
    )
    settled_by: SettledBy
    evidence_ref: str = Field(..., min_length=1, description="Pointer back to the source row.")


class FacultyRecordV1(BaseModel):
    """What the organ knows about one faculty, from its settled bets in the window."""

    model_config = ConfigDict(extra="forbid")

    faculty: str
    sources: List[str] = Field(default_factory=list)
    claim_kind: Optional[ClaimKind] = None
    gradeability: Gradeability
    unlock: Optional[str] = Field(None, description="For an ungradeable faculty: the fix that would grade it.")
    window_start: AwareDatetime
    window_end: AwareDatetime
    n_bets: int = Field(..., ge=0, description="All bets in the window, every outcome.")
    n_graded: int = Field(..., ge=0, description="upheld + refuted, settled by the world or Juniper.")
    n_failed_to_execute: int = Field(..., ge=0)
    n_expired: int = Field(..., ge=0)
    weight_graded: float = Field(..., ge=0.0, description="Sum of decay weights over graded bets.")
    reliability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Decay-weighted upheld share.")
    reliability_lo: Optional[float] = Field(None, ge=0.0, le=1.0)
    reliability_hi: Optional[float] = Field(None, ge=0.0, le=1.0)
    execution_failure: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Decay-weighted failed_to_execute share of attempts."
    )
    execution_failure_lo: Optional[float] = Field(None, ge=0.0, le=1.0)
    stated_mean: Optional[float] = Field(
        None, description="Mean stated confidence over graded bets that stated one."
    )
    brier: Optional[float] = None
    brier_skill: Optional[float] = Field(None, description="1 - Brier / base-rate Brier; None if undefined.")
    streak_kind: Optional[Literal["upheld", "refuted", "failed_to_execute"]] = None
    streak_len: int = Field(0, ge=0, description="Consecutive most-recent bets with the same outcome.")
    streak_since: Optional[AwareDatetime] = None
    candidate_status: FacultyStatus = Field(..., description="What this window alone says.")
    status: FacultyStatus = Field(..., description="Held status, after hysteresis.")
    status_since: Optional[AwareDatetime] = None
    candidate_held: int = Field(0, ge=0, description="Consecutive evaluations the candidate has held.")
    reasons: List[str] = Field(default_factory=list, description="Why the candidate status, in short tokens.")


class SelfMonitorStateV1(BaseModel):
    """The organ's whole picture at one moment."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["self_monitor.state.v1"] = SELF_MONITOR_STATE_KIND
    computed_at: AwareDatetime
    window_days: float = Field(..., gt=0.0)
    faculties: List[FacultyRecordV1] = Field(default_factory=list)
    source_errors: List[str] = Field(
        default_factory=list, description="Sources that could not be read this pass (named, never silent)."
    )
