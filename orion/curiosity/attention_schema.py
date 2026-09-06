"""Curiosity -> AttentionSchemaV1 projection (write-only adapter).

Settles the design doc's Missing Question 1 ("is curiosity's choice-reason
available, or must it be asked for?") from a real run, read live 2026-09-06
against Orion's own graph (`orion_worldview`, run `3b2d038cf18e`):

- WHAT Orion attended IS recoverable. The kickoff prompt has Orion stamp
  `p.last_run_id = "<RUN_ID>"` on a prior it tests, `p.run_id` on a prior it
  forms, and write a `:PriorRevision {run_id, from_confidence, to_confidence}`
  when a belief moves. Run `3b2d038cf18e` tested
  `substrate_domain_isolation_two_paths`, 0.95 -> 0.92, and that is exactly
  what the query below returns.
- WHY Orion chose that prior over the others on the menu is NOT recorded
  anywhere. `p.why` is why the prior was *formed*, `TurnOutcome.continue_note`
  is what Orion decided about *continuing* (that is `predicted_next`, and it
  is Orion's own words). Capturing the choice-reason means changing the
  kickoff prompt -- a live cognition-loop change, proposal mode per CLAUDE.md
  sec 0A -- and is deliberately NOT done here. So this lane's
  `reason_narrative` is computed from graph facts (which prior, how far the
  confidence moved), varies per run, and is honestly `narrative_kind=
  "computed"`. The design doc's Acceptance Check 3 ("choice becomes
  recoverable") is therefore met for the *what* and open for the *why*; the
  PR that ships this says so rather than folding the two together.

Never raises on the read path: `None` from `read_attended_priors` means the
graph could not answer, which the projection names as `graph_unreadable` --
not the same state as "Orion touched no prior" (`no_prior_touched`), the
unreadable-vs-empty distinction `worldview.py` refuses to collapse everywhere
else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from orion.curiosity.worldview import (
    _RUN_ID_RE,
    TurnOutcome,
    WorldviewReader,
    WorldviewUnavailable,
)
from orion.schemas.attention_schema import (
    MAX_LABEL_CHARS,
    MAX_NARRATIVE_CHARS,
    MAX_PREDICTED_NEXT_CHARS,
    AttentionSchemaV1,
    clip,
)

logger = logging.getLogger("orion.curiosity.attention_schema")


@dataclass(frozen=True)
class AttendedPrior:
    prior_id: str
    claim: str
    status: str
    confidence: Optional[float]
    touched: str  # "tested" (p.last_run_id == run) | "formed" (p.run_id == run)
    from_confidence: Optional[float] = None
    to_confidence: Optional[float] = None


def attended_priors_cypher(run_id: str) -> str:
    """Priors this run stamped, with the revision it wrote if any.

    Same hex guard as every other per-run Cypher in worldview.py: the run id
    is interpolated into a string literal, so a non-hex value is refused
    rather than quoted.
    """
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (p:Prior) WHERE p.last_run_id = '{run_id}' OR p.run_id = '{run_id}' "
        f"OPTIONAL MATCH (r:PriorRevision {{prior_id: p.prior_id, run_id: '{run_id}'}}) "
        "RETURN p.prior_id AS prior_id, p.claim AS claim, p.status AS status, "
        "p.confidence AS confidence, p.run_id AS run_id, p.last_run_id AS last_run_id, "
        "r.from_confidence AS from_confidence, r.to_confidence AS to_confidence"
    )


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_attended_priors(rows: Sequence[dict], run_id: str) -> list[AttendedPrior]:
    out: list[AttendedPrior] = []
    for row in rows:
        prior_id = str(row.get("prior_id") or "").strip()
        if not prior_id:
            continue
        touched = "tested" if str(row.get("last_run_id") or "") == run_id else "formed"
        out.append(
            AttendedPrior(
                prior_id=prior_id,
                claim=str(row.get("claim") or "").strip(),
                status=str(row.get("status") or "").strip(),
                confidence=_as_float(row.get("confidence")),
                touched=touched,
                from_confidence=_as_float(row.get("from_confidence")),
                to_confidence=_as_float(row.get("to_confidence")),
            )
        )
    # Tested priors first: a run that both tests one and forms one was
    # attending to the one it tested; the new one is what it left behind.
    out.sort(key=lambda p: (p.touched != "tested", p.prior_id))
    return out


def read_attended_priors(reader: WorldviewReader, run_id: str) -> Optional[list[AttendedPrior]]:
    """`[]` means the run touched no prior; `None` means the graph could not answer."""
    try:
        rows = reader.query(attended_priors_cypher(run_id))
    except (WorldviewUnavailable, ValueError) as exc:
        logger.warning("curiosity_attended_priors_read_failed run=%s err=%s", run_id, exc)
        return None
    return build_attended_priors(rows, run_id)


def _fmt(value: Optional[float]) -> str:
    return "?" if value is None else f"{value:.2f}"


def to_attention_schema(
    *,
    run_id: str,
    outcome: Optional[TurnOutcome],
    priors: Optional[Sequence[AttendedPrior]],
    correlation_id: Optional[str],
    generated_at: datetime,
) -> AttentionSchemaV1:
    """One row per investigation run. Pure; never touches I/O."""
    attended: Optional[AttendedPrior] = None
    confidence: Optional[float] = None
    basis: Optional[str] = None

    if priors is None:
        reason = "graph_unreadable"
        narrative = (
            "Orion's own graph could not be read after this run, so which prior it "
            "attended to is unknown -- not the same as none."
        )
    elif not priors:
        reason = "no_prior_touched"
        narrative = (
            "Orion neither tested nor formed a prior this run; what it attended to "
            "survives only in the finding prose, not as a stamped node."
        )
    else:
        attended = priors[0]
        others = len(priors) - 1
        tail = f" {others} more prior{'s' if others != 1 else ''} touched in the same run." if others else ""
        if attended.touched == "tested":
            reason = "tested_held_prior"
            narrative = (
                f"Orion tested a prior it already held: '{attended.claim}'. Confidence "
                f"{_fmt(attended.from_confidence)} -> {_fmt(attended.to_confidence or attended.confidence)}, "
                f"status now '{attended.status}'.{tail}"
            )
        else:
            reason = "formed_new_prior"
            narrative = (
                f"Orion formed a new prior: '{attended.claim}' at confidence "
                f"{_fmt(attended.confidence)}.{tail}"
            )
        confidence = attended.to_confidence if attended.to_confidence is not None else attended.confidence
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
            basis = "prior.confidence -- Orion's own belief after the test, self-assigned"

    predicted = None
    if outcome is not None and outcome.continue_note:
        predicted = clip(outcome.continue_note, MAX_PREDICTED_NEXT_CHARS) or None

    return AttentionSchemaV1(
        entry_id=f"curiosity-{run_id}",
        generated_at=generated_at,
        process="curiosity",
        correlation_id=correlation_id,
        attended_id=attended.prior_id if attended is not None else None,
        attended_label=clip(attended.claim if attended is not None else "", MAX_LABEL_CHARS),
        attention_reason=reason,
        reason_narrative=clip(narrative, MAX_NARRATIVE_CHARS),
        narrative_kind="computed",
        confidence=confidence,
        confidence_basis=basis,
        predicted_next=predicted,
    )
