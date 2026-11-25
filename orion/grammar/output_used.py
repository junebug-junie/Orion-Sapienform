"""The ``output_used`` act signal: another organ used what this act made (act consequence bets, Patch 1a).

Spec: docs/superpowers/specs/2026-11-25-act-consequence-bets-design.md (PR #3953).

An act that bets ``output_used`` is not closed by its producer's "I made it" report. A producer close
with value 1 only records that the output exists; the act stays open. The resolver closes it:

* **used** (1, ``resolver``) at the first use inside the horizon, from either link path:
  (a) an ``experience_use`` row for a ``temporal_self_event`` carrying the act's ``correlation_id``
  (the now never shows a reader its own organ's experiences, so every such row is another organ's
  use), or (b) the act's ``act_downstream_use`` row with ``used``.
* **unused** (0, ``expired``) at the horizon, when the output was observable: it landed as an
  experience, or its downstream-use row is ``linked``.
* **unobservable** (void, never graded) at the horizon, when neither link path exists: no reader's
  use could ever have been seen, so a 0 would teach "nobody used it" from an act nobody could
  watch. Closed with a ``void:`` note, which ``close_act`` keeps out of the posterior.
* **no_output** (0, ``producer``): the producer closed with value 0. Nothing was there to use: a
  real 0, graded.

Pure functions here; the SQL is run by ``orion.grammar.act_ledger``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from orion.grammar.promise_acts import VOID_NOTE_PREFIX
from orion.schemas.grammar import ActOutcomeV1

OUTPUT_USED_SIGNAL = "output_used"

USED_NOTE = "used: another organ used this act's output"
UNUSED_NOTE = "unused: output was observable, no other organ used it before horizon"
UNOBSERVABLE_NOTE = f"{VOID_NOTE_PREFIX}unobservable no link path to see a use (censored, not graded)"
NO_OUTPUT_NOTE = "no_output: producer reported nothing made"


@dataclass(frozen=True)
class UseEvidence:
    """What the link paths show for one open act."""

    first_use_at: Optional[datetime]
    use_site: Optional[str]
    observable: bool


def output_used_grade(evidence: UseEvidence, *, horizon_at: datetime, now: datetime,
                      grace_sec: float) -> Optional[ActOutcomeV1]:
    """The close for an open ``output_used`` act, or None while it should stay open."""
    if evidence.first_use_at is not None and evidence.first_use_at <= horizon_at:
        site = f" ({evidence.use_site})" if evidence.use_site else ""
        return ActOutcomeV1(value=1.0, observed_at=evidence.first_use_at, source="resolver",
                            note=f"{USED_NOTE}{site}")
    if (now - horizon_at).total_seconds() < grace_sec:
        return None
    if evidence.observable:
        return ActOutcomeV1(value=0.0, observed_at=horizon_at, source="expired", note=UNUSED_NOTE)
    return ActOutcomeV1(value=0.0, observed_at=horizon_at, source="expired", note=UNOBSERVABLE_NOTE)


def producer_close_disposition(value: float) -> str:
    """What a producer close does to an ``output_used`` act: ``defer`` (the output exists; wait for a
    use) or ``no_output`` (graded 0 now)."""
    return "defer" if value >= 1.0 else "no_output"


ANY_OPEN_OUTPUT_USED_SQL = text(
    "SELECT 1 FROM grammar_acts WHERE closed_at IS NULL AND signal_id = 'output_used' LIMIT 1"
)

# One row per open output_used act that is either used already or past its horizon + grace. A used act
# is found before its horizon, so a hit closes within one sweep. Its own limit and query, so up to a
# day of open output_used acts never crowds the main due list.
DUE_OUTPUT_USED_SQL = text(
    """
    SELECT a.act_id, a.level, a.actor, a.verb, a.signal_id, a.direction, a.horizon_sec, a.expect_basis,
           a.opened_at, a.horizon_at, a.correlation_id,
           u.first_use_at, u.use_site,
           (EXISTS (SELECT 1 FROM temporal_self_event e
                     WHERE a.correlation_id IS NOT NULL AND e.correlation_id = a.correlation_id
                       AND e.source_kind <> 'action_outcome')
            OR EXISTS (SELECT 1 FROM act_downstream_use d
                        WHERE d.act_id = a.act_id AND d.link_status = 'linked')) AS observable
      FROM grammar_acts a
      LEFT JOIN LATERAL (
            SELECT x.t AS first_use_at, x.site AS use_site FROM (
                SELECT eu.cited_at AS t, eu.citing_organ AS site
                  FROM temporal_self_event e
                  JOIN experience_use eu ON eu.event_id = e.event_id
                 WHERE a.correlation_id IS NOT NULL AND e.correlation_id = a.correlation_id
                   AND e.source_kind <> 'action_outcome' AND eu.cited_at >= a.opened_at
                UNION ALL
                SELECT d.first_use_at AS t, d.site AS site
                  FROM act_downstream_use d
                 WHERE d.act_id = a.act_id AND d.used AND d.first_use_at IS NOT NULL
                   AND d.first_use_at >= a.opened_at
            ) x ORDER BY x.t LIMIT 1
      ) u ON TRUE
     WHERE a.closed_at IS NULL AND a.opened_at IS NOT NULL AND a.signal_id = 'output_used'
       AND (u.first_use_at IS NOT NULL OR a.horizon_at < :expired_before)
     ORDER BY a.opened_at
     LIMIT :limit
    """
)
