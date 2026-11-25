"""The ``output_used`` act signal: another organ used what this act made (act consequence bets, Patch 1a).

Spec: docs/superpowers/specs/2026-11-25-act-consequence-bets-design.md (PR #3953).

An act that bets ``output_used`` is not closed by its producer's "I made it" report. A producer close
with value 1 only records that the output exists; the act stays open. The resolver closes it:

* **used** (1, ``resolver``) at the first use inside the horizon, from either link path:
  (a) an ``experience_use`` row by a DIFFERENT organ, for a ``temporal_self_event`` that carries the act's
  ``correlation_id`` AND whose ``source_kind`` belongs to the act's own organ. ``config/integration/
  organs.yaml`` says both: an organ's ``families`` are act actors, its ``temporal_sources`` are its
  experience kinds. One turn's correlation_id also carries other organs' rows (attention_row, gpu_wait,
  chat_turn), so without the kind match a cite of the turn's attention row would credit every act on the
  turn. (b) the act's ``act_downstream_use`` row with ``used``.
* **unused** (0, ``expired``) at the horizon, when the output was observable: an experience of the act's
  own organ landed with its correlation_id, or its downstream-use row is ``linked``.
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
from typing import Any, Mapping, Optional

from sqlalchemy import text

from orion.grammar.promise_acts import VOID_NOTE_PREFIX
from orion.schemas.grammar import ActOutcomeV1

OUTPUT_USED_SIGNAL = "output_used"

USED_NOTE = "used: another organ used this act's output"
UNUSED_NOTE = "unused: output was observable, no other organ used it before horizon"
UNOBSERVABLE_NOTE = f"{VOID_NOTE_PREFIX}unobservable no link path to see a use (censored, not graded)"
NO_OUTPUT_NOTE = "no_output: producer reported nothing made"


@dataclass(frozen=True)
class ActOutputKinds:
    """Arrays the due query joins on: (actor, organ) and (actor, experience kind) pairs from the registry."""

    actors: tuple[str, ...]
    organs: tuple[str, ...]
    kind_actors: tuple[str, ...]
    kinds: tuple[str, ...]

    def params(self) -> dict[str, list[str]]:
        return {"actors": list(self.actors), "organs": list(self.organs),
                "kind_actors": list(self.kind_actors), "kinds": list(self.kinds)}


def act_output_kinds(organs: Mapping[str, Any], organ_of: Mapping[str, str]) -> ActOutputKinds:
    """From the organ registry (``OrganRegistry.raw``, organ name -> its entry, and ``organ_of``): each act
    actor (an organ's ``families`` entry) -> its organ, and -> each experience kind of that organ. An actor
    no organ lists has no experience link path (downstream only)."""
    actors: list[str] = []
    actor_organs: list[str] = []
    kind_actors: list[str] = []
    kinds: list[str] = []
    for organ, spec in sorted((organs or {}).items()):
        # An act's own grade row (action_outcome, the prediction organ's kind) is never its output.
        organ_kinds = sorted(k for k, o in organ_of.items() if o == organ and k != "action_outcome")
        for actor in (spec or {}).get("families") or []:
            actors.append(str(actor))
            actor_organs.append(str(organ))
            for kind in organ_kinds:
                kind_actors.append(str(actor))
                kinds.append(kind)
    return ActOutputKinds(tuple(actors), tuple(actor_organs), tuple(kind_actors), tuple(kinds))


def producer_close_note(note: str) -> str:
    """The stored note of a producer's nothing-made close. The producer's own words stay as they are, so
    an execution failure ("failed: ...", "raised: ...") still reads as failed_to_execute downstream."""
    return note.strip() or NO_OUTPUT_NOTE


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
# day of open output_used acts never crowds the main due list. ``due_at`` falls back to created_at +
# horizon_sec for an act whose open never arrived (close-before-open), so it can never stay open forever.
DUE_OUTPUT_USED_SQL = text(
    """
    WITH org AS (
        SELECT * FROM unnest(CAST(:actors AS text[]), CAST(:organs AS text[])) AS t(actor, organ)
    ), kinds AS (
        SELECT * FROM unnest(CAST(:kind_actors AS text[]), CAST(:kinds AS text[])) AS t(actor, kind)
    ), due AS (
        SELECT a.*, COALESCE(a.opened_at, a.created_at) AS start_at,
               COALESCE(a.horizon_at, a.created_at + make_interval(secs => a.horizon_sec)) AS due_at
          FROM grammar_acts a
         WHERE a.closed_at IS NULL AND a.signal_id = 'output_used'
    )
    SELECT a.act_id, a.level, a.actor, a.verb, a.signal_id, a.direction, a.horizon_sec, a.expect_basis,
           a.opened_at, a.due_at, a.correlation_id,
           u.first_use_at, u.use_site,
           (EXISTS (SELECT 1 FROM temporal_self_event e JOIN kinds k ON k.kind = e.source_kind
                     WHERE a.correlation_id IS NOT NULL AND e.correlation_id = a.correlation_id
                       AND k.actor = a.actor)
            OR EXISTS (SELECT 1 FROM act_downstream_use d
                        WHERE d.act_id = a.act_id AND d.link_status = 'linked')) AS observable
      FROM due a
      LEFT JOIN LATERAL (
            SELECT x.t AS first_use_at, x.site AS use_site FROM (
                SELECT eu.cited_at AS t, eu.citing_organ AS site
                  FROM temporal_self_event e
                  JOIN kinds k ON k.kind = e.source_kind AND k.actor = a.actor
                  JOIN experience_use eu ON eu.event_id = e.event_id
                 WHERE a.correlation_id IS NOT NULL AND e.correlation_id = a.correlation_id
                   AND eu.cited_at >= a.start_at
                   AND eu.citing_organ IS DISTINCT FROM (SELECT o.organ FROM org o WHERE o.actor = a.actor LIMIT 1)
                UNION ALL
                SELECT d.first_use_at AS t, d.site AS site
                  FROM act_downstream_use d
                 WHERE d.act_id = a.act_id AND d.used AND d.first_use_at IS NOT NULL
                   AND d.first_use_at >= a.start_at
            ) x ORDER BY x.t LIMIT 1
      ) u ON TRUE
     WHERE u.first_use_at IS NOT NULL OR a.due_at < :expired_before
     ORDER BY a.start_at
     LIMIT :limit
    """
)
