"""One pure adapter per bet source: a row dict in, a ``SettledBetV1`` (or None) out.

Nothing here reads a database; ``report.py`` and the host do. Each adapter states which column is
the bet, which is the stated confidence (recorded before the outcome, never reconstructed), what
counts as the outcome, and who settled it. A row that is not settled yet returns None.

Vocabulary found live (2026-11-03, 7 days), kept here so a reader can check the mapping:

* ``grammar_acts`` (``orion/grammar/act_ledger.py``): ``predicted`` is the act cell's posterior mean
  stamped at open (or, for a close that arrives before its open, the prior read just before the
  close's own update -- still pre-outcome). The cell is keyed (verb, actor, signal), so every trigger
  name of one actor shares it: ``belief_key`` carries that key. ``outcome_source`` is ``producer`` (the doer reported), ``resolver`` (Juniper's
  turn did or didn't start) or ``expired`` (no close within horizon + 120 s). A close that never
  ran is written as ``outcome_value = 0`` with a note: ``failed: ...`` (RPC timeout, empty LLM
  reply), ``raised: ...`` (journal compose exception), ``abandoned:...``, or
  ``no_read_evidence:<why>`` (http_error, thin_fetch), ``control|treated run <id> died`` (Prime asks)
  or ``no_planner_decisions`` (retro play). Those are ``failed_to_execute``: the act did not happen,
  so the claim was never tested. A bare ``no_read_evidence`` (the read ran and found nothing),
  ``session_cap; 0 line(s) spoken`` (the session ran) and ``dropped unresolved: ...`` (a deliberate
  drop) are real ``refuted``. ``expired`` (no close within horizon + 120 s) is ``refuted`` too: the
  ledger defines an expired act as "a real 0, not a guess", and it is settled by the timer.
* ``prediction_turn_self_bet`` (``services/orion-prediction/app/turn_self_bets.py``): ``status`` is
  ``happened``/``not_happened`` (p faculties), ``upheld``/``broken`` (range faculties) or ``void``
  with ``void_reason``.
* ``prediction_action_bet`` (``services/orion-prediction/app/action_bets.py``): ``result`` is
  ``success``/``failure``/``void``; ``p`` is the served probability at placing.
* ``prediction_forecast``: champion rows; ``in_80`` is the outcome of the stated 80% range.
* ``prediction_bill_forecast`` (``services/orion-prediction/app/bill_forecast.py``): one row per day,
  Orion's projection of the current power bill with an 80 % range (``coverage``) on kWh and on pre-tax
  dollars, settled when RMP's bill lands (``kwh_in_80`` / ``usd_in_80``; ``usd_in_80`` is NULL when the
  bill has no tax line). Only ``origin = 'live'`` rows are bets: ``replay`` rows were backfilled from
  history the method was chosen on. Every daily guess at one bill shares one answer, so each carries
  ``answer_key`` = the bill (``record.py`` counts it once).
* ``vision_percept_expectation`` (``services/orion-sql-writer/app/vision_rhythm.py``): ``status``
  ``met``/``missed``/``open``/``unscorable``; ``confidence`` stored at emit time.
* ``orion_ask`` camera guesses (``services/orion-sql-writer/app/vision_individuals.py``,
  ``_open_person_confirm_asks`` and ``_open_face_asks``): only two of the camera ask kinds carry a
  guess of Orion's. ``vision_person_confirm`` asks "Is this <name>?" (``source_ref`` =
  ``<individual>|<name>``; ``rule:camera_prior`` in ``evidence_refs`` when the office camera's prior
  chose the name). ``vision_same_individual`` asks "Are these the same person?" for a pair whose face
  cosine is at least ``same_ask_min_face_cosine``; the street handoff loop
  (``vision_handoff_loop.py``, ``rule:handoff``) opens the same kind for two STORIES' crops, a
  different guess, so it is its own faculty ``vision.same_person.handoff``. Juniper settles both: ``answer`` yes/no, and on a
  confirm card optionally ``answer_detail.pictures[].verdict`` (same/different/unsure). Neither ask
  records a confidence: the attribution ``score`` is a median top-k cosine that the person model
  rewrites every cycle, and ``face_cosine`` is a similarity, not a probability. So ``stated`` is None
  and these faculties show their hit rate but stay ``ungradeable:no_stated_confidence``.
"""

from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from orion.schemas.self_monitor import SettledBetV1

Row = Mapping[str, Any]

# A name is a plain word (hyphens allowed, no digits): ids, hashes and versions carry digits.
_NAME = re.compile(r"^[a-z][a-z_-]*$")
_EXEC_FAIL_PREFIXES = ("failed:", "raised:", "abandoned", "no_read_evidence:")
_EXEC_FAIL_EXACT = frozenset({"no_planner_decisions"})


def _aware(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def act_faculty(actor: str, verb: str, object_ref: str) -> str:
    """``act.<actor>.<verb>[.<name>]``.

    The actor is the doer and the learner's own key (``thought.painting``, ``thought.dream_hop`` and
    ``durable.journal_compose`` all make a ``durable_run``; merged, one doer's failures would hide
    behind the others' successes). The name is the segment after ``kind:``, kept only when it is a
    plain word: ``journal_trigger:walkway_grade`` names a different faculty from
    ``journal_trigger:town_episode``, while an id, hash or version (``retro_session:<uuid>``,
    ``prime_thought:<hex>``, ``topic_foundry_model:<name>-<hash>:v1``) or a URL is one instance.
    """
    base = f"act.{actor or 'unknown'}.{verb or 'unknown'}"
    ref = (object_ref or "").strip()
    if ref.startswith(("http://", "https://")):
        return base
    parts = ref.split(":")
    if len(parts) >= 2 and _NAME.match(parts[1]):
        return f"{base}.{parts[1]}"
    return base


def act_outcome(row: Row) -> Optional[str]:
    """One grammar_acts close -> upheld | refuted | failed_to_execute | no_change. Also read by the body
    economy's weights (source (d), ``orion/physiology/weights.py``), so both organs grade an act alike."""
    source = (row.get("outcome_source") or "").strip()
    note = (row.get("outcome_note") or "").strip()
    value = _float(row.get("outcome_value"))
    if note.startswith(_EXEC_FAIL_PREFIXES) or note in _EXEC_FAIL_EXACT or note.endswith(" died"):
        return "failed_to_execute"
    if source == "expired":
        return "refuted"
    if value is None:
        return "no_change"
    return "upheld" if value >= 0.5 else "refuted"


def grammar_act(row: Row) -> Optional[SettledBetV1]:
    closed = _aware(row.get("closed_at"))
    opened = _aware(row.get("opened_at"))
    if closed is None or opened is None:
        return None
    outcome = act_outcome(row)
    source = (row.get("outcome_source") or "").strip()
    # An expired act is a real 0 (the deadline passed with no artifact): the world answered.
    settled_by = "juniper" if source == "resolver" else "world"
    actor = str(row.get("actor") or "unknown")
    verb = str(row.get("verb") or "unknown")
    return SettledBetV1(
        bet_id=f"grammar_acts:{row['act_id']}",
        faculty=act_faculty(actor, verb, str(row.get("object_ref") or "")),
        belief_key=f"grammar_acts:{verb}|{actor}|{row.get('signal_id') or ''}",
        source="grammar_acts",
        made_at=opened,
        settled_at=closed,
        claim_kind="success",
        stated=_float(row.get("predicted")),
        outcome=outcome,
        settled_by=settled_by,
        evidence_ref=f"grammar_acts:{row['act_id']}",
    )


_TURN_STATUS = {"happened": "upheld", "upheld": "upheld", "not_happened": "refuted", "broken": "refuted"}
_TURN_VOID = {"failed_to_execute": "failed_to_execute", "expired_unsettled": "expired_unsettled"}


def turn_self_bet(row: Row) -> Optional[SettledBetV1]:
    settled = _aware(row.get("settled_at"))
    if settled is None:
        return None
    status = row.get("status")
    if status == "void":
        # A turn that errored voids all four guesses at once: the turn failed, not the guess. It is
        # counted once, under ``turn.run`` (``turn_run``), never as a run of failures on each guess
        # faculty -- live 2026-11-03 that read turn.self.steps/reply_length as "broken" after stance
        # stopped stating them (relative bets, #3042) and only the failed turns were left.
        if (row.get("void_reason") or "") == "failed_to_execute":
            return None
        outcome = _TURN_VOID.get(row.get("void_reason") or "")
        if outcome is None:
            # ungradeable_no_stated / bad record: no bet was made. interrupted: the turn was
            # stopped from outside, so the guess never met its turn.
            return None
    else:
        outcome = _TURN_STATUS.get(status or "")
        if outcome is None:
            return None
    interval = row.get("claim_kind") == "interval"
    return SettledBetV1(
        bet_id=f"prediction_turn_self_bet:{row['bet_id']}",
        faculty=str(row["faculty"]),
        source="prediction_turn_self_bet",
        made_at=_aware(row["placed_at"]),
        settled_at=settled,
        claim_kind="interval" if interval else "event",
        stated=_float(row.get("coverage") if interval else row.get("stated_p")),
        outcome=outcome,
        settled_by="world",
        evidence_ref=f"prediction_turn_self_bet:{row['bet_id']}",
    )


def turn_run(row: Row) -> Optional[SettledBetV1]:
    """One row per unified turn (``collect`` groups ``prediction_turn_self_bet`` by correlation_id):
    did the turn run? ``failed`` = every guess on it was voided ``failed_to_execute``. A turn stopped
    from outside (void ``interrupted``) never reaches here: ``collect`` drops its rows."""
    settled = _aware(row.get("settled_at"))
    corr = row.get("correlation_id")
    if settled is None or not corr:
        return None
    return SettledBetV1(
        bet_id=f"prediction_turn_self_bet:turn:{corr}",
        faculty="turn.run",
        source="prediction_turn_run",  # the SOURCES name: a read error holds this faculty's status
        made_at=_aware(row["placed_at"]),
        settled_at=settled,
        claim_kind="success",
        stated=None,
        outcome="failed_to_execute" if row.get("failed") else "upheld",
        settled_by="world",
        evidence_ref=f"prediction_turn_self_bet:correlation_id={corr}",
    )


def action_bet(row: Row) -> Optional[SettledBetV1]:
    settled = _aware(row.get("settled_at"))
    result = row.get("result")
    if settled is None or result not in ("success", "failure"):
        return None
    return SettledBetV1(
        bet_id=f"prediction_action_bet:{row['bet_id']}",
        faculty=f"body.{row.get('domain') or 'unknown'}.{row.get('skill') or 'unknown'}",
        source="prediction_action_bet",
        made_at=_aware(row["placed_at"]),
        settled_at=settled,
        claim_kind="success",
        stated=_float(row.get("p")),
        outcome="upheld" if result == "success" else "refuted",
        settled_by="world",
        evidence_ref=f"prediction_action_bet:{row['bet_id']}",
    )


def forecast(row: Row) -> Optional[SettledBetV1]:
    scored = _aware(row.get("scored_at"))
    in_80 = row.get("in_80")
    if scored is None or in_80 is None or row.get("role") != "champion":
        return None
    return SettledBetV1(
        bet_id=f"prediction_forecast:{row['forecast_id']}",
        faculty=f"forecast.{row['series_id']}",
        source="prediction_forecast",
        made_at=_aware(row["issued_at"]),
        settled_at=scored,
        claim_kind="interval",
        stated=0.8,
        outcome="upheld" if in_80 else "refuted",
        settled_by="world",
        evidence_ref=f"prediction_forecast:{row['forecast_id']}",
    )


def _bill(row: Row, quantity: str) -> Optional[SettledBetV1]:
    settled = _aware(row.get("settled_at"))
    inside = row.get(f"{quantity}_in_80")
    if settled is None or inside is None or row.get("origin") != "live" or row.get("coverage") is None:
        return None
    return SettledBetV1(
        bet_id=f"prediction_bill_forecast:{row['bet_id']}:{quantity}",
        faculty=f"forecast.energy.bill_{quantity}",
        source=f"prediction_bill_forecast_{quantity}",
        made_at=_aware(row["as_of"]),
        settled_at=settled,
        claim_kind="interval",
        stated=_float(row.get("coverage")),
        answer_key=f"energy_bill:{row.get('usage_point_id') or ''}:{row.get('bill_period_start')}",
        outcome="upheld" if inside else "refuted",
        settled_by="world",
        evidence_ref=f"prediction_bill_forecast:{row['bet_id']}",
    )


def bill_kwh(row: Row) -> Optional[SettledBetV1]:
    """One live daily bill bet's kWh range against the kWh RMP billed."""
    return _bill(row, "kwh")


def bill_usd(row: Row) -> Optional[SettledBetV1]:
    """The same bet's pre-tax dollar range against the bill's pre-tax total."""
    return _bill(row, "usd")


def percept_expectation(row: Row) -> Optional[SettledBetV1]:
    scored = _aware(row.get("scored_at"))
    status = row.get("status")
    if scored is None or status not in ("met", "missed"):
        return None
    return SettledBetV1(
        bet_id=f"vision_percept_expectation:{row['expectation_id']}",
        faculty="vision.percept_expectation",
        source="vision_percept_expectation",
        made_at=_aware(row["emitted_at"]),
        settled_at=scored,
        claim_kind="event",
        stated=_float(row.get("confidence")),
        outcome="upheld" if status == "met" else "refuted",
        settled_by="world",
        evidence_ref=f"vision_percept_expectation:{row['expectation_id']}",
    )


# Camera asks that state a guess of Orion's (see the module docstring). "Who is this?" asks, walkway
# story names and person facts carry no guess, so they are not bets.
CAMERA_ASK_FACULTIES = {"vision_person_confirm": "vision.person_confirm", "vision_same_individual": "vision.same_person"}
CAMERA_PRIOR_REF = "rule:camera_prior"
HANDOFF_RULE_REF = "rule:handoff"
# Juniper's yes/no. Twin of ``_YES``/``_NO`` in services/orion-sql-writer/app/vision_individuals.py
# (the applier's parse_yes_no; an app module, so it cannot be imported here): change both together.
_ASK_YES = frozenset({"yes", "y", "same", "same person", "yes same person", "yes, same person"})
_ASK_NO = frozenset({"no", "n", "different", "different people", "not the same", "no different people",
                     "no, different people"})


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def camera_ask_outcome(row: Row) -> Optional[str]:
    """Juniper's answer to a camera guess -> upheld | refuted | no_change, or None (no usable answer).

    The applier's own order (``_apply_person_confirm``): an answer that is not yes/no applies
    nothing, so it grades nothing (a card closed by a name typed elsewhere stores ``answer`` NULL).
    Then per-picture answers win (only pictures with an ``image_ref``, as ``confirm_picture_verdicts``
    reads them): all "same" -> upheld, all "different" -> refuted, any other mix (the cluster holds
    more than one person, or she was unsure of some) -> no_change: the guess was about the cluster,
    and she said neither."""
    answer = " ".join(str(row.get("answer") or "").lower().replace(".", " ").replace("!", " ").split())
    if answer in _ASK_YES:
        said = "upheld"
    elif answer in _ASK_NO:
        said = "refuted"
    else:
        return None
    detail = _json(row.get("answer_detail")) or {}
    pictures = detail.get("pictures") if isinstance(detail, dict) else None
    verdicts = {p.get("verdict") for p in (pictures or []) if isinstance(p, dict) and p.get("image_ref")}
    verdicts &= {"same", "different", "unsure"}
    if not verdicts:
        return said
    if verdicts == {"same"}:
        return "upheld"
    if verdicts == {"different"}:
        return "refuted"
    return "no_change"


def camera_ask(row: Row) -> Optional[SettledBetV1]:
    """One answered camera guess. ``stated`` is None: no confidence is recorded before the ask."""
    faculty = CAMERA_ASK_FACULTIES.get(row.get("source_kind") or "")
    answered = _aware(row.get("answered_at"))
    if faculty is None or answered is None or row.get("status") != "answered":
        return None
    outcome = camera_ask_outcome(row)
    if outcome is None:
        return None
    refs = _json(row.get("evidence_refs")) or []
    if faculty == "vision.person_confirm" and CAMERA_PRIOR_REF in refs:
        # The office camera's prior chose the name, not the album scores: another mechanism.
        faculty = "vision.person_confirm.camera_prior"
    elif faculty == "vision.same_person" and HANDOFF_RULE_REF in refs:
        # Two stories' crops matched across cameras, not two face centroids: another guess.
        faculty = "vision.same_person.handoff"
    return SettledBetV1(
        bet_id=f"orion_ask:{row['ask_id']}",
        faculty=faculty,
        source="orion_ask",
        made_at=_aware(row["created_at"]),
        settled_at=answered,
        claim_kind="event",
        stated=None,
        outcome=outcome,
        settled_by="juniper",
        evidence_ref=f"orion_ask:{row['ask_id']}",
    )
