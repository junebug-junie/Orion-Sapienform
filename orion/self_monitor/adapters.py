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
* ``vision_percept_expectation`` (``services/orion-sql-writer/app/vision_rhythm.py``): ``status``
  ``met``/``missed``/``open``/``unscorable``; ``confidence`` stored at emit time.
"""

from __future__ import annotations

import re
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
            return None  # ungradeable_no_stated / bad record: no bet was made
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
    did the turn run? ``failed`` = every guess on it was voided ``failed_to_execute``."""
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
