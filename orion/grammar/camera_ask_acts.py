"""Camera asks as graded acts (attention patch 4a, docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md).

A camera ask is a card in ``orion_ask`` that asks Juniper about something the camera saw ("Is this
Theo?", "Are these the same person?", "Whose vehicle is this?"). From patch 4a each ask ORION made is
one act on the act ledger: actor ``vision.ask``, verb ``ask``, ``object_ref orion_ask:<ask_id>``,
``correlation_id`` the ask id, horizon the ask's ``expires_at``. Its grade is Juniper's answer
(``juniper_answered``), so the ``vision.ask`` family learns which asks she can answer.

The grade (``ask_grade``), read from the ask row by the act ledger's horizon resolver:

* answered (a name, yes, no, same, different) -> 1, ``resolver`` (Juniper graded it, not Orion);
* dismissed, or "I don't know" (a dismissal with ``answer_detail.dont_know``) -> 0, ``resolver``;
* expired unanswered (still open at ``expires_at``, or expired by the clock) -> 0, ``resolver``:
  seven days of silence is an answer, as for outreach (Juniper confirmed, PR #3213 question 3);
* withdrawn by Orion (``status = 'withdrawn'``; ``status = 'expired'`` with ``answered_at`` before
  ``expires_at``, which a split undo stamps; or a card closed as moot, ``answer_detail.resolved_by``)
  -> ``expired`` with an ``abandoned:`` note: not a grade for the market (``GRADED_SOURCES``), the
  self-monitor (``failed_to_execute``) or the effect posterior (skipped in ``close_act``);
* still open before its expiry -> no close yet.

Juniper's OWN cards are not Orion's asks and never open an act or count in any ask statistic: a
name she types on a walkway story is stored as an already-answered ``vision_individual`` row
("Who is this? (named on a walkway story)", services/orion-hub/scripts/motion_story_routes.py,
about 7 a day). From patch 4a Hub marks them with the evidence ref ``JUNIPER_STORY_CARD_REF``; rows
written before carry ``answer_detail.story_id`` (all 181 live rows, checked 2026-11-13), so the
filter is structural, never a match on the question text.

The rule (``ask_rule``) is which opener asked, the unit the expected outcome is learned per:
``confirm`` (Is this <name>?), ``same`` (same person?, incl. door/walkway handoffs), ``face``
(who is this? from a clear face), ``office`` (who is in the office?), ``vehicle`` and ``sighting``
(the slow "seen N times" rule, split by what was seen: Juniper dismisses 40 of 45 vehicle asks).

Pure and stdlib-only (plus the grammar schema), so the act ledger, sql-writer, Hub and the replay
eval share one copy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

ACTOR = "vision.ask"
VERB = "ask"
SIGNAL = "juniper_answered"

# The three orion_ask kinds Orion's camera openers write (app/vision_individuals.py, vision_handoff_loop.py,
# vision_office_sessions.py). vision_story_person is Juniper's own story label, never an ask.
CAMERA_ASK_KINDS = ("vision_individual", "vision_person_confirm", "vision_same_individual")

# Evidence ref Hub stamps on a name Juniper typed on a walkway story card (patch 4a).
JUNIPER_STORY_CARD_REF = "rule:juniper_story_naming"

FACE_RULE_REF = "rule:face"
OFFICE_RULE_REF = "rule:office_session"
RULES = ("confirm", "same", "face", "office", "vehicle", "sighting")

# Used when an ask row has no expires_at (none live): VISION_ASK_EXPIRY_DAYS' default.
DEFAULT_EXPIRY = timedelta(days=7)
UNKNOWN_STATUS_GRACE = timedelta(seconds=120)
# A withdrawn ask is not a grade anywhere: the "abandoned" prefix makes orion.self_monitor.adapters.act_outcome
# read it as failed_to_execute (not refuted), and the act ledger skips the posterior for it.
WITHDRAWN_PREFIX = "abandoned:"
WITHDRAWN_NOTE = "abandoned: withdrawn by orion"
OUTCOME_WINDOW = timedelta(days=30)

# SQL predicate (alias ``a`` on orion_ask): an ask ORION made with a camera opener. Juniper's own
# story cards are excluded by their marker, or (rows before the marker) by answer_detail.story_id.
ORION_MADE_CAMERA_ASK_SQL = (
    "a.source_kind IN ('vision_individual', 'vision_person_confirm', 'vision_same_individual') "
    "AND NOT coalesce(a.evidence_refs @> '[\"" + JUNIPER_STORY_CARD_REF + "\"]'::jsonb, false) "
    "AND NOT coalesce(a.answer_detail ? 'story_id', false)"
)

BASIS = (
    "juniper_answered by the ask's expires_at (VISION_ASK_EXPIRY_DAYS, 7 d): answered (a name, yes/no, "
    "same/different) = 1; dismissed, don't know, or unanswered at expiry = 0; withdrawn by Orion = expired "
    "(not a grade). Time to an answer over 30 d: median 3.7 h, p90 20.6 h, max 43 h (n=197, measured 2026-11-13)"
)


def _as_list(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except ValueError:
            return []
    return []


def _as_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except ValueError:
            return {}
    return {}


def is_juniper_card(evidence_refs: Any, answer_detail: Any) -> bool:
    """A card Juniper made herself (a name typed on a walkway story). Same rule as the SQL predicate."""
    return JUNIPER_STORY_CARD_REF in _as_list(evidence_refs) or "story_id" in _as_dict(answer_detail)


def is_orion_camera_ask(source_kind: Any, evidence_refs: Any, answer_detail: Any) -> bool:
    return str(source_kind or "") in CAMERA_ASK_KINDS and not is_juniper_card(evidence_refs, answer_detail)


def ask_rule(source_kind: str, evidence_refs: Any, individual_kind: Optional[str]) -> str:
    """Which opener asked. ``individual_kind``: vision_individual.kind of the ask's first individual."""
    if source_kind == "vision_person_confirm":
        return "confirm"
    if source_kind == "vision_same_individual":
        return "same"
    refs = _as_list(evidence_refs)
    if FACE_RULE_REF in refs:
        return "face"
    if OFFICE_RULE_REF in refs:
        return "office"
    return "vehicle" if individual_kind == "vehicle" else "sighting"


def _utc(v: Optional[datetime]) -> Optional[datetime]:
    if v is None:
        return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class AskGrade:
    value: float          # 0 or 1
    source: str           # resolver (Juniper's answer or her silence) | expired (withdrawn: not a grade)
    observed_at: datetime
    note: str


def ask_grade(ask: Optional[Mapping[str, Any]], *, now: datetime,
              horizon_at: Optional[datetime] = None) -> Optional[AskGrade]:
    """The close of one ask's act, from its ``orion_ask`` row (status, answer_detail, answered_at,
    expires_at). None: still open and not yet expired. ``ask`` None: the row is gone (withdrawn)."""
    now = _utc(now)
    if ask is None:
        return AskGrade(0.0, "expired", now, "ask row gone")
    status = str(ask.get("status") or "")
    answered_at = _utc(ask.get("answered_at"))
    expires_at = _utc(ask.get("expires_at")) or _utc(horizon_at)
    if status == "answered":
        if "resolved_by" in _as_dict(ask.get("answer_detail")):
            # Closed by Orion because a name Juniper typed elsewhere made this card moot: not her answer to it.
            return AskGrade(0.0, "expired", answered_at or now, WITHDRAWN_NOTE + " (moot: resolved elsewhere)")
        return AskGrade(1.0, "resolver", answered_at or now, "juniper answered")
    if status == "dismissed":
        dont_know = bool(_as_dict(ask.get("answer_detail")).get("dont_know"))
        return AskGrade(0.0, "resolver", answered_at or now,
                        "juniper said she does not know" if dont_know else "juniper dismissed")
    if status == "withdrawn":
        return AskGrade(0.0, "expired", answered_at or now, WITHDRAWN_NOTE)
    if status == "expired":
        # A withdrawal (split undo) stamps answered_at before expires_at; the clock sweep leaves it NULL.
        if answered_at is not None and (expires_at is None or answered_at < expires_at):
            return AskGrade(0.0, "expired", answered_at, WITHDRAWN_NOTE + " before expiry")
        if expires_at is not None and now < expires_at:
            return AskGrade(0.0, "expired", now, WITHDRAWN_NOTE + " before expiry")
        return AskGrade(0.0, "resolver", expires_at or now, "unanswered at expiry")
    if status == "open":
        if expires_at is not None and now >= expires_at:
            return AskGrade(0.0, "resolver", expires_at, "unanswered at expiry")
        return None
    # An unknown status is never guessed into a grade; past the horizon it is closed as not graded, so it cannot
    # sit in the resolver's due set forever.
    if expires_at is not None and now >= expires_at + UNKNOWN_STATUS_GRACE:
        return AskGrade(0.0, "expired", now, f"{WITHDRAWN_PREFIX} unknown_status:{status[:40]}")
    return None


# Pseudo-closes of the family's answered share each rule's value starts from (issue #3350). See rule_outcome_values.
PRIOR_WEIGHT = 2.0
# With no graded close in the whole family there is no family share: the uninformed 0.5.
EMPTY_FAMILY_PRIOR = 0.5


def rule_outcome_values(closes: Iterable[tuple[str, float]], *, k: float = PRIOR_WEIGHT) -> dict[str, float]:
    """Per rule, the answered share of its graded closes, shrunk toward the FAMILY's answered share:
    ``(answered + k * m) / (closed + k)``, ``m`` = answered / closed over every rule's closes in ``closes`` (the
    vision.ask family, Orion-made asks only, the same 30 days). A rule with no closes reads ``m``.

    Why not Laplace ``(answered + 1) / (closed + 2)`` (patch 4a, until issue #3350): it starts every rule at 0.5,
    below the family's ~0.82. Under 4b the market funds a camera ask only when its expected answer beats the
    family's (strict-below rank, hunger saturated), so a rare rule (office, 1 of 1 answered: 0.667) was held, never
    opened, never graded, and never moved. Starting from the family's share, one answer puts a rule above it and one
    "don't know" below it. ``k`` = 2: the prior weighs as much as two of the rule's own closes, as Laplace did (two
    pseudo-closes), so a well-sampled rule barely moves. Live, 2026-11-13 (262 closes, m = 0.817): vehicle 5/47
    0.122 -> 0.135, confirm 147/152 0.961 -> 0.965, office 1/1 0.667 -> 0.878, face 5/5 0.857 -> 0.948, sighting
    4/5 0.714 -> 0.805. In the live 30-day replay k = 1, 2, 4 and 8 open exactly the same asks
    (docs/superpowers/pr-reports/2026-11-13-camera-ask-prior-family-share-pr.md).

    ``m`` is the market's own family mean only if ``closes`` are the acts the market reads: services/orion-sql-writer
    ``GRADES_SQL`` selects them the same way (opened in 30 days, producer or resolver outcome), tested against
    ``build_family_appetite`` in tests/test_ask_bind_integration.py. A rule reading exactly ``m`` is rank-neutral: it
    beats every past event whose hunger was below saturation and ties (strict-below: no credit) the saturated ones,
    so it is funded unless EVERY past event was saturated (tests/test_camera_ask_acts.py pins both).
    ``closes``: (rule, outcome_value) of graded closes; unknown rules and non-finite values are ignored (the market
    skips them too). Every rule has a value."""
    answered = {r: 0 for r in RULES}
    closed = {r: 0 for r in RULES}
    for rule, value in closes:
        if rule not in closed or not math.isfinite(float(value)):
            continue
        closed[rule] += 1
        answered[rule] += 1 if float(value) >= 1.0 else 0
    n = sum(closed.values())
    m = sum(answered.values()) / n if n else EMPTY_FAMILY_PRIOR
    return {r: (answered[r] + k * m) / (closed[r] + k) for r in RULES}


def rule_from_test_before(test_before: Optional[str]) -> Optional[str]:
    """The rule stamped on an act at open (``test_before = camera_ask:<rule>``)."""
    if test_before and test_before.startswith("camera_ask:"):
        r = test_before.split(":", 1)[1]
        return r if r in RULES else None
    return None


def ask_act(ask_id: str, *, rule: str, created_at: datetime, expires_at: Optional[datetime]):
    """The ``ask`` act of one Orion-made camera ask; its horizon ends at the ask's expiry."""
    from orion.grammar.acts import build_act

    created = _utc(created_at)
    expires = _utc(expires_at) or created + DEFAULT_EXPIRY
    horizon = max(60.0, (expires - created).total_seconds())
    return build_act(actor=ACTOR, verb=VERB, key=str(ask_id), signal_id=SIGNAL, horizon_sec=horizon, basis=BASIS,
                     object_ref=f"orion_ask:{ask_id}", test_before=f"camera_ask:{rule}")


def ask_opened_event(ask_id: str, *, rule: str, created_at: datetime, expires_at: Optional[datetime]):
    """The act_opened event the writer applies straight to the act ledger (never published on the bus:
    sql-writer is the ledger, and the ask row is already in its database)."""
    from uuid import uuid4

    from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1

    act = ask_act(ask_id, rule=rule, created_at=created_at, expires_at=expires_at)
    at = _utc(created_at)
    return GrammarEventV1(
        event_id=f"act_opened:{act.act_id}:{uuid4().hex[:8]}", event_kind="act_opened", trace_id=f"act:{act.act_id}",
        correlation_id=str(ask_id), emitted_at=datetime.now(timezone.utc), observed_at=at, layer="act",
        dimensions=["act", act.verb, act.level], act=act,
        provenance=GrammarProvenanceV1(source_service="orion-sql-writer", source_component="camera_ask_acts"),
    )


# --- attention patch 4b: the ask bids BEFORE its card opens -------------------------------------------------
# The candidate want is keyed by what the ask is about, not by the ask id (a fresh uuid every opener tick), so a
# held candidate re-bids as the SAME want and the market's event history counts it once.
CANDIDATE_SOURCE = "camera_ask_candidate"
# Evidence ref an ask carries when its card opened on a market answer at open (``want:<want_id>``). The 5-minute
# shadow loop (services/orion-sql-writer/app/ask_event_wants.py) posts nothing for such an ask: one want per ask.
BID_WANT_REF_PREFIX = "want:"
# Re-bid backoff for a held candidate (decided on issue #3285: a held ask keeps bidding with NO deadline). The
# n-th consecutive hold waits REBID_FIRST_SEC x 2^(n-1), at least the market's own wait, capped at REBID_CAP_SEC.
# 600 s is the binding rule's default wait for a sated want (orion/physiology/binding.py REBID_DEFAULT_SEC). The cap
# is 3 h, not the rule's 1 h: in the live 30-day replay (event_wants_replay.py --camera-asks) 1 h and 3 h open the
# same asks with the same delays, but 3 h makes 8.6 re-bids per held candidate-day instead of 24, and ~100 vehicle
# candidates sit held at once (2026-11-13), so 1 h would triple want_ledger's ~800 rows a day.
REBID_FIRST_SEC = 600.0
REBID_CAP_SEC = 10800.0


def candidate_key(rule: str, source_kind: str, source_ref: str) -> str:
    """What a candidate ask is about: the same rule, kind and subject is the same candidate on every tick."""
    return f"{rule}:{source_kind}:{source_ref}"


def rebid_after_sec(n_held: int, market_wait_sec: Optional[float] = None) -> float:
    """Seconds until the next bid of a candidate held ``n_held`` times in a row (>= 1). Pure."""
    n = max(1, int(n_held))
    wait = REBID_FIRST_SEC * (2 ** min(n - 1, 16))
    if isinstance(market_wait_sec, (int, float)) and market_wait_sec == market_wait_sec:
        wait = max(wait, float(market_wait_sec))
    return min(REBID_CAP_SEC, wait)


def bid_want_ref(evidence_refs: Any) -> Optional[str]:
    """The want id an ask bid at open with (``want:<id>``), or None (opened without a market answer)."""
    for r in _as_list(evidence_refs):
        if isinstance(r, str) and r.startswith(BID_WANT_REF_PREFIX):
            return r[len(BID_WANT_REF_PREFIX):]
    return None


def ask_act_ids(ask_ids: Sequence[str]) -> dict[str, str]:
    """ask_id -> act_id (deterministic)."""
    from orion.grammar.acts import stable_act_id

    return {a: stable_act_id(ACTOR, VERB, str(a)) for a in ask_ids}


__all__ = [
    "ACTOR", "BID_WANT_REF_PREFIX", "CANDIDATE_SOURCE", "REBID_CAP_SEC", "REBID_FIRST_SEC", "bid_want_ref",
    "candidate_key", "rebid_after_sec", "AskGrade", "BASIS", "CAMERA_ASK_KINDS", "JUNIPER_STORY_CARD_REF", "ORION_MADE_CAMERA_ASK_SQL", "RULES",
    "SIGNAL", "VERB", "ask_act", "ask_act_ids", "ask_grade", "ask_opened_event", "ask_rule", "is_juniper_card",
    "is_orion_camera_ask", "PRIOR_WEIGHT", "rule_from_test_before", "rule_outcome_values",
]
