"""Prime's choice point (personality spec rev 4, move 1): one ``PersonalityChoiceV1`` per thought.

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md (rev 4 "Move 1",
rev 3 "Prime wiring"). Each Prime thought already chooses at most one held concern to act on, or
none. The thought's answer now says so explicitly (``PrimeAnswerV1.pick``: a concern_id from YOUR
CONCERNS, or "none"), in the same model call. No new model call.

The options are the concerns the frame showed (captured when the frame was built, so a concern the
same answer adds or drops does not change what Orion chose among). Rules, all deterministic:

* pick = a shown concern and at least 2 concerns shown: chosen = that concern, alternatives = the
  others.
* pick = "none" (or JSON null) and at least 1 concern shown: ``declined_all``, alternatives = all.
* a pick that contradicts the move's ``concern_id`` (both naming shown concerns, or "none" with a move about
  a shown concern): no row (``pick_conflicts_move``).
* no ``pick`` key: the move's ``concern_id`` counts as the pick when it names a shown concern (a
  move about a concern is a choice of it). Otherwise no row: the answer did not say.
* anything else (one concern and it was picked, an unknown id, "new", nothing held): no row, with
  the reason, so the log shows why a thought made no choice.

``market_rank``: Prime holds concerns in the order it added them, which is not a ranking. The one
real order is the body economy's: in event mode (PRIME_CONCERNS_FROM_EVENT_WANTS) each concern names
a funded event want, and the frame lists those "highest value first" (durable-runs
GET /physiology/event-wants). A concern's market rank is its event's position in that list, counted
among the shown concerns only (1 = the option the market valued most). A concern whose event is not
in the list (a turn promise, a day-letter thread, arc mode) has no rank (None). When fewer than 2
shown concerns are in the list, the market ordered nothing among them: every rank is None.

``situation``: what Prime's BODY section showed: the binding constraints and the strain dial (Prime's
frame shows no arousal reading; strain is the body-state number it does see).

Pure. The runtime (services/orion-mind/app/prime_loop.py) publishes the result, fail-open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from orion.schemas.personality import CHOICE_OPTION_TEXT_MAX, ChoiceOptionV1, PersonalityChoiceV1

LANE = "prime"
PICK_NONE = "none"


@dataclass(frozen=True)
class ShownConcern:
    """One concern as the frame showed it."""

    concern_id: str
    text: str
    event_ref: Optional[str] = None


def shown_concerns(concerns: Sequence[Any]) -> list[ShownConcern]:
    """Snapshot of the concerns in the frame (``PrimeConcernV1``-like objects)."""
    return [ShownConcern(c.concern_id, c.text, getattr(c, "event_ref", None)) for c in concerns]


def market_ranks(shown: Sequence[ShownConcern], ranked_event_ids: Sequence[str]) -> dict[str, Optional[int]]:
    """concern_id -> its rank among the shown concerns by the market's event order (1 = highest value);
    None when its event is not in the list. Fewer than 2 placed: the market ordered nothing among these
    options, so every rank is None (a lone placed concern is not "the market's top" over unranked ones)."""
    pos = {e: i for i, e in enumerate(ranked_event_ids)}
    placed = sorted((pos[c.event_ref], c.concern_id) for c in shown if c.event_ref in pos)
    ranks: dict[str, Optional[int]] = {c.concern_id: None for c in shown}
    if len(placed) < 2:
        return ranks
    for r, (_p, cid) in enumerate(placed, start=1):
        ranks[cid] = r
    return ranks


def _text(text: str) -> str:
    text = " ".join((text or "").split()) or "(no words)"
    return text if len(text) <= CHOICE_OPTION_TEXT_MAX else text[: CHOICE_OPTION_TEXT_MAX - 1] + "…"


def resolve_pick(answer: Any, shown: Sequence[ShownConcern]) -> tuple[Optional[str], Optional[str]]:
    """(pick, why-not). pick is a shown concern_id or PICK_NONE; None with the reason when the answer
    made no usable pick."""
    ids = {c.concern_id for c in shown}
    move_cid = getattr(getattr(answer, "move", None), "concern_id", None)
    move_cid = move_cid if move_cid in ids else None
    if "pick" in getattr(answer, "model_fields_set", set()):
        raw = answer.pick
        pick = PICK_NONE if raw is None else str(raw).strip()
        if pick.lower() in (PICK_NONE, "null", "no_move", "no move"):
            pick = PICK_NONE
        elif pick not in ids:
            return None, "pick_not_shown"
        # The prompt says a move about a concern picks it: a pick that contradicts the act Orion took is
        # not evidence either way (it would be a false decline or a false departure).
        if move_cid is not None and pick != move_cid:
            return None, "pick_conflicts_move"
        return pick, None
    if move_cid is not None:
        return move_cid, None
    return None, "no_pick"


def prime_choice(
    answer: Any,
    shown: Sequence[ShownConcern],
    *,
    thought_id: str,
    occurred_at: datetime,
    ranked_event_ids: Sequence[str] = (),
    situation: Optional[dict[str, Any]] = None,
    act_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> tuple[Optional[PersonalityChoiceV1], str]:
    """(the choice, reason). The choice is None when this thought made no recordable choice; the
    reason then says why (``no_concerns``, ``one_concern_picked``, ``pick_not_shown``, ``pick_conflicts_move``,
    ``no_pick``)."""
    if not shown:
        return None, "no_concerns"
    pick, why = resolve_pick(answer, shown)
    if pick is None:
        return None, why or "no_pick"
    ranks = market_ranks(shown, ranked_event_ids)
    options = {c.concern_id: ChoiceOptionV1(key=c.concern_id[:200], text=_text(c.text),
                                            market_rank=ranks[c.concern_id]) for c in shown}
    common = dict(choice_id=f"{LANE}:{thought_id}", lane=LANE, occurred_at=occurred_at,
                  situation=dict(situation or {}), act_id=act_id, correlation_id=correlation_id)
    if pick == PICK_NONE:
        return PersonalityChoiceV1(declined_all=True, alternatives=list(options.values()), **common), "declined_all"
    if len(shown) < 2:
        return None, "one_concern_picked"
    alternatives = [o for k, o in options.items() if k != pick]
    return PersonalityChoiceV1(chosen=options[pick], alternatives=alternatives, **common), "chosen"


def body_situation(body: Optional[dict[str, Any]], binding: Optional[Sequence[str]], *,
                   event_mode: bool) -> dict[str, Any]:
    """The flat situation dict: what the BODY section showed. Unread body -> body_read False."""
    out: dict[str, Any] = {"body_read": body is not None, "event_mode": bool(event_mode)}
    if binding is not None:
        out["binding"] = ",".join(binding)[:200]
        out["binding_count"] = len(binding)
    strain = (body or {}).get("strain")
    if isinstance(strain, dict):
        value = strain.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out["strain"] = round(float(value), 4)
        if strain.get("constraint"):
            out["strain_constraint"] = str(strain["constraint"])[:200]
    return out


__all__ = ["LANE", "PICK_NONE", "ShownConcern", "body_situation", "market_ranks", "prime_choice",
           "resolve_pick", "shown_concerns"]
