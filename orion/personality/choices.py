"""Pure reducer: curiosity offer/outcome rows -> ``PersonalityChoiceV1`` (spec rev 4).

The curiosity lane already stores a choice together with the options that lost.
``curiosity_offer_decisions.offered`` is the live priors Orion was shown: an ordered head (``slot``
``order``) followed by priors "DRAWN AT RANDOM, not ranked" (``slot`` ``coverage``;
``orion/curiosity/worldview.py`` ``select_priors``). Only the ordered head keeps its rank.
``stale_offered`` is the separate "tested repeatedly" section of the same prompt.
``curiosity_run_outcomes.per_prior`` says what the run touched.

**Chosen = per_prior entries with kind ``tested`` only.** ``formed`` is a NEW prior Orion wrote
during the run: it was never an option (live 2026-10-16: 0 of 30 appear in ``offered``).
``moved_untested`` is protocol drift, not a declared test. A tested prior is chosen wherever it came
from: the ordered list (has a rank), the stale section (rank None), or off-menu (rank None).

One run can test several priors. Each tested prior is ONE choice row
(``choice_id = curiosity:<run_id>:<prior_id>``) whose alternatives are the shown priors the run did
NOT test, with ``picks_in_decision`` = the number of tested priors, so the reducer weighs one run as
one decision. A run whose every shown prior was tested has no alternative and yields no row.

Option text is the prior's own claim, as Orion was shown it: the offer row stores it per offered prior
(``claim``, from 2026-11-13; ``services/orion-hub/scripts/curiosity_offer_decisions.py offered_rows``).
An older row (no ``claim``), or a tested prior that was not on the menu, falls back to the prior's
slug made readable (``self:peer_briefs_refused_budget_20260930`` -> ``peer briefs refused budget``).

Every function here never raises; ``explain_curiosity_choices`` also returns a reason bucket when
it yields nothing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from orion.curiosity.value import ARM_SELF_INQUIRY  # a self-inquiry run's decision row (nothing offered)
from orion.schemas.personality import (
    CHOICE_OPTION_TEXT_MAX,
    MAX_CHOICE_LIST,
    ChoiceOptionV1,
    PersonalityChoiceV1,
)

LANE_CURIOSITY = "curiosity"
KIND_TESTED = "tested"
CURIOSITY_CHOICE_PREFIX = "curiosity:"

_WS = re.compile(r"\s+")
_DATE_SUFFIX = re.compile(r"_\d{8}$")


def _json(value: Any) -> Any:
    """jsonb columns come back parsed from psycopg, but a text dump gives strings."""
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def _dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def _rank(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if out >= 1 else None


def _str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def readable_prior(prior_id: Any) -> str:
    """``self:reading_link_can_cross_region_boundary_20261015`` -> ``reading link can cross region
    boundary``."""
    s = _str(prior_id) or ""
    if s.startswith("curiosity_prior:"):
        s = s[len("curiosity_prior:"):]
    if s.startswith("self:"):
        s = s[len("self:"):]
    s = _DATE_SUFFIX.sub("", s)
    return _WS.sub(" ", s.replace("_", " ")).strip()[:CHOICE_OPTION_TEXT_MAX]


def claim_text(claim: Any) -> Optional[str]:
    """A stored claim, whitespace-collapsed and cut to the option cap; None when empty."""
    if not isinstance(claim, str):
        return None
    t = _WS.sub(" ", claim).strip()[:CHOICE_OPTION_TEXT_MAX]
    return t or None


def _option(prior_id: str, rank: Optional[int], claim: Optional[str] = None) -> Optional[ChoiceOptionV1]:
    text = claim_text(claim) or readable_prior(prior_id)
    if not text:
        return None
    return ChoiceOptionV1(key=prior_id[:200], text=text, market_rank=rank)


def _menu(rows: Any, *, ranked: bool) -> Dict[str, Tuple[Optional[int], Optional[str]]]:
    """prior_id -> (shown rank, stored claim), first occurrence wins. Stale-section and coverage rows
    are shown as unranked sections, so their rank is dropped (review 2026-10-16)."""
    out: Dict[str, Tuple[Optional[int], Optional[str]]] = {}
    for row in _json(rows) or []:
        if not isinstance(row, Mapping):
            continue
        prior_id = _str(row.get("prior_id"))
        if prior_id is None or prior_id in out:
            continue
        rank = _rank(row.get("rank")) if ranked and row.get("slot") != "coverage" else None
        out[prior_id] = (rank, claim_text(row.get("claim")))
    return out


def explain_curiosity_choices(
    offer: Optional[Mapping[str, Any]], outcome: Optional[Mapping[str, Any]]
) -> Tuple[List[PersonalityChoiceV1], Optional[str]]:
    """(choices, None) or ([], reason). Never raises."""
    try:
        if not isinstance(offer, Mapping):
            return [], "no_offer_row"
        if not isinstance(outcome, Mapping):
            return [], "no_outcome_row"
        run_id = _str(offer.get("run_id"))
        if run_id is None:
            return [], "no_run_id"
        if _str(offer.get("arm")) == ARM_SELF_INQUIRY:
            return [], "self_inquiry_no_offer"
        out_run = _str(outcome.get("run_id"))
        if out_run is not None and out_run != run_id:
            return [], "run_id_mismatch"
        per_prior = _json(outcome.get("per_prior"))
        if not isinstance(per_prior, list):
            return [], "malformed_per_prior"
        tested: List[str] = []
        for e in per_prior:
            pid = _str(e.get("prior_id")) if isinstance(e, Mapping) and e.get("kind") == KIND_TESTED else None
            if pid is not None and pid not in tested:
                tested.append(pid)
        if not tested:
            formed = any(isinstance(e, Mapping) and e.get("kind") == "formed" for e in per_prior)
            return [], "formed_only" if formed else "nothing_tested"

        shown = _menu(offer.get("offered"), ranked=True)
        for pid, entry in _menu(offer.get("stale_offered"), ranked=False).items():
            shown.setdefault(pid, entry)
        alternatives = [o for pid, (rank, claim) in shown.items() if pid not in tested
                        for o in [_option(pid, rank, claim)] if o is not None][:MAX_CHOICE_LIST]
        if not alternatives:
            return [], "no_alternative"

        occurred = (_dt(offer.get("turn_started_at")) or _dt(offer.get("decided_at"))
                    or _dt(outcome.get("completed_at")))
        if occurred is None:
            return [], "no_timestamp"
        situation: Dict[str, Any] = {}
        arm = _str(offer.get("arm"))
        if arm:
            situation["curiosity_arm"] = arm
        if isinstance(outcome.get("turn_ok"), bool):
            situation["turn_ok"] = outcome["turn_ok"]

        picks = min(len(tested), MAX_CHOICE_LIST)
        out: List[PersonalityChoiceV1] = []
        for pid in tested[:MAX_CHOICE_LIST]:
            rank, claim = shown.get(pid, (None, None))
            chosen = _option(pid, rank, claim)
            if chosen is None:
                continue
            try:
                out.append(PersonalityChoiceV1(
                    choice_id=f"{CURIOSITY_CHOICE_PREFIX}{run_id}:{pid}"[:300],
                    lane=LANE_CURIOSITY, occurred_at=occurred, chosen=chosen,
                    alternatives=alternatives, chosen_market_rank=chosen.market_rank,
                    picks_in_decision=picks, situation=situation, act_id=f"curiosity_run:{run_id}"[:300],
                ))
            except Exception:  # noqa: BLE001 -- one bad pick never sinks the run's other picks
                continue
        return (out, None) if out else ([], "invalid_model")
    except Exception:  # noqa: BLE001
        return [], "reducer_error"


def curiosity_choices_from_rows(
    offer: Optional[Mapping[str, Any]], outcome: Optional[Mapping[str, Any]]
) -> List[PersonalityChoiceV1]:
    return explain_curiosity_choices(offer, outcome)[0]
