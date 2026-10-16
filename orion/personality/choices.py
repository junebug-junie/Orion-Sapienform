"""Pure reducers: stored rows -> ``PersonalityChoiceV1`` (PR #2726, section 1 "Evidence").

Two lanes already persist a choice together with the options that lost:

1. ``curiosity_prior_selection``. ``curiosity_offer_decisions.offered`` is the ordered list of
   live priors Orion was shown; ``stale_offered`` is the separate "tested repeatedly" section of
   the same prompt. ``curiosity_run_outcomes.per_prior`` says what the run touched.

   **Chosen = per_prior entries with kind ``tested`` only.** Live distinct kinds (2026-10-16,
   all rows): ``tested`` 86, ``formed`` 30, ``moved_untested`` 0. ``formed`` is a NEW prior
   Orion wrote during the run: it was never an option (0 of 30 appear in ``offered``), so it is
   not a pick among alternatives. ``moved_untested`` is protocol drift (confidence moved without
   ``times_tested + 1``; ``orion/curiosity/value.py``), not a declared test. A tested prior is
   chosen wherever it came from: the ordered list (has a rank), the stale section (rank None),
   or off-menu (rank None, score None). Live: of 86 tested, 54 were in ``offered``, 20 in
   ``stale_offered``, 12 in neither.

2. ``voluntary_attention_override``. ``frame.voluntary_override`` names the loop Orion's goal
   bias picked (``chosen_loop_id``) and the bottom-up winner it beat (``beat_loop_id``).
   ``frame.open_loops`` holds every candidate; bottom-up score is ``loop.salience``
   (``orion/substrate/attention_broadcast.py`` ``_apply_voluntary_attention``). Rank is the
   bottom-up order, so ``chosen_best_rank`` > 1 by construction: an override always departs
   from the bottom-up order.

Every function here never raises and returns None for absent or malformed input. The
``explain_*`` variants also return a short reason bucket for the None case.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from orion.schemas.personality import MAX_CHOICE_LIST, ChoiceOptionV1, PersonalityChoiceV1

LANE_CURIOSITY = "curiosity_prior_selection"
LANE_OVERRIDE = "voluntary_attention_override"
KIND_TESTED = "tested"
PRIOR_REF_PREFIX = "curiosity_prior:"


# ------------------------------------------------------------------ helpers


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


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rank(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 1 else None


def _str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _dedupe(refs: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def _build(**kwargs: Any) -> Tuple[Optional[PersonalityChoiceV1], Optional[str]]:
    try:
        return PersonalityChoiceV1(**kwargs), None
    except Exception:  # noqa: BLE001 -- a reducer never raises
        return None, "invalid_model"


# ------------------------------------------------------------- curiosity lane


def _menu(rows: Any, *, ranked: bool) -> Dict[str, ChoiceOptionV1]:
    """prior_id -> option, first occurrence wins. Stale-section rows carry ranks within
    their own list; that is not the order Orion was told to read, so it is dropped."""
    out: Dict[str, ChoiceOptionV1] = {}
    for row in _json(rows) or []:
        if not isinstance(row, Mapping):
            continue
        prior_id = _str(row.get("prior_id"))
        if prior_id is None or prior_id in out:
            continue
        out[prior_id] = ChoiceOptionV1(
            ref=PRIOR_REF_PREFIX + prior_id,
            rank=_rank(row.get("rank")) if ranked else None,
            score=_num(row.get("expected_nats")),
        )
    return out


def explain_curiosity_choice(
    offer: Optional[Mapping[str, Any]], outcome: Optional[Mapping[str, Any]]
) -> Tuple[Optional[PersonalityChoiceV1], Optional[str]]:
    """(choice, None) or (None, reason). Never raises."""
    try:
        if not isinstance(offer, Mapping):
            return None, "no_offer_row"
        if not isinstance(outcome, Mapping):
            return None, "no_outcome_row"
        run_id = _str(offer.get("run_id"))
        if run_id is None:
            return None, "no_run_id"
        out_run = _str(outcome.get("run_id"))
        if out_run is not None and out_run != run_id:
            return None, "run_id_mismatch"
        per_prior = _json(outcome.get("per_prior"))
        if not isinstance(per_prior, list):
            return None, "malformed_per_prior"
        tested = _dedupe(
            [
                pid
                for e in per_prior
                if isinstance(e, Mapping) and e.get("kind") == KIND_TESTED
                for pid in [_str(e.get("prior_id"))]
                if pid is not None
            ]
        )
        if not tested:
            formed = any(isinstance(e, Mapping) and e.get("kind") == "formed" for e in per_prior)
            return None, "formed_only" if formed else "nothing_tested"

        live = _menu(offer.get("offered"), ranked=True)
        stale = _menu(offer.get("stale_offered"), ranked=False)
        chosen: List[ChoiceOptionV1] = []
        for pid in tested:
            opt = live.get(pid) or stale.get(pid) or ChoiceOptionV1(ref=PRIOR_REF_PREFIX + pid)
            chosen.append(opt)
        chosen_ids = set(tested)
        alternatives = [o for pid, o in live.items() if pid not in chosen_ids]
        alternatives += [o for pid, o in stale.items() if pid not in chosen_ids and pid not in live]
        ranks = [o.rank for o in chosen if o.rank is not None]

        started = _dt(offer.get("turn_started_at"))
        decided = _dt(offer.get("decided_at"))
        completed = _dt(outcome.get("completed_at"))
        occurred = started or decided or completed
        if occurred is None:
            return None, "no_timestamp"
        cost_seconds = None
        if started is not None and completed is not None:
            delta = (completed - started).total_seconds()
            cost_seconds = delta if delta >= 0 else None

        refs: List[str] = []
        arm = _str(offer.get("arm"))
        if arm:
            refs.append(f"curiosity_arm:{arm}")
        if isinstance(outcome.get("turn_ok"), bool):
            refs.append(f"curiosity_turn_ok:{str(outcome['turn_ok']).lower()}")
        for mid in _json(offer.get("material_ids")) or []:
            ref = _str(mid)
            if ref:
                refs.append(ref)

        return _build(
            choice_id=f"{LANE_CURIOSITY}:{run_id}",
            lane=LANE_CURIOSITY,
            chooser="orion_llm",
            occurred_at=occurred,
            chosen=chosen[:MAX_CHOICE_LIST],
            alternatives=alternatives[:MAX_CHOICE_LIST],
            chosen_best_rank=min(ranks) if ranks else None,
            cost_seconds=cost_seconds,
            cost_nats=_num(outcome.get("realized_nats")),
            situation_refs=_dedupe(refs)[:MAX_CHOICE_LIST],
            correlation_id=None,
            source_ref=f"curiosity_run:{run_id}",
        )
    except Exception:  # noqa: BLE001
        return None, "reducer_error"


def curiosity_choice_from_rows(
    offer: Optional[Mapping[str, Any]], outcome: Optional[Mapping[str, Any]]
) -> Optional[PersonalityChoiceV1]:
    return explain_curiosity_choice(offer, outcome)[0]


# ------------------------------------------------------- attention override lane


def explain_voluntary_override_choice(
    row: Optional[Mapping[str, Any]],
) -> Tuple[Optional[PersonalityChoiceV1], Optional[str]]:
    """(choice, None) or (None, reason). ``row`` is a ``substrate_attention_broadcast_log``
    row (``log_id``, ``generated_at``, ``projection_json``). Never raises."""
    try:
        if not isinstance(row, Mapping):
            return None, "no_row"
        log_id = _str(row.get("log_id"))
        if log_id is None:
            return None, "no_log_id"
        projection = _json(row.get("projection_json"))
        if not isinstance(projection, Mapping):
            return None, "malformed_projection"
        frame = projection.get("frame")
        if not isinstance(frame, Mapping):
            return None, "no_frame"
        vo = frame.get("voluntary_override")
        if vo is None:
            return None, "no_override"
        if not isinstance(vo, Mapping):
            return None, "malformed_override"
        chosen_id = _str(vo.get("chosen_loop_id"))
        beat_id = _str(vo.get("beat_loop_id"))
        if chosen_id is None or beat_id is None or chosen_id == beat_id:
            return None, "malformed_override"

        loops = [
            (i, loop)
            for i, loop in enumerate(frame.get("open_loops") or [])
            if isinstance(loop, Mapping) and _str(loop.get("id"))
        ]
        scored = sorted(
            [(i, loop) for i, loop in loops if _num(loop.get("salience")) is not None],
            # On a salience tie the beat goes first: the override record says it was the
            # bottom-up winner (live 2026-10-10: one row with both loops at 0.5).
            key=lambda t: (-_num(t[1].get("salience")), _str(t[1].get("id")) != beat_id, t[0]),
        )
        rank_of = {_str(loop.get("id")): r for r, (_, loop) in enumerate(scored, start=1)}
        options: Dict[str, ChoiceOptionV1] = {}
        source_refs: Dict[str, List[str]] = {}
        for _, loop in loops:
            lid = _str(loop.get("id"))
            if lid in options:
                continue
            options[lid] = ChoiceOptionV1(
                ref=lid, rank=rank_of.get(lid), score=_num(loop.get("salience"))
            )
            source_refs[lid] = [s for s in (loop.get("source_refs") or []) if _str(s)]
        # The override record itself is the ground truth for the pair; open_loops only adds
        # the rest of the field and the ranks. If the pair is missing from it, rank the pair
        # from the record (the beat was the bottom-up winner by definition).
        if chosen_id not in options or beat_id not in options:
            options.setdefault(
                beat_id, ChoiceOptionV1(ref=beat_id, rank=1, score=_num(vo.get("beat_bottom_up")))
            )
            options.setdefault(
                chosen_id,
                ChoiceOptionV1(ref=chosen_id, rank=2, score=_num(vo.get("chosen_bottom_up"))),
            )
        chosen = options[chosen_id]
        alternatives = [o for lid, o in options.items() if lid != chosen_id]
        alternatives.sort(key=lambda o: (o.rank is None, o.rank or 0))

        occurred = _dt(row.get("generated_at")) or _dt(frame.get("generated_at"))
        if occurred is None:
            return None, "no_timestamp"

        refs: List[str] = []
        goal = _str(vo.get("goal_artifact_id"))
        if goal:
            refs.append(goal)
        origin = _str(vo.get("goal_drive_origin"))
        if origin:
            refs.append(f"goal_drive_origin:{origin}")
        refs += source_refs.get(chosen_id, []) + source_refs.get(beat_id, [])

        return _build(
            choice_id=f"{LANE_OVERRIDE}:{log_id}",
            lane=LANE_OVERRIDE,
            chooser="goal_biased_formula",
            occurred_at=occurred,
            chosen=[chosen],
            alternatives=alternatives[:MAX_CHOICE_LIST],
            chosen_best_rank=chosen.rank,
            cost_seconds=None,
            cost_nats=None,
            cost_effort=_num(vo.get("effort_spent")),
            situation_refs=_dedupe(refs)[:MAX_CHOICE_LIST],
            correlation_id=_str(frame.get("correlation_id")) or _str(projection.get("correlation_id")),
            source_ref=f"attention_broadcast:{log_id}",
        )
    except Exception:  # noqa: BLE001
        return None, "reducer_error"


def voluntary_override_choice_from_row(
    row: Optional[Mapping[str, Any]],
) -> Optional[PersonalityChoiceV1]:
    return explain_voluntary_override_choice(row)[0]
