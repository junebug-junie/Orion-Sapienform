from __future__ import annotations

from orion.autonomy.models import AutonomyStateV1, AutonomySummaryV1


def _bounded_unique(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text and text not in out:
            out.append(text[:120])
        if len(out) >= limit:
            break
    return out


def summarize_autonomy_state(state: AutonomyStateV1 | None) -> AutonomySummaryV1:
    if state is None:
        return AutonomySummaryV1(
            stance_hint="maintain stable direct response",
            top_drives=[],
            active_tensions=[],
            proposal_headlines=[],
            response_hazards=[],
            raw_state_present=False,
        )

    dominant = (state.dominant_drive or "").strip().lower()
    stance_hint = {
        "coherence": "favor synthesis and reduction",
        "continuity": "preserve continuity and thread integrity",
        "relational_stability": "protect relational steadiness",
    }.get(dominant, "maintain stable direct response")

    drives_sorted = sorted(
        state.drive_pressures.items(),
        key=lambda item: (float(item[1]), item[0]),
        reverse=True,
    )
    drive_names = [name for name, _ in drives_sorted]
    top_drives = _bounded_unique(drive_names + list(state.active_drives), limit=3)

    active_tensions = _bounded_unique(list(state.tension_kinds), limit=3)
    proposal_headlines = _bounded_unique([goal.goal_statement[:90] for goal in state.goal_headlines], limit=3)

    hazards: list[str] = []
    if (state.drive_pressures.get("continuity") or 0.0) >= 0.6 or dominant == "continuity":
        hazards.append("avoid abrupt thread pivots")
    if (state.drive_pressures.get("coherence") or 0.0) >= 0.6 or dominant == "coherence":
        hazards.append("avoid contradictory framing")
    relational_pressure = max(
        float(state.drive_pressures.get("relational_stability") or 0.0),
        float(state.drive_pressures.get("relational") or 0.0),
    )
    if relational_pressure >= 0.6 or dominant == "relational_stability":
        hazards.append("avoid relational coldness")
    if state.goal_headlines:
        hazards.append("do not present proposals as commitments")

    return AutonomySummaryV1(
        stance_hint=stance_hint,
        top_drives=top_drives,
        active_tensions=active_tensions,
        proposal_headlines=proposal_headlines,
        response_hazards=_bounded_unique(hazards, limit=4),
        raw_state_present=True,
    )
