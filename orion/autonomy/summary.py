from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from orion.autonomy.models import (
    AutonomyStateQuality,
    AutonomyStateV1,
    AutonomyStateV2,
    AutonomyStanceMode,
    AutonomySummaryV1,
    DriveCompetitionSummaryV1,
)

if TYPE_CHECKING:
    from orion.autonomy.repository import AutonomyLookupV1

# Mirrors orion.spark.concept_induction.tensions.derive_pressure_competition_tensions spread logic
# so the summary reflects drive disagreement even when GraphDB has not yet materialized tension rows.
_DRIVE_KEYS_FOR_SPREAD = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")
# Tuned so saturated-but-not-identical drive rows still surface a competition signal in the UI.
_PRESSURE_SPREAD_THRESHOLD = 0.06
_PRESSURE_COMPETITION_KIND = "tension.drive_competition.v1"


def _canonical_pressures_for_spread(pressures: dict[str, float]) -> dict[str, float]:
    """Merge legacy / alternate labels into the six canonical drive keys before spread checks."""
    out = {k: float(pressures.get(k, 0.0)) for k in _DRIVE_KEYS_FOR_SPREAD}
    rs = float(pressures.get("relational_stability") or 0.0)
    if rs:
        out["relational"] = max(out["relational"], rs)
    return out


def _analyze_drive_competition(pressures: dict[str, float]) -> DriveCompetitionSummaryV1 | None:
    """Return structured top vs runner when pressure disagreement crosses the competition threshold."""
    if not pressures:
        return None
    canon = _canonical_pressures_for_spread(pressures)
    canon_vals = list(canon.values())
    spread_c = max(canon_vals) - min(canon_vals) if len(canon_vals) >= 2 else 0.0
    raw_nums = [float(v) for v in pressures.values() if isinstance(v, (int, float))]
    spread_r = max(raw_nums) - min(raw_nums) if len(raw_nums) >= 2 else 0.0
    ok_c = spread_c >= _PRESSURE_SPREAD_THRESHOLD
    ok_r = spread_r >= _PRESSURE_SPREAD_THRESHOLD
    if not ok_c and not ok_r:
        return None
    if ok_c:
        ranked = sorted(_DRIVE_KEYS_FOR_SPREAD, key=lambda k: float(canon.get(k, 0.0)), reverse=True)
        top_k, runner_k = ranked[0], ranked[1]
        p_top = float(canon.get(top_k, 0.0))
        p_run = float(canon.get(runner_k, 0.0))
        spread = float(spread_c)
    else:
        items = sorted(
            [(str(k), float(v)) for k, v in pressures.items() if isinstance(v, (int, float))],
            key=lambda x: x[1],
            reverse=True,
        )
        if len(items) < 2:
            return None
        top_k, p_top = items[0]
        runner_k, p_run = items[1]
        spread = float(spread_r)
    return DriveCompetitionSummaryV1(
        top_drive=top_k,
        runner_drive=runner_k,
        spread=min(1.0, max(0.0, spread)),
        pressure_top=min(1.0, max(0.0, p_top)),
        pressure_runner=min(1.0, max(0.0, p_run)),
    )


def _proposal_headline_for_display(raw: str) -> str:
    """Strip operational suffixes (trace, leaked chat) from persisted goal_statement for UI summaries."""
    text = " ".join(str(raw or "").split()).strip()
    if not text:
        return ""
    if " · " in text:
        return text.split(" · ", 1)[0].strip()[:120]
    return text[:120]


def _bounded_unique(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text and text not in out:
            out.append(text[:120])
        if len(out) >= limit:
            break
    return out


_FACET_NAMES = ("identity", "drives", "goals")
_OK_FACET_STATUSES = frozenset({"ok", "empty"})


def _facet_health_from_diagnostics(diagnostics: dict[str, dict[str, object]] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _FACET_NAMES:
        diag = (diagnostics or {}).get(name) or {}
        out[name] = str(diag.get("status") or "unknown")
    return out


def _derive_state_quality(*, availability: str, facet_health: dict[str, str]) -> AutonomyStateQuality:
    if availability == "unavailable":
        return "unavailable"
    if availability == "empty":
        return "empty"
    if all(facet_health.get(name, "unknown") in _OK_FACET_STATUSES for name in _FACET_NAMES):
        return "healthy"
    drives_status = facet_health.get("drives", "unknown")
    identity_status = facet_health.get("identity", "unknown")
    goals_status = facet_health.get("goals", "unknown")
    if drives_status == "timeout":
        return "degraded_drives_timeout"
    if drives_status not in _OK_FACET_STATUSES:
        return "degraded_drives_error"
    if identity_status == "timeout":
        return "degraded_identity_timeout"
    if identity_status not in _OK_FACET_STATUSES:
        return "degraded_partial"
    if goals_status == "timeout":
        return "degraded_goals_timeout"
    if goals_status not in _OK_FACET_STATUSES:
        return "degraded_partial"
    return "degraded_partial"


def _derive_degraded_reason(*, state_quality: AutonomyStateQuality, selected_subject: str | None, facet_health: dict[str, str]) -> str | None:
    subject_label = (selected_subject or "selected subject").replace("_", " ").title()
    if state_quality == "degraded_drives_timeout":
        return f"{subject_label} drives facet timed out"
    if state_quality == "degraded_drives_error":
        return f"{subject_label} drives facet failed ({facet_health.get('drives', 'error')})"
    if state_quality == "degraded_identity_timeout":
        return f"{subject_label} identity facet timed out"
    if state_quality == "degraded_goals_timeout":
        return f"{subject_label} goals facet timed out"
    if state_quality == "degraded_partial":
        failed = [name for name in _FACET_NAMES if facet_health.get(name) not in _OK_FACET_STATUSES]
        if failed:
            return f"{subject_label} partial facet failure ({', '.join(failed)})"
    if state_quality == "unavailable":
        return f"{subject_label} autonomy lookup unavailable"
    return None


def _derive_context_note(
    *,
    selected_subject: str | None,
    state_quality: AutonomyStateQuality,
    by_subject: Mapping[str, "AutonomyLookupV1"] | None,
) -> str | None:
    if not selected_subject or not by_subject:
        return None
    if state_quality not in {"degraded_drives_timeout", "degraded_drives_error", "degraded_partial"}:
        return None
    selected_drives = str(((by_subject.get(selected_subject).subquery_diagnostics or {}).get("drives") or {}).get("status", ""))
    if selected_drives in _OK_FACET_STATUSES:
        return None
    if selected_subject == "orion":
        rel = by_subject.get("relationship")
        rel_diag = (rel.subquery_diagnostics or {}).get("drives") if rel else None
        if rel_diag and str(rel_diag.get("status")) == "ok" and int(rel_diag.get("row_count") or 0) > 0:
            return "relationship drives are available, but were not substituted for Orion drives"
    return None


def _derive_stance_mode(*, state_quality: AutonomyStateQuality, has_proposals: bool) -> AutonomyStanceMode:
    if state_quality == "unavailable":
        return "unavailable"
    if state_quality in {"degraded_drives_timeout", "degraded_drives_error", "degraded_partial"}:
        return "proposal_only" if has_proposals else "unavailable"
    if state_quality == "empty":
        return "unavailable"
    return "proposal_only" if has_proposals else "normal"


def summarize_autonomy_lookup(
    state: AutonomyStateV1 | AutonomyStateV2 | None,
    *,
    selected_subject: str | None = None,
    availability: str = "empty",
    subquery_diagnostics: dict[str, dict[str, object]] | None = None,
    by_subject: Mapping[str, "AutonomyLookupV1"] | None = None,
) -> AutonomySummaryV1:
    base = summarize_autonomy_state(state)
    facet_health = _facet_health_from_diagnostics(subquery_diagnostics)
    state_quality = _derive_state_quality(availability=availability, facet_health=facet_health)
    has_proposals = bool(base.proposal_headlines)
    degraded_reason = _derive_degraded_reason(
        state_quality=state_quality,
        selected_subject=selected_subject,
        facet_health=facet_health,
    )
    context_note = _derive_context_note(
        selected_subject=selected_subject,
        state_quality=state_quality,
        by_subject=by_subject,
    )
    stance_mode = _derive_stance_mode(state_quality=state_quality, has_proposals=has_proposals)
    return base.model_copy(
        update={
            "state_quality": state_quality,
            "stance_mode": stance_mode,
            "degraded_reason": degraded_reason,
            "facet_health": facet_health,
            "context_note": context_note,
            "selected_subject": selected_subject,
        }
    )


def summarize_autonomy_state(state: AutonomyStateV1 | AutonomyStateV2 | None) -> AutonomySummaryV1:
    if state is None:
        return AutonomySummaryV1(
            stance_hint="maintain stable direct response",
            dominant_drive=None,
            top_drives=[],
            active_tensions=[],
            proposal_headlines=[],
            response_hazards=[],
            raw_state_present=False,
            drive_competition=None,
            state_quality="empty",
            stance_mode="unavailable",
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

    tension_sources = list(state.tension_kinds)
    drive_competition = _analyze_drive_competition(state.drive_pressures)
    if drive_competition and _PRESSURE_COMPETITION_KIND not in tension_sources:
        tension_sources.append(_PRESSURE_COMPETITION_KIND)
    active_tensions = _bounded_unique(tension_sources, limit=3)
    proposal_headlines = _bounded_unique(
        [_proposal_headline_for_display(goal.goal_statement) for goal in state.goal_headlines],
        limit=3,
    )
    if isinstance(state, AutonomyStateV2) and not state.goal_headlines and state.attention_items:
        proposal_headlines = _bounded_unique([a.summary for a in state.attention_items], limit=3)

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

    hazard_limit = 8 if isinstance(state, AutonomyStateV2) else 4
    if isinstance(state, AutonomyStateV2):
        if float(state.confidence) < 0.4:
            hazards.append("avoid overconfident inner-state claims")
        if state.unknowns:
            hazards.append("surface uncertainty when state evidence is thin")
        if any(
            getattr(i, "inhibition_reason", None) == "proxy_signal_not_canonical_state"
            for i in state.inhibited_impulses
        ):
            hazards.append("do not treat proxy telemetry as canonical state")

    return AutonomySummaryV1(
        stance_hint=stance_hint,
        dominant_drive=(state.dominant_drive or "").strip() or None,
        top_drives=top_drives,
        active_tensions=active_tensions,
        proposal_headlines=proposal_headlines,
        response_hazards=_bounded_unique(hazards, limit=hazard_limit),
        raw_state_present=True,
        drive_competition=drive_competition,
        state_quality="healthy",
        stance_mode="proposal_only" if proposal_headlines else "normal",
    )
