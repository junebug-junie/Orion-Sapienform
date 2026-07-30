from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from orion.autonomy.models import (
    AutonomyActiveGoalV1,
    AutonomyGoalHeadlineV1,
    AutonomyStateQuality,
    AutonomyStateV1,
    AutonomyStateV2,
    AutonomyStanceMode,
    AutonomySummaryV1,
)

if TYPE_CHECKING:
    from orion.autonomy.repository import AutonomyLookupV1

# Drive-pressure competition analysis (_analyze_drive_competition,
# _canonical_pressures_for_spread, DriveCompetitionSummaryV1 usage) removed
# 2026-07-30 (chore/delete-orion-drives Wave 2a): AutonomyStateV1/V2 no
# longer carry drive_pressures (Wave 2a full removal, mirroring Wave 1's
# deletion of the producer), so there is no real input left to analyze.
# AutonomySummaryV1.drive_competition/dominant_drive/top_drives/
# active_tensions are unconditionally empty/None below now -- their schema
# fields were NOT removed from AutonomySummaryV1 itself in this wave (out-of-
# scope consumers, e.g. services/orion-hub, still read that model), so this
# is a forced, honest consequence of the state-field deletion, not a fresh
# stub. See the Wave 2a PR report for the follow-up decision this leaves
# open: retire these AutonomySummaryV1 fields too, or accept them as
# permanently-empty historical vestiges.


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
        if name not in (diagnostics or {}):
            out[name] = "empty"
            continue
        diag = (diagnostics or {}).get(name) or {}
        out[name] = str(diag.get("status") or "unknown")
    return out


def _derive_state_quality(*, availability: str, facet_health: dict[str, str]) -> AutonomyStateQuality:
    if availability == "unavailable":
        return "unavailable"
    if availability == "empty":
        return "empty"
    if availability == "degraded":
        drives_status = facet_health.get("drives", "unknown")
        if drives_status == "timeout":
            return "degraded_drives_timeout"
        if drives_status not in _OK_FACET_STATUSES:
            return "degraded_drives_error"
        return "degraded_partial"
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
        drives_status = facet_health.get("drives", "error")
        if drives_status == "deferred":
            return f"{subject_label} drives facet deferred"
        return f"{subject_label} drives facet failed ({drives_status})"
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
    contextual_fallback: bool = False,
) -> str | None:
    if contextual_fallback and selected_subject == "relationship":
        return "Orion drives unavailable; stance context from relationship drives (not substituted as Orion drives)"
    if not selected_subject or not by_subject:
        return None
    if state_quality not in {"degraded_drives_timeout", "degraded_drives_error", "degraded_partial"}:
        return None
    selected = by_subject.get(selected_subject)
    selected_drives = str(((selected.subquery_diagnostics or {}).get("drives") or {}).get("status", "")) if selected else ""
    if selected_drives in _OK_FACET_STATUSES:
        return None
    if selected_subject == "orion":
        rel = by_subject.get("relationship")
        rel_diag = (rel.subquery_diagnostics or {}).get("drives") if rel else None
        if rel_diag and str(rel_diag.get("status")) == "ok" and int(rel_diag.get("row_count") or 0) > 0:
            return "relationship drives are available, but were not substituted for Orion drives"
    return None


def dedupe_goal_headlines_by_drive_origin(
    goals: list[AutonomyGoalHeadlineV1],
    *,
    limit: int = 3,
) -> list[AutonomyGoalHeadlineV1]:
    """Keep highest-priority goal per drive_origin (matches repository active-goal read path)."""
    ranked = sorted(goals, key=lambda goal: (-float(goal.priority), goal.artifact_id))
    seen_origins: set[str] = set()
    out: list[AutonomyGoalHeadlineV1] = []
    for goal in ranked:
        origin = str(goal.drive_origin or "").strip().lower()
        if not origin or origin in seen_origins:
            continue
        seen_origins.add(origin)
        out.append(goal)
        if len(out) >= limit:
            break
    return out


def _active_goals_from_state(state: AutonomyStateV1 | AutonomyStateV2) -> list[AutonomyActiveGoalV1]:
    out: list[AutonomyActiveGoalV1] = []
    for goal in dedupe_goal_headlines_by_drive_origin(state.goal_headlines, limit=3):
        headline = _proposal_headline_for_display(goal.goal_statement)
        if not headline:
            continue
        out.append(
            AutonomyActiveGoalV1(
                drive_origin=goal.drive_origin,
                headline=headline,
                priority=goal.priority,
                artifact_id=goal.artifact_id,
                proposal_status=goal.proposal_status,
                planned_task_id=goal.planned_task_id,
                completed_at=goal.completed_at,
            )
        )
    return out


def _derive_stance_mode(
    *,
    state_quality: AutonomyStateQuality,
    has_proposals: bool,
    contextual_fallback: bool = False,
) -> AutonomyStanceMode:
    if contextual_fallback and state_quality == "healthy":
        return "fallback_contextual"
    if state_quality == "unavailable":
        return "unavailable"
    if state_quality in {"degraded_drives_timeout", "degraded_drives_error", "degraded_partial"}:
        return "proposal_only" if has_proposals else "unavailable"
    if state_quality == "empty":
        return "unavailable"
    return "normal"


def summarize_autonomy_lookup(
    state: AutonomyStateV1 | AutonomyStateV2 | None,
    *,
    selected_subject: str | None = None,
    availability: str = "empty",
    subquery_diagnostics: dict[str, dict[str, object]] | None = None,
    by_subject: Mapping[str, "AutonomyLookupV1"] | None = None,
    contextual_fallback: bool = False,
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
        contextual_fallback=contextual_fallback,
    )
    stance_mode = _derive_stance_mode(
        state_quality=state_quality,
        has_proposals=has_proposals,
        contextual_fallback=contextual_fallback,
    )
    active_goals = _active_goals_from_state(state) if state else []
    return base.model_copy(
        update={
            "state_quality": state_quality,
            "stance_mode": stance_mode,
            "degraded_reason": degraded_reason,
            "facet_health": facet_health,
            "context_note": context_note,
            "selected_subject": selected_subject,
            "active_goals": active_goals,
            "goals_present": bool(active_goals),
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

    # dominant_drive/drive_pressures/active_drives/tension_kinds no longer
    # exist on AutonomyStateV1/V2 (removed 2026-07-30, chore/delete-orion-
    # drives Wave 2a) -- stance_hint, top_drives, active_tensions, and
    # drive_competition are honestly empty/default below, not derived from a
    # dead source. See the module-level comment above for the AutonomySummaryV1
    # schema follow-up this leaves open.
    stance_hint = "maintain stable direct response"
    top_drives: list[str] = []
    active_tensions: list[str] = []
    drive_competition = None

    proposal_headlines = _bounded_unique(
        [
            _proposal_headline_for_display(goal.goal_statement)
            for goal in dedupe_goal_headlines_by_drive_origin(state.goal_headlines, limit=3)
        ],
        limit=3,
    )
    # attention_items fallback removed 2026-07-30 (chore/delete-orion-drives
    # Wave 2a, found in review): upgrade_autonomy_state_v1_to_v2 was the last
    # producer of AutonomyStateV2.attention_items in the repo (it seeded them
    # from dominant_drive/tension_kinds, both now gone) -- confirmed zero
    # remaining non-test construction sites for AttentionItemV1 repo-wide, so
    # this branch could never fire again. Left as dead code, it would have
    # been exactly the "unreachable but plausible-looking" pattern CLAUDE.md
    # warns about.

    hazards: list[str] = []
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

    active_goals = _active_goals_from_state(state)
    return AutonomySummaryV1(
        stance_hint=stance_hint,
        dominant_drive=None,
        top_drives=top_drives,
        active_tensions=active_tensions,
        proposal_headlines=proposal_headlines,
        response_hazards=_bounded_unique(hazards, limit=hazard_limit),
        raw_state_present=True,
        drive_competition=drive_competition,
        state_quality="healthy",
        stance_mode="normal",
        active_goals=active_goals,
        goals_present=bool(active_goals),
    )
