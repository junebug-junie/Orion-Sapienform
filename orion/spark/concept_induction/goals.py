from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.schemas.drives import DriveStateV1, GoalProposalV1, TensionEventV1
from .dossier import build_evidence_items, build_source_event_ref, extract_trace_id, extract_turn_id


GOAL_TEMPLATES = {
    "coherence": "Stabilize internal coherence around the active evidence trail.",
    "continuity": "Preserve continuity across recent identity and relation shifts.",
    "capability": "Recover capability by reducing current cognitive load.",
    "relational": "Repair relational trust using the strongest current evidence.",
    "predictive": "Reduce predictive uncertainty by grounding on recent evidence.",
    "autonomy": "Clarify autonomy boundaries without executing any new action.",
}


@dataclass(frozen=True)
class GoalDecision:
    proposal: Optional[GoalProposalV1]
    suppressed_signature: Optional[str] = None


class GoalProposalEngine:
    def __init__(self, cooldown_minutes: int) -> None:
        self.cooldown = timedelta(minutes=max(0, cooldown_minutes))

    @staticmethod
    def _drive_origin(drive_state: DriveStateV1) -> str:
        if not drive_state.pressures:
            return "continuity"
        return max(sorted(drive_state.pressures), key=lambda key: drive_state.pressures.get(key, 0.0))

    @staticmethod
    def _priority(drive_state: DriveStateV1, drive_origin: str, tensions: List[TensionEventV1]) -> float:
        tension_weight = max((tension.magnitude for tension in tensions), default=0.0)
        return max(0.0, min(1.0, round((drive_state.pressures.get(drive_origin, 0.0) * 0.7) + (tension_weight * 0.3), 4)))

    def _goal_statement(self, drive_state: DriveStateV1, drive_origin: str, tensions: List[TensionEventV1]) -> str:
        base = GOAL_TEMPLATES.get(drive_origin, GOAL_TEMPLATES["continuity"])
        if tensions:
            lead = sorted(tensions, key=lambda tension: (-tension.magnitude, tension.kind))[0]
            text = f"{base} Primary tension: {lead.kind}."
        else:
            text = base
        # Lineage only: do not append evidence_summary (often user chat text from the intake envelope).
        extras: List[str] = []
        tr = str(drive_state.trace_id or "").strip()
        if tr:
            extras.append(f"trace={tr[:8]}" + ("…" if len(tr) > 8 else ""))
        if extras:
            text = f"{text} · {' · '.join(extras)}"
        return text

    @staticmethod
    def _signature(subject: str, model_layer: str, drive_origin: str, goal_statement: str, tensions: List[TensionEventV1]) -> str:
        tension_signature = ",".join(sorted(tension.kind for tension in tensions[:3]))
        material = "|".join([
            subject,
            model_layer,
            drive_origin,
            tension_signature,
            " ".join(goal_statement.lower().split()),
        ])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def propose(
        self,
        *,
        env: BaseEnvelope,
        intake_channel: str,
        drive_state: DriveStateV1,
        tensions: Iterable[TensionEventV1],
        store,
    ) -> GoalDecision:
        tension_list = sorted(list(tensions), key=lambda tension: (-tension.magnitude, tension.kind))
        drive_origin = self._drive_origin(drive_state)
        goal_statement = self._goal_statement(drive_state, drive_origin, tension_list)
        signature = self._signature(drive_state.subject, drive_state.model_layer, drive_origin, goal_statement, tension_list)
        now = drive_state.updated_at if drive_state.updated_at.tzinfo else drive_state.updated_at.replace(tzinfo=timezone.utc)

        cooldown_record = store.load_goal_cooldown(signature)
        if cooldown_record:
            cooldown_until_raw = cooldown_record.get("cooldown_until")
            if isinstance(cooldown_until_raw, str):
                try:
                    cooldown_until = datetime.fromisoformat(cooldown_until_raw)
                except ValueError:
                    cooldown_until = None
                if cooldown_until and cooldown_until > now:
                    store.record_goal_suppression(signature, now)
                    return GoalDecision(proposal=None, suppressed_signature=signature)

        source_event_ref = build_source_event_ref(env, intake_channel)
        evidence_items = build_evidence_items(env, intake_channel, drive_state.provenance.evidence_text)
        proposal = GoalProposalV1(
            artifact_id=f"goal-{signature}",
            subject=drive_state.subject,
            model_layer=drive_state.model_layer,
            entity_id=drive_state.entity_id,
            kind="memory.goals.proposed.v1",
            ts=now,
            confidence=drive_state.confidence,
            correlation_id=drive_state.correlation_id,
            trace_id=drive_state.trace_id or extract_trace_id(env),
            turn_id=drive_state.turn_id or extract_turn_id(env),
            provenance=drive_state.provenance.model_copy(update={
                "source_event_refs": [source_event_ref],
                "evidence_items": evidence_items,
                "tension_refs": [tension.artifact_id for tension in tension_list],
                "evidence_summary": evidence_items[0].summary if evidence_items else drive_state.provenance.evidence_summary,
            }),
            related_nodes=drive_state.related_nodes + [tension.artifact_id for tension in tension_list],
            goal_statement=goal_statement,
            proposal_signature=signature,
            drive_origin=drive_origin,
            priority=self._priority(drive_state, drive_origin, tension_list),
            cooldown_until=now + self.cooldown if self.cooldown else now,
            source_event_refs=[source_event_ref],
            evidence_items=evidence_items,
            tension_kinds=[tension.kind for tension in tension_list],
        )
        store.save_goal_cooldown(signature, proposal.cooldown_until or now)
        return GoalDecision(proposal=proposal)
