from __future__ import annotations

from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import compact, stable_id, unique


class SituationSignalDetector:
    detector_id = "situation_attention_v1"

    def detect(
        self,
        ctx: dict[str, Any],  # noqa: ARG002
        inputs: dict[str, Any],
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        situation = inputs.get("situation") if isinstance(inputs.get("situation"), dict) else {}
        if not situation:
            return []
        raw: list[tuple[str, str, float]] = []
        phase = situation.get("conversation_phase") if isinstance(situation.get("conversation_phase"), dict) else {}
        phase_change = compact(phase.get("phase_change"), 80)
        if phase_change:
            raw.append((phase_change, "conversation_phase", 0.58 if phase_change == "stale_thread" else 0.42))
        presence = situation.get("presence") if isinstance(situation.get("presence"), dict) else {}
        audience = compact(presence.get("audience_mode"), 80)
        if audience:
            raw.append((audience, "presence", 0.45))
        for affordance in (situation.get("affordances") or [])[:6]:
            if isinstance(affordance, dict):
                label = compact(affordance.get("kind") or affordance.get("suggestion"), 120)
                if label:
                    raw.append((label, "affordance", 0.52))

        out: list[AttentionSignalV1] = []
        for text in unique([item[0] for item in raw], limit=8):
            kind = next((kind for candidate, kind, _salience in raw if candidate == text), "situation")
            salience = next((_salience for candidate, _kind, _salience in raw if candidate == text), 0.45)
            out.append(
                AttentionSignalV1(
                    signal_id=stable_id("attention-signal", f"{self.detector_id}:{text.lower()}"),
                    source=self.detector_id,
                    target_text=text,
                    target_type_hint="other",
                    signal_kind=f"situation_{kind}",
                    salience=salience,
                    confidence=0.66,
                    evidence_refs=["inputs.situation"],
                    provenance={"detector": self.detector_id, "belief_lineage": list(belief_lineage or [])[:8]},
                )
            )
        return out
