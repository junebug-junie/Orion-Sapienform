from __future__ import annotations

from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import compact, stable_id, unique


# Fixed rank priors, not measurements: see the world-first audit note in
# detectors/base.py. A stale thread outranks an ordinary phase change.
SITUATION_SALIENCE_STALE_THREAD = 0.58
SITUATION_SALIENCE_PHASE_CHANGE = 0.42
SITUATION_SALIENCE_PRESENCE = 0.45
SITUATION_SALIENCE_AFFORDANCE = 0.52


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
            raw.append((phase_change, "conversation_phase", SITUATION_SALIENCE_STALE_THREAD if phase_change == "stale_thread" else SITUATION_SALIENCE_PHASE_CHANGE))
        presence = situation.get("presence") if isinstance(situation.get("presence"), dict) else {}
        audience = compact(presence.get("audience_mode"), 80)
        if audience:
            raw.append((audience, "presence", SITUATION_SALIENCE_PRESENCE))
        for affordance in (situation.get("affordances") or [])[:6]:
            if isinstance(affordance, dict):
                label = compact(affordance.get("kind") or affordance.get("suggestion"), 120)
                if label:
                    raw.append((label, "affordance", SITUATION_SALIENCE_AFFORDANCE))

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
                    provenance={
                        "detector": self.detector_id,
                        "source_kind": "external",
                        "belief_lineage": list(belief_lineage or [])[:8],
                    },
                )
            )
        return out
