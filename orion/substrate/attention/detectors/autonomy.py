from __future__ import annotations

from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import compact, stable_id, unique


class AutonomySignalDetector:
    detector_id = "autonomy_attention_v1"

    def detect(
        self,
        ctx: dict[str, Any],  # noqa: ARG002
        inputs: dict[str, Any],
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        autonomy = inputs.get("autonomy") if isinstance(inputs.get("autonomy"), dict) else {}
        summary = autonomy.get("summary") if isinstance(autonomy.get("summary"), dict) else {}
        state_v2 = autonomy.get("state_v2") if isinstance(autonomy.get("state_v2"), dict) else {}
        raw: list[tuple[str, str, float]] = []
        for item in (state_v2.get("attention_items") or [])[:5]:
            if isinstance(item, dict) and item.get("summary"):
                raw.append((str(item.get("summary")), "autonomy_attention_item", float(item.get("salience") or 0.72)))
        for drive in (summary.get("top_drives") or [])[:3]:
            raw.append((str(drive), "autonomy_drive", 0.58))
        for tension in (summary.get("active_tensions") or [])[:4]:
            raw.append((str(tension), "autonomy_tension", 0.68))

        out: list[AttentionSignalV1] = []
        for text in unique([item[0] for item in raw], limit=8):
            kind = next((kind for candidate, kind, _salience in raw if candidate == text), "autonomy_signal")
            salience = next((_salience for candidate, _kind, _salience in raw if candidate == text), 0.55)
            target = compact(text, 120)
            out.append(
                AttentionSignalV1(
                    signal_id=stable_id("attention-signal", f"{self.detector_id}:{target.lower()}"),
                    source=self.detector_id,
                    target_text=target,
                    target_type_hint="concept",
                    signal_kind=kind,
                    salience=salience,
                    confidence=0.72,
                    evidence_refs=["inputs.autonomy"],
                    provenance={"detector": self.detector_id, "belief_lineage": list(belief_lineage or [])[:8]},
                )
            )
        return out
