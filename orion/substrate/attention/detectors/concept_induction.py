from __future__ import annotations

from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import compact, stable_id, unique


class ConceptInductionSignalDetector:
    detector_id = "concept_induction_attention_v1"

    def detect(
        self,
        ctx: dict[str, Any],  # noqa: ARG002
        inputs: dict[str, Any],
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        concept = inputs.get("concept_induction") if isinstance(inputs.get("concept_induction"), dict) else {}
        pairs: list[tuple[str, str]] = []
        for bucket, values in concept.items():
            if not isinstance(values, list):
                continue
            for value in values[:6]:
                pairs.append((str(value), str(bucket)))

        out: list[AttentionSignalV1] = []
        for text in unique([p[0] for p in pairs], limit=10):
            bucket = next((bucket for candidate, bucket in pairs if candidate == text), "concept")
            target = compact(text, 120)
            out.append(
                AttentionSignalV1(
                    signal_id=stable_id("attention-signal", f"{self.detector_id}:{bucket}:{target.lower()}"),
                    source=self.detector_id,
                    target_text=target,
                    target_type_hint="relation" if bucket == "relationship" else "concept",
                    signal_kind=f"concept_{bucket}",
                    salience=0.5 if bucket != "tension" else 0.62,
                    confidence=0.68,
                    evidence_refs=["inputs.concept_induction"],
                    provenance={"detector": self.detector_id, "bucket": bucket, "belief_lineage": list(belief_lineage or [])[:8]},
                )
            )
        return out
