from __future__ import annotations

from typing import Any, Protocol

from orion.schemas.attention_frame import AttentionSignalV1


class AttentionSignalDetector(Protocol):
    detector_id: str

    def detect(
        self,
        ctx: dict[str, Any],
        inputs: dict[str, Any],
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        ...
