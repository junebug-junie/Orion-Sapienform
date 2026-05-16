from __future__ import annotations

from typing import Any

from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.common import compact
from orion.substrate.attention.detectors.legacy_regex import LegacyRegexSignalDetector


class CurrentTurnSignalDetector:
    """Current user-turn detector; delegates phrase fallback to legacy v1 extraction."""

    detector_id = "current_turn_v1"

    def __init__(self, *, fallback: LegacyRegexSignalDetector | None = None) -> None:
        self._fallback = fallback or LegacyRegexSignalDetector()

    def detect(
        self,
        ctx: dict[str, Any],
        inputs: dict[str, Any],
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        user_text = compact(ctx.get("user_message") or ctx.get("raw_user_text") or "", 600)
        if not user_text:
            return []
        signals = self._fallback.detect(ctx, inputs, belief_lineage)
        if signals:
            return signals
        return []
