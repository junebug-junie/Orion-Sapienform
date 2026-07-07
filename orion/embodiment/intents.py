from __future__ import annotations

from typing import Optional

from orion.schemas.embodiment import EmbodimentIntentV1, IntentKind, IntentSource


def build_intent(
    *,
    kind: IntentKind,
    source: IntentSource,
    reason: str,
    correlation_id: str,
    ref: Optional[str] = None,
    urgency: float = 0.0,
    world_id: Optional[str] = None,
    player_id: Optional[str] = None,
) -> EmbodimentIntentV1:
    """Single builder so the non-empty ``reason`` contract is enforced everywhere."""
    return EmbodimentIntentV1(
        kind=kind,
        source=source,
        reason=reason,
        correlation_id=correlation_id,
        ref=ref,
        urgency=urgency,
        world_id=world_id,
        player_id=player_id,
    )
