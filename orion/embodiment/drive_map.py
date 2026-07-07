from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from orion.core.schemas.drives import DriveStateV1
from orion.embodiment.intents import build_intent
from orion.schemas.embodiment import EmbodimentIntentV1


@dataclass(frozen=True)
class DriveMapThresholds:
    social_high: float = 0.8
    social: float = 0.4
    curiosity: float = 0.4
    idle: float = 0.05
    social_key: str = "social"
    curiosity_key: str = "predictive"


def map_drive_state_to_intent(
    drive: DriveStateV1,
    *,
    correlation_id: str,
    in_conversation: bool = False,
    thresholds: Optional[DriveMapThresholds] = None,
) -> Optional[EmbodimentIntentV1]:
    t = thresholds or DriveMapThresholds()
    pressures = drive.pressures or {}
    if not pressures:
        return None
    social = float(pressures.get(t.social_key, 0.0) or 0.0)
    curio = float(pressures.get(t.curiosity_key, 0.0) or 0.0)
    dominant = max(pressures, key=lambda k: float(pressures.get(k, 0.0) or 0.0))

    if social >= t.social_high and not in_conversation:
        return build_intent(kind="start_conversation", source="involuntary", correlation_id=correlation_id,
                            reason=f"social pressure {social:.2f} -> initiate", urgency=social)
    if dominant == t.social_key and social >= t.social:
        return build_intent(kind="approach_player", source="involuntary", correlation_id=correlation_id,
                            reason=f"social pressure {social:.2f}", urgency=social)
    if dominant == t.curiosity_key and curio >= t.curiosity:
        return build_intent(kind="wander", source="involuntary", correlation_id=correlation_id,
                            reason=f"predictive pressure {curio:.2f}", urgency=curio)
    if all(float(v or 0.0) < t.idle for v in pressures.values()):
        return build_intent(kind="idle", source="involuntary", correlation_id=correlation_id,
                            reason="all drives quiescent")
    return None
