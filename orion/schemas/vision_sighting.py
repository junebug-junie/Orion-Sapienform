"""A confident face match on a home camera, published once per sitting (2026-10-09).

The room camera matches Juniper's face routinely (live 10-09 01:56-02:00: 6 probable, 14
possible), but the match only lived in the camera's presence row while she was in view and was
rewritten to "unknown" when she stepped away. Nothing durable recorded "Juniper was seen at home".

Producer: orion-vision-window, only for streams it is configured to treat as home cameras and
on one "probable" match, or two "possible"-or-better matches within 10 min (never the laptop
webcam, which travels with her).
Consumer: the situation.update graph in orion-durable-runs, where it is positive evidence for
Juniper's whereabouts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

IDENTITY_SIGHTING_KIND = "vision.identity.sighting.v1"
IDENTITY_SIGHTING_CHANNEL = "orion:vision:identity:sighting"


class IdentitySightingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["vision.identity.sighting.v1"] = "vision.identity.sighting.v1"
    subject: str                      # the enrolled subject (one-subject gallery: "juniper")
    stream_id: str                    # camera stream, e.g. "cam0"
    place: Literal["home"] = "home"   # the producer only emits for configured home cameras
    seen_at: datetime
    # Which rule qualified: one "probable" match, or two "possible"-or-better matches within the
    # window ("corroborated"). Loosened 2026-10-10 from probable x2 (a wave at the camera got one
    # check and never qualified).
    outcome: Literal["probable", "corroborated"] = "probable"
    similarity: float = Field(..., ge=-1.0, le=1.0)
    correlation_id: str               # the identity artifact that matched
