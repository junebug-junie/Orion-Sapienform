"""Which frames an edge instance detects on.

The edge subscribes to orion:vision:frames, which also carries frames from
other capture services (e.g. the walkway's orion-vision-retina instance). It
must detect only its own camera's frames or it doubles activity events and
GPU work. A pointer with no
stream_id (legacy producers) belongs to the legacy default instance, cam0,
only -- otherwise every instance would detect it.
"""

from __future__ import annotations

from typing import Optional

LEGACY_STREAM_ID = "cam0"


def is_own_frame(pointer_stream_id: Optional[str], own_stream_id: str) -> bool:
    if not pointer_stream_id:
        return own_stream_id == LEGACY_STREAM_ID
    return pointer_stream_id == own_stream_id
