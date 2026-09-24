"""Which frames an edge instance detects on.

Every orion-vision-edge instance (cam0, walkway) subscribes to the same
orion:vision:frames channel. Each must detect only its own camera's frames or
a second instance doubles activity events and GPU work. A pointer with no
stream_id is kept (legacy producers) -- it cannot be attributed elsewhere.
"""

from __future__ import annotations

from typing import Optional


def is_own_frame(pointer_stream_id: Optional[str], own_stream_id: str) -> bool:
    return not pointer_stream_id or pointer_stream_id == own_stream_id
