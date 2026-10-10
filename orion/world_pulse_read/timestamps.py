"""Server-side time stamping for reading artifacts.

The Stage 1 handoff's and Stage 2 result's ``created_at`` used to come from
the model's JSON (the prompt even asked for it). Models wrote round, future
times (e.g. 02:30Z the next day), which then became ``observed_at`` on wp-read
concept nodes and ``created_at`` on journal rows. The pipeline's clock at the
moment it received the model output is the only time we stamp.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


def stamp_server_created_at(
    parsed: dict[str, Any],
    *,
    seed_id: str,
    stage: str,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> datetime:
    """Overwrite ``parsed["created_at"]`` with the server clock (UTC).

    A model-written value is discarded, not kept: nothing reads it, and a
    second timestamp field would just invite the wrong one being sorted on.
    It is logged so the drop stays visible.
    """
    stamped = now()
    model_value = parsed.pop("created_at", None)
    if model_value is not None:
        logger.info(
            "world_pulse_read_model_created_at_dropped stage=%s seed=%s model_value=%r server_value=%s",
            stage, seed_id, model_value, stamped.isoformat(),
        )
    parsed["created_at"] = stamped.isoformat()
    return stamped
