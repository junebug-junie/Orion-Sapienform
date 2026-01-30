from __future__ import annotations

import logging
from typing import Optional

from app.settings import Settings

logger = logging.getLogger("topic-rail.representation")


def build_representation(settings: Settings) -> Optional[object]:
    if not settings.topic_rail_use_keybert:
        return None
    try:
        from bertopic.representation import KeyBERTInspired
    except Exception as exc:  # noqa: BLE001
        logger.warning("KeyBERT not available: %s", exc)
        return None
    return KeyBERTInspired()
