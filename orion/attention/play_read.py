"""The one read behind ``world:play`` (game company patch 4), shared by both attention contests.

``orion.attention.world_first`` holds the candidate's rules and SQL and does no I/O. This module runs the
three statements on a SQLAlchemy connection, so the substrate broadcast (orion-substrate-runtime) and the
field contest (orion-attention-runtime) read play the same way:

1. ``PLAY_SITTINGS_SQL``: the sittings that can speak into the 7-day grid. A failure here makes the
   candidate ABSENT ("retro_sitting unreadable"), never "nobody plays".
2. For the open sitting only: its watch preferences and its newest stored look note. A failure here leaves
   the watch unknown, and ``play_watch_reason`` holds the candidate out (fail closed).

Each statement runs in its own SAVEPOINT-free connection call; a missing table (a database before the game
company migrations) reads as a failed read, with its reason.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from orion.attention.world_first import (
    PLAY_NEWEST_NOTE_SQL,
    PLAY_PREFS_SQL,
    PLAY_RATE_WINDOW,
    PLAY_SITTINGS_SQL,
    PlaySitting,
    PlayWatch,
    open_play_sitting,
    play_watch_from_rows,
)
from orion.substrate.prediction_error_magnitude import WINDOW_7D

logger = logging.getLogger(__name__)


def read_play_inputs(engine: Any, *, now: datetime) -> tuple[Optional[list[PlaySitting]], Optional[PlayWatch], Optional[str]]:
    """(sittings or None, the open sitting's watch or None, absent reason or None). Never raises."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(PLAY_SITTINGS_SQL), {"since": now - WINDOW_7D - PLAY_RATE_WINDOW}).mappings().all()
    except Exception as exc:  # noqa: BLE001 -- absent, not calm
        logger.warning("world_play_sittings_read_failed err=%s", type(exc).__name__)
        return None, None, f"retro_sitting unreadable ({type(exc).__name__})"
    sittings = [
        PlaySitting(
            sitting_id=str(r["sitting_id"]), started_at=r["started_at"], last_input_at=r["last_input_at"],
            ended_at=r.get("ended_at"), game_title=r.get("game_title"), game_id=r.get("game_id"),
            player_stated=r.get("player_stated"), backfilled=bool(r.get("backfilled")),
        )
        for r in rows
        if r.get("started_at") is not None and r.get("last_input_at") is not None
    ]
    sitting = open_play_sitting(sittings, now=now)
    if sitting is None:
        return sittings, None, None
    try:
        with engine.connect() as conn:
            prefs = conn.execute(text(PLAY_PREFS_SQL), {"sitting_id": sitting.sitting_id,
                                                        "game_id": str(sitting.game_id or "")}).mappings().all()
            note = conn.execute(text(PLAY_NEWEST_NOTE_SQL), {"sitting_id": sitting.sitting_id}).first()
    except Exception as exc:  # noqa: BLE001 -- watch unknown: held out, never "watching"
        logger.warning("world_play_watch_read_failed err=%s", type(exc).__name__)
        return sittings, None, None
    newest = (note[0], str(note[1] or "")) if note is not None and note[0] is not None else None
    return sittings, play_watch_from_rows(sitting, [dict(p) for p in prefs], newest), None


__all__ = ["read_play_inputs"]
