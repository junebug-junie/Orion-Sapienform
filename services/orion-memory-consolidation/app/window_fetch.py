from __future__ import annotations

from app.boundary import should_close_window
from app.settings import settings
from app.window_state import WindowStore


def window_fetch_should_close_by_phase(turn, scores) -> bool:
    return should_close_window(turn, scores, settings)


async def fetch_window_turns_for_retry(window_store: WindowStore, memory_window_id: str):
    return await window_store.get_window_turns(memory_window_id)
