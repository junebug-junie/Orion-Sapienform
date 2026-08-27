from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CORTEX_EXEC_ROOT = ROOT / "services" / "orion-cortex-exec"
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORTEX_EXEC_ROOT) not in sys.path:
    sys.path.insert(0, str(CORTEX_EXEC_ROOT))

from datetime import datetime, timedelta, timezone

from orion.situational.session_turn_phase import bind_session_turn_phase_bus, write_session_turn_state
from orion.situational.identity_ask_cooldown import bind_identity_ask_cooldown_bus
from orion.situational.context import build_situation_for_ctx


class _InMemoryRedis:
    """Standalone-script stand-in for `bus.redis` -- this smoke script has
    no live Redis/bus of its own (it only ever exercised in-process dict
    state before conversation-phase turn timestamps moved to Redis, see
    app/session_turn_phase.py), so it gets its own tiny in-memory store
    rather than requiring real infra just to render sample prompt text.

    `set(..., nx=True, ex=...)` added 2026-08-26 (review finding: this
    smoke bound session_turn_phase's bus but never identity_ask_cooldown's,
    so it never actually exercised the new Redis round-trip that feature
    adds to turn assembly) -- matches real redis-py SET-NX semantics: True
    the first call for a key, None while it still holds a value.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl_seconds: int, payload: str):
        self.store[key] = payload.encode("utf-8")

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        if nx and key in self.store:
            return None
        self.store[key] = value.encode("utf-8")
        return True


class _InMemoryBus:
    def __init__(self) -> None:
        self.redis = _InMemoryRedis()


bind_session_turn_phase_bus(_InMemoryBus())
bind_identity_ask_cooldown_bus(_InMemoryBus())


def _settings():
    return SimpleNamespace(
        orion_situation_enabled=True,
        orion_situation_ttl_seconds=0,
        orion_situation_prompt_max_chars=220,
        orion_situation_timezone="America/Denver",
        orion_situation_location_label="Utah",
        orion_situation_locality="Vernal",
        orion_situation_region="Utah",
        orion_situation_country="US",
        orion_situation_location_precision="city",
        orion_situation_weather_enabled=True,
        orion_situation_weather_provider="none",
        orion_situation_weather_lat=None,
        orion_situation_weather_lon=None,
        orion_situation_weather_ttl_seconds=600,
        orion_situation_umbrella_precip_prob_threshold=40,
        orion_situation_jacket_temp_f_threshold=55,
        orion_situation_high_wind_mph_threshold=25,
        orion_situation_hot_car_temp_f_threshold=80,
        orion_situation_agenda_enabled=False,
        orion_situation_lab_context_enabled=True,
        orion_situation_lab_provider="stub",
        orion_presence_default_requestor="Juniper",
        orion_presence_persist_allowed=False,
    )


async def _print_case(name: str, ctx: dict):
    brief, fragment = await build_situation_for_ctx(ctx, _settings())
    compact = fragment.get("compact_text", "")
    print(f"== {name} ==")
    phase = (brief.get("conversation_phase") or {}).get("phase_change")
    if phase:
        print(f"phase={phase}")
    print(compact)
    assert len(compact) <= 220
    assert brief.get("kind") == "situation.brief.v1"


async def main() -> None:
    await _print_case("default_solo", {"session_id": "smoke-default", "raw_user_text": "hello"})
    await _print_case(
        "child_present",
        {
            "session_id": "smoke-child",
            "raw_user_text": "Can you explain this for my kid?",
            "presence_context": {
                "audience_mode": "kid_present",
                "companions": [{"display_name": "Kid", "relationship": "child", "role": "listener", "age_band": "child"}],
            },
        },
    )
    await _print_case("long_gap_seed", {"session_id": "smoke-gap", "raw_user_text": "first turn"})
    # Conversation-phase turn timestamps live in Redis now (see
    # app/session_turn_phase.py) -- seed the same "10 hours ago" state
    # there instead of poking a since-removed in-process dict.
    await write_session_turn_state(
        "smoke-gap",
        last_user_turn_at=datetime.now(timezone.utc) - timedelta(hours=10),
        last_orion_turn_at=None,
    )
    await _print_case("long_gap_resume", {"session_id": "smoke-gap", "raw_user_text": "yeah do that"})
    await _print_case("heading_out", {"session_id": "smoke-out", "raw_user_text": "I am heading out, should I take a jacket?"})


if __name__ == "__main__":
    asyncio.run(main())
