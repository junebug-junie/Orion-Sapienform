from __future__ import annotations

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

from datetime import UTC, datetime, timedelta

import app.situation as situation_module
from app.situation import build_situation_for_ctx


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


def _print_case(name: str, ctx: dict):
    brief, fragment = build_situation_for_ctx(ctx, _settings())
    compact = fragment.get("compact_text", "")
    print(f"== {name} ==")
    phase = (brief.get("conversation_phase") or {}).get("phase_change")
    if phase:
        print(f"phase={phase}")
    print(compact)
    assert len(compact) <= 220
    assert brief.get("kind") == "situation.brief.v1"


def main() -> None:
    _print_case("default_solo", {"session_id": "smoke-default", "raw_user_text": "hello"})
    _print_case(
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
    _print_case("long_gap_seed", {"session_id": "smoke-gap", "raw_user_text": "first turn"})
    situation_module._SESSION_LAST_USER_TURN["smoke-gap"] = datetime.now(UTC) - timedelta(hours=10)
    _print_case("long_gap_resume", {"session_id": "smoke-gap", "raw_user_text": "yeah do that"})
    _print_case("heading_out", {"session_id": "smoke-out", "raw_user_text": "I am heading out, should I take a jacket?"})


if __name__ == "__main__":
    main()
