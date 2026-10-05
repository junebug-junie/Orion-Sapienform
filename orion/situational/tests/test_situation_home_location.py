"""2026-10-05 (correlation 5063fb71): Juniper was in a Chicago hotel and said a
camera would soon point "out towards the road". Orion answered with "whatever
Chicago looks like at 2 AM from your hotel window". The unified-turn Situation
block carried a timezone but no place: the Hub adapter hardcoded
`location_label="Unknown"`, `locality=None`. These tests pin the fix: the Hub
settings reach the block, and `home_location`/`physical_location` render as
fixed facts (and stay silent when unconfigured).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from orion.schemas.situation import SituationBriefV1, SurfaceContextV1
from orion.situational import context as situation_mod
from orion.situational.context import (
    _build_prompt_fragment,
    hub_settings_to_runtime_namespace,
    settings_from_runtime,
)

NOW = datetime.now(timezone.utc)
HOME = "Ogden, Utah"
BODY = "Server cabinet in basement, in office, laundry closet"


def _hub(**extra) -> SimpleNamespace:
    base = dict(
        ORION_SITUATION_LOCATION_LABEL="Utah",
        ORION_SITUATION_LOCALITY="Ogden",
        ORION_SITUATION_REGION="Utah",
        ORION_SITUATION_COUNTRY="US",
        ORION_SITUATION_HOME_LOCATION=HOME,
        ORION_SITUATION_PHYSICAL_LOCATION=BODY,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _fragment_text(hub: SimpleNamespace) -> str:
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(hub))
    diag = SimpleNamespace(provider_status={})
    time_ctx = situation_mod._build_time_context(cfg, diag)
    brief = SituationBriefV1(
        generated_at=NOW,
        time=time_ctx,
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, NOW)),
        place=situation_mod._build_place_context(cfg),
        surface=SurfaceContextV1(surface="hub_desktop", input_modality="typed"),
    )
    frag = _build_prompt_fragment(brief, situation_mod._DEFAULT_PROMPT_MAX_CHARS)
    return "\n".join(getattr(frag, "summary_lines", None) or []) or str(frag)


def test_hub_adapter_carries_place_fields_not_unknown() -> None:
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(_hub()))
    assert (cfg.location_label, cfg.locality, cfg.region, cfg.country) == ("Utah", "Ogden", "Utah", "US")
    assert cfg.home_location == HOME
    assert cfg.physical_location == BODY
    place = situation_mod._build_place_context(cfg)
    assert place.source == "configured_home"
    assert place.home_location == HOME and place.physical_location == BODY


def test_unconfigured_hub_still_unknown_and_blank_strings_are_none() -> None:
    cfg = settings_from_runtime(
        hub_settings_to_runtime_namespace(
            SimpleNamespace(ORION_SITUATION_HOME_LOCATION="   ", ORION_SITUATION_PHYSICAL_LOCATION="")
        )
    )
    assert cfg.location_label == "Unknown"
    assert cfg.home_location is None and cfg.physical_location is None


def test_rendered_block_states_home_and_body_as_fixed() -> None:
    text = _fragment_text(_hub())
    assert f"home_location={HOME}" in text
    assert f"physical_location={BODY}" in text
    assert "not where Orion's body" in text


def test_rendered_block_is_silent_when_unconfigured() -> None:
    text = _fragment_text(SimpleNamespace())
    assert "home_location" not in text and "physical_location" not in text
