from pydantic import ValidationError
import pytest

from orion.schemas.registry import _REGISTRY
from orion.schemas.social_chat import TownContinuityReadV1, TownContinuityTurnV1


def test_town_continuity_schemas_are_registered() -> None:
    assert _REGISTRY["TownContinuityTurnV1"] is TownContinuityTurnV1
    assert _REGISTRY["TownContinuityReadV1"] is TownContinuityReadV1


def test_empty_lists_are_valid() -> None:
    body = TownContinuityReadV1(
        thread_id="juniper-feld--nico-sable",
        speaker_id="nico-sable",
        pair_turns=[],
        town_turns=[],
    )
    assert body.pair_turns == []
    assert body.town_turns == []


def test_turn_roundtrip() -> None:
    turn = TownContinuityTurnV1(
        speaker="Nico Sable",
        other="Juniper Feld",
        text="the pie sat out",
        thread_id="juniper-feld--nico-sable",
        created_at="2026-08-29T22:00:00+00:00",
    )
    again = TownContinuityTurnV1.model_validate(turn.model_dump())
    assert again.text == "the pie sat out"


def test_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TownContinuityTurnV1(
            speaker="Nico Sable",
            other="Juniper Feld",
            text="hi",
            thread_id="juniper-feld--nico-sable",
            created_at="2026-08-29T22:00:00+00:00",
            client_meta={"nope": True},
        )
