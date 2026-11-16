"""PersonalityChoiceV1 (personality rev 4) is in both schema registries and on its bus channel; the nested ChoiceOptionV1
is in `_REGISTRY` only, per registry convention for nested payload models.

Lives here, not in orion/personality/tests: importing the registry pulls every schema module,
and the Personality workflow installs only pytest + pydantic.
"""

from orion.schemas.personality import ChoiceOptionV1, PersonalityChoiceV1
from orion.schemas.registry import SCHEMA_REGISTRY, resolve


def test_personality_schemas_resolve():
    assert resolve("PersonalityChoiceV1") is PersonalityChoiceV1
    assert resolve("ChoiceOptionV1") is ChoiceOptionV1
    assert SCHEMA_REGISTRY["PersonalityChoiceV1"].model is PersonalityChoiceV1
    assert "ChoiceOptionV1" not in SCHEMA_REGISTRY


def test_personality_frame_contracts_resolve():
    from orion.schemas.personality import ChoiceStyleV1, DispositionV1, PersonalityCueV1, PersonalityFrameV1

    assert resolve("PersonalityFrameV1") is PersonalityFrameV1
    assert SCHEMA_REGISTRY["PersonalityFrameV1"].model is PersonalityFrameV1
    for name, model in (("DispositionV1", DispositionV1), ("ChoiceStyleV1", ChoiceStyleV1), ("PersonalityCueV1", PersonalityCueV1)):
        assert resolve(name) is model
        assert name not in SCHEMA_REGISTRY


def test_choice_channel_is_catalogued():
    """Rev 4: producers publish PersonalityChoiceV1 on orion:personality:choice; durable-runs consumes."""
    from pathlib import Path

    import yaml

    from orion.schemas.personality import PERSONALITY_CHOICE_CHANNEL, PERSONALITY_CHOICE_KIND

    doc = yaml.safe_load((Path(__file__).resolve().parents[1] / "orion/bus/channels.yaml").read_text())
    entries = doc["channels"] if isinstance(doc, dict) else doc
    (entry,) = [c for c in entries if c["name"] == PERSONALITY_CHOICE_CHANNEL]
    assert entry["schema_id"] == "PersonalityChoiceV1" and entry["message_kind"] == PERSONALITY_CHOICE_KIND
    assert entry["consumer_services"] == ["orion-durable-runs"]


def test_choice_option_without_detail_dumps_like_the_pre_detail_model():
    """ChoiceOptionV1.detail is additive on an extra='forbid' model: an option without it must dump without
    the key, so a durable-runs built before the field still validates every producer's choices."""
    from datetime import datetime, timezone

    from pydantic import BaseModel, ConfigDict, Field

    from orion.schemas.personality import ChoiceOptionV1, PersonalityChoiceV1

    class OldOption(BaseModel):  # frozen copy of the model before 2026-11-16
        model_config = ConfigDict(extra="forbid")
        key: str = Field(min_length=1, max_length=200)
        text: str = Field(min_length=1, max_length=400)
        market_rank: int | None = Field(default=None, ge=1)

    c = PersonalityChoiceV1(choice_id="reading:x", lane="reading", occurred_at=datetime(2026, 11, 16, tzinfo=timezone.utc),
                            chosen=ChoiceOptionV1(key="a", text="A", market_rank=1),
                            alternatives=[ChoiceOptionV1(key="b", text="B", market_rank=2)], chosen_market_rank=1)
    dumped = c.model_dump(mode="json")
    for o in [dumped["chosen"], *dumped["alternatives"]]:
        OldOption.model_validate(o)
    assert "detail" in ChoiceOptionV1(key="a", text="A", detail="the words").model_dump()
