"""PersonalityChoiceV1 (PR #2726 P0) is in both schema registries; the nested ChoiceOptionV1
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
