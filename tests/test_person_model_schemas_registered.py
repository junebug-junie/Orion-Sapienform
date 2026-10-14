from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas import person_model as pm


def test_person_model_schemas_resolve():
    for name in ("PersonRefV1", "PersonMentionV1", "PersonLinkageDecisionV1", "EntityMergeLogEntryV1"):
        assert resolve(name) is getattr(pm, name)
        assert SCHEMA_REGISTRY[name].model is getattr(pm, name)
