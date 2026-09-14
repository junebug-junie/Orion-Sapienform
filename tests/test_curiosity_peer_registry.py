from orion.schemas.curiosity_peer import (
    HELP_REQUEST_KIND,
    PEER_BRIEF_CONSUMED_KIND,
    PEER_BRIEF_KIND,
)
from orion.schemas.registry import SCHEMA_REGISTRY, _REGISTRY


def test_curiosity_peer_models_registered_in_both_maps() -> None:
    assert "HelpRequestV1" in _REGISTRY
    assert "PeerBriefV1" in _REGISTRY
    assert "PeerBriefConsumedV1" in _REGISTRY
    assert SCHEMA_REGISTRY["HelpRequestV1"].kind == HELP_REQUEST_KIND
    assert SCHEMA_REGISTRY["PeerBriefV1"].kind == PEER_BRIEF_KIND
    assert SCHEMA_REGISTRY["PeerBriefConsumedV1"].kind == PEER_BRIEF_CONSUMED_KIND
