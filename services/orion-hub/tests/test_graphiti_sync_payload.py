from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.schemas import CrystallizationGovernanceV1, CrystallizationLinkV1, MemoryCrystallizationV1


def _crys_with_link():
    now = datetime.now(timezone.utc)
    return MemoryCrystallizationV1(
        crystallization_id="crys_a",
        kind="stance",
        subject="A",
        summary="summary",
        status="active",
        governance=CrystallizationGovernanceV1(proposed_by="test", approved_by="test"),
        created_at=now,
        updated_at=now,
        links=[
            CrystallizationLinkV1(target_crystallization_id="crys_b", relation="supports", confidence=0.8)
        ],
    )


def test_sync_payload_includes_links():
    crys = _crys_with_link()
    adapter = GraphitiAdapter(enabled=True, url="http://graphiti")
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"episode_id": "gep_crys_a", "entity_id": "gent_crys_a", "edge_id": "ged_x", "canonical_mutated": False}
        return resp

    with patch("httpx.Client.post", side_effect=fake_post):
        adapter.sync_crystallization(crys)

    assert captured["json"]["links"] == [
        {"target_crystallization_id": "crys_b", "relation": "supports", "confidence": 0.8}
    ]
    assert captured["json"]["metadata"]["sensitivity"] == "private"
