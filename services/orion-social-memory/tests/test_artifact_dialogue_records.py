from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.synthesizer import artifact_dialogue_records
from orion.schemas.social_chat import SocialRoomTurnStoredV1


def test_artifact_dialogue_records_ignores_empty_placeholder_dicts() -> None:
    turn = SocialRoomTurnStoredV1.model_validate(
        {
            "prompt": "k, we're back",
            "response": "Back and ready.",
            "client_meta": {
                "external_room": {"platform": "hub", "room_id": "hub-direct"},
                "external_participant": {"participant_id": "juniper"},
                "social_artifact_proposal": {},
                "social_artifact_revision": {},
                "social_artifact_confirmation": {},
            },
        }
    )

    proposal, revision, confirmation = artifact_dialogue_records(turn, scope="peer_local")

    assert proposal is None
    assert revision is None
    assert confirmation is None
