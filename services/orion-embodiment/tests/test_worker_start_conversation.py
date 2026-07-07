from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.worker import START_CONVERSATION_INPUT, EmbodimentWorker
from orion.embodiment.arbiter import ArbiterState
from orion.schemas.embodiment import EmbodimentIntentV1


def _worker() -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._arbiter = ArbiterState()
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._hold_sec = 8.0
    w._wander_radius = 3.0
    w._locations = {}
    w._social_cooldown_sec = 120.0
    w._move_cooldown_sec = 0.0
    w._last_conversation_start = None
    w._last_move_at = None
    return w


_T0 = datetime(2026, 7, 7, 0, 0, 0, tzinfo=timezone.utc)
_PLAYERS = [
    {"id": "orion", "name": "Orion", "position": {"x": 0.0, "y": 0.0}},
    {"id": "p9", "name": "Juniper", "position": {"x": 3.0, "y": 0.0}},
]


def test_start_conversation_actuates_and_sets_cooldown():
    w = _worker()
    intent = EmbodimentIntentV1(kind="start_conversation", source="deliberate", ref="Juniper",
                                reason="say hi", correlation_id="c1", player_id="orion")
    with patch("app.worker.aitown_client.list_players", return_value=_PLAYERS), \
         patch("app.worker.aitown_client.move_to", return_value={"ok": True}) as mv, \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}) as si:
        outcome = w.process_intent(intent, now=_T0)
    mv.assert_called_once()
    si.assert_called_once()
    assert si.call_args.kwargs["name"] == START_CONVERSATION_INPUT
    assert si.call_args.kwargs["args"]["invitee"] == "p9"
    assert outcome.status == "actuated"
    assert w._last_conversation_start == _T0


def test_second_start_conversation_within_cooldown_denied():
    w = _worker()
    w._last_conversation_start = _T0
    intent = EmbodimentIntentV1(kind="start_conversation", source="deliberate", ref="Juniper",
                                reason="say hi again", correlation_id="c2", player_id="orion")
    with patch("app.worker.aitown_client.list_players", return_value=_PLAYERS), \
         patch("app.worker.aitown_client.move_to") as mv, \
         patch("app.worker.aitown_client.send_input") as si:
        outcome = w.process_intent(intent, now=_T0 + timedelta(seconds=30))
    si.assert_not_called()
    assert outcome.status == "denied"
    assert "cooldown" in outcome.reason.lower()


def test_start_conversation_no_target_denied():
    w = _worker()
    intent = EmbodimentIntentV1(kind="start_conversation", source="deliberate", ref="Nobody",
                                reason="lonely", correlation_id="c3", player_id="orion")
    with patch("app.worker.aitown_client.list_players", return_value=_PLAYERS), \
         patch("app.worker.aitown_client.move_to") as mv, \
         patch("app.worker.aitown_client.send_input") as si:
        outcome = w.process_intent(intent, now=_T0)
    si.assert_not_called()
    assert outcome.status == "denied"
