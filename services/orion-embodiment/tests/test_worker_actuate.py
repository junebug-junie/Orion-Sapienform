from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.worker import EmbodimentWorker
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
    w._walkable = None
    w._walkable_loaded = True
    return w


def test_go_to_location_actuates():
    w = _worker()
    w._locations = {"fountain": {"x": 5.0, "y": 5.0}}
    intent = EmbodimentIntentV1(kind="go_to_location", source="deliberate", ref="fountain",
                                reason="r", correlation_id="c", player_id="orion")
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.aitown_client.move_to", return_value={"ok": True}) as mv:
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    mv.assert_called_once_with(player_id="orion", x=5.0, y=5.0, world_id="w1")
    assert outcome.status == "actuated"
    assert outcome.send_input_ok is True
    assert outcome.resolved_destination == {"x": 5.0, "y": 5.0}


def test_involuntary_preempted_during_hold_emits_no_move():
    w = _worker()
    w._arbiter.deliberate_hold_until = datetime(2026, 7, 7, 0, 0, 30, tzinfo=timezone.utc)
    intent = EmbodimentIntentV1(kind="wander", source="involuntary", reason="r", correlation_id="c")
    with patch("app.worker.aitown_client.move_to") as mv:
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, 0, 0, 25, tzinfo=timezone.utc))
    mv.assert_not_called()
    assert outcome.status == "preempted"


def test_missing_player_id_denied():
    w = _worker()
    w._orion_player_id = ""
    intent = EmbodimentIntentV1(kind="wander", source="deliberate", reason="r", correlation_id="c")
    with patch("app.worker.aitown_client.move_to") as mv:
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    mv.assert_not_called()
    assert outcome.status == "denied"


def test_convex_error_becomes_error_outcome():
    w = _worker()
    intent = EmbodimentIntentV1(kind="wander", source="deliberate", reason="r", correlation_id="c",
                                player_id="orion")
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    from orion.embodiment.aitown_client import AitownClientError
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.aitown_client.move_to", side_effect=AitownClientError("down")):
        outcome = w.process_intent(intent, now=datetime(2026, 7, 7, tzinfo=timezone.utc))
    assert outcome.status == "error"
    assert outcome.send_input_ok is False


def test_move_cooldown_debounces_rapid_moves():
    w = _worker()
    w._move_cooldown_sec = 6.0
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    intent = EmbodimentIntentV1(kind="wander", source="involuntary", reason="r", correlation_id="c",
                                player_id="orion")
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.aitown_client.move_to") as mv:
        first = w.process_intent(intent, now=datetime(2026, 7, 7, 0, 0, 0, tzinfo=timezone.utc))
        # A second move 2s later is within the 6s cooldown -> no actuation.
        second = w.process_intent(intent, now=datetime(2026, 7, 7, 0, 0, 2, tzinfo=timezone.utc))
        # A third move past the cooldown actuates again.
        third = w.process_intent(intent, now=datetime(2026, 7, 7, 0, 0, 10, tzinfo=timezone.utc))
    assert first.status == "actuated"
    assert second.status == "resolved_noop"
    assert "move cooldown" in (second.reason or "")
    assert third.status == "actuated"
    assert mv.call_count == 2
