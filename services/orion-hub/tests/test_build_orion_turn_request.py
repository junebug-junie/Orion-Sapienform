from __future__ import annotations

from orion.hub.turn_request import build_orion_turn_request


def test_build_orion_turn_request_is_thin() -> None:
    req = build_orion_turn_request(
        correlation_id="c-1",
        session_id="sess-1",
        user_message="fix the deploy",
        repair_bundle=None,
    )
    assert req["mode"] == "orion"
    assert "verb_name" not in req
    assert req["user_message"] == "fix the deploy"
