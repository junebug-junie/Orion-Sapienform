from __future__ import annotations

from orion.substrate.appraisal.turn_window import build_turn_window


def test_build_turn_window_includes_assistant_messages() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "be specific"},
    ]
    window = build_turn_window(messages, max_turns=8)
    roles = [m.role for m in window]
    assert roles == ["user", "assistant", "user"]


def test_build_turn_window_caps_at_max_turns() -> None:
    messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
    window = build_turn_window(messages, max_turns=4)
    assert len(window) == 4
    assert window[-1].content == "msg-19"


def test_build_turn_window_skips_empty_and_unknown_roles() -> None:
    messages = [
        {"role": "user", "content": "ok"},
        {"role": "tool", "content": "ignored"},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "visible"},
    ]
    window = build_turn_window(messages, max_turns=8)
    assert len(window) == 2
    assert window[0].role == "user"
    assert window[1].content == "visible"
