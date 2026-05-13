from __future__ import annotations

from orion.autonomy.fanout_policy import autonomy_subject_fanout_from_runtime_ctx


def test_fanout_bounded_plain_chat_quick() -> None:
    assert (
        autonomy_subject_fanout_from_runtime_ctx(
            {"verb": "chat_quick", "mode": "brain", "options": {}}
        )
        == "bounded"
    )


def test_fanout_full_chat_quick_with_hub_full_stance() -> None:
    assert (
        autonomy_subject_fanout_from_runtime_ctx(
            {"verb": "chat_quick", "mode": "brain", "options": {"chat_quick_full_stance": True}}
        )
        == "full"
    )


def test_fanout_full_agent_mode_even_if_verb_chat_quick() -> None:
    assert autonomy_subject_fanout_from_runtime_ctx({"verb": "chat_quick", "mode": "agent"}) == "full"


def test_fanout_full_chat_general() -> None:
    assert autonomy_subject_fanout_from_runtime_ctx({"verb": "chat_general", "mode": "brain"}) == "full"


def test_fanout_full_empty_ctx() -> None:
    assert autonomy_subject_fanout_from_runtime_ctx(None) == "full"
