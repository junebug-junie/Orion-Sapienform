"""The stance pass reads the message against the recent conversation.

Live 2026-10-20 (corr=98bde7cf): Juniper wrote "That was on Sunday. Today is
Monday :) It was eventful, thanks for asking!" -- "it" was Lagoon, two days
earlier. The stance pass saw only that message, read "eventful" as Monday and
told the motor to "meet Juniper's Monday update". The motor had the timestamped
history but followed the stance. Now the stance gets the same history block.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from test_turn_orchestrator_utterance_origin import (
    _CORR_ID,
    _harness_run,
    _hub_client_patches,
    _thought,
)

from orion.hub.turn_orchestrator import STANCE_RECENT_TURN_MAX_CHARS, execute_unified_turn

_USER_MESSAGE = "That was on Sunday. Today is Monday :) \n\nIt was eventful, thanks for asking!"
_HISTORY = [
    {
        "role": "user",
        "content": "Big day later today at Lagoon before end of season :) night night",
        "ts": "2026-10-18T07:40:35+00:00",
    },
    {"role": "assistant", "content": "Lagoon before the season closes!", "ts": "2026-10-18T07:40:35+00:00"},
    {"role": "user", "content": "hey hey", "ts": "2026-10-20T04:52:47+00:00"},
    {
        "role": "assistant",
        "content": "Hey hey back. Hope Lagoon was magical today. " + "x" * 1000,
        "ts": "2026-10-20T04:52:47+00:00",
    },
    # Callers include this turn's own message last; it is stripped.
    {"role": "user", "content": _USER_MESSAGE, "ts": "2026-10-20T05:00:39+00:00"},
]


async def _stance_inputs(continuity_messages):
    harness_client_run = AsyncMock(return_value=_harness_run())
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1] as react_mock, patches[2]:
        await execute_unified_turn(
            bus=MagicMock(),
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message=_USER_MESSAGE,
            payload={},
            emit_observation_fn=lambda **_kwargs: None,
            utterance_origin="juniper",
            continuity_messages=continuity_messages,
        )
    return react_mock.await_args.args[0].stance_inputs, harness_client_run.await_args.args[0]


@pytest.mark.asyncio
async def test_stance_gets_timestamped_recent_conversation() -> None:
    stance_inputs, harness_req = await _stance_inputs(_HISTORY)
    lines = stance_inputs["recent_conversation"]
    text = "\n".join(lines)
    assert "Sun Oct 18, 1:40 am" in text and "Lagoon" in text
    assert "about 45 hours pass" in text
    # This turn's own message is not repeated as history.
    assert "It was eventful" not in text
    # Long replies are trimmed for stance; the motor's copy is not.
    assert all(len(line) < STANCE_RECENT_TURN_MAX_CHARS + 80 for line in lines)
    assert any(len(t.content) > 1000 for t in harness_req.recent_turns)


@pytest.mark.asyncio
async def test_no_history_adds_no_recent_conversation() -> None:
    stance_inputs, _ = await _stance_inputs(None)
    assert "recent_conversation" not in stance_inputs
