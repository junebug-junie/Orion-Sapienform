from __future__ import annotations

import importlib
import asyncio
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

HUB_ROOT = Path(__file__).resolve().parents[1]
if str(HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(HUB_ROOT))

hub_api_routes = importlib.import_module('scripts.api_routes')


class _FakeBus:
    def __init__(self) -> None:
        self.enabled = True
        self.published = []

    async def publish(self, channel, env):
        self.published.append((channel, env))


def test_api_chat_response_feedback_rejects_invalid_payload(monkeypatch) -> None:
    fake_bus = _FakeBus()
    monkeypatch.setitem(sys.modules, "scripts.main", types.SimpleNamespace(bus=fake_bus))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(hub_api_routes.api_chat_response_feedback(
            {
                'feedback_id': 'fb-invalid',
                'feedback_value': 'up',
                'categories': ['made_up_facts'],
            }
        ))
    assert exc.value.status_code == 422
    assert fake_bus.published == []


def test_api_chat_response_feedback_publishes_valid_payload(monkeypatch) -> None:
    fake_bus = _FakeBus()
    monkeypatch.setitem(sys.modules, "scripts.main", types.SimpleNamespace(bus=fake_bus))
    chat_history_mod = importlib.import_module("scripts.chat_history")
    monkeypatch.setattr(
        chat_history_mod,
        "settings",
        types.SimpleNamespace(
            SERVICE_NAME="hub",
            NODE_NAME="athena",
            SERVICE_VERSION="0.3.0",
            PUBLISH_CHAT_HISTORY_LOG=True,
        ),
    )

    resp = asyncio.run(hub_api_routes.api_chat_response_feedback(
        {
            'feedback_id': 'fb-valid',
            'target_turn_id': 'turn-1',
            'target_message_id': 'turn-1:assistant',
            'target_correlation_id': 'turn-1',
            'session_id': 'sid-1',
            'feedback_value': 'up',
            'categories': ['helpful_actionable'],
            'free_text': '  useful  ',
        }
    ))
    assert resp['ok'] is True
    assert resp['feedback_id'] == 'fb-valid'
    assert len(fake_bus.published) == 1
    channel, env = fake_bus.published[0]
    assert channel == 'orion:chat:response:feedback'
    assert env.kind == 'chat.response.feedback.v1'
    assert env.payload.free_text == 'useful'


def test_feedback_downvote_emits_pressure_event_telemetry(monkeypatch) -> None:
    fake_bus = _FakeBus()
    monkeypatch.setitem(sys.modules, "scripts.main", types.SimpleNamespace(bus=fake_bus))
    recorded = []
    monkeypatch.setattr(hub_api_routes.SUBSTRATE_REVIEW_TELEMETRY_STORE, "record", lambda entry: recorded.append(entry))

    resp = asyncio.run(hub_api_routes.api_chat_response_feedback(
        {
            'feedback_id': 'fb-pressure',
            'target_turn_id': 'turn-2',
            'target_message_id': 'turn-2:assistant',
            'target_correlation_id': 'corr-2',
            'session_id': 'sid-2',
            'feedback_value': 'down',
            'categories': ['wrong_tool_wrong_routing_wrong_mode', 'missed_relevant_context'],
            'free_text': 'route felt wrong',
        }
    ))
    assert resp['ok'] is True
    assert recorded, "expected pressure telemetry to be recorded"
    categories = {item.pressure_category for item in recorded[-1].pressure_events}
    assert "recall_miss_or_dissatisfaction" in categories
    assert "routing_false_downgrade" in categories
    assert "routing_false_escalation" in categories


def test_api_chat_response_feedback_options_reflect_canonical_contract() -> None:
    payload = hub_api_routes.api_chat_response_feedback_options()
    assert payload['feedback_values'] == ['up', 'down']
    assert any(item['value'] == 'should_have_probed_more_about_stated_topics' for item in payload['categories']['down'])


def test_artifact_rating_emits_no_chat_pressure_events() -> None:
    """An artifact rating is not chat-lane evidence.

    Verified live in review that before this gate, an artifact-targeted
    thumbs-down carrying missed_relevant_context +
    wrong_tool_wrong_routing_wrong_mode emitted 3 MutationPressureEvidenceV1
    events with correlation_id=None, filed under
    invocation_surface="chat_reflective_lane".

    Two things wrong with that. A rating of a journal entry is not a recall
    miss in a conversation that never happened. And routing it here feeds the
    human verdict back into the same self-graded mutation-pressure machinery
    that the artifact-rating path exists to sit OUTSIDE of -- counting one
    human opinion twice, through two mechanisms, with different provenance.
    """
    sys.path.insert(0, str(HUB_ROOT))
    api_routes = importlib.import_module("scripts.api_routes")
    from orion.schemas.chat_response_feedback import (
        ChatResponseFeedbackV1,
        build_artifact_ref,
    )

    dispatch_id = (
        "dispatch:proposal:prune_stopped_containers:tick_fc7585176059:none:"
        "execution_dispatch_policy.v1"
    )
    pressure_categories = ["missed_relevant_context", "wrong_tool_wrong_routing_wrong_mode"]

    # Same verdict, same categories, differing only in what is being rated.
    chat = ChatResponseFeedbackV1(
        feedback_id="chat-1",
        target_turn_id="turn-1",
        target_correlation_id="corr-1",
        feedback_value="down",
        categories=pressure_categories,
    )
    artifact = ChatResponseFeedbackV1(
        feedback_id="artifact-1",
        target_artifact_ref=build_artifact_ref("journal", dispatch_id),
        user_id="juniper",
        feedback_value="down",
        categories=pressure_categories,
    )

    # The chat one must still produce evidence -- this gate must not have
    # silently disabled the existing behaviour.
    assert api_routes._producer_pressure_events_from_feedback(chat)
    assert api_routes._producer_pressure_events_from_feedback(artifact) == []
