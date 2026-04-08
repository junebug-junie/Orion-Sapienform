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


def test_api_chat_response_feedback_options_reflect_canonical_contract() -> None:
    payload = hub_api_routes.api_chat_response_feedback_options()
    assert payload['feedback_values'] == ['up', 'down']
    assert any(item['value'] == 'should_have_probed_more_about_stated_topics' for item in payload['categories']['down'])
