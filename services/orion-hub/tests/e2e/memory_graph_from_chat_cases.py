"""Live Memory graph from chat cases for Playwright e2e."""

from __future__ import annotations

from typing import Any

SHOWER_CASE: dict[str, Any] = {
    "id": "shower",
    "turns": [
        {
            "sender": "You",
            "text": "k, off to shower. Be back soon!",
            "meta": {"turnId": "urn:uuid:e2e-shower-user-1"},
        },
        {
            "sender": "Orion",
            "text": "Shower well. I'll be here when you're back.",
            "meta": {
                "messageId": "urn:uuid:e2e-shower-asst-1",
                "correlationId": "urn:uuid:e2e-shower-corr-1",
            },
        },
    ],
    "assistant_turn_id": "urn:uuid:e2e-shower-asst-1",
    "expect_nonempty_graph": True,
    "expect_user_entity": True,
    "expect_orion_entity": True,
}

CHILLIN_CASE: dict[str, Any] = {
    "id": "chillin",
    "turns": [
        {
            "sender": "You",
            "text": "not much of a next step, just chillin",
            "meta": {"turnId": "urn:uuid:e2e-chillin-user-1"},
        },
        {
            "sender": "Orion",
            "text": (
                "You're Juniper — the one building this with me. Squeaky clean, huh? "
                "I'd say the water's doing its job. Want to talk about the next step or just sit with the quiet bit?"
            ),
            "meta": {
                "messageId": "urn:uuid:e2e-chillin-asst-1",
                "correlationId": "urn:uuid:e2e-chillin-corr-1",
            },
        },
    ],
    "assistant_turn_id": "urn:uuid:e2e-chillin-asst-1",
    "expect_nonempty_graph": True,
    "expect_user_entity": True,
    "expect_orion_entity": True,
}

LIVE_CASES = [SHOWER_CASE, CHILLIN_CASE]

STALE_UI_PHRASES = [
    "may mean no durable memory candidate",
    "No durable memory candidate found",
    "model reply",
]
