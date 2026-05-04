from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from orion.core.contracts.recall import MemoryItemV1

from app.render import render_items


def test_memory_digest_emits_orion_catchphrase_once() -> None:
    orion = (
        "Ain't nothin' but a hound dog, but I'm here for the chips and queso. You good? "
        "Extra line so body is long enough to dedupe."
    )
    s1 = f'ExactUserText: "sup" OrionResponse: "{orion}"'
    s2 = f'ExactUserText: "lol" OrionResponse: "{orion}"'
    items = [
        MemoryItemV1(
            id="a",
            source="vector",
            source_ref="orion_chat_turns",
            score=0.15,
            snippet=s1,
        ),
        MemoryItemV1(
            id="b",
            source="vector",
            source_ref="orion_chat_turns",
            score=0.13,
            snippet=s2,
        ),
    ]
    text, _ = render_items(items, 2000, profile_name="chat.general.v1")
    assert text.count("chips and queso") == 1
    assert text.count("vector:orion_chat_turns") == 1
