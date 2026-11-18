"""The metacog upstream names which core page version a turn saw, never a copy of the page text."""
from __future__ import annotations

from app.chat_turn_metacog_gate import _capped_grounding_capsule_for_upstream


def test_core_pages_are_reduced_to_id_and_version():
    capsule = {
        "memory_digest": "short",
        "core_pages": [{"concept_id": "page:juniper", "version": 4, "text": "x " * 400, "evidence_count": 30}],
    }
    capped = _capped_grounding_capsule_for_upstream(capsule)
    assert capped["core_pages"] == [{"concept_id": "page:juniper", "version": 4}]
    assert capsule["core_pages"][0]["text"].startswith("x ")   # the turn's own capsule is untouched


def test_capsule_without_core_pages_is_unchanged():
    assert _capped_grounding_capsule_for_upstream({"memory_digest": "m"}) == {"memory_digest": "m"}
    assert _capped_grounding_capsule_for_upstream(None) is None
