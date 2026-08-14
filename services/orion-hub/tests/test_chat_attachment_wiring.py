"""Hub-side validation of client-supplied attachment refs on a chat turn."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

from scripts.cortex_request_builder import _attachments_from_payload  # noqa: E402


def ref(**overrides):
    base = {
        "kind": "image",
        "sha256": "a" * 64,
        "mime": "image/png",
        "bytes": 128,
        "width": 64,
        "height": 64,
        "source_url": "http://100.92.216.81:8080/api/chat/attachments/" + "a" * 64,
        "filename": "shot.png",
    }
    base.update(overrides)
    return base


def test_absent_attachments_is_empty():
    assert _attachments_from_payload({}) == []
    assert _attachments_from_payload({"attachments": None}) == []
    assert _attachments_from_payload({"attachments": []}) == []


def test_non_list_attachments_is_empty():
    assert _attachments_from_payload({"attachments": {"sha256": "x"}}) == []
    assert _attachments_from_payload({"attachments": "a" * 64}) == []


def test_valid_ref_round_trips():
    out = _attachments_from_payload({"attachments": [ref()]})
    assert len(out) == 1
    assert out[0].sha256 == "a" * 64
    assert out[0].mime == "image/png"


def test_malformed_refs_are_dropped_not_fatal():
    out = _attachments_from_payload(
        {"attachments": [{"garbage": True}, ref(), "not-an-object", 42]}
    )
    assert len(out) == 1


def test_extra_fields_are_rejected_by_the_contract():
    """extra='forbid' on AttachmentRefV1 means a client cannot smuggle fields."""
    out = _attachments_from_payload({"attachments": [ref(injected="../../etc/passwd")]})
    assert out == []


def test_per_turn_cap_is_enforced(monkeypatch):
    # Patch the module's settings reference rather than the pydantic instance:
    # setattr on the Settings object does not stick (confirmed by this test
    # failing that way first), and patching the reference is the pattern the
    # rest of the Hub suite already uses.
    from scripts import cortex_request_builder as crb

    monkeypatch.setattr(crb, "settings", SimpleNamespace(HUB_CHAT_ATTACHMENT_MAX_PER_TURN=2))
    # Call through the module, not the top-level-imported name: conftest purges
    # cached `scripts.*` imports, so a re-import inside a test yields a NEW module
    # object whose globals the top-level-bound function does not share.
    out = crb._attachments_from_payload({"attachments": [ref(sha256=c * 64) for c in "abcdef"]})
    assert len(out) == 2


def test_zero_cap_refuses_everything(monkeypatch):
    """0 is a kill switch, not 'unlimited'."""
    from scripts import cortex_request_builder as crb

    monkeypatch.setattr(crb, "settings", SimpleNamespace(HUB_CHAT_ATTACHMENT_MAX_PER_TURN=0))
    assert crb._attachments_from_payload({"attachments": [ref()]}) == []


def test_default_cap_applies_without_patching():
    """Guards the real configured default, not just a patched one."""
    out = _attachments_from_payload({"attachments": [ref(sha256=c * 64) for c in "abcdefgh"]})
    assert 0 < len(out) <= 8
