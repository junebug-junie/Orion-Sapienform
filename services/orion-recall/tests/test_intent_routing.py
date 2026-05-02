from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.intent import classify_intent_v1, resolve_profile_for_intent


def test_biographical_intent() -> None:
    ic = classify_intent_v1("Where do I live?")
    assert ic.intent == "biographical"
    prof = resolve_profile_for_intent(ic.intent, fallback_profile="reflect.v1")
    assert prof == "biographical.v1"


def test_fallback_profile() -> None:
    ic = classify_intent_v1("")
    assert ic.intent == "unknown"
    assert resolve_profile_for_intent(ic.intent, fallback_profile="chat.general.v1") == "chat.general.v1"
