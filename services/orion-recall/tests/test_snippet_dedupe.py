from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.snippet_dedupe import (
    OrionDigestDeduper,
    duplicate_orion_reply_assistant,
    extract_orion_assistant_from_snippet,
)


def test_extract_handles_unicode_apostrophe() -> None:
    s = 'ExactUserText: "hi" OrionResponse: "Ain\u2019t nothin\' here"'
    assert "not" in extract_orion_assistant_from_snippet(s).lower()


def test_duplicate_detects_same_opener() -> None:
    long_t = (
        "Ain't nothin' but a hound dog, but I'm here for the chips and queso. You good? "
        "(You just called me silly.)"
    )
    short_t = "Ain't nothin' but a hound dog, but I'm here for the chips and queso. You good?"
    assert duplicate_orion_reply_assistant(long_t, short_t)


def test_deduper_skips_second_identical() -> None:
    body = "Same Orion line repeated."
    a = f'ExactUserText: "x" OrionResponse: "{body}"'
    b = f'ExactUserText: "y" OrionResponse: "{body}"'
    d = OrionDigestDeduper()
    assert d.should_emit_snippet(a) is True
    assert d.should_emit_snippet(b) is False
