"""2026-08-27: the `handleTtsFields` refactor and its `shouldAppendOrionWsPayload`
regression, checked against the real `app.js` source.

Review finding: no JS test/smoke was added for the HTTP-fallback TTS fix's
frontend half, only a Python backend test -- and as a direct consequence,
a real regression in `shouldAppendOrionWsPayload` (a merged
llm_response+tts_error frame, only possible via the new HTTP path, was
misclassified as "no text" and replaced the real reply with a false
"HTTP completed but no assistant text was returned" message) shipped
undetected.

`app.js` has no `module.exports` (it is a pure browser IIFE that touches
`document`/`window` at load time, not requireable in Node -- unlike the
small, pure-logic modules this repo's `*.test.js` files already cover, e.g.
`cognitive-loop-card.js`). Extracting the touched functions into a
requireable module is a legitimate follow-up, but a materially bigger
refactor than this fix warrants. Matches this file's own established
convention instead (see test_attention_frame_debug_panel.py,
test_websocket_agent_claude_routing.py): assert on the real source's
control-flow shape.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def _source() -> str:
    return APP_JS_PATH.read_text(encoding="utf-8")


def test_handle_tts_fields_is_shared_by_both_delivery_paths() -> None:
    """The whole point of the refactor: one function, two call sites (the
    live WS onmessage handler and the HTTP fallback's fetch().then()) --
    not two independent copies that can drift, which is exactly the shape
    of bug this whole incident chain has been about."""
    source = _source()
    # ".count(...)" on the bare "handleTtsFields(d)" substring would also
    # match its own `function handleTtsFields(d) {` definition -- count the
    # CALL form (trailing `;`, no `function` keyword) specifically.
    assert source.count("handleTtsFields(d);") == 2, (
        "handleTtsFields(d) must be CALLED from exactly two places -- the "
        "WS onmessage handler and the HTTP fallback's .then()"
    )
    assert "function handleTtsFields(d) {" in source


def test_should_append_orion_ws_payload_checks_llm_response_before_tts_error() -> None:
    """Regression guard for the actual incident: the old shape
    (`if (d.tts_error) return false;` as the very first check) treated ANY
    tts_error as "no text to show" -- correct for the WS lane, where a
    tts_error frame never carries real text, but wrong for the HTTP
    fallback path, which can merge a real llm_response with a failed TTS
    into the SAME object. The fix must gate specifically on the ABSENCE of
    llm_response, not on tts_error alone."""
    source = _source()
    idx = source.index("function shouldAppendOrionWsPayload(d)")
    body = source[idx : idx + 1200]
    assert "d.tts_error && !d.llm_response" in body, (
        "the tts_error check must require the absence of llm_response -- "
        "a bare `if (d.tts_error) return false` reintroduces the incident"
    )


def test_handle_tts_fields_still_queues_audio_and_reports_errors() -> None:
    """The extraction must not have dropped any of the three original
    behaviors: queueing playback, logging a debug line, and surfacing a
    tts_error as a visible system message."""
    source = _source()
    idx = source.index("function handleTtsFields(d)")
    body = source[idx : idx + 800]
    assert "audioQueue.push(" in body
    assert "processAudioQueue()" in body
    assert "d.tts_debug" in body
    assert "TTS warning" in body


def test_http_fallback_then_handler_calls_handle_tts_fields() -> None:
    """Placement check: the HTTP fallback's own .then(d => {...}) handler
    must actually call the shared function -- extracting it is useless if
    the new call site was never wired in."""
    source = _source()
    # A distinctive string unique to this specific .then() branch --
    # "updateMemoryPanelFromResponse(d);" alone appears 3 times in the file
    # (the WS handler has more than one call site of that function too), so
    # anchoring on it directly is not specific enough to isolate the HTTP
    # fallback block.
    marker = "HTTP completed but no assistant text was returned"
    assert source.count(marker) == 1
    idx = source.index(marker)
    following = source[idx : idx + 400]
    assert "handleTtsFields(d);" in following
