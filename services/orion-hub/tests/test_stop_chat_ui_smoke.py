from pathlib import Path

_HUB_ROOT = Path(__file__).resolve().parents[1]


def test_template_has_stop_button():
    html = (_HUB_ROOT / "templates" / "index.html").read_text()
    assert 'id="stopButton"' in html
    assert "hidden" in html.split('id="stopButton"', 1)[1].split(">", 1)[0]


def test_app_js_wires_stop_button():
    js = (_HUB_ROOT / "static" / "js" / "app.js").read_text()
    assert "getElementById('stopButton')" in js
    assert "stopCurrentTurn" in js
    assert "stopButton.addEventListener('click', stopCurrentTurn)" in js
    assert "/api/chat/turn/cancel" in js


def test_app_js_stop_request_uses_connection_id_not_session_id():
    """Regression guard: the cancel request must key off the per-connection id
    captured from the WS 'connection_ready' frame, not session_id — session_id is
    shared across every browser tab via localStorage, so keying on it would let a
    stop click in one tab cancel a different tab's turn.
    """
    js = (_HUB_ROOT / "static" / "js" / "app.js").read_text()
    stop_fn = js.split("async function stopCurrentTurn()", 1)[1].split("\n  }\n", 1)[0]
    assert "connection_id" in stop_fn
    assert "activeConnectionId" in stop_fn
    assert "orionSessionId" not in stop_fn


def test_app_js_shows_stop_button_on_every_turn_kind_send():
    """The Stop button must be shown for every WS send path that starts a
    server-cancellable turn (orion text and voice), not just text chat.
    """
    js = (_HUB_ROOT / "static" / "js" / "app.js").read_text()
    assert js.count("stopButton.classList.remove('hidden')") >= 2


def test_app_js_captures_connection_id_from_connection_ready_frame():
    js = (_HUB_ROOT / "static" / "js" / "app.js").read_text()
    assert "d.type === 'connection_ready'" in js
    assert "activeConnectionId = d.connection_id" in js
