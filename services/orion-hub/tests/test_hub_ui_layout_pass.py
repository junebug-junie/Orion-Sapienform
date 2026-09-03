"""Structural guards for the Hub chat-column layout pass.

Each assertion pins a claim the layout pass is selling, so a later edit that
quietly undoes one fails here rather than only on Juniper's screen:

* Skill Runner and Container bring-up occupy ONE bar, and their controls live
  in the Operator tools modal rather than two stacked full-width panels.
* The voice visualizer sits in the chat column between the "Oríon + Juniper"
  rule and the transcript, and is height-constrained by its card.
* Every chat message can be opened full-screen.
* Pending Attention and Notifications each have a Dismiss-all control.
* Cognitive Loops scrolls inside its own window.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
APP_JS = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _js() -> str:
    return APP_JS.read_text(encoding="utf-8")


# --- Skill Runner + Container bring-up collapse to one bar ------------------


def test_operator_tools_bar_replaces_the_two_stacked_panels() -> None:
    html = _html()
    assert 'id="operatorToolsBar"' in html
    assert 'id="operatorToolsOpenBtn"' in html


def test_both_operator_panels_live_inside_the_operator_tools_modal() -> None:
    """The point of the pass: they cost one bar of height, not two panels."""
    html = _html()
    modal_start = html.index('id="operatorToolsModalRoot"')
    modal_end = html.index('id="chatMessageExpandModalRoot"')
    modal = html[modal_start:modal_end]
    assert 'id="skillRunnerPanel"' in modal
    assert 'id="containerBringupPanel"' in modal
    # ...and nowhere else, or the height would be back.
    assert html.count('id="skillRunnerPanel"') == 1
    assert html.count('id="containerBringupPanel"') == 1


def test_operator_tools_modal_is_wired_open_and_closed() -> None:
    js = _js()
    assert "openOperatorToolsModal" in js
    assert "closeOperatorToolsModal" in js
    assert "isModalVisible(operatorToolsModalRoot)" in js


# --- Voice card moves into the chat column ---------------------------------


def test_voice_visualizer_sits_between_the_chat_rule_and_the_transcript() -> None:
    html = _html()
    rule = html.index("Oríon + Juniper")
    visualizer = html.index('id="visualizerContainer"')
    transcript = html.index('id="conversation"')
    assert rule < visualizer < transcript, "voice card is not in the chat column"


def test_voice_card_is_height_constrained_and_cannot_grow_the_column() -> None:
    html = _html()
    start = html.index('id="visualizerContainer"')
    block = html[start : start + 400]
    assert "h-28" in block and "shrink-0" in block
    assert "min-h-0" in block


def test_canvas_is_sized_from_its_own_box_not_the_padded_container() -> None:
    """Measuring the container counted padding + the label row, so the bars
    drew past the bottom of the card."""
    js = _js()
    assert "visualizerCanvas.clientHeight" in js
    assert "visualizerContainer.clientHeight" not in js
    assert "ResizeObserver" in js


# --- Per-message expand ----------------------------------------------------


def test_messages_can_be_opened_full_screen() -> None:
    html = _html()
    assert 'id="chatMessageExpandModalRoot"' in html
    assert 'id="chatMessageExpandModalBody"' in html
    js = _js()
    assert "openChatMessageExpandModal" in js
    assert "closeChatMessageExpandModal" in js


def test_expand_modal_clones_the_message_so_the_transcript_node_survives() -> None:
    js = _js()
    start = js.index("function openChatMessageExpandModal")
    body = js[start : js.index("function closeChatMessageExpandModal")]
    assert "cloneNode(true)" in body


# --- Elapsed turn timer ----------------------------------------------------


def test_turn_timer_chip_exists_and_is_driven_by_turn_in_flight() -> None:
    assert 'id="chatTurnTimer"' in _html()
    js = _js()
    assert "function setTurnInFlight(" in js
    assert "startTurnTimer()" in js and "stopTurnTimer()" in js


def test_no_raw_turn_in_flight_assignment_bypasses_the_timer() -> None:
    """Every transition must go through setTurnInFlight, or the timer and the
    Stop button drift apart."""
    js = _js()
    assert "turnInFlight = true;" not in js
    # The declaration is the only bare write left; every later transition is a
    # setTurnInFlight call. (The setter itself assigns Boolean(next).)
    assert js.count("turnInFlight = false;") == 1
    assert "let turnInFlight = false;" in js
    assert "turnInFlight = Boolean(next);" in js


def test_http_fallback_stops_the_timer_on_both_outcomes() -> None:
    js = _js()
    assert ".finally(() => { stopTurnTimer(); });" in js


# --- Dismiss all -----------------------------------------------------------


def test_pending_attention_and_notifications_each_have_dismiss_all() -> None:
    html = _html()
    assert 'id="attentionDismissAllBtn"' in html
    assert 'id="notificationsDismissAllBtn"' in html
    js = _js()
    assert "dismissAllPendingAttention" in js
    assert "dismissAllNotifications" in js


def test_attention_dismiss_all_reuses_the_single_item_ack() -> None:
    """No new server shape: bulk dismiss is N of the existing ack call."""
    js = _js()
    start = js.index("async function dismissAllPendingAttention")
    body = js[start : js.index("function dismissAllNotifications")]
    assert "handleAttentionAck(id, 'dismissed')" in body


def test_notification_dismiss_all_is_scoped_to_the_active_filter() -> None:
    js = _js()
    start = js.index("function dismissAllNotifications")
    body = js[start : start + 1200]
    assert "notificationFilter" in body
    assert "isNotificationTrayItem" in body


# --- Cognitive Loops scroll window -----------------------------------------


def test_cognitive_loops_scrolls_like_the_notification_tray() -> None:
    html = _html()
    loops = html[html.index('id="cognitiveLoopsList"') :][:200]
    assert "overflow-y-auto" in loops
    assert "max-h-56" in loops


# --- Toggle strip no longer overflows the card -----------------------------


def test_recall_no_write_tts_social_room_strip_wraps() -> None:
    """Recall / No-write / TTS / Social room / Solo used to bleed out of the
    card because the row could not wrap."""
    html = _html()
    start = html.index('for="recallToggle"')
    row_open = html.rindex("<div class=", 0, start)
    row = html[row_open:start]
    assert "flex-wrap" in row
