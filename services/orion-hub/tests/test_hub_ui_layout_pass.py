"""Structural guards for the Hub chat-column layout pass.

Every assertion here is mutation-tested: reverting the change it names makes it
fail. An earlier version of this file passed a check that the new HTML and the
new function names EXIST, which stayed green when the Expand button, both
Dismiss-all handlers, and the wrapping fix itself were deleted. Existence is not
wiring -- assert the call site, not the definition.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
APP_JS = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _slice(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    return text[start : text.index(end_marker, start)]


def _opening_tag(html: str, element_id: str) -> str:
    """The full opening tag carrying this id, so class assertions cannot drift
    onto a nested element's classes."""
    at = html.index(f'id="{element_id}"')
    return html[html.rindex("<", 0, at) : html.index(">", at) + 1]


def _classes(html: str, element_id: str) -> set[str]:
    """Class TOKENS on that element. Substring checks are not safe here: the
    modal roots carry aria-hidden="true", so `"hidden" in tag` passes even with
    the hidden class stripped -- which is exactly how this suite once let that
    mutation through."""
    tag = _opening_tag(html, element_id)
    match = re.search(r'\sclass="([^"]*)"', tag)
    return set(match.group(1).split()) if match else set()


# --- Skill Runner + Container bring-up collapse to one bar ------------------


def test_operator_tools_bar_replaces_the_two_stacked_panels() -> None:
    html = _html()
    assert 'id="operatorToolsBar"' in html
    assert 'id="operatorToolsOpenBtn"' in html


def test_both_operator_panels_live_inside_the_operator_tools_modal() -> None:
    """The point of the pass: they cost one bar of height, not two panels."""
    html = _html()
    modal = _slice(html, 'id="operatorToolsModalRoot"', 'id="chatMessageExpandModalRoot"')
    assert 'id="skillRunnerPanel"' in modal
    assert 'id="containerBringupPanel"' in modal
    # ...and nowhere else, or the height would be back.
    assert html.count('id="skillRunnerPanel"') == 1
    assert html.count('id="containerBringupPanel"') == 1


def test_operator_tools_button_is_wired_to_the_opener() -> None:
    js = _js()
    assert "operatorToolsOpenBtn.addEventListener('click', () => openOperatorToolsModal())" in js
    assert "operatorToolsModalClose.addEventListener" in js
    assert "isModalVisible(operatorToolsModalRoot)" in js


def test_new_modals_start_hidden() -> None:
    """Without `hidden` on the root these render over the whole page on load --
    the one regression here that would make the Hub unusable."""
    html = _html()
    for modal_id in ("operatorToolsModalRoot", "chatMessageExpandModalRoot"):
        assert "hidden" in _classes(html, modal_id), modal_id


# --- Voice card moves into the chat column ---------------------------------


def test_voice_visualizer_sits_between_the_chat_rule_and_the_transcript() -> None:
    html = _html()
    rule = html.index("Oríon + Juniper")
    visualizer = html.index('id="visualizerContainer"')
    transcript = html.index('id="conversation"')
    assert rule < visualizer < transcript, "voice card is not in the chat column"


def test_voice_card_is_height_constrained_and_cannot_grow_the_column() -> None:
    # The container's OWN tag -- scanning a window past it would pass on the
    # canvas's classes instead.
    classes = _classes(_html(), "visualizerContainer")
    assert "h-28" in classes
    assert "shrink-0" in classes
    assert "min-h-0" in classes


def test_canvas_is_sized_from_its_own_box_not_the_padded_container() -> None:
    """Measuring the container counted padding + the label row, so the bars
    drew past the bottom of the card."""
    js = _js()
    assert "visualizerCanvas.clientWidth" in js
    assert "visualizerCanvas.clientHeight" in js
    # Both dimensions, not just height: the container is gone from app.js
    # entirely now, so neither can quietly come back.
    assert "visualizerContainer" not in js
    assert "new ResizeObserver(setAllCanvasSizes)" in js


# --- Per-message expand ----------------------------------------------------


def test_messages_can_be_opened_full_screen() -> None:
    html = _html()
    assert 'id="chatMessageExpandModalRoot"' in html
    assert 'id="chatMessageExpandModalBody"' in html


def test_append_message_actually_builds_and_wires_the_expand_button() -> None:
    """Asserting only that openChatMessageExpandModal is DEFINED stayed green
    with the whole Expand block deleted."""
    js = _js()
    body = _slice(js, "function appendMessage(", "function collectConversationTurnsUpTo")
    assert "openChatMessageExpandModal(sender, [attachmentStrip, body], displayText)" in body
    assert "expandBtn.addEventListener('click'" in body


def test_expand_button_is_gated_on_trimmed_text() -> None:
    """A whitespace-only turn is also a workflowOnlyTurn, so its body node is
    never appended -- an ungated button opens a blank full-screen modal."""
    js = _js()
    body = _slice(js, "function appendMessage(", "function collectConversationTurnsUpTo")
    assert "if (displayText.trim()) {" in body


def test_expand_modal_clones_and_refuses_to_open_on_nothing() -> None:
    js = _js()
    body = _slice(js, "function openChatMessageExpandModal", "function closeChatMessageExpandModal")
    assert "cloneNode(true)" in body
    assert "if (!nodes.length) return;" in body


# --- Elapsed turn timer ----------------------------------------------------


def test_turn_timer_chip_exists_and_is_driven_by_turn_in_flight() -> None:
    assert 'id="chatTurnTimer"' in _html()
    js = _js()
    assert "function setTurnInFlight(" in js
    assert "startTurnTimer(owner)" in js


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


def test_each_transport_owns_the_clock_it_started() -> None:
    """A WebSocket reconnect landing mid-HTTP-turn must not stop that turn's
    timer, nor report the reconnect gap as the turn's duration."""
    js = _js()
    assert "if (owner && turnTimerOwner && owner !== turnTimerOwner) return;" in js
    assert "startTurnTimer('http')" in js
    assert ".finally(() => { stopTurnTimer('http'); });" in js


def test_socket_close_freezes_the_clock_instead_of_reconnect_repainting_it() -> None:
    js = _js()
    onclose = _slice(js, "socket.onclose = (e) => {", "socket.onerror")
    assert "stopTurnTimer('ws', { repaint: false });" in onclose


def test_a_new_turn_cannot_inherit_the_previous_turn_start_time() -> None:
    js = _js()
    body = _slice(js, "function startTurnTimer(", "function stopTurnTimer(")
    # The old early-return guard would hand turn B turn A's accumulated elapsed.
    assert "if (turnTimerHandle) return;" not in body
    assert "if (turnTimerHandle) window.clearInterval(turnTimerHandle);" in body


def test_the_timer_chip_cannot_throw_out_of_the_websocket_frame_handler() -> None:
    """paintTurnTimer is reached from updateStatusBasedOnState inside
    socket.onmessage's try/catch -- an unguarded module call there would abandon
    handleTtsFields and error rendering for the rest of the frame."""
    js = _js()
    body = _slice(js, "function paintTurnTimer(", "function startTurnTimer(")
    assert "window.OrionTurnTimer && window.OrionTurnTimer.formatTurnElapsed" in body
    assert "if (typeof format !== 'function') return;" in body


# --- Dismiss all -----------------------------------------------------------


def test_dismiss_all_buttons_exist_and_are_wired_to_their_handlers() -> None:
    html = _html()
    assert 'id="attentionDismissAllBtn"' in html
    assert 'id="notificationsDismissAllBtn"' in html
    js = _js()
    # The click wiring, not just the function definitions.
    assert "attentionDismissAllBtn.addEventListener('click', () => dismissAllPendingAttention())" in js
    assert "notificationsDismissAllBtn.addEventListener('click', () => dismissAllNotifications())" in js


def test_attention_dismiss_all_reuses_the_single_item_ack() -> None:
    """No new server shape: bulk dismiss is N of the existing ack call."""
    js = _js()
    body = _slice(js, "async function dismissAllPendingAttention", "function dismissAllNotifications")
    assert "handleAttentionAck(id, 'dismissed')" in body


def test_attention_dismiss_all_stays_disabled_until_every_ack_settles() -> None:
    """Each ack re-renders, and the render re-derives disabled from the
    remaining list -- so the first ack landing would re-enable the button
    mid-flight and invite a second racing wave."""
    js = _js()
    body = _slice(js, "async function dismissAllPendingAttention", "function dismissAllNotifications")
    assert "if (dismissAllAttentionInFlight) return;" in body
    assert "} finally {" in body
    assert "dismissAllAttentionInFlight || pendingAttention.length === 0" in js


def test_notification_dismiss_all_is_scoped_to_the_active_filter() -> None:
    js = _js()
    body = _slice(js, "function dismissAllNotifications", "async function handleChatMessageReceipt")
    assert "notificationFilter" in body
    assert "isNotificationTrayItem" in body


# --- Cognitive Loops scroll window -----------------------------------------


def test_cognitive_loops_scrolls_like_the_notification_tray() -> None:
    classes = _classes(_html(), "cognitiveLoopsList")
    assert "overflow-y-auto" in classes
    assert "max-h-56" in classes


# --- Toggle strip no longer overflows the card -----------------------------


def test_recall_no_write_tts_social_room_strip_wraps() -> None:
    """Recall / No-write / TTS / Social room / Solo used to bleed out of the
    card because the OUTER row could not wrap. Anchoring on the nearest
    preceding <div> finds the inner toggle group instead, and passes with the
    outer row's flex-wrap stripped back off."""
    html = _html()
    start = html.index('for="recallToggle"')
    # The border-t rule is unique to the outer row of this strip.
    outer_open = html.rindex('<div class="', 0, html.rindex("border-t border-gray-700/80", 0, start))
    outer_tag = html[outer_open : html.index(">", outer_open) + 1]
    assert "border-t border-gray-700/80" in outer_tag, outer_tag
    assert "flex-wrap" in outer_tag, outer_tag


def test_the_actions_half_of_the_strip_wraps_independently() -> None:
    """Presence / Ask Claude / Solo sit in their own wrapping group so a long
    row breaks in a sensible place instead of one ragged line."""
    html = _html()
    strip = _slice(html, 'for="recallToggle"', 'id="recallModeSelect"')
    assert strip.count("flex-wrap") >= 1
    assert 'id="presenceStatusChip"' in strip
