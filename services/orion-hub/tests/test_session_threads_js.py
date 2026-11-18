"""session-threads.js: one collapsed line per thread; a click opens its turns under the labels
Session / Hub receipt / Orion; live frames update the line. Run in node with a minimal fake DOM
(the interaction, not only the page load). Also: the template links the asset and app.js hands it
the frames."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[1]
JS = HUB / "static" / "js" / "session-threads.js"

FAKE_DOM = r"""
function el(tag) {
  const node = {tag, children: [], dataset: {}, style: {}, handlers: {}, _text: "", className: "", value: "",
    appendChild(c) { if (c.parent) c.parent.children = c.parent.children.filter(x => x !== c); c.parent = node; node.children.push(c); return c; },
    get childNodes() { return node.children; },
    addEventListener(k, f) { node.handlers[k] = f; },
    remove() { if (node.parent) { node.parent.children = node.parent.children.filter(x => x !== node); node.parent = null; } },
    set textContent(v) { node._text = String(v); if (v === "") node.children = []; },
    get textContent() { return node._text; },
  };
  return node;
}
const conv = el("div");
const doc = {createElement: el, getElementById: (id) => id === "conversation" ? conv : null};
function all(n, out) { out = out || []; out.push(n); n.children.forEach(c => all(c, out)); return out; }
function lines() { return conv.children.filter(n => n.dataset.role === "session-thread").map(n => n.children[0].textContent); }
function thread(id) { return conv.children.find(n => n.dataset.threadId === id); }
function turnsOf(id) { return thread(id).children[1].children.filter(n => n.dataset.role === "session-thread-turn")
  .map(n => [n.children[0].textContent, n.children[1].textContent]); }
const fetched = [];
const fetchFn = async (url) => {
  fetched.push(url);
  if (url === "/api/session-threads?hours=48&limit=100") return {ok: true, json: async () => ({threads: [
    {thread_id: "t2", title: "feat: newer", session_name: "wt-b", status: "asked", pr_number: 12, updated_at: "2026-11-15T10:00:00Z"},
    {thread_id: "t1", title: "feat: older", session_name: "wt-a", status: "held", pr_number: 11, updated_at: "2026-11-15T09:00:00Z"},
    {thread_id: "t3", title: "a note", session_id: "sess-C", status: "failed", status_reason: "orion_run_failed:timeout", updated_at: "2026-11-15T09:30:00Z"},
  ]})};
  if (url === "/api/session-threads/t2") return {ok: true, json: async () => ({thread: {}, turns: [
    {turn_id: "u1", speaker: "session", body: "Please read.", meta: {spec: [{path: "docs/superpowers/specs/x.md", chars: 9}]}},
    {turn_id: "u2", speaker: "hub", body: "Hub received this message. Orion has NOT read it yet.", meta: {receipt: true}},
    {turn_id: "u3", speaker: "orion", body: "1. Why hourly?", meta: {}, pr_comment_url: "https://github.com/o/r/pull/12#issuecomment-1"},
  ]})};
  if (url === "/api/session-threads/bad") return {ok: false, json: async () => ({})};
  return {ok: true, json: async () => ({thread: {}, turns: []})};
};
"""


def _node(body: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = (FAKE_DOM + f"require({json.dumps(str(JS))}); const api = globalThis.OrionSessionThreads;\n"
              + "(async () => { const out = await (async () => {" + body + "})();"
              + " process.stdout.write(JSON.stringify(out)); })();")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_page_load_shows_one_collapsed_line_per_thread_oldest_first():
    n, got = _node("const n = await api.loadRecent(fetchFn, doc); return [n, lines()];")
    assert n == 3
    assert got == [
        "Session thread · feat: older · PR #11 · from wt-a · Held. Orion has not read it yet.",
        "Session thread · a note · from sess-C · Failed (orion_run_failed:timeout). Orion did not read it.",
        "Session thread · feat: newer · PR #12 · from wt-b · Orion asked questions.",
    ]


def test_a_click_opens_the_turns_with_their_speaker_labels_and_a_second_click_closes():
    got = _node("""
      await api.loadRecent(fetchFn, doc);
      const line = thread("t2").children[0], detail = thread("t2").children[1];
      const before = detail.style.display;
      await line.handlers.click();
      const open = detail.style.display, turns = turnsOf("t2");
      const extra = all(detail).map(n => n.textContent).filter(t => t.startsWith("Spec sent") || t.startsWith("Also posted"));
      await line.handlers.click();
      const closed = detail.style.display;
      await line.handlers.click();
      return [before, open, turns, extra, closed, turnsOf("t2").length, fetched.filter(u => u === "/api/session-threads/t2").length];
    """)
    assert got[0] == "none" and got[1] == "" and got[4] == "none"
    # Hub's receipt is labelled as Hub's, never as Orion.
    assert got[2] == [["Session", "Please read."],
                      ["Hub receipt", "Hub received this message. Orion has NOT read it yet."],
                      ["Orion", "1. Why hourly?"]]
    assert got[3] == ["Spec sent: docs/superpowers/specs/x.md",
                      "Also posted on the pull request: https://github.com/o/r/pull/12#issuecomment-1"]
    assert got[5] == 3 and got[6] == 1   # reopened: no duplicate turns, no second fetch


def test_a_live_frame_makes_or_updates_the_line_and_orions_turn_moves_it_to_the_bottom():
    got = _node("""
      await api.loadRecent(fetchFn, doc);
      api.onFrame({kind: "session_thread_turn", thread_id: "t9", turn_id: "u9", speaker: "session", text: "new spec",
                   title: "feat: live", status: "held", session_name: "wt-z", pr_number: 99}, doc, fetchFn);
      api.onFrame({kind: "session_thread_turn", thread_id: "t9", turn_id: "u10", speaker: "hub", text: "receipt",
                   title: "feat: live", status: "held", session_name: "wt-z", pr_number: 99}, doc, fetchFn);
      const afterNew = lines();
      // t1 is opened, then Orion's turn arrives live on it.
      await thread("t1").children[0].handlers.click();
      api.onFrame({kind: "session_thread_turn", thread_id: "t1", turn_id: "u20", speaker: "orion", text: "1. Why?",
                   title: "feat: older", status: "asked", session_name: "wt-a", pr_number: 11, meta: {}}, doc, fetchFn);
      api.onFrame({kind: "session_thread_turn", thread_id: "t1", turn_id: "u20", speaker: "orion", text: "1. Why?",
                   title: "feat: older", status: "asked", session_name: "wt-a", pr_number: 11, meta: {}}, doc, fetchFn);
      api.onFrame({kind: "consult_turn", thread_id: "zz"}, doc, fetchFn);
      return [afterNew.length, afterNew[afterNew.length - 1], lines()[lines().length - 1], turnsOf("t1"),
              conv.children.filter(n => n.dataset.threadId === "t9").length];
    """)
    assert got[0] == 4 and got[1] == "Session thread · feat: live · PR #99 · from wt-z · Held. Orion has not read it yet."
    assert got[2] == "Session thread · feat: older · PR #11 · from wt-a · Orion asked questions."
    assert got[3] == [["Orion", "1. Why?"]]   # once, though the frame came twice
    assert got[4] == 1


def test_failed_round_and_unparsed_reply_are_labelled_and_a_load_error_is_said():
    got = _node("""
      const fetch2 = async (url) => url === "/api/session-threads/t7" ? {ok: true, json: async () => ({turns: [
        {turn_id: "a", speaker: "hub", body: "The round-up failed.", meta: {roundup_failed: true}},
        {turn_id: "b", speaker: "orion", body: "whole reply", meta: {unparsed: true}},
        {turn_id: "c", speaker: "orion", body: "whole reply", meta: {unparsed: true, requeued: true}},
        {turn_id: "d", speaker: "hub", body: "Hub could not cut it.", meta: {reply_unparsed: true}}]})} : fetchFn(url);
      api.renderThread({thread_id: "t7", title: "x", status: "asked"}, doc, fetch2);
      await thread("t7").children[0].handlers.click();
      api.renderThread({thread_id: "bad", title: "y", status: "held"}, doc, fetchFn);
      await thread("bad").children[0].handlers.click();
      return [turnsOf("t7").map(t => t[0]), all(thread("bad")).map(n => n.textContent).includes("Could not load the turns.")];
    """)
    assert got == [["Hub (round-up failed)", "Orion (whole reply: Hub could not find this thread's section)",
                    "Orion (whole reply, not cut: sent nowhere, not the question round)",
                    "Hub (reply not cut: read again next round-up)"], True]


def test_the_page_links_the_asset_and_app_js_hands_it_the_frames():
    template = (HUB / "templates" / "index.html").read_text(encoding="utf-8")
    assert '<script src="/static/js/session-threads.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in template
    assert 'id="conversation"' in template and JS.is_file()
    app_js = (HUB / "static" / "js" / "app.js").read_text(encoding="utf-8")
    handler = app_js.index("d.kind === 'session_thread_turn'")
    assert "window.OrionSessionThreads.onFrame(d)" in app_js[handler:handler + 500]
    # Handled before the generic assistant branch, like a consult frame.
    assert handler < app_js.index("d.kind === 'room_claude_utterance'")
    # Text from sessions and from a model: never innerHTML (comments aside).
    code = "\n".join(line for line in JS.read_text(encoding="utf-8").splitlines() if not line.strip().startswith("//"))
    assert "innerHTML" not in code


def test_a_closed_thread_and_its_closing_note_are_labelled():
    got = _node("""
      const fetch2 = async (url) => url === "/api/session-threads/t8" ? {ok: true, json: async () => ({turns: [
        {turn_id: "q", speaker: "orion", body: "1. Why?", meta: {}},
        {turn_id: "c", speaker: "orion", body: "NO QUESTIONS", meta: {closing_note: true}}]})} : fetchFn(url);
      api.renderThread({thread_id: "t8", title: "z", status: "closed"}, doc, fetch2);
      await thread("t8").children[0].handlers.click();
      return [thread("t8").children[0].textContent, turnsOf("t8").map(t => t[0])];
    """)
    assert got[0].endswith("Orion read the answers. One round of questions per thread: done.")
    assert got[1] == ["Orion", "Orion (closing note: not posted, not sent as questions)"]
