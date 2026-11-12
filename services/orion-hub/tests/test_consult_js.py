"""consult.js: a brief renders once with its buttons, and each button posts the right answer. Run in node
with a minimal fake DOM (the interaction, not only the page load)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parents[1] / "static" / "js" / "consult.js"

FAKE_DOM = r"""
function el(tag) {
  const node = {tag, children: [], dataset: {}, style: {}, handlers: {}, _text: "", className: "", value: "",
    appendChild(c) { c.parent = node; node.children.push(c); return c; },
    get childNodes() { return node.children; },
    addEventListener(k, f) { node.handlers[k] = f; },
    remove() { if (node.parent) node.parent.children = node.parent.children.filter(x => x !== node); },
    focus() {},
    set textContent(v) { node._text = String(v); if (v === "") node.children = []; },
    get textContent() { return node._text; },
  };
  return node;
}
const conv = el("div");
const doc = {createElement: el, getElementById: (id) => id === "conversation" ? conv : null};
function all(n, out) { out = out || []; out.push(n); n.children.forEach(c => all(c, out)); return out; }
function buttons() { return all(conv).filter(n => n.tag === "button"); }
function byText(t) { return buttons().find(b => b.textContent.startsWith(t)); }
const posts = [];
const fetchFn = async (url, init) => {
  if (init && init.method === "POST") { posts.push({url, body: JSON.parse(init.body)}); return {ok: true, status: 200, json: async () => ({})}; }
  if (url === "/api/consults?recent_hours=48") return {ok: true, status: 200, json: async () => ({consults: [
    {consult_id: "c9", status: "brief_sent", decision: null, brief_text: "B9", question: "fix the lease?"},
    {consult_id: "c8", status: "failed", status_reason: "orion_run_failed:timeout", question: "why does x stall?"},
    {consult_id: "c7", status: "answered", answer: "by design", question: "q7"},
  ]})};
  return {ok: true, status: 200, json: async () => ({turns: [{speaker: "Orion", text: "hi"}, {speaker: "Claude", text: "why?"}]})};
};
"""


def _node(body: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = (FAKE_DOM + f"require({json.dumps(str(JS))}); const api = globalThis.OrionConsult;\n"
              + "(async () => { const out = await (async () => {" + body + "})();"
              + " process.stdout.write(JSON.stringify(out)); })();")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_brief_renders_once_with_its_buttons():
    got = _node("""
      const first = api.renderBrief("c1", "@Juniper -- brief", doc, fetchFn);
      const again = api.renderBrief("c1", "@Juniper -- brief", doc, fetchFn);
      return [first !== null, again === null, buttons().map(b => b.textContent)];
    """)
    assert got == [True, True, ["Yes", "No", "No, because…", "Show the conversation"]]


def test_yes_posts_yes_and_removes_the_buttons():
    got = _node("""
      api.renderBrief("c1", "b", doc, fetchFn);
      await byText("Yes").handlers.click();
      await new Promise(r => setTimeout(r, 0));
      return [posts, buttons().length];
    """)
    assert got[0] == [{"url": "/api/consults/c1/decision", "body": {"decision": "yes", "reason": None}}]
    assert got[1] == 0


def test_no_because_needs_a_reason_before_it_sends():
    got = _node("""
      api.renderBrief("c2", "b", doc, fetchFn);
      const because = byText("No, because");
      await because.handlers.click();           // opens the box
      await because.handlers.click();           // empty: refused
      const refused = posts.length;
      const box = all(conv).find(n => n.tag === "textarea");
      box.value = "too costly";
      await because.handlers.click();
      await new Promise(r => setTimeout(r, 0));
      return [refused, posts];
    """)
    assert got[0] == 0
    assert got[1] == [{"url": "/api/consults/c2/decision", "body": {"decision": "no_because", "reason": "too costly"}}]


def test_show_the_conversation_lists_the_turns():
    got = _node("""
      api.renderBrief("c3", "b", doc, fetchFn);
      await byText("Show the conversation").handlers.click();
      return all(conv).map(n => n.textContent).filter(t => t === "hi" || t === "why?");
    """)
    assert got == ["hi", "why?"]


def test_waiting_briefs_load_on_page_open_and_turn_frames_render():
    got = _node("""
      const n = await api.loadRecentConsults(fetchFn, doc);
      globalThis.document = doc;
      api.onFrame({kind: "consult_turn", consult_id: "c9", speaker: "Claude", llm_response: "checked x.py:3"});
      return [n, all(conv).some(x => x.textContent === "B9"), all(conv).some(x => x.textContent === "checked x.py:3")];
    """)
    assert got == [3, True, True]


def test_other_recent_consults_are_one_collapsed_line_each_and_open_on_click():
    got = _node("""
      await api.loadRecentConsults(fetchFn, doc);
      const rows = conv.children.map(c => [c.dataset.role, c.dataset.consultId]);
      const failed = conv.children.find(c => c.dataset.consultId === "c8");
      const [line, detail] = failed.children;
      const closed = detail.style.display;
      await line.handlers.click();
      await new Promise(r => setTimeout(r, 0));
      const texts = all(detail).map(n => n.textContent);
      await line.handlers.click();
      return [rows, line.textContent, closed, texts, detail.style.display];
    """)
    rows, line, closed, texts, reclosed = got
    # Oldest first (the API is newest first); the undecided brief keeps its full card and buttons.
    assert rows == [["consult-summary", "c7"], ["consult-summary", "c8"], ["consult-brief", "c9"]]
    assert line == "Consult · failed · why does x stall?" and closed == "none"
    assert "Failed: orion_run_failed:timeout." in texts and "hi" in texts and "why?" in texts
    assert reclosed == "none"


def test_status_text_says_where_each_consult_stands():
    got = _node("""
      return [
        api.statusText({status: "open", status_reason: "awaiting_orion_run"}),
        api.statusText({status: "open", status_reason: "budget:weekly_limited"}),
        api.statusText({status: "answered", answer: "by design"}),
        api.statusText({status: "brief_sent", decision: "no_because", decision_reason: "too costly"}),
        api.statusText({status: "brief_refused", status_reason: "missing:cost"}),
      ];
    """)
    assert got == ["Orion is working on their turn.", "Waiting for Claude budget (weekly_limited).",
                   "Answered. Nothing for you. Answer: by design", "Brief sent. You said: no, because too costly.",
                   "Stopped: the brief failed its check (missing:cost). Nothing was sent."]


def test_consult_rows_stay_out_of_orions_context():
    """The memory bridge and Ask Claude read chat rows by turnId / data-message-body; consult rows have neither."""
    got = _node("""
      await api.loadRecentConsults(fetchFn, doc);
      return all(conv).some(n => "turnId" in n.dataset || "messageBody" in n.dataset);
    """)
    assert got is False
