(function (global) {
  // Session threads in Hub chat (scripts/session_roundup_loop.py,
  // docs/superpowers/specs/2026-11-15-session-threads-design.md).
  //
  // A coding session sends Orion a message (usually a pull request's design
  // spec). Hub holds it and answers with a RECEIPT at once; Orion reads every
  // held thread together once an hour and comes back with questions.
  //
  // One collapsed line per thread: title, sender session, status. A click
  // opens its turns, each under one of three labels: "Session", "Hub receipt"
  // (Hub's own words: never shown as Orion's) and "Orion". Turns arrive live
  // on the chat socket as kind "session_thread_turn"; Hub chat keeps no history
  // across a reload, so on page load GET /api/session-threads?hours=48 brings
  // the last 48 h back.
  //
  // Not in Orion's chat context: these rows carry no turnId and no
  // data-message-body, and are never written to chat history.
  //
  // Everything is rendered with textContent, never innerHTML: the text comes
  // from coding sessions and from a model.

  const RECENT_HOURS = 48;
  const rows = new Map();   // thread_id -> {div, line, detail, thread, loaded, turnIds}

  const SPEAKERS = {
    session: { label: "Session", colour: "text-sky-300" },
    hub: { label: "Hub receipt", colour: "text-slate-400" },
    orion: { label: "Orion", colour: "text-green-300" },
  };

  function conversation(doc) {
    return (doc || global.document).getElementById("conversation");
  }

  function label(doc, text, cls) {
    const el = doc.createElement("div");
    el.className = "text-xs font-semibold " + cls;
    el.textContent = text;
    return el;
  }

  function body(doc, text) {
    const el = doc.createElement("div");
    el.className = "text-sm text-white";
    el.style.whiteSpace = "pre-wrap";
    el.textContent = String(text || "");
    return el;
  }

  function scroll(container) {
    if (container) container.scrollTop = container.scrollHeight;
  }

  // One plain sentence for where a thread stands.
  function statusText(t) {
    switch (String(t.status || "")) {
      case "held": return "Held. Orion has not read it yet.";
      case "reading": return "Orion is reading it now.";
      case "asked": return "Orion asked questions.";
      case "no_questions": return "Orion read it. No questions.";
      case "closed": return "Orion read the answers. One round of questions per thread: done.";
      case "failed": return "Failed" + (t.status_reason ? " (" + t.status_reason + ")" : "") + ". Orion did not read it.";
      default: return String(t.status || "unknown") + ".";
    }
  }

  function speakerLabel(turn) {
    const s = SPEAKERS[String(turn.speaker || "")] || { label: String(turn.speaker || "?"), colour: "text-slate-400" };
    const meta = turn.meta || {};
    let text = s.label;
    if (turn.speaker === "hub" && meta.roundup_failed) text = "Hub (round-up failed)";
    if (turn.speaker === "hub" && meta.reply_unparsed) text = "Hub (reply not cut: read again next round-up)";
    if (turn.speaker === "orion" && meta.unparsed) text = "Orion (whole reply: Hub could not find this thread's section)";
    if (turn.speaker === "orion" && meta.requeued) text = "Orion (whole reply, not cut: sent nowhere, not the question round)";
    if (turn.speaker === "orion" && meta.closing_note) text = "Orion (closing note: not posted, not sent as questions)";
    return { text: text, colour: s.colour };
  }

  function lineText(t) {
    const title = String(t.title || "").replace(/\s+/g, " ");
    const who = String(t.session_name || t.session_id || "unknown session");
    return "Session thread · " + (title.length > 90 ? title.slice(0, 90) + "…" : title) +
      (t.pr_number ? " · PR #" + t.pr_number : "") + " · from " + who + " · " + statusText(t);
  }

  function turnRow(doc, turn) {
    const row = doc.createElement("div");
    row.className = "mt-2 pl-3 border-l-2 border-slate-600";
    row.dataset.role = "session-thread-turn";
    row.dataset.speaker = String(turn.speaker || "");
    const who = speakerLabel(turn);
    row.appendChild(label(doc, who.text, who.colour));
    row.appendChild(body(doc, turn.body !== undefined ? turn.body : turn.text));
    const specs = turn.spec_paths || ((turn.meta && turn.meta.spec) || []).map(function (s) { return s.path; });
    if (specs && specs.length) row.appendChild(label(doc, "Spec sent: " + specs.join(", "), "text-slate-500"));
    if (turn.pr_comment_url) row.appendChild(label(doc, "Also posted on the pull request: " + turn.pr_comment_url, "text-slate-500"));
    return row;
  }

  function appendTurn(entry, doc, turn) {
    const id = String(turn.turn_id || "");
    if (id && entry.turnIds.has(id)) return;
    if (id) entry.turnIds.add(id);
    entry.detail.appendChild(turnRow(doc, turn));
  }

  async function loadTurns(entry, doc, fetchFn) {
    try {
      const resp = await fetchFn("/api/session-threads/" + encodeURIComponent(entry.thread.thread_id));
      const data = resp.ok ? await resp.json() : null;
      if (!data) throw new Error("not ok");
      (data.turns || []).forEach(function (t) { appendTurn(entry, doc, t); });
      entry.loaded = true;
    } catch (_e) {
      entry.detail.appendChild(label(doc, "Could not load the turns.", "text-red-300"));
    }
  }

  // The collapsed line for one thread; created once, updated after.
  function renderThread(t, doc, fetchFn) {
    doc = doc || global.document;
    fetchFn = fetchFn || global.fetch.bind(global);
    const container = conversation(doc);
    const threadId = String((t && t.thread_id) || "");
    if (!container || !threadId) return null;
    let entry = rows.get(threadId);
    if (entry) {
      entry.thread = Object.assign({}, entry.thread, t);
      entry.line.textContent = lineText(entry.thread);
      return entry.div;
    }
    const div = doc.createElement("div");
    div.className = "mb-2 pl-3 border-l-2 border-sky-700";
    div.dataset.role = "session-thread";
    div.dataset.threadId = threadId;
    const line = doc.createElement("button");
    line.type = "button";
    line.className = "text-left text-xs text-slate-400 hover:text-slate-200";
    const detail = doc.createElement("div");
    detail.style.display = "none";
    entry = { div: div, line: line, detail: detail, thread: Object.assign({}, t), loaded: false, turnIds: new Set() };
    line.textContent = lineText(entry.thread);
    line.addEventListener("click", async function () {
      if (detail.style.display !== "none") { detail.style.display = "none"; return; }
      detail.style.display = "";
      if (!entry.loaded) await loadTurns(entry, doc, fetchFn);
    });
    div.appendChild(line);
    div.appendChild(detail);
    container.appendChild(div);
    rows.set(threadId, entry);
    return div;
  }

  // A live turn: the thread's line shows the new status; an open thread also gets the turn.
  function onFrame(frame, doc, fetchFn) {
    if (!frame || typeof frame !== "object" || frame.kind !== "session_thread_turn") return null;
    doc = doc || global.document;
    const div = renderThread({ thread_id: frame.thread_id, title: frame.title, status: frame.status,
      session_name: frame.session_name, pr_number: frame.pr_number }, doc, fetchFn);
    const entry = rows.get(String(frame.thread_id || ""));
    if (!entry) return null;
    // An opened thread that already loaded its turns gets the new one; a closed one loads all on its click.
    if (entry.loaded) appendTurn(entry, doc, frame);
    if (frame.speaker === "orion") {
      // Orion's questions are the news: the line moves to the bottom of chat, where new things are.
      const container = conversation(doc);
      if (container && container.appendChild) { div.remove(); container.appendChild(div); scroll(container); }
    }
    return div;
  }

  // Oldest first, so the newest ends up at the bottom like the rest of chat.
  async function loadRecent(fetchFn, doc) {
    fetchFn = fetchFn || global.fetch.bind(global);
    try {
      const resp = await fetchFn("/api/session-threads?hours=" + RECENT_HOURS + "&limit=100");
      if (!resp.ok) return 0;
      const data = await resp.json();
      const list = (data && data.threads) || [];
      list.slice().sort(function (a, b) { return String(a.updated_at || "").localeCompare(String(b.updated_at || "")); })
        .forEach(function (t) { renderThread(t, doc, fetchFn); });
      scroll(conversation(doc || global.document));
      return list.length;
    } catch (_e) {
      return 0;
    }
  }

  global.OrionSessionThreads = { onFrame: onFrame, renderThread: renderThread, statusText: statusText,
    lineText: lineText, loadRecent: loadRecent };
  if (global.document && global.document.addEventListener) {
    global.document.addEventListener("DOMContentLoaded", function () { loadRecent(); });
  }
})(typeof window !== "undefined" ? window : globalThis);
