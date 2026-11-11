(function (global) {
  // Consults in Hub chat (scripts/consult_loop.py,
  // docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md).
  //
  // Orion takes a plumbing question to Claude before Juniper. Each turn arrives
  // live on the chat socket as kind "consult_turn"; the brief as
  // "consult_brief", with Yes / No / No-because buttons that POST to
  // /api/consults/{id}/decision. On page load, briefs still waiting for her
  // answer are fetched from GET /api/consults, because Hub chat keeps no
  // history across a reload and the brief must not depend on her being online.
  //
  // Everything is rendered with textContent, never innerHTML: the text comes
  // from two models.

  const shownBriefs = new Set();

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

  function renderTurn(frame, doc) {
    doc = doc || global.document;
    const container = conversation(doc);
    if (!container) return null;
    const speaker = String(frame.speaker || "");
    const div = doc.createElement("div");
    div.className = "mb-3 pl-3 border-l-2 border-slate-600";
    div.dataset.role = "consult";
    div.dataset.consultId = String(frame.consult_id || "");
    div.dataset.sender = speaker;
    const colour = speaker === "Claude" ? "text-amber-300" : (speaker === "Orion" ? "text-green-300" : "text-slate-400");
    div.appendChild(label(doc, "Consult · " + speaker, colour));
    div.appendChild(body(doc, frame.llm_response));
    container.appendChild(div);
    scroll(container);
    return div;
  }

  async function postDecision(fetchFn, consultId, decision, reason) {
    const resp = await fetchFn("/api/consults/" + encodeURIComponent(consultId) + "/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision: decision, reason: reason || null }),
    });
    let data = null;
    try { data = await resp.json(); } catch (_e) { data = null; }
    return { ok: resp.ok, status: resp.status, body: data };
  }

  function button(doc, text, onClick) {
    const b = doc.createElement("button");
    b.type = "button";
    b.className = "mr-2 mt-2 px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 text-white text-xs";
    b.textContent = text;
    b.addEventListener("click", onClick);
    return b;
  }

  function renderBrief(consultId, text, doc, fetchFn) {
    doc = doc || global.document;
    fetchFn = fetchFn || global.fetch.bind(global);
    const container = conversation(doc);
    if (!container || !consultId || shownBriefs.has(consultId)) return null;
    shownBriefs.add(consultId);
    const div = doc.createElement("div");
    div.className = "mb-4 p-3 rounded border border-amber-500/60 bg-slate-800/60";
    div.dataset.role = "consult-brief";
    div.dataset.consultId = consultId;
    div.appendChild(label(doc, "Orion & Claude → Juniper", "text-amber-300"));
    div.appendChild(body(doc, text));

    const status = doc.createElement("div");
    status.className = "text-xs text-slate-400 mt-2";
    const actions = doc.createElement("div");
    const reasonBox = doc.createElement("textarea");
    reasonBox.className = "w-full mt-2 p-2 rounded bg-slate-900 text-white text-sm";
    reasonBox.placeholder = "Why not?";
    reasonBox.style.display = "none";

    async function send(decision) {
      const reason = decision === "no_because" ? reasonBox.value.trim() : "";
      if (decision === "no_because" && !reason) {
        status.textContent = "Say why, then send.";
        return;
      }
      status.textContent = "Sending…";
      const result = await postDecision(fetchFn, consultId, decision, reason);
      if (result.ok) {
        actions.remove();
        reasonBox.remove();
        status.textContent = "Answered: " + (decision === "no_because" ? "no, because " + reason : decision) +
          ". Orion sees it at their next curiosity run.";
      } else {
        status.textContent = "Not saved (" + result.status + "): " + ((result.body && result.body.detail) || "error");
      }
    }

    actions.appendChild(button(doc, "Yes", function () { send("yes"); }));
    actions.appendChild(button(doc, "No", function () { send("no"); }));
    const because = button(doc, "No, because…", function () {
      if (reasonBox.style.display === "none") {
        reasonBox.style.display = "";
        because.textContent = "Send “no, because”";
        reasonBox.focus();
      } else {
        send("no_because");
      }
    });
    actions.appendChild(because);

    const transcript = doc.createElement("div");
    transcript.className = "mt-2";
    const show = button(doc, "Show the conversation", async function () {
      if (transcript.childNodes.length) {
        transcript.textContent = "";
        show.textContent = "Show the conversation";
        return;
      }
      const resp = await fetchFn("/api/consults/" + encodeURIComponent(consultId));
      const data = resp.ok ? await resp.json() : null;
      (data && data.turns || []).forEach(function (t) {
        const row = doc.createElement("div");
        row.className = "mt-2 pl-3 border-l-2 border-slate-600";
        row.appendChild(label(doc, t.speaker, t.speaker === "Claude" ? "text-amber-300" : "text-green-300"));
        row.appendChild(body(doc, t.text || t.error || ""));
        transcript.appendChild(row);
      });
      show.textContent = "Hide the conversation";
    });
    actions.appendChild(show);

    div.appendChild(actions);
    div.appendChild(reasonBox);
    div.appendChild(status);
    div.appendChild(transcript);
    container.appendChild(div);
    scroll(container);
    return div;
  }

  function onFrame(frame) {
    if (!frame || typeof frame !== "object") return;
    if (frame.kind === "consult_turn") renderTurn(frame);
    if (frame.kind === "consult_brief") renderBrief(String(frame.consult_id || ""), frame.llm_response);
  }

  async function loadWaitingBriefs(fetchFn, doc) {
    fetchFn = fetchFn || global.fetch.bind(global);
    try {
      const resp = await fetchFn("/api/consults");
      if (!resp.ok) return 0;
      const data = await resp.json();
      const rows = (data && data.consults) || [];
      rows.slice().reverse().forEach(function (c) {
        if (c.brief_text) renderBrief(String(c.consult_id), c.brief_text, doc, fetchFn);
      });
      return rows.length;
    } catch (_e) {
      return 0;
    }
  }

  global.OrionConsult = { onFrame: onFrame, renderTurn: renderTurn, renderBrief: renderBrief, loadWaitingBriefs: loadWaitingBriefs };
  if (global.document && global.document.addEventListener) {
    global.document.addEventListener("DOMContentLoaded", function () { loadWaitingBriefs(); });
  }
})(typeof window !== "undefined" ? window : globalThis);
