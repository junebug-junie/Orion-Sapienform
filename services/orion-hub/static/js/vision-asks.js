(function (global) {
  // "Orion is asking" card in the Vision panel (walkway camera idea 3,
  // docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md).
  // Lists open questions from GET /api/asks and posts Juniper's answer or
  // dismissal to /api/asks/{id}/answer | /dismiss (scripts/ask_routes.py).
  //
  // Polls instead of listening on a socket: the Hub has no push channel for
  // panels, and the orion_ask table is the only truth either way.
  //
  // Everything is rendered with textContent / DOM attributes, never
  // innerHTML: question text and image refs come from the database.

  const POLL_MS = 60000;

  function askImageView(imageRef) {
    const ref = typeof imageRef === "string" ? imageRef.trim() : "";
    if (!ref) return { kind: "none", ref: "" };
    // Only a real web URL can be shown as a picture. Anything else (a path on
    // the vision host's disk, an embedding ref) is not something this Hub can
    // serve, so it is shown as text rather than a broken image.
    if (/^https?:\/\//i.test(ref)) return { kind: "img", ref: ref };
    return { kind: "text", ref: ref };
  }

  function askCardViewModel(ask) {
    const a = ask || {};
    return {
      askId: String(a.ask_id || ""),
      question: String(a.question || ""),
      image: askImageView(a.image_ref),
      askedAt: a.created_at ? String(a.created_at) : "",
      about: a.source_kind ? String(a.source_kind) + ": " + String(a.source_ref || "") : "",
    };
  }

  function statusLine(asks) {
    const n = Array.isArray(asks) ? asks.length : 0;
    if (n === 0) return "Orion has no open questions for you.";
    if (n === 1) return "Orion has 1 question for you.";
    return "Orion has " + n + " questions for you.";
  }

  function errorLine(status, detail) {
    if (status === 409) return "Someone already answered this one, or it expired.";
    if (status === 404) return "That question no longer exists.";
    if (status === 503) return "Can't reach the question store right now.";
    return "Something went wrong" + (detail ? ": " + detail : ".");
  }

  function el(doc, tag, cls, text) {
    const node = doc.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderAsk(doc, ask, onAction) {
    const vm = askCardViewModel(ask);
    const card = el(doc, "div", "rounded-lg border border-sky-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    if (vm.image.kind === "img") {
      const img = el(doc, "img", "max-h-48 rounded border border-gray-700");
      img.setAttribute("src", vm.image.ref);
      img.setAttribute("alt", "What Orion is asking about");
      card.appendChild(img);
    } else if (vm.image.kind === "text") {
      card.appendChild(el(doc, "div", "text-[11px] text-gray-500 break-all", "Picture: " + vm.image.ref));
    }
    if (vm.askedAt) card.appendChild(el(doc, "div", "text-[11px] text-gray-500", "Asked " + vm.askedAt));
    const row = el(doc, "div", "flex items-center gap-2");
    const input = el(doc, "input", "flex-1 bg-gray-900 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700");
    input.setAttribute("type", "text");
    input.setAttribute("maxlength", "500");
    input.setAttribute("placeholder", "Your answer");
    input.setAttribute("data-ask-input", vm.askId);
    const answerBtn = el(doc, "button", "text-xs bg-sky-700 hover:bg-sky-600 text-white rounded px-3 py-1", "Answer");
    answerBtn.setAttribute("type", "button");
    answerBtn.setAttribute("data-ask-action", "answer");
    const dismissBtn = el(doc, "button", "text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded px-3 py-1 border border-gray-700", "Dismiss");
    dismissBtn.setAttribute("type", "button");
    dismissBtn.setAttribute("data-ask-action", "dismiss");
    const note = el(doc, "div", "text-[11px] text-amber-300 hidden");
    answerBtn.addEventListener("click", function () {
      onAction(vm.askId, "answer", input.value, note);
    });
    dismissBtn.addEventListener("click", function () {
      onAction(vm.askId, "dismiss", "", note);
    });
    row.appendChild(input);
    row.appendChild(answerBtn);
    row.appendChild(dismissBtn);
    card.appendChild(row);
    card.appendChild(note);
    return card;
  }

  async function submitAction(fetchFn, askId, action, answerText) {
    const url = "/api/asks/" + encodeURIComponent(askId) + "/" + action;
    const init = { method: "POST", headers: { "Content-Type": "application/json" } };
    if (action === "answer") init.body = JSON.stringify({ answer: String(answerText || "").trim() });
    const resp = await fetchFn(url, init);
    let body = null;
    try {
      body = await resp.json();
    } catch (_e) {
      body = null;
    }
    return { ok: resp.ok, status: resp.status, body: body };
  }

  function mount(doc, fetchFn) {
    const list = doc.getElementById("visionAsksList");
    const status = doc.getElementById("visionAsksStatus");
    if (!list || !status) return null;

    async function refresh() {
      try {
        const resp = await fetchFn("/api/asks?status=open");
        if (!resp.ok) {
          status.textContent = errorLine(resp.status);
          return;
        }
        const data = await resp.json();
        const asks = (data && data.asks) || [];
        status.textContent = statusLine(asks);
        while (list.firstChild) list.removeChild(list.firstChild);
        asks.forEach(function (ask) {
          list.appendChild(renderAsk(doc, ask, onAction));
        });
      } catch (_e) {
        status.textContent = "Can't reach the Hub to load Orion's questions.";
      }
    }

    async function onAction(askId, action, answerText, note) {
      if (action === "answer" && !String(answerText || "").trim()) {
        note.textContent = "Type an answer first, or press Dismiss.";
        note.classList.remove("hidden");
        return;
      }
      try {
        const res = await submitAction(fetchFn, askId, action, answerText);
        if (!res.ok) {
          note.textContent = errorLine(res.status, res.body && res.body.detail);
          note.classList.remove("hidden");
          if (res.status !== 409 && res.status !== 404) return;
        }
      } catch (_e) {
        note.textContent = "Can't reach the Hub; your answer was not saved.";
        note.classList.remove("hidden");
        return;
      }
      await refresh();
    }

    refresh();
    const timer = global.setInterval(function () {
      if (!doc.hidden) refresh();
    }, POLL_MS);
    return { refresh: refresh, onAction: onAction, stop: function () { global.clearInterval(timer); } };
  }

  const api = {
    askImageView: askImageView,
    askCardViewModel: askCardViewModel,
    statusLine: statusLine,
    errorLine: errorLine,
    submitAction: submitAction,
    renderAsk: renderAsk,
    mount: mount,
  };

  global.OrionVisionAsks = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof document !== "undefined" && typeof global.fetch === "function") {
    const start = function () {
      api.mount(document, global.fetch.bind(global));
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
  }
})(typeof window !== "undefined" ? window : globalThis);
