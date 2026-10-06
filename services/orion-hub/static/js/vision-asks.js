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
  //
  // Memory confirmation cards (2026-10-06, memory-episode spec sections 3/5):
  // a card whose source_kind is in RESOLVABLE_KINDS gets Confirm / Revise /
  // Reject instead of Answer / Dismiss, and posts to /api/asks/{id}/resolve.
  // Revise opens a box prefilled with the memory's current wording; it cannot
  // be sent empty. The panel is mounted at the top of the Hub home.
  const RESOLVABLE_KINDS = ["memory_confirmation", "open_question"];

  function isResolvable(ask) {
    return !!ask && RESOLVABLE_KINDS.indexOf(String(ask.source_kind || "")) >= 0;
  }

  const POLL_MS = 60000;

  // "thumb:<sha256>" is orion-vision-host's crop thumbnail (one embedded box,
  // never the patio). The Hub serves it by hash from a read-only mount.
  const THUMB_REF = /^thumb:([0-9a-f]{64})$/;

  function askImageView(imageRef) {
    const ref = typeof imageRef === "string" ? imageRef.trim() : "";
    if (!ref) return { kind: "none", ref: "" };
    const thumb = THUMB_REF.exec(ref);
    if (thumb) return { kind: "img", ref: "/api/vision/crop-thumbs/" + thumb[1] };
    // Otherwise only a real web URL can be shown as a picture. Anything else
    // (a path on the vision host's disk, an embedding ref) is not something
    // this Hub can serve, so it is shown as text rather than a broken image.
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
      resolvable: isResolvable(a),
      statement: a.memory_statement ? String(a.memory_statement) : "",
    };
  }

  function statusLine(asks) {
    const n = Array.isArray(asks) ? asks.length : 0;
    if (n === 0) return "Orion has no open questions for you.";
    if (n === 1) return "Orion has 1 question for you.";
    return "Orion has " + n + " questions for you.";
  }

  // Mirrors orion.memory.episode.confirmation.MIN_REVISION_WORDS / revision_problem: structural
  // checks only. A note that only says "no" belongs on Reject, which the hint says.
  const MIN_REVISION_WORDS = 6;

  function revisionProblem(note, current) {
    const words = String(note || "").split(/\s+/).filter(Boolean);
    if (words.length === 0) return "revised_needs_note";
    if (words.length < MIN_REVISION_WORDS) return "revised_too_short";
    const norm = function (t) { return String(t || "").split(/\s+/).filter(Boolean).join(" ").toLowerCase(); };
    if (current && norm(note) === norm(current)) return "revised_unchanged";
    return null;
  }

  function errorLine(status, detail) {
    if (detail === "revised_needs_note") return "Write how I should remember it first.";
    if (detail === "revised_too_short") return "Write it as a full sentence (at least " + MIN_REVISION_WORDS + " words). To drop it, press Reject.";
    if (detail === "revised_unchanged") return "That is the same as what I have. Change it, or press Confirm.";
    if (status === 409) return "Someone already answered this one, or it expired.";
    if (status === 404) return "That question no longer exists.";
    if (status === 503) return "Can't reach the question store right now.";
    const text = typeof detail === "string" ? detail : "";
    return "Something went wrong" + (text ? ": " + text : ".");
  }

  function el(doc, tag, cls, text) {
    const node = doc.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderResolvable(doc, vm, onAction) {
    const card = el(doc, "div", "rounded-lg border border-violet-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.setAttribute("data-ask-kind", "resolvable");
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    if (vm.askedAt) card.appendChild(el(doc, "div", "text-[11px] text-gray-500", "Asked " + vm.askedAt));
    const row = el(doc, "div", "flex items-center gap-2");
    function btn(label, action, cls) {
      const b = el(doc, "button", "text-xs rounded px-3 py-1 " + cls, label);
      b.setAttribute("type", "button");
      b.setAttribute("data-ask-action", action);
      return b;
    }
    const confirmBtn = btn("Confirm", "confirm", "bg-emerald-700 hover:bg-emerald-600 text-white");
    const reviseBtn = btn("Revise", "revise", "bg-sky-700 hover:bg-sky-600 text-white");
    const rejectBtn = btn("Reject", "reject", "bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700");
    const revise = el(doc, "div", "space-y-1 hidden");
    revise.setAttribute("data-ask-revise", vm.askId);
    // The hidden attribute as well as the class: hidden must not depend on the stylesheet.
    revise.hidden = true;
    revise.appendChild(el(doc, "div", "text-[11px] text-gray-400",
      "How should I remember it? Rewrite it in full. If it should not be kept at all, press Reject instead."));
    const box = el(doc, "textarea", "w-full bg-gray-900 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700");
    box.setAttribute("maxlength", "500");
    box.setAttribute("rows", "3");
    box.setAttribute("data-ask-input", vm.askId);
    const saveBtn = btn("Save revision", "save-revision", "bg-sky-700 hover:bg-sky-600 text-white");
    revise.appendChild(box);
    revise.appendChild(saveBtn);
    const note = el(doc, "div", "text-[11px] text-amber-300 hidden");
    const buttons = [confirmBtn, reviseBtn, rejectBtn, saveBtn];
    function run(resolution, text) {
      buttons.forEach(function (b) { b.disabled = true; });
      return Promise.resolve(onAction(vm.askId, "resolve", text, note, resolution)).finally(function () {
        buttons.forEach(function (b) { b.disabled = false; });
      });
    }
    confirmBtn.addEventListener("click", function () { return run("confirmed", ""); });
    rejectBtn.addEventListener("click", function () { return run("rejected", ""); });
    reviseBtn.addEventListener("click", function () {
      revise.classList.remove("hidden");
      revise.hidden = false;
      if (!String(box.value || "").trim()) box.value = vm.statement;
      if (box.focus) box.focus();
    });
    saveBtn.addEventListener("click", function () {
      const problem = revisionProblem(box.value, vm.statement);
      if (problem) {
        note.textContent = errorLine(422, problem);
        note.classList.remove("hidden");
        return;
      }
      return run("revised", box.value);
    });
    row.appendChild(confirmBtn);
    row.appendChild(reviseBtn);
    row.appendChild(rejectBtn);
    card.appendChild(row);
    card.appendChild(revise);
    card.appendChild(note);
    return card;
  }

  function renderAsk(doc, ask, onAction) {
    const vm = askCardViewModel(ask);
    if (vm.resolvable) return renderResolvable(doc, vm, onAction);
    const card = el(doc, "div", "rounded-lg border border-sky-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    if (vm.image.kind === "img") {
      const img = el(doc, "img", "max-h-48 rounded border border-gray-700");
      img.setAttribute("src", vm.image.ref);
      img.setAttribute("alt", "What Orion is asking about");
      // A pruned or missing thumbnail must not leave a broken-image icon.
      img.addEventListener("error", function () {
        img.hidden = true;
      });
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
    // Both buttons are locked while a request is in flight, so a double
    // click cannot send a second POST and flash a spurious 409.
    const buttons = [answerBtn, dismissBtn];
    function run(action, text) {
      buttons.forEach(function (b) { b.disabled = true; });
      return Promise.resolve(onAction(vm.askId, action, text, note)).finally(function () {
        buttons.forEach(function (b) { b.disabled = false; });
      });
    }
    answerBtn.addEventListener("click", function () {
      return run("answer", input.value);
    });
    dismissBtn.addEventListener("click", function () {
      return run("dismiss", "");
    });
    row.appendChild(input);
    row.appendChild(answerBtn);
    row.appendChild(dismissBtn);
    card.appendChild(row);
    card.appendChild(note);
    return card;
  }

  async function submitAction(fetchFn, askId, action, answerText, resolution) {
    const url = "/api/asks/" + encodeURIComponent(askId) + "/" + action;
    const init = { method: "POST", headers: { "Content-Type": "application/json" } };
    if (action === "answer") init.body = JSON.stringify({ answer: String(answerText || "").trim() });
    if (action === "resolve") {
      init.body = JSON.stringify({ resolution: String(resolution || ""), note: String(answerText || "").trim() });
    }
    const resp = await fetchFn(url, init);
    let body = null;
    try {
      body = await resp.json();
    } catch (_e) {
      body = null;
    }
    return { ok: resp.ok, status: resp.status, body: body };
  }

  // At most this many questions show at once; the rest scroll inside the list so
  // a long queue cannot push the chat off the top of the Hub.
  const MAX_VISIBLE_ASKS = 2;

  // Pixel height of the first `max` cards plus the gaps between them, or null
  // when everything already fits (no cap needed).
  function visibleAsksMaxHeight(heights, gap, max) {
    const limit = max == null ? MAX_VISIBLE_ASKS : max;
    if (!heights || heights.length <= limit) return null;
    let total = 0;
    for (let i = 0; i < limit; i++) total += Number(heights[i]) || 0;
    return total + (Number(gap) || 0) * (limit - 1);
  }

  function capVisibleAsks(win, list) {
    const kids = Array.prototype.slice.call(list.children);
    const gap = parseFloat(win.getComputedStyle(list).rowGap) || 0;
    const px = visibleAsksMaxHeight(kids.map(function (k) { return k.offsetHeight; }), gap);
    list.style.maxHeight = px == null ? "" : Math.ceil(px) + "px";
  }

  function mount(doc, fetchFn) {
    const list = doc.getElementById("visionAsksList");
    const status = doc.getElementById("visionAsksStatus");
    if (!list || !status) return null;

    // True while Juniper is mid-answer: a box has focus or unsent text. The
    // timed poll skips then, because refresh() rebuilds every card and would
    // wipe what she is typing.
    function isTyping() {
      const active = doc.activeElement;
      if (active && active.getAttribute && active.getAttribute("data-ask-input")) return true;
      return hasText(list);
    }

    function hasText(node) {
      if (!node) return false;
      if (node.getAttribute && node.getAttribute("data-ask-input") && String(node.value || "").trim()) return true;
      const kids = node.children || [];
      for (let i = 0; i < kids.length; i++) if (hasText(kids[i])) return true;
      return false;
    }

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
        if (doc.defaultView) capVisibleAsks(doc.defaultView, list);
      } catch (_e) {
        status.textContent = "Can't reach the Hub to load Orion's questions.";
      }
    }

    async function onAction(askId, action, answerText, note, resolution) {
      if (action === "answer" && !String(answerText || "").trim()) {
        note.textContent = "Type an answer first, or press Dismiss.";
        note.classList.remove("hidden");
        return;
      }
      if (action === "resolve" && resolution === "revised" && !String(answerText || "").trim()) {
        note.textContent = errorLine(422, "revised_needs_note");
        note.classList.remove("hidden");
        return;
      }
      try {
        const res = await submitAction(fetchFn, askId, action, answerText, resolution);
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
      if (!doc.hidden && !isTyping()) refresh();
    }, POLL_MS);
    return { refresh: refresh, onAction: onAction, isTyping: isTyping, stop: function () { global.clearInterval(timer); } };
  }

  const api = {
    askImageView: askImageView,
    askCardViewModel: askCardViewModel,
    statusLine: statusLine,
    errorLine: errorLine,
    isResolvable: isResolvable,
    revisionProblem: revisionProblem,
    submitAction: submitAction,
    renderAsk: renderAsk,
    mount: mount,
    visibleAsksMaxHeight: visibleAsksMaxHeight,
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
