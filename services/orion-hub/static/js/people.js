(function (global) {
  // "People Orion knows" panel (2026-10-22,
  // docs/superpowers/specs/2026-10-22-people-one-record-design.md section 5).
  // One card per person from GET /api/people (scripts/people_routes.py): picture,
  // days seen, what Juniper said about them, the open question about them.
  // Juniper can add what she knows, remove a fact, or say a name is not a person.
  // "Same person as..." (2026-10-29) joins two names of one person (a first name
  // and a full name, a nickname); the joined name shows as "Also called".
  //
  // Everything is rendered with textContent / attributes, never innerHTML:
  // names and facts come from the database.

  const THUMB_RE = /^thumb:([0-9a-f]{64})$/;
  const POLL_MS = 60000;

  function el(doc, tag, cls, text) {
    const node = doc.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function thumbUrl(ref) {
    const m = THUMB_RE.exec(String(ref || ""));
    return m ? "/api/vision/crop-thumbs/" + m[1] : "";
  }

  function seenLine(p) {
    const days = Number(p.days || 0);
    if (!days) return "Not seen on camera";
    return "Seen on camera on " + days + (days === 1 ? " day" : " days");
  }

  function statusLine(people) {
    const n = (people || []).filter(function (p) { return p.status === "active"; }).length;
    if (!n) return "Orion does not know anyone by name yet.";
    return "Orion knows " + n + (n === 1 ? " person" : " people") + " by name.";
  }

  function factLine(f) {
    return String(f.label || f.category || "") + ": " + String(f.value || "");
  }

  function renderPerson(doc, p, categories, onActionRaw, people) {
    const active = p.status === "active";
    const buttons = [];
    // Every button in the card is locked while a request is in flight: a double
    // tap cannot save the same fact twice.
    function onAction(personId, action, data) {
      buttons.forEach(function (b) { b.disabled = true; });
      return Promise.resolve(onActionRaw(personId, action, data)).finally(function () {
        buttons.forEach(function (b) { b.disabled = false; });
      });
    }
    const card = el(doc, "div", "rounded-lg border border-gray-700 bg-gray-800/60 p-3 space-y-2" + (active ? "" : " opacity-60"));
    card.setAttribute("data-person-id", String(p.person_id || ""));
    const head = el(doc, "div", "flex items-center gap-3");
    const url = thumbUrl(p.image_ref);
    if (url) {
      const img = el(doc, "img", "w-12 h-12 rounded object-cover bg-gray-900");
      img.setAttribute("src", url);
      img.setAttribute("alt", "Picture of " + String(p.name || "this person"));
      img.setAttribute("loading", "lazy");
      head.appendChild(img);
    }
    const title = el(doc, "div", "min-w-0");
    title.appendChild(el(doc, "div", "text-sm font-semibold text-gray-100", p.name));
    title.appendChild(el(doc, "div", "text-[11px] text-gray-400", active ? seenLine(p) : "Not a person (Orion will not ask about this name)"));
    if ((p.also_called || []).length) {
      title.appendChild(el(doc, "div", "text-[11px] text-gray-400", "Also called " + p.also_called.join(", ")));
    }
    head.appendChild(title);
    card.appendChild(head);

    const list = el(doc, "ul", "space-y-1");
    (p.facts || []).forEach(function (f) {
      const li = el(doc, "li", "flex items-start justify-between gap-2 text-xs text-gray-200");
      li.appendChild(el(doc, "span", "min-w-0", factLine(f)));
      const rm = el(doc, "button", "text-[11px] text-gray-400 hover:text-red-300", "Remove");
      rm.setAttribute("type", "button");
      rm.setAttribute("data-person-action", "remove-fact");
      buttons.push(rm);
      rm.setAttribute("aria-label", "Remove " + factLine(f));
      rm.addEventListener("click", function () { onAction(p.person_id, "remove-fact", { fact_id: f.fact_id }); });
      li.appendChild(rm);
      list.appendChild(li);
    });
    if (!(p.facts || []).length && active) {
      list.appendChild(el(doc, "li", "text-[11px] text-gray-500", "You have not told Orion anything about them yet."));
    }
    card.appendChild(list);
    if (p.open_question && active) {
      card.appendChild(el(doc, "div", "text-[11px] text-sky-300", "Orion is asking: " + String(p.open_question.question || "")));
    }

    const row = el(doc, "div", "flex flex-wrap items-center gap-2");
    if (active) {
      const select = el(doc, "select", "bg-gray-900 text-gray-100 text-xs rounded px-1 py-1 border border-gray-700");
      select.setAttribute("aria-label", "What kind of thing");
      select.setAttribute("data-person-field", "category");
      (categories || []).forEach(function (c) {
        const opt = el(doc, "option", "", c.label);
        opt.setAttribute("value", c.key);
        select.appendChild(opt);
      });
      const input = el(doc, "input", "flex-1 min-w-0 bg-gray-900 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700");
      input.setAttribute("type", "text");
      input.setAttribute("maxlength", "500");
      input.setAttribute("placeholder", "Tell Orion something about " + String(p.name || "them"));
      input.setAttribute("aria-label", "Tell Orion something about " + String(p.name || "them"));
      input.setAttribute("data-person-field", "value");
      const save = el(doc, "button", "text-xs bg-sky-700 hover:bg-sky-600 text-white rounded px-3 py-1", "Save");
      save.setAttribute("type", "button");
      save.setAttribute("data-person-action", "add-fact");
      buttons.push(save);
      save.addEventListener("click", function () {
        onAction(p.person_id, "add-fact", { category: select.value, value: input.value });
      });
      row.appendChild(select);
      row.appendChild(input);
      row.appendChild(save);
    }
    let sameRow = null;
    const others = (people || []).filter(function (o) { return o.person_id !== p.person_id && o.status === "active"; });
    if (active && others.length) {
      sameRow = el(doc, "div", "flex flex-wrap items-center gap-2");
      const sameSelect = el(doc, "select", "bg-gray-900 text-gray-100 text-xs rounded px-1 py-1 border border-gray-700");
      sameSelect.setAttribute("aria-label", "Same person as");
      sameSelect.setAttribute("data-person-field", "same-as");
      const none = el(doc, "option", "", "Same person as...");
      none.setAttribute("value", "");
      sameSelect.appendChild(none);
      others.forEach(function (o) {
        const opt = el(doc, "option", "", o.name);
        opt.setAttribute("value", o.person_id);
        sameSelect.appendChild(opt);
      });
      const join = el(doc, "button", "text-[11px] text-gray-300 hover:text-white border border-gray-700 rounded px-2 py-1", "Join");
      join.setAttribute("type", "button");
      join.setAttribute("data-person-action", "same-as");
      buttons.push(join);
      join.addEventListener("click", function () {
        if (!sameSelect.value) return;
        // The other card is kept: this name becomes "also called" there.
        onAction(p.person_id, "same-as", { into: sameSelect.value });
      });
      sameRow.appendChild(sameSelect);
      sameRow.appendChild(join);
    }
    const toggle = el(doc, "button", "text-[11px] text-gray-400 hover:text-gray-200 border border-gray-700 rounded px-2 py-1",
      active ? "Not a person" : "This is a person");
    toggle.setAttribute("type", "button");
    toggle.setAttribute("data-person-action", "status");
    buttons.push(toggle);
    toggle.addEventListener("click", function () {
      onAction(p.person_id, "status", { status: active ? "not_a_person" : "active" });
    });
    row.appendChild(toggle);
    card.appendChild(row);
    if (sameRow) card.appendChild(sameRow);
    return card;
  }

  function request(action, personId, data) {
    const base = "/api/people/" + encodeURIComponent(personId);
    if (action === "add-fact") return { url: base + "/facts", body: { category: data.category, value: String(data.value || "").trim() } };
    if (action === "remove-fact") return { url: base + "/facts/" + encodeURIComponent(data.fact_id) + "/remove", body: {} };
    if (action === "status") return { url: base + "/status", body: { status: data.status } };
    if (action === "same-as") return { url: base + "/same-as", body: { into: data.into || null } };
    return null;
  }

  function mount(doc, fetchFn) {
    const status = doc.getElementById("peopleStatus");
    const listEl = doc.getElementById("peopleList");
    if (!status || !listEl) return null;
    let typing = false;
    listEl.addEventListener("focusin", function () { typing = true; });
    listEl.addEventListener("focusout", function () { typing = false; });
    // Text typed but not saved yet: a poll must not wipe it.
    function hasUnsaved() {
      const inputs = listEl.querySelectorAll('[data-person-field="value"]');
      for (let i = 0; i < inputs.length; i++) if (String(inputs[i].value || "").trim()) return true;
      return false;
    }

    async function refresh() {
      try {
        const resp = await fetchFn("/api/people");
        const body = await resp.json();
        if (!resp.ok || !body || !body.ok) throw new Error("status " + resp.status);
        status.textContent = statusLine(body.people);
        while (listEl.firstChild) listEl.removeChild(listEl.firstChild);
        (body.people || []).forEach(function (p) {
          listEl.appendChild(renderPerson(doc, p, body.categories, onAction, body.people));
        });
      } catch (_e) {
        status.textContent = "Could not load the people Orion knows.";
      }
    }

    async function onAction(personId, action, data) {
      const req = request(action, personId, data || {});
      if (!req) return;
      if (action === "add-fact" && !req.body.value) return;
      let ok = false;
      try {
        const resp = await fetchFn(req.url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req.body) });
        ok = !!(resp && resp.ok);
      } catch (_e) {
        ok = false;
      }
      if (!ok) {
        // Keep what she typed on screen and say it did not save.
        status.textContent = "That did not save. Please try again.";
        return;
      }
      if (action === "add-fact") {
        const input = listEl.querySelector('[data-person-id="' + cssEscape(personId) + '"] [data-person-field="value"]');
        if (input) input.value = "";
      }
      typing = false;
      if (!hasUnsaved()) await refresh();
    }

    function cssEscape(value) {
      return global.CSS && global.CSS.escape ? global.CSS.escape(String(value)) : String(value).replace(/["\\]/g, "\\$&");
    }

    refresh();
    const timer = global.setInterval(function () {
      if (!doc.hidden && !typing && !hasUnsaved()) refresh();
    }, POLL_MS);
    return { refresh: refresh, onAction: onAction, stop: function () { global.clearInterval(timer); } };
  }

  const api = { thumbUrl: thumbUrl, seenLine: seenLine, statusLine: statusLine, factLine: factLine,
                request: request, renderPerson: renderPerson, mount: mount };
  global.OrionPeople = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document !== "undefined" && typeof global.fetch === "function") {
    const start = function () { api.mount(document, global.fetch.bind(global)); };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
  }
})(typeof window !== "undefined" ? window : globalThis);
