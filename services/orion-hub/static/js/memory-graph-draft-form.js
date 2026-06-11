/* Orion Hub — human-readable SuggestDraftV1 form editor (syncs with JSON textarea). */
(function () {
  const ENTITY_KINDS = ["person", "animal", "pet", "breed", "taxon", "collective", "abstract"];
  const AFFECT_LABELS = [
    "anger",
    "annoyance",
    "affection",
    "fear",
    "trust",
    "distrust",
    "neutral",
    "ambivalence",
    "none",
  ];
  const TIME_QUAL = ["today", "yesterday", "last_week", "recent", "unknown"];
  const TRUST_POL = ["trust", "distrust", "ambivalent", "unknown"];
  const EDGE_PREDS = [
    "orionmem:inSituation",
    "orionmem:stimulusEntity",
    "orionmem:aboutEntity",
    "orionmem:targetOfNegativeAffect",
    "orionmem:generalizationOf",
    "orionmem:contradictsSituation",
    "prov:wasDerivedFrom",
    "schema:about",
  ];
  const PARTICIPANT_ROLES = ["agent", "patient", "topic", "observer", "participant"];
  const MEMORY_CONFIDENCE = ["certain", "likely", "possible", "uncertain"];
  const MEMORY_SENSITIVITY = ["public", "private", "intimate"];
  const MEMORY_PRIORITY = ["always_inject", "high_recall", "episodic_detail", "archival"];
  const MEMORY_PROVENANCE = ["operator_highlight", "operator_distiller", "auto_extractor", "imported"];
  const MEMORY_VISIBILITY = ["chat", "social", "intimate", "all"];
  const MEMORY_TIME_KIND = ["timeless", "era_bound", "current", "expiring"];
  const USER_SPEAKER_LABEL = "Juniper";

  function localDateISO(d) {
    const dt = d instanceof Date ? d : new Date();
    const y = dt.getFullYear();
    const m = String(dt.getMonth() + 1).padStart(2, "0");
    const day = String(dt.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }

  function occurredAtNeedsToday(value) {
    const s = String(value == null ? "" : value).trim();
    if (!s || s.toLowerCase() === "null" || s.toLowerCase() === "none") return true;
    const m = /^(\d{4})-\d{2}-\d{2}$/.exec(s);
    if (!m) return true;
    const year = parseInt(m[1], 10);
    return year < new Date().getFullYear() - 1;
  }

  function normalizeUserSpeakerLabel(label) {
    const s = String(label || "").trim();
    if (!s) return USER_SPEAKER_LABEL;
    if (s.toLowerCase() === "user" || s.toLowerCase() === "the user" || s.toLowerCase() === "operator") {
      return USER_SPEAKER_LABEL;
    }
    return s;
  }

  function normalizeDraftForEditor(obj) {
    if (!obj || typeof obj !== "object") return obj;
    const out = { ...obj };
    out.entities = (Array.isArray(obj.entities) ? obj.entities : []).map((ent) => {
      if (!ent || typeof ent !== "object") return ent;
      return { ...ent, label: normalizeUserSpeakerLabel(ent.label) };
    });
    const today = localDateISO();
    out.situations = (Array.isArray(obj.situations) ? obj.situations : []).map((sit) => {
      if (!sit || typeof sit !== "object") return sit;
      const row = { ...sit };
      if (occurredAtNeedsToday(row.occurredAt)) {
        row.occurredAt = today;
        const tq = String(row.timeQualitative || "").trim().toLowerCase();
        if (!tq || tq === "unknown") row.timeQualitative = "today";
      }
      return row;
    });
    return out;
  }

  function defaultCardProjectionDefaults() {
    return {
      confidence: "likely",
      sensitivity: "private",
      priority: "episodic_detail",
      provenance: "operator_highlight",
      visibility_scope: ["chat"],
      summary: "",
      still_true: [],
      evidence: [],
      time_horizon: { kind: "timeless", start: null, end: null, as_of: null },
    };
  }

  function debounce(fn, ms) {
    let t = null;
    return function debounced() {
      const args = arguments;
      const self = this;
      clearTimeout(t);
      t = setTimeout(() => fn.apply(self, args), ms);
    };
  }

  function newEntityId() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      return `urn:uuid:${crypto.randomUUID()}`;
    }
    return `urn:uuid:${Date.now()}-draft`;
  }

  function normalizeRef(value) {
    const s = String(value == null ? "" : value).trim();
    if (!s || s.toLowerCase() === "null" || s.toLowerCase() === "none") return null;
    return s;
  }

  function splitCsv(s) {
    return String(s || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
  }

  function joinCsv(arr) {
    return (Array.isArray(arr) ? arr : [])
      .map((x) => String(x || "").trim())
      .filter(Boolean)
      .join(", ");
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function fieldRow(labelText, control) {
    const row = el("label", "block space-y-0.5");
    const lab = el("span", "text-[10px] text-gray-500 uppercase tracking-wide");
    lab.textContent = labelText;
    row.appendChild(lab);
    row.appendChild(control);
    return row;
  }

  function textInput(value, placeholder) {
    const inp = document.createElement("input");
    inp.type = "text";
    inp.className =
      "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100 text-[11px]";
    inp.value = value != null ? String(value) : "";
    if (placeholder) inp.placeholder = placeholder;
    return inp;
  }

  function textArea(value, rows) {
    const ta = document.createElement("textarea");
    ta.className =
      "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100 text-[11px] font-mono leading-relaxed";
    ta.rows = rows || 2;
    ta.value = value != null ? String(value) : "";
    return ta;
  }

  function selectInput(value, options) {
    const sel = document.createElement("select");
    sel.className =
      "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100 text-[11px]";
    options.forEach((opt) => {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      if (String(value) === opt) o.selected = true;
      sel.appendChild(o);
    });
    return sel;
  }

  function dateInput(value) {
    const inp = document.createElement("input");
    inp.type = "date";
    inp.className =
      "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100 text-[11px]";
    const s = String(value == null ? "" : value).trim();
    inp.value = /^\d{4}-\d{2}-\d{2}$/.test(s) && !occurredAtNeedsToday(s) ? s : localDateISO();
    return inp;
  }

  function entityOptionLabel(ent) {
    const label = String((ent && ent.label) || "?").trim();
    const id = String((ent && ent.id) || "").trim();
    if (!id) return label;
    const short = id.length > 28 ? `${id.slice(0, 24)}…` : id;
    return `${label} — ${short}`;
  }

  function entitySelect(entities, selectedId, { allowEmpty = true } = {}) {
    const sel = document.createElement("select");
    sel.className =
      "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100 text-[11px] font-mono";
    if (allowEmpty) {
      const none = document.createElement("option");
      none.value = "";
      none.textContent = "(none)";
      sel.appendChild(none);
    }
    (Array.isArray(entities) ? entities : []).forEach((ent) => {
      if (!ent || !ent.id) return;
      const o = document.createElement("option");
      o.value = String(ent.id);
      o.textContent = entityOptionLabel(ent);
      if (String(selectedId || "").trim() === o.value) o.selected = true;
      sel.appendChild(o);
    });
    const sid = String(selectedId || "").trim();
    if (sid && !Array.from(sel.options).some((o) => o.value === sid)) {
      const orphan = document.createElement("option");
      orphan.value = sid;
      orphan.textContent = `Unknown entity — ${sid.slice(0, 24)}…`;
      orphan.selected = true;
      sel.appendChild(orphan);
    }
    return sel;
  }

  function entityMultiSelect(entities, selectedIds) {
    const sel = document.createElement("select");
    sel.multiple = true;
    sel.className =
      "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100 text-[11px] font-mono";
    const chosen = new Set((Array.isArray(selectedIds) ? selectedIds : []).map((x) => String(x || "").trim()));
    const ents = Array.isArray(entities) ? entities : [];
    sel.size = Math.min(5, Math.max(2, ents.length));
    ents.forEach((ent) => {
      if (!ent || !ent.id) return;
      const o = document.createElement("option");
      o.value = String(ent.id);
      o.textContent = entityOptionLabel(ent);
      if (chosen.has(o.value)) o.selected = true;
      sel.appendChild(o);
    });
    chosen.forEach((id) => {
      if (!id || Array.from(sel.options).some((o) => o.value === id)) return;
      const orphan = document.createElement("option");
      orphan.value = id;
      orphan.textContent = `Unknown — ${id.slice(0, 24)}…`;
      orphan.selected = true;
      sel.appendChild(orphan);
    });
    return sel;
  }

  function selectedOptionValues(sel) {
    if (!sel) return [];
    return Array.from(sel.selectedOptions || [])
      .map((o) => String(o.value || "").trim())
      .filter(Boolean);
  }

  function entityLabelById(entities, id) {
    const sid = String(id || "").trim();
    if (!sid) return "";
    const ents = Array.isArray(entities) ? entities : [];
    for (let i = 0; i < ents.length; i += 1) {
      const e = ents[i];
      if (e && String(e.id) === sid) return String(e.label || e.id);
    }
    return sid.slice(0, 24);
  }

  function participantsToText(parts) {
    if (!Array.isArray(parts)) return "";
    return parts
      .map((p) => {
        if (!p || typeof p !== "object") return "";
        const role = String(p.role || "").trim();
        const eid = String(p.entity_id || p.entityId || "").trim();
        if (!role && !eid) return "";
        return `${role || "participant"}: ${eid}`;
      })
      .filter(Boolean)
      .join("\n");
  }

  function participantsFromText(text) {
    return String(text || "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const idx = line.indexOf(":");
        if (idx < 0) return { entity_id: line, role: "participant" };
        return {
          role: line.slice(0, idx).trim() || "participant",
          entity_id: line.slice(idx + 1).trim(),
        };
      });
  }

  function emptyDraftFromUi(ui) {
    if (ui && typeof ui.emptySuggestDraft === "function") return ui.emptySuggestDraft();
    return {
      ontology_version: "orionmem-2026-05",
      utterance_ids: [],
      entities: [],
      situations: [],
      edges: [],
      dispositions: [],
      utterance_text_by_id: {},
    };
  }

  /**
   * @param {object} options
   * @param {HTMLTextAreaElement} options.draftTextarea
   * @param {HTMLElement} options.formHost
   * @param {() => void} [options.onDraftChange]
   */
  function attachFormEditor(options) {
    const draftTextarea = options.draftTextarea;
    const formHost = options.formHost;
    const onDraftChange = options.onDraftChange || function () {};
    const ui = window.OrionMemoryGraphDraftUI || {};
    const parseFn =
      typeof ui.parseMemoryGraphDraftJson === "function"
        ? ui.parseMemoryGraphDraftJson
        : null;
    const looksDraft =
      typeof ui.looksLikeMemoryGraphDraftObject === "function"
        ? ui.looksLikeMemoryGraphDraftObject
        : function () {
            return false;
          };

    let syncing = false;
    let draftObj = null;
    let cardDefaults = defaultCardProjectionDefaults();

    function readCardDefaults() {
      const root = formHost;
      const cardMeta = root && root.querySelector("[data-card-meta-root]");
      if (!cardMeta) return { ...cardDefaults };
      const get = (name) => {
        const elNode = cardMeta.querySelector(`[data-card-meta="${name}"]`);
        return elNode ? String(elNode.value || "").trim() : "";
      };
      const vis = [];
      cardMeta.querySelectorAll("input[data-card-vis]").forEach((cb) => {
        if (cb.checked) vis.push(cb.getAttribute("data-card-vis"));
      });
      let evidence = [];
      const evidenceRaw = get("evidence");
      if (evidenceRaw) {
        try {
          const parsed = JSON.parse(evidenceRaw);
          if (Array.isArray(parsed)) evidence = parsed;
        } catch (_) {
          /* ignore until save */
        }
      }
      const stillLines = get("still_true")
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
      const thKind = get("time_horizon.kind") || "timeless";
      return {
        confidence: get("confidence") || "likely",
        sensitivity: get("sensitivity") || "private",
        priority: get("priority") || "episodic_detail",
        provenance: get("provenance") || "operator_highlight",
        visibility_scope: vis.length ? vis : ["chat"],
        summary: get("summary") || "",
        still_true: stillLines,
        evidence,
        time_horizon: {
          kind: thKind,
          start: get("time_horizon.start") || null,
          end: get("time_horizon.end") || null,
          as_of: get("time_horizon.as_of") || null,
        },
      };
    }

    function setCardDefaults(next) {
      cardDefaults = { ...defaultCardProjectionDefaults(), ...(next && typeof next === "object" ? next : {}) };
    }

    function readTextarea() {
      if (!draftTextarea || !parseFn) return { ok: false, object: null };
      return parseFn(draftTextarea.value);
    }

    function writeTextarea(obj) {
      if (!draftTextarea) return;
      syncing = true;
      draftTextarea.value = JSON.stringify(obj, null, 2);
      syncing = false;
      draftObj = obj;
      onDraftChange();
      try {
        draftTextarea.dispatchEvent(new Event("input", { bubbles: true }));
      } catch (_) {
        /* ignore */
      }
    }

    function collectDraftFromForm() {
      const base = draftObj && typeof draftObj === "object" ? { ...draftObj } : emptyDraftFromUi(ui);
      const root = formHost;
      if (!root) return base;

      const uidInp = root.querySelector("[data-draft-field=utterance_ids]");
      if (uidInp) base.utterance_ids = splitCsv(uidInp.value);

      base.entities = [];
      root.querySelectorAll("[data-entity-card]").forEach((card) => {
        base.entities.push({
          id: card.querySelector("[data-ent=id]")?.value?.trim() || newEntityId(),
          label: card.querySelector("[data-ent=label]")?.value?.trim() || "",
          entityKind: card.querySelector("[data-ent=entityKind]")?.value || "abstract",
          surfaceForms: splitCsv(card.querySelector("[data-ent=surfaceForms]")?.value),
          generalizes_to: normalizeRef(card.querySelector("[data-ent=generalizes_to]")?.value),
        });
      });

      base.situations = [];
      root.querySelectorAll("[data-situation-card]").forEach((card) => {
        const occurredInp = card.querySelector("[data-sit=occurredAt]");
        const occurred = occurredInp ? String(occurredInp.value || "").trim() : "";
        const aboutSel = card.querySelector("[data-sit=about_entity_ids]");
        const targetSel = card.querySelector("[data-sit=target_entity_ids]");
        const participants = [];
        card.querySelectorAll("[data-participant-row]").forEach((prow) => {
          const eid = prow.querySelector("[data-part=entity_id]")?.value?.trim();
          const role = prow.querySelector("[data-part=role]")?.value?.trim() || "participant";
          if (eid) participants.push({ entity_id: eid, role });
        });
        base.situations.push({
          id: card.querySelector("[data-sit=id]")?.value?.trim() || newEntityId(),
          utterance_ids: splitCsv(card.querySelector("[data-sit=utterance_ids]")?.value),
          label: card.querySelector("[data-sit=label]")?.value?.trim() || "",
          stimulus_entity_id: normalizeRef(card.querySelector("[data-sit=stimulus_entity_id]")?.value),
          about_entity_ids: aboutSel ? selectedOptionValues(aboutSel) : splitCsv(card.querySelector("[data-sit=about_entity_ids]")?.value),
          target_entity_ids: targetSel ? selectedOptionValues(targetSel) : splitCsv(card.querySelector("[data-sit=target_entity_ids]")?.value),
          affectLabel: card.querySelector("[data-sit=affectLabel]")?.value || "neutral",
          timeQualitative: card.querySelector("[data-sit=timeQualitative]")?.value || "unknown",
          occurredAt: normalizeRef(occurred) || localDateISO(),
          participants,
        });
      });

      base.edges = [];
      root.querySelectorAll("[data-edge-row]").forEach((row) => {
        const confRaw = row.querySelector("[data-edge=confidence]")?.value;
        let confidence = 0.8;
        const c = parseFloat(confRaw);
        if (!Number.isNaN(c)) confidence = c;
        base.edges.push({
          s: row.querySelector("[data-edge=s]")?.value?.trim() || "",
          p: row.querySelector("[data-edge=p]")?.value || "schema:about",
          o: row.querySelector("[data-edge=o]")?.value?.trim() || "",
          confidence,
        });
      });

      base.dispositions = [];
      root.querySelectorAll("[data-disposition-card]").forEach((card) => {
        const idVal = card.querySelector("[data-disp=id]")?.value?.trim();
        base.dispositions.push({
          id: idVal || null,
          holder_id: card.querySelector("[data-disp=holder_id]")?.value?.trim() || "",
          target_id: card.querySelector("[data-disp=target_id]")?.value?.trim() || "",
          trustPolarity: card.querySelector("[data-disp=trustPolarity]")?.value || "unknown",
          description: card.querySelector("[data-disp=description]")?.value?.trim() || "",
        });
      });

      return base;
    }

    const debouncedWrite = debounce(() => {
      if (syncing) return;
      writeTextarea(collectDraftFromForm());
    }, 180);

    function bindInput(node) {
      if (!node) return;
      node.addEventListener("input", debouncedWrite);
      node.addEventListener("change", debouncedWrite);
    }

    function bindCardMeta(node) {
      if (!node) return;
      const sync = () => {
        cardDefaults = readCardDefaults();
      };
      node.addEventListener("input", sync);
      node.addEventListener("change", sync);
    }

    function renderParticipantRows(container, participants, entities) {
      container.innerHTML = "";
      const parts = Array.isArray(participants) ? participants : [];
      const addRow = (part) => {
        const row = el("div", "flex flex-wrap gap-2 items-end border border-gray-800/60 rounded p-1");
        row.dataset.participantRow = "1";
        const roleSel = selectInput(part && part.role ? part.role : "participant", PARTICIPANT_ROLES);
        roleSel.dataset.part = "role";
        const entSel = entitySelect(entities, part && part.entity_id, { allowEmpty: false });
        entSel.dataset.part = "entity_id";
        [roleSel, entSel].forEach(bindInput);
        row.appendChild(fieldRow("Role", roleSel));
        row.appendChild(fieldRow("Entity", entSel));
        const rm = el("button", "text-[10px] text-red-300/90 shrink-0");
        rm.type = "button";
        rm.textContent = "Remove";
        rm.addEventListener("click", () => {
          row.remove();
          debouncedWrite();
        });
        row.appendChild(rm);
        container.appendChild(row);
      };
      if (!parts.length) addRow({ role: "agent", entity_id: "" });
      else parts.forEach((p) => addRow(p));
      const addBtn = el("button", "text-[10px] text-indigo-300/90 mt-1");
      addBtn.type = "button";
      addBtn.textContent = "+ Participant";
      addBtn.addEventListener("click", () => {
        addRow({ role: "participant", entity_id: "" });
        debouncedWrite();
      });
      container.appendChild(addBtn);
    }

    function renderCardDefaultsSection(defaults) {
      const d = { ...defaultCardProjectionDefaults(), ...(defaults || {}) };
      const th = d.time_horizon && typeof d.time_horizon === "object" ? d.time_horizon : {};
      const wrap = el("div", "space-y-2 border border-indigo-900/40 rounded-lg p-2 bg-indigo-950/20");
      wrap.dataset.cardMetaRoot = "1";
      wrap.appendChild(
        el(
          "div",
          "text-[11px] font-semibold text-indigo-200",
          "Card metadata (applied on Approve to every projected card)",
        ),
      );
      const grid = el("div", "grid grid-cols-2 gap-2");
      const confSel = selectInput(d.confidence, MEMORY_CONFIDENCE);
      confSel.dataset.cardMeta = "confidence";
      const sensSel = selectInput(d.sensitivity, MEMORY_SENSITIVITY);
      sensSel.dataset.cardMeta = "sensitivity";
      const priSel = selectInput(d.priority, MEMORY_PRIORITY);
      priSel.dataset.cardMeta = "priority";
      const provSel = selectInput(d.provenance, MEMORY_PROVENANCE);
      provSel.dataset.cardMeta = "provenance";
      [confSel, sensSel, priSel, provSel].forEach(bindCardMeta);
      grid.appendChild(fieldRow("Confidence", confSel));
      grid.appendChild(fieldRow("Sensitivity", sensSel));
      grid.appendChild(fieldRow("Priority", priSel));
      grid.appendChild(fieldRow("Provenance", provSel));
      wrap.appendChild(grid);
      const visWrap = el("div", "flex flex-wrap gap-2");
      const selectedVis = new Set(Array.isArray(d.visibility_scope) ? d.visibility_scope : ["chat"]);
      MEMORY_VISIBILITY.forEach((v) => {
        const lab = el("label", "inline-flex items-center gap-1 text-[10px] text-gray-300");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.dataset.cardVis = v;
        cb.checked = selectedVis.has(v);
        cb.addEventListener("change", () => {
          cardDefaults = readCardDefaults();
        });
        lab.appendChild(cb);
        lab.appendChild(document.createTextNode(v));
        visWrap.appendChild(lab);
      });
      wrap.appendChild(fieldRow("Visibility", visWrap));
      const sumTa = textArea(d.summary || "", 2);
      sumTa.dataset.cardMeta = "summary";
      sumTa.placeholder = "Optional summary override (leave empty to use situation text)";
      const stillTa = textArea((d.still_true || []).join("\n"), 2);
      stillTa.dataset.cardMeta = "still_true";
      stillTa.placeholder = "Still true (one per line)";
      const evTa = textArea(JSON.stringify(d.evidence || [], null, 2), 3);
      evTa.dataset.cardMeta = "evidence";
      evTa.placeholder = '[{"source":"…","excerpt":"…"}]';
      [sumTa, stillTa, evTa].forEach(bindCardMeta);
      wrap.appendChild(fieldRow("Summary override", sumTa));
      wrap.appendChild(fieldRow("Still true", stillTa));
      wrap.appendChild(fieldRow("Evidence JSON", evTa));
      const thGrid = el("div", "grid grid-cols-2 gap-2");
      const thKindSel = selectInput(th.kind || "timeless", MEMORY_TIME_KIND);
      thKindSel.dataset.cardMeta = "time_horizon.kind";
      const thStart = textInput(th.start || "", "YYYY-MM-DD");
      thStart.dataset.cardMeta = "time_horizon.start";
      const thEnd = textInput(th.end || "", "YYYY-MM-DD");
      thEnd.dataset.cardMeta = "time_horizon.end";
      const thAsOf = textInput(th.as_of || "", "YYYY-MM-DD");
      thAsOf.dataset.cardMeta = "time_horizon.as_of";
      [thKindSel, thStart, thEnd, thAsOf].forEach(bindCardMeta);
      thGrid.appendChild(fieldRow("Time kind", thKindSel));
      thGrid.appendChild(fieldRow("Time start", thStart));
      thGrid.appendChild(fieldRow("Time end", thEnd));
      thGrid.appendChild(fieldRow("Time as of", thAsOf));
      wrap.appendChild(fieldRow("Card time horizon", thGrid));
      return wrap;
    }

    function sectionHeader(title, addLabel, onAdd) {
      const head = el("div", "flex items-center justify-between gap-2 mt-3 mb-1");
      head.appendChild(el("div", "text-[11px] font-semibold text-gray-300", title));
      if (onAdd) {
        const btn = el("button", "px-2 py-0.5 rounded border border-gray-600 bg-gray-800 text-[10px] text-gray-200");
        btn.type = "button";
        btn.textContent = addLabel || "+ Add";
        btn.addEventListener("click", onAdd);
        head.appendChild(btn);
      }
      return head;
    }

    function renderEntityCard(ent, index) {
      const card = el("div", "border border-gray-800 rounded-lg p-2 space-y-2 bg-gray-950/40");
      card.dataset.entityCard = String(index);
      const title = el("div", "text-[11px] text-indigo-200 font-medium");
      title.textContent = `Entity: ${ent.label || ent.id || `#${index + 1}`}`;
      card.appendChild(title);
      const idInp = textInput(ent.id, "urn:uuid:…");
      idInp.dataset.ent = "id";
      const labelInp = textInput(ent.label, "display label");
      labelInp.dataset.ent = "label";
      const kindSel = selectInput(ent.entityKind || "abstract", ENTITY_KINDS);
      kindSel.dataset.ent = "entityKind";
      const sfInp = textInput(joinCsv(ent.surfaceForms), "surface forms, comma-separated");
      sfInp.dataset.ent = "surfaceForms";
      const genInp = textInput(ent.generalizes_to || "", "generalizes_to entity id");
      genInp.dataset.ent = "generalizes_to";
      [idInp, labelInp, kindSel, sfInp, genInp].forEach(bindInput);
      card.appendChild(fieldRow("ID", idInp));
      card.appendChild(fieldRow("Label", labelInp));
      card.appendChild(fieldRow("Kind", kindSel));
      card.appendChild(fieldRow("Surface forms", sfInp));
      card.appendChild(fieldRow("Generalizes to", genInp));
      const rm = el("button", "text-[10px] text-red-300/90 hover:text-red-200");
      rm.type = "button";
      rm.textContent = "Remove entity";
      rm.addEventListener("click", () => {
        card.remove();
        debouncedWrite();
      });
      card.appendChild(rm);
      return card;
    }

    function renderSituationCard(sit, index, entities) {
      const card = el("div", "border border-gray-800 rounded-lg p-2 space-y-2 bg-gray-950/40");
      card.dataset.situationCard = String(index);
      const title = el("div", "text-[11px] text-emerald-200 font-medium");
      title.textContent = `Situation: ${sit.label || sit.id || `#${index + 1}`}`;
      card.appendChild(title);
      const idInp = textInput(sit.id, "urn:uuid:…");
      idInp.dataset.sit = "id";
      const uidInp = textInput(joinCsv(sit.utterance_ids), "turn ids");
      uidInp.dataset.sit = "utterance_ids";
      const labelInp = textInput(sit.label, "short description");
      labelInp.dataset.sit = "label";
      const stimSel = entitySelect(entities, sit.stimulus_entity_id || "");
      stimSel.dataset.sit = "stimulus_entity_id";
      const aboutSel = entityMultiSelect(entities, sit.about_entity_ids);
      aboutSel.dataset.sit = "about_entity_ids";
      const targetSel = entityMultiSelect(entities, sit.target_entity_ids);
      targetSel.dataset.sit = "target_entity_ids";
      [idInp, uidInp, labelInp, stimSel, aboutSel, targetSel].forEach(bindInput);
      card.appendChild(fieldRow("ID", idInp));
      card.appendChild(fieldRow("Utterance ids", uidInp));
      card.appendChild(fieldRow("Label", labelInp));
      card.appendChild(fieldRow("Stimulus entity", stimSel));
      card.appendChild(fieldRow("About entities (Ctrl+click)", aboutSel));
      card.appendChild(fieldRow("Target entities (Ctrl+click)", targetSel));
      const affSel = selectInput(sit.affectLabel || "neutral", AFFECT_LABELS);
      affSel.dataset.sit = "affectLabel";
      const timeSel = selectInput(sit.timeQualitative || "today", TIME_QUAL);
      timeSel.dataset.sit = "timeQualitative";
      const occInp = dateInput(sit.occurredAt);
      occInp.dataset.sit = "occurredAt";
      [affSel, timeSel, occInp].forEach(bindInput);
      card.appendChild(fieldRow("Affect", affSel));
      card.appendChild(fieldRow("Time qual.", timeSel));
      card.appendChild(fieldRow("Occurred (local date)", occInp));
      const partWrap = el("div", "space-y-1");
      partWrap.dataset.sitParticipants = "1";
      card.appendChild(fieldRow("Participants", partWrap));
      renderParticipantRows(partWrap, sit.participants, entities);
      const hint = el(
        "p",
        "text-[10px] text-gray-500",
        "Pick entities by label — no UUID typing. Roles: agent = doer, patient = affected, topic = discourse center.",
      );
      card.appendChild(hint);
      const rm = el("button", "text-[10px] text-red-300/90 hover:text-red-200");
      rm.type = "button";
      rm.textContent = "Remove situation";
      rm.addEventListener("click", () => {
        card.remove();
        debouncedWrite();
      });
      card.appendChild(rm);
      return card;
    }

    function renderEdgeRow(edge, index) {
      const row = el("div", "grid grid-cols-1 sm:grid-cols-4 gap-2 items-end border-b border-gray-800/80 pb-2");
      row.dataset.edgeRow = String(index);
      const sInp = textInput(edge.s, "source id");
      sInp.dataset.edge = "s";
      const pSel = selectInput(edge.p || "schema:about", EDGE_PREDS);
      pSel.dataset.edge = "p";
      const oInp = textInput(edge.o, "target id");
      oInp.dataset.edge = "o";
      const cInp = textInput(edge.confidence != null ? String(edge.confidence) : "0.8", "0–1");
      cInp.dataset.edge = "confidence";
      [sInp, pSel, oInp, cInp].forEach(bindInput);
      row.appendChild(fieldRow("Source", sInp));
      row.appendChild(fieldRow("Predicate", pSel));
      row.appendChild(fieldRow("Target", oInp));
      row.appendChild(fieldRow("Conf.", cInp));
      const rmWrap = el("div", "sm:col-span-4");
      const rm = el("button", "text-[10px] text-red-300/90");
      rm.type = "button";
      rm.textContent = "Remove edge";
      rm.addEventListener("click", () => {
        row.remove();
        debouncedWrite();
      });
      rmWrap.appendChild(rm);
      row.appendChild(rmWrap);
      return row;
    }

    function renderDispositionCard(disp, index, entities) {
      const card = el("div", "border border-gray-800 rounded-lg p-2 space-y-2 bg-gray-950/40");
      card.dataset.dispositionCard = String(index);
      const title = el("div", "text-[11px] text-fuchsia-200 font-medium");
      title.textContent = `Disposition #${index + 1}`;
      card.appendChild(title);
      const idInp = textInput(disp.id || "", "optional id");
      idInp.dataset.disp = "id";
      const holderSel = entitySelect(entities, disp.holder_id, { allowEmpty: false });
      holderSel.dataset.disp = "holder_id";
      const targetSel = entitySelect(entities, disp.target_id, { allowEmpty: false });
      targetSel.dataset.disp = "target_id";
      [idInp, holderSel, targetSel].forEach(bindInput);
      card.appendChild(fieldRow("ID (optional)", idInp));
      card.appendChild(fieldRow("Holder (who feels)", holderSel));
      card.appendChild(fieldRow("Target (toward whom/what)", targetSel));
      const polSel = selectInput(disp.trustPolarity || "unknown", TRUST_POL);
      polSel.dataset.disp = "trustPolarity";
      const descTa = textArea(disp.description, 2);
      descTa.dataset.disp = "description";
      descTa.placeholder = "Grounded note, e.g. Juniper trusts Orion on this topic";
      [polSel, descTa].forEach(bindInput);
      card.appendChild(fieldRow("Trust polarity", polSel));
      card.appendChild(fieldRow("Description", descTa));
      const rm = el("button", "text-[10px] text-red-300/90");
      rm.type = "button";
      rm.textContent = "Remove disposition";
      rm.addEventListener("click", () => {
        card.remove();
        debouncedWrite();
      });
      card.appendChild(rm);
      return card;
    }

    function render(obj) {
      if (!formHost) return;
      formHost.innerHTML = "";
      draftObj = obj;

      if (!obj || !looksDraft(obj)) {
        const msg = el(
          "p",
          "text-[11px] text-gray-500 leading-relaxed",
          "No valid SuggestDraftV1 yet. Use Suggest from chat, paste JSON in Advanced below, or add entities here after the first suggest.",
        );
        formHost.appendChild(msg);
        const seed = el("button", "mt-2 px-2 py-1 rounded border border-gray-600 bg-gray-800 text-[10px] text-gray-200");
        seed.type = "button";
        seed.textContent = "Start empty draft";
        seed.addEventListener("click", () => writeTextarea(emptyDraftFromUi(ui)));
        formHost.appendChild(seed);
        return;
      }

      formHost.appendChild(renderCardDefaultsSection(cardDefaults));

      const meta = el("div", "space-y-2 border border-gray-800 rounded-lg p-2 bg-gray-950/30");
      meta.appendChild(el("div", "text-[10px] text-gray-500", `Ontology: ${obj.ontology_version || "?"}`));
      const uidInp = textInput(joinCsv(obj.utterance_ids), "utterance turn ids");
      uidInp.dataset.draftField = "utterance_ids";
      bindInput(uidInp);
      meta.appendChild(fieldRow("Utterance IDs", uidInp));
      formHost.appendChild(meta);

      const textMap = obj.utterance_text_by_id;
      if (textMap && typeof textMap === "object" && Object.keys(textMap).length) {
        const utWrap = el("details", "border border-gray-800 rounded-lg p-2 bg-gray-950/20");
        const sum = document.createElement("summary");
        sum.className = "text-[10px] text-gray-400 cursor-pointer";
        sum.textContent = "Source utterances (read-only)";
        utWrap.appendChild(sum);
        Object.keys(textMap).forEach((id) => {
          const block = el("div", "mt-2 text-[10px] text-gray-400");
          block.innerHTML = `<span class="text-gray-500 font-mono">${id}</span><p class="text-gray-300 mt-0.5 whitespace-pre-wrap">${String(
            textMap[id] || "",
          )
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")}</p>`;
          utWrap.appendChild(block);
        });
        formHost.appendChild(utWrap);
      }

      const entWrap = el("div", "");
      formHost.appendChild(
        sectionHeader("Entities (relational + topical)", "+ Entity", () => {
          entWrap.appendChild(
            renderEntityCard(
              {
                id: newEntityId(),
                label: "",
                entityKind: "abstract",
                surfaceForms: [],
                generalizes_to: null,
              },
              entWrap.children.length,
            ),
          );
          debouncedWrite();
        }),
      );
      formHost.appendChild(entWrap);
      (obj.entities || []).forEach((e, i) => entWrap.appendChild(renderEntityCard(e, i)));

      const sitWrap = el("div", "");
      formHost.appendChild(
        sectionHeader("Situations", "+ Situation", () => {
          sitWrap.appendChild(
            renderSituationCard(
              {
                id: newEntityId(),
                utterance_ids: obj.utterance_ids || [],
                label: "",
                stimulus_entity_id: null,
                about_entity_ids: [],
                target_entity_ids: [],
                affectLabel: "neutral",
                timeQualitative: "today",
                occurredAt: localDateISO(),
                participants: [],
              },
              sitWrap.children.length,
              collectDraftFromForm().entities,
            ),
          );
          debouncedWrite();
        }),
      );
      formHost.appendChild(sitWrap);
      (obj.situations || []).forEach((s, i) =>
        sitWrap.appendChild(renderSituationCard(s, i, obj.entities)),
      );

      const edgeWrap = el("div", "space-y-2");
      formHost.appendChild(
        sectionHeader("Edges", "+ Edge", () => {
          edgeWrap.appendChild(
            renderEdgeRow({ s: "", p: "schema:about", o: "", confidence: 0.85 }, edgeWrap.children.length),
          );
          debouncedWrite();
        }),
      );
      formHost.appendChild(edgeWrap);
      (obj.edges || []).forEach((e, i) => edgeWrap.appendChild(renderEdgeRow(e, i)));

      const dispWrap = el("div", "");
      formHost.appendChild(
        sectionHeader("Dispositions", "+ Disposition", () => {
          dispWrap.appendChild(
            renderDispositionCard(
              {
                id: newEntityId(),
                holder_id: "",
                target_id: "",
                trustPolarity: "unknown",
                description: "",
              },
              dispWrap.children.length,
              collectDraftFromForm().entities,
            ),
          );
          debouncedWrite();
        }),
      );
      formHost.appendChild(dispWrap);
      (obj.dispositions || []).forEach((d, i) =>
        dispWrap.appendChild(renderDispositionCard(d, i, obj.entities)),
      );
    }

    function buildCardProjectionPayload() {
      const d = readCardDefaults();
      const out = {
        confidence: d.confidence,
        sensitivity: d.sensitivity,
        priority: d.priority,
        provenance: d.provenance,
        visibility_scope: d.visibility_scope,
      };
      if (d.summary && String(d.summary).trim()) out.summary = String(d.summary).trim();
      if (Array.isArray(d.still_true) && d.still_true.length) out.still_true = d.still_true;
      if (Array.isArray(d.evidence) && d.evidence.length) out.evidence = d.evidence;
      const th = d.time_horizon || {};
      const hasTh =
        th.kind && th.kind !== "timeless"
          ? true
          : Boolean(th.start || th.end || th.as_of);
      if (hasTh) out.time_horizon = th;
      return out;
    }

    function refresh() {
      if (syncing) return;
      const parsed = readTextarea();
      if (parsed.ok && parsed.object) {
        const normalized = normalizeDraftForEditor(parsed.object);
        render(normalized);
        if (JSON.stringify(normalized) !== JSON.stringify(parsed.object)) {
          writeTextarea(normalized);
        }
        return;
      }
      render(null);
    }

    const debouncedRefresh = debounce(refresh, 100);
    if (draftTextarea) {
      draftTextarea.addEventListener("input", debouncedRefresh);
    }

    refresh();

    return {
      refresh,
      readCardDefaults,
      setCardDefaults,
      buildCardProjectionPayload,
      destroy() {
        if (draftTextarea) draftTextarea.removeEventListener("input", debouncedRefresh);
        if (formHost) formHost.innerHTML = "";
      },
      flushToTextarea() {
        writeTextarea(collectDraftFromForm());
      },
    };
  }

  window.OrionMemoryGraphDraftForm = {
    attachFormEditor,
    normalizeDraftForEditor,
    localDateISO,
    USER_SPEAKER_LABEL,
  };
})();
