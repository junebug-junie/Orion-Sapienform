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
          generalizes_to: card.querySelector("[data-ent=generalizes_to]")?.value?.trim() || null,
        });
      });

      base.situations = [];
      root.querySelectorAll("[data-situation-card]").forEach((card) => {
        const occurred = card.querySelector("[data-sit=occurredAt]")?.value?.trim();
        base.situations.push({
          id: card.querySelector("[data-sit=id]")?.value?.trim() || newEntityId(),
          utterance_ids: splitCsv(card.querySelector("[data-sit=utterance_ids]")?.value),
          label: card.querySelector("[data-sit=label]")?.value?.trim() || "",
          stimulus_entity_id: card.querySelector("[data-sit=stimulus_entity_id]")?.value?.trim() || null,
          about_entity_ids: splitCsv(card.querySelector("[data-sit=about_entity_ids]")?.value),
          target_entity_ids: splitCsv(card.querySelector("[data-sit=target_entity_ids]")?.value),
          affectLabel: card.querySelector("[data-sit=affectLabel]")?.value || "neutral",
          timeQualitative: card.querySelector("[data-sit=timeQualitative]")?.value || "unknown",
          occurredAt: occurred || null,
          participants: participantsFromText(card.querySelector("[data-sit=participants]")?.value),
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
      const fields = [
        ["id", sit.id, "urn:uuid:…"],
        ["utterance_ids", joinCsv(sit.utterance_ids), "turn ids"],
        ["label", sit.label, "short description"],
        ["stimulus_entity_id", sit.stimulus_entity_id || "", "entity id"],
        ["about_entity_ids", joinCsv(sit.about_entity_ids), "entity ids"],
        ["target_entity_ids", joinCsv(sit.target_entity_ids), "entity ids"],
      ];
      fields.forEach(([key, val, ph]) => {
        const inp = textInput(val, ph);
        inp.dataset.sit = key;
        bindInput(inp);
        card.appendChild(fieldRow(key.replace(/_/g, " "), inp));
      });
      const affSel = selectInput(sit.affectLabel || "neutral", AFFECT_LABELS);
      affSel.dataset.sit = "affectLabel";
      const timeSel = selectInput(sit.timeQualitative || "unknown", TIME_QUAL);
      timeSel.dataset.sit = "timeQualitative";
      const occInp = textInput(sit.occurredAt || "", "YYYY-MM-DD or empty");
      occInp.dataset.sit = "occurredAt";
      const partTa = textArea(participantsToText(sit.participants), 3);
      partTa.dataset.sit = "participants";
      partTa.placeholder = "role: entity_id (one per line), e.g. topic: urn:uuid:…";
      [affSel, timeSel, occInp, partTa].forEach(bindInput);
      card.appendChild(fieldRow("Affect", affSel));
      card.appendChild(fieldRow("Time", timeSel));
      card.appendChild(fieldRow("Occurred at", occInp));
      card.appendChild(fieldRow("Participants", partTa));
      const hint = el(
        "p",
        "text-[10px] text-gray-500",
        `Entities: ${(entities || [])
          .map((e) => `${e.label || "?"} → ${e.id}`)
          .join(" · ") || "(none yet)"}`,
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

    function renderDispositionCard(disp, index) {
      const card = el("div", "border border-gray-800 rounded-lg p-2 space-y-2 bg-gray-950/40");
      card.dataset.dispositionCard = String(index);
      const fields = [
        ["id", disp.id || "", "optional"],
        ["holder_id", disp.holder_id, "entity id"],
        ["target_id", disp.target_id, "entity id"],
      ];
      fields.forEach(([key, val, ph]) => {
        const inp = textInput(val, ph);
        inp.dataset.disp = key;
        bindInput(inp);
        card.appendChild(fieldRow(key, inp));
      });
      const polSel = selectInput(disp.trustPolarity || "unknown", TRUST_POL);
      polSel.dataset.disp = "trustPolarity";
      const descTa = textArea(disp.description, 2);
      descTa.dataset.disp = "description";
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
                timeQualitative: "recent",
                occurredAt: null,
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
            ),
          );
          debouncedWrite();
        }),
      );
      formHost.appendChild(dispWrap);
      (obj.dispositions || []).forEach((d, i) => dispWrap.appendChild(renderDispositionCard(d, i)));
    }

    function refresh() {
      if (syncing) return;
      const parsed = readTextarea();
      if (parsed.ok && parsed.object) {
        render(parsed.object);
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
  };
})();
