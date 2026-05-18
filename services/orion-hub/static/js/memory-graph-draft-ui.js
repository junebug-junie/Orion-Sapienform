/* Orion Hub — shared Memory graph SuggestDraftV1 draft preview (parse + Cytoscape + detail panel). */
(function () {
  function debounce(fn, ms) {
    let t = null;
    return function debounced() {
      const args = arguments;
      const self = this;
      clearTimeout(t);
      t = setTimeout(() => fn.apply(self, args), ms);
    };
  }

  /**
   * One successfully parsed JSON object; salvage modes show explicit warning (no silent graph).
   */
  function parseMemoryGraphDraftJson(text) {
    const trimmed = String(text || "").trim();
    if (!trimmed) {
      return { ok: false, object: null, mode: null, warning: "Empty draft." };
    }
    if (trimmed.startsWith("[Error:")) {
      return { ok: false, object: null, mode: null, warning: trimmed };
    }
    try {
      const o = JSON.parse(trimmed);
      if (o !== null && typeof o === "object" && !Array.isArray(o)) {
        return { ok: true, object: o, mode: "strict", warning: null };
      }
    } catch (_) {
      /* try salvage */
    }
    const fence = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fence) {
      try {
        const o = JSON.parse(fence[1].trim());
        if (o !== null && typeof o === "object" && !Array.isArray(o)) {
          return {
            ok: true,
            object: o,
            mode: "fenced",
            warning: "Parsed JSON from a fenced code block (textarea had extra wrapping).",
          };
        }
      } catch (_) {
        /* continue */
      }
    }
    const start = trimmed.indexOf("{");
    if (start >= 0) {
      let depth = 0;
      for (let i = start; i < trimmed.length; i += 1) {
        const c = trimmed[i];
        if (c === "{") depth += 1;
        else if (c === "}") {
          depth -= 1;
          if (depth === 0) {
            const slice = trimmed.slice(start, i + 1);
            try {
              const o = JSON.parse(slice);
              if (o !== null && typeof o === "object" && !Array.isArray(o)) {
                return {
                  ok: true,
                  object: o,
                  mode: "salvage",
                  warning: "Parsed the first top-level { … } object; leading/trailing text was ignored.",
                };
              }
            } catch (_) {
              /* fall through */
            }
            break;
          }
        }
      }
    }
    return { ok: false, object: null, mode: null, warning: "Could not parse a single JSON object." };
  }

  /**
   * Pull gateway/LLM failure text from cortex-exec raw.steps (e.g. "[Error: llamacpp timed out…]").
   * Do not scan full model JSON for the word "error" — memory-graph drafts legitimately contain it.
   */
  function contentLooksLikeGatewayFailureBlurb(c) {
    const s = String(c || "").trim();
    if (!s) return false;
    if (s.startsWith("[Error:")) return true;
    const head = s.length > 600 ? `${s.slice(0, 600)}…` : s;
    return /\b(timed out|timeout)\b/i.test(head);
  }

  function extractLlmGatewayErrorFromRaw(raw) {
    if (!raw || typeof raw !== "object") return "";
    const topErr = raw.error != null ? String(raw.error).trim() : "";
    const parts = [];
    if (topErr) parts.push(topErr);
    const steps = raw.steps;
    if (!Array.isArray(steps)) {
      return parts.filter(Boolean).join(" · ") || "";
    }
    steps.forEach((step) => {
      if (!step || typeof step !== "object") return;
      if (step.error != null) parts.push(String(step.error));
      const res = step.result;
      if (!res || typeof res !== "object") return;
      Object.keys(res).forEach((svcKey) => {
        const block = res[svcKey];
        if (!block || typeof block !== "object") return;
        const be = block.error;
        if (be != null) {
          if (typeof be === "string" && be.trim()) parts.push(be.trim());
          else if (typeof be === "object" && typeof be.message === "string" && be.message.trim()) {
            parts.push(be.message.trim());
          }
        }
        const c = String(block.content || "").trim();
        if (contentLooksLikeGatewayFailureBlurb(c)) parts.push(c);
      });
    });
    const merged = parts.filter(Boolean);
    return merged.filter((v, i, a) => a.indexOf(v) === i).join(" · ");
  }

  const SUGGEST_DRAFT_ONTOLOGY_VERSION = "orionmem-2026-05";

  function looksLikeMemoryGraphDraftObject(obj) {
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return false;
    if (obj.ontology_version !== SUGGEST_DRAFT_ONTOLOGY_VERSION) return false;
    if (!Array.isArray(obj.utterance_ids)) return false;
    if (!Array.isArray(obj.entities)) return false;
    if (!Array.isArray(obj.situations)) return false;
    if (!Array.isArray(obj.edges)) return false;
    if (!Array.isArray(obj.dispositions)) return false;
    return true;
  }

  /** Selected-turn evidence only (request input), not a SuggestDraftV1 draft. */
  function looksLikeEvidenceEnvelopeOnly(obj) {
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return false;
    if (obj.ontology_version === SUGGEST_DRAFT_ONTOLOGY_VERSION) return false;
    const hasIds = Array.isArray(obj.utterance_ids);
    const hasTextMap =
      obj.utterance_text_by_id != null &&
      typeof obj.utterance_text_by_id === "object" &&
      !Array.isArray(obj.utterance_text_by_id);
    if (!hasIds && !hasTextMap) return false;
    if (Array.isArray(obj.entities) || Array.isArray(obj.situations)) return false;
    if (Array.isArray(obj.edges) || Array.isArray(obj.dispositions)) return false;
    return hasIds || hasTextMap;
  }

  function emptySuggestDraft(options) {
    const opts = options && typeof options === "object" ? options : {};
    const utteranceIds = opts.utteranceIds;
    const utteranceTextById = opts.utteranceTextById;
    const ids = Array.isArray(utteranceIds)
      ? utteranceIds.map((x) => String(x || "").trim()).filter(Boolean)
      : [];
    return {
      ontology_version: SUGGEST_DRAFT_ONTOLOGY_VERSION,
      utterance_ids: ids,
      entities: [],
      situations: [],
      edges: [],
      dispositions: [],
      utterance_text_by_id:
        utteranceTextById && typeof utteranceTextById === "object" && !Array.isArray(utteranceTextById)
          ? utteranceTextById
          : {},
    };
  }

  /** @deprecated use emptySuggestDraft */
  function emptyValidSuggestDraft(utteranceIds) {
    return emptySuggestDraft({ utteranceIds });
  }

  function collectSuggestDiagnostics(data) {
    const attempts = Array.isArray(data.suggest_attempts)
      ? data.suggest_attempts
      : Array.isArray(data.attempts)
        ? data.attempts
        : [];
    const validationErrors = [];
    if (Array.isArray(data.validation_errors)) {
      data.validation_errors.forEach((e) => {
        if (e != null && String(e).trim()) validationErrors.push(String(e).trim());
      });
    }
    attempts.forEach((att) => {
      if (!att || typeof att !== "object") return;
      const phase = att.phase != null ? String(att.phase).trim() : "";
      if (phase && validationErrors.indexOf(phase) < 0) validationErrors.push(phase);
      const ve = att.validation_errors;
      if (Array.isArray(ve)) {
        ve.forEach((e) => {
          const s = e != null ? String(e).trim() : "";
          if (s && validationErrors.indexOf(s) < 0) validationErrors.push(s);
        });
      }
    });
    return {
      route_used: data.suggest_route_used != null ? data.suggest_route_used : data.route_used ?? null,
      attempts,
      validation_errors: validationErrors,
      api_error: data.error != null ? String(data.error) : null,
      diagnostic_raw:
        typeof data.diagnostic_raw === "string" && data.diagnostic_raw.trim()
          ? data.diagnostic_raw
          : null,
    };
  }

  /**
   * Coalesce /api/memory/graph/suggest (or compatible) envelopes into textarea-safe draft JSON.
   * Never returns assistant prose as draftText.
   */
  function coalesceMemoryGraphSuggestEnvelope(data, options) {
    const opts = options && typeof options === "object" ? options : {};
    const utteranceIds = opts.utteranceIds;
    const utteranceTextById = opts.utteranceTextById;
    const emptyDraft = emptySuggestDraft({ utteranceIds, utteranceTextById });
    const emptyText = JSON.stringify(emptyDraft, null, 2);

    function rejectResult(err, extraDiagnostics, preserveFrom) {
      let ids = utteranceIds;
      let textMap = utteranceTextById;
      if (preserveFrom && typeof preserveFrom === "object" && !Array.isArray(preserveFrom)) {
        if (Array.isArray(preserveFrom.utterance_ids) && preserveFrom.utterance_ids.length) {
          ids = preserveFrom.utterance_ids;
        }
        if (
          preserveFrom.utterance_text_by_id &&
          typeof preserveFrom.utterance_text_by_id === "object" &&
          !Array.isArray(preserveFrom.utterance_text_by_id)
        ) {
          textMap = preserveFrom.utterance_text_by_id;
        }
      }
      const diagnostics = collectSuggestDiagnostics(data || {});
      if (extraDiagnostics && typeof extraDiagnostics === "object") {
        Object.keys(extraDiagnostics).forEach((k) => {
          diagnostics[k] = extraDiagnostics[k];
        });
      }
      return {
        draftText: JSON.stringify(emptySuggestDraft({ utteranceIds: ids, utteranceTextById: textMap }), null, 2),
        error: err,
        diagnostics,
      };
    }

    if (!data || typeof data !== "object") {
      return rejectResult("memory_graph_suggest_failed", { reason: "empty_response" });
    }

    function successFromObject(obj) {
      const diagnostics = collectSuggestDiagnostics(data);
      if (Array.isArray(data.violations) && data.violations.length) {
        data.violations.forEach((v) => {
          const s = v != null ? String(v).trim() : "";
          if (s && diagnostics.validation_errors.indexOf(s) < 0) {
            diagnostics.validation_errors.push(s);
          }
        });
      }
      const ents = Array.isArray(obj.entities) ? obj.entities : [];
      const sits = Array.isArray(obj.situations) ? obj.situations : [];
      const edgs = Array.isArray(obj.edges) ? obj.edges : [];
      const disps = Array.isArray(obj.dispositions) ? obj.dispositions : [];
      const graphEmpty = !ents.length && !sits.length && !edgs.length && !disps.length;
      const apiErr = data.error != null ? String(data.error).trim() : "";
      const failed = data.ok === false;
      return {
        draftText: JSON.stringify(obj, null, 2),
        error: failed ? apiErr || "memory_graph_suggest_failed" : null,
        diagnostics,
        graphEmpty,
      };
    }

    function tryAcceptObject(obj) {
      if (!obj || typeof obj !== "object" || Array.isArray(obj)) return null;
      if (looksLikeEvidenceEnvelopeOnly(obj)) {
        return { rejected: "evidence_envelope_not_draft", object: obj };
      }
      if (looksLikeMemoryGraphDraftObject(obj)) {
        return { accepted: successFromObject(obj) };
      }
      if (Array.isArray(obj.utterance_ids) && !looksLikeMemoryGraphDraftObject(obj)) {
        return { rejected: "missing_required_suggest_draft_fields" };
      }
      return null;
    }

    if (data.draft && typeof data.draft === "object" && !Array.isArray(data.draft)) {
      const verdict = tryAcceptObject(data.draft);
      if (verdict && verdict.accepted) return verdict.accepted;
      if (verdict && verdict.rejected) {
        return rejectResult(verdict.rejected, null, verdict.object);
      }
    }

    if (typeof data.appendix_c_json === "string" && data.appendix_c_json.trim()) {
      const parsed = parseMemoryGraphDraftJson(data.appendix_c_json.trim());
      if (parsed.ok && parsed.object) {
        const verdict = tryAcceptObject(parsed.object);
        if (verdict && verdict.accepted) return verdict.accepted;
        if (verdict && verdict.rejected) {
          return rejectResult(verdict.rejected, null, verdict.object);
        }
      }
    }

    const raw = data.raw && typeof data.raw === "object" ? data.raw : null;
    const textCandidates = [];
    if (typeof data.text === "string" && data.text.trim()) textCandidates.push(data.text.trim());
    if (raw && typeof raw.final_text === "string" && raw.final_text.trim()) {
      textCandidates.push(raw.final_text.trim());
    }
    if (raw) {
      const fromSteps = extractLlmGatewayDraftFromSteps(raw);
      if (fromSteps) textCandidates.push(fromSteps);
    }

    for (let i = 0; i < textCandidates.length; i += 1) {
      const candidate = textCandidates[i];
      const parsed = parseMemoryGraphDraftJson(candidate);
      if (parsed.ok && parsed.object) {
        const verdict = tryAcceptObject(parsed.object);
        if (verdict && verdict.accepted) return verdict.accepted;
        if (verdict && verdict.rejected) {
          return rejectResult(
            verdict.rejected,
            {
              prose_rejected: verdict.rejected === "evidence_envelope_not_draft",
              prose_preview: candidate.slice(0, 240),
            },
            verdict.object,
          );
        }
      }
    }

    const extraDiag = {};
    if (textCandidates.length > 0) {
      extraDiag.prose_rejected = true;
      extraDiag.prose_preview = textCandidates[0].slice(0, 240);
      const firstParsed = parseMemoryGraphDraftJson(textCandidates[0]);
      if (firstParsed.ok && firstParsed.object && looksLikeEvidenceEnvelopeOnly(firstParsed.object)) {
        return rejectResult("evidence_envelope_not_draft", extraDiag, firstParsed.object);
      }
    }

    const apiErr = data.error != null ? String(data.error).trim() : "";
    let err = "invalid_model_output";
    if (
      apiErr === "memory_graph_suggest_failed" ||
      apiErr === "memory_graph_suggest_exhausted" ||
      data.ok === false
    ) {
      err = apiErr || "memory_graph_suggest_failed";
    }

    return rejectResult(err, extraDiag);
  }

  /**
   * When Hub `text` / raw.final_text are empty, recover JSON from step payloads (same source cortex-exec uses).
   */
  function extractLlmGatewayDraftFromSteps(raw) {
    if (!raw || typeof raw !== "object") return "";
    const steps = raw.steps;
    if (!Array.isArray(steps)) return "";
    for (let si = steps.length - 1; si >= 0; si -= 1) {
      const step = steps[si];
      if (!step || typeof step !== "object") continue;
      const res = step.result;
      if (!res || typeof res !== "object") continue;
      const keys = Object.keys(res);
      for (let ki = 0; ki < keys.length; ki += 1) {
        const block = res[keys[ki]];
        if (!block || typeof block !== "object") continue;
        const candidate =
          (typeof block.content === "string" && block.content.trim() && block.content) ||
          (typeof block.final_text === "string" && block.final_text.trim() && block.final_text) ||
          (typeof block.text === "string" && block.text.trim() && block.text) ||
          "";
        const s = String(candidate || "").trim();
        if (!s || contentLooksLikeGatewayFailureBlurb(s)) continue;
        const parsed = parseMemoryGraphDraftJson(s);
        if (parsed.ok && parsed.object && looksLikeMemoryGraphDraftObject(parsed.object)) {
          return JSON.stringify(parsed.object);
        }
      }
    }
    return "";
  }

  /**
   * Operator-facing status line from coalesceMemoryGraphSuggestEnvelope result.
   */
  function formatSuggestCoalesceUserStatus(coalesce) {
    if (!coalesce || typeof coalesce !== "object") {
      return "Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics.";
    }
    const err = coalesce.error != null ? String(coalesce.error).trim() : "";
    if (!err) {
      if (coalesce.graphEmpty) {
        return "Loaded valid empty SuggestDraftV1 JSON.";
      }
      return "Loaded validated role-grounded SuggestDraftV1 JSON.";
    }
    if (err === "evidence_envelope_not_draft") {
      return "Blocked selected-turn evidence envelope from Draft JSON; evidence is request input, not graph output.";
    }
    return "Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics.";
  }

  /**
   * Legacy chat-envelope coalescer; delegates to suggest-route envelope (never returns prose).
   */
  function coalesceChatSuggestDraft(data) {
    const out = coalesceMemoryGraphSuggestEnvelope(data);
    if (!out.draftText) {
      const raw = data && data.raw && typeof data.raw === "object" ? data.raw : null;
      const stepErr = raw ? extractLlmGatewayErrorFromRaw(raw) : "";
      if (stepErr) {
        return {
          draftText: out.draftText || JSON.stringify(emptySuggestDraft(), null, 2),
          error: stepErr,
          diagnostics: out.diagnostics,
        };
      }
    }
    return out;
  }

  function draftToCyElements(obj) {
    const nodes = [];
    const edges = [];
    const seen = new Set();
    function addNode(id, label, kind) {
      const sid = String(id);
      if (seen.has(sid)) return;
      seen.add(sid);
      nodes.push({ data: { id: sid, label: String(label || sid), kind } });
    }

    const ents = Array.isArray(obj.entities) ? obj.entities : [];
    ents.forEach((e) => {
      if (e && e.id) addNode(e.id, e.label || e.id, "entity");
    });
    const sits = Array.isArray(obj.situations) ? obj.situations : [];
    sits.forEach((s) => {
      if (s && s.id) addNode(s.id, s.label || s.id, "situation");
    });
    const disps = Array.isArray(obj.dispositions) ? obj.dispositions : [];
    disps.forEach((d) => {
      if (d && d.id) addNode(d.id, (d.description && String(d.description).slice(0, 40)) || String(d.id), "disposition");
    });

    const edgs = Array.isArray(obj.edges) ? obj.edges : [];
    edgs.forEach((e, i) => {
      if (!e || e.s == null || e.o == null) return;
      const sid = String(e.s);
      const oid = String(e.o);
      if (!seen.has(sid)) addNode(sid, sid, "ref");
      if (!seen.has(oid)) addNode(oid, oid, "ref");
      edges.push({
        data: {
          id: `draft-edge-${i}`,
          source: sid,
          target: oid,
          label: String(e.p || ""),
          edgeIndex: i,
        },
      });
    });

    return { nodes, edges };
  }

  function renderDetail(selection, obj, onApply) {
    const wrap = document.createElement("div");
    wrap.className = "space-y-2 text-[11px]";

    if (!selection) {
      wrap.innerHTML = '<p class="text-gray-500">Select a node or edge in the graph.</p>';
      return wrap;
    }

    if (selection.type === "edge") {
      const e = selection.edge;
      const idx = selection.index;
      const head = document.createElement("div");
      head.className = "text-gray-400 font-mono text-[10px] break-all";
      head.textContent = `edge[${idx}] ${e.s} —${e.p}→ ${e.o}`;
      wrap.appendChild(head);
      ["s", "p", "o"].forEach((field) => {
        const lab = document.createElement("label");
        lab.className = "block text-gray-500";
        lab.textContent = field;
        const inp = document.createElement("input");
        inp.type = "text";
        inp.className = "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100";
        inp.value = e[field] != null ? String(e[field]) : "";
        inp.dataset.field = field;
        lab.appendChild(inp);
        wrap.appendChild(lab);
      });
      if (idx >= 0 && Array.isArray(obj.edges) && obj.edges[idx]) {
        const applyBtn = document.createElement("button");
        applyBtn.type = "button";
        applyBtn.className = "mt-2 px-2 py-1 rounded border border-violet-600 bg-violet-900/30 text-violet-100";
        applyBtn.textContent = "Apply to draft JSON";
        applyBtn.addEventListener("click", () => {
          const edges = [...obj.edges];
          const cur = { ...(edges[idx] || {}) };
          wrap.querySelectorAll("input[data-field]").forEach((inp) => {
            const f = inp.dataset.field;
            if (f) cur[f] = inp.value;
          });
          edges[idx] = cur;
          onApply({ ...obj, edges });
        });
        wrap.appendChild(applyBtn);
      } else {
        const note = document.createElement("p");
        note.className = "text-amber-200/90 text-[10px]";
        note.textContent = "This edge is not indexed in draft.edges — edit JSON or fix the draft.";
        wrap.appendChild(note);
      }
      return wrap;
    }

    const arrName = selection.arrayName;
    const idx = selection.index;
    const arr = Array.isArray(obj[arrName]) ? obj[arrName] : [];
    const row = arr[idx];
    if (!row || typeof row !== "object") {
      wrap.innerHTML = '<p class="text-gray-500">Missing row in draft.</p>';
      return wrap;
    }

    const title = document.createElement("div");
    title.className = "text-gray-300 font-semibold capitalize";
    title.textContent = `${arrName.slice(0, -1)} ${row.id != null ? String(row.id) : `#${idx}`}`;
    wrap.appendChild(title);

    const keys = Object.keys(row);
    keys.forEach((k) => {
      const val = row[k];
      if (val != null && typeof val === "object") return;
      const lab = document.createElement("label");
      lab.className = "block text-gray-500";
      lab.textContent = k;
      const inp = document.createElement("input");
      inp.type = "text";
      inp.className = "w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100";
      inp.value = val != null ? String(val) : "";
      inp.dataset.key = k;
      lab.appendChild(inp);
      wrap.appendChild(lab);
    });

    const applyBtn = document.createElement("button");
    applyBtn.type = "button";
    applyBtn.className = "mt-2 px-2 py-1 rounded border border-violet-600 bg-violet-900/30 text-violet-100";
    applyBtn.textContent = "Apply to draft JSON";
    applyBtn.addEventListener("click", () => {
      const copy = { ...row };
      wrap.querySelectorAll("input[data-key]").forEach((inp) => {
        const k = inp.dataset.key;
        if (k) copy[k] = inp.value;
      });
      const narr = [...arr];
      narr[idx] = copy;
      const next = { ...obj, [arrName]: narr };
      onApply(next);
    });
    wrap.appendChild(applyBtn);
    return wrap;
  }

  function findNodeSelection(obj, targetId) {
    const id = String(targetId);
    const ents = Array.isArray(obj.entities) ? obj.entities : [];
    for (let i = 0; i < ents.length; i += 1) {
      if (ents[i] && String(ents[i].id) === id) return { type: "node", arrayName: "entities", index: i };
    }
    const sits = Array.isArray(obj.situations) ? obj.situations : [];
    for (let i = 0; i < sits.length; i += 1) {
      if (sits[i] && String(sits[i].id) === id) return { type: "node", arrayName: "situations", index: i };
    }
    const disps = Array.isArray(obj.dispositions) ? obj.dispositions : [];
    for (let i = 0; i < disps.length; i += 1) {
      if (disps[i] && String(disps[i].id) === id) return { type: "node", arrayName: "dispositions", index: i };
    }
    return null;
  }

  /**
   * @param {object} options
   * @param {HTMLTextAreaElement|null} options.draftTextarea
   * @param {HTMLElement|null} options.cyHost
   * @param {HTMLElement|null} options.detailHost
   * @param {HTMLElement|null} [options.bannerEl]
   * @param {() => void} [options.onDraftJsonChange]
   */
  function attach(options) {
    const draftTextarea = options.draftTextarea;
    const cyHost = options.cyHost;
    const detailHost = options.detailHost;
    const bannerEl = options.bannerEl || null;
    const onDraftJsonChange = options.onDraftJsonChange || function () {};

    if (!window.cytoscape || !cyHost) {
      return { refresh: function () {}, destroy: function () {}, getParsed: function () { return null; } };
    }

    let cy = null;
    let lastParsed = null;
    let lastObj = null;

    function destroyCy() {
      if (cy) {
        try {
          cy.destroy();
        } catch (_) {
          /* ignore */
        }
        cy = null;
      }
    }

    function renderDetailFor(evt) {
      if (!detailHost || !lastObj) return;
      detailHost.innerHTML = "";
      const t = evt && evt.target;
      if (!t || typeof t !== "object") return;
      if (t.isEdge && t.isEdge()) {
        const ed = t.data();
        const edges = Array.isArray(lastObj.edges) ? lastObj.edges : [];
        const idx = typeof ed.edgeIndex === "number" ? ed.edgeIndex : -1;
        const edge = idx >= 0 && edges[idx] ? edges[idx] : { s: ed.source, p: ed.label, o: ed.target };
        const panel = renderDetail(
          { type: "edge", edge, index: idx >= 0 ? idx : 0 },
          lastObj,
          applyObj,
        );
        detailHost.appendChild(panel);
        return;
      }
      if (t.isNode && t.isNode()) {
        const nid = t.id();
        const sel = findNodeSelection(lastObj, nid);
        if (sel && sel.type === "node") {
          detailHost.appendChild(renderDetail(sel, lastObj, applyObj));
        } else {
          const el = document.createElement("div");
          el.className = "text-gray-500 text-[11px]";
          el.textContent = `Node "${nid}" is not listed under entities / situations / dispositions (edge endpoint or ref).`;
          detailHost.appendChild(el);
        }
      }
    }

    function applyObj(nextObj) {
      lastObj = nextObj;
      const json = JSON.stringify(nextObj, null, 2);
      if (draftTextarea) draftTextarea.value = json;
      onDraftJsonChange();
      refresh();
    }

    function refresh() {
      if (!draftTextarea) return;
      const raw = draftTextarea.value;
      const parsed = parseMemoryGraphDraftJson(raw);
      lastParsed = parsed;

      if (bannerEl) {
        bannerEl.textContent = "";
        bannerEl.className =
          "text-[11px] rounded border px-2 py-1 whitespace-pre-wrap hidden border-gray-700 text-gray-400 bg-gray-950/60";
        if (!parsed.ok) {
          bannerEl.classList.remove("hidden");
          bannerEl.classList.add("border-amber-700", "text-amber-100", "bg-amber-950/50");
          bannerEl.textContent = parsed.warning || "Invalid JSON.";
        } else if (parsed.warning) {
          bannerEl.classList.remove("hidden");
          bannerEl.classList.add("border-amber-700", "text-amber-100", "bg-amber-950/50");
          bannerEl.textContent = parsed.warning;
        }
      }

      destroyCy();
      cyHost.innerHTML = "";
      detailHost.innerHTML = "";

      if (!parsed.ok || !parsed.object) {
        lastObj = null;
        if (detailHost) {
          const p = document.createElement("p");
          p.className = "text-gray-500 text-[11px]";
          p.textContent = "No graph preview until JSON parses.";
          detailHost.appendChild(p);
        }
        return;
      }

      lastObj = parsed.object;
      const { nodes, edges } = draftToCyElements(parsed.object);
      if (!nodes.length && !edges.length) {
        const p = document.createElement("p");
        p.className = "text-gray-500 text-[11px]";
        const keys = parsed.object && typeof parsed.object === "object" ? Object.keys(parsed.object) : [];
        const emptyShape = keys.length === 0;
        const isValidEmptyDraft =
          looksLikeMemoryGraphDraftObject(parsed.object) &&
          !draftToCyElements(parsed.object).nodes.length &&
          !draftToCyElements(parsed.object).edges.length;
        p.textContent = emptyShape
          ? "Parsed JSON is an empty object — nothing to draw. Check the status line for suggest outcome."
          : isValidEmptyDraft
            ? "Valid SuggestDraftV1 with no entities, situations, edges, or dispositions — no graph to preview (may mean no durable memory candidate)."
            : "Parsed JSON has no entities, situations, edges, or dispositions to show.";
        cyHost.appendChild(p);
        return;
      }

      cy = window.cytoscape({
        container: cyHost,
        elements: [...nodes, ...edges],
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              "font-size": 10,
              color: "#e2e8f0",
              "background-color": "#475569",
              width: 28,
              height: 28,
            },
          },
          { selector: 'node[kind = "entity"]', style: { "background-color": "#4f46e5" } },
          { selector: 'node[kind = "situation"]', style: { "background-color": "#059669" } },
          { selector: 'node[kind = "disposition"]', style: { "background-color": "#c026d3" } },
          { selector: 'node[kind = "ref"]', style: { "background-color": "#64748b" } },
          {
            selector: "edge",
            style: {
              width: 2,
              "line-color": "#64748b",
              "target-arrow-color": "#64748b",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              label: "data(label)",
              "font-size": 8,
              color: "#94a3b8",
            },
          },
        ],
        layout: { name: "cose", animate: false },
        wheelSensitivity: 0.35,
      });
      cy.on("tap", "node", renderDetailFor);
      cy.on("tap", "edge", renderDetailFor);
      cy.userPanningEnabled(true);
      cy.boxSelectionEnabled(false);
      requestAnimationFrame(() => {
        try {
          cy.resize();
          cy.fit(undefined, 24);
        } catch (_) {
          /* ignore */
        }
      });
    }

    const debounced = debounce(refresh, 120);
    if (draftTextarea) {
      draftTextarea.addEventListener("input", debounced);
    }

    return {
      refresh,
      destroy: function () {
        if (draftTextarea) draftTextarea.removeEventListener("input", debounced);
        destroyCy();
      },
      getParsed: function () {
        return lastParsed;
      },
    };
  }

  window.OrionMemoryGraphDraftUI = {
    parseMemoryGraphDraftJson,
    looksLikeMemoryGraphDraftObject,
    looksLikeEvidenceEnvelopeOnly,
    emptySuggestDraft,
    emptyValidSuggestDraft,
    draftToCyElements,
    attach,
    coalesceChatSuggestDraft,
    coalesceMemoryGraphSuggestEnvelope,
    formatSuggestCoalesceUserStatus,
    extractLlmGatewayErrorFromRaw,
  };
})();
