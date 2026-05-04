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
   */
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
        const c = String(block.content || "").trim();
        if (c.startsWith("[Error:") || /timed out|timeout|error/i.test(c)) parts.push(c);
      });
    });
    const merged = parts.filter(Boolean);
    return merged.filter((v, i, a) => a.indexOf(v) === i).join(" · ");
  }

  /**
   * Prefer top-level `text` / raw.final_text only. Never fall back to the full HTTP response body
   * (that leaks the whole JSON envelope into the draft textarea and breaks the graph preview).
   */
  function coalesceChatSuggestDraft(data) {
    const raw = data && typeof data.raw === "object" ? data.raw : null;
    let draft = "";
    if (typeof (data && data.text) === "string" && data.text.trim()) draft = data.text.trim();
    else if (raw && typeof raw.final_text === "string" && raw.final_text.trim()) draft = raw.final_text.trim();
    const stepErr = raw ? extractLlmGatewayErrorFromRaw(raw) : "";
    if (!draft && stepErr) return { draftText: "", error: stepErr };
    if (draft && draft.startsWith("[Error:")) return { draftText: "", error: draft };
    if (stepErr && (!draft || draft === "{}")) return { draftText: "", error: stepErr };
    return { draftText: draft, error: null };
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
        p.textContent = emptyShape
          ? "Parsed JSON is an empty object — nothing to draw. If Suggest failed upstream, check the status line for [Error: …] / timeout, then retry."
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
    draftToCyElements,
    attach,
    coalesceChatSuggestDraft,
    extractLlmGatewayErrorFromRaw,
  };
})();
