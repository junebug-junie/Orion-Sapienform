/* Orion Hub — organ signal causal graph from /api/signals/active or correlation chain. */
(function () {
  const GENERIC_PLACEHOLDER_DIMS = { level: 0.5, confidence: 0.5 };

  const STUB_ORGAN_IDS = new Set([
    "social_room_bridge",
    "vision",
    "agent_chain",
    "planner",
    "dream",
    "state_journaler",
    "topic_foundry",
    "concept_induction",
    "graph_cognition",
    "power_guard",
    "security_watcher",
  ]);

  const DEFAULT_ORGAN_LAYERS = {
    cortex_exec: "runtime",
    llm_gateway: "runtime",
    cortex_gateway: "runtime",
    cortex_orch: "runtime",
    hub: "runtime",
    graph_cognition: "cognition",
    chat_stance: "cognition",
    recall: "cognition",
    mind: "cognition",
    spark_introspector: "cognition",
    autonomy: "cognition",
    biometrics: "infra",
    equilibrium: "infra",
    journaler: "memory",
    collapse_mirror: "memory",
    social_memory: "social",
    world_pulse: "vision",
    sql_writer: "persistence",
    rdf_writer: "persistence",
    vector_writer: "persistence",
  };

  function filterSignalsByLayer(signalsMap, layersMap, layerKey) {
    const key = String(layerKey || "all").toLowerCase();
    if (key === "all") return signalsMap;
    const layers = layersMap && typeof layersMap === "object" ? layersMap : DEFAULT_ORGAN_LAYERS;
    return Object.fromEntries(
      Object.entries(signalsMap || {}).filter(([oid]) => {
        const layer = String(layers[oid] || DEFAULT_ORGAN_LAYERS[oid] || "cognition").toLowerCase();
        return layer === key;
      }),
    );
  }

  function destroyCy(cy) {
    if (cy && typeof cy.destroy === "function") {
      try {
        cy.destroy();
      } catch (_) {
        /* ignore */
      }
    }
  }

  function organClassColor(cls) {
    const c = String(cls || "").toLowerCase();
    if (c === "exogenous") return "#059669";
    if (c === "endogenous") return "#7c3aed";
    if (c === "hybrid") return "#ca8a04";
    return "#475569";
  }

  function parseCorrelationIdFromSearch(search) {
    const qs = String(search || window.location.search || "");
    if (!qs) return "";
    const params = new URLSearchParams(qs.startsWith("?") ? qs : `?${qs}`);
    return String(params.get("correlation_id") || "").trim();
  }

  function isPlaceholderDimensions(dimensions) {
    const dims = dimensions && typeof dimensions === "object" ? dimensions : {};
    const keys = Object.keys(dims);
    if (keys.length !== 2) return false;
    if (!keys.includes("level") || !keys.includes("confidence")) return false;
    return dims.level === 0.5 && dims.confidence === 0.5;
  }

  function isStubSignal(row) {
    const sig = row && typeof row === "object" ? row : {};
    const notes = Array.isArray(sig.notes) ? sig.notes : [];
    for (let i = 0; i < notes.length; i += 1) {
      if (String(notes[i]).toLowerCase().includes("stub adapter")) return true;
    }
    if (sig.summary && String(sig.summary).toLowerCase().includes("stub adapter")) {
      return true;
    }
    if (isPlaceholderDimensions(sig.dimensions)) return true;
    const oid = sig.organ_id != null ? String(sig.organ_id) : "";
    if (STUB_ORGAN_IDS.has(oid) && !sig.source_event_id) {
      return true;
    }
    return false;
  }

  function buildGraphElements(signalsMap) {
    const map = signalsMap && typeof signalsMap === "object" ? signalsMap : {};
    const organs = Object.keys(map);
    const sidToOrgan = {};
    organs.forEach((oid) => {
      const s = map[oid];
      if (s && typeof s === "object" && s.signal_id) {
        sidToOrgan[String(s.signal_id)] = oid;
      }
    });
    const nodes = organs.map((oid) => {
      const s = map[oid] || {};
      const ocls = s.organ_class != null ? String(s.organ_class) : "";
      const label = [oid, s.signal_kind ? String(s.signal_kind) : ""].filter(Boolean).join("\n");
      return {
        data: {
          id: oid,
          label: label || oid,
          organ_class: ocls,
          organ_class_color: organClassColor(ocls),
          signal_kind: s.signal_kind != null ? String(s.signal_kind) : "",
          otel_trace_id: s.otel_trace_id != null ? String(s.otel_trace_id) : "",
          signal_id: s.signal_id != null ? String(s.signal_id) : "",
          is_stub: isStubSignal(s),
          raw_json: JSON.stringify(s, null, 2),
        },
      };
    });
    const edges = [];
    const edgeKey = new Set();
    organs.forEach((child) => {
      const s = map[child];
      if (!s || typeof s !== "object") return;
      const parents = Array.isArray(s.causal_parents) ? s.causal_parents : [];
      parents.forEach((pid, idx) => {
        const src = sidToOrgan[String(pid)];
        if (!src || src === child) return;
        const ek = `${src}|${child}|${idx}`;
        if (edgeKey.has(ek)) return;
        edgeKey.add(ek);
        edges.push({
          data: {
            id: `e-${src}-${child}-${idx}`,
            source: src,
            target: child,
            label: "from",
          },
        });
      });
    });
    return { nodes, edges };
  }

  function buildCorrelationGraphElements(chain, options) {
    const opts = options && typeof options === "object" ? options : {};
    const showStubs = Boolean(opts.showStubs);
    const rows = Array.isArray(chain) ? chain : [];
    const visible = showStubs ? rows : rows.filter((row) => !isStubSignal(row));
    const visibleIds = new Set(
      visible.map((row) => (row && row.signal_id ? String(row.signal_id) : "")).filter(Boolean),
    );
    const nodes = visible.map((row) => {
      const sid = row.signal_id != null ? String(row.signal_id) : "";
      const oid = row.organ_id != null ? String(row.organ_id) : "";
      const kind = row.signal_kind != null ? String(row.signal_kind) : "";
      const label = [oid, kind].filter(Boolean).join("\n");
      const stub = isStubSignal(row);
      return {
        data: {
          id: sid || oid,
          label: label || sid || oid,
          organ_id: oid,
          organ_class: "",
          organ_class_color: stub ? "#64748b" : organClassColor("endogenous"),
          signal_kind: kind,
          otel_trace_id: "",
          signal_id: sid,
          is_stub: stub,
          raw_json: JSON.stringify(row, null, 2),
        },
      };
    });
    const edges = [];
    const edgeKey = new Set();
    visible.forEach((row) => {
      const child = row && row.signal_id != null ? String(row.signal_id) : "";
      if (!child) return;
      const parents = Array.isArray(row.causal_parents) ? row.causal_parents : [];
      parents.forEach((pid, idx) => {
        const src = String(pid);
        if (!src || !visibleIds.has(src) || src === child) return;
        const ek = `${src}|${child}|${idx}`;
        if (edgeKey.has(ek)) return;
        edgeKey.add(ek);
        edges.push({
          data: {
            id: `e-${src}-${child}-${idx}`,
            source: src,
            target: child,
            label: "from",
          },
        });
      });
    });
    return { nodes, edges, visibleChain: visible };
  }

  function buildCorrelationLayout(chain) {
    const positions = {};
    const rows = Array.isArray(chain) ? chain : [];
    rows.forEach((row, i) => {
      const id = row && row.signal_id != null ? String(row.signal_id) : "";
      if (!id) return;
      positions[id] = { x: 72 + i * 132, y: 120 };
    });
    return {
      name: "preset",
      fit: true,
      padding: 28,
      rankDir: "LR",
      positions: function (node) {
        const id = node.id();
        return positions[id] || { x: 72, y: 120 };
      },
    };
  }

  /**
   * @param {object} options
   * @param {string} options.apiBaseUrl
   * @param {HTMLElement} options.cyHost
   * @param {HTMLElement} [options.statusEl]
   * @param {HTMLElement} [options.detailEl]
   * @param {HTMLButtonElement} [options.refreshBtn]
   * @param {HTMLInputElement} [options.autoRefreshCheckbox]
   * @param {HTMLSelectElement} [options.layerFilterEl]
   */
  function attach(options) {
    const apiBaseUrl = String(options.apiBaseUrl || "").replace(/\/$/, "");
    const cyHost = options.cyHost;
    if (!cyHost || typeof window.cytoscape !== "function") {
      return {
        refresh: function () {},
        destroy: function () {},
      };
    }

    let cy = null;
    let pollTimer = null;
    let lastPayload = null;
    let correlationId = parseCorrelationIdFromSearch();
    let correlationMode = Boolean(correlationId);
    let showStubs = false;
    let lastCorrelationChain = [];
    let stubToggleEl = null;
    let lastLayersMap = DEFAULT_ORGAN_LAYERS;

    function selectedLayer() {
      const el = options.layerFilterEl;
      if (!el || correlationMode) return "all";
      return String(el.value || "all").toLowerCase();
    }

    function setStatus(text) {
      if (options.statusEl) options.statusEl.textContent = text;
    }

    function setDetail(text) {
      if (options.detailEl) options.detailEl.textContent = text;
    }

    function countHiddenStubs(chain) {
      const rows = Array.isArray(chain) ? chain : [];
      return rows.filter((row) => isStubSignal(row)).length;
    }

    function syncStubToggle(hiddenCount) {
      if (!options.statusEl || !correlationMode) {
        if (stubToggleEl && stubToggleEl.parentNode) {
          stubToggleEl.parentNode.removeChild(stubToggleEl);
        }
        stubToggleEl = null;
        return;
      }
      const host = options.statusEl.parentElement;
      if (!host) return;
      if (!stubToggleEl) {
        stubToggleEl = document.createElement("label");
        stubToggleEl.className = "inline-flex items-center gap-2 text-xs text-amber-200/90 ml-3";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.className = "rounded border border-gray-600 bg-gray-800";
        input.addEventListener("change", () => {
          showStubs = input.checked;
          renderCorrelationGraph(lastCorrelationChain);
        });
        stubToggleEl.appendChild(input);
        const text = document.createElement("span");
        text.className = "organ-signals-stub-toggle-label";
        stubToggleEl.appendChild(text);
        host.insertBefore(stubToggleEl, options.statusEl.nextSibling);
      }
      const input = stubToggleEl.querySelector("input");
      const label = stubToggleEl.querySelector(".organ-signals-stub-toggle-label");
      if (input) input.checked = showStubs;
      if (label) {
        label.textContent = hiddenCount
          ? `Hidden stubs: ${hiddenCount} (show)`
          : "Hidden stubs: 0";
      }
    }

    function renderDetailFor(evt) {
      const t = evt && evt.target;
      if (!t || !options.detailEl) return;
      if (t.isNode && t.isNode()) {
        const raw = t.data("raw_json");
        setDetail(raw || t.data("label") || "");
        return;
      }
      if (t.isEdge && t.isEdge()) {
        setDetail(
          `Edge ${t.data("source")} → ${t.data("target")}\n${t.data("label") || ""}`.trim(),
        );
      }
    }

    async function fetchTrace(tid) {
      const url = `${apiBaseUrl}/api/signals/trace/${encodeURIComponent(tid)}`;
      const res = await fetch(url);
      if (res.status === 503) {
        return { error: "Trace cache disabled (Hub TRACE_CACHE_* / signals inspect settings)." };
      }
      if (res.status === 404) {
        return { error: "Trace not in Hub rolling cache (ttl / max traces, or different id)." };
      }
      if (!res.ok) {
        const body = await res.text();
        return { error: `HTTP ${res.status} ${body || ""}`.trim() };
      }
      return { body: await res.json() };
    }

    async function fetchCorrelation(corr) {
      const url = `${apiBaseUrl}/api/signals/correlation/${encodeURIComponent(corr)}`;
      const res = await fetch(url);
      if (res.status === 404) {
        return { error: "Correlation not in Hub cache (ttl / no matching source_event_id)." };
      }
      if (!res.ok) {
        const body = await res.text();
        return { error: `HTTP ${res.status} ${body || ""}`.trim() };
      }
      return { body: await res.json() };
    }

    async function onTapNode(evt) {
      renderDetailFor(evt);
      const t = evt && evt.target;
      if (!t || !t.isNode || !t.isNode()) return;
      const tid = String(t.data("otel_trace_id") || "").trim();
      if (!tid || tid.length < 32) return;
      setDetail("Loading trace…");
      const out = await fetchTrace(tid);
      if (out.error) {
        setDetail(`${t.data("raw_json") || ""}\n\n--- trace ---\n${out.error}`);
        return;
      }
      const b = out.body || {};
      const chain = Array.isArray(b.chain) ? b.chain : [];
      const lines = chain.map(
        (row, i) =>
          `${i + 1}. ${row.organ_id}  ${row.signal_kind || ""}  parents=${JSON.stringify(row.causal_parents || [])}`,
      );
      const grafana = b.grafana_explore_trace_url ? `\nGrafana: ${b.grafana_explore_trace_url}` : "";
      setDetail(
        `${t.data("raw_json") || ""}\n\n--- Hub trace ${b.trace_id || tid} ---\ncomplete=${b.complete}\ngaps=${JSON.stringify(b.gaps || [])}\n${lines.join("\n")}${grafana}`,
      );
    }

    function renderGraph(signalsMap, graphOptions) {
      destroyCy(cy);
      cy = null;
      const opts = graphOptions && typeof graphOptions === "object" ? graphOptions : {};
      const filtered =
        opts.showStubs === false
          ? Object.fromEntries(
              Object.entries(signalsMap || {}).filter(([, sig]) => !isStubSignal(sig)),
            )
          : signalsMap;
      const { nodes, edges } = buildGraphElements(filtered);
      if (!nodes.length) {
        cyHost.textContent =
          "No signals in Hub cache. Run orion-signal-gateway against Redis, ensure Hub bus + SIGNALS_INSPECT_ENABLED, and wait for orion:signals:* traffic.";
        return;
      }
      cyHost.textContent = "";
      cy = window.cytoscape({
        container: cyHost,
        elements: [...nodes, ...edges],
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              "font-size": 9,
              color: "#e2e8f0",
              "background-color": "data(organ_class_color)",
              width: 36,
              height: 36,
              "text-wrap": "wrap",
              "text-max-width": 80,
            },
          },
          {
            selector: "node[is_stub = true]",
            style: {
              "background-color": "#475569",
              "border-width": 2,
              "border-style": "dashed",
              "border-color": "#94a3b8",
            },
          },
          {
            selector: "edge",
            style: {
              width: 2,
              "line-color": "#64748b",
              "target-arrow-color": "#64748b",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              "font-size": 8,
              color: "#94a3b8",
            },
          },
        ],
        layout: { name: "cose", animate: false, randomize: true },
        wheelSensitivity: 0.35,
      });
      cy.on("tap", "node", onTapNode);
      cy.on("tap", "edge", renderDetailFor);
      requestAnimationFrame(() => {
        try {
          cy.resize();
          cy.fit(undefined, 28);
        } catch (_) {
          /* ignore */
        }
      });
    }

    function renderCorrelationGraph(chain) {
      destroyCy(cy);
      cy = null;
      lastCorrelationChain = Array.isArray(chain) ? chain : [];
      const hidden = countHiddenStubs(lastCorrelationChain);
      syncStubToggle(hidden);
      const { nodes, edges, visibleChain } = buildCorrelationGraphElements(lastCorrelationChain, {
        showStubs,
      });
      if (!nodes.length) {
        cyHost.textContent = correlationId
          ? `No non-stub signals for correlation_id=${correlationId}. Toggle stubs to reveal placeholders.`
          : "No correlation chain in Hub cache.";
        return;
      }
      cyHost.textContent = "";
      cy = window.cytoscape({
        container: cyHost,
        elements: [...nodes, ...edges],
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              "font-size": 9,
              color: "#e2e8f0",
              "background-color": "data(organ_class_color)",
              width: 36,
              height: 36,
              "text-wrap": "wrap",
              "text-max-width": 80,
            },
          },
          {
            selector: "node[is_stub = true]",
            style: {
              "background-color": "#475569",
              "border-width": 2,
              "border-style": "dashed",
              "border-color": "#94a3b8",
            },
          },
          {
            selector: "edge",
            style: {
              width: 2,
              "line-color": "#64748b",
              "target-arrow-color": "#64748b",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              "font-size": 8,
              color: "#94a3b8",
            },
          },
        ],
        layout: buildCorrelationLayout(visibleChain),
        wheelSensitivity: 0.35,
      });
      cy.on("tap", "node", renderDetailFor);
      cy.on("tap", "edge", renderDetailFor);
      requestAnimationFrame(() => {
        try {
          cy.resize();
          cy.fit(undefined, 28);
        } catch (_) {
          /* ignore */
        }
      });
    }

    async function refresh() {
      if (!apiBaseUrl) {
        setStatus("Missing API base URL.");
        return;
      }
      correlationId = parseCorrelationIdFromSearch();
      correlationMode = Boolean(correlationId);
      if (correlationMode) {
        setStatus(`Loading /api/signals/correlation/${correlationId}…`);
        try {
          const out = await fetchCorrelation(correlationId);
          if (out.error) {
            throw new Error(out.error);
          }
          const data = out.body || {};
          lastPayload = data;
          const chain = Array.isArray(data.chain) ? data.chain : [];
          const hidden =
            typeof data.hidden_stubs === "number" ? data.hidden_stubs : countHiddenStubs(chain);
          const sourceHint =
            data.source === "cognition_trace_fallback" ? "\tsource=cognition_trace" : "";
          setStatus(
            `correlation_id=${data.correlation_id || correlationId}\tsignals=${chain.length}\thidden_stubs=${hidden}${sourceHint}`,
          );
          renderCorrelationGraph(chain);
        } catch (err) {
          setStatus(`Failed: ${err && err.message ? err.message : err}`);
          destroyCy(cy);
          cy = null;
          cyHost.textContent = "";
          syncStubToggle(0);
        }
        return;
      }
      syncStubToggle(0);
      setStatus("Loading /api/signals/active…");
      try {
        const res = await fetch(`${apiBaseUrl}/api/signals/active`);
        if (!res.ok) {
          const t = await res.text();
          throw new Error(`${res.status} ${t || ""}`.trim());
        }
        const data = await res.json();
        lastPayload = data;
        const sigs = data.signals && typeof data.signals === "object" ? data.signals : {};
        lastLayersMap =
          data.layers && typeof data.layers === "object" ? data.layers : DEFAULT_ORGAN_LAYERS;
        const layerKey = selectedLayer();
        const filtered = filterSignalsByLayer(sigs, lastLayersMap, layerKey);
        const n = Object.keys(filtered).length;
        const hidden = countHiddenStubs(Object.values(sigs));
        setStatus(
          `as_of=${data.as_of || "?"}\torgans=${n}\tlayer=${layerKey}\thidden_stubs=${hidden}`,
        );
        renderGraph(filtered, { showStubs });
      } catch (err) {
        setStatus(`Failed: ${err && err.message ? err.message : err}`);
        destroyCy(cy);
        cy = null;
        cyHost.textContent = "";
      }
    }

    function stopPolling() {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    function syncPolling() {
      stopPolling();
      if (correlationMode) return;
      const el = options.autoRefreshCheckbox;
      if (el && el.checked) {
        pollTimer = setInterval(() => {
          refresh();
        }, 5000);
      }
    }

    if (options.refreshBtn) {
      options.refreshBtn.addEventListener("click", () => {
        refresh();
      });
    }
    if (options.autoRefreshCheckbox) {
      options.autoRefreshCheckbox.addEventListener("change", syncPolling);
    }
    if (options.layerFilterEl) {
      options.layerFilterEl.addEventListener("change", () => {
        if (!correlationMode && lastPayload && lastPayload.signals) {
          const sigs =
            lastPayload.signals && typeof lastPayload.signals === "object"
              ? lastPayload.signals
              : {};
          const layerKey = selectedLayer();
          const filtered = filterSignalsByLayer(sigs, lastLayersMap, layerKey);
          const hidden = countHiddenStubs(Object.values(sigs));
          setStatus(
            `as_of=${lastPayload.as_of || "?"}\torgans=${Object.keys(filtered).length}\tlayer=${layerKey}\thidden_stubs=${hidden}`,
          );
          renderGraph(filtered, { showStubs });
        } else {
          refresh();
        }
      });
    }

    return {
      refresh,
      destroy: function () {
        stopPolling();
        destroyCy(cy);
        cy = null;
        if (stubToggleEl && stubToggleEl.parentNode) {
          stubToggleEl.parentNode.removeChild(stubToggleEl);
        }
        stubToggleEl = null;
      },
      getLastPayload: function () {
        return lastPayload;
      },
    };
  }

  window.OrionOrganSignalsGraphUI = {
    attach,
    buildCorrelationGraphElements,
    filterSignalsByLayer,
    isStubSignal,
    parseCorrelationIdFromSearch,
    DEFAULT_ORGAN_LAYERS,
  };
})();
