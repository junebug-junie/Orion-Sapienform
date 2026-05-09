/* Orion Hub — organ signal causal graph from /api/signals/active (+ optional trace chain). */
(function () {
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

  /**
   * @param {object} options
   * @param {string} options.apiBaseUrl
   * @param {HTMLElement} options.cyHost
   * @param {HTMLElement} [options.statusEl]
   * @param {HTMLElement} [options.detailEl]
   * @param {HTMLButtonElement} [options.refreshBtn]
   * @param {HTMLInputElement} [options.autoRefreshCheckbox]
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

    function setStatus(text) {
      if (options.statusEl) options.statusEl.textContent = text;
    }

    function setDetail(text) {
      if (options.detailEl) options.detailEl.textContent = text;
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

    function renderGraph(signalsMap) {
      destroyCy(cy);
      cy = null;
      const { nodes, edges } = buildGraphElements(signalsMap);
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

    async function refresh() {
      if (!apiBaseUrl) {
        setStatus("Missing API base URL.");
        return;
      }
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
        const n = Object.keys(sigs).length;
        setStatus(`as_of=${data.as_of || "?"}\torgans=${n}`);
        renderGraph(sigs);
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

    return {
      refresh,
      destroy: function () {
        stopPolling();
        destroyCy(cy);
        cy = null;
      },
      getLastPayload: function () {
        return lastPayload;
      },
    };
  }

  window.OrionOrganSignalsGraphUI = { attach };
})();
