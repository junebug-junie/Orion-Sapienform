(function () {
  const pathSegments = window.location.pathname.split('/').filter((p) => p.length > 0);
  const URL_PREFIX = pathSegments.length > 0 ? `/${pathSegments[0]}` : "";
  const API_BASE_URL = window.location.origin + URL_PREFIX;

  const MAX_LINES = 1500;

  function createTerminalState(element) {
    return {
      element,
      lines: [],
      pending: [],
      flushing: false,
      autoScroll: true,
    };
  }

  function bindTerminalScroll(state) {
    state.element.addEventListener("scroll", () => {
      const threshold = 24;
      const distanceFromBottom = state.element.scrollHeight - state.element.scrollTop - state.element.clientHeight;
      state.autoScroll = distanceFromBottom <= threshold;
    });
  }

  function queueLine(state, line) {
    state.pending.push(line);
    if (!state.flushing) {
      state.flushing = true;
      requestAnimationFrame(() => flushTerminal(state));
    }
  }

  function flushTerminal(state) {
    state.flushing = false;
    if (state.pending.length === 0) return;
    state.lines.push(...state.pending);
    state.pending = [];
    if (state.lines.length > MAX_LINES) {
      state.lines = state.lines.slice(state.lines.length - MAX_LINES);
    }
    state.element.textContent = state.lines.join("\n");
    if (state.autoScroll) {
      state.element.scrollTop = state.element.scrollHeight;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const selectEl = document.getElementById("serviceLogsServiceSelect");
    const statusEl = document.getElementById("serviceLogsStatus");
    const masterEl = document.getElementById("serviceLogsMasterTerminal");
    const terminalsGrid = document.getElementById("serviceLogsTerminals");

    if (!selectEl || !statusEl || !masterEl || !terminalsGrid) {
      return;
    }

    const masterState = createTerminalState(masterEl);
    bindTerminalScroll(masterState);

    const terminalStates = new Map();
    let socket = null;
    let reconnectTimer = null;
    let selectedServices = [];
    let inventory = [];

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function ensureServiceTerminal(serviceName) {
      if (terminalStates.has(serviceName)) {
        return terminalStates.get(serviceName);
      }

      const card = document.createElement("section");
      card.className = "bg-gray-900 rounded-xl border border-gray-800 p-3 flex flex-col gap-2 min-h-[14rem]";
      card.dataset.serviceName = serviceName;

      const heading = document.createElement("div");
      heading.className = "text-xs uppercase tracking-wide text-indigo-300";
      heading.textContent = serviceName;

      const terminal = document.createElement("pre");
      terminal.className = "h-[18rem] max-h-[18rem] bg-gray-950/80 border border-gray-800 rounded-lg p-3 overflow-auto text-[11px] leading-4 text-gray-200 font-mono";
      terminal.textContent = "";

      card.appendChild(heading);
      card.appendChild(terminal);
      terminalsGrid.appendChild(card);

      const state = createTerminalState(terminal);
      bindTerminalScroll(state);
      terminalStates.set(serviceName, { card, state });
      return terminalStates.get(serviceName);
    }

    function removeServiceTerminal(serviceName) {
      const entry = terminalStates.get(serviceName);
      if (!entry) return;
      entry.card.remove();
      terminalStates.delete(serviceName);
    }

    function syncTerminals() {
      const selectedSet = new Set(selectedServices);
      Array.from(terminalStates.keys()).forEach((name) => {
        if (!selectedSet.has(name)) {
          removeServiceTerminal(name);
        }
      });
      selectedServices.forEach((name) => {
        ensureServiceTerminal(name);
      });
    }

    function selectedFromControl() {
      return Array.from(selectEl.selectedOptions).map((opt) => opt.value).filter(Boolean);
    }

    function sendSelection() {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({ action: "subscribe", services: selectedServices }));
    }

    function renderInventory(options) {
      inventory = options.slice().sort();
      const currentSet = new Set(selectedServices);
      selectEl.innerHTML = "";
      inventory.forEach((serviceName) => {
        const option = document.createElement("option");
        option.value = serviceName;
        option.textContent = serviceName;
        option.selected = currentSet.has(serviceName);
        selectEl.appendChild(option);
      });
      const valid = selectedServices.filter((name) => inventory.includes(name));
      if (valid.length !== selectedServices.length) {
        selectedServices = valid;
        syncTerminals();
      }
    }

    function diagnosticsHint(meta) {
      if (!meta || typeof meta !== "object") return "";
      const hints = [];
      if (meta.repo_root) hints.push(`repo_root=${meta.repo_root}`);
      if (meta.services_root) hints.push(`services_root=${meta.services_root}`);
      if (meta.services_root_exists === false) hints.push("services_root_missing");
      if (meta.root_env_exists === false) hints.push("root_env_missing");
      if (meta.docker_available === false) hints.push("docker_cli_missing");
      if (meta.docker_socket_exists === false) hints.push("docker_socket_missing");
      return hints.join(" | ");
    }

    function appendLog(serviceName, line, stream) {
      const entry = ensureServiceTerminal(serviceName);
      queueLine(entry.state, line);
      const prefixed = `[${serviceName}${stream === "stderr" ? ":stderr" : ""}] ${line}`;
      queueLine(masterState, prefixed);
    }

    async function loadInventory() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/service-logs/services`);
        if (!response.ok) {
          throw new Error(`inventory status=${response.status}`);
        }
        const payload = await response.json();
        const names = Array.isArray(payload.services) ? payload.services.map((item) => item.name).filter(Boolean) : [];
        renderInventory(names);
        const meta = payload && typeof payload.meta === "object" ? payload.meta : {};
        if (names.length === 0) {
          const rootHint = meta.services_root || meta.repo_root || "services/";
          const hint = diagnosticsHint(meta);
          setStatus(`No loggable services found under ${rootHint}. ${hint}`.trim());
        }
      } catch (err) {
        console.warn("[ServiceLogs] Failed to load inventory", err);
        setStatus(`Inventory error: ${err.message || err}`);
      }
    }

    function connectSocket() {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${proto}//${window.location.host}${URL_PREFIX}/ws/service-logs`;
      socket = new WebSocket(wsUrl);
      setStatus("Connecting log stream...");

      socket.onopen = () => {
        setStatus("Live log stream connected.");
        sendSelection();
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data || "{}");
          if (payload.type === "service_inventory" && Array.isArray(payload.services)) {
            renderInventory(payload.services);
            const meta = payload && typeof payload.meta === "object" ? payload.meta : {};
            if (payload.services.length === 0) {
              const hint = diagnosticsHint(meta);
              if (hint) setStatus(`No services discovered. ${hint}`);
            }
            sendSelection();
            return;
          }
          if (payload.type === "selection_updated" && Array.isArray(payload.services)) {
            selectedServices = payload.services;
            renderInventory(inventory);
            syncTerminals();
            return;
          }
          if (payload.type === "log_line" && payload.service && typeof payload.line === "string") {
            appendLog(payload.service, payload.line, payload.stream || "stdout");
            return;
          }
          if (payload.type === "error") {
            setStatus(`Log stream error: ${payload.error || "unknown"}`);
          }
        } catch (err) {
          console.warn("[ServiceLogs] WS parse error", err);
        }
      };

      socket.onclose = () => {
        setStatus("Log stream disconnected. Reconnecting...");
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connectSocket, 2000);
      };

      socket.onerror = () => {
        setStatus("Log stream socket error.");
      };
    }

    selectEl.addEventListener("change", () => {
      selectedServices = selectedFromControl();
      syncTerminals();
      sendSelection();
    });

    loadInventory();
    connectSocket();
  });
})();
