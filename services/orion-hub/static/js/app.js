// services/orion-hub/static/js/app.js

// ───────────────────────────────────────────────────────────────
// Global State
// ───────────────────────────────────────────────────────────────
const API_BASE_URL = window.location.origin;
const VISION_EDGE_BASE = "https://athena.tail348bbe.ts.net/vision-edge";

let socket;
let mediaRecorder;
let audioChunks = [];
let audioContext = new (window.AudioContext || window.webkitAudioContext)();
let currentAudioSource = null;
let analyser;
let animationFrameId;
let audioQueue = [];
let isPlayingAudio = false;
let orionState = 'idle';
let particles = [];
let visionIsFloating = false;
let currentMode = "brain";
let selectedPacks = [];
let selectedVerbs = [];
let orionSessionId = null;
let cognitionLibrary = { packs: {}, verbs: [], map: {} };
let selectedBiometricsNode = "cluster";
let lastBiometricsPayload = null;
let topicAutoRefreshTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  console.log("[Main] DOM Content Loaded - Initializing UI...");

  // --- 1. Element References ---
  const recordButton = document.getElementById('recordButton');
  const interruptButton = document.getElementById('interruptButton');
  const statusDiv = document.getElementById('status');
  const conversationDiv = document.getElementById('conversation');
  const chatInput = document.getElementById('chatInput');
  const sendButton = document.getElementById('sendButton');
  const textToSpeechToggle = document.getElementById('textToSpeechToggle');
  const recallToggle = document.getElementById('recallToggle');
  const recallRequiredToggle = document.getElementById('recallRequiredToggle');
  const recallModeSelect = document.getElementById('recallModeSelect');
  const recallProfileSelect = document.getElementById('recallProfileSelect');

  // Controls
  const speedControl = document.getElementById('speedControl');
  const speedValue = document.getElementById('speedValue');
  const tempControl = document.getElementById('tempControl');
  const tempValue = document.getElementById('tempValue');
  const contextControl = document.getElementById('contextControl');
  const contextValue = document.getElementById('contextValue');
  const clearButton = document.getElementById('clearButton');
  const copyButton = document.getElementById('copyButton');

  // Settings
  const settingsToggle = document.getElementById('settingsToggle');
  const settingsPanel = document.getElementById('settingsPanel');
  const settingsClose = document.getElementById('settingsClose');

  // Visualizers
  const visualizerCanvas = document.getElementById('visualizer');
  const canvasCtx = visualizerCanvas ? visualizerCanvas.getContext('2d') : null;
  const visualizerContainer = document.getElementById('visualizerContainer');
  
  // NOTE: stateVisualizer was replaced by an iframe in the HTML. 
  // We check for its existence to prevent crashes.
  const stateVisualizerCanvas = document.getElementById('stateVisualizer');
  const stateCtx = stateVisualizerCanvas ? stateVisualizerCanvas.getContext('2d') : null;
  const stateVisualizerContainer = document.getElementById('stateVisualizerContainer');

  // Vision
  const visionPopoutButton = document.getElementById("visionPopoutButton");
  const visionDockedContainer = document.getElementById("visionDockedContainer");
  const visionFloatingContainer = document.getElementById("visionFloatingContainer");
  const visionCloseFloatingButton = document.getElementById("visionCloseFloating");
  const visionSourceSelect = document.getElementById("visionSource");

  // Biometrics
  const biometricsPanel = document.getElementById("biometricsPanel");
  const bioStatus = document.getElementById("bioStatus");
  const bioConstraint = document.getElementById("bioConstraint");
  const bioNodeSelect = document.getElementById("bioNodeSelect");
  const bioStrainValue = document.getElementById("bioStrainValue");
  const bioStrainTrend = document.getElementById("bioStrainTrend");
  const bioHomeostasisValue = document.getElementById("bioHomeostasisValue");
  const bioHomeostasisTrend = document.getElementById("bioHomeostasisTrend");
  const bioStabilityValue = document.getElementById("bioStabilityValue");
  const bioStabilityTrend = document.getElementById("bioStabilityTrend");

  // Topic Rail
  const topicWindowSelect = document.getElementById("topicWindowSelect");
  const topicMaxSelect = document.getElementById("topicMaxSelect");
  const topicMinTurnsSelect = document.getElementById("topicMinTurnsSelect");
  const topicModelVersion = document.getElementById("topicModelVersion");
  const topicRefreshButton = document.getElementById("topicRefreshButton");
  const topicAutoRefresh = document.getElementById("topicAutoRefresh");
  const topicRailStatus = document.getElementById("topicRailStatus");
  const topicRailRetry = document.getElementById("topicRailRetry");
  const topicSummaryBody = document.getElementById("topicSummaryBody");
  const topicDriftBody = document.getElementById("topicDriftBody");
  const topicSummaryMeta = document.getElementById("topicSummaryMeta");
  const topicDriftMeta = document.getElementById("topicDriftMeta");
  const toastContainer = document.getElementById("toastContainer");
  const toastMessage = document.getElementById("toastMessage");

  // Collapse Mirror
  const collapseModeGuided = document.getElementById('collapseModeGuided');
  const collapseModeRaw = document.getElementById('collapseModeRaw');
  const collapseGuidedSection = document.getElementById('collapseGuidedSection');
  const collapseRawSection = document.getElementById('collapseRawSection');
  const collapseStatus = document.getElementById('collapseStatus');
  const collapseTooltipToggle = document.getElementById('collapseTooltipToggle');
  const collapseTooltip = document.getElementById('collapseTooltip');

  // Packs & Verbs
  const packContainer = document.getElementById('packContainer');
  const verbSelectTrigger = document.getElementById('verbSelectTrigger');
  const verbSelectLabel = document.getElementById('verbSelectLabel');
  const verbDropdown = document.getElementById('verbDropdown');
  const verbList = document.getElementById('verbList');
  const clearVerbsBtn = document.getElementById('clearVerbs');

  // --- 2. Helpers ---

  function updateStatus(msg) { 
      if (statusDiv) {
          statusDiv.textContent = msg; 
          // Visual cue for disconnected state
          if(msg.includes("Disconnected") || msg.includes("Error")) {
              statusDiv.classList.add("text-red-400");
          } else {
              statusDiv.classList.remove("text-red-400");
          }
      }
  }

  function showToast(message) {
    if (!toastContainer || !toastMessage) return;
    toastMessage.textContent = message;
    toastContainer.classList.remove("hidden");
    setTimeout(() => {
      toastContainer.classList.add("hidden");
    }, 4000);
  }

  function updateStatusBasedOnState() {
    if (orionState === 'idle') updateStatus('Ready.');
    else if (orionState === 'speaking') updateStatus('Speaking...');
    else if (orionState === 'processing') updateStatus('Processing...');
  }

  function appendMessage(sender, text, colorClass = 'text-white') {
    if (!conversationDiv) return;
    const div = document.createElement('div');
    const color = sender === 'You' ? 'text-blue-300' : 'text-green-300';
    const header = document.createElement('p');
    header.className = `font-bold ${color}`;
    header.textContent = sender;
    const body = document.createElement('p');
    body.className = `${colorClass} whitespace-pre-wrap`;
    body.textContent = text || "";
    div.className = "mb-2 border-b border-gray-800/50 pb-2 last:border-0";
    div.appendChild(header);
    div.appendChild(body);
    conversationDiv.appendChild(div);
    conversationDiv.scrollTop = conversationDiv.scrollHeight;
  }

  function formatMetric(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "--";
    return `${(Number(value) * 100).toFixed(0)}%`;
  }

  function formatPercent(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) return "--";
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) return "--";
    return Number(value).toFixed(digits);
  }

  function trendArrow(trendValue) {
    const eps = 0.001;
    const normalized = Number(trendValue);
    if (Number.isNaN(normalized)) return "→";
    if (normalized > eps) return "↗";
    if (normalized < -eps) return "↘";
    return "→";
  }

  function updateBiometricsPanel(biometrics) {
    if (!biometricsPanel) return;
    lastBiometricsPayload = biometrics;
    if (bioNodeSelect) {
      const nodes = Object.keys(biometrics?.nodes || {});
      const options = ["cluster", ...nodes];
      const existing = Array.from(bioNodeSelect.options).map((opt) => opt.value);
      const changed = options.length !== existing.length || options.some((opt, i) => opt !== existing[i]);
      if (changed) {
        bioNodeSelect.innerHTML = "";
        options.forEach((opt) => {
          const option = document.createElement("option");
          option.value = opt;
          option.textContent = opt === "cluster" ? "Cluster" : opt;
          bioNodeSelect.appendChild(option);
        });
      }
      if (!options.includes(selectedBiometricsNode)) {
        selectedBiometricsNode = "cluster";
      }
      bioNodeSelect.value = selectedBiometricsNode;
    }
    const selectedNodePayload =
      selectedBiometricsNode !== "cluster"
        ? biometrics?.nodes?.[selectedBiometricsNode]
        : null;
    const status = selectedNodePayload?.status || biometrics?.status || "NO_SIGNAL";
    const freshness =
      selectedNodePayload?.freshness_s !== undefined
        ? selectedNodePayload?.freshness_s
        : biometrics?.freshness_s;
    const displayFreshness =
      typeof freshness === "number" && Number.isFinite(freshness) ? `${freshness.toFixed(0)}s` : "--";
    const statusLabel = status === "OK" ? "LIVE" : String(status).replace(/_/g, " ");
    if (bioStatus) {
      bioStatus.textContent = `${statusLabel} • ${displayFreshness}`;
    }
    const constraint =
      selectedNodePayload?.summary?.constraint || biometrics?.constraint || "NONE";
    if (bioConstraint) {
      if (constraint && constraint !== "NONE") {
        bioConstraint.textContent = constraint;
        bioConstraint.classList.remove("hidden");
      } else {
        bioConstraint.classList.add("hidden");
      }
    }
    let composite = biometrics?.cluster?.composite || {};
    let trend = biometrics?.cluster?.trend || {};
    if (selectedBiometricsNode !== "cluster") {
      const node = biometrics?.nodes?.[selectedBiometricsNode] || {};
      composite = node?.summary?.composites || {};
      trend = node?.induction?.metrics || {};
    }
    const strainTrend = (trend?.strain?.trend ?? 0.5) - 0.5;
    const homeostasisTrend = (trend?.homeostasis?.trend ?? 0.5) - 0.5;
    const stabilityTrend = (trend?.stability?.trend ?? 0.5) - 0.5;

    if (bioStrainValue) bioStrainValue.textContent = formatMetric(composite?.strain ?? null);
    if (bioHomeostasisValue) bioHomeostasisValue.textContent = formatMetric(composite?.homeostasis ?? null);
    if (bioStabilityValue) bioStabilityValue.textContent = formatMetric(composite?.stability ?? null);

    if (bioStrainTrend) bioStrainTrend.textContent = trendArrow(strainTrend);
    if (bioHomeostasisTrend) bioHomeostasisTrend.textContent = trendArrow(homeostasisTrend);
    if (bioStabilityTrend) bioStabilityTrend.textContent = trendArrow(stabilityTrend);
  }

  function setTopicRailStatus(text, isError = false) {
    if (!topicRailStatus) return;
    topicRailStatus.textContent = text;
    if (isError) {
      topicRailStatus.classList.add("text-red-300");
    } else {
      topicRailStatus.classList.remove("text-red-300");
    }
  }

  function clearTopicRailTables() {
    if (topicSummaryBody) topicSummaryBody.innerHTML = "";
    if (topicDriftBody) topicDriftBody.innerHTML = "";
  }

  function parseKeywords(value) {
    if (!value) return [];
    if (Array.isArray(value)) return value;
    if (typeof value === "string") {
      try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed : [value];
      } catch (err) {
        return [value];
      }
    }
    return [];
  }

  function renderTopicSummary(topics) {
    if (!topicSummaryBody) return;
    topicSummaryBody.innerHTML = "";
    if (!topics || topics.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="5" class="py-3 text-center text-gray-500">No topics found.</td>`;
      topicSummaryBody.appendChild(row);
      return;
    }

    topics.forEach((topic) => {
      const row = document.createElement("tr");
      const keywords = parseKeywords(topic.topic_keywords);
      const outlierPct = topic.outlier_pct;
      const outlierCount = topic.outlier_count;
      const outlierValue =
        outlierPct !== null && outlierPct !== undefined
          ? formatPercent(outlierPct, 1)
          : outlierCount !== null && outlierCount !== undefined
            ? `${outlierCount}`
            : "--";

      const label = topic.topic_label || "Untitled";
      const topicId = topic.topic_id !== null && topic.topic_id !== undefined ? String(topic.topic_id) : "--";
      const keywordChips = keywords.length
        ? keywords
            .slice(0, 5)
            .map(
              (word) =>
                `<span class="inline-flex items-center rounded-full bg-gray-700/70 px-2 py-0.5 text-[10px] text-gray-200">${word}</span>`,
            )
            .join(" ")
        : `<span class="text-gray-500">--</span>`;

      row.innerHTML = `
        <td class="py-2 pr-3 align-top">
          <div class="text-sm text-gray-100">${label}</div>
          <div class="text-[10px] text-gray-500">#${topicId}</div>
        </td>
        <td class="py-2 pr-3 align-top">${keywordChips}</td>
        <td class="py-2 pr-3 align-top">${topic.doc_count ?? "--"}</td>
        <td class="py-2 pr-3 align-top">${formatPercent(topic.pct_of_window, 1)}</td>
        <td class="py-2 pr-3 align-top">${outlierValue}</td>
      `;
      topicSummaryBody.appendChild(row);
    });
  }

  function renderTopicDrift(sessions, topicLabelMap) {
    if (!topicDriftBody) return;
    topicDriftBody.innerHTML = "";
    if (!sessions || sessions.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="6" class="py-3 text-center text-gray-500">No drifting sessions found.</td>`;
      topicDriftBody.appendChild(row);
      return;
    }

    sessions.forEach((session) => {
      const row = document.createElement("tr");
      const sessionId = session.session_id || "--";
      const shortId = sessionId.length > 10 ? `${sessionId.slice(0, 10)}…` : sessionId;
      const switchRate = Number(session.switch_rate);
      const entropy = Number(session.entropy);
      const switchClass = switchRate >= 0.35 ? "text-yellow-300 font-semibold" : "";
      const entropyClass = entropy >= 1.5 ? "text-red-300 font-semibold" : "";
      const dominantId = session.dominant_topic_id;
      const dominantLabel = dominantId !== null && dominantId !== undefined ? topicLabelMap.get(dominantId) : null;
      const dominantDisplay = dominantLabel ? `${dominantLabel} (#${dominantId})` : dominantId ?? "--";

      row.innerHTML = `
        <td class="py-2 pr-3 align-top" title="${sessionId}">${shortId}</td>
        <td class="py-2 pr-3 align-top">${session.turns ?? "--"}</td>
        <td class="py-2 pr-3 align-top">${session.unique_topics ?? "--"}</td>
        <td class="py-2 pr-3 align-top ${entropyClass}">${formatNumber(entropy)}</td>
        <td class="py-2 pr-3 align-top ${switchClass}">${formatPercent(switchRate, 1)}</td>
        <td class="py-2 pr-3 align-top">
          <div>${dominantDisplay ?? "--"}</div>
          <div class="text-[10px] text-gray-500">${formatPercent(session.dominant_pct, 1)}</div>
        </td>
      `;
      topicDriftBody.appendChild(row);
    });
  }

  async function fetchTopicRailData() {
    if (!topicWindowSelect || !topicMaxSelect || !topicMinTurnsSelect) return;
    setTopicRailStatus("Loading...");
    if (topicRailRetry) topicRailRetry.classList.add("hidden");

    const windowMinutes = topicWindowSelect.value;
    const maxTopics = topicMaxSelect.value;
    const minTurns = topicMinTurnsSelect.value;
    const modelVersion = topicModelVersion?.value?.trim();

    const summaryParams = new URLSearchParams({
      window_minutes: windowMinutes,
      max_topics: maxTopics,
    });
    if (modelVersion) summaryParams.set("model_version", modelVersion);

    const driftParams = new URLSearchParams({
      window_minutes: windowMinutes,
      min_turns: minTurns,
    });
    if (modelVersion) driftParams.set("model_version", modelVersion);

    try {
      const [summaryResp, driftResp] = await Promise.all([
        fetch(`${API_BASE_URL}/api/topics/summary?${summaryParams.toString()}`),
        fetch(`${API_BASE_URL}/api/topics/drift?${driftParams.toString()}`),
      ]);

      if (!summaryResp.ok || !driftResp.ok) {
        throw new Error("Topic Rail request failed.");
      }

      const summaryPayload = await summaryResp.json();
      const driftPayload = await driftResp.json();

      const topics = summaryPayload.topics || [];
      const sessions = driftPayload.sessions || [];
      const summaryModel = summaryPayload.model_version || "latest";
      const driftModel = driftPayload.model_version || "latest";
      if (topicSummaryMeta) {
        topicSummaryMeta.textContent = `model: ${summaryModel}`;
      }
      if (topicDriftMeta) {
        topicDriftMeta.textContent = `model: ${driftModel}`;
      }

      const topicLabelMap = new Map();
      topics.forEach((topic) => {
        if (topic.topic_id !== null && topic.topic_id !== undefined) {
          topicLabelMap.set(topic.topic_id, topic.topic_label || `Topic ${topic.topic_id}`);
        }
      });

      renderTopicSummary(topics);
      renderTopicDrift(sessions, topicLabelMap);
      setTopicRailStatus(`Updated ${new Date().toLocaleTimeString()}.`);
    } catch (err) {
      console.error("[TopicRail]", err);
      clearTopicRailTables();
      setTopicRailStatus("Failed to load Topic Rail data.", true);
      if (topicRailRetry) topicRailRetry.classList.remove("hidden");
      showToast("Topic Rail failed to load. Check Landing Pad connectivity.");
    }
  }

  function startTopicAutoRefresh() {
    if (topicAutoRefreshTimer) {
      clearInterval(topicAutoRefreshTimer);
      topicAutoRefreshTimer = null;
    }
    if (topicAutoRefresh?.checked) {
      topicAutoRefreshTimer = setInterval(fetchTopicRailData, 60000);
    }
  }

  // --- 3. Event Listeners ---

  if (recordButton) {
    recordButton.addEventListener('mousedown', startRecording);
    recordButton.addEventListener('mouseup', stopRecording);
    recordButton.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    recordButton.addEventListener('touchend', stopRecording);
  }

  if (sendButton) sendButton.addEventListener('click', sendTextMessage);
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
          e.preventDefault(); 
          sendTextMessage();
      }
    });
  }

  if (interruptButton) {
    interruptButton.addEventListener('click', () => {
      if (currentAudioSource) currentAudioSource.stop();
      audioQueue = [];
      isPlayingAudio = false;
      updateStatusBasedOnState();
      interruptButton.classList.add('hidden');
    });
  }

  // UI Toggles
  if (settingsToggle && settingsPanel) {
    settingsToggle.addEventListener('click', () => {
      settingsPanel.classList.toggle('translate-x-full');
      settingsPanel.classList.toggle('translate-x-0');
    });
  }
  if (settingsClose && settingsPanel) {
    settingsClose.addEventListener('click', () => {
      settingsPanel.classList.add('translate-x-full');
      settingsPanel.classList.remove('translate-x-0');
    });
  }

  if (clearButton && conversationDiv) clearButton.addEventListener('click', () => conversationDiv.innerHTML = '');
  if (copyButton && conversationDiv) {
    copyButton.addEventListener('click', () => {
      navigator.clipboard.writeText(conversationDiv.innerText);
      const originalText = copyButton.textContent;
      copyButton.textContent = "Copied!";
      setTimeout(() => copyButton.textContent = originalText, 1500);
    });
  }

  // Mode Switching
  const modeButtons = document.querySelectorAll('.mode-btn');
  modeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      currentMode = btn.dataset.mode || 'brain';
      modeButtons.forEach(b => {
        b.classList.remove('bg-indigo-600', 'text-white');
        b.classList.add('bg-gray-700', 'text-gray-200');
      });
      btn.classList.add('bg-indigo-600', 'text-white');
      btn.classList.remove('bg-gray-700', 'text-gray-200');
      updateStatus(`Switched to ${currentMode} mode.`);
    });
  });

  // --- 4. Logic Functions ---

  async function loadCognitionLibrary() {
      try {
          const res = await fetch(`${API_BASE_URL}/api/cognition/library`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          cognitionLibrary = await res.json();
          renderPackButtons();
          renderVerbList();
      } catch (e) {
          console.error("Failed to load packs:", e);
          if (packContainer) packContainer.innerHTML = '<span class="text-red-400 text-xs">Error loading packs</span>';
      }
  }

  function renderPackButtons() {
      if (!packContainer) return;
      packContainer.innerHTML = '';
      const packs = Object.keys(cognitionLibrary.packs || {});
      
      if (packs.length === 0) {
          packContainer.innerHTML = '<span class="text-gray-500 text-xs">No packs available.</span>';
          return;
      }

      const defaults = ['executive_pack'];
      packs.forEach(packName => {
          const btn = document.createElement('button');
          btn.className = `pack-btn inline-flex items-center px-2 py-1 rounded text-[10px] bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors border border-gray-600`;
          btn.textContent = cognitionLibrary.packs[packName].label || packName;
          btn.dataset.pack = packName;

          if (defaults.includes(packName) && !selectedPacks.includes(packName)) {
               selectedPacks.push(packName);
          }

          if (selectedPacks.includes(packName)) {
               btn.classList.remove('bg-gray-700', 'text-gray-300');
               btn.classList.add('bg-emerald-700', 'text-white', 'border-emerald-500');
          }

          btn.addEventListener('click', () => togglePack(packName));
          packContainer.appendChild(btn);
      });
      renderVerbList();
  }

  function togglePack(packName) {
      const idx = selectedPacks.indexOf(packName);
      if (idx >= 0) selectedPacks.splice(idx, 1);
      else selectedPacks.push(packName);
      
      // Update styling
      document.querySelectorAll('.pack-btn').forEach(btn => {
          if (selectedPacks.includes(btn.dataset.pack)) {
              btn.classList.remove('bg-gray-700', 'text-gray-300');
              btn.classList.add('bg-emerald-700', 'text-white', 'border-emerald-500');
          } else {
              btn.classList.remove('bg-emerald-700', 'text-white', 'border-emerald-500');
              btn.classList.add('bg-gray-700', 'text-gray-300');
          }
      });
      renderVerbList();
  }

  function renderVerbList() {
      if (!verbList) return;
      verbList.innerHTML = '';
      let availableVerbs = [];

      if (selectedPacks.length === 0) {
          availableVerbs = cognitionLibrary.verbs || [];
      } else {
          const verbSet = new Set();
          selectedPacks.forEach(p => {
              const pVerbs = cognitionLibrary.map[p] || [];
              pVerbs.forEach(v => verbSet.add(v));
          });
          availableVerbs = Array.from(verbSet).sort();
      }

      if (availableVerbs.length === 0) {
          verbList.innerHTML = '<span class="text-gray-500 text-xs p-2">No verbs found.</span>';
          return;
      }

      availableVerbs.forEach(verb => {
          const div = document.createElement('div');
          div.className = "flex items-center gap-2 p-1 hover:bg-gray-700 rounded cursor-pointer";
          const cb = document.createElement('input');
          cb.type = "checkbox";
          cb.className = "form-checkbox h-3 w-3 text-red-600 bg-gray-600 border-gray-500 rounded focus:ring-red-500";
          cb.checked = selectedVerbs.includes(verb);
          
          const toggle = () => {
              const idx = selectedVerbs.indexOf(verb);
              if (idx >= 0) selectedVerbs.splice(idx, 1);
              else selectedVerbs.push(verb);
              cb.checked = selectedVerbs.includes(verb);
              updateVerbLabel();
          };

          cb.addEventListener('change', toggle);
          div.addEventListener('click', (e) => { if(e.target !== cb) toggle(); });

          const span = document.createElement('span');
          span.textContent = verb;
          span.className = "text-xs text-gray-300";
          
          div.appendChild(cb);
          div.appendChild(span);
          verbList.appendChild(div);
      });
      updateVerbLabel();
  }

  function updateVerbLabel() {
      if (!verbSelectLabel) return;
      if (selectedVerbs.length === 0) {
          verbSelectLabel.textContent = "All verbs available";
          verbSelectLabel.className = "text-gray-400 italic";
      } else {
          verbSelectLabel.textContent = `${selectedVerbs.length} verbs selected`;
          verbSelectLabel.className = "text-white font-semibold";
      }
  }

  // Verbs Dropdown
  if (verbSelectTrigger) {
      verbSelectTrigger.addEventListener('click', (e) => {
          e.stopPropagation();
          verbDropdown.classList.toggle('hidden');
      });
      document.addEventListener('click', (e) => {
          if (verbDropdown && !verbDropdown.classList.contains('hidden')) {
              if (!verbDropdown.contains(e.target) && !verbSelectTrigger.contains(e.target)) {
                  verbDropdown.classList.add('hidden');
              }
          }
      });
  }

  // --- WebSocket ---
  function setupWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/ws`;
    
    console.log(`[WS] Connecting to ${wsUrl}...`);
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("[WS] Connected");
        updateStatus('Connected.');
    };

    socket.onmessage = (e) => {
      try {
          const d = JSON.parse(e.data);
          if (d.transcript && !d.is_text_input) appendMessage('You', d.transcript);
          if (d.llm_response) appendMessage('Orion', d.llm_response);
          if (d.state) { orionState = d.state; updateStatusBasedOnState(); }
          if (d.audio_response) { audioQueue.push(d.audio_response); processAudioQueue(); }
          if (d.error) appendMessage('System', `Error: ${d.error}`, 'text-red-400');
          if (d.biometrics) updateBiometricsPanel(d.biometrics);
      } catch (err) {
          console.error("WS Parse Error", err);
      }
    };

    socket.onclose = (e) => {
        console.warn("[WS] Closed", e.code, e.reason);
        updateStatus('Disconnected. Reconnecting...');
        setTimeout(setupWebSocket, 2000);
    };

    socket.onerror = (err) => {
        console.error("[WS] Error", err);
        updateStatus('Connection Error.');
    };
  }

  function sendTextMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    appendMessage('You', text);
    chatInput.value = '';

    const recallMode = recallModeSelect ? recallModeSelect.value : "auto";
    const recallProfile = recallProfileSelect ? recallProfileSelect.value : "auto";
    const payload = {
       text_input: text,
       mode: currentMode,
       session_id: orionSessionId,
       disable_tts: textToSpeechToggle ? !textToSpeechToggle.checked : false,
       use_recall: recallToggle ? recallToggle.checked : false,
       recall_mode: recallMode !== "auto" ? recallMode : null,
       recall_profile: recallProfile !== "auto" ? recallProfile : null,
       recall_required: recallRequiredToggle ? recallRequiredToggle.checked : false,
       packs: selectedPacks,
       verbs: selectedVerbs,
    };

    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(payload));
        updateStatus('Sent...');
    } else {
        appendMessage('System', 'WebSocket not connected. Trying HTTP fallback...', 'text-yellow-400');
        // Fallback to HTTP if WS is down
        fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({messages:[{role:'user', content:text}], ...payload})
        })
        .then(r => r.json())
        .then(d => {
            if(d.text) appendMessage('Orion', d.text);
            else if(d.error) appendMessage('System', d.error, 'text-red-400');
        })
        .catch(e => appendMessage('System', "HTTP Failed: " + e.message, 'text-red-400'));
    }
  }

  // --- Audio ---
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks);
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
           if(socket && socket.readyState === WebSocket.OPEN) {
              socket.send(JSON.stringify({
               audio: reader.result.split(',')[1],
               mode: currentMode,
               session_id: orionSessionId,
               use_recall: recallToggle ? recallToggle.checked : false,
               recall_mode: recallModeSelect && recallModeSelect.value !== "auto" ? recallModeSelect.value : null,
               recall_profile: recallProfileSelect && recallProfileSelect.value !== "auto" ? recallProfileSelect.value : null,
               recall_required: recallRequiredToggle ? recallRequiredToggle.checked : false
             }));
             updateStatus('Audio sent.');
           } else {
               updateStatus('Offline. Cannot send audio.');
           }
        };
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      updateStatus('Recording...');
      recordButton.classList.add('pulse');
    } catch (e) { console.error(e); updateStatus('Mic Access Denied'); }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      recordButton.classList.remove('pulse');
    }
  }

  function processAudioQueue() {
    if (isPlayingAudio || !audioQueue.length) return;
    isPlayingAudio = true;
    playAudio(audioQueue.shift());
  }

  async function playAudio(b64) {
    try {
        const bin = atob(b64);
        const arr = new Uint8Array(bin.length);
        for(let i=0; i<bin.length; i++) arr[i] = bin.charCodeAt(i);
        const buf = await audioContext.decodeAudioData(arr.buffer);
        const src = audioContext.createBufferSource();
        src.buffer = buf;
        const gain = audioContext.createGain();
        src.connect(gain);
        gain.connect(audioContext.destination);
        
        analyser = audioContext.createAnalyser();
        src.connect(analyser);
        drawVisualizer();

        src.start(0);
        currentAudioSource = src;
        src.onended = () => {
          isPlayingAudio = false;
          cancelAnimationFrame(animationFrameId);
          // Clear canvas
          if(canvasCtx) canvasCtx.clearRect(0,0,visualizerCanvas.width, visualizerCanvas.height);
          processAudioQueue();
        };
    } catch(e) {
        console.error("Audio Playback Error", e);
        isPlayingAudio = false;
        processAudioQueue();
    }
  }

  function drawVisualizer() {
    if (!analyser || !canvasCtx) return;
    const bufLen = analyser.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyser.getByteFrequencyData(data);
    canvasCtx.clearRect(0,0,visualizerCanvas.width, visualizerCanvas.height);
    const barW = (visualizerCanvas.width / bufLen) * 2.5;
    let x = 0;
    for(let i=0; i<bufLen; i++) {
      const h = (data[i] / 255) * visualizerCanvas.height;
      canvasCtx.fillStyle = `rgb(${h+100}, 100, 200)`;
      canvasCtx.fillRect(x, visualizerCanvas.height - h, barW, h);
      x += barW + 1;
    }
    animationFrameId = requestAnimationFrame(drawVisualizer);
  }

  // --- Session & Init ---
  async function initSession() {
    try {
       const sid = localStorage.getItem('orion_sid');
       const r = await fetch(`${API_BASE_URL}/api/session`, {headers: sid ? {'X-Orion-Session-Id': sid} : {}});
       const d = await r.json();
       if(d.session_id) {
         orionSessionId = d.session_id;
         localStorage.setItem('orion_sid', orionSessionId);
       }
    } catch(e) { console.warn("Session init fail", e); }
  }

  function setAllCanvasSizes() {
    if (visualizerCanvas && visualizerContainer) {
      visualizerCanvas.width = visualizerContainer.clientWidth;
      visualizerCanvas.height = visualizerContainer.clientHeight;
    }
    // Note: State visualizer is now an iframe, so no canvas resize logic needed for it.
  }

  // --- Boot Sequence ---

  // ───────────────────────────────────────────────────────────────
  // Collapse Mirror (Legacy submit path)
  // ───────────────────────────────────────────────────────────────
  (function initCollapseMirror() {
    const guidedBtn = document.getElementById("collapseModeGuided");
    const rawBtn = document.getElementById("collapseModeRaw");
    const guidedSection = document.getElementById("collapseGuidedSection");
    const rawSection = document.getElementById("collapseRawSection");

    const tooltipToggle = document.getElementById("collapseTooltipToggle");
    const tooltip = document.getElementById("collapseTooltip");

    const statusEl = document.getElementById("collapseStatus");

    const guidedSubmit =
      document.getElementById("collapseGuidedSubmit") ||
      document.getElementById("collapseSubmit"); // legacy id fallback

    const rawSubmit =
      document.getElementById("collapseRawSubmit") ||
      document.getElementById("collapseSubmitRaw"); // legacy id fallback

    const rawJsonEl = document.getElementById("collapseRawJson");

    // Guided inputs
    const el = (id) => document.getElementById(id);
    const observerEl = el("collapseObserver");
    const typeEl = el("collapseType");
    const triggerEl = el("collapseTrigger");
    const observerStateEl = el("collapseObserverState");
    const fieldResonanceEl = el("collapseFieldResonance");
    const emergentEntityEl = el("collapseEmergentEntity");
    const summaryEl = el("collapseSummary");
    const mantraEl = el("collapseMantra");
    const causalEchoEl = el("collapseCausalEcho");

    function setCollapseStatus(msg, isError=false) {
      if (statusEl) {
        statusEl.textContent = msg;
        statusEl.classList.toggle("text-red-400", !!isError);
        statusEl.classList.toggle("text-gray-400", !isError);
        return;
      }
      // If template is missing status element, do something visible.
      console[isError ? "error" : "log"]("[collapse]", msg);
      if (isError) alert(msg);
    }

    function showGuided() {
      if (guidedSection) guidedSection.classList.remove("hidden");
      if (rawSection) rawSection.classList.add("hidden");
      if (guidedBtn) {
        guidedBtn.classList.add("bg-gray-700", "text-gray-100");
        guidedBtn.classList.remove("text-gray-300");
      }
      if (rawBtn) {
        rawBtn.classList.remove("bg-gray-700", "text-gray-100");
        rawBtn.classList.add("text-gray-300");
      }
    }

    function showRaw() {
      if (rawSection) rawSection.classList.remove("hidden");
      if (guidedSection) guidedSection.classList.add("hidden");
      if (rawBtn) {
        rawBtn.classList.add("bg-gray-700", "text-gray-100");
        rawBtn.classList.remove("text-gray-300");
      }
      if (guidedBtn) {
        guidedBtn.classList.remove("bg-gray-700", "text-gray-100");
        guidedBtn.classList.add("text-gray-300");
      }
    }

    if (guidedBtn) guidedBtn.addEventListener("click", showGuided);
    if (rawBtn) rawBtn.addEventListener("click", showRaw);

    if (tooltipToggle && tooltip) {
      tooltipToggle.addEventListener("click", () => tooltip.classList.toggle("hidden"));
    }

    async function postCollapse(payload) {
      setCollapseStatus("Submitting…");
      try {
        const resp = await fetch("/submit-collapse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const txt = await resp.text();
        let data = null;
        try { data = txt ? JSON.parse(txt) : null; } catch { data = { raw: txt }; }

        if (!resp.ok) {
          const detail =
            (data && (data.error || data.detail || data.message)) ||
            `HTTP ${resp.status}`;
          setCollapseStatus(`❌ ${detail}`, true);
          return;
        }

        if (data && data.success === false) {
          setCollapseStatus(`❌ ${data.error || "Unknown error"}`, true);
          return;
        }

        setCollapseStatus("✅ Collapse submitted.");
      } catch (e) {
        setCollapseStatus(`❌ Network error: ${e}`, true);
      }
    }

    function buildGuidedPayload() {
      // If the guided inputs don't exist, fail loudly.
      const missing = [];
      const must = [
        ["observer", observerEl],
        ["type", typeEl],
        ["trigger", triggerEl],
        ["observer_state", observerStateEl],
        ["field_resonance", fieldResonanceEl],
        ["emergent_entity", emergentEntityEl],
        ["summary", summaryEl],
        ["mantra", mantraEl],
      ];
      for (const [name, node] of must) if (!node) missing.push(name);
      if (missing.length) {
        setCollapseStatus(`❌ UI missing required inputs: ${missing.join(", ")}`, true);
        return null;
      }

      const observer_state = (observerStateEl.value || "")
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter(Boolean);

      const payload = {
        observer: observerEl.value.trim(),
        trigger: triggerEl.value.trim(),
        observer_state,
        field_resonance: fieldResonanceEl.value.trim(),
        type: typeEl.value.trim(),
        emergent_entity: emergentEntityEl.value.trim(),
        summary: summaryEl.value.trim(),
        mantra: mantraEl.value.trim(),
        causal_echo: (causalEchoEl && causalEchoEl.value.trim()) || null,
      };

      // Basic validation (FastAPI will also validate)
      const req = [
        ["observer", payload.observer],
        ["trigger", payload.trigger],
        ["observer_state", payload.observer_state.length ? "ok" : ""],
        ["field_resonance", payload.field_resonance],
        ["type", payload.type],
        ["emergent_entity", payload.emergent_entity],
        ["summary", payload.summary],
        ["mantra", payload.mantra],
      ];
      const missing2 = req.filter(([_, v]) => !v).map(([k]) => k);
      if (missing2.length) {
        setCollapseStatus(`❌ Missing: ${missing2.join(", ")}`, true);
        return null;
      }

      return payload;
    }

    if (guidedSubmit) {
      guidedSubmit.addEventListener("click", async (e) => {
        e.preventDefault();
        const payload = buildGuidedPayload();
        if (!payload) return;
        await postCollapse(payload);
      });
    }

    if (rawSubmit) {
      rawSubmit.addEventListener("click", async (e) => {
        e.preventDefault();
        const raw = (rawJsonEl && rawJsonEl.value) ? rawJsonEl.value.trim() : "";
        if (!raw) {
          setCollapseStatus("❌ Paste JSON first.", true);
          return;
        }
        try {
          const payload = JSON.parse(raw);
          await postCollapse(payload);
        } catch (err) {
          setCollapseStatus(`❌ Invalid JSON: ${err}`, true);
        }
      });
    }

    // default to guided
    showGuided();
  })();

  (async () => {
      // 1. Session
      await initSession();
      // 2. Library (Safe)
      await loadCognitionLibrary();
      // 3. WS
      setupWebSocket();
      // 4. UI
      setAllCanvasSizes();
      window.addEventListener('resize', setAllCanvasSizes);
      
      // Vision UI Init
      function updateVisionUi() {
        if (!visionDockedContainer) return;
        const value = visionSourceSelect ? visionSourceSelect.value : "";
        
        // Simple visibility toggles based on source selection
        const hasSource = value !== 'none';
        if (!hasSource) {
            visionFloatingContainer.classList.add("hidden");
            // Show placeholder
            visionDockedContainer.innerHTML = '<div id="visionPlaceholder" class="text-gray-400 text-sm text-center px-4">Vision service stub.</div>';
            return;
        }

        let endpoint = value === "gopro-1" ? `${VISION_EDGE_BASE}/stream.mjpg` : "/static/img/vision-simulated.gif";
        // prevent caching
        if(value === "gopro-1") endpoint += "?t=" + Date.now();

        const imgHtml = `<img src="${endpoint}" class="w-full h-full object-cover">`;
        
        if (visionIsFloating) {
            visionFloatingContainer.classList.remove("hidden");
            visionFloatingContainer.querySelector('.flex-1').innerHTML = imgHtml;
            visionDockedContainer.innerHTML = `<div class="text-gray-500 text-xs">Viewing in Pop-out</div>`;
            if (visionPopoutButton) visionPopoutButton.textContent = "Dock";
        } else {
            visionFloatingContainer.classList.add("hidden");
            visionDockedContainer.innerHTML = imgHtml;
            if (visionPopoutButton) visionPopoutButton.textContent = "Pop Out";
        }
      }

      if (visionPopoutButton) visionPopoutButton.addEventListener("click", () => {
        visionIsFloating = !visionIsFloating;
        updateVisionUi();
      });
      if (visionCloseFloatingButton) visionCloseFloatingButton.addEventListener("click", () => {
        visionIsFloating = false;
        updateVisionUi();
      });
      if (visionSourceSelect) visionSourceSelect.addEventListener("change", () => {
        visionIsFloating = false;
        updateVisionUi();
      });
      
      // Initial call
      updateVisionUi();
  })();

  if (topicRefreshButton) {
    topicRefreshButton.addEventListener("click", () => {
      fetchTopicRailData();
    });
  }
  if (topicRailRetry) {
    topicRailRetry.addEventListener("click", () => {
      fetchTopicRailData();
    });
  }
  if (topicAutoRefresh) {
    topicAutoRefresh.addEventListener("change", () => {
      startTopicAutoRefresh();
    });
  }
  if (topicWindowSelect) {
    topicWindowSelect.addEventListener("change", () => {
      fetchTopicRailData();
    });
  }
  if (topicMaxSelect) {
    topicMaxSelect.addEventListener("change", () => {
      fetchTopicRailData();
    });
  }
  if (topicMinTurnsSelect) {
    topicMinTurnsSelect.addEventListener("change", () => {
      fetchTopicRailData();
    });
  }
  if (topicModelVersion) {
    topicModelVersion.addEventListener("change", () => {
      fetchTopicRailData();
    });
    topicModelVersion.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        fetchTopicRailData();
      }
    });
  }

  if (topicWindowSelect && topicMaxSelect && topicMinTurnsSelect) {
    fetchTopicRailData();
    startTopicAutoRefresh();
  }

  if (bioNodeSelect) {
    bioNodeSelect.addEventListener("change", () => {
      selectedBiometricsNode = bioNodeSelect.value || "cluster";
      if (lastBiometricsPayload) {
        updateBiometricsPanel(lastBiometricsPayload);
      }
    });
  }

});
