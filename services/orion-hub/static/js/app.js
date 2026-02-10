// services/orion-hub/static/js/app.js

// ───────────────────────────────────────────────────────────────
// Global State
// ───────────────────────────────────────────────────────────────
window.__HUB_LAST_STEP = "boot";
const HUB_DEBUG = new URLSearchParams(window.location.search).get("debug") === "1";
const HUB_CFG = window.__HUB_CFG__ || {};

function setHubStep(step) {
  window.__HUB_LAST_STEP = step;
}

function ensureHubPanelHost() {
  let host = document.getElementById("hub-panel-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "hub-panel-host";
    host.setAttribute(
      "style",
      "position:fixed; inset:0; padding:80px 24px 24px 24px; overflow:auto; z-index:10; pointer-events:none;"
    );
    document.body.appendChild(host);
  }
  return host;
}

function truncateCrashText(value, maxLength = 4000) {
  if (!value) return "";
  const text = String(value);
  return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text;
}

function renderCrashOverlay(errorLike) {
  const host = ensureHubPanelHost();
  const message = errorLike?.message || errorLike?.toString?.() || "Unknown error";
  const stack = errorLike?.stack ? String(errorLike.stack) : "";
  host.style.pointerEvents = "auto";
  host.innerHTML = `
    <div style="color:#0f0; background:#111; border:2px solid #0f0; padding:16px; font-family:monospace; max-width:960px; margin:0 auto;">
      <div style="font-size:18px; font-weight:bold; margin-bottom:8px;">HUB UI CRASH</div>
      <div style="margin-bottom:8px;">${truncateCrashText(message)}</div>
      <pre style="white-space:pre-wrap; margin:0 0 8px 0;">${truncateCrashText(stack)}</pre>
      <div>href: ${truncateCrashText(window.location.href, 400)}</div>
      <div>hash: ${truncateCrashText(window.location.hash, 200)}</div>
      <div>lastStep: ${truncateCrashText(window.__HUB_LAST_STEP, 200)}</div>
    </div>
  `;
}

window.addEventListener("error", (event) => {
  renderCrashOverlay(event.error || event.message || event);
});

window.addEventListener("unhandledrejection", (event) => {
  renderCrashOverlay(event.reason || event);
});

function hubOrigin() {
  return window.location.origin;
}

function apiUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (HUB_API_BASE_OVERRIDE) {
    return `${HUB_API_BASE_OVERRIDE}${normalized}`;
  }
  return normalized;
}

function wsProto() {
  return window.location.protocol === "https:" ? "wss:" : "ws:";
}

function wsBase() {
  if (HUB_WS_BASE_OVERRIDE) {
    return HUB_WS_BASE_OVERRIDE;
  }
  return `${wsProto()}//${window.location.host}`;
}

function wsUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${wsBase()}${normalized}`;
}

function normalizeApiBaseOverride(value) {
  if (!value) return "";
  const trimmed = String(value).trim();
  if (!trimmed) return "";
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed.replace(/\/+$/, "");
  }
  if (trimmed.startsWith("/")) {
    return trimmed.replace(/\/+$/, "");
  }
  return `/${trimmed.replace(/\/+$/, "")}`;
}

function normalizeWsBaseOverride(value) {
  if (!value) return "";
  const trimmed = String(value).trim();
  if (!trimmed) return "";
  if (trimmed.startsWith("ws://") || trimmed.startsWith("wss://")) {
    return trimmed.replace(/\/+$/, "");
  }
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed.replace(/^http/, "ws").replace(/\/+$/, "");
  }
  if (trimmed.startsWith("/")) {
    return `${wsProto()}//${window.location.host}${trimmed.replace(/\/+$/, "")}`;
  }
  return "";
}

const HUB_API_BASE_OVERRIDE = normalizeApiBaseOverride(HUB_CFG.apiBaseOverride);
const HUB_WS_BASE_OVERRIDE = normalizeWsBaseOverride(HUB_CFG.wsBaseOverride);
const DEV_HOSTS = new Set(["localhost", "127.0.0.1"]);
const IS_DEV = DEV_HOSTS.has(window.location.hostname);

function warnIfCrossOrigin(url) {
  if (!IS_DEV) return;
  if ((url.startsWith("http://") || url.startsWith("https://")) && !url.startsWith(hubOrigin())) {
    console.warn(`[Hub] Cross-origin fetch detected: ${url}`);
  }
}

function hubFetch(url, options) {
  warnIfCrossOrigin(url);
  return fetch(url, options);
}
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
let orionSessionId = localStorage.getItem('orion_sid') || null;
let cognitionLibrary = { packs: {}, verbs: [], map: {} };
let selectedBiometricsNode = "cluster";
let lastBiometricsPayload = null;
let notifications = [];
let pendingAttention = [];
let chatMessages = [];
const seenMessageIds = new Set();
const openedMessageIds = new Set();
const dismissedMessageIds = new Set();
const NOTIFICATION_MAX = 200;
let notificationToastSeconds = 8;
const ATTENTION_EVENT_KIND = "orion.chat.attention";
const CHAT_MESSAGE_EVENT_KIND = "orion.chat.message";
const RECIPIENT_GROUP = "juniper_primary";
let topicAutoRefreshTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  console.log("[Main] DOM Content Loaded - Initializing UI...");

// --- 0. Local persistence for message dismissals ---
// Backend may not reflect receipts into /api/chat/messages yet; this prevents dismissed items from reappearing.
const DISMISSED_STORAGE_KEY = "orion_chat_dismissed_ids_v1";
function loadDismissedIds() {
  try {
    const raw = localStorage.getItem(DISMISSED_STORAGE_KEY);
    if (!raw) return;
    const arr = JSON.parse(raw);
    if (Array.isArray(arr)) arr.forEach((id) => dismissedMessageIds.add(String(id)));
  } catch (e) {
    console.warn("[Messages] Failed to load dismissed ids", e);
  }
}
function persistDismissedIds() {
  try {
    localStorage.setItem(DISMISSED_STORAGE_KEY, JSON.stringify(Array.from(dismissedMessageIds)));
  } catch (e) {
    console.warn("[Messages] Failed to persist dismissed ids", e);
  }
}
function markDismissed(messageId) {
  if (!messageId) return;
  dismissedMessageIds.add(String(messageId));
  persistDismissedIds();
}
function isDismissed(messageId) {
  if (!messageId) return false;
  return dismissedMessageIds.has(String(messageId));
}

loadDismissedIds();

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
  const noWriteToggle = document.getElementById('noWriteToggle');
  const recallModeSelect = document.getElementById('recallModeSelect');
  const recallProfileSelect = document.getElementById('recallProfileSelect');
  const memoryPanelToggle = document.getElementById('memoryPanelToggle');
  const memoryPanelBody = document.getElementById('memoryPanelBody');
  const memoryUsedValue = document.getElementById('memoryUsedValue');
  const recallCountValue = document.getElementById('recallCountValue');
  const backendCountsValue = document.getElementById('backendCountsValue');
  const memoryDigestPre = document.getElementById('memoryDigestPre');
  const notificationList = document.getElementById('notificationList');
  const notificationFilter = document.getElementById('notificationFilter');
  const attentionList = document.getElementById('attentionList');
  const attentionCount = document.getElementById('attentionCount');
  const messageList = document.getElementById('messageList');
  const messageFilter = document.getElementById('messageFilter');
  const toastContainer = document.getElementById('toastContainer');

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
  const notifyDisplayName = document.getElementById('notifyDisplayName');
  const notifyTimezone = document.getElementById('notifyTimezone');
  const notifyQuietEnabled = document.getElementById('notifyQuietEnabled');
  const notifyQuietStart = document.getElementById('notifyQuietStart');
  const notifyQuietEnd = document.getElementById('notifyQuietEnd');
  const notifySettingsSave = document.getElementById('notifySettingsSave');
  const notifySettingsStatus = document.getElementById('notifySettingsStatus');

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
  const toastMessage = document.getElementById("toastMessage");

  // Topic Studio
  const hubTabButton = document.getElementById("hubTabButton");
  const topicsTabButton = document.getElementById("topicsTabButton");
  const topicStudioTabButton = document.getElementById("topicStudioTabButton");
  const appPanels = document.getElementById("appPanels");
  const hubPanel = document.getElementById("hub");
  const topicStudioPanel = document.getElementById("topic-studio");
  const topicStudioRoot = document.getElementById("topicStudioRoot");
  const topicFoundryBaseLabel = document.getElementById("topicFoundryBaseLabel");
  const tsDatasetSelect = document.getElementById("tsDatasetSelect");
  const tsDatasetName = document.getElementById("tsDatasetName");
  const tsDatasetSchema = document.getElementById("tsDatasetSchema");
  const tsDatasetTableSelect = document.getElementById("tsDatasetTableSelect");
  const tsDatasetTable = document.getElementById("tsDatasetTable");
  const tsDatasetIdColumn = document.getElementById("tsDatasetIdColumn");
  const tsDatasetTimeColumn = document.getElementById("tsDatasetTimeColumn");
  const tsDatasetTextColumns = document.getElementById("tsDatasetTextColumns");
  const tsDatasetBoundaryColumn = document.getElementById("tsDatasetBoundaryColumn");
  const tsDatasetWhereSql = document.getElementById("tsDatasetWhereSql");
  const tsDatasetTimezone = document.getElementById("tsDatasetTimezone");
  const tsSaveDataset = document.getElementById("tsSaveDataset");
  const tsDatasetSaveStatus = document.getElementById("tsDatasetSaveStatus");
  const tsStartAt = document.getElementById("tsStartAt");
  const tsEndAt = document.getElementById("tsEndAt");
  const tsWindowingMode = document.getElementById("tsWindowingMode");
  const tsGroupByRow = document.getElementById("tsGroupByRow");
  const tsGroupByColumn = document.getElementById("tsGroupByColumn");
  const tsSegmentationMode = document.getElementById("tsSegmentationMode");
  const tsTimeGap = document.getElementById("tsTimeGap");
  const tsMaxWindow = document.getElementById("tsMaxWindow");
  const tsFixedKRows = document.getElementById("tsFixedKRows");
  const tsFixedKRowsStep = document.getElementById("tsFixedKRowsStep");
  const tsMinBlocks = document.getElementById("tsMinBlocks");
  const tsMaxChars = document.getElementById("tsMaxChars");
  const tsRunPreset = document.getElementById("tsRunPreset");
  const tsCreateDataset = document.getElementById("tsCreateDataset");
  const tsPreviewDataset = document.getElementById("tsPreviewDataset");
  const tsPreviewDocs = document.getElementById("tsPreviewDocs");
  const tsPreviewSegments = document.getElementById("tsPreviewSegments");
  const tsPreviewAvgChars = document.getElementById("tsPreviewAvgChars");
  const tsPreviewP95Chars = document.getElementById("tsPreviewP95Chars");
  const tsPreviewMaxChars = document.getElementById("tsPreviewMaxChars");
  const tsPreviewObserved = document.getElementById("tsPreviewObserved");
  const tsPreviewSamples = document.getElementById("tsPreviewSamples");
  const tsPreviewError = document.getElementById("tsPreviewError");
  const tsModelName = document.getElementById("tsModelName");
  const tsModelVersion = document.getElementById("tsModelVersion");
  const tsModelStage = document.getElementById("tsModelStage");
  const tsModelEmbeddingUrl = document.getElementById("tsModelEmbeddingUrl");
  const tsModelMinCluster = document.getElementById("tsModelMinCluster");
  const tsModelMetric = document.getElementById("tsModelMetric");
  const tsModelParams = document.getElementById("tsModelParams");
  const tsCreateModel = document.getElementById("tsCreateModel");
  const tsPromoteModelSelect = document.getElementById("tsPromoteModelSelect");
  const tsPromoteStage = document.getElementById("tsPromoteStage");
  const tsPromoteReason = document.getElementById("tsPromoteReason");
  const tsPromoteModel = document.getElementById("tsPromoteModel");
  const tsEnrichEnricher = document.getElementById("tsEnrichEnricher");
  const tsEnrichForce = document.getElementById("tsEnrichForce");
  const tsEnrichRun = document.getElementById("tsEnrichRun");
  const tsEnrichStatus = document.getElementById("tsEnrichStatus");
  const tsTrainModelSelect = document.getElementById("tsTrainModelSelect");
  const tsTrainRun = document.getElementById("tsTrainRun");
  const tsRunId = document.getElementById("tsRunId");
  const tsPollRun = document.getElementById("tsPollRun");
  const tsRunState = document.getElementById("tsRunState");
  const tsRunClusters = document.getElementById("tsRunClusters");
  const tsRunDocs = document.getElementById("tsRunDocs");
  const tsRunSegments = document.getElementById("tsRunSegments");
  const tsRunOutliers = document.getElementById("tsRunOutliers");
  const tsRunEnriched = document.getElementById("tsRunEnriched");
  const tsRunArtifacts = document.getElementById("tsRunArtifacts");
  const tsRunError = document.getElementById("tsRunError");
  const tsRunsSelect = document.getElementById("tsRunsSelect");
  const tsRunsWarning = document.getElementById("tsRunsWarning");
  const tsSegmentsRunId = document.getElementById("tsSegmentsRunId");
  const tsSegmentsEnrichment = document.getElementById("tsSegmentsEnrichment");
  const tsSegmentsAspect = document.getElementById("tsSegmentsAspect");
  const tsLoadSegments = document.getElementById("tsLoadSegments");
  const tsSegmentsError = document.getElementById("tsSegmentsError");
  const tsSegmentsTableBody = document.getElementById("tsSegmentsTableBody");
  const tsSegmentDetailsPanel = document.getElementById("tsSegmentDetailsPanel");
  const tsSegmentDetailHeader = document.getElementById("tsSegmentDetailHeader");
  const tsSegmentDetailMeta = document.getElementById("tsSegmentDetailMeta");
  const tsSegmentDetailSnippet = document.getElementById("tsSegmentDetailSnippet");
  const tsSegmentDetailMeaning = document.getElementById("tsSegmentDetailMeaning");
  const tsSegmentFullText = document.getElementById("tsSegmentFullText");
  const tsSegmentFullTextStatus = document.getElementById("tsSegmentFullTextStatus");
  const tsSegmentFullTextLoad = document.getElementById("tsSegmentFullTextLoad");
  const tsSegmentFullTextExpand = document.getElementById("tsSegmentFullTextExpand");
  const tsSegmentFullTextCopy = document.getElementById("tsSegmentFullTextCopy");
  const topicStudioError = document.getElementById("topicStudioError");
  const tsDebugDrawer = document.getElementById("tsDebugDrawer");
  const tsDebugPreview = document.getElementById("tsDebugPreview");
  const tsDebugTrain = document.getElementById("tsDebugTrain");
  const tsDebugEnrich = document.getElementById("tsDebugEnrich");
  const tsDebugSegments = document.getElementById("tsDebugSegments");
  const tsSegmentsLoading = document.getElementById("tsSegmentsLoading");
  const tsSegmentsRefresh = document.getElementById("tsSegmentsRefresh");
  const tsSegmentsPageSize = document.getElementById("tsSegmentsPageSize");
  const tsSegmentsPrev = document.getElementById("tsSegmentsPrev");
  const tsSegmentsNext = document.getElementById("tsSegmentsNext");
  const tsSegmentsRange = document.getElementById("tsSegmentsRange");
  const tsSegmentsExport = document.getElementById("tsSegmentsExport");
  const tsSubviewRunsBtn = document.getElementById("tsSubviewRunsBtn");
  const tsSubviewConversationsBtn = document.getElementById("tsSubviewConversationsBtn");
  const tsSubviewRuns = document.getElementById("tsSubviewRuns");
  const tsSubviewConversations = document.getElementById("tsSubviewConversations");
  const tsSubviewTopicsBtn = document.getElementById("tsSubviewTopicsBtn");
  const tsSubviewCompareBtn = document.getElementById("tsSubviewCompareBtn");
  const tsSubviewDriftBtn = document.getElementById("tsSubviewDriftBtn");
  const tsSubviewEventsBtn = document.getElementById("tsSubviewEventsBtn");
  const tsSubviewKgBtn = document.getElementById("tsSubviewKgBtn");
  const tsSubviewTopics = document.getElementById("tsSubviewTopics");
  const tsSubviewCompare = document.getElementById("tsSubviewCompare");
  const tsSubviewDrift = document.getElementById("tsSubviewDrift");
  const tsSubviewEvents = document.getElementById("tsSubviewEvents");
  const tsSubviewKg = document.getElementById("tsSubviewKg");
  const tsTopicsRunId = document.getElementById("tsTopicsRunId");
  const tsTopicsScope = document.getElementById("tsTopicsScope");
  const tsTopicsLimit = document.getElementById("tsTopicsLimit");
  const tsTopicsOffset = document.getElementById("tsTopicsOffset");
  const tsTopicsLoad = document.getElementById("tsTopicsLoad");
  const tsTopicsStatus = document.getElementById("tsTopicsStatus");
  const tsTopicsError = document.getElementById("tsTopicsError");
  const tsTopicsTableBody = document.getElementById("tsTopicsTableBody");
  const tsTopicSelectedId = document.getElementById("tsTopicSelectedId");
  const tsTopicKeywords = document.getElementById("tsTopicKeywords");
  const tsTopicSegmentsLimit = document.getElementById("tsTopicSegmentsLimit");
  const tsTopicSegmentsOffset = document.getElementById("tsTopicSegmentsOffset");
  const tsTopicSegmentsPrev = document.getElementById("tsTopicSegmentsPrev");
  const tsTopicSegmentsNext = document.getElementById("tsTopicSegmentsNext");
  const tsTopicSegmentsStatus = document.getElementById("tsTopicSegmentsStatus");
  const tsTopicSegmentsTableBody = document.getElementById("tsTopicSegmentsTableBody");
  const tsCompareLeftRunId = document.getElementById("tsCompareLeftRunId");
  const tsCompareRightRunId = document.getElementById("tsCompareRightRunId");
  const tsCompareRun = document.getElementById("tsCompareRun");
  const tsCompareStatus = document.getElementById("tsCompareStatus");
  const tsCompareError = document.getElementById("tsCompareError");
  const tsCompareTableBody = document.getElementById("tsCompareTableBody");
  const tsCompareAspectBody = document.getElementById("tsCompareAspectBody");
  const tsCompareDocs = document.getElementById("tsCompareDocs");
  const tsCompareSegments = document.getElementById("tsCompareSegments");
  const tsCompareClusters = document.getElementById("tsCompareClusters");
  const tsCompareOutliers = document.getElementById("tsCompareOutliers");
  const tsDriftModelName = document.getElementById("tsDriftModelName");
  const tsDriftWindowHours = document.getElementById("tsDriftWindowHours");
  const tsDriftLimit = document.getElementById("tsDriftLimit");
  const tsDriftLoad = document.getElementById("tsDriftLoad");
  const tsDriftRunNow = document.getElementById("tsDriftRunNow");
  const tsDriftStatus = document.getElementById("tsDriftStatus");
  const tsDriftError = document.getElementById("tsDriftError");
  const tsDriftTableBody = document.getElementById("tsDriftTableBody");
  const tsEventsKind = document.getElementById("tsEventsKind");
  const tsEventsLimit = document.getElementById("tsEventsLimit");
  const tsEventsOffset = document.getElementById("tsEventsOffset");
  const tsEventsLoad = document.getElementById("tsEventsLoad");
  const tsEventsExport = document.getElementById("tsEventsExport");
  const tsEventsStatus = document.getElementById("tsEventsStatus");
  const tsEventsError = document.getElementById("tsEventsError");
  const tsEventsTableBody = document.getElementById("tsEventsTableBody");
  const tsKgRunId = document.getElementById("tsKgRunId");
  const tsKgLimit = document.getElementById("tsKgLimit");
  const tsKgOffset = document.getElementById("tsKgOffset");
  const tsKgQuery = document.getElementById("tsKgQuery");
  const tsKgPredicate = document.getElementById("tsKgPredicate");
  const tsKgLoad = document.getElementById("tsKgLoad");
  const tsKgExport = document.getElementById("tsKgExport");
  const tsKgStatus = document.getElementById("tsKgStatus");
  const tsKgError = document.getElementById("tsKgError");
  const tsKgTableBody = document.getElementById("tsKgTableBody");
  const tsConvoDatasetSelect = document.getElementById("tsConvoDatasetSelect");
  const tsConvoStartAt = document.getElementById("tsConvoStartAt");
  const tsConvoEndAt = document.getElementById("tsConvoEndAt");
  const tsConvoLimit = document.getElementById("tsConvoLimit");
  const tsConvoLoad = document.getElementById("tsConvoLoad");
  const tsConvoLoading = document.getElementById("tsConvoLoading");
  const tsConvoList = document.getElementById("tsConvoList");
  const tsConvoDetail = document.getElementById("tsConvoDetail");
  const tsConvoMerge = document.getElementById("tsConvoMerge");
  const tsConvoMergeReason = document.getElementById("tsConvoMergeReason");
  const tsConvoMergeStatus = document.getElementById("tsConvoMergeStatus");
  const tsConvoOverrides = document.getElementById("tsConvoOverrides");
  const tsConvoError = document.getElementById("tsConvoError");
  const tsConvoRebuildPreview = document.getElementById("tsConvoRebuildPreview");
  const tsStatusBadge = document.getElementById("tsStatusBadge");
  const tsStatusPg = document.getElementById("tsStatusPg");
  const tsStatusEmbedding = document.getElementById("tsStatusEmbedding");
  const tsStatusModelDir = document.getElementById("tsStatusModelDir");
  const tsStatusDetail = document.getElementById("tsStatusDetail");
  const tsStatusLoading = document.getElementById("tsStatusLoading");
  const tsReadyWarning = document.getElementById("tsReadyWarning");
  const tsCapabilitiesWarning = document.getElementById("tsCapabilitiesWarning");
  const tsCopyReadyUrl = document.getElementById("tsCopyReadyUrl");
  const tsCopyCapabilitiesUrl = document.getElementById("tsCopyCapabilitiesUrl");
  const tsSkeletonMain = document.getElementById("tsSkeletonMain");
  const tsSkeletonStatus = document.getElementById("tsSkeletonStatus");
  const tsSkeletonRetry = document.getElementById("tsSkeletonRetry");
  const tsLlmNote = document.getElementById("tsLlmNote");
  const tsPreviewLoading = document.getElementById("tsPreviewLoading");
  const tsRunLoading = document.getElementById("tsRunLoading");
  const tsEnrichLoading = document.getElementById("tsEnrichLoading");
  const tsPreviewWarning = document.getElementById("tsPreviewWarning");
  const tsUsePreviewSpec = document.getElementById("tsUsePreviewSpec");
  const tsRunWarning = document.getElementById("tsRunWarning");
  const tsEnrichWarning = document.getElementById("tsEnrichWarning");
  const tsCopyRunId = document.getElementById("tsCopyRunId");
  const tsCopyRunUrl = document.getElementById("tsCopyRunUrl");
  const tsCopyArtifacts = document.getElementById("tsCopyArtifacts");
  const tsCopyDatasetId = document.getElementById("tsCopyDatasetId");
  const tsCopyModelId = document.getElementById("tsCopyModelId");
  const tsSegmentsSearch = document.getElementById("tsSegmentsSearch");
  const tsSegmentsSort = document.getElementById("tsSegmentsSort");
  const tsSegmentsFacets = document.getElementById("tsSegmentsFacets");
  const tsCopySegmentId = document.getElementById("tsCopySegmentId");

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

  function showToastText(message) {
    if (!toastContainer || !toastMessage) return;
    toastMessage.textContent = message;
    toastContainer.classList.remove("hidden");
    setTimeout(() => {
      toastContainer.classList.add("hidden");
    }, 4000);
  }

  function showToast(x) {
    if (typeof x === "string") {
      showToastText(x);
    } else {
      showToastNotification(x);
    }
  }

  function truncate(s, n = 240) {
    const text = s == null ? "" : String(s);
    if (text.length <= n) return text;
    return `${text.slice(0, Math.max(0, n - 1))}…`;
  }

  const TOPIC_FOUNDRY_PROXY_BASE = apiUrl("/api/topic-foundry");
  const TOPIC_STUDIO_STATE_KEY = "topic_studio_state_v1";
  const MIN_PREVIEW_DOCS = 20;
  const TOPIC_STUDIO_SPLIT_SENTINEL = "TOPIC STUDIO SPLIT PANE v2 ACTIVE";

  const panelRenderSteps = {
    hub: [],
    "topic-studio": [],
  };

  function resetPanelSteps(tabKey) {
    panelRenderSteps[tabKey] = [];
  }

  function recordPanelStep(tabKey, step) {
    if (!panelRenderSteps[tabKey]) {
      panelRenderSteps[tabKey] = [];
    }
    panelRenderSteps[tabKey].push(step);
    updatePanelDebug(tabKey);
  }

  function panelDebugMarkup(tabKey) {
    if (!HUB_DEBUG) return "";
    const steps = panelRenderSteps[tabKey] || [];
    const lastStep = steps.length ? steps[steps.length - 1] : "--";
    const items = steps.length ? steps.map((step) => `<li>${step}</li>`).join("") : "<li>--</li>";
    return `
      <div style="margin-top:12px; font-size:11px; color:#9ca3af;">
        <div style="font-weight:600; margin-bottom:4px;">Render steps (last: ${lastStep})</div>
        <ul style="padding-left:18px; margin:0; line-height:1.4;">${items}</ul>
      </div>
    `;
  }

  function updatePanelDebug(tabKey) {
    if (!HUB_DEBUG) return;
    const debugEl = document.getElementById(`hub-panel-debug-${tabKey}`);
    if (debugEl) {
      debugEl.innerHTML = panelDebugMarkup(tabKey);
    }
  }

  function renderPanelError(tabKey, error, retryLabel = "Retry", retryFn) {
    const message = error?.message || error?.toString?.() || "Unknown error";
    const stack = error?.stack ? String(error.stack) : "";
    if (tabKey === "topic-studio" && topicStudioError) {
      topicStudioError.classList.remove("hidden");
      topicStudioError.innerHTML = `
        <div class="flex items-center justify-between gap-2">
          <div><strong>Topic Studio error:</strong> ${truncateCrashText(message)}</div>
          <button id="ts-route-retry" class="bg-gray-900/60 hover:bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700 text-xs">${retryLabel}</button>
        </div>
        <pre class="mt-2 text-[10px] whitespace-pre-wrap">${truncateCrashText(stack)}</pre>
      `;
      const retryButton = document.getElementById("ts-route-retry");
      if (retryButton && typeof retryFn === "function") {
        retryButton.addEventListener("click", retryFn);
      }
      return;
    }
    const host = ensureHubPanelHost();
    host.style.pointerEvents = "auto";
    host.innerHTML = `
      <div style="color:#86efac; background:#000; border:2px solid #ef4444; padding:16px; font-family:monospace; max-width:960px; margin:0 auto; border-radius:12px;">
        <div style="font-size:18px; font-weight:bold; margin-bottom:8px; color:#86efac;">${tabKey.toUpperCase()} PANEL ERROR</div>
        <div style="margin-bottom:8px; color:#fca5a5;">${truncateCrashText(message)}</div>
        <pre style="white-space:pre-wrap; margin:0 0 8px 0; color:#fca5a5;">${truncateCrashText(stack)}</pre>
        <button id="hub-panel-retry" style="background:#111; color:#86efac; border:1px solid #86efac; padding:6px 12px; border-radius:8px; cursor:pointer;">${retryLabel}</button>
        <div id="hub-panel-debug-${tabKey}">${panelDebugMarkup(tabKey)}</div>
      </div>
    `;
    const retryButton = document.getElementById("hub-panel-retry");
    if (retryButton && typeof retryFn === "function") {
      retryButton.addEventListener("click", retryFn);
    }
  }

  function normalizeHash(hash) {
    const cleaned = (hash || "").replace("#", "");
    if (!cleaned) return "hub";
    if (cleaned === "hub" || cleaned === "topic-studio" || cleaned === "topics") return cleaned;
    return "hub";
  }

  function setActiveNav(activeKey) {
    const isHub = activeKey === "hub";
    const isTopicStudio = activeKey === "topic-studio";
    if (hubTabButton) {
      hubTabButton.classList.toggle("bg-indigo-600", isHub);
      hubTabButton.classList.toggle("text-white", isHub);
      hubTabButton.classList.toggle("border-indigo-500", isHub);
      hubTabButton.classList.toggle("bg-gray-800", !isHub);
      hubTabButton.classList.toggle("text-gray-200", !isHub);
      hubTabButton.classList.toggle("border-gray-700", !isHub);
    }
    if (topicStudioTabButton) {
      topicStudioTabButton.classList.toggle("bg-indigo-600", isTopicStudio);
      topicStudioTabButton.classList.toggle("text-white", isTopicStudio);
      topicStudioTabButton.classList.toggle("border-indigo-500", isTopicStudio);
      topicStudioTabButton.classList.toggle("bg-gray-800", !isTopicStudio);
      topicStudioTabButton.classList.toggle("text-gray-200", !isTopicStudio);
      topicStudioTabButton.classList.toggle("border-gray-700", !isTopicStudio);
    }
  }

  function teardownTopicStudioUI() {
    const inlineSentinel = document.getElementById("tsSplitPaneSentinel");
    if (inlineSentinel) inlineSentinel.classList.add("hidden");
    if (topicStudioError) {
      topicStudioError.classList.add("hidden");
      topicStudioError.textContent = "";
    }
    if (topicStudioDebugState.overlay) {
      topicStudioDebugState.overlay.remove();
      topicStudioDebugState.overlay = null;
      topicStudioDebugState.overlayBody = null;
    }
    const host = ensureHubPanelHost();
    host.style.pointerEvents = "none";
    host.innerHTML = "";
  }

  async function initTopicStudioUI() {
    if (!topicStudioRoot) {
      throw new Error("Topic Studio root not found.");
    }
    if (!window.__topicStudioInitDone) {
      window.__topicStudioInitDone = true;
      bindTopicStudioPanel();
    }
    if (topicStudioError) {
      topicStudioError.classList.add("hidden");
      topicStudioError.textContent = "";
    }
    ensureTopicStudioSentinel();
    await refreshTopicStudio();
  }

  function showPanel(panelKey) {
    if (!hubPanel || !topicStudioPanel || !hubTabButton || !topicStudioTabButton) return;
    const panels = Array.from(document.querySelectorAll("#appPanels section[data-panel]"));
    panels.forEach((panel) => panel.classList.add("hidden"));
    const target = document.querySelector(`#appPanels section[data-panel="${panelKey}"]`);
    if (target) {
      target.classList.remove("hidden");
    } else if (panelKey !== "hub") {
      showPanel("hub");
      return;
    }
    setActiveNav(panelKey);
    if (panelKey !== "topic-studio") {
      teardownTopicStudioUI();
    } else {
      initTopicStudioUI().catch((err) => {
        renderPanelError("topic-studio", err, "Retry", () => showPanel("topic-studio"));
        window.location.hash = "#hub";
      });
    }
  }

  function renderTopicStudioSkeleton(message = "Loading...") {
    const host = topicStudioRoot;
    if (!host) return;
    host.style.pointerEvents = "auto";
    host.innerHTML = `
      <div style="color:#ddd; background:#0b0b0b; border:1px solid #333; border-radius:12px; padding:16px; max-width:960px; margin:0 auto; pointer-events:auto;">
        <div style="font-size:20px; font-weight:600; margin-bottom:12px;">Topic Studio</div>
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px;">
          <div><strong>Ready:</strong> <span id="ts-host-ready">loading</span></div>
          <div><strong>Capabilities:</strong> <span id="ts-host-capabilities">loading</span></div>
          <div><strong>Runs:</strong> <span id="ts-host-runs">loading</span></div>
        </div>
        <div id="ts-host-errors" style="display:flex; flex-direction:column; gap:8px; margin-bottom:12px;"></div>
        <button id="ts-host-retry" style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer;">Retry</button>
        <div style="margin-top:12px; font-size:12px; color:#888;">${message}</div>
        <div style="margin-top:8px; font-size:11px; color:#666;">lastStep: <span id="ts-host-step">${window.__HUB_LAST_STEP}</span></div>
      </div>
    `;
    const retryButton = document.getElementById("ts-host-retry");
    if (retryButton) {
      retryButton.addEventListener("click", () => {
        renderTopicStudioSkeleton("Retrying...");
        refreshTopicStudioRoute();
      });
    }
    setTopicStudioRenderStep("mounted skeleton");
    setHubStep("topic-skeleton-mounted");
  }

  function renderTopicStudioPanel() {
    const host = topicStudioRoot;
    if (!host) return;
    host.style.pointerEvents = "auto";
    host.innerHTML = `
      <div style="color:#ddd; background:#0b0b0b; border:1px solid #333; border-radius:12px; padding:16px; max-width:1200px; margin:0 auto; pointer-events:auto;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
          <div>
            <div style="font-size:20px; font-weight:600;">Topic Studio</div>
            <div style="font-size:11px; color:#6ee7b7; font-family:monospace;">${TOPIC_STUDIO_SPLIT_SENTINEL}</div>
            <div style="font-size:12px; color:#888;">Manage datasets, models, runs, and segments.</div>
          </div>
          <div style="display:flex; gap:8px; align-items:center;">
            <span id="ts-mvp-last-refresh" style="font-size:11px; color:#777;">Last refresh: --</span>
            <button id="ts-mvp-refresh" style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer;">Refresh</button>
          </div>
        </div>
        <div style="margin-top:12px; padding:10px; border:1px solid #333; border-radius:10px; background:#111;">
          <div style="display:flex; gap:16px; flex-wrap:wrap; font-size:12px;">
            <div>Ready: <span id="ts-mvp-ready">loading</span></div>
            <div>Capabilities: <span id="ts-mvp-capabilities">loading</span></div>
            <div>Details: <span id="ts-mvp-ready-detail">--</span></div>
          </div>
        </div>
        <div id="ts-mvp-errors" style="margin-top:10px; display:flex; flex-direction:column; gap:8px;"></div>

        <div style="margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:16px;">
          <div style="border:1px solid #333; border-radius:10px; padding:12px; background:#111;">
            <div style="font-weight:600; margin-bottom:8px;">Datasets</div>
            <div id="ts-mvp-datasets-list" style="font-size:12px; color:#aaa; margin-bottom:8px;">(loading)</div>
            <div id="ts-mvp-introspect-warning" style="display:none; font-size:12px; color:#fbb; margin-bottom:8px;"></div>
            <div style="font-size:12px; margin-bottom:6px;">Create dataset</div>
            <div style="display:grid; gap:6px;">
              <input id="ts-mvp-dataset-name" placeholder="name" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <select id="ts-mvp-schema" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <select id="ts-mvp-table" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <select id="ts-mvp-column-id" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <select id="ts-mvp-column-time" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <div id="ts-mvp-column-texts" style="border:1px solid #333; padding:6px; border-radius:6px; color:#aaa; font-size:12px; max-height:140px; overflow:auto;">(no columns)</div>
              <div id="ts-mvp-manual-fields" style="display:none; gap:6px; flex-direction:column;">
                <input id="ts-mvp-dataset-table" placeholder="source_table" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
                <input id="ts-mvp-dataset-id" placeholder="id_column" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
                <input id="ts-mvp-dataset-time" placeholder="time_column" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
                <input id="ts-mvp-dataset-text" placeholder="text_columns (comma)" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              </div>
              <input id="ts-mvp-dataset-where" placeholder="where_sql (optional)" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-dataset-tz" value="UTC" placeholder="timezone" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <button id="ts-mvp-create-dataset" style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer;">Create dataset</button>
            </div>
            <div style="margin-top:12px; font-weight:600;">Preview</div>
            <div style="display:grid; gap:6px; margin-top:6px;">
              <select id="ts-mvp-preview-dataset" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <input id="ts-mvp-preview-start" placeholder="start_at (ISO)" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-preview-end" placeholder="end_at (ISO)" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-preview-limit" value="200" placeholder="limit" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <select id="ts-mvp-preview-block" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;">
                <option value="turn_pairs">turn_pairs</option>
                <option value="fixed_k_rows">fixed_k_rows</option>
                <option value="time_gap">time_gap</option>
                <option value="conversation_bound">conversation_bound</option>
                <option value="conversation_bound_then_time_gap">conversation_bound_then_time_gap</option>
                <option value="group_by_column">group_by_column</option>
              </select>
              <input id="ts-mvp-preview-gap" value="900" placeholder="time_gap_seconds" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-preview-maxchars" value="6000" placeholder="max_chars" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <button id="ts-mvp-preview" disabled style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer; opacity:0.6;">Preview</button>
            </div>
            <div style="margin-top:6px; font-size:11px; color:#666;">Tip: set a start/end range for faster previews.</div>
            <div id="ts-mvp-preview-stats" style="margin-top:8px; font-size:12px; color:#aaa;">--</div>
            <div id="ts-mvp-preview-samples" style="margin-top:6px; font-size:12px; color:#aaa;">--</div>
          </div>

          <div style="border:1px solid #333; border-radius:10px; padding:12px; background:#111;">
            <div style="font-weight:600; margin-bottom:8px;">Models</div>
            <div id="ts-mvp-models-list" style="font-size:12px; color:#aaa; margin-bottom:8px;">(loading)</div>
            <div style="font-size:12px; margin-bottom:6px;">Create model</div>
            <div style="display:grid; gap:6px;">
              <input id="ts-mvp-model-name" placeholder="name" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-model-version" placeholder="version" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-model-stage" value="development" placeholder="stage" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <select id="ts-mvp-model-dataset" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <div style="font-size:11px; color:#888;">Min cluster size (HDBSCAN)</div>
              <input id="ts-mvp-model-mincluster" value="20" placeholder="min_cluster_size" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-model-metric" value="cosine" placeholder="metric" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <div style="font-size:11px; color:#888;">Advanced params (optional)</div>
              <textarea id="ts-mvp-model-params" rows="2" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;">{}</textarea>
              <div id="ts-mvp-embed-hint" style="font-size:11px; color:#666;"></div>
              <button id="ts-mvp-create-model" style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer;">Create model</button>
            </div>

            <div style="margin-top:12px; font-weight:600;">Runs</div>
            <div style="display:grid; gap:6px; margin-top:6px;">
              <select id="ts-mvp-run-model" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <select id="ts-mvp-run-dataset" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
              <input id="ts-mvp-run-start" placeholder="start_at (ISO)" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <input id="ts-mvp-run-end" placeholder="end_at (ISO)" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;" />
              <button id="ts-mvp-train-run" style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer;">Train run</button>
              <button id="ts-mvp-enrich-run" style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer;">Enrich run</button>
            </div>
            <div id="ts-mvp-runs-list" style="margin-top:8px; font-size:12px; color:#aaa;">(loading)</div>
          </div>
        </div>

        <div style="margin-top:16px; display:grid; grid-template-columns:minmax(0,2fr) minmax(0,1fr); gap:12px;">
          <div style="border:1px solid #333; border-radius:10px; padding:12px; background:#111;">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
              <div style="font-weight:600;">Segments</div>
              <div style="display:flex; gap:8px; align-items:center;">
                <select id="ts-mvp-segments-run" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;"></select>
                <select id="ts-mvp-segments-enriched" style="background:#0b0b0b; color:#ddd; border:1px solid #333; padding:6px; border-radius:6px;">
                  <option value="">all</option>
                  <option value="true">has_enrichment=true</option>
                  <option value="false">has_enrichment=false</option>
                </select>
                <button id="ts-mvp-load-segments" style="background:#222; color:#ddd; border:1px solid #444; padding:6px 12px; border-radius:8px; cursor:pointer;">Load</button>
              </div>
            </div>
            <div id="ts-mvp-segments-table" style="margin-top:8px; font-size:12px; color:#aaa;">--</div>
          </div>
          <div style="border:1px solid #333; border-radius:10px; padding:12px; background:#111; display:flex; flex-direction:column; gap:8px; min-height:0;">
            <div style="font-weight:600;">Segment Details</div>
            <div id="ts-mvp-segment-detail" style="font-size:12px; color:#bbb;">Select a segment to view details.</div>
          </div>
        </div>
        <div style="margin-top:8px; font-size:11px; color:#666;">lastStep: <span id="ts-mvp-step">${window.__HUB_LAST_STEP}</span></div>
      </div>
    `;
  }

  function renderPanelWithBoundary(tabKey, renderFn) {
    resetPanelSteps(tabKey);
    recordPanelStep(tabKey, "route matched");
    try {
      const result = renderFn();
      if (result && typeof result.then === "function") {
        result.catch((err) => {
          renderPanelError(tabKey, err, "Retry", () => renderPanelWithBoundary(tabKey, renderFn));
        });
      }
    } catch (err) {
      renderPanelError(tabKey, err, "Retry", () => renderPanelWithBoundary(tabKey, renderFn));
    }
  }

  function routeFromHash() {
    if (!hubTabButton || !topicStudioTabButton) return;
    const panelKey = normalizeHash(window.location.hash);
    const nextHash = `#${panelKey}`;
    if (window.location.hash !== nextHash) {
      window.location.hash = nextHash;
      return;
    }
    const target = document.querySelector(`#appPanels section[data-panel="${panelKey}"]`);
    if (!target) {
      window.location.hash = "#hub";
      return;
    }
    showPanel(panelKey);
    updateTopicStudioDebugOverlay();
  }

  function navigateToHash(nextHash) {
    if (window.location.hash === nextHash) {
      routeFromHash();
      return;
    }
    window.location.hash = nextHash;
  }

  function ensureTopicStudioSentinel() {
    const inlineSentinel = document.getElementById("tsSplitPaneSentinel");
    if (inlineSentinel) {
      inlineSentinel.textContent = TOPIC_STUDIO_SPLIT_SENTINEL;
      inlineSentinel.classList.remove("hidden");
    }
  }

  function updateTopicStudioHostStep() {
    const stepEl = document.getElementById("ts-host-step");
    if (stepEl) {
      stepEl.textContent = window.__HUB_LAST_STEP || "";
    }
  }

  function updateTopicStudioPanelStep() {
    const stepEl = document.getElementById("ts-mvp-step");
    if (stepEl) stepEl.textContent = window.__HUB_LAST_STEP || "";
  }

  function setTopicStudioHostStatus(id, value, color = "#ddd") {
    const target = document.getElementById(id);
    if (!target) return;
    target.textContent = value;
    target.style.color = color;
  }

  function appendTopicStudioHostError(label, err) {
    const container = document.getElementById("ts-host-errors");
    if (!container) return;
    const message = err?.message || err?.toString?.() || "Request failed";
    const detail = truncateCrashText(err?.body || err?.stack || message, 400);
    const box = document.createElement("div");
    box.setAttribute(
      "style",
      "background:#220; color:#f88; border:1px solid #a33; padding:8px; border-radius:8px; font-size:12px; font-family:monospace;"
    );
    box.textContent = `${label}: ${detail}`;
    container.appendChild(box);
  }

  async function refreshTopicStudioRoute() {
    setHubStep("fetch:ready");
    updateTopicStudioHostStep();
    updateTopicStudioPanelStep();
    setTopicStudioHostStatus("ts-host-ready", "loading", "#ddd");
    setTopicStudioHostStatus("ts-host-capabilities", "loading", "#ddd");
    setTopicStudioHostStatus("ts-host-runs", "loading", "#ddd");
    const errorContainer = document.getElementById("ts-host-errors");
    if (errorContainer) errorContainer.innerHTML = "";

    try {
      const data = await tfFetchJson("/ready");
      const status = data?.ok ? "ok" : "degraded";
      setTopicStudioHostStatus("ts-host-ready", status, data?.ok ? "#6f6" : "#ff6");
    } catch (err) {
      setTopicStudioHostStatus("ts-host-ready", "error", "#f66");
      appendTopicStudioHostError("ready", err);
    }

    setHubStep("fetch:capabilities");
    updateTopicStudioHostStep();
    updateTopicStudioPanelStep();
    try {
      await tfFetchJson("/capabilities");
      setTopicStudioHostStatus("ts-host-capabilities", "ok", "#6f6");
    } catch (err) {
      setTopicStudioHostStatus("ts-host-capabilities", "error", "#f66");
      appendTopicStudioHostError("capabilities", err);
    }

    setHubStep("fetch:runs");
    updateTopicStudioHostStep();
    updateTopicStudioPanelStep();
    try {
      await tfFetchJson("/runs?limit=20");
      setTopicStudioHostStatus("ts-host-runs", "ok", "#6f6");
    } catch (err) {
      setTopicStudioHostStatus("ts-host-runs", "error", "#f66");
      appendTopicStudioHostError("runs", err);
    }

    setHubStep("render:done");
    updateTopicStudioHostStep();
    updateTopicStudioPanelStep();
  }

  function resolveTopicStudioSubview() {
    const valid = new Set(["runs", "conversations", "topics", "compare", "drift", "events", "kg"]);
    return valid.has(topicStudioLastSubview) ? topicStudioLastSubview : "runs";
  }

  function setTopicStudioSubview(viewKey) {
    const views = [
      { key: "runs", panel: tsSubviewRuns, button: tsSubviewRunsBtn },
      { key: "conversations", panel: tsSubviewConversations, button: tsSubviewConversationsBtn },
      { key: "topics", panel: tsSubviewTopics, button: tsSubviewTopicsBtn },
      { key: "compare", panel: tsSubviewCompare, button: tsSubviewCompareBtn },
      { key: "drift", panel: tsSubviewDrift, button: tsSubviewDriftBtn },
      { key: "events", panel: tsSubviewEvents, button: tsSubviewEventsBtn },
      { key: "kg", panel: tsSubviewKg, button: tsSubviewKgBtn },
    ];
    views.forEach(({ key, panel, button }) => {
      if (panel) {
        panel.classList.toggle("hidden", key !== viewKey);
      }
      if (button) {
        const isActive = key === viewKey;
        button.classList.toggle("bg-gray-900/60", isActive);
        button.classList.toggle("text-gray-200", isActive);
        button.classList.toggle("text-gray-400", !isActive);
        button.classList.toggle("bg-gray-800", !isActive);
      }
    });
    topicStudioLastSubview = viewKey;
    saveTopicStudioState();
  }

  function saveTopicStudioState() {
    const state = {
      dataset: {
        name: tsDatasetName?.value || "",
        schema: tsDatasetSchema?.value || "",
        table: tsDatasetTableSelect?.value || "",
        source_table: tsDatasetTable?.value || "",
        id_column: tsDatasetIdColumn?.value || "",
        time_column: tsDatasetTimeColumn?.value || "",
        text_columns: tsDatasetTextColumns ? getSelectedValues(tsDatasetTextColumns) : tsDatasetTextColumns?.value || "",
        boundary_column: tsDatasetBoundaryColumn?.value || "",
        where_sql: tsDatasetWhereSql?.value || "",
        timezone: tsDatasetTimezone?.value || "",
      },
      windowing: {
        windowing_mode: tsWindowingMode?.value || "",
        group_by: tsGroupByColumn?.value || "",
        segmentation_mode: tsSegmentationMode?.value || "",
        time_gap_seconds: tsTimeGap?.value || "",
        max_window_seconds: tsMaxWindow?.value || "",
        fixed_k_rows: tsFixedKRows?.value || "",
        fixed_k_rows_step: tsFixedKRowsStep?.value || "",
        min_blocks_per_segment: tsMinBlocks?.value || "",
        max_chars: tsMaxChars?.value || "",
        preset: tsRunPreset?.value || "",
      },
      model: {
        name: tsModelName?.value || "",
        version: tsModelVersion?.value || "",
        stage: tsModelStage?.value || "",
        embedding_source_url: tsModelEmbeddingUrl?.value || "",
        min_cluster_size: tsModelMinCluster?.value || "",
        metric: tsModelMetric?.value || "",
        params: tsModelParams?.value || "",
      },
      run: {
        run_id: tsRunId?.value || "",
        start_at: tsStartAt?.value || "",
        end_at: tsEndAt?.value || "",
      },
      segments: {
        run_id: tsSegmentsRunId?.value || "",
        search: tsSegmentsSearch?.value || "",
        sort: tsSegmentsSort?.value || "",
        has_enrichment: tsSegmentsEnrichment?.value || "",
        aspect: tsSegmentsAspect?.value || "",
        page_size: tsSegmentsPageSize?.value || "",
      },
      conversations: {
        dataset_id: tsConvoDatasetSelect?.value || "",
        start_at: tsConvoStartAt?.value || "",
        end_at: tsConvoEndAt?.value || "",
        limit: tsConvoLimit?.value || "",
      },
      topics: {
        run_id: tsTopicsRunId?.value || "",
        scope: tsTopicsScope?.value || "",
        limit: tsTopicsLimit?.value || "",
        offset: tsTopicsOffset?.value || "",
        segment_limit: tsTopicSegmentsLimit?.value || "",
        segment_offset: tsTopicSegmentsOffset?.value || "",
      },
      compare: {
        left_run_id: tsCompareLeftRunId?.value || "",
        right_run_id: tsCompareRightRunId?.value || "",
      },
      drift: {
        model_name: tsDriftModelName?.value || "",
        window_hours: tsDriftWindowHours?.value || "",
        limit: tsDriftLimit?.value || "",
      },
      events: {
        kind: tsEventsKind?.value || "",
        limit: tsEventsLimit?.value || "",
        offset: tsEventsOffset?.value || "",
      },
      kg: {
        run_id: tsKgRunId?.value || "",
        limit: tsKgLimit?.value || "",
        offset: tsKgOffset?.value || "",
        query: tsKgQuery?.value || "",
        predicate: tsKgPredicate?.value || "",
      },
      last_subview: topicStudioLastSubview || "runs",
    };
    localStorage.setItem(TOPIC_STUDIO_STATE_KEY, JSON.stringify(state));
  }

  function applyTopicStudioState() {
    const raw = localStorage.getItem(TOPIC_STUDIO_STATE_KEY);
    if (!raw) return;
    try {
      const state = JSON.parse(raw);
      if (state.dataset) {
        if (tsDatasetName && state.dataset.name) tsDatasetName.value = state.dataset.name;
        if (tsDatasetSchema && state.dataset.schema) tsDatasetSchema.value = state.dataset.schema;
        if (tsDatasetTableSelect && state.dataset.table) tsDatasetTableSelect.value = state.dataset.table;
        if (tsDatasetTable && state.dataset.source_table) tsDatasetTable.value = state.dataset.source_table;
        if (tsDatasetIdColumn && state.dataset.id_column) tsDatasetIdColumn.value = state.dataset.id_column;
        if (tsDatasetTimeColumn && state.dataset.time_column) tsDatasetTimeColumn.value = state.dataset.time_column;
        if (tsDatasetTextColumns && state.dataset.text_columns) {
          const values = Array.isArray(state.dataset.text_columns)
            ? state.dataset.text_columns
            : String(state.dataset.text_columns).split(",").map((col) => col.trim()).filter(Boolean);
          setSelectedValues(tsDatasetTextColumns, values);
        }
        if (tsDatasetBoundaryColumn && state.dataset.boundary_column) tsDatasetBoundaryColumn.value = state.dataset.boundary_column;
        if (tsDatasetWhereSql && state.dataset.where_sql) tsDatasetWhereSql.value = state.dataset.where_sql;
        if (tsDatasetTimezone && state.dataset.timezone) tsDatasetTimezone.value = state.dataset.timezone;
        updateSourceTableInput();
      }
      if (state.windowing) {
        if (tsWindowingMode && state.windowing.windowing_mode) tsWindowingMode.value = state.windowing.windowing_mode;
        if (tsGroupByColumn && state.windowing.group_by) tsGroupByColumn.value = state.windowing.group_by;
        if (tsSegmentationMode && state.windowing.segmentation_mode) tsSegmentationMode.value = state.windowing.segmentation_mode;
        if (tsTimeGap && state.windowing.time_gap_seconds) tsTimeGap.value = state.windowing.time_gap_seconds;
        if (tsMaxWindow && state.windowing.max_window_seconds) tsMaxWindow.value = state.windowing.max_window_seconds;
        if (tsFixedKRows && state.windowing.fixed_k_rows) tsFixedKRows.value = state.windowing.fixed_k_rows;
        if (tsFixedKRowsStep && state.windowing.fixed_k_rows_step) tsFixedKRowsStep.value = state.windowing.fixed_k_rows_step;
        if (tsMinBlocks && state.windowing.min_blocks_per_segment) tsMinBlocks.value = state.windowing.min_blocks_per_segment;
        if (tsMaxChars && state.windowing.max_chars) tsMaxChars.value = state.windowing.max_chars;
        if (tsRunPreset && state.windowing.preset) tsRunPreset.value = state.windowing.preset;
      }
      if (state.model) {
        if (tsModelName && state.model.name) tsModelName.value = state.model.name;
        if (tsModelVersion && state.model.version) tsModelVersion.value = state.model.version;
        if (tsModelStage && state.model.stage) tsModelStage.value = state.model.stage;
        if (tsModelEmbeddingUrl && state.model.embedding_source_url) tsModelEmbeddingUrl.value = state.model.embedding_source_url;
        if (tsModelMinCluster && state.model.min_cluster_size) tsModelMinCluster.value = state.model.min_cluster_size;
        if (tsModelMetric && state.model.metric) tsModelMetric.value = state.model.metric;
        if (tsModelParams && state.model.params) tsModelParams.value = state.model.params;
      }
      if (state.run) {
        if (tsRunId && state.run.run_id) tsRunId.value = state.run.run_id;
        if (tsStartAt && state.run.start_at) tsStartAt.value = state.run.start_at;
        if (tsEndAt && state.run.end_at) tsEndAt.value = state.run.end_at;
      }
      if (state.segments) {
        if (tsSegmentsRunId && state.segments.run_id) tsSegmentsRunId.value = state.segments.run_id;
        if (tsSegmentsSearch && state.segments.search) tsSegmentsSearch.value = state.segments.search;
        if (tsSegmentsSort && state.segments.sort) tsSegmentsSort.value = state.segments.sort;
        if (tsSegmentsEnrichment && state.segments.has_enrichment) tsSegmentsEnrichment.value = state.segments.has_enrichment;
        if (tsSegmentsAspect && state.segments.aspect) tsSegmentsAspect.value = state.segments.aspect;
        if (tsSegmentsPageSize && state.segments.page_size) tsSegmentsPageSize.value = state.segments.page_size;
      }
      if (state.conversations) {
        if (tsConvoDatasetSelect && state.conversations.dataset_id) tsConvoDatasetSelect.value = state.conversations.dataset_id;
        if (tsConvoStartAt && state.conversations.start_at) tsConvoStartAt.value = state.conversations.start_at;
        if (tsConvoEndAt && state.conversations.end_at) tsConvoEndAt.value = state.conversations.end_at;
        if (tsConvoLimit && state.conversations.limit) tsConvoLimit.value = state.conversations.limit;
      }
      if (state.topics) {
        if (tsTopicsRunId && state.topics.run_id) tsTopicsRunId.value = state.topics.run_id;
        if (tsTopicsScope && state.topics.scope) tsTopicsScope.value = state.topics.scope;
        if (tsTopicsLimit && state.topics.limit) tsTopicsLimit.value = state.topics.limit;
        if (tsTopicsOffset && state.topics.offset) tsTopicsOffset.value = state.topics.offset;
        if (tsTopicSegmentsLimit && state.topics.segment_limit) tsTopicSegmentsLimit.value = state.topics.segment_limit;
        if (tsTopicSegmentsOffset && state.topics.segment_offset) tsTopicSegmentsOffset.value = state.topics.segment_offset;
      }
      if (state.compare) {
        if (tsCompareLeftRunId && state.compare.left_run_id) tsCompareLeftRunId.value = state.compare.left_run_id;
        if (tsCompareRightRunId && state.compare.right_run_id) tsCompareRightRunId.value = state.compare.right_run_id;
      }
      if (state.drift) {
        if (tsDriftModelName && state.drift.model_name) tsDriftModelName.value = state.drift.model_name;
        if (tsDriftWindowHours && state.drift.window_hours) tsDriftWindowHours.value = state.drift.window_hours;
        if (tsDriftLimit && state.drift.limit) tsDriftLimit.value = state.drift.limit;
      }
      if (state.events) {
        if (tsEventsKind && state.events.kind) tsEventsKind.value = state.events.kind;
        if (tsEventsLimit && state.events.limit) tsEventsLimit.value = state.events.limit;
        if (tsEventsOffset && state.events.offset) tsEventsOffset.value = state.events.offset;
      }
      if (state.kg) {
        if (tsKgRunId && state.kg.run_id) tsKgRunId.value = state.kg.run_id;
        if (tsKgLimit && state.kg.limit) tsKgLimit.value = state.kg.limit;
        if (tsKgOffset && state.kg.offset) tsKgOffset.value = state.kg.offset;
        if (tsKgQuery && state.kg.query) tsKgQuery.value = state.kg.query;
        if (tsKgPredicate && state.kg.predicate) tsKgPredicate.value = state.kg.predicate;
      }
      if (state.last_subview) {
        topicStudioLastSubview = state.last_subview;
      }
    } catch (err) {
      console.warn("[TopicStudio] Failed to restore state", err);
    }
    updateGroupByVisibility();
    topicStudioTopicSegmentsOffset = Number(tsTopicSegmentsOffset?.value || 0);
  }

  function bindTopicStudioPersistence() {
    const inputs = [
      tsDatasetName,
      tsDatasetSchema,
      tsDatasetTableSelect,
      tsDatasetTable,
      tsDatasetIdColumn,
      tsDatasetTimeColumn,
      tsDatasetTextColumns,
      tsDatasetBoundaryColumn,
      tsDatasetWhereSql,
      tsDatasetTimezone,
      tsStartAt,
      tsEndAt,
      tsWindowingMode,
      tsGroupByColumn,
      tsSegmentationMode,
      tsTimeGap,
      tsMaxWindow,
      tsFixedKRows,
      tsFixedKRowsStep,
      tsMinBlocks,
      tsMaxChars,
      tsRunPreset,
      tsModelName,
      tsModelVersion,
      tsModelStage,
      tsModelEmbeddingUrl,
      tsModelMinCluster,
      tsModelMetric,
      tsModelParams,
      tsRunId,
      tsSegmentsRunId,
      tsSegmentsSearch,
      tsSegmentsSort,
      tsSegmentsEnrichment,
      tsSegmentsAspect,
      tsSegmentsPageSize,
      tsConvoDatasetSelect,
      tsConvoStartAt,
      tsConvoEndAt,
      tsConvoLimit,
      tsTopicsRunId,
      tsTopicsScope,
      tsTopicsLimit,
      tsTopicsOffset,
      tsTopicSegmentsLimit,
      tsTopicSegmentsOffset,
      tsCompareLeftRunId,
      tsCompareRightRunId,
      tsDriftModelName,
      tsDriftWindowHours,
      tsDriftLimit,
      tsEventsKind,
      tsEventsLimit,
      tsEventsOffset,
      tsKgRunId,
      tsKgLimit,
      tsKgOffset,
      tsKgQuery,
      tsKgPredicate,
    ];
    inputs.forEach((input) => {
      if (!input) return;
      input.addEventListener("change", saveTopicStudioState);
      input.addEventListener("input", saveTopicStudioState);
    });
  }

  function parseJsonInput(value, fallback = null) {
    if (!value || !value.trim()) return fallback;
    try {
      return JSON.parse(value);
    } catch (err) {
      showToast("Invalid JSON input.");
      return fallback;
    }
  }

  function parseDateInput(value) {
    if (!value) return null;
    const parsed = new Date(value);
    return Number.isNaN(parsed.valueOf()) ? null : parsed.toISOString();
  }

  function buildWindowingSpec() {
    const windowingMode = tsWindowingMode?.value || "turn_pairs";
    const groupBy = tsGroupByColumn?.value || "";
    const blockMode = windowingMode === "turn_pairs" ? "turn_pairs" : windowingMode === "group_by_column" ? "group_by_column" : "rows";
    return {
      block_mode: blockMode,
      windowing_mode: windowingMode,
      fixed_k_rows: Number(tsFixedKRows?.value || 2),
      fixed_k_rows_step: Number(tsFixedKRowsStep?.value || 0) || null,
      boundary_column: tsDatasetBoundaryColumn?.value || null,
      group_by: windowingMode === "group_by_column" ? groupBy : null,
      segmentation_mode: tsSegmentationMode?.value || "time_gap",
      time_gap_seconds: Number(tsTimeGap?.value || 900),
      max_window_seconds: Number(tsMaxWindow?.value || 7200),
      min_blocks_per_segment: Number(tsMinBlocks?.value || 1),
      max_chars: Number(tsMaxChars?.value || 6000),
    };
  }

  function boundaryColumnRequired() {
    const mode = tsWindowingMode?.value || "turn_pairs";
    return mode.startsWith("conversation");
  }

  function updateBoundaryRequirement() {
    const required = boundaryColumnRequired();
    const boundaryValue = tsDatasetBoundaryColumn?.value || "";
    const missing = required && !boundaryValue;
    if (tsPreviewDataset) tsPreviewDataset.disabled = missing;
    if (tsTrainRun) tsTrainRun.disabled = missing;
    if (missing) {
      const action = tsDatasetSelect?.value
        ? "Save the dataset after selecting a boundary column."
        : "Select a boundary column and save the dataset.";
      setWarning(tsPreviewWarning, `Boundary column required for conversation-bound windowing. ${action}`);
      setWarning(tsRunWarning, `Boundary column required for conversation-bound windowing. ${action}`);
    } else {
      setWarning(tsPreviewWarning, null);
      setWarning(tsRunWarning, null);
    }
  }

  function updateGroupByVisibility() {
    if (!tsGroupByRow || !tsGroupByColumn) return;
    const isGroupBy = tsWindowingMode?.value === "group_by_column";
    tsGroupByColumn.disabled = !isGroupBy;
    tsGroupByColumn.classList.toggle("opacity-50", !isGroupBy);
  }

  function renderGroupByOptions() {
    if (!tsGroupByColumn) return;
    tsGroupByColumn.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select column…";
    tsGroupByColumn.appendChild(placeholder);
    topicStudioGroupByCandidates.forEach((col) => {
      const option = document.createElement("option");
      option.value = col.column_name;
      option.textContent = `${col.column_name} (${col.data_type || col.udt_name || "--"})`;
      tsGroupByColumn.appendChild(option);
    });
  }

  function renderBoundaryOptions() {
    if (!tsDatasetBoundaryColumn) return;
    tsDatasetBoundaryColumn.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "No boundary";
    tsDatasetBoundaryColumn.appendChild(placeholder);
    topicStudioBoundaryCandidates.forEach((col) => {
      const option = document.createElement("option");
      option.value = col.column_name;
      option.textContent = `${col.column_name} (${col.data_type || col.udt_name || "--"})`;
      tsDatasetBoundaryColumn.appendChild(option);
    });
  }

  function renderSchemaOptionsForDataset() {
    if (!tsDatasetSchema) return;
    tsDatasetSchema.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select schema…";
    tsDatasetSchema.appendChild(placeholder);
    topicStudioIntrospectionSchemas.forEach((schema) => {
      const option = document.createElement("option");
      option.value = schema;
      option.textContent = schema;
      tsDatasetSchema.appendChild(option);
    });
  }

  function renderTableOptionsForDataset() {
    if (!tsDatasetTableSelect) return;
    tsDatasetTableSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select table…";
    tsDatasetTableSelect.appendChild(placeholder);
    topicStudioIntrospectionTables.forEach((table) => {
      const option = document.createElement("option");
      option.value = table.table_name;
      option.textContent = table.table_name;
      tsDatasetTableSelect.appendChild(option);
    });
  }

  function updateSourceTableInput() {
    if (!tsDatasetTable) return;
    if (!tsDatasetTable.readOnly) return;
    const schema = tsDatasetSchema?.value || "";
    const table = tsDatasetTableSelect?.value || "";
    if (schema && table) {
      tsDatasetTable.value = `${schema}.${table}`;
    } else if (table) {
      tsDatasetTable.value = table;
    } else {
      tsDatasetTable.value = "";
    }
  }

  function setDatasetSourceEditable(enabled) {
    if (!tsDatasetTable) return;
    tsDatasetTable.readOnly = !enabled;
    tsDatasetTable.classList.toggle("opacity-60", !enabled);
  }

  function renderDatasetColumnOptions() {
    if (tsDatasetIdColumn) {
      tsDatasetIdColumn.innerHTML = '<option value="">Select id column…</option>';
    }
    if (tsDatasetTimeColumn) {
      tsDatasetTimeColumn.innerHTML = '<option value="">Select time column…</option>';
    }
    if (tsDatasetTextColumns) {
      tsDatasetTextColumns.innerHTML = "";
    }
    if (!topicStudioDatasetColumns.length) {
      renderBoundaryOptions();
      return;
    }
    if (tsDatasetIdColumn) {
      topicStudioDatasetColumns.forEach((col) => {
        const option = document.createElement("option");
        option.value = col.column_name;
        option.textContent = `${col.column_name} (${col.data_type || col.udt_name || "--"})`;
        tsDatasetIdColumn.appendChild(option);
      });
    }
    if (tsDatasetTimeColumn) {
      topicStudioDatasetColumns.forEach((col) => {
        const option = document.createElement("option");
        option.value = col.column_name;
        option.textContent = `${col.column_name} (${col.data_type || col.udt_name || "--"})`;
        tsDatasetTimeColumn.appendChild(option);
      });
    }
    if (tsDatasetTextColumns) {
      topicStudioDatasetColumns.forEach((col) => {
        const option = document.createElement("option");
        option.value = col.column_name;
        option.textContent = `${col.column_name} (${col.data_type || col.udt_name || "--"})`;
        tsDatasetTextColumns.appendChild(option);
      });
    }
    renderBoundaryOptions();
  }

  function parseSourceTableValue(value) {
    const trimmed = String(value || "").trim();
    if (!trimmed) return { schema: "", table: "" };
    if (trimmed.includes(".")) {
      const [schema, table] = trimmed.split(".", 2);
      return { schema, table };
    }
    const schema = topicStudioIntrospectionSchemas[0] || "";
    return { schema, table: trimmed };
  }

  async function loadDatasetColumnsFromSourceTable(sourceTable) {
    const { schema, table } = parseSourceTableValue(sourceTable);
    if (!schema || !table) {
      topicStudioDatasetColumns = [];
      renderDatasetColumnOptions();
      return;
    }
    try {
      const response = await topicFoundryFetch(`/introspect/columns?schema=${encodeURIComponent(schema)}&table=${encodeURIComponent(table)}`);
      topicStudioDatasetColumns = response?.columns || [];
      topicStudioGroupByCandidates = [...topicStudioDatasetColumns];
      topicStudioBoundaryCandidates = [...topicStudioDatasetColumns];
      renderGroupByOptions();
      renderDatasetColumnOptions();
    } catch (err) {
      topicStudioDatasetColumns = [];
      topicStudioGroupByCandidates = [];
      topicStudioBoundaryCandidates = [];
      renderGroupByOptions();
      renderDatasetColumnOptions();
    }
  }

  async function loadDatasetTables(schema) {
    if (!schema) {
      topicStudioIntrospectionTables = [];
      renderTableOptionsForDataset();
      return;
    }
    try {
      const priorTable = tsDatasetTableSelect?.value || "";
      const response = await topicFoundryFetch(`/introspect/tables?schema=${encodeURIComponent(schema)}`);
      topicStudioIntrospectionTables = response?.tables || [];
      renderTableOptionsForDataset();
      if (tsDatasetTableSelect && priorTable) {
        const exists = topicStudioIntrospectionTables.some((table) => table.table_name === priorTable);
        if (exists) tsDatasetTableSelect.value = priorTable;
      }
    } catch (err) {
      topicStudioIntrospectionTables = [];
      renderTableOptionsForDataset();
    }
  }

  async function loadGroupByCandidates(schema, table) {
    if (!schema || !table) {
      topicStudioGroupByCandidates = [];
      renderGroupByOptions();
      return;
    }
    try {
      const response = await topicFoundryFetch(`/introspect/columns?schema=${encodeURIComponent(schema)}&table=${encodeURIComponent(table)}`);
      const columns = response?.columns || [];
      topicStudioGroupByCandidates = [...columns];
      renderGroupByOptions();
      topicStudioBoundaryCandidates = topicStudioGroupByCandidates;
      renderBoundaryOptions();
    } catch (err) {
      topicStudioGroupByCandidates = [];
      renderGroupByOptions();
      topicStudioBoundaryCandidates = [];
      renderBoundaryOptions();
    }
  }

  async function loadGroupByCandidatesFromDataset(dataset) {
    if (!dataset?.source_table) {
      topicStudioGroupByCandidates = [];
      renderGroupByOptions();
      return;
    }
    const parts = String(dataset.source_table).split(".");
    if (parts.length === 2) {
      await loadGroupByCandidates(parts[0], parts[1]);
      return;
    }
    const schema = topicStudioIntrospectionSchemas[0] || "";
    if (!schema) {
      topicStudioGroupByCandidates = [];
      renderGroupByOptions();
      return;
    }
    await loadGroupByCandidates(schema, parts[0]);
  }

  function recordTopicStudioDebug(panel, payload) {
    if (!HUB_DEBUG) return;
    if (tsDebugDrawer) tsDebugDrawer.classList.remove("hidden");
    const pretty = JSON.stringify(payload, null, 2);
    if (panel === "preview" && tsDebugPreview) tsDebugPreview.textContent = pretty;
    if (panel === "train" && tsDebugTrain) tsDebugTrain.textContent = pretty;
    if (panel === "enrich" && tsDebugEnrich) tsDebugEnrich.textContent = pretty;
    if (panel === "segments" && tsDebugSegments) tsDebugSegments.textContent = pretty;
  }

  function formatDetailValue(value) {
    if (value === null || value === undefined || value === "") return "--";
    if (typeof value === "string") return value;
    return JSON.stringify(value, null, 2);
  }

  function renderSegmentDetailPanel(detail, options = {}) {
    topicStudioSelectedSegment = detail || null;
    topicStudioSelectedSegmentId = detail?.segment_id || null;
    if (tsSegmentDetailHeader) {
      tsSegmentDetailHeader.textContent = detail?.segment_id || "Select a segment";
    }
    if (tsSegmentDetailMeta) {
      if (!detail) {
        tsSegmentDetailMeta.textContent = "--";
      } else {
        const meta = [
          `run_id=${detail.run_id || "--"}`,
          `created_at=${detail.created_at || "--"}`,
          `rows=${detail.row_ids_count ?? detail.size ?? "--"}`,
          `chars=${detail.chars ?? "--"}`,
          `start=${detail.start_at || "--"}`,
          `end=${detail.end_at || "--"}`,
        ];
        tsSegmentDetailMeta.textContent = meta.join(" · ");
      }
    }
    if (tsSegmentDetailSnippet) {
      tsSegmentDetailSnippet.textContent = formatDetailValue(detail?.snippet || detail?.text);
    }
    if (tsSegmentDetailMeaning) {
      if (!detail) {
        tsSegmentDetailMeaning.textContent = "--";
      } else {
        const payload = {
          title: detail.title || detail.label || null,
          aspects: detail.aspects || null,
          meaning: detail.meaning || null,
          enrichment: detail.enrichment || null,
          sentiment: detail.sentiment || null,
          topic_id: detail.topic_id ?? null,
          is_outlier: detail.is_outlier ?? null,
          provenance: detail.provenance || null,
        };
        tsSegmentDetailMeaning.textContent = JSON.stringify(payload, null, 2);
      }
    }
    const requestedFullText = options.include_full_text === true;
    if (!detail) {
      topicStudioSegmentFullText = "";
      if (tsSegmentFullTextStatus) tsSegmentFullTextStatus.textContent = "--";
    } else if (requestedFullText) {
      const hasFullText = Object.prototype.hasOwnProperty.call(detail, "full_text");
      if (!hasFullText) {
        topicStudioSegmentFullText = "";
        if (tsSegmentFullTextStatus) tsSegmentFullTextStatus.textContent = "backend missing include_full_text support";
      } else {
        topicStudioSegmentFullText = detail.full_text || "";
        if (tsSegmentFullTextStatus) {
          tsSegmentFullTextStatus.textContent = `Full text loaded (${topicStudioSegmentFullText.length} chars).`;
        }
      }
    } else {
      topicStudioSegmentFullText = "";
      if (tsSegmentFullTextStatus) tsSegmentFullTextStatus.textContent = "Full text not loaded.";
    }
    topicStudioSegmentFullTextExpanded = true;
    if (tsSegmentFullTextExpand) tsSegmentFullTextExpand.textContent = "Collapse";
    if (tsSegmentFullText) {
      tsSegmentFullText.style.maxHeight = "none";
    }
    renderSegmentFullText();
    if (tsSegmentFullTextLoad) tsSegmentFullTextLoad.disabled = !detail;
    if (tsSegmentFullTextExpand) tsSegmentFullTextExpand.disabled = !detail;
    if (tsSegmentFullTextCopy) tsSegmentFullTextCopy.disabled = !detail;
  }

  function renderSegmentFullText() {
    if (!tsSegmentFullText) return;
    if (!topicStudioSegmentFullText) {
      tsSegmentFullText.textContent = "--";
      return;
    }
    tsSegmentFullText.textContent = topicStudioSegmentFullText;
    if (tsSegmentFullTextExpand) {
      tsSegmentFullTextExpand.disabled = false;
    }
    if (tsSegmentFullTextStatus && !topicStudioSegmentFullTextExpanded) {
      tsSegmentFullTextStatus.textContent = `Full text loaded (${topicStudioSegmentFullText.length} chars).`;
    }
  }

  function applyRunPreset(value) {
    if (!value) return;
    if (value === "conversation_view") {
      if (tsWindowingMode) tsWindowingMode.value = "conversation_bound_then_time_gap";
      if (tsSegmentationMode) tsSegmentationMode.value = "time_gap";
      if (tsMaxChars) tsMaxChars.value = "6000";
      if (tsTimeGap) tsTimeGap.value = "900";
      if (tsMinBlocks) tsMinBlocks.value = "1";
      if (tsFixedKRows) tsFixedKRows.value = "2";
    }
    if (value === "global_themes") {
      if (tsWindowingMode) tsWindowingMode.value = "turn_pairs";
      if (tsSegmentationMode) tsSegmentationMode.value = "time_gap";
      if (tsMaxChars) tsMaxChars.value = "12000";
      if (tsTimeGap) tsTimeGap.value = "900";
      if (tsMinBlocks) tsMinBlocks.value = "1";
    }
    updateGroupByVisibility();
    saveTopicStudioState();
  }

  function getSelectedValues(selectEl) {
    if (!selectEl) return [];
    return Array.from(selectEl.selectedOptions || []).map((option) => option.value).filter(Boolean);
  }

  function setSelectedValues(selectEl, values) {
    if (!selectEl) return;
    const valueSet = new Set(values || []);
    Array.from(selectEl.options || []).forEach((option) => {
      option.selected = valueSet.has(option.value);
    });
  }

  function resolveDatasetSourceTable() {
    const schema = tsDatasetSchema?.value?.trim() || "";
    const table = tsDatasetTableSelect?.value?.trim() || "";
    if (schema && table) return `${schema}.${table}`;
    if (table) return table;
    return tsDatasetTable?.value?.trim() || "";
  }

  function buildDatasetSpec() {
    const textColumns = tsDatasetTextColumns
      ? getSelectedValues(tsDatasetTextColumns)
      : (tsDatasetTextColumns?.value || "")
          .split(",")
          .map((col) => col.trim())
          .filter(Boolean);
    return {
      name: tsDatasetName?.value?.trim() || "",
      source_table: resolveDatasetSourceTable(),
      id_column: tsDatasetIdColumn?.value?.trim() || "",
      time_column: tsDatasetTimeColumn?.value?.trim() || "",
      text_columns: textColumns,
      boundary_column: tsDatasetBoundaryColumn?.value || null,
      boundary_strategy: tsDatasetBoundaryColumn?.value ? "column" : null,
      where_sql: tsDatasetWhereSql?.value?.trim() || "",
      timezone: tsDatasetTimezone?.value?.trim() || "UTC",
    };
  }

  function setDatasetSaveStatus(message, isError = false) {
    if (!tsDatasetSaveStatus) return;
    tsDatasetSaveStatus.textContent = message || "--";
    tsDatasetSaveStatus.classList.toggle("text-red-400", isError);
    tsDatasetSaveStatus.classList.toggle("text-gray-500", !isError);
  }

  async function tfFetchJson(path, options = {}) {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    const response = await hubFetch(`${TOPIC_FOUNDRY_PROXY_BASE}${normalizedPath}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    const payloadText = await response.text();
    const contentType = response.headers.get("content-type", "");
    let payload = null;
    if (payloadText && contentType.includes("application/json")) {
      try {
        payload = JSON.parse(payloadText);
      } catch (err) {
        payload = null;
      }
    }
    if (!response.ok) {
      const error = {
        status: response.status,
        body: payloadText,
        message: payloadText || response.statusText || `Request failed (${response.status})`,
      };
      throw error;
    }
    if (response.status === 204) return {};
    return payload || {};
  }

  async function topicFoundryFetch(path, options = {}) {
    return tfFetchJson(path, options);
  }

  async function topicFoundryFetchWithHeaders(path, options = {}) {
    const response = await hubFetch(`${TOPIC_FOUNDRY_PROXY_BASE}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    const payloadText = await response.text();
    const contentType = response.headers.get("content-type", "");
    let payload = payloadText;
    if (payloadText && contentType.includes("application/json")) {
      try {
        payload = JSON.parse(payloadText);
      } catch (err) {
        payload = payloadText;
      }
    }
    if (!response.ok) {
      const error = new Error(payloadText || response.statusText || `Request failed (${response.status})`);
      error.status = response.status;
      error.body = payloadText;
      throw error;
    }
    return { payload, headers: response.headers };
  }

  function formatStatusBadge(target, ok, label) {
    if (!target) return;
    target.textContent = label;
    target.classList.remove("text-green-300", "text-yellow-300", "text-red-300");
    if (ok === true) {
      target.classList.add("text-green-300");
    } else if (ok === false) {
      target.classList.add("text-red-300");
    } else {
      target.classList.add("text-yellow-300");
    }
  }

  function setLoading(target, isLoading, label = "Loading...") {
    if (!target) return;
    target.textContent = label;
    target.classList.toggle("hidden", !isLoading);
  }

  function debounce(fn, wait = 300) {
    let timer = null;
    return (...args) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  async function copyText(value, successMessage = "Copied to clipboard.") {
    if (!value) {
      showToast("Nothing to copy.");
      return;
    }
    try {
      await navigator.clipboard.writeText(String(value));
      showToast(successMessage);
    } catch (err) {
      console.warn("[TopicStudio] Clipboard copy failed", err);
      showToast("Failed to copy.");
    }
  }

  let topicStudioDatasets = [];
  let topicStudioModels = [];
  let topicStudioRuns = [];
  let topicStudioRunPoller = null;
  let topicStudioCapabilities = null;
  let topicStudioSegmentsPolling = false;
  let topicStudioEnrichPolling = false;
  let topicStudioSegmentsOffset = 0;
  let topicStudioSegmentsLimit = 50;
  let topicStudioSegmentsTotal = null;
  let topicStudioSegmentsPage = [];
  let topicStudioSegmentsDisplayed = [];
  let topicStudioSegmentsQueryKey = null;
  let topicStudioLastPreview = null;
  let topicStudioLastSubview = "runs";
  let topicStudioSelectedSegmentId = null;
  let topicStudioSegmentsFacetFilter = null;
  let topicStudioSegmentsLastFacets = null;
  let topicStudioConversations = [];
  let topicStudioConversationSelection = new Set();
  let topicStudioSelectedConversationId = null;
  let topicStudioSelectedTopicId = null;
  let topicStudioTopicSegmentsOffset = 0;
  let topicStudioDriftPolling = false;
  let topicStudioEventsPage = [];
  let topicStudioKgEdgesPage = [];
  let topicStudioIntrospectionSchemas = [];
  let topicStudioIntrospectionTables = [];
  let topicStudioGroupByCandidates = [];
  let topicStudioBoundaryCandidates = [];
  let topicStudioDatasetColumns = [];
  let topicStudioSelectedSegment = null;
  let topicStudioSegmentFullText = "";
  let topicStudioSegmentFullTextExpanded = false;
  const TOPIC_STUDIO_RUN_ID_KEY = "topic_studio_run_id_v1";
  const topicStudioMvpState = {
    datasets: [],
    models: [],
    runs: [],
    previewDatasetId: "",
    selectedDatasetId: "",
    selectedModelId: "",
    selectedRunId: "",
    runPoller: null,
    introspectionOk: false,
    schemas: [],
    tables: [],
    columns: [],
    defaultEmbeddingUrl: "",
  };
  const topicStudioDebugState = {
    enabled: new URLSearchParams(window.location.search).get("debug") === "1",
    lastRenderStep: "init",
    fetchStatus: {
      ready: null,
      capabilities: null,
      runs: null,
    },
    overlay: null,
    overlayBody: null,
  };

  function formatFetchStatus(status) {
    if (!status) return "--";
    const okLabel = status.ok === true ? "ok" : status.ok === false ? "fail" : "unknown";
    const detail = status.detail ? ` · ${truncateText(status.detail, 80)}` : "";
    return `${status.status ?? "--"} (${okLabel})${detail}`;
  }

  function ensureTopicStudioDebugOverlay() {
    if (!topicStudioDebugState.enabled || topicStudioDebugState.overlay || !topicStudioPanel) return;
    const overlay = document.createElement("div");
    overlay.className = "fixed bottom-3 right-3 z-50 bg-gray-900/95 border border-gray-700 rounded-lg px-3 py-2 text-[10px] text-gray-200 shadow-lg";
    overlay.style.maxWidth = "240px";
    overlay.innerHTML = `
      <div class="flex items-center justify-between gap-2 mb-1">
        <div class="font-semibold text-xs">Topic Studio Debug</div>
        <button type="button" class="text-gray-400 hover:text-gray-200 text-[10px]" data-debug-hide>Hide</button>
      </div>
      <div data-debug-body class="space-y-1"></div>
    `;
    overlay.querySelector("[data-debug-hide]")?.addEventListener("click", () => {
      overlay.classList.add("hidden");
    });
    topicStudioDebugState.overlay = overlay;
    topicStudioDebugState.overlayBody = overlay.querySelector("[data-debug-body]");
    topicStudioPanel.appendChild(overlay);
  }

  function updateTopicStudioDebugOverlay() {
    if (!topicStudioDebugState.enabled) return;
    if (window.location.hash !== "#topic-studio") {
      topicStudioDebugState.overlay?.classList.add("hidden");
      return;
    }
    ensureTopicStudioDebugOverlay();
    topicStudioDebugState.overlay?.classList.remove("hidden");
    if (!topicStudioDebugState.overlayBody) return;
    const hash = window.location.hash || "(none)";
    const exists = Boolean(topicStudioPanel);
    topicStudioDebugState.overlayBody.innerHTML = `
      <div>hash: <span class="text-gray-400">${hash}</span></div>
      <div>container: <span class="text-gray-400">${exists ? "found" : "missing"}</span></div>
      <div>step: <span class="text-gray-400">${topicStudioDebugState.lastRenderStep}</span></div>
      <div>/ready: <span class="text-gray-400">${formatFetchStatus(topicStudioDebugState.fetchStatus.ready)}</span></div>
      <div>/capabilities: <span class="text-gray-400">${formatFetchStatus(topicStudioDebugState.fetchStatus.capabilities)}</span></div>
      <div>/runs: <span class="text-gray-400">${formatFetchStatus(topicStudioDebugState.fetchStatus.runs)}</span></div>
    `;
  }

  function setTopicStudioRenderStep(step) {
    topicStudioDebugState.lastRenderStep = step;
    updateTopicStudioDebugOverlay();
  }

  function recordTopicStudioFetchStatus(key, status, ok, detail) {
    topicStudioDebugState.fetchStatus[key] = { status, ok, detail };
    updateTopicStudioDebugOverlay();
  }

  function setMvpError(message) {
    const container = document.getElementById("ts-mvp-errors");
    if (!container) return;
    const box = document.createElement("div");
    box.setAttribute(
      "style",
      "background:#220; color:#f88; border:1px solid #a33; padding:8px; border-radius:8px; font-size:12px; font-family:monospace;"
    );
    box.textContent = message;
    container.appendChild(box);
  }

  function parseDatasets(json) {
    const arr = json && Array.isArray(json.datasets)
      ? json.datasets
      : json && Array.isArray(json.items)
        ? json.items
        : Array.isArray(json)
          ? json
          : [];
    return arr
      .map((d) => {
        const id = d.dataset_id || d.id;
        return {
          id,
          name: d.name || id,
          raw: d,
        };
      })
      .filter((x) => x.id);
  }

  function formatHttpError(label, error) {
    const status = error?.status ? `status ${error.status}` : "status unknown";
    let detail = extractHttpErrorDetail(error);
    detail = truncateCrashText(detail, 300);
    return `${label}: ${status} ${detail}`.trim();
  }

  function extractHttpErrorDetail(error) {
    let detail = error?.body || error?.message || "";
    if (typeof detail === "string") {
      try {
        const parsed = JSON.parse(detail);
        if (parsed?.detail !== undefined) {
          detail = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
        } else if (parsed?.error !== undefined) {
          detail = typeof parsed.error === "string" ? parsed.error : JSON.stringify(parsed.error);
        } else {
          detail = JSON.stringify(parsed);
        }
      } catch (err) {
        detail = detail;
      }
    } else if (detail && typeof detail === "object") {
      detail = JSON.stringify(detail);
    }
    return detail || "Request failed.";
  }

  function setIntrospectionWarning(message) {
    const warning = document.getElementById("ts-mvp-introspect-warning");
    if (!warning) return;
    if (!message) {
      warning.style.display = "none";
      warning.textContent = "";
      return;
    }
    warning.style.display = "block";
    warning.textContent = message;
  }

  function setManualDatasetFields(enabled) {
    const manual = document.getElementById("ts-mvp-manual-fields");
    const schemaSelect = document.getElementById("ts-mvp-schema");
    const tableSelect = document.getElementById("ts-mvp-table");
    const idSelect = document.getElementById("ts-mvp-column-id");
    const timeSelect = document.getElementById("ts-mvp-column-time");
    const textList = document.getElementById("ts-mvp-column-texts");
    if (manual) {
      manual.style.display = enabled ? "flex" : "none";
    }
    if (schemaSelect) schemaSelect.disabled = enabled;
    if (tableSelect) tableSelect.disabled = enabled;
    if (idSelect) idSelect.disabled = enabled;
    if (timeSelect) timeSelect.disabled = enabled;
    if (textList) textList.style.opacity = enabled ? "0.6" : "1";
  }

  function renderSchemaOptions() {
    const select = document.getElementById("ts-mvp-schema");
    if (!select) return;
    select.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select schema…";
    select.appendChild(placeholder);
    topicStudioMvpState.schemas.forEach((schema) => {
      const option = document.createElement("option");
      option.value = schema;
      option.textContent = schema;
      select.appendChild(option);
    });
    const preferred = topicStudioMvpState.schemas.includes("public") ? "public" : topicStudioMvpState.schemas[0];
    if (preferred) {
      select.value = preferred;
    }
  }

  function renderTableOptions() {
    const select = document.getElementById("ts-mvp-table");
    if (!select) return;
    select.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select table…";
    select.appendChild(placeholder);
    topicStudioMvpState.tables.forEach((table) => {
      const option = document.createElement("option");
      option.value = table.table_name;
      option.textContent = `${table.table_name} (${table.table_type})`;
      select.appendChild(option);
    });
  }

  function renderColumnOptions() {
    const idSelect = document.getElementById("ts-mvp-column-id");
    const timeSelect = document.getElementById("ts-mvp-column-time");
    const textList = document.getElementById("ts-mvp-column-texts");
    if (idSelect) {
      idSelect.innerHTML = '<option value="">Select id column…</option>';
    }
    if (timeSelect) {
      timeSelect.innerHTML = '<option value="">Select time column…</option>';
    }
    if (textList) {
      textList.innerHTML = "";
    }
    if (topicStudioMvpState.columns.length === 0) {
      if (textList) textList.textContent = "(no columns)";
      return;
    }
    topicStudioMvpState.columns.forEach((col) => {
      const label = `${col.column_name} (${col.data_type}/${col.udt_name})${col.is_nullable ? "" : " [not null]"}`;
      if (idSelect) {
        const option = document.createElement("option");
        option.value = col.column_name;
        option.textContent = label;
        idSelect.appendChild(option);
      }
      if (timeSelect) {
        const option = document.createElement("option");
        option.value = col.column_name;
        option.textContent = label;
        timeSelect.appendChild(option);
      }
      if (textList) {
        const row = document.createElement("label");
        row.style.display = "flex";
        row.style.gap = "6px";
        row.style.alignItems = "center";
        row.style.marginBottom = "4px";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = col.column_name;
        const span = document.createElement("span");
        span.textContent = label;
        row.appendChild(checkbox);
        row.appendChild(span);
        textList.appendChild(row);
      }
    });
  }

  async function loadSchemas() {
    try {
      const result = await tfFetchJson("/introspect/schemas");
      topicStudioMvpState.schemas = result?.schemas || [];
      if (topicStudioMvpState.schemas.length === 0) {
        setIntrospectionWarning("No schemas available; manual entry enabled.");
        setManualDatasetFields(true);
        return;
      }
      setIntrospectionWarning("");
      setManualDatasetFields(false);
      renderSchemaOptions();
    } catch (err) {
      topicStudioMvpState.schemas = [];
      setIntrospectionWarning("Introspection unavailable; manual entry enabled.");
      setManualDatasetFields(true);
    }
  }

  async function loadTables(schema) {
    if (!schema) return;
    try {
      const result = await tfFetchJson(`/introspect/tables?schema=${encodeURIComponent(schema)}`);
      topicStudioMvpState.tables = result?.tables || [];
      renderTableOptions();
    } catch (err) {
      topicStudioMvpState.tables = [];
      renderTableOptions();
      setIntrospectionWarning("Failed to load tables; manual entry enabled.");
      setManualDatasetFields(true);
    }
  }

  async function loadColumns(schema, table) {
    if (!schema || !table) return;
    try {
      const result = await tfFetchJson(
        `/introspect/columns?schema=${encodeURIComponent(schema)}&table=${encodeURIComponent(table)}`
      );
      topicStudioMvpState.columns = result?.columns || [];
      renderColumnOptions();
    } catch (err) {
      topicStudioMvpState.columns = [];
      renderColumnOptions();
      setIntrospectionWarning("Failed to load columns; manual entry enabled.");
      setManualDatasetFields(true);
    }
  }

  function clearMvpErrors() {
    const container = document.getElementById("ts-mvp-errors");
    if (container) container.innerHTML = "";
  }

  async function safeJsonFetch(path, options) {
    try {
      const resp = await hubFetch(apiUrl(path), options);
      const text = await resp.text();
      let json = null;
      if (text) {
        try {
          json = JSON.parse(text);
        } catch (err) {
          json = text;
        }
      }
      if (!resp.ok) {
        const error = new Error(`status ${resp.status}`);
        error.status = resp.status;
        error.body = text;
        return { ok: false, resp, json, error };
      }
      return { ok: true, resp, json };
    } catch (err) {
      return { ok: false, resp: null, json: null, error: err };
    }
  }

  function renderDatasetList() {
    const container = document.getElementById("ts-mvp-datasets-list");
    if (!container) return;
    if (topicStudioMvpState.datasets.length === 0) {
      container.textContent = "(none)";
      return;
    }
    container.innerHTML = topicStudioMvpState.datasets
      .map((ds) => `${ds.name} (${ds.id})`)
      .join("<br />");
  }

  function renderModelList() {
    const container = document.getElementById("ts-mvp-models-list");
    if (!container) return;
    if (topicStudioMvpState.models.length === 0) {
      container.textContent = "(none)";
      return;
    }
    container.innerHTML = topicStudioMvpState.models
      .map((model) => `${model.name || model.model_id} ${model.version || ""} (${model.model_id})`)
      .join("<br />");
  }

  function renderRunsList() {
    const container = document.getElementById("ts-mvp-runs-list");
    if (!container) return;
    if (topicStudioMvpState.runs.length === 0) {
      container.textContent = "(none)";
      return;
    }
    container.innerHTML = topicStudioMvpState.runs
      .map((run) => `${run.run_id} · ${run.status || "--"} ${run.stage || ""} · ${run.created_at || "--"}`)
      .join("<br />");
  }

  function updateSelectOptions(selectId, items, valueKey, labelFn) {
    const select = document.getElementById(selectId);
    if (!select) return;
    select.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select…";
    select.appendChild(placeholder);
    items.forEach((item) => {
      const option = document.createElement("option");
      option.value = item[valueKey];
      option.textContent = labelFn(item);
      select.appendChild(option);
    });
  }

  function updateMvpSelectors() {
    updateSelectOptions(
      "ts-mvp-preview-dataset",
      topicStudioMvpState.datasets,
      "id",
      (ds) => `${ds.name}`
    );
    updateSelectOptions(
      "ts-mvp-model-dataset",
      topicStudioMvpState.datasets,
      "id",
      (ds) => `${ds.name}`
    );
    updateSelectOptions(
      "ts-mvp-run-dataset",
      topicStudioMvpState.datasets,
      "id",
      (ds) => `${ds.name}`
    );
    updateSelectOptions(
      "ts-mvp-run-model",
      topicStudioMvpState.models,
      "model_id",
      (model) => `${model.name || model.model_id}`
    );
    updateSelectOptions(
      "ts-mvp-segments-run",
      topicStudioMvpState.runs,
      "run_id",
      (run) => `${run.run_id}`
    );
    const previewSelect = document.getElementById("ts-mvp-preview-dataset");
    const previewButton = document.getElementById("ts-mvp-preview");
    if (previewSelect) {
      const validIds = new Set(topicStudioMvpState.datasets.map((ds) => ds.id));
      if (!validIds.has(topicStudioMvpState.previewDatasetId)) {
        topicStudioMvpState.previewDatasetId = "";
      }
      previewSelect.value = topicStudioMvpState.previewDatasetId;
    }
    if (previewSelect && previewButton) {
      const enabled = Boolean(topicStudioMvpState.previewDatasetId);
      previewButton.disabled = !enabled;
      previewButton.style.opacity = enabled ? "1" : "0.6";
      previewButton.style.cursor = enabled ? "pointer" : "not-allowed";
    }
    if (HUB_DEBUG) {
      const debugEl = document.getElementById("ts-mvp-preview-debug");
      if (debugEl) {
        debugEl.textContent = `datasetsList count = ${topicStudioMvpState.datasets.length}`;
      }
    }
  }

  function setLastRefresh() {
    const el = document.getElementById("ts-mvp-last-refresh");
    if (el) el.textContent = `Last refresh: ${new Date().toLocaleTimeString()}`;
  }

  function setTopicStudioDebugLine(message) {
    if (!tsDebugLine) return;
    tsDebugLine.textContent = message || "";
    tsDebugLine.classList.toggle("hidden", !message);
  }

  function bindTopicStudioPanel() {
    const refreshButton = document.getElementById("ts-mvp-refresh");
    if (refreshButton) {
      refreshButton.addEventListener("click", () => {
        clearMvpErrors();
        refreshTopicStudioMvp();
      });
    }
    const schemaSelect = document.getElementById("ts-mvp-schema");
    if (schemaSelect) {
      schemaSelect.addEventListener("change", () => {
        const schema = schemaSelect.value;
        loadTables(schema);
        topicStudioMvpState.columns = [];
        renderColumnOptions();
      });
    }
    const tableSelect = document.getElementById("ts-mvp-table");
    if (tableSelect) {
      tableSelect.addEventListener("change", () => {
        const schema = schemaSelect ? schemaSelect.value : "";
        loadColumns(schema, tableSelect.value);
      });
    }
    const createDatasetBtn = document.getElementById("ts-mvp-create-dataset");
    if (createDatasetBtn) {
      createDatasetBtn.addEventListener("click", () => {
        createDataset().catch((err) => setMvpError(err?.message || "Failed to create dataset"));
      });
    }
    const previewBtn = document.getElementById("ts-mvp-preview");
    if (previewBtn) {
      previewBtn.addEventListener("click", () => {
        previewDataset().catch((err) => setMvpError(err?.message || "Failed to preview dataset"));
      });
    }
    const previewSelect = document.getElementById("ts-mvp-preview-dataset");
    if (previewSelect) {
      previewSelect.addEventListener("change", () => {
        topicStudioMvpState.previewDatasetId = previewSelect.value;
        updateMvpSelectors();
      });
    }
    try {
      const qs = new URLSearchParams();
      qs.set("run_id", "debug");
      setTopicStudioDebugLine(`Topic Studio ready (segments qs ok: ${qs.toString()})`);
    } catch (err) {
      setTopicStudioDebugLine(`Topic Studio init error: ${err?.message || err}`);
    }
    updateBoundaryRequirement();
    const createModelBtn = document.getElementById("ts-mvp-create-model");
    if (createModelBtn) {
      createModelBtn.addEventListener("click", () => {
        createModel().catch((err) => setMvpError(err?.message || "Failed to create model"));
      });
    }
    const trainRunBtn = document.getElementById("ts-mvp-train-run");
    if (trainRunBtn) {
      trainRunBtn.addEventListener("click", () => {
        trainRun().catch((err) => setMvpError(err?.message || "Failed to train run"));
      });
    }
    const enrichBtn = document.getElementById("ts-mvp-enrich-run");
    if (enrichBtn) {
      enrichBtn.addEventListener("click", () => {
        enrichRun().catch((err) => setMvpError(err?.message || "Failed to enrich run"));
      });
    }
    const loadSegmentsBtn = document.getElementById("ts-mvp-load-segments");
    if (loadSegmentsBtn) {
      loadSegmentsBtn.addEventListener("click", () => {
        loadSegmentsMvp().catch((err) => setMvpError(err?.message || "Failed to load segments"));
      });
    }
  }

  async function refreshTopicStudioMvp() {
    clearMvpErrors();
    setHubStep("fetch:ready");
    updateTopicStudioPanelStep();
    let readyResult = { ok: false, json: null, error: null };
    try {
      const json = await tfFetchJson("/ready");
      readyResult = { ok: true, json, error: null };
    } catch (err) {
      readyResult = { ok: false, json: null, error: err };
    }
    const readyEl = document.getElementById("ts-mvp-ready");
    const readyDetail = document.getElementById("ts-mvp-ready-detail");
    if (readyEl) {
      if (readyResult.ok && readyResult.json) {
        readyEl.textContent = readyResult.json.ok ? "ok" : "degraded";
        readyEl.style.color = readyResult.json.ok ? "#6f6" : "#ff6";
        if (readyDetail) {
          const checks = readyResult.json.checks || {};
          readyDetail.textContent = `PG:${checks.pg?.detail || "--"} · Embed:${checks.embedding?.detail || "--"}`;
        }
      } else {
        readyEl.textContent = "error";
        readyEl.style.color = "#f66";
        setMvpError(formatHttpError("ready", readyResult.error));
      }
    }

    setHubStep("fetch:capabilities");
    updateTopicStudioPanelStep();
    let capResult = { ok: false, json: null, error: null };
    try {
      const json = await tfFetchJson("/capabilities");
      capResult = { ok: true, json, error: null };
    } catch (err) {
      capResult = { ok: false, json: null, error: err };
    }
    const capEl = document.getElementById("ts-mvp-capabilities");
    if (capEl) {
      if (capResult.ok) {
        capEl.textContent = "ok";
        capEl.style.color = "#6f6";
        const introspection = capResult.json?.introspection;
        topicStudioMvpState.introspectionOk = Boolean(introspection?.ok);
        topicStudioMvpState.defaultEmbeddingUrl =
          capResult.json?.defaults?.embedding_source_url || capResult.json?.default_embedding_url || "";
        const embedHint = document.getElementById("ts-mvp-embed-hint");
        if (embedHint) {
          embedHint.textContent = topicStudioMvpState.defaultEmbeddingUrl
            ? `Embedding URL (default): ${topicStudioMvpState.defaultEmbeddingUrl}`
            : "";
        }
        if (topicStudioMvpState.introspectionOk) {
          topicStudioMvpState.schemas = introspection.schemas || [];
          await loadSchemas();
          const schemaSelect = document.getElementById("ts-mvp-schema");
          if (schemaSelect && schemaSelect.value) {
            await loadTables(schemaSelect.value);
          }
        } else {
          setIntrospectionWarning("Introspection unavailable; manual entry enabled.");
          setManualDatasetFields(true);
        }
      } else {
        capEl.textContent = "error";
        capEl.style.color = "#f66";
        setMvpError(formatHttpError("capabilities", capResult.error));
        setIntrospectionWarning("Introspection unavailable; manual entry enabled.");
        setManualDatasetFields(true);
      }
    }

    setHubStep("fetch:datasets");
    updateTopicStudioPanelStep();
    try {
      const datasetResult = await tfFetchJson("/datasets");
      topicStudioMvpState.datasets = parseDatasets(datasetResult);
      renderDatasetList();
      updateMvpSelectors();
    } catch (err) {
      topicStudioMvpState.datasets = [];
      renderDatasetList();
      setMvpError(formatHttpError("datasets", err));
    }

    setHubStep("fetch:models");
    updateTopicStudioPanelStep();
    try {
      const modelResult = await tfFetchJson("/models");
      topicStudioMvpState.models = modelResult?.models || [];
      renderModelList();
      updateMvpSelectors();
    } catch (err) {
      topicStudioMvpState.models = [];
      renderModelList();
      setMvpError(formatHttpError("models", err));
    }

    setHubStep("fetch:runs");
    updateTopicStudioPanelStep();
    try {
      const runsResult = await tfFetchJson("/runs?limit=20");
      topicStudioMvpState.runs = normalizeRunsResponse(runsResult);
      renderRunsList();
      updateMvpSelectors();
    } catch (err) {
      topicStudioMvpState.runs = [];
      renderRunsList();
      setMvpError(formatHttpError("runs", err));
    }

    setHubStep("render:done");
    updateTopicStudioPanelStep();
    setLastRefresh();
  }

  async function createDataset() {
    const name = document.getElementById("ts-mvp-dataset-name")?.value?.trim() || "";
    let sourceTable = "";
    let idColumn = null;
    let timeColumn = null;
    let textColumns = [];
    if (topicStudioMvpState.introspectionOk) {
      const schema = document.getElementById("ts-mvp-schema")?.value;
      const table = document.getElementById("ts-mvp-table")?.value;
      if (schema && table) {
        sourceTable = `${schema}.${table}`;
      }
      idColumn = document.getElementById("ts-mvp-column-id")?.value || null;
      timeColumn = document.getElementById("ts-mvp-column-time")?.value || null;
      const textList = document.getElementById("ts-mvp-column-texts");
      if (textList) {
        textColumns = Array.from(textList.querySelectorAll("input[type=checkbox]"))
          .filter((input) => input.checked)
          .map((input) => input.value);
      }
    } else {
      sourceTable = document.getElementById("ts-mvp-dataset-table")?.value?.trim() || "";
      idColumn = document.getElementById("ts-mvp-dataset-id")?.value?.trim() || null;
      timeColumn = document.getElementById("ts-mvp-dataset-time")?.value?.trim() || null;
      textColumns = (document.getElementById("ts-mvp-dataset-text")?.value || "")
        .split(",")
        .map((c) => c.trim())
        .filter(Boolean);
    }
    if (!name || !sourceTable) {
      setMvpError("Dataset name and source_table are required.");
      return;
    }
    const payload = {
      name,
      source_table: sourceTable,
      id_column: idColumn,
      time_column: timeColumn,
      text_columns: textColumns,
      where_sql: document.getElementById("ts-mvp-dataset-where")?.value?.trim() || "",
      timezone: document.getElementById("ts-mvp-dataset-tz")?.value?.trim() || "UTC",
    };
    let result = null;
    try {
      result = await tfFetchJson("/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      setMvpError(formatHttpError("create dataset", err));
      return;
    }
    await refreshTopicStudioMvp();
    if (result?.dataset_id) {
      topicStudioMvpState.selectedDatasetId = result.dataset_id;
      updateMvpSelectors();
    }
  }

  async function previewDataset() {
    const datasetId = topicStudioMvpState.previewDatasetId || document.getElementById("ts-mvp-preview-dataset")?.value;
    const savedDataset = topicStudioMvpState.datasets.find((ds) => ds.id === datasetId)?.raw || null;
    if (!datasetId) {
      setMvpError("Select a saved dataset first.");
      return;
    }
    if (!savedDataset) {
      setMvpError("Select a saved dataset first.");
      return;
    }
    const windowingMode = document.getElementById("ts-mvp-preview-block")?.value || "turn_pairs";
    const blockMode = windowingMode === "turn_pairs" ? "turn_pairs" : windowingMode === "group_by_column" ? "group_by_column" : "rows";
    const windowing = {
      block_mode: blockMode,
      windowing_mode: windowingMode,
      time_gap_seconds: Number(document.getElementById("ts-mvp-preview-gap")?.value || 900),
      max_chars: Number(document.getElementById("ts-mvp-preview-maxchars")?.value || 6000),
    };
    const payload = {
      dataset: savedDataset,
      start_at: document.getElementById("ts-mvp-preview-start")?.value || null,
      end_at: document.getElementById("ts-mvp-preview-end")?.value || null,
      limit: Number(document.getElementById("ts-mvp-preview-limit")?.value || 200),
      windowing,
      windowing_spec: windowing,
    };
    let result = null;
    try {
      result = await tfFetchJson("/datasets/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      const statsEl = document.getElementById("ts-mvp-preview-stats");
      const samplesEl = document.getElementById("ts-mvp-preview-samples");
      if (statsEl) statsEl.textContent = "--";
      if (samplesEl) samplesEl.textContent = "--";
      const message = formatHttpError("preview", err);
      setMvpError(message);
      showToast(message);
      return;
    }
    const statsEl = document.getElementById("ts-mvp-preview-stats");
    const samplesEl = document.getElementById("ts-mvp-preview-samples");
    const stats = result?.stats || {};
    if (statsEl) {
      statsEl.textContent = `docs=${stats.docs_generated ?? "--"} segments=${stats.segments_generated ?? "--"} avg_chars=${stats.avg_chars ?? "--"}`;
    }
    if (samplesEl) {
      const samples = result?.samples || result?.segments || [];
      if (!samples.length) {
        samplesEl.textContent = "(none)";
      } else {
        samplesEl.innerHTML = samples
          .slice(0, 5)
          .map((s) => `${s.segment_id || ""} · ${s.snippet || s.text || ""}`)
          .join("<br />");
      }
    }
  }

  async function createModel() {
    const name = document.getElementById("ts-mvp-model-name")?.value?.trim() || "";
    const version = document.getElementById("ts-mvp-model-version")?.value?.trim() || "";
    const stage = document.getElementById("ts-mvp-model-stage")?.value?.trim() || "development";
    const datasetId = document.getElementById("ts-mvp-model-dataset")?.value;
    if (!name || !version || !datasetId) {
      setMvpError("Model name, version, and dataset are required.");
      return;
    }
    let params = {};
    try {
      const raw = document.getElementById("ts-mvp-model-params")?.value?.trim();
      params = raw ? JSON.parse(raw) : {};
    } catch (err) {
      setMvpError("Model params must be valid JSON.");
      return;
    }
    const windowingMode = document.getElementById("ts-mvp-preview-block")?.value || "turn_pairs";
    const blockMode = windowingMode === "turn_pairs" ? "turn_pairs" : windowingMode === "group_by_column" ? "group_by_column" : "rows";
    const payload = {
      name,
      version,
      stage,
      dataset_id: datasetId,
      model_spec: {
        algorithm: "hdbscan",
        min_cluster_size: Number(document.getElementById("ts-mvp-model-mincluster")?.value || 20),
        metric: document.getElementById("ts-mvp-model-metric")?.value || "cosine",
        params,
      },
      windowing_spec: {
        block_mode: blockMode,
        windowing_mode: windowingMode,
        time_gap_seconds: Number(document.getElementById("ts-mvp-preview-gap")?.value || 900),
        max_chars: Number(document.getElementById("ts-mvp-preview-maxchars")?.value || 6000),
      },
    };
    let result = null;
    try {
      result = await tfFetchJson("/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      setMvpError(formatHttpError("create model", err));
      return;
    }
    await refreshTopicStudioMvp();
    if (result?.model_id) {
      topicStudioMvpState.selectedModelId = result.model_id;
      updateMvpSelectors();
    }
  }

  async function trainRun() {
    const modelId = document.getElementById("ts-mvp-run-model")?.value;
    const datasetId = document.getElementById("ts-mvp-run-dataset")?.value;
    if (!modelId || !datasetId) {
      setMvpError("Select a model and dataset to train.");
      return;
    }
    const payload = {
      model_id: modelId,
      dataset_id: datasetId,
      start_at: document.getElementById("ts-mvp-run-start")?.value || null,
      end_at: document.getElementById("ts-mvp-run-end")?.value || null,
    };
    let result = null;
    try {
      result = await tfFetchJson("/runs/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      setMvpError(formatHttpError("train run", err));
      return;
    }
    await refreshTopicStudioMvp();
    const runId = result?.run_id;
    if (runId) {
      startRunPolling(runId);
    }
    await refreshTopicStudioMvp();
  }

  async function enrichRun() {
    const targetRun = document.getElementById("ts-mvp-segments-run")?.value;
    if (!targetRun) {
      setMvpError("Select a run to enrich.");
      return;
    }
    try {
      await tfFetchJson(`/runs/${targetRun}/enrich`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enricher: "heuristic" }),
      });
    } catch (err) {
      setMvpError(formatHttpError("enrich", err));
      return;
    }
    await refreshTopicStudioMvp();
  }

  function startRunPolling(runId) {
    if (topicStudioMvpState.runPoller) {
      clearInterval(topicStudioMvpState.runPoller);
    }
    topicStudioMvpState.runPoller = setInterval(async () => {
      let run = null;
      try {
        run = await tfFetchJson(`/runs/${runId}`);
      } catch (err) {
        setMvpError(formatHttpError("run poll", err));
        clearInterval(topicStudioMvpState.runPoller);
        topicStudioMvpState.runPoller = null;
        return;
      }
      const idx = topicStudioMvpState.runs.findIndex((r) => r.run_id === runId);
      if (idx >= 0) {
        topicStudioMvpState.runs[idx] = run;
        renderRunsList();
      }
      if (["complete", "failed"].includes((run.status || "").toLowerCase())) {
        clearInterval(topicStudioMvpState.runPoller);
        topicStudioMvpState.runPoller = null;
      }
    }, 2000);
  }

  async function loadSegmentsMvp() {
    const runId = document.getElementById("ts-mvp-segments-run")?.value;
    if (!runId) {
      setMvpError("Select a run to load segments.");
      return;
    }
    const enriched = document.getElementById("ts-mvp-segments-enriched")?.value;
    const params = new URLSearchParams({
      run_id: runId,
      include_snippet: "true",
      include_bounds: "true",
    });
    if (enriched) params.set("has_enrichment", enriched);
    let result = null;
    try {
      result = await tfFetchJson(`/segments?${params.toString()}`);
    } catch (err) {
      table.textContent = "--";
      setMvpError(formatHttpError("segments", err));
      return;
    }
    const table = document.getElementById("ts-mvp-segments-table");
    if (!table) return;
    const items = asItems(result);
    if (!items.length) {
      table.textContent = "(none)";
      return;
    }
    table.innerHTML = `
      <table style="width:100%; border-collapse:collapse;">
        <thead><tr style="text-align:left; color:#888;">
          <th style="padding:4px;">segment_id</th>
          <th style="padding:4px;">rows</th>
          <th style="padding:4px;">label</th>
          <th style="padding:4px;">bounds</th>
          <th style="padding:4px;">snippet</th>
        </tr></thead>
        <tbody>
          ${items
            .map(
              (segment) => `
                <tr data-segment-id="${segment.segment_id}" style="cursor:pointer; border-top:1px solid #222;">
                  <td style="padding:4px; color:#7fd;">${segment.segment_id || "--"}</td>
                  <td style="padding:4px;">${segment.row_ids_count ?? segment.size ?? "--"}</td>
                  <td style="padding:4px;">${segment.label || "--"}</td>
                  <td style="padding:4px;">${segment.bounds || segment.bounds_text || "--"}</td>
                  <td style="padding:4px;">${segment.snippet || "--"}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    `;
    table.querySelectorAll("[data-segment-id]").forEach((row) => {
      row.addEventListener("click", () => {
        const id = row.getAttribute("data-segment-id");
        if (id) loadSegmentDetail(id);
      });
    });
  }

  async function loadSegmentDetail(segmentId) {
    const detailEl = document.getElementById("ts-mvp-segment-detail");
    if (!detailEl) return;
    let seg = null;
    try {
      seg = await tfFetchJson(`/segments/${segmentId}?include_full_text=true`);
    } catch (err) {
      detailEl.textContent = "Failed to load segment detail.";
      setMvpError(formatHttpError("segment detail", err));
      return;
    }
    const hasFullText = Object.prototype.hasOwnProperty.call(seg, "full_text");
    const provenance = seg.provenance ? JSON.stringify(seg.provenance, null, 2) : "--";
    const meaningPayload = {
      title: seg.title || seg.label || null,
      aspects: seg.aspects || null,
      meaning: seg.meaning || null,
      enrichment: seg.enrichment || null,
      sentiment: seg.sentiment || null,
      topic_id: seg.topic_id ?? null,
      is_outlier: seg.is_outlier ?? null,
    };
    const fullTextLabel = hasFullText ? "Full text" : "Full text (error)";
    const fullTextValue = hasFullText ? seg.full_text || "--" : "backend missing include_full_text support";
    detailEl.innerHTML = `
      <div><strong>${seg.segment_id || ""}</strong></div>
      <div style="margin-top:4px;">${seg.text || seg.snippet || "--"}</div>
      <div style="margin-top:6px; color:#777;">Provenance</div>
      <pre style="margin-top:4px; background:#0b0b0b; border:1px solid #333; padding:6px; border-radius:6px; white-space:pre-wrap;">${provenance}</pre>
      <div style="margin-top:6px; color:#777;">Meaning & enrichment</div>
      <pre style="margin-top:4px; background:#0b0b0b; border:1px solid #333; padding:6px; border-radius:6px; white-space:pre-wrap;">${JSON.stringify(meaningPayload, null, 2)}</pre>
      <div style="margin-top:6px; color:#777;">${fullTextLabel}</div>
      <pre style="margin-top:4px; background:#0b0b0b; border:1px solid #333; padding:6px; border-radius:6px; white-space:pre-wrap;">${fullTextValue}</pre>
    `;
  }

  function renderError(target, error, fallback = "Request failed.") {
    if (!target) return;
    if (!error) {
      target.textContent = "--";
      return;
    }
    const status = error.status ? `status ${error.status}` : "status unknown";
    const detail = error.body || error.message || fallback;
    target.textContent = `${status}: ${detail}`;
  }

  function renderTopicStudioError(target, error, label, panel, request = null) {
    const detail = formatHttpError(label.toLowerCase(), error);
    if (target) {
      target.textContent = detail;
    }
    showToast(detail);
    if (HUB_DEBUG) {
      recordTopicStudioDebug(panel, {
        request,
        error: {
          status: error?.status,
          message: error?.message || "",
          body: error?.body || "",
          detail,
        },
      });
    }
  }

  function asItems(value) {
    if (Array.isArray(value)) return value;
    if (value && Array.isArray(value.items)) return value.items;
    if (value && Array.isArray(value.segments)) return value.segments;
    return [];
  }

  function getTotal(data, items) {
    if (data && data.total !== undefined) return Number(data.total);
    if (data && data.count !== undefined) return Number(data.count);
    return Array.isArray(items) ? items.length : 0;
  }

  function truncateText(value, maxLength = 200) {
    if (!value) return "";
    const trimmed = String(value).replace(/\s+/g, " ").trim();
    if (trimmed.length <= maxLength) return trimmed;
    return `${trimmed.slice(0, maxLength)}…`;
  }

  function renderEndpointWarning(target, endpoint, error) {
    if (!target) return;
    if (!error) {
      target.textContent = "";
      target.classList.add("hidden");
      return;
    }
    const status = error.status ? `status ${error.status}` : "status unknown";
    const detail = truncateText(error.body || error.message || "Request failed.");
    target.textContent = `${endpoint} · ${status} · ${detail}`;
    target.classList.remove("hidden");
  }

  function setWarning(target, message) {
    if (!target) return;
    if (!message) {
      target.textContent = "";
      target.classList.add("hidden");
      return;
    }
    target.textContent = message;
    target.classList.remove("hidden");
  }

  function renderDatasetOptions() {
    if (!tsDatasetSelect) return;
    tsDatasetSelect.innerHTML = '<option value="">New dataset…</option>';
    topicStudioDatasets.forEach((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.raw?.dataset_id || dataset.id;
      option.textContent = `${dataset.name} (${dataset.id})`;
      tsDatasetSelect.appendChild(option);
    });
  }

  function getSelectedSavedDataset() {
    const selectedDatasetId = tsDatasetSelect?.value || "";
    if (!selectedDatasetId) return null;
    const dataset = topicStudioDatasets.find(
      (item) => (item.raw?.dataset_id || item.id) === selectedDatasetId
    );
    return dataset?.raw || null;
  }

  function renderModelOptions() {
    if (tsPromoteModelSelect) {
      tsPromoteModelSelect.innerHTML = "";
      topicStudioModels.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.model_id;
        option.textContent = `${model.name}:${model.version} (${model.stage || "stage?"})`;
        tsPromoteModelSelect.appendChild(option);
      });
    }
    if (tsTrainModelSelect) {
      tsTrainModelSelect.innerHTML = "";
      topicStudioModels.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.model_id;
        option.textContent = `${model.name}:${model.version}`;
        tsTrainModelSelect.appendChild(option);
      });
    }
  }

  function formatRunLabel(run) {
    const createdAt = run.created_at || run.started_at || "--";
    const modelName = run.model?.name || run.model_name || "--";
    const modelVersion = run.model?.version || run.model_version || "";
    const stage = run.stage || run.model?.stage || run.model_stage || "--";
    const windowStart = run.window?.start_at || run.window_start || run.start_at || "--";
    const windowEnd = run.window?.end_at || run.window_end || run.end_at || "--";
    const stats = run.stats_summary || run.stats || {};
    const docs = stats.docs_generated ?? stats.doc_count;
    const segments = stats.segments_generated ?? stats.segment_count;
    const counts = docs || segments ? `${docs ?? "--"} docs · ${segments ?? "--"} segs` : "--";
    const status = run.status || "--";
    return `${createdAt} · ${modelName} ${modelVersion}`.trim() + ` · ${status}/${stage} · ${windowStart} → ${windowEnd} · ${counts}`;
  }

  function normalizeRunsResponse(response) {
    if (!response) return [];
    const items = asItems(response);
    if (items.length > 0) return items;
    if (Array.isArray(response.runs)) return response.runs;
    return [];
  }

  function renderRunsSelect() {
    if (!tsRunsSelect) return;
    tsRunsSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select a run…";
    tsRunsSelect.appendChild(placeholder);
    topicStudioRuns.forEach((run) => {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = formatRunLabel(run);
      tsRunsSelect.appendChild(option);
    });
    const stored = localStorage.getItem(TOPIC_STUDIO_RUN_ID_KEY);
    const storedMatch = topicStudioRuns.find((run) => run.run_id === stored);
    if (storedMatch) {
      tsRunsSelect.value = storedMatch.run_id;
      return;
    }
    const completed = topicStudioRuns
      .filter((run) => run.status === "complete")
      .sort((a, b) => new Date(b.created_at || b.started_at || 0) - new Date(a.created_at || a.started_at || 0));
    if (completed.length > 0) {
      tsRunsSelect.value = completed[0].run_id;
      return;
    }
    if (topicStudioRuns.length > 0) {
      tsRunsSelect.value = topicStudioRuns[0].run_id;
    }
  }

  function renderCompareRunOptions() {
    const selects = [tsCompareLeftRunId, tsCompareRightRunId];
    const current = selects.map((select) => select?.value || "");
    selects.forEach((select, index) => {
      if (!select) return;
      const placeholder = index === 0 ? "Left run…" : "Right run…";
      select.innerHTML = `<option value="">${placeholder}</option>`;
      topicStudioRuns.forEach((run) => {
        const option = document.createElement("option");
        option.value = run.run_id;
        option.textContent = formatRunLabel(run);
        select.appendChild(option);
      });
      if (current[index]) {
        select.value = current[index];
      }
    });
    if (topicStudioRuns.length > 0) {
      if (tsCompareLeftRunId && !tsCompareLeftRunId.value) {
        tsCompareLeftRunId.value = topicStudioRuns[0].run_id;
      }
      if (tsCompareRightRunId && !tsCompareRightRunId.value) {
        tsCompareRightRunId.value = topicStudioRuns[1]?.run_id || "";
      }
    }
  }

  function renderConversationDatasetOptions() {
    if (!tsConvoDatasetSelect) return;
    const current = tsConvoDatasetSelect.value;
    tsConvoDatasetSelect.innerHTML = '<option value="">Select dataset…</option>';
    topicStudioDatasets.forEach((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = `${dataset.name} (${dataset.id})`;
      tsConvoDatasetSelect.appendChild(option);
    });
    if (current) {
      tsConvoDatasetSelect.value = current;
    }
    if (!tsConvoDatasetSelect.value) {
      if (tsDatasetSelect?.value) {
        tsConvoDatasetSelect.value = tsDatasetSelect.value;
      } else if (topicStudioDatasets.length > 0) {
        tsConvoDatasetSelect.value = topicStudioDatasets[0].id;
      }
    }
  }

  function renderConversationList(conversations) {
    if (!tsConvoList) return;
    tsConvoList.innerHTML = "";
    if (!conversations || conversations.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="5">No conversations found.</td>';
      tsConvoList.appendChild(row);
      return;
    }
    conversations.forEach((convo) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      const start = convo.observed_start_at || "--";
      const end = convo.observed_end_at || "--";
      const windowText = start === "--" && end === "--" ? "--" : `${start} → ${end}`;
      const snippet = convo.text_snippet || convo.snippet || "--";
      const checked = topicStudioConversationSelection.has(convo.conversation_id);
      row.innerHTML = `
        <td class="py-2 pr-2"><input type="checkbox" data-convo-select="${convo.conversation_id}" ${checked ? "checked" : ""} /></td>
        <td class="py-2 pr-2">${windowText}</td>
        <td class="py-2 pr-2">${convo.block_count ?? "--"}</td>
        <td class="py-2 pr-2">${snippet}</td>
        <td class="py-2 pr-2">
          <span class="text-indigo-300 cursor-pointer" data-convo-id="${convo.conversation_id}">${convo.conversation_id}</span>
          <button class="ml-2 text-[10px] text-gray-400 hover:text-gray-200" data-convo-copy="${convo.conversation_id}">Copy</button>
        </td>
      `;
      row.querySelector(`[data-convo-select="${convo.conversation_id}"]`)?.addEventListener("change", (event) => {
        const checkedNow = event.target.checked;
        if (checkedNow) {
          topicStudioConversationSelection.add(convo.conversation_id);
        } else {
          topicStudioConversationSelection.delete(convo.conversation_id);
        }
      });
      row.querySelector(`[data-convo-id="${convo.conversation_id}"]`)?.addEventListener("click", () => {
        loadConversationDetail(convo.conversation_id);
      });
      row.querySelector(`[data-convo-copy="${convo.conversation_id}"]`)?.addEventListener("click", () => {
        copyText(convo.conversation_id, "Conversation id copied.");
      });
      tsConvoList.appendChild(row);
    });
  }

  function renderConversationDetail(conversation) {
    if (!tsConvoDetail) return;
    if (!conversation) {
      tsConvoDetail.textContent = "Select a conversation to view blocks.";
      return;
    }
    const blocks = conversation.blocks || [];
    if (!blocks.length) {
      tsConvoDetail.textContent = "No blocks found.";
      return;
    }
    const container = document.createElement("div");
    container.className = "space-y-2";
    blocks.forEach((block, idx) => {
      const blockWrap = document.createElement("div");
      blockWrap.className = "bg-gray-950/60 border border-gray-800 rounded p-2 space-y-1";
      const title = document.createElement("div");
      title.className = "text-[10px] text-gray-500";
      const times = Array.isArray(block.timestamps) ? block.timestamps : [];
      const range = times.length ? `${times[0]} → ${times[times.length - 1]}` : "--";
      title.textContent = `Block ${block.block_index} · ${range}`;
      const snippet = document.createElement("div");
      snippet.textContent = block.text_snippet || "--";
      blockWrap.appendChild(title);
      blockWrap.appendChild(snippet);
      if (idx < blocks.length - 1) {
        const splitBtn = document.createElement("button");
        splitBtn.className = "text-[10px] text-indigo-300 hover:text-indigo-200";
        splitBtn.textContent = `Split after block ${block.block_index}`;
        splitBtn.addEventListener("click", () => {
          splitConversation(conversation.conversation_id, block.block_index);
        });
        blockWrap.appendChild(splitBtn);
      }
      container.appendChild(blockWrap);
    });
    tsConvoDetail.innerHTML = "";
    tsConvoDetail.appendChild(container);
  }

  function renderConversationOverrides(overrides) {
    if (!tsConvoOverrides) return;
    if (!overrides || overrides.length === 0) {
      tsConvoOverrides.textContent = "--";
      return;
    }
    tsConvoOverrides.innerHTML = "";
    overrides.forEach((override) => {
      const row = document.createElement("div");
      row.className = "border-b border-gray-800 py-1";
      row.textContent = `${override.kind} · ${override.reason || "no reason"} · ${override.created_at}`;
      tsConvoOverrides.appendChild(row);
    });
  }

  function setSelectedRun(runId) {
    if (!runId) return;
    if (tsRunId) tsRunId.value = runId;
    if (tsSegmentsRunId) tsSegmentsRunId.value = runId;
    if (tsTopicsRunId && !tsTopicsRunId.value) tsTopicsRunId.value = runId;
    localStorage.setItem(TOPIC_STUDIO_RUN_ID_KEY, runId);
    topicStudioLastSubview = "runs";
    saveTopicStudioState();
  }

  function resetSegmentsPaging() {
    topicStudioSegmentsOffset = 0;
    topicStudioSegmentsTotal = null;
  }

  function updateSegmentsRange() {
    if (!tsSegmentsRange) return;
    const total = topicStudioSegmentsTotal;
    const start = topicStudioSegmentsPage.length === 0 ? 0 : topicStudioSegmentsOffset + 1;
    const end = topicStudioSegmentsOffset + topicStudioSegmentsPage.length;
    if (total !== null && total !== undefined) {
      tsSegmentsRange.textContent = `Showing ${start}–${end} of ${total}`;
    } else {
      tsSegmentsRange.textContent = `Showing ${start}–${end}`;
    }
    if (tsSegmentsPrev) tsSegmentsPrev.disabled = topicStudioSegmentsOffset <= 0;
    if (tsSegmentsNext) {
      if (total !== null && total !== undefined) {
        tsSegmentsNext.disabled = topicStudioSegmentsOffset + topicStudioSegmentsLimit >= total;
      } else {
        tsSegmentsNext.disabled = topicStudioSegmentsPage.length < topicStudioSegmentsLimit;
      }
    }
  }

  function exportSegmentsCsv() {
    const exportRows = topicStudioSegmentsDisplayed.length ? topicStudioSegmentsDisplayed : topicStudioSegmentsPage;
    if (!exportRows || exportRows.length === 0) {
      showToast("No segments to export.");
      return;
    }
    const headers = ["segment_id", "rows", "start_at", "end_at", "row_ids_count", "title", "aspects", "valence", "friction", "snippet"];
    const lines = [headers.join(",")];
    const escapeCsv = (value) => {
      if (value === null || value === undefined) return "";
      const str = String(value).replace(/"/g, "\"\"");
      return `"${str}"`;
    };
    exportRows.forEach((segment) => {
      const sentiment = segment.sentiment || {};
      const aspects = Array.isArray(segment.aspects) ? segment.aspects.join("|") : "";
      const rowIdsCount = segment.row_ids_count ?? segment.size ?? "";
      const row = [
        escapeCsv(segment.segment_id),
        escapeCsv(rowIdsCount),
        escapeCsv(segment.start_at ?? ""),
        escapeCsv(segment.end_at ?? ""),
        escapeCsv(segment.row_ids_count ?? ""),
        escapeCsv(segment.title || segment.label || ""),
        escapeCsv(aspects),
        escapeCsv(sentiment.valence ?? ""),
        escapeCsv(sentiment.friction ?? ""),
        escapeCsv(segment.snippet || ""),
      ];
      lines.push(row.join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `topic_foundry_segments_${tsSegmentsRunId?.value || "run"}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function applySegmentsClientFilters(segments) {
    let filtered = Array.isArray(segments) ? [...segments] : [];
    if (topicStudioSegmentsFacetFilter) {
      const { type, value } = topicStudioSegmentsFacetFilter;
      if (type === "intent") {
        filtered = filtered.filter((segment) => segment.intent === value || segment.meaning?.intent === value);
      }
      if (type === "friction") {
        filtered = filtered.filter((segment) => {
          const friction = Number(segment.sentiment?.friction ?? segment.friction ?? 0);
          if (Number.isNaN(friction)) return false;
          if (value === "0-0.3") return friction <= 0.3;
          if (value === "0.3-0.7") return friction > 0.3 && friction <= 0.7;
          if (value === "0.7-1.0") return friction > 0.7;
          return false;
        });
      }
    }
    return filtered;
  }

  function renderSegmentsFacets(facets) {
    if (!tsSegmentsFacets) return;
    tsSegmentsFacets.innerHTML = "";
    if (!facets) {
      const note = document.createElement("div");
      note.className = "text-[10px] text-gray-500";
      note.textContent = "Facets unavailable";
      tsSegmentsFacets.appendChild(note);
      return;
    }
    if (facets.ok === false) {
      const note = document.createElement("div");
      note.className = "text-[10px] text-gray-500";
      note.textContent = "Facets unavailable";
      tsSegmentsFacets.appendChild(note);
      return;
    }
    const makeChip = (label, type, value) => {
      const button = document.createElement("button");
      button.className = "bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-full px-2 py-0.5 border border-gray-700 text-[10px]";
      button.textContent = label;
      button.addEventListener("click", () => {
        if (type === "aspect") {
          if (tsSegmentsAspect) tsSegmentsAspect.value = value;
          topicStudioSegmentsFacetFilter = null;
          resetSegmentsPaging();
          loadSegments();
          refreshSegmentFacets();
          return;
        }
        topicStudioSegmentsFacetFilter = { type, value };
        const filtered = applySegmentsClientFilters(topicStudioSegmentsPage);
        topicStudioSegmentsDisplayed = filtered;
        renderSegmentsTable(filtered);
        renderSegmentsFacets(topicStudioSegmentsLastFacets);
      });
      tsSegmentsFacets.appendChild(button);
    };
    const clearButton = document.createElement("button");
    clearButton.className = "bg-gray-900/60 hover:bg-gray-800 text-gray-200 rounded-full px-2 py-0.5 border border-gray-700 text-[10px]";
    clearButton.textContent = "Clear filters";
    clearButton.addEventListener("click", () => {
      topicStudioSegmentsFacetFilter = null;
      if (tsSegmentsAspect) tsSegmentsAspect.value = "";
      resetSegmentsPaging();
      loadSegments();
      refreshSegmentFacets();
    });
    tsSegmentsFacets.appendChild(clearButton);
    (facets.aspects || []).slice(0, 6).forEach(({ key, count }) => makeChip(`Aspect: ${key} (${count})`, "aspect", key));
    (facets.intents || []).slice(0, 6).forEach(({ key, count }) => makeChip(`Intent: ${key} (${count})`, "intent", key));
    (facets.friction_buckets || []).forEach(({ key, count }) => makeChip(`Friction ${key} (${count})`, "friction", key));
  }

  async function loadConversations() {
    if (!tsConvoDatasetSelect?.value) {
      showToast("Select a dataset to load conversations.");
      return;
    }
    try {
      setLoading(tsConvoLoading, true);
      if (tsConvoError) tsConvoError.textContent = "--";
      if (tsConvoMergeStatus) tsConvoMergeStatus.textContent = "--";
      topicStudioConversationSelection = new Set();
      const params = new URLSearchParams({ dataset_id: tsConvoDatasetSelect.value });
      const startAt = parseDateInput(tsConvoStartAt?.value);
      const endAt = parseDateInput(tsConvoEndAt?.value);
      const limit = Number(tsConvoLimit?.value || 200);
      if (startAt) params.set("start_at", startAt);
      if (endAt) params.set("end_at", endAt);
      params.set("limit", String(limit));
      const response = await topicFoundryFetch(`/conversations?${params.toString()}`);
      const items = response.items || response.conversations || response;
      topicStudioConversations = Array.isArray(items) ? items : [];
      renderConversationList(topicStudioConversations);
      setLoading(tsConvoLoading, false);
      await refreshConversationOverrides();
      saveTopicStudioState();
    } catch (err) {
      renderError(tsConvoError, err, "Failed to load conversations.");
      setLoading(tsConvoLoading, false);
    }
  }

  async function loadConversationDetail(conversationId) {
    if (!conversationId) return;
    try {
      topicStudioSelectedConversationId = conversationId;
      const detail = await topicFoundryFetch(`/conversations/${conversationId}`);
      renderConversationDetail(detail);
    } catch (err) {
      renderError(tsConvoError, err, "Failed to load conversation detail.");
    }
  }

  async function splitConversation(conversationId, splitIndex) {
    if (!conversationId || splitIndex === null || splitIndex === undefined) return;
    if (!tsConvoDatasetSelect?.value) {
      showToast("Select a dataset before splitting.");
      return;
    }
    try {
      await topicFoundryFetch(`/conversations/${conversationId}/split`, {
        method: "POST",
        body: JSON.stringify({
          dataset_id: tsConvoDatasetSelect.value,
          split_at_block_index: splitIndex,
          reason: tsConvoMergeReason?.value?.trim() || null,
        }),
      });
      showToast("Conversation split.");
      await loadConversations();
    } catch (err) {
      renderError(tsConvoError, err, "Failed to split conversation.");
    }
  }

  async function mergeConversations() {
    if (!tsConvoDatasetSelect?.value) {
      showToast("Select a dataset before merging.");
      return;
    }
    const ids = Array.from(topicStudioConversationSelection);
    if (ids.length < 2) {
      showToast("Select at least two conversations to merge.");
      return;
    }
    try {
      const result = await topicFoundryFetch("/conversations/merge", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: tsConvoDatasetSelect.value,
          conversation_ids: ids,
          reason: tsConvoMergeReason?.value?.trim() || null,
        }),
      });
      topicStudioConversationSelection = new Set();
      if (tsConvoMergeStatus) {
        tsConvoMergeStatus.textContent = `Merged. New conversation: ${result.new_conversation_id || "--"}`;
      }
      await loadConversations();
    } catch (err) {
      renderError(tsConvoError, err, "Failed to merge conversations.");
    }
  }

  async function refreshConversationOverrides() {
    if (!tsConvoDatasetSelect?.value) return;
    try {
      const overrides = await topicFoundryFetch(`/conversations/overrides?dataset_id=${tsConvoDatasetSelect.value}`);
      const items = overrides.items || overrides.overrides || overrides;
      renderConversationOverrides(Array.isArray(items) ? items : []);
    } catch (err) {
      console.warn("[TopicStudio] Failed to load conversation overrides", err);
    }
  }

  async function populateDatasetForm(dataset) {
    if (!dataset) return;
    if (tsDatasetName) tsDatasetName.value = dataset.name || "";
    if (tsDatasetTable) tsDatasetTable.value = dataset.source_table || "";
    const sourceTable = dataset.source_table || "";
    const [schemaPart, tablePart] = sourceTable.includes(".") ? sourceTable.split(".", 2) : ["", sourceTable];
    if (tsDatasetSchema) {
      if (!topicStudioIntrospectionSchemas.includes(schemaPart)) {
        renderSchemaOptionsForDataset();
      }
      tsDatasetSchema.value = schemaPart || tsDatasetSchema.value || "";
    }
    if (tsDatasetSchema?.value) {
      await loadDatasetTables(tsDatasetSchema.value);
    }
    if (tsDatasetTableSelect) {
      tsDatasetTableSelect.value = tablePart || "";
    }
    updateSourceTableInput();
    if (tsDatasetSchema?.value && tsDatasetTableSelect?.value) {
      await loadDatasetColumnsFromSourceTable(`${tsDatasetSchema.value}.${tsDatasetTableSelect.value}`);
    } else if (sourceTable) {
      await loadDatasetColumnsFromSourceTable(sourceTable);
    }
    if (tsDatasetIdColumn) tsDatasetIdColumn.value = dataset.id_column || "";
    if (tsDatasetTimeColumn) tsDatasetTimeColumn.value = dataset.time_column || "";
    if (tsDatasetTextColumns) setSelectedValues(tsDatasetTextColumns, dataset.text_columns || []);
    if (tsDatasetBoundaryColumn) tsDatasetBoundaryColumn.value = dataset.boundary_column || "";
    if (tsDatasetWhereSql) tsDatasetWhereSql.value = dataset.where_sql || "";
    if (tsDatasetTimezone) tsDatasetTimezone.value = dataset.timezone || "UTC";
    setDatasetSaveStatus("--");
  }

  function clearPreview() {
    if (tsPreviewDocs) tsPreviewDocs.textContent = "--";
    if (tsPreviewSegments) tsPreviewSegments.textContent = "--";
    if (tsPreviewAvgChars) tsPreviewAvgChars.textContent = "--";
    if (tsPreviewP95Chars) tsPreviewP95Chars.textContent = "--";
    if (tsPreviewMaxChars) tsPreviewMaxChars.textContent = "--";
    if (tsPreviewObserved) tsPreviewObserved.textContent = "--";
    if (tsPreviewSamples) tsPreviewSamples.textContent = "--";
    if (tsPreviewError) tsPreviewError.textContent = "--";
    setWarning(tsPreviewWarning, null);
    topicStudioLastPreview = null;
    if (tsUsePreviewSpec) tsUsePreviewSpec.disabled = true;
  }

  function renderPreviewSamples(samples) {
    if (!tsPreviewSamples) return;
    tsPreviewSamples.innerHTML = "";
    if (!samples || samples.length === 0) {
      tsPreviewSamples.textContent = "No samples returned.";
      return;
    }
    samples.slice(0, 5).forEach((sample) => {
      const wrapper = document.createElement("div");
      wrapper.className = "bg-gray-950/60 border border-gray-800 rounded p-2";
      const header = document.createElement("div");
      header.className = "text-[10px] text-gray-500 mb-1";
      header.textContent = `Segment ${sample.segment_id} · ${sample.chars} chars`;
      const snippet = document.createElement("div");
      snippet.textContent = sample.snippet || "--";
      wrapper.appendChild(header);
      wrapper.appendChild(snippet);
      tsPreviewSamples.appendChild(wrapper);
    });
  }

  async function executePreview(payload) {
    const result = await topicFoundryFetch("/datasets/preview", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    topicStudioLastPreview = payload;
    if (tsUsePreviewSpec) {
      tsUsePreviewSpec.disabled = false;
    }
    if (tsPreviewDocs) tsPreviewDocs.textContent = result.docs_generated ?? "--";
    if (tsPreviewSegments) tsPreviewSegments.textContent = result.segments_generated ?? "--";
    if (tsPreviewAvgChars) tsPreviewAvgChars.textContent = result.avg_chars ?? "--";
    if (tsPreviewP95Chars) tsPreviewP95Chars.textContent = result.p95_chars ?? "--";
    if (tsPreviewMaxChars) tsPreviewMaxChars.textContent = result.max_chars ?? "--";
    if (tsPreviewObserved) {
      const start = result.observed_start_at || "--";
      const end = result.observed_end_at || "--";
      tsPreviewObserved.textContent = `${start} → ${end}`;
    }
    renderPreviewSamples(result.samples || []);
    const docsGenerated = Number(result.docs_generated);
    if (!Number.isNaN(docsGenerated) && docsGenerated < MIN_PREVIEW_DOCS) {
      setWarning(tsPreviewWarning, "Low document count. Widen the date range or adjust windowing for more docs.");
    } else {
      setWarning(tsPreviewWarning, null);
    }
    saveTopicStudioState();
    if (tsPreviewError) tsPreviewError.textContent = "--";
    setLoading(tsPreviewLoading, false);
    return result;
  }

  function formatRunStats(run) {
    const stats = run?.stats || {};
    const docs = stats.docs_generated ?? "--";
    const segments = stats.segments_generated ?? "--";
    const clusters = stats.cluster_count ?? "--";
    const outlierPct = stats.outlier_pct ?? "--";
    const enriched = stats.segments_enriched ?? "--";
    return { docs, segments, clusters, outlierPct, enriched };
  }

  function renderRunStatus(run) {
    if (!run) return;
    const stats = formatRunStats(run);
    if (tsRunState) tsRunState.textContent = run.status || "--";
    if (tsRunClusters) tsRunClusters.textContent = stats.clusters;
    if (tsRunDocs) tsRunDocs.textContent = stats.docs;
    if (tsRunSegments) tsRunSegments.textContent = stats.segments;
    if (tsRunOutliers) tsRunOutliers.textContent = stats.outlierPct;
    if (tsRunEnriched) tsRunEnriched.textContent = stats.enriched;
    if (tsRunArtifacts) tsRunArtifacts.textContent = JSON.stringify(run.artifact_paths || {}, null, 2);
    if (tsRunError) tsRunError.textContent = run.error || "--";
    const outlierPct = Number(stats.outlierPct);
    const clusters = Number(stats.clusters);
    if (!Number.isNaN(outlierPct) && outlierPct > 0.8) {
      setWarning(tsRunWarning, "High outlier rate detected. Consider adjusting windowing or model parameters.");
    } else if (!Number.isNaN(clusters) && clusters <= 1) {
      setWarning(tsRunWarning, "Low cluster count detected. Consider increasing min_cluster_size or widening date range.");
    } else {
      setWarning(tsRunWarning, null);
    }
  }

  function stopRunPolling() {
    if (topicStudioRunPoller) {
      clearInterval(topicStudioRunPoller);
      topicStudioRunPoller = null;
    }
    if (topicStudioEnrichPolling) {
      topicStudioEnrichPolling = false;
      setLoading(tsEnrichLoading, false);
    }
  }

  function startRunPolling(runId) {
    if (!runId) return;
    stopRunPolling();
    setLoading(tsRunLoading, true);
    topicStudioRunPoller = setInterval(async () => {
      try {
        const run = await topicFoundryFetch(`/runs/${runId}`);
        renderRunStatus(run);
        if (run.status === "complete" || run.status === "failed") {
          stopRunPolling();
          setLoading(tsRunLoading, false);
          await refreshTopicStudio();
        }
      } catch (err) {
        stopRunPolling();
        setLoading(tsRunLoading, false);
        renderError(tsRunError, err);
      }
    }, 2000);
  }

  function updateStatusBasedOnState() {
    if (orionState === 'idle') updateStatus('Ready.');
    else if (orionState === 'speaking') updateStatus('Speaking...');
    else if (orionState === 'processing') updateStatus('Processing...');
  }

  function updateMemoryPanelFromResponse(data) {
    if (!memoryUsedValue || !recallCountValue || !backendCountsValue || !memoryDigestPre) return;
    const recallDebug = data && typeof data.recall_debug === 'object' ? data.recall_debug : null;
    const recallCount = recallDebug && typeof recallDebug.count === 'number' ? recallDebug.count : null;
    const backendCounts = recallDebug && recallDebug.backend_counts
      ? recallDebug.backend_counts
      : (recallDebug && recallDebug.debug && recallDebug.debug.backend_counts) || null;
    const memoryUsed = typeof data.memory_used === 'boolean'
      ? data.memory_used
      : (typeof recallCount === 'number' ? recallCount > 0 : false);
    const memoryDigest = data.memory_digest || (recallDebug && recallDebug.memory_digest) || "";

    memoryUsedValue.textContent = memoryUsed ? "true" : "false";
    recallCountValue.textContent = typeof recallCount === 'number' ? recallCount : "--";
    backendCountsValue.textContent = backendCounts ? JSON.stringify(backendCounts, null, 2) : "--";
    memoryDigestPre.textContent = memoryDigest || "--";
  }

  function toggleMemoryPanel() {
    if (!memoryPanelBody) return;
    memoryPanelBody.classList.toggle('hidden');
  }

  function isAttentionNotification(notification) {
    return notification && notification.event_kind === ATTENTION_EVENT_KIND;
  }

  function normalizeAttentionItem(notification) {
    if (!notification) return null;
    return {
      attention_id: notification.attention_id,
      created_at: notification.created_at,
      severity: notification.severity || 'info',
      title: notification.title || 'Attention',
      message: notification.body_text || '',
      source_service: notification.source_service || 'unknown',
    };
  }

  function upsertPendingAttention(item) {
    if (!item || !item.attention_id) return;
    const idx = pendingAttention.findIndex((a) => a.attention_id === item.attention_id);
    if (idx >= 0) {
      pendingAttention[idx] = { ...pendingAttention[idx], ...item };
    } else {
      pendingAttention.unshift(item);
    }
    renderPendingAttention();
  }

  function removePendingAttention(attentionId) {
    if (!attentionId) return;
    pendingAttention = pendingAttention.filter((item) => item.attention_id !== attentionId);
    renderPendingAttention();
  }

  function renderPendingAttention() {
    if (!attentionList || !attentionCount) return;
    attentionCount.textContent = String(pendingAttention.length);
    attentionList.innerHTML = '';
    if (pendingAttention.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-500 italic';
      empty.textContent = 'No pending attention';
      attentionList.appendChild(empty);
      return;
    }
    pendingAttention.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'bg-gray-900/60 border border-gray-700 rounded-lg p-2 space-y-2';

      const header = document.createElement('div');
      header.className = 'flex items-center justify-between gap-2';

      const title = document.createElement('div');
      title.className = 'text-gray-100 font-semibold text-xs';
      title.textContent = item.title || 'Attention';

      const badge = document.createElement('span');
      badge.className = `text-[10px] px-2 py-0.5 rounded-full uppercase ${severityBadgeClass(item.severity)}`;
      badge.textContent = (item.severity || 'info').toUpperCase();

      header.appendChild(title);
      header.appendChild(badge);

      const body = document.createElement('div');
      body.className = 'text-[11px] text-gray-300 whitespace-pre-wrap';
      body.textContent = item.message || '';

      const actions = document.createElement('div');
      actions.className = 'flex items-center gap-2 text-[10px]';

      const openBtn = document.createElement('button');
      openBtn.className = 'px-2 py-1 rounded bg-indigo-600/80 hover:bg-indigo-500 text-white';
      openBtn.textContent = 'Open chat';
      openBtn.addEventListener('click', () => focusChatInput());

      const dismissBtn = document.createElement('button');
      dismissBtn.className = 'px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
      dismissBtn.textContent = 'Dismiss';
      dismissBtn.addEventListener('click', () => handleAttentionAck(item.attention_id, 'dismissed'));

      const snoozeBtn = document.createElement('button');
      snoozeBtn.className = 'px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
      snoozeBtn.textContent = 'Snooze 30m';
      snoozeBtn.addEventListener('click', () =>
        handleAttentionAck(item.attention_id, 'snooze', 'Snoozed for 30 minutes')
      );

      actions.appendChild(openBtn);
      actions.appendChild(dismissBtn);
      actions.appendChild(snoozeBtn);

      card.appendChild(header);
      card.appendChild(body);
      card.appendChild(actions);
      attentionList.appendChild(card);
    });
  }

  function isChatMessageNotification(notification) {
    return (
      notification &&
      (notification.notification_type === 'chat_message' || notification.event_kind === CHAT_MESSAGE_EVENT_KIND)
    );
  }

  function normalizeChatMessage(notification) {
    if (!notification) return null;
    const msgId = notification.message_id || notification.messageId;
    if (!msgId) return null;
    return {
      message_id: msgId,
      session_id: notification.session_id || notification.sessionId,
      created_at: notification.created_at || notification.createdAt,
      severity: notification.severity || 'info',
      title: notification.title || 'New message from Orion',
      preview_text: notification.body_text || notification.preview_text || notification.previewText || '',
      status: (notification.status || 'unread').toLowerCase(),
      silent: Boolean(notification.silent),
    };
  }

  function upsertChatMessage(item) {
    if (!item || !item.message_id) return;
    // Never re-insert locally dismissed messages (backend may still report them as unread)
    if (isDismissed(item.message_id)) return;
    const idx = chatMessages.findIndex((m) => m.message_id === item.message_id);
    if (idx >= 0) {
      chatMessages[idx] = { ...chatMessages[idx], ...item };
    } else {
      chatMessages.unshift(item);
    }
    renderChatMessages();
  }

  function updateChatMessageStatus(messageId, status) {
    const idx = chatMessages.findIndex((m) => m.message_id === messageId);
    if (idx >= 0) {
      chatMessages[idx].status = status;
      renderChatMessages();
    }
  }

  function renderChatMessages() {
    if (!messageList) return;
    const filter = messageFilter ? messageFilter.value : 'unread';
    // Filter out dismissed regardless of unread/all
    const visible = chatMessages.filter((m) => !isDismissed(m.message_id));
    const filtered =
      filter === 'all' ? visible : visible.filter((m) => (m.status || '').toLowerCase() === 'unread');

    messageList.innerHTML = '';
    if (filtered.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-500 italic';
      empty.textContent = 'No messages';
      messageList.appendChild(empty);
      return;
    }

    const grouped = filtered.reduce((acc, item) => {
      const key = item.session_id || 'unknown';
      acc[key] = acc[key] || [];
      acc[key].push(item);
      return acc;
    }, {});

    Object.entries(grouped).forEach(([sessionId, items]) => {
      const groupHeader = document.createElement('div');
      groupHeader.className = 'text-[10px] text-gray-500 uppercase tracking-wide';
      groupHeader.textContent = `Session ${sessionId} (${items.length})`;
      messageList.appendChild(groupHeader);

      items.forEach((item) => {
        const details = document.createElement('details');
        details.className = 'bg-gray-900/60 border border-gray-700 rounded-lg p-2';

        const summary = document.createElement('summary');
        summary.className = 'cursor-pointer list-none space-y-1';

        const headerRow = document.createElement('div');
        headerRow.className = 'flex items-center justify-between gap-2';

        const title = document.createElement('div');
        title.className = 'text-gray-100 font-semibold text-xs';
        title.textContent = item.title || 'Message';

        const meta = document.createElement('div');
        meta.className = 'text-[10px] text-gray-400';
        meta.textContent = item.created_at ? new Date(item.created_at).toLocaleString() : '--';

        headerRow.appendChild(title);
        headerRow.appendChild(meta);

        const preview = document.createElement('div');
        preview.className = 'text-[11px] text-gray-300 line-clamp-2';
        preview.textContent = item.preview_text || '';

        summary.appendChild(headerRow);
        summary.appendChild(preview);

        const body = document.createElement('div');
        body.className = 'mt-2 space-y-2';

        const bodyText = document.createElement('div');
        bodyText.className = 'text-[11px] text-gray-300 whitespace-pre-wrap';
        bodyText.textContent = item.preview_text || '';

        const actions = document.createElement('div');
        actions.className = 'flex items-center gap-2 text-[10px]';

        const openBtn = document.createElement('button');
        openBtn.className = 'px-2 py-1 rounded bg-indigo-600/80 hover:bg-indigo-500 text-white';
        openBtn.textContent = 'Open chat';
        openBtn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          setSessionId(item.session_id);
          focusChatInput();
        });

        const dismissBtn = document.createElement('button');
        dismissBtn.className = 'px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
        dismissBtn.textContent = 'Dismiss';
        dismissBtn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (item && item.message_id) {
            // Make dismiss visibly do something even if backend doesn't change message status yet
            markDismissed(item.message_id);
            chatMessages = chatMessages.filter((m) => m.message_id !== item.message_id);
            renderChatMessages();
          }
          if (!item._synthetic_id) handleChatMessageReceipt(item.message_id, item.session_id, 'dismissed');
        });

        actions.appendChild(openBtn);
        actions.appendChild(dismissBtn);

        body.appendChild(bodyText);
        body.appendChild(actions);

        details.appendChild(summary);
        details.appendChild(body);
        details.addEventListener('toggle', () => {
          if (!details.open) return;
          if (!item || !item.message_id || item._synthetic_id) return;
          if (openedMessageIds.has(item.message_id)) return;
          openedMessageIds.add(item.message_id);
          handleChatMessageReceipt(item.message_id, item.session_id, 'opened');
        });

        messageList.appendChild(details);
      });
    });
  }

  function addNotification(notification) {
    notifications.unshift(notification);
    if (notifications.length > NOTIFICATION_MAX) notifications = notifications.slice(0, NOTIFICATION_MAX);
    renderNotifications();
    if (isAttentionNotification(notification)) {
      const item = normalizeAttentionItem(notification);
      upsertPendingAttention(item);
    }
    if (isChatMessageNotification(notification)) {
      const item = normalizeChatMessage(notification);
      upsertChatMessage(item);
      if (item && !seenMessageIds.has(item.message_id)) {
        seenMessageIds.add(item.message_id);
        handleChatMessageReceipt(item.message_id, item.session_id, 'seen');
      }
    }
    showToast(notification);
  }

  function notificationIdentity(notification) {
    if (notification && notification.event_id) return `event:${notification.event_id}`;
    const createdAt = notification?.created_at || '';
    const title = notification?.title || '';
    const body = notification?.body_text || '';
    return `fallback:${createdAt}|${title}|${body}`;
  }

  function renderNotifications() {
    if (!notificationList) return;
    const filter = notificationFilter ? notificationFilter.value : 'all';
    notificationList.innerHTML = '';
    const filtered = notifications.filter((n) => filter === 'all' || (n.severity || '').toLowerCase() === filter);

    if (filtered.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-500 italic';
      empty.textContent = 'No notifications';
      notificationList.appendChild(empty);
      return;
    }

    filtered.forEach((n) => {
      const item = document.createElement('div');
      item.className = 'bg-gray-900/60 border border-gray-700 rounded-lg p-2 space-y-2';

      const header = document.createElement('div');
      header.className = 'flex items-center justify-between gap-2';

      const title = document.createElement('div');
      title.className = 'text-gray-100 font-semibold text-xs';
      title.textContent = n.title || 'Notification';

      const headerActions = document.createElement('div');
      headerActions.className = 'flex items-center gap-2';

      const badge = document.createElement('span');
      badge.className = `text-[10px] px-2 py-0.5 rounded-full uppercase ${severityBadgeClass(n.severity)}`;
      badge.textContent = (n.severity || 'info').toUpperCase();

      const dismissBtn = document.createElement('button');
      dismissBtn.className = 'text-[10px] px-2 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
      dismissBtn.textContent = 'Dismiss';
      dismissBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const identity = notificationIdentity(n);
        notifications = notifications.filter((item) => notificationIdentity(item) !== identity);
        renderNotifications();
        if (n && n.message_id && n.session_id) {
          handleChatMessageReceipt(n.message_id, n.session_id, 'dismissed');
        }
      });

      headerActions.appendChild(badge);
      headerActions.appendChild(dismissBtn);

      header.appendChild(title);
      header.appendChild(headerActions);

      const meta = document.createElement('div');
      meta.className = 'text-[10px] text-gray-400';
      const createdAt = n.created_at ? new Date(n.created_at).toLocaleString() : '--';
      meta.textContent = `${createdAt} • ${n.event_kind || 'event'} • ${n.source_service || 'unknown'}`;

      const fullText = String(n.full_text || n.body_md || n.body_text || n.preview_text || '').trim();
      const bodyTextEl = document.createElement('div');
      bodyTextEl.className = 'text-[11px] text-gray-300 whitespace-pre-wrap line-clamp-2 overflow-hidden';
      bodyTextEl.textContent = fullText;

      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'text-[10px] text-gray-400 hover:text-gray-200';
      toggle.textContent = 'Expand';
      let isExpanded = false;
      const toggleVisible = fullText.length > 240;
      toggle.classList.toggle('hidden', !toggleVisible);
      toggle.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        isExpanded = !isExpanded;
        bodyTextEl.classList.toggle('line-clamp-2', !isExpanded);
        bodyTextEl.classList.toggle('overflow-hidden', !isExpanded);
        toggle.textContent = isExpanded ? 'Collapse' : 'Expand';
      });

      item.appendChild(header);
      item.appendChild(meta);
      item.appendChild(bodyTextEl);
      if (toggleVisible) item.appendChild(toggle);
      notificationList.appendChild(item);
    });
  }

  function focusChatInput() {
    if (chatInput) {
      chatInput.focus();
      chatInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      window.location.hash = 'chat';
    }
  }

  function setSessionId(sessionId) {
    if (!sessionId) return;
    orionSessionId = sessionId;
    localStorage.setItem('orion_sid', sessionId);
  }

  async function handleAttentionAck(attentionId, ackType, note) {
    if (!attentionId) return;
    try {
      const resp = await hubFetch(apiUrl(`/api/attention/${attentionId}/ack`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ack_type: ackType, note }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      removePendingAttention(attentionId);
    } catch (err) {
      console.warn('Failed to acknowledge attention', err);
    }
  }

  async function handleChatMessageReceipt(messageId, sessionId, receiptType) {
    if (!messageId || !sessionId) return;
    if (String(messageId).startsWith('synthetic:')) return;
    // Deduplicate double-clicks / bubbling
    const key = `${messageId}:${receiptType}`;
    window.__orionReceiptInFlight = window.__orionReceiptInFlight || new Set();
    if (window.__orionReceiptInFlight.has(key)) return;
    window.__orionReceiptInFlight.add(key);
    try {
      const resp = await hubFetch(apiUrl(`/api/chat/message/${messageId}/receipt`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, receipt_type: receiptType }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (receiptType === 'dismissed' || receiptType === 'opened' || receiptType === 'seen') {
        updateChatMessageStatus(messageId, 'seen');
      }
    } catch (err) {
      console.warn('Failed to send chat message receipt', err);
    } finally {
      try { window.__orionReceiptInFlight.delete(key); } catch (_) {}
    }
  }

  function showAttentionToast(notification) {
    if (!toastContainer || !notification) return;
    const severity = (notification.severity || 'info').toLowerCase();
    const toast = document.createElement('div');
    toast.className = `bg-gray-900/95 border border-gray-700 rounded-lg shadow-lg p-3 w-80 ${toastBorderClass(severity)}`;

    const title = document.createElement('div');
    title.className = 'text-sm font-semibold text-gray-100';
    title.textContent = notification.title || 'Attention';

    const body = document.createElement('div');
    body.className = 'text-xs text-gray-300 mt-1 line-clamp-3';
    body.textContent = notification.body_text || '';

    const actions = document.createElement('div');
    actions.className = 'flex items-center gap-2 mt-3 text-[10px]';

    const openBtn = document.createElement('button');
    openBtn.className = 'px-2 py-1 rounded bg-indigo-600/80 hover:bg-indigo-500 text-white';
    openBtn.textContent = 'Open chat';
    openBtn.addEventListener('click', () => {
      focusChatInput();
      toast.remove();
    });

    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
    dismissBtn.textContent = 'Dismiss';
    dismissBtn.addEventListener('click', () => {
      handleAttentionAck(notification.attention_id, 'dismissed');
      toast.remove();
    });

    const snoozeBtn = document.createElement('button');
    snoozeBtn.className = 'px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
    snoozeBtn.textContent = 'Snooze 30m';
    snoozeBtn.addEventListener('click', () => {
      handleAttentionAck(notification.attention_id, 'snooze', 'Snoozed for 30 minutes');
      toast.remove();
    });

    actions.appendChild(openBtn);
    actions.appendChild(dismissBtn);
    actions.appendChild(snoozeBtn);

    toast.appendChild(title);
    toast.appendChild(body);
    toast.appendChild(actions);
    toastContainer.appendChild(toast);

    if (severity !== 'error' && severity !== 'critical') {
      setTimeout(() => toast.remove(), notificationToastSeconds * 1000);
    }
  }

  function showChatMessageToast(notification) {
    if (!toastContainer || !notification) return;
    if (notification.silent) return;
    const severity = (notification.severity || 'info').toLowerCase();
    const toast = document.createElement('div');
    toast.className = `bg-gray-900/95 border border-gray-700 rounded-lg shadow-lg p-3 w-80 ${toastBorderClass(severity)}`;

    const title = document.createElement('div');
    title.className = 'text-sm font-semibold text-gray-100';
    title.textContent = notification.title || 'New message from Orion';

    const body = document.createElement('div');
    body.className = 'text-xs text-gray-300 mt-1 line-clamp-3';
    body.textContent = notification.body_text || '';

    const actions = document.createElement('div');
    actions.className = 'flex items-center gap-2 mt-3 text-[10px]';

    const openBtn = document.createElement('button');
    openBtn.className = 'px-2 py-1 rounded bg-indigo-600/80 hover:bg-indigo-500 text-white';
    openBtn.textContent = 'Open chat';
    openBtn.addEventListener('click', () => {
      setSessionId(notification.session_id);
      focusChatInput();
      handleChatMessageReceipt(notification.message_id, notification.session_id, 'opened');
      toast.remove();
    });

    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
    dismissBtn.textContent = 'Dismiss';
    dismissBtn.addEventListener('click', () => {
      handleChatMessageReceipt(notification.message_id, notification.session_id, 'dismissed');
      toast.remove();
    });

    actions.appendChild(openBtn);
    actions.appendChild(dismissBtn);

    toast.appendChild(title);
    toast.appendChild(body);
    toast.appendChild(actions);
    toastContainer.appendChild(toast);

    if (severity !== 'error' && severity !== 'critical') {
      setTimeout(() => toast.remove(), notificationToastSeconds * 1000);
    }
  }

  function showToastNotification(notification) {
    if (!toastContainer || !notification) return;

    if (isAttentionNotification(notification) && notification.attention_id) {
      showAttentionToast(notification);
      return;
    }
    if (isChatMessageNotification(notification) && notification.message_id) {
      showChatMessageToast(notification);
      return;
    }
    const severity = (notification.severity || 'info').toLowerCase();
    const toast = document.createElement('div');
    toast.className = `bg-gray-900/90 border border-gray-700 rounded-lg shadow-lg p-3 w-72 ${toastBorderClass(severity)}`;

    const title = document.createElement('div');
    title.className = 'text-sm font-semibold text-gray-100';
    title.textContent = notification.title || 'Notification';

    const body = document.createElement('div');
    body.className = 'text-xs text-gray-300 mt-1 line-clamp-3';
    body.textContent = notification.body_text || '';

    const footer = document.createElement('div');
    footer.className = 'flex items-center justify-between mt-2';

    const meta = document.createElement('span');
    meta.className = 'text-[10px] text-gray-500';
    meta.textContent = notification.severity ? notification.severity.toUpperCase() : 'INFO';

    const dismiss = document.createElement('button');
    dismiss.className = 'text-[10px] text-gray-400 hover:text-white';
    dismiss.textContent = 'Dismiss';
    dismiss.addEventListener('click', () => toast.remove());

    footer.appendChild(meta);
    footer.appendChild(dismiss);

    toast.appendChild(title);
    toast.appendChild(body);
    toast.appendChild(footer);
    toastContainer.appendChild(toast);

    if (severity !== 'error' && severity !== 'critical') {
      setTimeout(() => toast.remove(), notificationToastSeconds * 1000);
    }
  }

  function severityBadgeClass(severity) {
    switch ((severity || '').toLowerCase()) {
      case 'critical':
        return 'bg-red-700/80 text-red-100';
      case 'error':
        return 'bg-red-600/70 text-red-100';
      case 'warning':
        return 'bg-yellow-600/70 text-yellow-100';
      default:
        return 'bg-gray-700/70 text-gray-200';
    }
  }

  function toastBorderClass(severity) {
    switch ((severity || '').toLowerCase()) {
      case 'critical':
        return 'border-red-500';
      case 'error':
        return 'border-red-400';
      case 'warning':
        return 'border-yellow-400';
      default:
        return 'border-gray-600';
    }
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

  async function loadNotifications() {
    try {
      const resp = await hubFetch(apiUrl("/api/notifications?limit=50"));
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data)) {
        notifications = data;
        renderNotifications();
      }
    } catch (err) {
      console.warn("Failed to load notifications", err);
    }
  }

  function setSettingsStatus(message, isError = false) {
    if (!notifySettingsStatus) return;
    notifySettingsStatus.textContent = message;
    notifySettingsStatus.classList.toggle('text-red-400', isError);
    notifySettingsStatus.classList.toggle('text-gray-500', !isError);
  }

  function readPreferenceRows() {
    const rows = Array.from(document.querySelectorAll('.pref-row'));
    return rows.map((row) => {
      const scopeType = row.dataset.scopeType;
      const scopeValue = row.dataset.scopeValue;
      const channels = Array.from(row.querySelectorAll('.pref-channel'))
        .filter((input) => input.checked)
        .map((input) => input.dataset.channel);
      const escalationToggle = row.querySelector('.pref-escalation');
      const delayInput = row.querySelector('.pref-delay');
      const dedupeInput = row.querySelector('.pref-dedupe');
      const throttleMaxInput = row.querySelector('.pref-throttle-max');
      const throttleWindowInput = row.querySelector('.pref-throttle-window');
      return {
        recipient_group: RECIPIENT_GROUP,
        scope_type: scopeType,
        scope_value: scopeValue,
        channels_enabled: channels,
        escalation_enabled: escalationToggle ? escalationToggle.checked : null,
        escalation_delay_minutes: delayInput && delayInput.value ? parseInt(delayInput.value, 10) : null,
        dedupe_window_seconds: dedupeInput && dedupeInput.value ? parseInt(dedupeInput.value, 10) : null,
        throttle_max_per_window: throttleMaxInput && throttleMaxInput.value ? parseInt(throttleMaxInput.value, 10) : null,
        throttle_window_seconds: throttleWindowInput && throttleWindowInput.value ? parseInt(throttleWindowInput.value, 10) : null,
      };
    });
  }

  function applyPreferenceRows(preferences) {
    const rows = Array.from(document.querySelectorAll('.pref-row'));
    rows.forEach((row) => {
      const scopeType = row.dataset.scopeType;
      const scopeValue = row.dataset.scopeValue;
      const pref = preferences.find(
        (item) => item.scope_type === scopeType && item.scope_value === scopeValue
      );
      if (!pref) return;
      row.querySelectorAll('.pref-channel').forEach((input) => {
        input.checked = (pref.channels_enabled || []).includes(input.dataset.channel);
      });
      const escalationToggle = row.querySelector('.pref-escalation');
      if (escalationToggle) {
        escalationToggle.checked = pref.escalation_enabled ?? false;
      }
      const delayInput = row.querySelector('.pref-delay');
      if (delayInput) {
        delayInput.value = pref.escalation_delay_minutes ?? '';
      }
      const dedupeInput = row.querySelector('.pref-dedupe');
      if (dedupeInput) {
        dedupeInput.value = pref.dedupe_window_seconds ?? '';
      }
      const throttleMaxInput = row.querySelector('.pref-throttle-max');
      if (throttleMaxInput) {
        throttleMaxInput.value = pref.throttle_max_per_window ?? '';
      }
      const throttleWindowInput = row.querySelector('.pref-throttle-window');
      if (throttleWindowInput) {
        throttleWindowInput.value = pref.throttle_window_seconds ?? '';
      }
    });
  }

  async function loadNotifySettings() {
    try {
      const profileResp = await hubFetch(apiUrl(`/api/notify/recipients/${RECIPIENT_GROUP}`));
      if (profileResp.ok) {
        const profile = await profileResp.json();
        if (notifyDisplayName) notifyDisplayName.value = profile.display_name || '';
        if (notifyTimezone) notifyTimezone.value = profile.timezone || '';
        if (notifyQuietEnabled) notifyQuietEnabled.checked = Boolean(profile.quiet_hours_enabled);
        if (notifyQuietStart) notifyQuietStart.value = profile.quiet_start_local || '22:00';
        if (notifyQuietEnd) notifyQuietEnd.value = profile.quiet_end_local || '07:00';
      }
    } catch (err) {
      console.warn('Failed to load notify profile', err);
    }
    try {
      const prefsResp = await hubFetch(apiUrl(`/api/notify/recipients/${RECIPIENT_GROUP}/preferences`));
      if (prefsResp.ok) {
        const prefs = await prefsResp.json();
        if (Array.isArray(prefs)) applyPreferenceRows(prefs);
      }
      setSettingsStatus('Loaded');
    } catch (err) {
      setSettingsStatus('Failed to load settings', true);
      console.warn('Failed to load notify preferences', err);
    }
  }

  async function saveNotifySettings() {
    if (!notifySettingsSave) return;
    setSettingsStatus('Saving...');
    try {
      const profilePayload = {
        display_name: notifyDisplayName ? notifyDisplayName.value.trim() : null,
        timezone: notifyTimezone ? notifyTimezone.value.trim() : null,
        quiet_hours_enabled: notifyQuietEnabled ? notifyQuietEnabled.checked : null,
        quiet_start_local: notifyQuietStart ? notifyQuietStart.value : null,
        quiet_end_local: notifyQuietEnd ? notifyQuietEnd.value : null,
      };
      await hubFetch(apiUrl(`/api/notify/recipients/${RECIPIENT_GROUP}`), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profilePayload),
      });

      const preferencesPayload = { preferences: readPreferenceRows() };
      const prefResp = await hubFetch(apiUrl(`/api/notify/recipients/${RECIPIENT_GROUP}/preferences`), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(preferencesPayload),
      });
      if (!prefResp.ok) {
        const errData = await prefResp.json();
        throw new Error(errData.detail || 'Failed to save preferences');
      }
      setSettingsStatus('Saved');
    } catch (err) {
      setSettingsStatus('Save failed', true);
      console.warn('Failed to save notify settings', err);
    }
  }

  async function loadChatMessages() {
    try {
      const filter = messageFilter ? messageFilter.value : 'unread';
      const statusParam = filter === 'all' ? '' : 'unread';
      const resp = await hubFetch(apiUrl(`/api/chat/messages?limit=50&status=${statusParam}`));
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data)) {
        chatMessages = data
          .map((item) => ({
          message_id: item.message_id || item.messageId,
          session_id: item.session_id || item.sessionId,
          created_at: item.created_at || item.createdAt,
          severity: item.severity || 'info',
          title: item.title || 'New message from Orion',
          preview_text: item.preview_text || item.previewText || '',
          status: (item.status || 'unread').toLowerCase(),
          silent: false,
          }))
          .filter((m) => m && m.session_id && m.message_id)
          .filter((m) => !isDismissed(m.message_id));
        renderChatMessages();
        chatMessages.forEach((item) => {
          if (isDismissed(item.message_id)) return;
          if ((item.status || '').toLowerCase() === 'unread' && !seenMessageIds.has(item.message_id)) {
            seenMessageIds.add(item.message_id);
            handleChatMessageReceipt(item.message_id, item.session_id, 'seen');
          }
        });
      }
    } catch (err) {
      console.warn('Failed to load chat messages', err);
    }
  }

  async function loadPendingAttention() {
    try {
      const resp = await hubFetch(apiUrl("/api/attention?status=pending&limit=50"));
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data)) {
        pendingAttention = data.map((item) => ({
          attention_id: item.attention_id,
          created_at: item.created_at,
          severity: item.severity || 'info',
          title: item.reason || 'Attention',
          message: item.message || '',
          source_service: item.source_service || 'unknown',
        }));
        renderPendingAttention();
      }
    } catch (err) {
      console.warn('Failed to load pending attention', err);
    }
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
        hubFetch(apiUrl(`/api/topics/summary?${summaryParams.toString()}`)),
        hubFetch(apiUrl(`/api/topics/drift?${driftParams.toString()}`)),
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
      if (settingsPanel.classList.contains('translate-x-0')) {
        loadNotifySettings();
      }
    });
  }
  if (settingsClose && settingsPanel) {
    settingsClose.addEventListener('click', () => {
      settingsPanel.classList.add('translate-x-full');
      settingsPanel.classList.remove('translate-x-0');
    });
  }
  if (notifySettingsSave) {
    notifySettingsSave.addEventListener('click', () => saveNotifySettings());
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
          const res = await hubFetch(apiUrl("/api/cognition/library"));
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
          if (verbDropdown) verbDropdown.classList.toggle('hidden');
      });
      document.addEventListener('click', (e) => {
          if (verbDropdown && !verbDropdown.classList.contains('hidden')) {
              if (!verbDropdown.contains(e.target) && !verbSelectTrigger.contains(e.target)) {
                  verbDropdown.classList.add('hidden');
              }
          }
      });
  }

  if (memoryPanelToggle) {
    memoryPanelToggle.addEventListener('click', toggleMemoryPanel);
  }

  // --- WebSocket ---
  function setupWebSocket() {
    const wsEndpoint = wsUrl("/ws");
    console.log(`[WS] Connecting to ${wsEndpoint}...`);
    socket = new WebSocket(wsEndpoint);

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
          if (d.kind === 'notification' && d.notification) {
            addNotification(d.notification);
          }
          if (d.memory_digest || d.recall_debug || typeof d.memory_used === 'boolean') {
            updateMemoryPanelFromResponse(d);
          }
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

  async function sendTextMessage() {
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
       no_write: noWriteToggle ? noWriteToggle.checked : false,
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
        
        if (!orionSessionId) await initSession();
        payload.session_id = orionSessionId;

        const headers = {'Content-Type': 'application/json'};
        if (orionSessionId) headers['X-Orion-Session-Id'] = orionSessionId;
        if (noWriteToggle && noWriteToggle.checked) headers['X-Orion-No-Write'] = '1';

        // Fallback to HTTP if WS is down
        hubFetch(apiUrl("/api/chat"), {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({messages:[{role:'user', content:text}], ...payload})
        })
        .then(r => r.json())
        .then(d => {
            if(d.text) appendMessage('Orion', d.text);
            else if(d.error) appendMessage('System', d.error, 'text-red-400');
            updateMemoryPanelFromResponse(d);
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
               no_write: noWriteToggle ? noWriteToggle.checked : false,
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
       const sid = orionSessionId || localStorage.getItem('orion_sid');
       const r = await hubFetch(apiUrl("/api/session"), {headers: sid ? {'X-Orion-Session-Id': sid} : {}});
       const d = await r.json();
       if(d.session_id) {
         orionSessionId = d.session_id;
         localStorage.setItem('orion_sid', orionSessionId);
         console.log("[Session] Initialized:", orionSessionId);
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
        const resp = await hubFetch(apiUrl("/submit-collapse"), {
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
      if (document.body && document.body.dataset.toastSeconds) {
        const parsed = parseInt(document.body.dataset.toastSeconds, 10);
        if (!Number.isNaN(parsed)) notificationToastSeconds = parsed;
      }
      await loadNotifications();
      await loadChatMessages();
      await loadPendingAttention();
      if (notificationFilter) {
        notificationFilter.addEventListener('change', renderNotifications);
      }
      if (messageFilter) {
        messageFilter.addEventListener('change', () => {
          renderChatMessages();
          loadChatMessages();
        });
      }
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
        
        if (visionIsFloating && visionFloatingContainer) {
            visionFloatingContainer.classList.remove("hidden");
            const flex = visionFloatingContainer.querySelector('.flex-1');
            if (flex) flex.innerHTML = imgHtml;
            visionDockedContainer.innerHTML = `<div class="text-gray-500 text-xs">Viewing in Pop-out</div>`;
            if (visionPopoutButton) visionPopoutButton.textContent = "Dock";
        } else {
            if (visionFloatingContainer) visionFloatingContainer.classList.add("hidden");
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

  async function refreshTopicStudioStatus() {
    if (!tsStatusBadge) return;
    try {
      setLoading(tsStatusLoading, true);
      setTopicStudioRenderStep("fetching /ready");
      const result = await topicFoundryFetch("/ready");
      const checks = result?.checks || {};
      const pgOk = typeof checks.pg?.ok === "boolean" ? checks.pg.ok : null;
      const embeddingOk = typeof checks.embedding?.ok === "boolean" ? checks.embedding.ok : null;
      const modelDirOk = typeof checks.model_dir?.ok === "boolean" ? checks.model_dir.ok : null;
      formatStatusBadge(tsStatusBadge, true, "Reachable");
      formatStatusBadge(tsStatusPg, pgOk, pgOk === null ? "--" : pgOk ? "ok" : "fail");
      formatStatusBadge(tsStatusEmbedding, embeddingOk, embeddingOk === null ? "--" : embeddingOk ? "ok" : "fail");
      formatStatusBadge(tsStatusModelDir, modelDirOk, modelDirOk === null ? "--" : modelDirOk ? "ok" : "fail");
      if (tsStatusDetail) {
        tsStatusDetail.textContent = `PG: ${checks.pg?.detail || "--"} · Embedding: ${checks.embedding?.detail || "--"} · Model dir: ${checks.model_dir?.detail || "--"}`;
      }
      renderEndpointWarning(tsReadyWarning, null, null);
      recordTopicStudioFetchStatus("ready", 200, true);
      setTopicStudioRenderStep("fetched /ready");
      setLoading(tsStatusLoading, false);
    } catch (err) {
      formatStatusBadge(tsStatusBadge, false, "Unreachable");
      formatStatusBadge(tsStatusPg, null, "--");
      formatStatusBadge(tsStatusEmbedding, null, "--");
      formatStatusBadge(tsStatusModelDir, null, "--");
      renderError(tsStatusDetail, err, "Failed to read /ready.");
      renderEndpointWarning(tsReadyWarning, "/ready", err);
      recordTopicStudioFetchStatus("ready", err.status ?? "error", false, err.body || err.message);
      setTopicStudioRenderStep("failed /ready");
      setLoading(tsStatusLoading, false);
    }
  }

  function renderSegmentationModes(modes = [], llmEnabled = true) {
    if (!tsSegmentationMode) return;
    tsSegmentationMode.innerHTML = "";
    modes.forEach((mode) => {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = mode;
      if (!llmEnabled && (mode.includes("llm"))) {
        option.disabled = true;
      }
      tsSegmentationMode.appendChild(option);
    });
    if (!llmEnabled) {
      if (tsLlmNote) tsLlmNote.classList.remove("hidden");
    } else if (tsLlmNote) {
      tsLlmNote.classList.add("hidden");
    }
    const current = tsSegmentationMode.value;
    const selectedOption = tsSegmentationMode.querySelector(`option[value="${current}"]`);
    if (selectedOption && selectedOption.disabled) {
      const firstEnabled = Array.from(tsSegmentationMode.options).find((opt) => !opt.disabled);
      if (firstEnabled) {
        tsSegmentationMode.value = firstEnabled.value;
      }
    }
  }

  function renderMetricOptions(metrics = [], defaultMetric = "") {
    if (!tsModelMetric) return;
    const current = tsModelMetric.value;
    tsModelMetric.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select metric";
    tsModelMetric.appendChild(placeholder);
    metrics.forEach((metric) => {
      const option = document.createElement("option");
      option.value = metric;
      option.textContent = metric;
      tsModelMetric.appendChild(option);
    });
    if (current && metrics.includes(current)) {
      tsModelMetric.value = current;
    } else if (defaultMetric && metrics.includes(defaultMetric)) {
      tsModelMetric.value = defaultMetric;
    }
  }

  function applyCapabilityDefaults(defaults = {}, embeddingDefault = "") {
    if (tsModelEmbeddingUrl) {
      const current = tsModelEmbeddingUrl.value || "";
      const isStale = current.includes("orion-vector-host");
      if (!current || isStale) {
        tsModelEmbeddingUrl.value = embeddingDefault || defaults.embedding_source_url || "";
      }
    }
    if (tsModelMetric && !tsModelMetric.value) {
      tsModelMetric.value = defaults.metric || "";
    }
    if (tsModelMinCluster && !tsModelMinCluster.value && defaults.min_cluster_size) {
      tsModelMinCluster.value = defaults.min_cluster_size;
    }
  }

  async function refreshTopicStudioCapabilities() {
    if (tsCapabilitiesWarning) {
      tsCapabilitiesWarning.classList.add("hidden");
      tsCapabilitiesWarning.textContent = "";
    }
    try {
      setLoading(tsStatusLoading, true, "Loading capabilities...");
      setTopicStudioRenderStep("fetching /capabilities");
      const result = await topicFoundryFetch("/capabilities");
      topicStudioCapabilities = result;
      const modes = result?.segmentation_modes_supported ?? [];
      const metrics = result.supported_metrics || [];
      renderSegmentationModes(modes, Boolean(result.llm_enabled));
      renderMetricOptions(metrics, result.default_metric);
      const embeddingDefault = result?.defaults?.embedding_source_url || result?.default_embedding_url || "";
      applyCapabilityDefaults(result.defaults || {}, embeddingDefault);
      if (tsEnrichRun) {
        const llmEnabled = Boolean(result.llm_enabled);
        tsEnrichRun.disabled = !llmEnabled;
        tsEnrichRun.title = llmEnabled ? "" : "LLM disabled";
      }
      renderEndpointWarning(tsCapabilitiesWarning, null, null);
      recordTopicStudioFetchStatus("capabilities", 200, true);
      setTopicStudioRenderStep("fetched /capabilities");
      setLoading(tsStatusLoading, false);
    } catch (err) {
      const fallbackModes = ["time_gap", "semantic", "hybrid"];
      const fallbackMetrics = ["euclidean", "cosine"];
      renderSegmentationModes(fallbackModes, false);
      renderMetricOptions(fallbackMetrics, "cosine");
      renderEndpointWarning(tsCapabilitiesWarning, "/capabilities", err);
      recordTopicStudioFetchStatus("capabilities", err.status ?? "error", false, err.body || err.message);
      setTopicStudioRenderStep("failed /capabilities");
      if (tsEnrichRun) {
        tsEnrichRun.disabled = true;
        tsEnrichRun.title = "LLM disabled";
      }
      setLoading(tsStatusLoading, false);
    }
  }

  async function refreshTopicStudio() {
    if (topicFoundryBaseLabel) {
      topicFoundryBaseLabel.textContent = TOPIC_FOUNDRY_PROXY_BASE;
    }
    setTopicStudioRenderStep("refresh topic studio");
    await refreshTopicStudioCapabilities();
    await refreshTopicStudioStatus();
    try {
      const schemasResponse = await topicFoundryFetch("/introspect/schemas");
      topicStudioIntrospectionSchemas = schemasResponse?.schemas || [];
      const priorSchema = tsDatasetSchema?.value || "";
      renderSchemaOptionsForDataset();
      setDatasetSourceEditable(topicStudioIntrospectionSchemas.length === 0);
      if (tsDatasetSchema && priorSchema && topicStudioIntrospectionSchemas.includes(priorSchema)) {
        tsDatasetSchema.value = priorSchema;
      }
      if (tsDatasetSchema && !tsDatasetSchema.value && topicStudioIntrospectionSchemas.length > 0) {
        tsDatasetSchema.value = topicStudioIntrospectionSchemas[0];
      }
      if (tsDatasetSchema?.value) {
        await loadDatasetTables(tsDatasetSchema.value);
      }
      if (tsDatasetSchema?.value && tsDatasetTableSelect?.value) {
        updateSourceTableInput();
        await loadDatasetColumnsFromSourceTable(`${tsDatasetSchema.value}.${tsDatasetTableSelect.value}`);
      }
    } catch (err) {
      topicStudioIntrospectionSchemas = [];
      setDatasetSourceEditable(true);
    }
    try {
      const datasetsResponse = await topicFoundryFetch("/datasets");
      topicStudioDatasets = parseDatasets(datasetsResponse);
      topicStudioMvpState.datasets = topicStudioDatasets;
      renderDatasetOptions();
      renderConversationDatasetOptions();
      if (tsDatasetSelect?.value) {
        const selected = topicStudioDatasets.find((dataset) => dataset.id === tsDatasetSelect.value);
        if (selected?.raw) {
          loadGroupByCandidatesFromDataset(selected.raw);
        }
      }
      if (HUB_DEBUG) {
        const debugLine = document.getElementById("tsDebugLine");
        if (debugLine) {
          const firstId = topicStudioDatasets[0]?.id || "--";
          debugLine.textContent = `datasetsCount=${topicStudioDatasets.length} first=${firstId}`;
          debugLine.classList.remove("hidden");
        }
      }
    } catch (err) {
      console.warn("[TopicStudio] Failed to load datasets", err);
      if (HUB_DEBUG) {
        const debugLine = document.getElementById("tsDebugLine");
        if (debugLine) {
          debugLine.textContent = "datasetsCount=0 first=--";
          debugLine.classList.remove("hidden");
        }
      }
    }
    try {
      const modelsResponse = await topicFoundryFetch("/models");
      topicStudioModels = modelsResponse?.models || [];
      renderModelOptions();
      if (tsDriftModelName && !tsDriftModelName.value) {
        if (topicStudioModels.length > 0) {
          tsDriftModelName.value = topicStudioModels[0].name || "";
        } else if (tsModelName?.value) {
          tsDriftModelName.value = tsModelName.value;
        }
      }
    } catch (err) {
      console.warn("[TopicStudio] Failed to load models", err);
    }
    try {
      if (tsRunsWarning) {
        tsRunsWarning.classList.add("hidden");
        tsRunsWarning.textContent = "";
      }
      const runsResponse = await topicFoundryFetch("/runs?limit=20");
      topicStudioRuns = normalizeRunsResponse(runsResponse);
      renderRunsSelect();
      renderCompareRunOptions();
      if (tsRunsSelect?.value) {
        setSelectedRun(tsRunsSelect.value);
        loadSegments();
      }
      if (tsKgRunId && !tsKgRunId.value && tsRunsSelect?.value) {
        tsKgRunId.value = tsRunsSelect.value;
      }
      recordTopicStudioFetchStatus("runs", 200, true);
      setTopicStudioRenderStep("fetched /runs");
    } catch (err) {
      console.warn("[TopicStudio] Failed to load runs", err);
      recordTopicStudioFetchStatus("runs", err.status ?? "error", false, err.body || err.message);
      setTopicStudioRenderStep("failed /runs");
      if (tsRunsWarning) {
        tsRunsWarning.textContent = `Failed to load runs. Enter a run id manually. ${err.status ? `status ${err.status}` : ""} ${err.body || err.message || ""}`.trim();
        tsRunsWarning.classList.remove("hidden");
      }
    }
    setTopicStudioSubview(resolveTopicStudioSubview());
  }

  applyTopicStudioState();
  bindTopicStudioPersistence();
  if (HUB_DEBUG && tsDebugDrawer) {
    tsDebugDrawer.classList.remove("hidden");
  }
  if (tsUsePreviewSpec) {
    tsUsePreviewSpec.disabled = true;
  }
  setTopicStudioSubview(resolveTopicStudioSubview());
  if (tsSkeletonRetry) {
    tsSkeletonRetry.addEventListener("click", () => {
      renderTopicStudioSkeleton("Loading...");
      refreshTopicStudio().catch((err) => {
        console.warn("[TopicStudio] Retry failed", err);
      });
    });
  }

  document.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target.closest("[data-hash-target]") : null;
    if (!target) return;
    const hash = target.getAttribute("data-hash-target");
    if (!hash) return;
    event.preventDefault();
    navigateToHash(hash);
  });
  window.addEventListener("hashchange", routeFromHash);
  if (!window.location.hash) {
    window.location.hash = "#hub";
  } else {
    routeFromHash();
  }

  if (tsDatasetSelect) {
    tsDatasetSelect.addEventListener("change", async () => {
      const selected = topicStudioDatasets.find((dataset) => dataset.id === tsDatasetSelect.value);
      if (selected?.raw) {
        await populateDatasetForm(selected.raw);
        loadGroupByCandidatesFromDataset(selected.raw);
        updateBoundaryRequirement();
      }
    });
  }

  if (tsWindowingMode) {
    tsWindowingMode.addEventListener("change", () => {
      updateGroupByVisibility();
      updateBoundaryRequirement();
      saveTopicStudioState();
    });
  }

  if (tsRunPreset) {
    tsRunPreset.addEventListener("change", () => {
      applyRunPreset(tsRunPreset.value);
      updateBoundaryRequirement();
    });
  }

  if (tsDatasetSchema) {
    tsDatasetSchema.addEventListener("change", async () => {
      await loadDatasetTables(tsDatasetSchema.value);
      topicStudioDatasetColumns = [];
      renderDatasetColumnOptions();
      updateSourceTableInput();
      saveTopicStudioState();
    });
  }

  if (tsDatasetTableSelect) {
    tsDatasetTableSelect.addEventListener("change", async () => {
      updateSourceTableInput();
      if (tsDatasetSchema?.value && tsDatasetTableSelect.value) {
        await loadDatasetColumnsFromSourceTable(`${tsDatasetSchema.value}.${tsDatasetTableSelect.value}`);
      }
      saveTopicStudioState();
    });
  }

  if (tsGroupByColumn) {
    tsGroupByColumn.addEventListener("change", saveTopicStudioState);
  }

  if (tsDatasetBoundaryColumn) {
    tsDatasetBoundaryColumn.addEventListener("change", () => {
      updateBoundaryRequirement();
      saveTopicStudioState();
    });
  }

  if (tsRunsSelect) {
    tsRunsSelect.addEventListener("change", () => {
      if (!tsRunsSelect.value) return;
      setSelectedRun(tsRunsSelect.value);
      resetSegmentsPaging();
      loadSegments();
    });
  }

  if (tsSubviewRunsBtn) {
    tsSubviewRunsBtn.addEventListener("click", () => {
      setTopicStudioSubview("runs");
    });
  }

  if (tsSubviewConversationsBtn) {
    tsSubviewConversationsBtn.addEventListener("click", () => {
      setTopicStudioSubview("conversations");
      loadConversations();
    });
  }

  if (tsSubviewTopicsBtn) {
    tsSubviewTopicsBtn.addEventListener("click", () => {
      setTopicStudioSubview("topics");
    });
  }

  if (tsSubviewCompareBtn) {
    tsSubviewCompareBtn.addEventListener("click", () => {
      setTopicStudioSubview("compare");
    });
  }

  if (tsSubviewDriftBtn) {
    tsSubviewDriftBtn.addEventListener("click", () => {
      setTopicStudioSubview("drift");
      loadDriftRecords();
    });
  }

  if (tsSubviewEventsBtn) {
    tsSubviewEventsBtn.addEventListener("click", () => {
      setTopicStudioSubview("events");
      loadEvents();
    });
  }

  if (tsSubviewKgBtn) {
    tsSubviewKgBtn.addEventListener("click", () => {
      setTopicStudioSubview("kg");
      loadKgEdges();
    });
  }

  if (tsConvoDatasetSelect) {
    tsConvoDatasetSelect.addEventListener("change", () => {
      saveTopicStudioState();
      loadConversations();
    });
  }

  if (tsConvoLoad) {
    tsConvoLoad.addEventListener("click", () => {
      loadConversations();
    });
  }

  if (tsTopicsLoad) {
    tsTopicsLoad.addEventListener("click", () => {
      loadTopicExplorer();
    });
  }

  if (tsTopicSegmentsLimit) {
    tsTopicSegmentsLimit.addEventListener("change", () => {
      saveTopicStudioState();
      updateTopicSegmentsOffset(0);
    });
  }

  if (tsTopicSegmentsOffset) {
    tsTopicSegmentsOffset.addEventListener("change", () => {
      updateTopicSegmentsOffset(Number(tsTopicSegmentsOffset.value || 0));
    });
  }

  if (tsTopicSegmentsPrev) {
    tsTopicSegmentsPrev.addEventListener("click", () => {
      const limit = Number(tsTopicSegmentsLimit?.value || 50);
      updateTopicSegmentsOffset(Math.max(0, topicStudioTopicSegmentsOffset - limit));
    });
  }

  if (tsTopicSegmentsNext) {
    tsTopicSegmentsNext.addEventListener("click", () => {
      const limit = Number(tsTopicSegmentsLimit?.value || 50);
      updateTopicSegmentsOffset(topicStudioTopicSegmentsOffset + limit);
    });
  }

  if (tsCompareRun) {
    tsCompareRun.addEventListener("click", () => {
      loadRunCompare();
    });
  }

  if (tsDriftLoad) {
    tsDriftLoad.addEventListener("click", () => {
      loadDriftRecords();
    });
  }

  if (tsDriftRunNow) {
    tsDriftRunNow.addEventListener("click", () => {
      runDriftNow();
    });
  }

  if (tsEventsLoad) {
    tsEventsLoad.addEventListener("click", () => {
      loadEvents();
    });
  }

  if (tsEventsExport) {
    tsEventsExport.addEventListener("click", () => {
      exportEventsCsv();
    });
  }

  if (tsKgLoad) {
    tsKgLoad.addEventListener("click", () => {
      loadKgEdges();
    });
  }

  if (tsKgExport) {
    tsKgExport.addEventListener("click", () => {
      exportKgCsv();
    });
  }

  if (tsConvoMerge) {
    tsConvoMerge.addEventListener("click", () => {
      mergeConversations();
    });
  }

  if (tsConvoRebuildPreview) {
    tsConvoRebuildPreview.addEventListener("click", async () => {
      try {
        setLoading(tsPreviewLoading, true);
        clearPreview();
        const payload = {
          dataset: buildDatasetSpec(),
          windowing: buildWindowingSpec(),
          start_at: parseDateInput(tsConvoStartAt?.value) || parseDateInput(tsStartAt?.value),
          end_at: parseDateInput(tsConvoEndAt?.value) || parseDateInput(tsEndAt?.value),
          limit: 200,
        };
        await executePreview(payload);
      } catch (err) {
        renderError(tsPreviewError, err, "Failed to rebuild preview.");
        showToast("Failed to rebuild preview.");
        setLoading(tsPreviewLoading, false);
      }
    });
  }

  if (tsCopyReadyUrl) {
    tsCopyReadyUrl.addEventListener("click", () => {
      copyText(`${TOPIC_FOUNDRY_PROXY_BASE}/ready`, "Ready URL copied.");
    });
  }

  if (tsCopyCapabilitiesUrl) {
    tsCopyCapabilitiesUrl.addEventListener("click", () => {
      copyText(`${TOPIC_FOUNDRY_PROXY_BASE}/capabilities`, "Capabilities URL copied.");
    });
  }

  if (tsCopyRunId) {
    tsCopyRunId.addEventListener("click", () => {
      copyText(tsRunId?.value, "Run id copied.");
    });
  }

  if (tsCopyRunUrl) {
    tsCopyRunUrl.addEventListener("click", () => {
      if (!tsRunId?.value) {
        showToast("Enter a run id to copy the URL.");
        return;
      }
      copyText(`${TOPIC_FOUNDRY_PROXY_BASE}/runs/${tsRunId.value}`, "Run URL copied.");
    });
  }

  if (tsCopyArtifacts) {
    tsCopyArtifacts.addEventListener("click", () => {
      copyText(tsRunArtifacts?.textContent, "Artifacts copied.");
    });
  }

  if (tsCopyDatasetId) {
    tsCopyDatasetId.addEventListener("click", () => {
      copyText(tsDatasetSelect?.value, "Dataset id copied.");
    });
  }

  if (tsCopyModelId) {
    tsCopyModelId.addEventListener("click", () => {
      copyText(tsTrainModelSelect?.value, "Model id copied.");
    });
  }

  if (tsCopySegmentId) {
    tsCopySegmentId.addEventListener("click", () => {
      copyText(topicStudioSelectedSegmentId, "Segment id copied.");
    });
  }

  if (tsSegmentFullTextLoad) {
    tsSegmentFullTextLoad.addEventListener("click", async () => {
      if (!topicStudioSelectedSegmentId) {
        showToast("Select a segment first.");
        return;
      }
      try {
        if (tsSegmentFullTextStatus) tsSegmentFullTextStatus.textContent = "Loading full text...";
        const result = await topicFoundryFetch(`/segments/${topicStudioSelectedSegmentId}?include_full_text=true`);
        renderSegmentDetailPanel(result, { include_full_text: true });
        recordTopicStudioDebug("segments", {
          request: `/segments/${topicStudioSelectedSegmentId}?include_full_text=true`,
          response: result,
        });
      } catch (err) {
        if (tsSegmentFullTextStatus) tsSegmentFullTextStatus.textContent = "Failed to load full text.";
        renderTopicStudioError(tsSegmentsError, err, "Load full text", "segments", {
          request: `/segments/${topicStudioSelectedSegmentId}?include_full_text=true`,
        });
      }
    });
  }

  if (tsSegmentFullTextExpand) {
    tsSegmentFullTextExpand.addEventListener("click", () => {
      if (!topicStudioSegmentFullText) return;
      topicStudioSegmentFullTextExpanded = !topicStudioSegmentFullTextExpanded;
      tsSegmentFullTextExpand.textContent = topicStudioSegmentFullTextExpanded ? "Collapse" : "Expand";
      if (tsSegmentFullText) {
        tsSegmentFullText.style.maxHeight = topicStudioSegmentFullTextExpanded ? "none" : "320px";
      }
      renderSegmentFullText();
    });
  }

  if (tsSegmentFullTextCopy) {
    tsSegmentFullTextCopy.addEventListener("click", () => {
      if (!topicStudioSegmentFullText) {
        showToast("Load full text first.");
        return;
      }
      copyText(topicStudioSegmentFullText, "Full text copied.");
    });
  }

  if (tsCreateDataset) {
    tsCreateDataset.addEventListener("click", async () => {
      try {
        const payload = buildDatasetSpec();
        const response = await topicFoundryFetch("/datasets", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        showToast("Dataset created.");
        setDatasetSaveStatus("Dataset created.");
        await refreshTopicStudio();
        if (response?.dataset_id && tsDatasetSelect) {
          tsDatasetSelect.value = response.dataset_id;
        }
      } catch (err) {
        showToast("Failed to create dataset.");
        setDatasetSaveStatus("Failed to create dataset.", true);
      }
    });
  }

  if (tsSaveDataset) {
    tsSaveDataset.addEventListener("click", async () => {
      if (!tsDatasetSelect?.value) {
        showToast("Select a dataset to save.");
        setDatasetSaveStatus("Select a dataset to save.", true);
        return;
      }
      try {
        const payload = buildDatasetSpec();
        await topicFoundryFetch(`/datasets/${tsDatasetSelect.value}`, {
          method: "PATCH",
          body: JSON.stringify(payload),
        });
        showToast("Dataset saved.");
        setDatasetSaveStatus("Dataset saved.");
        await refreshTopicStudio();
      } catch (err) {
        showToast("Failed to save dataset.");
        setDatasetSaveStatus("Failed to save dataset.", true);
      }
    });
  }

  if (tsPreviewDataset) {
    tsPreviewDataset.addEventListener("click", async () => {
      try {
        const savedDataset = getSelectedSavedDataset();
        if (!savedDataset) {
          const message = "Select a saved dataset first.";
          if (tsPreviewError) tsPreviewError.textContent = message;
          setWarning(tsPreviewWarning, message);
          showToast(message);
          return;
        }
        if (boundaryColumnRequired() && !tsDatasetBoundaryColumn?.value) {
          setWarning(tsPreviewWarning, "Boundary column required. Select a boundary column and save the dataset before previewing.");
          return;
        }
        setLoading(tsPreviewLoading, true);
        clearPreview();
        const payload = {
          dataset: savedDataset,
          windowing: buildWindowingSpec(),
          start_at: parseDateInput(tsStartAt?.value),
          end_at: parseDateInput(tsEndAt?.value),
          limit: 200,
        };
        recordTopicStudioDebug("preview", { request: "/datasets/preview", payload });
        const result = await executePreview(payload);
        recordTopicStudioDebug("preview", { request: "/datasets/preview", response: result });
      } catch (err) {
        const detail = formatHttpError("preview", err);
        if (tsPreviewError) {
          tsPreviewError.textContent = detail;
        }
        showToast(detail);
        renderTopicStudioError(tsPreviewError, err, "Preview", "preview", {
          request: "/datasets/preview",
        });
        setWarning(tsPreviewWarning, null);
        setLoading(tsPreviewLoading, false);
      }
    });
  }

  if (tsCreateModel) {
    tsCreateModel.addEventListener("click", async () => {
      if (!tsDatasetSelect?.value) {
        const message = "Select a dataset before creating a model.";
        if (tsRunError) tsRunError.textContent = message;
        showToast(message);
        return;
      }
      const modelName = tsModelName?.value?.trim() || "";
      const modelVersion = tsModelVersion?.value?.trim() || "";
      if (!modelName || !modelVersion) {
        const message = "Model name and version are required.";
        if (tsRunError) tsRunError.textContent = message;
        showToast(message);
        return;
      }
      try {
        if (tsRunError) tsRunError.textContent = "--";
        const payload = {
          name: modelName,
          version: modelVersion,
          stage: tsModelStage?.value || "candidate",
          dataset_id: tsDatasetSelect.value,
          model_spec: {
            algorithm: "hdbscan",
            embedding_source_url: tsModelEmbeddingUrl?.value?.trim() || "",
            min_cluster_size: Number(tsModelMinCluster?.value || 15),
            metric: tsModelMetric?.value?.trim() || "cosine",
            params: parseJsonInput(tsModelParams?.value || "", {}),
          },
          windowing_spec: buildWindowingSpec(),
          metadata: {},
        };
        console.log("[TopicStudio] create model payload", payload);
        const response = await topicFoundryFetch("/models", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        console.log("[TopicStudio] create model response", response);
        showToast("Model created.");
        await refreshTopicStudio();
      } catch (err) {
        console.error("[TopicStudio] create model failed", err);
        renderTopicStudioError(tsRunError, err, "Create model", "train", { request: "/models" });
      }
    });
  }

  if (tsUsePreviewSpec) {
    tsUsePreviewSpec.addEventListener("click", async () => {
      if (!topicStudioLastPreview) {
        showToast("Run a preview before using the preview spec.");
        return;
      }
      if (!tsModelName?.value?.trim() || !tsModelVersion?.value?.trim()) {
        showToast("Enter a model name and version before creating a model.");
        return;
      }
      try {
        const datasetResponse = await topicFoundryFetch("/datasets", {
          method: "POST",
          body: JSON.stringify(topicStudioLastPreview.dataset),
        });
        const datasetId = datasetResponse?.dataset_id;
        if (!datasetId) {
          showToast("Dataset creation failed.");
          return;
        }
        const modelPayload = {
          name: tsModelName?.value?.trim() || "",
          version: tsModelVersion?.value?.trim() || "",
          stage: tsModelStage?.value || "candidate",
          dataset_id: datasetId,
          model_spec: {
            algorithm: "hdbscan",
            embedding_source_url: tsModelEmbeddingUrl?.value?.trim() || "",
            min_cluster_size: Number(tsModelMinCluster?.value || 15),
            metric: tsModelMetric?.value?.trim() || "cosine",
            params: parseJsonInput(tsModelParams?.value || "", {}),
          },
          windowing_spec: topicStudioLastPreview.windowing,
          metadata: {},
        };
        await topicFoundryFetch("/models", {
          method: "POST",
          body: JSON.stringify(modelPayload),
        });
        showToast("Preview spec applied. Dataset + model created.");
        await refreshTopicStudio();
        if (tsDatasetSelect) tsDatasetSelect.value = datasetId;
      } catch (err) {
        renderError(tsPreviewError, err, "Failed to apply preview spec.");
        showToast("Failed to use preview spec.");
      }
    });
  }

  if (tsPromoteModel) {
    tsPromoteModel.addEventListener("click", async () => {
      if (!tsPromoteModelSelect?.value) {
        showToast("Select a model to promote.");
        return;
      }
      try {
        await topicFoundryFetch(`/models/${tsPromoteModelSelect.value}/promote`, {
          method: "POST",
          body: JSON.stringify({
            stage: tsPromoteStage?.value || "candidate",
            reason: tsPromoteReason?.value?.trim() || null,
          }),
        });
        showToast("Model stage updated.");
        await refreshTopicStudio();
      } catch (err) {
        showToast("Failed to promote model.");
      }
    });
  }

  if (tsTrainRun) {
    tsTrainRun.addEventListener("click", async () => {
      const modelId = tsTrainModelSelect?.value;
      const model = topicStudioModels.find((entry) => entry.model_id === modelId);
      if (!model) {
        showToast("Select a model to train.");
        return;
      }
      if (boundaryColumnRequired() && !tsDatasetBoundaryColumn?.value) {
        setWarning(tsRunWarning, "Boundary column required. Select a boundary column and save the dataset before training.");
        return;
      }
      try {
        setLoading(tsRunLoading, true);
        const requestPayload = {
          model_id: model.model_id,
          dataset_id: model.dataset_id,
          start_at: parseDateInput(tsStartAt?.value),
          end_at: parseDateInput(tsEndAt?.value),
          windowing_spec: buildWindowingSpec(),
        };
        recordTopicStudioDebug("train", { request: "/runs/train", payload: requestPayload });
        const result = await topicFoundryFetch("/runs/train", {
          method: "POST",
          body: JSON.stringify(requestPayload),
        });
        recordTopicStudioDebug("train", { request: "/runs/train", response: result });
        if (tsRunId) tsRunId.value = result.run_id;
        if (tsRunError) tsRunError.textContent = "--";
        renderRunStatus(result);
        startRunPolling(result.run_id);
        await refreshTopicStudio();
      } catch (err) {
        if (tsRunError) tsRunError.textContent = formatHttpError("train", err);
        renderTopicStudioError(tsRunError, err, "Train run", "train", { request: "/runs/train" });
        setLoading(tsRunLoading, false);
      }
    });
  }

  if (tsPollRun) {
    tsPollRun.addEventListener("click", async () => {
      if (!tsRunId?.value) {
        showToast("Enter a run id to poll.");
        return;
      }
      try {
        setLoading(tsRunLoading, true);
        const result = await topicFoundryFetch(`/runs/${tsRunId.value}`);
        renderRunStatus(result);
        await refreshTopicStudio();
        setLoading(tsRunLoading, false);
      } catch (err) {
        renderTopicStudioError(tsRunError, err, "Poll run", "train", { request: `/runs/${tsRunId?.value}` });
        setLoading(tsRunLoading, false);
      }
    });
  }

  if (tsEnrichRun) {
    tsEnrichRun.addEventListener("click", async () => {
      if (!tsRunId?.value) {
        showToast("Enter a run id to enrich.");
        return;
      }
      try {
        setLoading(tsEnrichLoading, true);
        topicStudioEnrichPolling = true;
        const payload = {
          enricher: tsEnrichEnricher?.value || "heuristic",
          force: Boolean(tsEnrichForce?.checked),
        };
        recordTopicStudioDebug("enrich", { request: `/runs/${tsRunId.value}/enrich`, payload });
        const result = await topicFoundryFetch(`/runs/${tsRunId.value}/enrich`, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        recordTopicStudioDebug("enrich", { request: `/runs/${tsRunId.value}/enrich`, response: result });
        if (tsEnrichStatus) {
          tsEnrichStatus.textContent = `enriched=${result.enriched_count ?? 0} failed=${result.failed_count ?? 0}`;
        }
        const enrichedCount = Number(result.enriched_count);
        if (!Number.isNaN(enrichedCount) && enrichedCount === 0) {
          setWarning(tsEnrichWarning, "No segments enriched yet. Verify enricher configuration or run after training completes.");
        } else {
          setWarning(tsEnrichWarning, null);
        }
        startRunPolling(tsRunId.value);
      } catch (err) {
        const errorMsg = formatHttpError("enrich", err);
        if (tsEnrichStatus) tsEnrichStatus.textContent = errorMsg;
        renderTopicStudioError(tsEnrichStatus, err, "Enrich run", "enrich", { request: `/runs/${tsRunId?.value}/enrich` });
        setWarning(tsEnrichWarning, null);
        setLoading(tsEnrichLoading, false);
        topicStudioEnrichPolling = false;
      }
    });
  }

  function renderSegmentsTable(segments) {
    if (!tsSegmentsTableBody) return;
    tsSegmentsTableBody.innerHTML = "";
    if (!segments || segments.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="13">0 segments returned.</td>';
      tsSegmentsTableBody.appendChild(row);
      return;
    }
    segments.forEach((segment) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      const sentiment = segment.sentiment || {};
      const aspects = Array.isArray(segment.aspects) ? segment.aspects : [];
      const aspectChips = aspects.length
        ? aspects.map((aspect) => `<span class="bg-gray-800 text-gray-200 px-2 py-0.5 rounded-full text-[10px]">${aspect}</span>`).join(" ")
        : "--";
      const startAt = segment.start_at || "--";
      const endAt = segment.end_at || "--";
      const snippet = truncate(segment.snippet || "", 160) || "--";
      const rowIdsCount = segment.row_ids_count ?? "--";
      const size = segment.size ?? rowIdsCount ?? "--";
      const chars = segment.chars ?? "--";
      const topicId = segment.topic_id ?? "--";
      const topicProbValue = segment.topic_prob;
      const topicProb = Number.isFinite(topicProbValue) ? Number(topicProbValue).toFixed(3) : topicProbValue ?? "--";
      const outlier = segment.is_outlier === true ? "yes" : segment.is_outlier === false ? "no" : "--";
      const segmentId = segment.segment_id || "--";
      const shortId = segmentId !== "--" && segmentId.length > 10 ? `${segmentId.slice(0, 10)}…` : segmentId;
      row.innerHTML = `
        <td class="py-2 pr-3 text-indigo-300 cursor-pointer" data-segment-id="${segmentId}" title="${segmentId}">${shortId}</td>
        <td class="py-2 pr-3">${size}</td>
        <td class="py-2 pr-3">${rowIdsCount}</td>
        <td class="py-2 pr-3">${chars}</td>
        <td class="py-2 pr-3">${segment.title || segment.label || "--"}</td>
        <td class="py-2 pr-3">${aspectChips}</td>
        <td class="py-2 pr-3">${startAt}</td>
        <td class="py-2 pr-3">${endAt}</td>
        <td class="py-2 pr-3">${snippet}</td>
        <td class="py-2 pr-3">${topicId}</td>
        <td class="py-2 pr-3">${topicProb}</td>
        <td class="py-2 pr-3">${outlier}</td>
        <td class="py-2 pr-3">${sentiment.friction ?? "--"}</td>
      `;
      row.querySelector("[data-segment-id]")?.addEventListener("click", async () => {
        try {
          topicStudioSelectedSegmentId = segment.segment_id;
          const detail = await topicFoundryFetch(`/segments/${segment.segment_id}?include_full_text=true`);
          renderSegmentDetailPanel(detail, { include_full_text: true });
          recordTopicStudioDebug("segments", { request: `/segments/${segment.segment_id}?include_full_text=true`, response: detail });
        } catch (err) {
          renderTopicStudioError(tsSegmentsError, err, "Load segment detail", "segments", {
            request: `/segments/${segment.segment_id}?include_full_text=true`,
          });
        }
      });
      tsSegmentsTableBody.appendChild(row);
    });
  }

  async function showSegmentDetailFromId(segmentId) {
    if (!segmentId) return;
    try {
      const detail = await topicFoundryFetch(`/segments/${segmentId}?include_full_text=true`);
      renderSegmentDetailPanel(detail, { include_full_text: true });
      recordTopicStudioDebug("segments", { request: `/segments/${segmentId}?include_full_text=true`, response: detail });
      setTopicStudioSubview("runs");
    } catch (err) {
      renderTopicStudioError(tsSegmentsError, err, "Load segment detail", "segments", {
        request: `/segments/${segmentId}?include_full_text=true`,
      });
    }
  }

  function renderTopicsTable(topics) {
    if (!tsTopicsTableBody) return;
    tsTopicsTableBody.innerHTML = "";
    if (!topics || topics.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="4">No topics found.</td>';
      tsTopicsTableBody.appendChild(row);
      return;
    }
    topics.forEach((topic) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      const pct = Number.isFinite(topic.outlier_pct) ? `${(topic.outlier_pct * 100).toFixed(1)}%` : "--";
      row.innerHTML = `
        <td class="py-2 pr-3 text-indigo-300">${topic.topic_id ?? "--"}</td>
        <td class="py-2 pr-3">${topic.scope ?? "--"}</td>
        <td class="py-2 pr-3">${topic.count ?? "--"}</td>
        <td class="py-2 pr-3">${pct}</td>
        <td class="py-2 pr-3">${topic.parent_topic_id ?? "--"}</td>
        <td class="py-2 pr-3"><button class="text-indigo-300 hover:text-indigo-200" data-topic-id="${topic.topic_id}">View</button></td>
      `;
      row.querySelector("[data-topic-id]")?.addEventListener("click", () => {
        loadTopicDetails(topic.topic_id);
      });
      tsTopicsTableBody.appendChild(row);
    });
  }

  function renderTopicKeywords(keywords) {
    if (!tsTopicKeywords) return;
    if (!keywords || keywords.length === 0) {
      tsTopicKeywords.textContent = "--";
      return;
    }
    tsTopicKeywords.innerHTML = keywords
      .map((keyword) => `<span class="bg-gray-800 text-gray-200 px-2 py-0.5 rounded-full text-[10px]">${keyword}</span>`)
      .join(" ");
  }

  function renderTopicSegmentsTable(segments) {
    if (!tsTopicSegmentsTableBody) return;
    tsTopicSegmentsTableBody.innerHTML = "";
    if (!segments || segments.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="5">No segments found.</td>';
      tsTopicSegmentsTableBody.appendChild(row);
      return;
    }
    segments.forEach((segment) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      const snippet = truncate(segment.snippet || "", 160) || "--";
      const startAt = segment.start_at || "--";
      const endAt = segment.end_at || "--";
      const bounds = startAt === "--" && endAt === "--" ? "--" : `${startAt} → ${endAt}`;
      const rowIdsCount = segment.row_ids_count ?? segment.size ?? "--";
      row.innerHTML = `
        <td class="py-2 pr-3">${segment.segment_id ?? "--"}</td>
        <td class="py-2 pr-3">${rowIdsCount}</td>
        <td class="py-2 pr-3">${segment.title || segment.label || "--"}</td>
        <td class="py-2 pr-3">${bounds}</td>
        <td class="py-2 pr-3">${snippet}</td>
      `;
      tsTopicSegmentsTableBody.appendChild(row);
    });
  }

  function renderDriftTable(records) {
    if (!tsDriftTableBody) return;
    tsDriftTableBody.innerHTML = "";
    if (!records || records.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="6">No drift records found.</td>';
      tsDriftTableBody.appendChild(row);
      return;
    }
    records.forEach((record) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      const created = record.created_at || "--";
      const js = record.js_divergence ?? "--";
      const outlierPct = record.outlier_pct !== undefined && record.outlier_pct !== null ? `${(record.outlier_pct * 100).toFixed(1)}%` : "--";
      const outlierDelta = record.outlier_pct_delta !== undefined && record.outlier_pct_delta !== null ? `${(record.outlier_pct_delta * 100).toFixed(1)}%` : "--";
      const topDelta = record.top_topic_share_delta !== undefined && record.top_topic_share_delta !== null ? `${(record.top_topic_share_delta * 100).toFixed(1)}%` : "--";
      const thresholdJs = record.threshold_js ?? "--";
      const thresholdOutlier = record.threshold_outlier ?? "--";
      row.innerHTML = `
        <td class="py-2 pr-3">${created}</td>
        <td class="py-2 pr-3">${js}</td>
        <td class="py-2 pr-3">${outlierPct}</td>
        <td class="py-2 pr-3">${outlierDelta}</td>
        <td class="py-2 pr-3">${topDelta}</td>
        <td class="py-2 pr-3">${thresholdJs} / ${thresholdOutlier}</td>
      `;
      tsDriftTableBody.appendChild(row);
    });
  }

  async function loadDriftRecords() {
    if (!tsDriftModelName?.value) {
      showToast("Enter a model name to load drift.");
      return;
    }
    try {
      if (tsDriftStatus) tsDriftStatus.textContent = "Loading...";
      renderError(tsDriftError, null);
      const params = new URLSearchParams({
        model_name: tsDriftModelName.value,
        limit: tsDriftLimit?.value || "50",
      });
      const response = await topicFoundryFetch(`/drift?${params.toString()}`);
      renderDriftTable(response.records || []);
      if (tsDriftStatus) {
        tsDriftStatus.textContent = `Loaded ${response.records?.length || 0} drift records.`;
      }
    } catch (err) {
      renderError(tsDriftError, err);
      if (tsDriftStatus) tsDriftStatus.textContent = "Failed to load drift.";
    }
  }

  async function pollForDriftRecord(modelName, driftId) {
    if (topicStudioDriftPolling) return;
    topicStudioDriftPolling = true;
    const maxAttempts = 8;
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      try {
        const params = new URLSearchParams({
          model_name: modelName,
          limit: tsDriftLimit?.value || "50",
        });
        const response = await topicFoundryFetch(`/drift?${params.toString()}`);
        const records = response.records || [];
        renderDriftTable(records);
        const found = records.find((row) => row.drift_id === driftId);
        if (found) {
          if (tsDriftStatus) tsDriftStatus.textContent = "Drift record created.";
          topicStudioDriftPolling = false;
          return;
        }
      } catch (err) {
        renderError(tsDriftError, err);
      }
      if (tsDriftStatus) tsDriftStatus.textContent = "Waiting for drift record...";
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    topicStudioDriftPolling = false;
    if (tsDriftStatus) tsDriftStatus.textContent = "Drift record not found yet.";
  }

  async function runDriftNow() {
    if (!tsDriftModelName?.value) {
      showToast("Enter a model name to run drift.");
      return;
    }
    try {
      if (tsDriftStatus) tsDriftStatus.textContent = "Running drift...";
      renderError(tsDriftError, null);
      const payload = {
        model_name: tsDriftModelName.value,
        window_hours: Number(tsDriftWindowHours?.value || 24),
      };
      const response = await topicFoundryFetch("/drift/run", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (response?.drift_id) {
        await pollForDriftRecord(tsDriftModelName.value, response.drift_id);
      }
    } catch (err) {
      renderError(tsDriftError, err);
      if (tsDriftStatus) tsDriftStatus.textContent = "Drift run failed.";
    }
  }

  function renderEventsTable(items) {
    if (!tsEventsTableBody) return;
    tsEventsTableBody.innerHTML = "";
    if (!items || items.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="5">No events found.</td>';
      tsEventsTableBody.appendChild(row);
      return;
    }
    items.forEach((event) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      row.innerHTML = `
        <td class="py-2 pr-3">${event.created_at || "--"}</td>
        <td class="py-2 pr-3">${event.kind || "--"}</td>
        <td class="py-2 pr-3">${event.run_id || "--"}</td>
        <td class="py-2 pr-3">${event.model_id || "--"}</td>
        <td class="py-2 pr-3">${event.bus_status || "--"}</td>
      `;
      tsEventsTableBody.appendChild(row);
    });
  }

  async function loadEvents() {
    try {
      if (tsEventsStatus) tsEventsStatus.textContent = "Loading...";
      renderError(tsEventsError, null);
      const params = new URLSearchParams({
        limit: tsEventsLimit?.value || "50",
        offset: tsEventsOffset?.value || "0",
      });
      if (tsEventsKind?.value) {
        const kindValue = tsEventsKind.value.replaceAll("_", ".");
        params.set("kind", kindValue);
      }
      const response = await topicFoundryFetch(`/events?${params.toString()}`);
      const items = asItems(response);
      topicStudioEventsPage = items.length > 0 ? items : Array.isArray(response?.events) ? response.events : [];
      renderEventsTable(topicStudioEventsPage);
      if (tsEventsStatus) tsEventsStatus.textContent = `Loaded ${topicStudioEventsPage.length} events.`;
    } catch (err) {
      renderError(tsEventsError, err);
      if (tsEventsStatus) tsEventsStatus.textContent = "Failed to load events.";
    }
  }

  function exportEventsCsv() {
    if (!topicStudioEventsPage.length) {
      showToast("No events to export.");
      return;
    }
    const headers = ["event_id", "kind", "run_id", "model_id", "drift_id", "bus_status", "created_at"];
    const lines = [headers.join(",")];
    const escapeCsv = (value) => {
      if (value === null || value === undefined) return "";
      const str = String(value).replace(/"/g, "\"\"");
      return `"${str}"`;
    };
    topicStudioEventsPage.forEach((event) => {
      const row = [
        escapeCsv(event.event_id),
        escapeCsv(event.kind),
        escapeCsv(event.run_id || ""),
        escapeCsv(event.model_id || ""),
        escapeCsv(event.drift_id || ""),
        escapeCsv(event.bus_status || ""),
        escapeCsv(event.created_at || ""),
      ];
      lines.push(row.join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "topic_foundry_events.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function renderKgTable(items) {
    if (!tsKgTableBody) return;
    tsKgTableBody.innerHTML = "";
    if (!items || items.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="5">No edges found.</td>';
      tsKgTableBody.appendChild(row);
      return;
    }
    items.forEach((edge) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      row.innerHTML = `
        <td class="py-2 pr-3">${edge.subject || "--"}</td>
        <td class="py-2 pr-3">${edge.predicate || "--"}</td>
        <td class="py-2 pr-3">${edge.object || "--"}</td>
        <td class="py-2 pr-3">${edge.confidence ?? "--"}</td>
        <td class="py-2 pr-3 text-indigo-300 cursor-pointer" data-segment-id="${edge.segment_id}">${edge.segment_id || "--"}</td>
      `;
      row.querySelector("[data-segment-id]")?.addEventListener("click", () => {
        showSegmentDetailFromId(edge.segment_id);
      });
      tsKgTableBody.appendChild(row);
    });
  }

  async function loadKgEdges() {
    if (!tsKgRunId?.value) {
      showToast("Enter a run id to load edges.");
      return;
    }
    try {
      if (tsKgStatus) tsKgStatus.textContent = "Loading...";
      renderError(tsKgError, null);
      const params = new URLSearchParams({
        run_id: tsKgRunId.value,
        limit: tsKgLimit?.value || "100",
        offset: tsKgOffset?.value || "0",
      });
      if (tsKgPredicate?.value) {
        params.set("predicate", tsKgPredicate.value);
      }
      if (tsKgQuery?.value) {
        params.set("q", tsKgQuery.value);
      }
      const response = await topicFoundryFetch(`/kg/edges?${params.toString()}`);
      const items = asItems(response);
      topicStudioKgEdgesPage = items.length > 0 ? items : Array.isArray(response?.edges) ? response.edges : [];
      renderKgTable(topicStudioKgEdgesPage);
      if (tsKgStatus) tsKgStatus.textContent = `Loaded ${topicStudioKgEdgesPage.length} edges.`;
    } catch (err) {
      renderError(tsKgError, err);
      if (tsKgStatus) tsKgStatus.textContent = "Failed to load edges.";
    }
  }

  function exportKgCsv() {
    if (!topicStudioKgEdgesPage.length) {
      showToast("No edges to export.");
      return;
    }
    const headers = ["edge_id", "segment_id", "subject", "predicate", "object", "confidence", "created_at"];
    const lines = [headers.join(",")];
    const escapeCsv = (value) => {
      if (value === null || value === undefined) return "";
      const str = String(value).replace(/"/g, "\"\"");
      return `"${str}"`;
    };
    topicStudioKgEdgesPage.forEach((edge) => {
      const row = [
        escapeCsv(edge.edge_id),
        escapeCsv(edge.segment_id),
        escapeCsv(edge.subject),
        escapeCsv(edge.predicate),
        escapeCsv(edge.object),
        escapeCsv(edge.confidence ?? ""),
        escapeCsv(edge.created_at || ""),
      ];
      lines.push(row.join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "topic_foundry_kg_edges.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  async function loadTopicExplorer() {
    if (!tsTopicsRunId?.value) {
      showToast("Enter a run id to load topics.");
      return;
    }
    try {
      if (tsTopicsStatus) tsTopicsStatus.textContent = "Loading...";
      renderError(tsTopicsError, null);
      topicStudioSelectedTopicId = null;
      topicStudioTopicSegmentsOffset = 0;
      if (tsTopicSelectedId) tsTopicSelectedId.textContent = "--";
      if (tsTopicSegmentsStatus) tsTopicSegmentsStatus.textContent = "--";
      if (tsTopicSegmentsOffset) tsTopicSegmentsOffset.value = "0";
      renderTopicKeywords([]);
      renderTopicSegmentsTable([]);
      const params = new URLSearchParams({
        run_id: tsTopicsRunId.value,
        limit: tsTopicsLimit?.value || "200",
        offset: tsTopicsOffset?.value || "0",
      });
      if (tsTopicsScope?.value) {
        params.set("scope", tsTopicsScope.value);
      }
      const response = await topicFoundryFetch(`/topics?${params.toString()}`);
      const items = asItems(response);
      const topics = items.length > 0 ? items : Array.isArray(response?.topics) ? response.topics : [];
      renderTopicsTable(topics);
      if (tsTopicsStatus) {
        tsTopicsStatus.textContent = `Loaded ${topics.length} topics.`;
      }
    } catch (err) {
      renderError(tsTopicsError, err);
      if (tsTopicsStatus) tsTopicsStatus.textContent = "Failed to load topics.";
    }
  }

  async function loadTopicDetails(topicId) {
    if (!tsTopicsRunId?.value) {
      showToast("Enter a run id to load topic details.");
      return;
    }
    if (!Number.isFinite(Number(topicId))) {
      showToast("Select a topic to load details.");
      return;
    }
    topicStudioSelectedTopicId = topicId;
    if (tsTopicSelectedId) {
      tsTopicSelectedId.textContent = `Topic ${topicId}`;
    }
    if (tsTopicSegmentsStatus) {
      tsTopicSegmentsStatus.textContent = "Loading...";
    }
    try {
      const keywordsResponse = await topicFoundryFetch(`/topics/${topicId}/keywords?run_id=${tsTopicsRunId.value}`);
      renderTopicKeywords(keywordsResponse.keywords || []);
    } catch (err) {
      renderTopicKeywords([]);
    }
    try {
      const limit = tsTopicSegmentsLimit?.value || "50";
      topicStudioTopicSegmentsOffset = Math.max(0, Number(tsTopicSegmentsOffset?.value || 0));
      if (tsTopicSegmentsOffset) {
        tsTopicSegmentsOffset.value = String(topicStudioTopicSegmentsOffset);
      }
      const params = new URLSearchParams({
        run_id: tsTopicsRunId.value,
        limit,
        offset: String(topicStudioTopicSegmentsOffset),
        include_snippet: "true",
        include_bounds: "true",
      });
      const response = await topicFoundryFetch(`/topics/${topicId}/segments?${params.toString()}`);
      const items = asItems(response);
      const segments = items.length > 0 ? items : Array.isArray(response?.segments) ? response.segments : [];
      renderTopicSegmentsTable(segments);
      if (tsTopicSegmentsStatus) {
        const rangeStart = segments.length === 0 ? 0 : topicStudioTopicSegmentsOffset + 1;
        const rangeEnd = topicStudioTopicSegmentsOffset + segments.length;
        tsTopicSegmentsStatus.textContent = `Loaded ${segments.length} segments (${rangeStart}–${rangeEnd}).`;
      }
    } catch (err) {
      if (tsTopicSegmentsStatus) tsTopicSegmentsStatus.textContent = "Failed to load segments.";
      renderTopicSegmentsTable([]);
    }
  }

  function updateTopicSegmentsOffset(nextOffset) {
    topicStudioTopicSegmentsOffset = Math.max(0, nextOffset);
    if (tsTopicSegmentsOffset) {
      tsTopicSegmentsOffset.value = String(topicStudioTopicSegmentsOffset);
    }
    saveTopicStudioState();
    if (topicStudioSelectedTopicId !== null) {
      loadTopicDetails(topicStudioSelectedTopicId);
    }
  }

  function renderCompareTable(diffs, leftStats, rightStats) {
    if (!tsCompareTableBody) return;
    tsCompareTableBody.innerHTML = "";
    if (!diffs || Object.keys(diffs).length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="4">No comparison data.</td>';
      tsCompareTableBody.appendChild(row);
      return;
    }
    Object.keys(diffs).forEach((key) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      const leftValue = leftStats?.[key] ?? "--";
      const rightValue = rightStats?.[key] ?? "--";
      const deltaValue = diffs[key] ?? "--";
      row.innerHTML = `
        <td class="py-2 pr-3">${key}</td>
        <td class="py-2 pr-3">${leftValue}</td>
        <td class="py-2 pr-3">${rightValue}</td>
        <td class="py-2 pr-3">${deltaValue}</td>
      `;
      tsCompareTableBody.appendChild(row);
    });
  }

  function renderCompareAspects(aspects) {
    if (!tsCompareAspectBody) return;
    tsCompareAspectBody.innerHTML = "";
    if (!aspects || aspects.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="4">No aspect diffs.</td>';
      tsCompareAspectBody.appendChild(row);
      return;
    }
    aspects.forEach((rowData) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-gray-800/40";
      row.innerHTML = `
        <td class="py-2 pr-3">${rowData.aspect ?? "--"}</td>
        <td class="py-2 pr-3">${rowData.left_count ?? "--"}</td>
        <td class="py-2 pr-3">${rowData.right_count ?? "--"}</td>
        <td class="py-2 pr-3">${rowData.delta ?? "--"}</td>
      `;
      tsCompareAspectBody.appendChild(row);
    });
  }

  function renderCompareSummary(leftStats, rightStats, diffs) {
    const formatDelta = (value) => {
      if (value === null || value === undefined) return "--";
      return Number.isFinite(value) ? value : value;
    };
    const formatStat = (value, fallback = "--") => (value === null || value === undefined ? fallback : value);
    const formatPct = (value) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return `${(Number(value) * 100).toFixed(1)}%`;
    };
    const docsLeft = formatStat(leftStats?.docs_generated);
    const docsRight = formatStat(rightStats?.docs_generated);
    const segmentsLeft = formatStat(leftStats?.segments_generated);
    const segmentsRight = formatStat(rightStats?.segments_generated);
    const clustersLeft = formatStat(leftStats?.cluster_count);
    const clustersRight = formatStat(rightStats?.cluster_count);
    const outliersLeft = formatPct(leftStats?.outlier_pct);
    const outliersRight = formatPct(rightStats?.outlier_pct);
    const docsDelta = formatDelta(diffs?.docs_generated);
    const segmentsDelta = formatDelta(diffs?.segments_generated);
    const clustersDelta = formatDelta(diffs?.cluster_count);
    const outliersDelta = formatDelta(diffs?.outlier_pct);
    if (tsCompareDocs) tsCompareDocs.textContent = `${docsLeft} / ${docsRight} (Δ ${docsDelta})`;
    if (tsCompareSegments) tsCompareSegments.textContent = `${segmentsLeft} / ${segmentsRight} (Δ ${segmentsDelta})`;
    if (tsCompareClusters) tsCompareClusters.textContent = `${clustersLeft} / ${clustersRight} (Δ ${clustersDelta})`;
    if (tsCompareOutliers) {
      const deltaValue = Number(outliersDelta);
      const outlierDeltaText = Number.isFinite(deltaValue) ? `${(deltaValue * 100).toFixed(1)}%` : "--";
      tsCompareOutliers.textContent = `${outliersLeft} / ${outliersRight} (Δ ${outlierDeltaText})`;
    }
  }

  async function loadRunCompare() {
    if (!tsCompareLeftRunId?.value || !tsCompareRightRunId?.value) {
      showToast("Enter both run ids to compare.");
      return;
    }
    try {
      if (tsCompareStatus) tsCompareStatus.textContent = "Loading...";
      renderError(tsCompareError, null);
      const params = new URLSearchParams({
        left_run_id: tsCompareLeftRunId.value,
        right_run_id: tsCompareRightRunId.value,
      });
      const response = await topicFoundryFetch(`/runs/compare?${params.toString()}`);
      renderCompareTable(response.diffs || {}, response.left_stats || {}, response.right_stats || {});
      renderCompareAspects(response.aspect_diffs || []);
      renderCompareSummary(response.left_stats || {}, response.right_stats || {}, response.diffs || {});
      if (tsCompareStatus) tsCompareStatus.textContent = "Comparison loaded.";
    } catch (err) {
      renderError(tsCompareError, err);
      if (tsCompareStatus) tsCompareStatus.textContent = "Failed to compare runs.";
      renderCompareTable({}, {}, {});
      renderCompareAspects([]);
      renderCompareSummary({}, {}, {});
    }
  }

  async function loadSegments() {
    if (!tsSegmentsRunId?.value) {
      showToast("Enter a run id to load segments.");
      return;
    }
    if (topicStudioSegmentsPolling) {
      return;
    }
    topicStudioSegmentsPolling = true;
    const qs = new URLSearchParams();
    try {
      setLoading(tsSegmentsLoading, true);
      if (tsSegmentsError) tsSegmentsError.textContent = "--";
      const queryKey = `${tsSegmentsRunId.value}|${tsSegmentsEnrichment?.value || ""}|${tsSegmentsAspect?.value || ""}|${topicStudioSegmentsLimit}`;
      if (topicStudioSegmentsQueryKey !== queryKey) {
        topicStudioSegmentsQueryKey = queryKey;
        resetSegmentsPaging();
      }
      topicStudioSegmentsOffset = 0;
      qs.set("run_id", tsSegmentsRunId.value);
      qs.set("include_snippet", "true");
      qs.set("include_bounds", "true");
      qs.set("limit", "50");
      qs.set("offset", "0");
      qs.set("format", "wrapped");
      qs.set("sort_by", "created_at");
      qs.set("sort_dir", "desc");
      const enrichmentValue = tsSegmentsEnrichment?.value;
      if (enrichmentValue && enrichmentValue !== "all") {
        qs.set("has_enrichment", enrichmentValue);
      }
      const aspectValue = tsSegmentsAspect?.value?.trim();
      if (aspectValue && aspectValue !== "all") {
        qs.set("aspect", aspectValue);
      }
      const query = tsSegmentsSearch?.value?.trim();
      if (query) {
        qs.set("q", query);
      }
      const { payload } = await topicFoundryFetchWithHeaders(`/segments?${qs.toString()}`);
      const items = asItems(payload);
      const segments = items;
      topicStudioSegmentsPage = segments;
      topicStudioSegmentsTotal = getTotal(payload, items);
      if (HUB_DEBUG) {
        const keys = Object.keys(payload || {});
        console.log("segments keys", keys, "items", items.length);
        const debugEl = document.getElementById("tsSegmentsDebug");
        if (debugEl) {
          debugEl.textContent = `segments response keys: ${keys.join(", ") || "--"}`;
          debugEl.classList.remove("hidden");
        }
        recordTopicStudioDebug("segments", {
          request: `/segments?${qs.toString()}`,
          response_keys: keys,
          items: items.length,
        });
      }
      const filtered = applySegmentsClientFilters(topicStudioSegmentsPage);
      topicStudioSegmentsDisplayed = filtered;
      renderSegmentsTable(filtered);
      renderSegmentsFacets(topicStudioSegmentsLastFacets);
      topicStudioLastSubview = "runs";
      saveTopicStudioState();
      updateSegmentsRange();
      refreshSegmentFacets();
    } catch (err) {
      renderTopicStudioError(tsSegmentsError, err, "Load segments", "segments", {
        request: `/segments?${qs.toString()}`,
      });
    } finally {
      setLoading(tsSegmentsLoading, false);
      topicStudioSegmentsPolling = false;
    }
  }

  async function refreshSegmentFacets() {
    if (!tsSegmentsRunId?.value) {
      return;
    }
    try {
      const params = new URLSearchParams({ run_id: tsSegmentsRunId.value });
      const query = tsSegmentsSearch?.value?.trim();
      if (query) {
        params.set("q", query);
      }
      if (tsSegmentsEnrichment?.value) {
        params.set("has_enrichment", tsSegmentsEnrichment.value);
      }
      if (tsSegmentsAspect?.value) {
        params.set("aspect", tsSegmentsAspect.value);
      }
      const facets = await topicFoundryFetch(`/segments/facets?${params.toString()}`);
      if (!facets || facets.ok === false) {
        topicStudioSegmentsLastFacets = null;
        renderSegmentsFacets(null);
        return;
      }
      topicStudioSegmentsLastFacets = facets;
      renderSegmentsFacets(facets);
    } catch (err) {
      console.warn("[TopicStudio] Failed to load segment facets", err);
      topicStudioSegmentsLastFacets = null;
      renderSegmentsFacets(null);
    }
  }

  if (tsLoadSegments) {
    tsLoadSegments.addEventListener("click", loadSegments);
  }

  if (tsSegmentsRefresh) {
    tsSegmentsRefresh.addEventListener("click", loadSegments);
  }

  const debouncedSegmentSearch = debounce(() => {
    resetSegmentsPaging();
    loadSegments();
    refreshSegmentFacets();
  }, 300);

  if (tsSegmentsSearch) {
    tsSegmentsSearch.addEventListener("input", () => {
      saveTopicStudioState();
      debouncedSegmentSearch();
    });
  }

  if (tsSegmentsSort) {
    tsSegmentsSort.addEventListener("change", () => {
      saveTopicStudioState();
      resetSegmentsPaging();
      loadSegments();
      refreshSegmentFacets();
    });
  }

  if (tsSegmentsAspect) {
    tsSegmentsAspect.addEventListener("change", () => {
      saveTopicStudioState();
      resetSegmentsPaging();
      loadSegments();
      refreshSegmentFacets();
    });
  }

  if (tsSegmentsEnrichment) {
    tsSegmentsEnrichment.addEventListener("change", () => {
      saveTopicStudioState();
      resetSegmentsPaging();
      loadSegments();
      refreshSegmentFacets();
    });
  }

  if (tsSegmentsPageSize) {
    topicStudioSegmentsLimit = Number(tsSegmentsPageSize.value || 50);
    tsSegmentsPageSize.addEventListener("change", () => {
      topicStudioSegmentsLimit = Number(tsSegmentsPageSize.value || 50);
      resetSegmentsPaging();
      loadSegments();
      refreshSegmentFacets();
    });
  }

  if (tsSegmentsPrev) {
    tsSegmentsPrev.addEventListener("click", () => {
      if (topicStudioSegmentsOffset <= 0) return;
      topicStudioSegmentsOffset = Math.max(0, topicStudioSegmentsOffset - topicStudioSegmentsLimit);
      loadSegments();
      refreshSegmentFacets();
    });
  }

  if (tsSegmentsNext) {
    tsSegmentsNext.addEventListener("click", () => {
      if (topicStudioSegmentsTotal !== null && topicStudioSegmentsOffset + topicStudioSegmentsLimit >= topicStudioSegmentsTotal) {
        return;
      }
      topicStudioSegmentsOffset += topicStudioSegmentsLimit;
      loadSegments();
      refreshSegmentFacets();
    });
  }

  if (tsSegmentsExport) {
    tsSegmentsExport.addEventListener("click", exportSegmentsCsv);
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
