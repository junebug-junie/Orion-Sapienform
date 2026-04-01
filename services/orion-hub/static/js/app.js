// services/orion-hub/static/js/app.js

// ───────────────────────────────────────────────────────────────
// Global State
// ───────────────────────────────────────────────────────────────
const pathSegments = window.location.pathname.split('/').filter(p => p.length > 0);
const URL_PREFIX = pathSegments.length > 0 ? `/${pathSegments[0]}` : "";
const API_BASE_URL = window.location.origin + URL_PREFIX;
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
let latestSocialInspectionState = null;
const socialInspectionCache = new Map();
let workflowSchedules = [];
let selectedSchedule = null;

document.addEventListener("DOMContentLoaded", () => {
  console.log("[Main] DOM Content Loaded - Initializing UI...");
  const hubUiVersion = (window.__HUB_UI_VERSION__ || document.body?.dataset?.uiVersion || "unknown").trim() || "unknown";
  console.log(`[HubUI] version=${hubUiVersion}`);

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
  const agentTraceDebugPanel = document.getElementById('agentTraceDebugPanel');
  const agentTraceDebugToggle = document.getElementById('agentTraceDebugToggle');
  const agentTraceDebugCaret = document.getElementById('agentTraceDebugCaret');
  const agentTraceDebugBody = document.getElementById('agentTraceDebugBody');
  const agentTraceDebugMeta = document.getElementById('agentTraceDebugMeta');
  const agentTraceDebugOverview = document.getElementById('agentTraceDebugOverview');
  const agentTraceDebugSummary = document.getElementById('agentTraceDebugSummary');
  const agentTraceDebugToolGroups = document.getElementById('agentTraceDebugToolGroups');
  const agentTraceDebugTimeline = document.getElementById('agentTraceDebugTimeline');
  const agentTraceDebugRaw = document.getElementById('agentTraceDebugRaw');
  const autonomyDebugPanel = document.getElementById('autonomyDebugPanel');
  const autonomyDebugToggle = document.getElementById('autonomyDebugToggle');
  const autonomyDebugCaret = document.getElementById('autonomyDebugCaret');
  const autonomyDebugBody = document.getElementById('autonomyDebugBody');
  const autonomyDebugMeta = document.getElementById('autonomyDebugMeta');
  const autonomyDebugOverview = document.getElementById('autonomyDebugOverview');
  const autonomyDebugState = document.getElementById('autonomyDebugState');
  const autonomyDebugProposals = document.getElementById('autonomyDebugProposals');
  const autonomyDebugAlignment = document.getElementById('autonomyDebugAlignment');
  const autonomyDebugRaw = document.getElementById('autonomyDebugRaw');
  const autonomyDebugOpenModal = document.getElementById('autonomyDebugOpenModal');
  const autonomyDebugModalRoot = document.getElementById('autonomyDebugModalRoot');
  const autonomyDebugModalBackdrop = document.getElementById('autonomyDebugModalBackdrop');
  const autonomyDebugModalDialog = document.getElementById('autonomyDebugModalDialog');
  const autonomyDebugModalClose = document.getElementById('autonomyDebugModalClose');
  const autonomyDebugModalMeta = document.getElementById('autonomyDebugModalMeta');
  const autonomyDebugModalBody = document.getElementById('autonomyDebugModalBody');
  const notificationList = document.getElementById('notificationList');
  const notificationFilter = document.getElementById('notificationFilter');
  const attentionList = document.getElementById('attentionList');
  const attentionCount = document.getElementById('attentionCount');
  const messageList = document.getElementById('messageList');
  const messageFilter = document.getElementById('messageFilter');
  const toastContainer = document.getElementById('toastContainer');
  const agentTraceApi = window.OrionAgentTrace || {};
  const socialInspectionApi = window.OrionSocialInspection || {};
  const workflowUiApi = window.OrionWorkflowUI || {};
  const scheduleUiApi = window.OrionWorkflowScheduleUI || {};
  const agentTraceModal = document.getElementById('agentTraceModal');
  const agentTraceModalClose = document.getElementById('agentTraceModalClose');
  const agentTraceModalMeta = document.getElementById('agentTraceModalMeta');
  const agentTraceEmptyState = document.getElementById('agentTraceEmptyState');
  const agentTraceContent = document.getElementById('agentTraceContent');
  const agentTraceOverview = document.getElementById('agentTraceOverview');
  const agentTraceSummary = document.getElementById('agentTraceSummary');
  const agentTraceToolGroups = document.getElementById('agentTraceToolGroups');
  const agentTraceTimelineBody = document.getElementById('agentTraceTimelineBody');
  const agentTraceRawSummary = document.getElementById('agentTraceRawSummary');
  const agentTraceRawPayloads = document.getElementById('agentTraceRawPayloads');
  const workflowModal = document.getElementById('workflowModal');
  const workflowModalClose = document.getElementById('workflowModalClose');
  const workflowModalTitle = document.getElementById('workflowModalTitle');
  const workflowModalMeta = document.getElementById('workflowModalMeta');
  const workflowModalBadges = document.getElementById('workflowModalBadges');
  const workflowModalSummary = document.getElementById('workflowModalSummary');
  const workflowModalDetailSurface = document.getElementById('workflowModalDetailSurface');
  const workflowModalRaw = document.getElementById('workflowModalRaw');
  const outboundRoutingDebug = document.getElementById('outboundRoutingDebug');
  const scheduleInventoryMeta = document.getElementById('scheduleInventoryMeta');
  const scheduleAttentionSummary = document.getElementById('scheduleAttentionSummary');
  const scheduleFilter = document.getElementById('scheduleFilter');
  const scheduleRefreshButton = document.getElementById('scheduleRefreshButton');
  const scheduleInventoryStatus = document.getElementById('scheduleInventoryStatus');
  const scheduleInventoryList = document.getElementById('scheduleInventoryList');
  const scheduleModal = document.getElementById('scheduleModal');
  const scheduleModalClose = document.getElementById('scheduleModalClose');
  const scheduleModalTitle = document.getElementById('scheduleModalTitle');
  const scheduleModalMeta = document.getElementById('scheduleModalMeta');
  const scheduleModalBadges = document.getElementById('scheduleModalBadges');
  const scheduleModalSummary = document.getElementById('scheduleModalSummary');
  const scheduleModalAnalytics = document.getElementById('scheduleModalAnalytics');
  const scheduleModalAnalyticsSummary = document.getElementById('scheduleModalAnalyticsSummary');
  const scheduleModalAnalyticsTrend = document.getElementById('scheduleModalAnalyticsTrend');
  const scheduleModalHistory = document.getElementById('scheduleModalHistory');
  const scheduleEditModal = document.getElementById('scheduleEditModal');
  const scheduleEditModalClose = document.getElementById('scheduleEditModalClose');
  const scheduleEditCadence = document.getElementById('scheduleEditCadence');
  const scheduleEditNotify = document.getElementById('scheduleEditNotify');
  const scheduleEditHour = document.getElementById('scheduleEditHour');
  const scheduleEditMinute = document.getElementById('scheduleEditMinute');
  const scheduleEditStatus = document.getElementById('scheduleEditStatus');
  const scheduleEditSave = document.getElementById('scheduleEditSave');
  const socialInspectionOpen = document.getElementById('socialInspectionOpen');
  const socialInspectionPanelStatus = document.getElementById('socialInspectionPanelStatus');
  const socialInspectionBadgeRow = document.getElementById('socialInspectionBadgeRow');
  const socialInspectionPanelSummary = document.getElementById('socialInspectionPanelSummary');
  const socialInspectionModal = document.getElementById('socialInspectionModal');
  const socialInspectionModalClose = document.getElementById('socialInspectionModalClose');
  const socialInspectionModalMeta = document.getElementById('socialInspectionModalMeta');
  const socialInspectionModalBadges = document.getElementById('socialInspectionModalBadges');
  const socialInspectionModalSummary = document.getElementById('socialInspectionModalSummary');
  const socialInspectionModalLoading = document.getElementById('socialInspectionModalLoading');
  const socialInspectionModalError = document.getElementById('socialInspectionModalError');
  const socialInspectionLiveSurface = document.getElementById('socialInspectionLiveSurface');
  const socialInspectionMemorySurface = document.getElementById('socialInspectionMemorySurface');
  const socialInspectionMemoryMeta = document.getElementById('socialInspectionMemoryMeta');

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
  const topicStudioTabButton = document.getElementById("topicStudioTabButton");
  const hubTabPanel = document.getElementById("hubTabPanel");
  const topicStudioPanel = document.getElementById("topicStudioPanel");
  const topicFoundryBaseLabel = document.getElementById("topicFoundryBaseLabel");
  const tsDatasetSelect = document.getElementById("tsDatasetSelect");
  const tsDatasetName = document.getElementById("tsDatasetName");
  const tsDatasetTable = document.getElementById("tsDatasetTable");
  const tsDatasetIdColumn = document.getElementById("tsDatasetIdColumn");
  const tsDatasetTimeColumn = document.getElementById("tsDatasetTimeColumn");
  const tsDatasetTextColumns = document.getElementById("tsDatasetTextColumns");
  const tsDatasetWhereSql = document.getElementById("tsDatasetWhereSql");
  const tsDatasetTimezone = document.getElementById("tsDatasetTimezone");
  const tsStartAt = document.getElementById("tsStartAt");
  const tsEndAt = document.getElementById("tsEndAt");
  const tsBlockMode = document.getElementById("tsBlockMode");
  const tsSegmentationMode = document.getElementById("tsSegmentationMode");
  const tsTimeGap = document.getElementById("tsTimeGap");
  const tsMaxWindow = document.getElementById("tsMaxWindow");
  const tsMinBlocks = document.getElementById("tsMinBlocks");
  const tsMaxChars = document.getElementById("tsMaxChars");
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
  const tsSegmentDetail = document.getElementById("tsSegmentDetail");
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
  const tsCapabilitiesWarning = document.getElementById("tsCapabilitiesWarning");
  const tsCopyReadyUrl = document.getElementById("tsCopyReadyUrl");
  const tsCopyCapabilitiesUrl = document.getElementById("tsCopyCapabilitiesUrl");
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

  const TOPIC_FOUNDRY_PROXY_BASE = `${API_BASE_URL}/api/topic-foundry`;
  const TOPIC_STUDIO_STATE_KEY = "topic_studio_state_v1";
  const MIN_PREVIEW_DOCS = 20;

  function setActiveTab(tabKey) {
    if (!hubTabPanel || !topicStudioPanel || !hubTabButton || !topicStudioTabButton) return;
    const isHub = tabKey === "hub";
    hubTabPanel.classList.toggle("hidden", !isHub);
    topicStudioPanel.classList.toggle("hidden", isHub);
    hubTabButton.classList.toggle("bg-indigo-600", isHub);
    hubTabButton.classList.toggle("text-white", isHub);
    hubTabButton.classList.toggle("border-indigo-500", isHub);
    hubTabButton.classList.toggle("bg-gray-800", !isHub);
    hubTabButton.classList.toggle("text-gray-200", !isHub);
    hubTabButton.classList.toggle("border-gray-700", !isHub);
    topicStudioTabButton.classList.toggle("bg-indigo-600", !isHub);
    topicStudioTabButton.classList.toggle("text-white", !isHub);
    topicStudioTabButton.classList.toggle("border-indigo-500", !isHub);
    topicStudioTabButton.classList.toggle("bg-gray-800", isHub);
    topicStudioTabButton.classList.toggle("text-gray-200", isHub);
    topicStudioTabButton.classList.toggle("border-gray-700", isHub);
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
        source_table: tsDatasetTable?.value || "",
        id_column: tsDatasetIdColumn?.value || "",
        time_column: tsDatasetTimeColumn?.value || "",
        text_columns: tsDatasetTextColumns?.value || "",
        where_sql: tsDatasetWhereSql?.value || "",
        timezone: tsDatasetTimezone?.value || "",
      },
      windowing: {
        block_mode: tsBlockMode?.value || "",
        segmentation_mode: tsSegmentationMode?.value || "",
        time_gap_seconds: tsTimeGap?.value || "",
        max_window_seconds: tsMaxWindow?.value || "",
        min_blocks_per_segment: tsMinBlocks?.value || "",
        max_chars: tsMaxChars?.value || "",
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
        if (tsDatasetTable && state.dataset.source_table) tsDatasetTable.value = state.dataset.source_table;
        if (tsDatasetIdColumn && state.dataset.id_column) tsDatasetIdColumn.value = state.dataset.id_column;
        if (tsDatasetTimeColumn && state.dataset.time_column) tsDatasetTimeColumn.value = state.dataset.time_column;
        if (tsDatasetTextColumns && state.dataset.text_columns) tsDatasetTextColumns.value = state.dataset.text_columns;
        if (tsDatasetWhereSql && state.dataset.where_sql) tsDatasetWhereSql.value = state.dataset.where_sql;
        if (tsDatasetTimezone && state.dataset.timezone) tsDatasetTimezone.value = state.dataset.timezone;
      }
      if (state.windowing) {
        if (tsBlockMode && state.windowing.block_mode) tsBlockMode.value = state.windowing.block_mode;
        if (tsSegmentationMode && state.windowing.segmentation_mode) tsSegmentationMode.value = state.windowing.segmentation_mode;
        if (tsTimeGap && state.windowing.time_gap_seconds) tsTimeGap.value = state.windowing.time_gap_seconds;
        if (tsMaxWindow && state.windowing.max_window_seconds) tsMaxWindow.value = state.windowing.max_window_seconds;
        if (tsMinBlocks && state.windowing.min_blocks_per_segment) tsMinBlocks.value = state.windowing.min_blocks_per_segment;
        if (tsMaxChars && state.windowing.max_chars) tsMaxChars.value = state.windowing.max_chars;
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
    topicStudioTopicSegmentsOffset = Number(tsTopicSegmentsOffset?.value || 0);
  }

  function bindTopicStudioPersistence() {
    const inputs = [
      tsDatasetName,
      tsDatasetTable,
      tsDatasetIdColumn,
      tsDatasetTimeColumn,
      tsDatasetTextColumns,
      tsDatasetWhereSql,
      tsDatasetTimezone,
      tsStartAt,
      tsEndAt,
      tsBlockMode,
      tsSegmentationMode,
      tsTimeGap,
      tsMaxWindow,
      tsMinBlocks,
      tsMaxChars,
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
    return {
      block_mode: tsBlockMode?.value || "turn_pairs",
      segmentation_mode: tsSegmentationMode?.value || "time_gap",
      time_gap_seconds: Number(tsTimeGap?.value || 900),
      max_window_seconds: Number(tsMaxWindow?.value || 7200),
      min_blocks_per_segment: Number(tsMinBlocks?.value || 1),
      max_chars: Number(tsMaxChars?.value || 6000),
    };
  }

  function buildDatasetSpec() {
    const textColumns = (tsDatasetTextColumns?.value || "")
      .split(",")
      .map((col) => col.trim())
      .filter(Boolean);
    return {
      name: tsDatasetName?.value?.trim() || "",
      source_table: tsDatasetTable?.value?.trim() || "",
      id_column: tsDatasetIdColumn?.value?.trim() || "",
      time_column: tsDatasetTimeColumn?.value?.trim() || "",
      text_columns: textColumns,
      where_sql: tsDatasetWhereSql?.value?.trim() || null,
      timezone: tsDatasetTimezone?.value?.trim() || "UTC",
    };
  }

  async function topicFoundryFetch(path, options = {}) {
    const response = await fetch(`${TOPIC_FOUNDRY_PROXY_BASE}${path}`, {
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
    if (response.status === 204) return null;
    return payload;
  }

  async function topicFoundryFetchWithHeaders(path, options = {}) {
    const response = await fetch(`${TOPIC_FOUNDRY_PROXY_BASE}${path}`, {
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
  const TOPIC_STUDIO_RUN_ID_KEY = "topic_studio_run_id_v1";

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
      option.value = dataset.dataset_id;
      option.textContent = `${dataset.name} (${dataset.dataset_id})`;
      tsDatasetSelect.appendChild(option);
    });
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
    if (Array.isArray(response)) return response;
    if (Array.isArray(response.items)) return response.items;
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
      option.value = dataset.dataset_id;
      option.textContent = `${dataset.name} (${dataset.dataset_id})`;
      tsConvoDatasetSelect.appendChild(option);
    });
    if (current) {
      tsConvoDatasetSelect.value = current;
    }
    if (!tsConvoDatasetSelect.value) {
      if (tsDatasetSelect?.value) {
        tsConvoDatasetSelect.value = tsDatasetSelect.value;
      } else if (topicStudioDatasets.length > 0) {
        tsConvoDatasetSelect.value = topicStudioDatasets[0].dataset_id;
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
    const headers = ["segment_id", "size", "start_at", "end_at", "row_ids_count", "title", "aspects", "valence", "friction", "snippet"];
    const lines = [headers.join(",")];
    const escapeCsv = (value) => {
      if (value === null || value === undefined) return "";
      const str = String(value).replace(/"/g, "\"\"");
      return `"${str}"`;
    };
    exportRows.forEach((segment) => {
      const sentiment = segment.sentiment || {};
      const aspects = Array.isArray(segment.aspects) ? segment.aspects.join("|") : "";
      const rowIdsCount = segment.row_ids_count ?? "";
      const row = [
        escapeCsv(segment.segment_id),
        escapeCsv(segment.size ?? ""),
        escapeCsv(segment.start_at ?? ""),
        escapeCsv(segment.end_at ?? ""),
        escapeCsv(rowIdsCount),
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

  function populateDatasetForm(dataset) {
    if (!dataset) return;
    if (tsDatasetName) tsDatasetName.value = dataset.name || "";
    if (tsDatasetTable) tsDatasetTable.value = dataset.source_table || "";
    if (tsDatasetIdColumn) tsDatasetIdColumn.value = dataset.id_column || "";
    if (tsDatasetTimeColumn) tsDatasetTimeColumn.value = dataset.time_column || "";
    if (tsDatasetTextColumns) tsDatasetTextColumns.value = (dataset.text_columns || []).join(", ");
    if (tsDatasetWhereSql) tsDatasetWhereSql.value = dataset.where_sql || "";
    if (tsDatasetTimezone) tsDatasetTimezone.value = dataset.timezone || "UTC";
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

  function updateRoutingDebugPanel(data) {
    if (!outboundRoutingDebug) return;
    const routeDebug = data && typeof data.routing_debug === 'object' ? data.routing_debug : null;
    outboundRoutingDebug.textContent = routeDebug ? JSON.stringify(routeDebug, null, 2) : '--';
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
    updateRoutingDebugPanel(data);
  }

  function clearAgentTraceDebugPanel() {
    if (agentTraceDebugBody) agentTraceDebugBody.classList.add('hidden');
    if (agentTraceDebugCaret) agentTraceDebugCaret.textContent = '▾';
    if (agentTraceDebugMeta) agentTraceDebugMeta.textContent = 'No agent trace on this turn.';
    if (agentTraceDebugOverview) agentTraceDebugOverview.innerHTML = '';
    if (agentTraceDebugSummary) agentTraceDebugSummary.textContent = 'No agent trace on this turn.';
    if (agentTraceDebugToolGroups) agentTraceDebugToolGroups.innerHTML = '';
    if (agentTraceDebugTimeline) agentTraceDebugTimeline.innerHTML = '';
    if (agentTraceDebugRaw) agentTraceDebugRaw.textContent = 'No agent trace on this turn.';
  }

  function updateAgentTraceDebugPanel(summary, meta = {}) {
    if (
      !agentTraceDebugPanel
      || !agentTraceDebugBody
      || !agentTraceDebugOverview
      || !agentTraceDebugSummary
      || !agentTraceDebugToolGroups
      || !agentTraceDebugTimeline
      || !agentTraceDebugRaw
    ) {
      return;
    }
    if (!agentTraceApi.shouldShowAgentTrace || !agentTraceApi.shouldShowAgentTrace(summary)) {
      clearAgentTraceDebugPanel();
      return;
    }

    agentTraceDebugPanel.classList.remove('hidden');
    if (agentTraceDebugMeta) {
      const corr = summary.corr_id || meta.correlationId || '--';
      agentTraceDebugMeta.textContent = `corr ${corr} · status ${summary.status || '--'} · ${summary.step_count || 0} steps`;
    }

    agentTraceDebugOverview.innerHTML = '';
    Array.from(buildAgentTraceOverviewNode(summary).children).forEach((child) => agentTraceDebugOverview.appendChild(child));

    agentTraceDebugSummary.textContent = summary.summary_text || 'No deterministic summary available.';

    agentTraceDebugToolGroups.innerHTML = '';
    Array.from(buildAgentTraceToolGroupsNode(summary).children).forEach((child) => agentTraceDebugToolGroups.appendChild(child));

    agentTraceDebugTimeline.innerHTML = '';
    agentTraceDebugTimeline.appendChild(buildAgentTraceTimelineNode(summary));

    agentTraceDebugRaw.innerHTML = '';
    agentTraceDebugRaw.appendChild(buildAgentTraceRawPayloadsNode(summary));
  }

  function clearAutonomyDebugPanel() {
    if (autonomyDebugBody) autonomyDebugBody.classList.add('hidden');
    if (autonomyDebugCaret) autonomyDebugCaret.textContent = '▾';
    if (autonomyDebugMeta) autonomyDebugMeta.textContent = 'No autonomy payload on this turn.';
    if (autonomyDebugOverview) autonomyDebugOverview.textContent = 'No autonomy payload on this turn.';
    if (autonomyDebugState) autonomyDebugState.textContent = 'No meaningful autonomy signal for this turn.';
    if (autonomyDebugProposals) autonomyDebugProposals.textContent = 'No autonomy payload on this turn.';
    if (autonomyDebugAlignment) autonomyDebugAlignment.textContent = 'No meaningful autonomy signal for this turn.';
    if (autonomyDebugRaw) autonomyDebugRaw.textContent = 'No autonomy payload on this turn.';
    if (autonomyDebugModalMeta) autonomyDebugModalMeta.textContent = 'No autonomy payload on this turn.';
    if (autonomyDebugModalBody) autonomyDebugModalBody.textContent = 'No autonomy payload on this turn.';
  }

  function expectedPostureFromDrive(drive) {
    const key = String(drive || '').trim().toLowerCase();
    if (key === 'coherence') return 'synthesis/reduction';
    if (key === 'continuity') return 'continuity-preserving';
    if (key === 'relational_stability') return 'relationally steady';
    if (key === 'capability_expansion') return 'forward-building';
    if (key === 'predictive_mastery') return 'clarifying / forecasting';
    return key ? 'neutral' : null;
  }

  function deriveVisibleAutonomyCues(replyText) {
    const text = String(replyText || '').toLowerCase();
    if (!text) return [];
    const cues = [];
    const cueRules = [
      { label: 'continuity-preserving', patterns: ['continue', 'earlier', 'you said', 'as we were saying'] },
      { label: 'synthesis/reduction', patterns: ['the key point', 'in short', 'this means', 'so the answer is'] },
      { label: 'relationally steady', patterns: ['i hear you', 'that makes sense', 'you’re right', "you're right"] },
    ];
    cueRules.forEach((rule) => {
      if (rule.patterns.some((pattern) => text.includes(pattern))) {
        cues.push(rule.label);
      }
    });
    return cues.slice(0, 3);
  }

  function computeAutonomyAlignment(model, replyText) {
    const expectedPosture = expectedPostureFromDrive(model && model.dominantDrive);
    const visibleCues = deriveVisibleAutonomyCues(replyText);
    let alignmentNote = 'no strong posture expected';
    if (expectedPosture) {
      alignmentNote = visibleCues.includes(expectedPosture)
        ? 'reply appears aligned'
        : 'reply posture not clearly visible';
    }
    return {
      expected_posture: expectedPosture,
      visible_cues: visibleCues,
      alignment_note: alignmentNote,
    };
  }

  function normalizeAutonomyModel(summary, debug, meta = {}) {
    const safeSummary = summary && typeof summary === 'object' ? summary : null;
    const safeDebug = debug && typeof debug === 'object' ? debug : null;
    if (!safeSummary && !safeDebug) return null;
    const topDrives = Array.isArray(safeSummary && safeSummary.top_drives)
      ? safeSummary.top_drives.map((v) => String(v || '').trim()).filter(Boolean).slice(0, 3)
      : [];
    const tensions = Array.isArray(safeSummary && safeSummary.active_tensions)
      ? safeSummary.active_tensions.map((v) => String(v || '').trim()).filter(Boolean).slice(0, 3)
      : [];
    const proposals = Array.isArray(safeSummary && safeSummary.proposal_headlines)
      ? safeSummary.proposal_headlines.map((v) => String(v || '').trim()).filter(Boolean).slice(0, 3)
      : [];
    const availabilityRows = safeDebug
      ? Object.entries(safeDebug).filter(([subject, value]) => !String(subject).startsWith('_') && value && typeof value === 'object')
      : [];
    const availableCount = availabilityRows.filter(([, value]) => value.availability === 'available').length;
    const unavailableCount = availabilityRows.filter(([, value]) => value.availability === 'unavailable').length;
    const subjectCount = availabilityRows.length;
    const dominantDrive = String((safeSummary && safeSummary.dominant_drive) || '').trim();
    const hasSemanticSignal = !!(dominantDrive || topDrives.length || tensions.length || proposals.length);
    const hasDebugSignal = !!(safeDebug && typeof safeDebug === 'object' && Object.keys(safeDebug).length);
    const hasAnySignal = !!(hasSemanticSignal || hasDebugSignal || String((safeSummary && safeSummary.stance_hint) || '').trim());
    if (!hasAnySignal) return null;

    return {
      dominantDrive,
      topDrives,
      tensions,
      proposals,
      stanceHint: String((safeSummary && safeSummary.stance_hint) || '').trim(),
      hasSemanticSignal,
      hasDebugSignal,
      backend: String((meta.autonomyBackend || (safeDebug && safeDebug._runtime && safeDebug._runtime.backend) || meta.backend || '')).trim() || '--',
      selectedSubject: String((meta.autonomySelectedSubject || (safeDebug && safeDebug._runtime && safeDebug._runtime.selected_subject) || meta.selectedSubject || '')).trim() || '--',
      availability: {
        available: availableCount,
        unavailable: unavailableCount,
        subjects: subjectCount,
      },
      fallback: unavailableCount > 0 ? 'yes' : 'no',
      alignment: computeAutonomyAlignment({ dominantDrive }, meta.replyText || meta.reply_text || ''),
      raw: {
        summary: safeSummary || {},
        debug: safeDebug || {},
      },
    };
  }

  function shouldRenderAutonomyInline(model) {
    if (!model || typeof model !== 'object') return false;
    return !!(model.dominantDrive || (model.topDrives || []).length || (model.tensions || []).length || (model.proposals || []).length);
  }

  function updateAutonomyDebugPanel(summary, debug, meta = {}) {
    if (
      !autonomyDebugPanel
      || !autonomyDebugBody
      || !autonomyDebugMeta
      || !autonomyDebugOverview
      || !autonomyDebugState
      || !autonomyDebugProposals
      || !autonomyDebugAlignment
      || !autonomyDebugRaw
    ) {
      return;
    }
    const model = normalizeAutonomyModel(summary, debug, meta);
    if (!model) {
      clearAutonomyDebugPanel();
      return;
    }
    autonomyDebugPanel.classList.remove('hidden');
    if (autonomyDebugMeta) {
      autonomyDebugMeta.textContent = `backend ${model.backend} · selected ${model.selectedSubject} · proposal-only`;
    }

    autonomyDebugOverview.innerHTML = '';
    [
      `backend: ${model.backend}`,
      `selected subject: ${model.selectedSubject}`,
      `availability: ${model.availability.available}/${model.availability.subjects} available`,
      `fallback: ${model.fallback}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      autonomyDebugOverview.appendChild(row);
    });

    autonomyDebugState.innerHTML = '';
    [
      `dominant drive: ${model.dominantDrive || '--'}`,
      `top drives: ${model.topDrives.length ? model.topDrives.join(', ') : '--'}`,
      `top tensions: ${model.tensions.length ? model.tensions.join(', ') : '--'}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      autonomyDebugState.appendChild(row);
    });

    autonomyDebugProposals.innerHTML = '';
    if (model.proposals.length) {
      model.proposals.forEach((proposal) => {
        const row = document.createElement('div');
        row.className = 'rounded-lg border border-amber-500/30 bg-amber-500/5 px-2 py-1';
        row.textContent = `proposal-only: ${proposal}`;
        autonomyDebugProposals.appendChild(row);
      });
    } else {
      autonomyDebugProposals.textContent = '--';
    }

    autonomyDebugAlignment.innerHTML = '';
    [
      `expected posture: ${model.alignment.expected_posture || '--'}`,
      `visible cues: ${model.alignment.visible_cues.length ? model.alignment.visible_cues.join(', ') : '--'}`,
      `alignment note: ${model.alignment.alignment_note}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      autonomyDebugAlignment.appendChild(row);
    });

    autonomyDebugRaw.textContent = JSON.stringify(model.raw, null, 2);
    if (autonomyDebugModalMeta) autonomyDebugModalMeta.textContent = autonomyDebugMeta ? autonomyDebugMeta.textContent : '';
    if (autonomyDebugModalBody) autonomyDebugModalBody.innerHTML = autonomyDebugBody ? autonomyDebugBody.innerHTML : '';
  }

  function ensureAutonomyModalRootOnBody() {
    if (!autonomyDebugModalRoot || !document.body) return;
    if (autonomyDebugModalRoot.parentElement !== document.body) {
      document.body.appendChild(autonomyDebugModalRoot);
    }
  }

  function openAutonomyDebugModal() {
    if (!autonomyDebugModalRoot) return;
    ensureAutonomyModalRootOnBody();
    if (autonomyDebugModalMeta) autonomyDebugModalMeta.textContent = autonomyDebugMeta ? autonomyDebugMeta.textContent : '--';
    if (autonomyDebugModalBody) autonomyDebugModalBody.innerHTML = autonomyDebugBody ? autonomyDebugBody.innerHTML : '--';
    autonomyDebugModalRoot.classList.remove('hidden');
    autonomyDebugModalRoot.setAttribute('aria-hidden', 'false');
    if (document.body) document.body.classList.add('overflow-hidden');
  }

  function closeAutonomyDebugModal() {
    if (!autonomyDebugModalRoot) return;
    autonomyDebugModalRoot.classList.add('hidden');
    autonomyDebugModalRoot.setAttribute('aria-hidden', 'true');
    if (document.body) document.body.classList.remove('overflow-hidden');
  }

  function renderSocialInspectionBadges(container, badges) {
    if (!container) return;
    container.innerHTML = '';
    const toneMap = {
      indigo: 'border-indigo-500/40 bg-indigo-500/10 text-indigo-200',
      sky: 'border-sky-500/40 bg-sky-500/10 text-sky-200',
      amber: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
      emerald: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
      violet: 'border-violet-500/40 bg-violet-500/10 text-violet-200',
      fuchsia: 'border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-200',
      orange: 'border-orange-500/40 bg-orange-500/10 text-orange-200',
      rose: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
      gray: 'border-gray-600 bg-gray-800/70 text-gray-200',
    };
    (Array.isArray(badges) ? badges : []).forEach((badge) => {
      const pill = document.createElement('div');
      pill.className = `rounded-full border px-2.5 py-1 text-[11px] ${toneMap[badge.tone] || toneMap.gray}`;
      pill.textContent = `${badge.label}: ${badge.value}`;
      container.appendChild(pill);
    });
  }

  function renderSocialInspectionSummary(container, rows) {
    if (!container) return;
    container.innerHTML = '';
    (Array.isArray(rows) ? rows : []).forEach((row) => {
      const wrap = document.createElement('div');
      wrap.className = 'rounded-xl border border-gray-800 bg-gray-900/60 px-3 py-2';
      const label = document.createElement('div');
      label.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      label.textContent = row.label || '--';
      const value = document.createElement('div');
      value.className = 'mt-1 text-sm text-gray-100';
      value.textContent = row.value || '--';
      wrap.appendChild(label);
      wrap.appendChild(value);
      container.appendChild(wrap);
    });
  }

  function appendSocialInspectionStateList(parent, labelText, items, accentClass) {
    if (!parent || !Array.isArray(items) || !items.length) return;
    const wrap = document.createElement('div');
    wrap.className = 'space-y-1';
    const label = document.createElement('div');
    label.className = `text-[10px] uppercase tracking-wide ${accentClass}`;
    label.textContent = labelText;
    const list = document.createElement('ul');
    list.className = 'space-y-1 text-[11px] text-gray-300';
    items.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'rounded-lg border border-gray-800 bg-gray-950/50 px-2 py-1';
      li.textContent = item;
      list.appendChild(li);
    });
    wrap.appendChild(label);
    wrap.appendChild(list);
    parent.appendChild(wrap);
  }

  function renderSocialInspectionSurface(container, surfaceModel, emptyMessage) {
    if (!container) return;
    container.innerHTML = '';
    const model = surfaceModel && typeof surfaceModel === 'object' ? surfaceModel : null;
    if (!model || !Array.isArray(model.sections) || !model.sections.length) {
      const empty = document.createElement('div');
      empty.className = 'rounded-xl border border-dashed border-gray-700 bg-gray-950/40 p-3 text-sm text-gray-500';
      empty.textContent = emptyMessage;
      container.appendChild(empty);
      return;
    }
    const summary = document.createElement('div');
    summary.className = 'rounded-xl border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-200';
    summary.textContent = model.summary || 'No inspection summary available.';
    container.appendChild(summary);
    model.sections.forEach((section) => {
      const details = document.createElement('details');
      details.className = 'rounded-xl border border-gray-800 bg-gray-950/40 p-3';
      if (section.kind === 'context_window' || section.kind === 'routing' || section.kind === 'epistemic') {
        details.open = true;
      }
      const summaryRow = document.createElement('summary');
      summaryRow.className = 'cursor-pointer list-none';
      const top = document.createElement('div');
      top.className = 'flex items-start justify-between gap-3';
      const titleWrap = document.createElement('div');
      const title = document.createElement('div');
      title.className = 'text-sm font-semibold text-white';
      title.textContent = section.title;
      const why = document.createElement('div');
      why.className = 'mt-1 text-[11px] text-gray-400';
      why.textContent = section.why || 'No explanation captured.';
      titleWrap.appendChild(title);
      titleWrap.appendChild(why);
      const counts = document.createElement('div');
      counts.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      counts.textContent = `${(section.selected || []).length} selected · ${(section.softened || []).length} softened · ${(section.excluded || []).length} excluded`;
      top.appendChild(titleWrap);
      top.appendChild(counts);
      summaryRow.appendChild(top);
      details.appendChild(summaryRow);

      const body = document.createElement('div');
      body.className = 'mt-3 space-y-3';
      appendSocialInspectionStateList(body, 'Selected', section.selected && section.selected.length ? section.selected : section.included, 'text-emerald-300');
      appendSocialInspectionStateList(body, 'Softened', section.softened, 'text-amber-300');
      appendSocialInspectionStateList(body, 'Excluded / omitted', section.excluded, 'text-rose-300');
      appendSocialInspectionStateList(body, 'Freshness / confidence', [...(section.freshness || []), ...(section.confidence || [])], 'text-sky-300');
      if (Array.isArray(section.traces) && section.traces.length) {
        const traceWrap = document.createElement('div');
        traceWrap.className = 'space-y-2';
        const traceLabel = document.createElement('div');
        traceLabel.className = 'text-[10px] uppercase tracking-wide text-gray-500';
        traceLabel.textContent = 'Decision traces';
        traceWrap.appendChild(traceLabel);
        section.traces.forEach((trace) => {
          const item = document.createElement('div');
          item.className = 'rounded-lg border border-gray-800 bg-gray-900/80 px-3 py-2';
          const traceHead = document.createElement('div');
          traceHead.className = 'text-[11px] font-semibold text-gray-100';
          traceHead.textContent = trace.summary || trace.trace_kind || 'Trace';
          const traceWhy = document.createElement('div');
          traceWhy.className = 'mt-1 text-[11px] text-gray-400';
          traceWhy.textContent = trace.why_it_mattered || 'No explanation captured.';
          item.appendChild(traceHead);
          item.appendChild(traceWhy);
          traceWrap.appendChild(item);
        });
        body.appendChild(traceWrap);
      }
      details.appendChild(body);
      container.appendChild(details);
    });
  }

  function setSocialInspectionModalState({ loading = false, error = '', meta = '', memoryMeta = '' } = {}) {
    if (socialInspectionModalLoading) socialInspectionModalLoading.classList.toggle('hidden', !loading);
    if (socialInspectionModalError) {
      socialInspectionModalError.classList.toggle('hidden', !error);
      socialInspectionModalError.textContent = error || '';
    }
    if (socialInspectionModalMeta && meta) socialInspectionModalMeta.textContent = meta;
    if (socialInspectionMemoryMeta) socialInspectionMemoryMeta.textContent = memoryMeta || 'Not loaded';
  }

  function renderSocialInspectionState(state) {
    latestSocialInspectionState = state;
    if (!socialInspectionApi.buildOperatorSummary) return;
    if (!state || !state.routeDebug) {
      if (socialInspectionPanelStatus) socialInspectionPanelStatus.textContent = 'Waiting for a social-room turn.';
      if (socialInspectionOpen) socialInspectionOpen.disabled = true;
      renderSocialInspectionBadges(socialInspectionBadgeRow, []);
      renderSocialInspectionSummary(socialInspectionPanelSummary, []);
      renderSocialInspectionBadges(socialInspectionModalBadges, []);
      renderSocialInspectionSummary(socialInspectionModalSummary, []);
      renderSocialInspectionSurface(socialInspectionLiveSurface, null, 'No live social inspection has been captured yet.');
      renderSocialInspectionSurface(socialInspectionMemorySurface, null, 'Open a social-room turn to load the social-memory inspection snapshot.');
      setSocialInspectionModalState({ loading: false, error: '', meta: 'Live Hub routing debug with social-memory inspection on demand.', memoryMeta: 'Not loaded' });
      return;
    }
    const model = socialInspectionApi.buildOperatorSummary(state.routeDebug, state.liveSnapshot, state.memorySnapshot);
    if (socialInspectionPanelStatus) {
      socialInspectionPanelStatus.textContent = `${model.query.platform || '--'} · ${model.query.room_id || '--'} · ${model.liveSnapshot && model.liveSnapshot.sections ? model.liveSnapshot.sections.length : 0} sections`;
    }
    if (socialInspectionOpen) socialInspectionOpen.disabled = !model.query.available;
    renderSocialInspectionBadges(socialInspectionBadgeRow, model.badges);
    renderSocialInspectionSummary(socialInspectionPanelSummary, model.summaryRows.slice(0, 4));
    renderSocialInspectionBadges(socialInspectionModalBadges, model.badges);
    renderSocialInspectionSummary(socialInspectionModalSummary, model.summaryRows);
    renderSocialInspectionSurface(
      socialInspectionLiveSurface,
      socialInspectionApi.buildSurfaceModel(state.liveSnapshot, 'Live turn'),
      'No live social inspection has been captured yet.'
    );
    renderSocialInspectionSurface(
      socialInspectionMemorySurface,
      socialInspectionApi.buildSurfaceModel(state.memorySnapshot, 'Memory baseline'),
      'Open this inspector to fetch the bounded social-memory snapshot.'
    );
    setSocialInspectionModalState({
      loading: Boolean(state.loadingMemory),
      error: state.error || '',
      meta: `${model.query.platform || '--'} · room ${model.query.room_id || '--'} · participant ${model.query.participant_id || 'room'}`,
      memoryMeta: state.memorySnapshot ? 'Loaded from /inspection' : (state.loadingMemory ? 'Loading…' : 'Not loaded'),
    });
  }

  async function fetchSocialMemoryInspection(query) {
    const params = new URLSearchParams();
    params.set('platform', query.platform);
    params.set('room_id', query.room_id);
    if (query.participant_id) params.set('participant_id', query.participant_id);
    const response = await fetch(`${API_BASE_URL}/api/social-memory/inspection?${params.toString()}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload && payload.detail ? payload.detail : `HTTP ${response.status}`);
    }
    return payload;
  }

  async function openSocialInspectionModal(sourceState = latestSocialInspectionState) {
    if (!socialInspectionModal || !sourceState || !sourceState.routeDebug || !socialInspectionApi.resolveInspectionQuery) return;
    const query = socialInspectionApi.resolveInspectionQuery(sourceState.routeDebug);
    socialInspectionModal.classList.remove('hidden');
    socialInspectionModal.setAttribute('aria-hidden', 'false');
    renderSocialInspectionState({ ...sourceState, loadingMemory: Boolean(sourceState.loadingMemory) });
    if (!query.available || sourceState.memorySnapshot || sourceState.loadingMemory) return;
    const cached = socialInspectionCache.get(query.cache_key);
    if (cached) {
      renderSocialInspectionState({ ...sourceState, memorySnapshot: cached, loadingMemory: false, error: '' });
      return;
    }
    renderSocialInspectionState({ ...sourceState, loadingMemory: true, error: '' });
    try {
      const memorySnapshot = await fetchSocialMemoryInspection(query);
      socialInspectionCache.set(query.cache_key, memorySnapshot);
      renderSocialInspectionState({ ...sourceState, memorySnapshot, loadingMemory: false, error: '' });
    } catch (error) {
      renderSocialInspectionState({ ...sourceState, loadingMemory: false, error: error.message || 'Failed to load social-memory inspection.' });
    }
  }

  function closeSocialInspectionModal() {
    if (!socialInspectionModal) return;
    socialInspectionModal.classList.add('hidden');
    socialInspectionModal.setAttribute('aria-hidden', 'true');
  }

  function syncSocialInspectionFromRouteDebug(routeDebug) {
    if (!socialInspectionApi.shouldShowSocialInspection || !socialInspectionApi.shouldShowSocialInspection(routeDebug)) return;
    renderSocialInspectionState({
      routeDebug,
      liveSnapshot: routeDebug.social_inspection,
      memorySnapshot: latestSocialInspectionState && latestSocialInspectionState.routeDebug && socialInspectionApi.resolveInspectionQuery(routeDebug).cache_key === socialInspectionApi.resolveInspectionQuery(latestSocialInspectionState.routeDebug).cache_key
        ? latestSocialInspectionState.memorySnapshot
        : null,
      loadingMemory: false,
      error: '',
    });
  }

  function toggleMemoryPanel() {
    if (!memoryPanelBody) return;
    memoryPanelBody.classList.toggle('hidden');
  }

  function toggleAgentTraceDebugPanel() {
    if (!agentTraceDebugBody) return;
    const nextHidden = !agentTraceDebugBody.classList.contains('hidden');
    agentTraceDebugBody.classList.toggle('hidden', nextHidden);
    if (agentTraceDebugCaret) agentTraceDebugCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function toggleAutonomyDebugPanel() {
    if (!autonomyDebugBody) return;
    const nextHidden = !autonomyDebugBody.classList.contains('hidden');
    autonomyDebugBody.classList.toggle('hidden', nextHidden);
    if (autonomyDebugCaret) autonomyDebugCaret.textContent = nextHidden ? '▾' : '▴';
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

  function statusChipClass(status) {
    switch (String(status || '').toLowerCase()) {
      case 'completed':
        return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200';
      case 'failed':
        return 'border-red-500/40 bg-red-500/10 text-red-200';
      case 'running':
        return 'border-amber-500/40 bg-amber-500/10 text-amber-200';
      case 'requested':
        return 'border-sky-500/40 bg-sky-500/10 text-sky-200';
      default:
        return 'border-violet-500/40 bg-violet-500/10 text-violet-200';
    }
  }

  function normalizeWorkflow(messageLike, explicitStatus) {
    if (!workflowUiApi.normalizeWorkflow) return null;
    const options = explicitStatus ? { status: explicitStatus } : {};
    return workflowUiApi.normalizeWorkflow(
      (messageLike && (messageLike.workflow || messageLike.raw_workflow || messageLike.rawWorkflow)) || messageLike,
      options,
    );
  }

  function buildWorkflowMetaBadges(workflow) {
    const badges = [];
    if (!workflow) return badges;
    badges.push({ label: workflowUiApi.getWorkflowBadgeLabel ? workflowUiApi.getWorkflowBadgeLabel(workflow) : `Workflow · ${workflow.display_name || workflow.id}`, className: 'border-violet-500/40 bg-violet-500/10 text-violet-200' });
    const statusLabel = workflowUiApi.getWorkflowStatusLabel ? workflowUiApi.getWorkflowStatusLabel(workflow.status) : (workflow.status || '');
    if (statusLabel) badges.push({ label: statusLabel, className: statusChipClass(workflow.status) });
    if (Array.isArray(workflow.persisted) && workflow.persisted.length) badges.push({ label: 'Persisted', className: 'border-gray-600/50 bg-gray-800/70 text-gray-200' });
    if (workflow.action_assistance_used) badges.push({ label: 'Actions assist', className: 'border-gray-600/50 bg-gray-800/70 text-gray-200' });
    return badges;
  }

  function renderBadgeRow(badges) {
    const row = document.createElement('div');
    row.className = 'flex flex-wrap items-center gap-2';
    (badges || []).forEach((badge) => {
      if (!badge || !badge.label) return;
      const chip = document.createElement('span');
      chip.className = `inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold ${badge.className || 'border-gray-700 bg-gray-800 text-gray-200'}`;
      chip.textContent = badge.label;
      row.appendChild(chip);
    });
    return row;
  }

  function buildWorkflowSummaryLine(workflow) {
    if (!workflow) return '';
    const summary = [];
    if (workflow.summary) summary.push(workflow.summary);
    if (Array.isArray(workflow.persisted) && workflow.persisted.length) summary.push(`persisted ${workflow.persisted.length}`);
    if (Array.isArray(workflow.scheduled) && workflow.scheduled.length) summary.push(`scheduled ${workflow.scheduled.length}`);
    return summary.join(' · ');
  }

  function buildWorkflowModalSummaryCard(label, value) {
    const card = document.createElement('div');
    card.className = 'rounded-xl border border-gray-700 bg-gray-800/50 p-3';
    const title = document.createElement('div');
    title.className = 'text-[10px] uppercase tracking-wide text-gray-500';
    title.textContent = label;
    const body = document.createElement('div');
    body.className = 'mt-2 text-sm font-semibold text-gray-100 whitespace-pre-wrap';
    body.textContent = value || '--';
    card.appendChild(title);
    card.appendChild(body);
    return card;
  }

  function buildCodePre(value) {
    const pre = document.createElement('pre');
    pre.className = 'mt-2 overflow-x-auto rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-[11px] text-gray-200';
    pre.textContent = JSON.stringify(value, null, 2);
    return pre;
  }

  function renderConceptInductionDetails(normalized) {
    const details = workflowUiApi.normalizeConceptInductionDetails ? workflowUiApi.normalizeConceptInductionDetails(normalized) : null;
    if (!details) return null;
    const root = document.createElement('div');
    root.className = 'space-y-4';
    const sections = workflowUiApi.buildConceptInductionSections ? workflowUiApi.buildConceptInductionSections(normalized) : [];

    const overview = document.createElement('section');
    overview.className = 'rounded-xl border border-gray-800 bg-gray-900/30 p-3';
    overview.innerHTML = `<div class="text-xs uppercase tracking-wide text-gray-500">${sections[0] || 'Overview'}</div>`;
    const resolution = (details.trace && details.trace.repository_resolution) || {};
    overview.appendChild(buildCodePre({
      generated_at: details.generated_at,
      reviewed_subjects: details.profiles.map((p) => p.subject),
      backend: {
        requested: resolution.requested_backend || null,
        resolved: resolution.resolved_backend || null,
        fallback_used: Boolean(resolution.fallback_used),
      },
    }));
    root.appendChild(overview);

    details.profiles.forEach((profile) => {
      const profileSection = document.createElement('section');
      profileSection.className = 'rounded-xl border border-gray-800 bg-gray-900/30 p-3';
      profileSection.innerHTML = `<div class="text-xs uppercase tracking-wide text-gray-500">${sections[1] || 'Profiles'} · ${profile.subject || '--'}</div>`;
      profileSection.appendChild(buildCodePre({
        profile_id: profile.profile_id,
        revision: profile.revision,
        created_at: profile.created_at,
        window_start: profile.window_start,
        window_end: profile.window_end,
        concept_count: profile.concept_count,
        cluster_count: profile.cluster_count,
      }));
      root.appendChild(profileSection);
      const concepts = document.createElement('section');
      concepts.className = 'rounded-xl border border-gray-800 bg-gray-900/30 p-3';
      concepts.innerHTML = `<div class="text-xs uppercase tracking-wide text-gray-500">${sections[2] || 'Concepts'} · ${profile.subject || '--'}</div>`;
      concepts.appendChild(buildCodePre(profile.concepts));
      root.appendChild(concepts);
      const clusters = document.createElement('section');
      clusters.className = 'rounded-xl border border-gray-800 bg-gray-900/30 p-3';
      clusters.innerHTML = `<div class="text-xs uppercase tracking-wide text-gray-500">${sections[3] || 'Clusters'} · ${profile.subject || '--'}</div>`;
      clusters.appendChild(buildCodePre(profile.clusters));
      root.appendChild(clusters);
      const state = document.createElement('section');
      state.className = 'rounded-xl border border-gray-800 bg-gray-900/30 p-3';
      state.innerHTML = `<div class="text-xs uppercase tracking-wide text-gray-500">${sections[4] || 'State estimate'} · ${profile.subject || '--'}</div>`;
      state.appendChild(buildCodePre(profile.state_estimate || {}));
      root.appendChild(state);
    });

    const traceSection = document.createElement('section');
    traceSection.className = 'rounded-xl border border-gray-800 bg-gray-900/30 p-3';
    traceSection.innerHTML = `<div class="text-xs uppercase tracking-wide text-gray-500">${sections[5] || 'Trace / Artifacts'}</div>`;
    traceSection.appendChild(buildCodePre(details.trace));
    root.appendChild(traceSection);
    return root;
  }

  async function synthesizeConceptInductionToJournal(workflow, statusElement) {
    const details = workflowUiApi.normalizeConceptInductionDetails ? workflowUiApi.normalizeConceptInductionDetails(workflow) : null;
    if (!details) return;
    if (statusElement) statusElement.textContent = 'Synthesizing concept induction learnings to journal…';
    const payload = {
      mode: 'brain',
      session_id: orionSessionId,
      use_recall: false,
      no_write: false,
      messages: [{ role: 'user', content: 'Synthesize concept induction review to journal.' }],
      workflow_request_override: {
        workflow_id: 'concept_induction_pass',
        action: 'synthesize_to_journal',
        concept_induction_details: details,
      },
    };
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(orionSessionId ? { 'X-Orion-Session-Id': orionSessionId } : {}) },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (data.session_id && !orionSessionId) {
      orionSessionId = data.session_id;
      localStorage.setItem('orion_sid', orionSessionId);
    }
    if (data.text) {
      appendMessage('Orion', data.text, 'text-white', {
        workflow: data.workflow,
        correlationId: data.correlation_id,
      });
    }
    if (statusElement) statusElement.textContent = data.error ? `Synthesis failed: ${data.error}` : 'Synthesis persisted to journal.';
  }

  function closeWorkflowModal() {
    if (!workflowModal) return;
    workflowModal.classList.add('hidden');
    workflowModal.setAttribute('aria-hidden', 'true');
  }

  function openWorkflowModal(workflow) {
    const normalized = normalizeWorkflow(workflow);
    if (!workflowModal || !normalized) return;
    if (workflowModalTitle) workflowModalTitle.textContent = normalized.display_name || normalized.id || 'Workflow';
    if (workflowModalMeta) {
      const statusLabel = workflowUiApi.getWorkflowStatusLabel ? workflowUiApi.getWorkflowStatusLabel(normalized.status) : (normalized.status || '--');
      workflowModalMeta.textContent = `${normalized.id || '--'} · ${statusLabel || '--'}`;
    }
    if (workflowModalBadges) {
      workflowModalBadges.innerHTML = '';
      workflowModalBadges.appendChild(renderBadgeRow(buildWorkflowMetaBadges(normalized)));
    }
    if (workflowModalSummary) {
      workflowModalSummary.innerHTML = '';
      const rows = [
        ['Summary', normalized.summary || '--'],
        ['Run again prompt', normalized.rerun_prompt || '--'],
      ];
      rows.forEach(([label, value]) => workflowModalSummary.appendChild(buildWorkflowModalSummaryCard(label, value)));
      if (normalized.id === 'concept_induction_pass') {
        const card = document.createElement('div');
        card.className = 'rounded-xl border border-indigo-700/40 bg-indigo-900/20 p-3';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'rounded-full border border-indigo-500/50 bg-indigo-500/20 px-3 py-1 text-[11px] font-semibold text-indigo-100 hover:bg-indigo-500/30';
        button.textContent = 'Synthesize to Journal';
        const status = document.createElement('div');
        status.className = 'mt-2 text-[11px] text-gray-300';
        button.addEventListener('click', async () => {
          button.disabled = true;
          try {
            await synthesizeConceptInductionToJournal(normalized, status);
          } finally {
            button.disabled = false;
          }
        });
        card.appendChild(button);
        card.appendChild(status);
        workflowModalSummary.appendChild(card);
      }
    }
    if (workflowModalDetailSurface) {
      workflowModalDetailSurface.innerHTML = '';
      const conceptSurface = normalized.id === 'concept_induction_pass' ? renderConceptInductionDetails(normalized) : null;
      if (conceptSurface) workflowModalDetailSurface.appendChild(conceptSurface);
      else {
        const rows = workflowUiApi.buildWorkflowDetailRows ? workflowUiApi.buildWorkflowDetailRows(normalized) : [];
        rows.forEach(([label, value]) => {
          const row = document.createElement('div');
          row.className = 'grid grid-cols-1 gap-1 border-b border-gray-800 pb-3 last:border-b-0 last:pb-0 md:grid-cols-[160px_minmax(0,1fr)]';
          const key = document.createElement('div');
          key.className = 'text-[10px] uppercase tracking-wide text-gray-500';
          key.textContent = label;
          const val = document.createElement('div');
          val.className = 'text-sm text-gray-200 whitespace-pre-wrap break-words';
          val.textContent = String(value ?? '--');
          row.appendChild(key);
          row.appendChild(val);
          workflowModalDetailSurface.appendChild(row);
        });
      }
    }
    if (workflowModalRaw) workflowModalRaw.textContent = JSON.stringify(normalized.raw_metadata || normalized, null, 2);
    workflowModal.classList.remove('hidden');
    workflowModal.setAttribute('aria-hidden', 'false');
  }

  function createWorkflowPanel(workflow, options = {}) {
    const normalized = normalizeWorkflow(workflow, options.status);
    if (!normalized) return null;

    const panel = document.createElement('div');
    panel.className = 'mt-3 rounded-xl border border-violet-500/20 bg-violet-500/5 p-3 space-y-3';

    const header = document.createElement('div');
    header.className = 'flex flex-wrap items-center justify-between gap-2';
    header.appendChild(renderBadgeRow(buildWorkflowMetaBadges(normalized)));

    const actions = document.createElement('div');
    actions.className = 'flex flex-wrap items-center gap-2';

    const detailsButton = document.createElement('button');
    detailsButton.type = 'button';
    detailsButton.className = 'rounded-full border border-violet-500/40 bg-violet-500/10 px-2 py-1 text-[10px] font-semibold text-violet-200 hover:bg-violet-500/20';
    detailsButton.textContent = normalized.id === 'concept_induction_pass' ? 'Details' : 'Workflow details';
    detailsButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openWorkflowModal(normalized);
    });
    actions.appendChild(detailsButton);

    if (workflowUiApi.canRunAgain && workflowUiApi.canRunAgain(normalized)) {
      const rerunButton = document.createElement('button');
      rerunButton.type = 'button';
      rerunButton.className = 'rounded-full border border-gray-700 bg-gray-800 px-2 py-1 text-[10px] font-semibold text-gray-200 hover:bg-gray-700';
      rerunButton.textContent = 'Run again';
      rerunButton.addEventListener('click', async (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (typeof options.onRunAgain === 'function') {
          await options.onRunAgain(normalized);
        }
      });
      actions.appendChild(rerunButton);
    }

    header.appendChild(actions);
    panel.appendChild(header);

    const summary = document.createElement('div');
    summary.className = 'text-[11px] text-violet-100/90 whitespace-pre-wrap';
    summary.textContent = buildWorkflowSummaryLine(normalized) || 'Structured workflow turn.';
    panel.appendChild(summary);

    return panel;
  }

  function normalizeSchedule(entry) {
    if (!scheduleUiApi.normalizeSchedule) return null;
    return scheduleUiApi.normalizeSchedule(entry);
  }

  function scheduleStateBadge(state) {
    const cls = scheduleUiApi.stateChipClass ? scheduleUiApi.stateChipClass(state) : 'border-gray-700 bg-gray-800 text-gray-200';
    const chip = document.createElement('span');
    chip.className = `inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold ${cls}`;
    chip.textContent = String(state || 'unknown');
    return chip;
  }

  function scheduleHealthBadge(health) {
    const cls = scheduleUiApi.healthChipClass ? scheduleUiApi.healthChipClass(health) : 'border-gray-700 bg-gray-800 text-gray-200';
    const chip = document.createElement('span');
    chip.className = `inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold ${cls}`;
    chip.textContent = `health: ${String(health || 'idle')}`;
    return chip;
  }

  function scheduleTimeLabel(value) {
    return scheduleUiApi.asLocal ? scheduleUiApi.asLocal(value) : String(value || '--');
  }

  function formatOverdue(seconds) {
    if (!Number.isFinite(Number(seconds))) return 'Overdue';
    const total = Math.max(0, Number(seconds));
    if (total < 3600) return `Overdue ${Math.round(total / 60)}m`;
    if (total < 86400) return `Overdue ${Math.round(total / 3600)}h`;
    return `Overdue ${Math.round(total / 86400)}d`;
  }

  function scheduleNeedsAttention(item) {
    return Boolean(item?.analytics?.needs_attention || item?.analytics?.is_overdue);
  }

  function scheduleOverdueBadge(seconds) {
    const chip = document.createElement('span');
    chip.className = 'inline-flex items-center rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-200';
    chip.textContent = formatOverdue(seconds);
    return chip;
  }

  async function fetchScheduleInventory() {
    const res = await fetch(`${API_BASE_URL}/api/workflow/schedules`);
    if (!res.ok) throw new Error(`Schedule list failed (${res.status})`);
    const payload = await res.json();
    const schedules = Array.isArray(payload.schedules) ? payload.schedules : [];
    workflowSchedules = schedules.map((item) => normalizeSchedule(item)).filter(Boolean);
    if (scheduleInventoryMeta) scheduleInventoryMeta.textContent = `${workflowSchedules.length} schedule(s) loaded`;
    return payload;
  }

  function filteredSchedules() {
    const mode = scheduleFilter ? scheduleFilter.value : 'active';
    if (mode === 'needs_attention') return workflowSchedules.filter((item) => scheduleNeedsAttention(item));
    if (mode === 'all') return workflowSchedules;
    if (mode === 'paused') return workflowSchedules.filter((item) => item.state === 'paused');
    if (mode === 'cancelled') return workflowSchedules.filter((item) => item.state === 'cancelled');
    return workflowSchedules.filter((item) => !['cancelled', 'completed'].includes(item.state));
  }

  function renderScheduleAttentionSummary() {
    if (!scheduleAttentionSummary) return;
    const needsAttention = workflowSchedules.filter((item) => scheduleNeedsAttention(item));
    if (!needsAttention.length) {
      scheduleAttentionSummary.classList.add('hidden');
      scheduleAttentionSummary.textContent = '';
      return;
    }
    const overdueCount = needsAttention.filter((item) => item.analytics?.is_overdue).length;
    scheduleAttentionSummary.classList.remove('hidden');
    scheduleAttentionSummary.textContent = `${needsAttention.length} schedule(s) need attention${overdueCount ? ` · ${overdueCount} overdue` : ''}.`;
  }

  function renderScheduleAnalyticsDetails(schedule) {
    if (!scheduleModalAnalytics || !scheduleModalAnalyticsSummary || !scheduleModalAnalyticsTrend) return;
    const analytics = schedule?.analytics;
    if (!analytics) {
      scheduleModalAnalytics.classList.add('hidden');
      return;
    }
    scheduleModalAnalytics.classList.remove('hidden');
    scheduleModalAnalyticsSummary.innerHTML = '';
    scheduleModalAnalyticsTrend.innerHTML = '';
    const summaryRows = [
      ['Health', analytics.health || '--'],
      ['Needs attention', analytics.needs_attention ? 'yes' : 'no'],
      ['Overdue', analytics.is_overdue ? formatOverdue(analytics.overdue_seconds) : 'no'],
      ['Missed runs', String(analytics.missed_run_count || 0)],
      ['Last success', scheduleTimeLabel(analytics.last_success_at)],
      ['Last failure', scheduleTimeLabel(analytics.last_failure_at)],
      ['Recent counts', `${analytics.recent_success_count || 0} success / ${analytics.recent_failure_count || 0} failure`],
      ['Most recent result', analytics.most_recent_result_status || '--'],
    ];
    summaryRows.forEach(([label, value]) => scheduleModalAnalyticsSummary.appendChild(buildWorkflowModalSummaryCard(label, value)));

    const trend = document.createElement('div');
    trend.className = 'text-[11px] text-gray-300';
    trend.textContent = analytics.trend_text || 'No recent runs.';
    scheduleModalAnalyticsTrend.appendChild(trend);
    (analytics.recent_outcomes || []).forEach((status) => {
      const chip = document.createElement('span');
      chip.className = `inline-flex rounded-full border px-2 py-0.5 text-[10px] ${scheduleUiApi.historyStatusClass ? scheduleUiApi.historyStatusClass(status) : 'border-gray-700 bg-gray-800 text-gray-200'}`;
      chip.textContent = status;
      scheduleModalAnalyticsTrend.appendChild(chip);
    });
  }

  async function performScheduleAction(scheduleId, operation, body) {
    const req = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    };
    const res = await fetch(`${API_BASE_URL}/api/workflow/schedules/${encodeURIComponent(scheduleId)}/${operation}`, req);
    const payload = await res.json();
    if (!res.ok || payload.ok === false) {
      const err = new Error(payload.message || `Schedule ${operation} failed`);
      err.code = payload.error_code || null;
      err.details = payload.error_details || {};
      throw err;
    }
    return payload;
  }

  async function openScheduleDetails(schedule) {
    const normalized = normalizeSchedule(schedule);
    if (!normalized || !scheduleModal) return;
    selectedSchedule = normalized;
    if (scheduleModalTitle) scheduleModalTitle.textContent = normalized.workflow_display_name;
    if (scheduleModalMeta) scheduleModalMeta.textContent = `${normalized.workflow_id} · id:${normalized.schedule_id_short} · rev:${normalized.revision}`;
    if (scheduleModalBadges) {
      scheduleModalBadges.innerHTML = '';
      scheduleModalBadges.appendChild(scheduleStateBadge(normalized.state));
      if (normalized.analytics?.health) scheduleModalBadges.appendChild(scheduleHealthBadge(normalized.analytics.health));
      if (normalized.analytics?.is_overdue) scheduleModalBadges.appendChild(scheduleOverdueBadge(normalized.analytics.overdue_seconds));
    }
    if (scheduleModalSummary) {
      scheduleModalSummary.innerHTML = '';
      const rows = [
        ['Next run', scheduleTimeLabel(normalized.next_run_at)],
        ['Cadence', normalized.cadence_summary || '--'],
        ['Notify', normalized.notify_on || 'none'],
        ['Last run', scheduleTimeLabel(normalized.last_run_at)],
        ['Last result', normalized.last_result_status || '--'],
        ['Schedule ID', normalized.schedule_id],
        ['Revision', String(normalized.revision || 0)],
        ['Source', `${normalized.source_service || '--'} / ${normalized.source_kind || '--'}`],
      ];
      rows.forEach(([label, value]) => scheduleModalSummary.appendChild(buildWorkflowModalSummaryCard(label, value)));
    }
    renderScheduleAnalyticsDetails(normalized);

    if (scheduleModalHistory) {
      scheduleModalHistory.innerHTML = '<div class="text-gray-500">Loading history…</div>';
      try {
        const res = await fetch(`${API_BASE_URL}/api/workflow/schedules/${encodeURIComponent(normalized.schedule_id)}/history`);
        const payload = await res.json();
        const refreshedSchedule = normalizeSchedule(payload.schedule || normalized.raw || {});
        if (refreshedSchedule) {
          selectedSchedule = refreshedSchedule;
          renderScheduleAnalyticsDetails(refreshedSchedule);
        }
        const history = Array.isArray(payload.history)
          ? payload.history.map((item) => (scheduleUiApi.normalizeHistoryItem ? scheduleUiApi.normalizeHistoryItem(item) : item)).filter(Boolean)
          : [];
        const events = Array.isArray(payload.events)
          ? payload.events.map((item) => (scheduleUiApi.normalizeEventItem ? scheduleUiApi.normalizeEventItem(item) : item)).filter(Boolean)
          : [];
        scheduleModalHistory.innerHTML = '';
        if (!history.length && !events.length) {
          scheduleModalHistory.innerHTML = '<div class="text-gray-500">No run history yet.</div>';
        } else {
          if (history.length) {
            const runHeader = document.createElement('div');
            runHeader.className = 'text-[10px] uppercase tracking-wide text-gray-500';
            runHeader.textContent = 'Run attempts';
            scheduleModalHistory.appendChild(runHeader);
            history.slice(0, 5).forEach((item) => {
              const row = document.createElement('div');
              row.className = 'rounded border border-gray-800 bg-gray-900/60 px-2 py-2 flex items-center justify-between gap-2';
              const left = document.createElement('div');
              left.className = 'flex items-center gap-2';
              const chip = document.createElement('span');
              chip.className = `inline-flex rounded-full border px-2 py-0.5 text-[10px] ${scheduleUiApi.historyStatusClass ? scheduleUiApi.historyStatusClass(item.status) : 'border-gray-700 bg-gray-800 text-gray-200'}`;
              chip.textContent = item.status || 'unknown';
              left.appendChild(chip);
              const when = document.createElement('span');
              when.className = 'text-[11px] text-gray-300';
              when.textContent = scheduleTimeLabel(item.dispatch_at || item.completed_at);
              left.appendChild(when);
              row.appendChild(left);
              if (item.error) {
                const err = document.createElement('span');
                err.className = 'text-[11px] text-red-300 truncate';
                err.textContent = item.error;
                row.appendChild(err);
              }
              scheduleModalHistory.appendChild(row);
            });
          }
          if (events.length) {
            const evtHeader = document.createElement('div');
            evtHeader.className = 'mt-2 text-[10px] uppercase tracking-wide text-gray-500';
            evtHeader.textContent = 'Lifecycle events';
            scheduleModalHistory.appendChild(evtHeader);
            events.slice(0, 5).forEach((event) => {
              const row = document.createElement('div');
              row.className = 'rounded border border-gray-800 bg-gray-900/50 px-2 py-1 text-[11px] text-gray-300';
              row.textContent = `${String(event.kind || 'event').replace(/_/g, ' ')} · ${scheduleTimeLabel(event.occurred_at)}`;
              scheduleModalHistory.appendChild(row);
            });
          }
        }
      } catch (err) {
        scheduleModalHistory.innerHTML = `<div class="text-red-300">${err.message}</div>`;
      }
    }

    scheduleModal.classList.remove('hidden');
    scheduleModal.setAttribute('aria-hidden', 'false');
  }

  function closeScheduleModal() {
    if (!scheduleModal) return;
    scheduleModal.classList.add('hidden');
    scheduleModal.setAttribute('aria-hidden', 'true');
  }

  function openScheduleEdit(schedule) {
    selectedSchedule = normalizeSchedule(schedule);
    if (!selectedSchedule || !scheduleEditModal) return;
    if (scheduleEditCadence) scheduleEditCadence.value = '';
    if (scheduleEditNotify) scheduleEditNotify.value = '';
    if (scheduleEditHour) scheduleEditHour.value = '';
    if (scheduleEditMinute) scheduleEditMinute.value = '';
    if (scheduleEditStatus) scheduleEditStatus.textContent = `Editing ${selectedSchedule.workflow_display_name} (${selectedSchedule.schedule_id})`;
    scheduleEditModal.classList.remove('hidden');
    scheduleEditModal.setAttribute('aria-hidden', 'false');
  }

  function closeScheduleEdit() {
    if (!scheduleEditModal) return;
    scheduleEditModal.classList.add('hidden');
    scheduleEditModal.setAttribute('aria-hidden', 'true');
  }

  async function renderScheduleInventory() {
    if (!scheduleInventoryList) return;
    renderScheduleAttentionSummary();
    const items = filteredSchedules().slice().sort((a, b) => String(a.next_run_at || '').localeCompare(String(b.next_run_at || '')));
    scheduleInventoryList.innerHTML = '';
    if (!items.length) {
      scheduleInventoryList.innerHTML = '<div class="text-gray-500 text-xs">No schedules for this filter.</div>';
      return;
    }
    items.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'rounded-xl border border-gray-800 bg-gray-900/50 p-3 space-y-2';

      const top = document.createElement('div');
      top.className = 'flex items-center justify-between gap-2';
      const title = document.createElement('div');
      title.className = 'text-sm font-semibold text-gray-100';
      title.textContent = item.workflow_display_name;
      top.appendChild(title);
      const badges = document.createElement('div');
      badges.className = 'flex items-center gap-1';
      badges.appendChild(scheduleStateBadge(item.state));
      if (item.analytics?.health) badges.appendChild(scheduleHealthBadge(item.analytics.health));
      if (item.analytics?.is_overdue) badges.appendChild(scheduleOverdueBadge(item.analytics.overdue_seconds));
      top.appendChild(badges);

      const meta = document.createElement('div');
      meta.className = 'text-[11px] text-gray-300';
      meta.textContent = `#${item.schedule_id_short} · Next: ${scheduleTimeLabel(item.next_run_at)} · ${item.cadence_summary} · notify=${item.notify_on}`;
      const analyticsLine = document.createElement('div');
      analyticsLine.className = 'text-[11px] text-gray-400';
      if (item.analytics) {
        const parts = [];
        if (item.analytics.trend_text) parts.push(item.analytics.trend_text);
        if (item.analytics.last_success_at) parts.push(`last success ${scheduleTimeLabel(item.analytics.last_success_at)}`);
        if (item.analytics.last_failure_at) parts.push(`last failure ${scheduleTimeLabel(item.analytics.last_failure_at)}`);
        analyticsLine.textContent = parts.join(' · ') || 'No recent analytics.';
      } else {
        analyticsLine.textContent = 'Analytics unavailable.';
      }

      const actions = document.createElement('div');
      actions.className = 'flex flex-wrap gap-2';
      const actionDefs = [
        ['Details', () => openScheduleDetails(item), 'bg-gray-800', false],
        ['Edit', () => openScheduleEdit(item), 'bg-gray-800', false],
      ];
      if (item.state === 'paused') actionDefs.push(['Resume', async () => { await performScheduleAction(item.schedule_id, 'resume'); await loadScheduleInventory(); }, 'bg-emerald-700', true]);
      else if (!['cancelled', 'completed'].includes(item.state)) actionDefs.push(['Pause', async () => { await performScheduleAction(item.schedule_id, 'pause'); await loadScheduleInventory(); }, 'bg-amber-700', true]);
      if (!['cancelled', 'completed'].includes(item.state)) actionDefs.push(['Cancel', async () => { await performScheduleAction(item.schedule_id, 'cancel'); await loadScheduleInventory(); }, 'bg-red-700', true]);

      actionDefs.forEach(([label, fn, cls, notifyAction]) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `px-2 py-1 rounded text-[10px] text-white ${cls} hover:opacity-90`;
        btn.textContent = label;
        btn.addEventListener('click', async () => {
          try {
            await fn();
            if (notifyAction) showToast(`Schedule ${label.toLowerCase()} succeeded.`);
          } catch (err) {
            if (scheduleInventoryStatus) scheduleInventoryStatus.textContent = err.message;
            if (notifyAction) showToast(`Schedule ${label.toLowerCase()} failed: ${err.message}`);
          }
        });
        actions.appendChild(btn);
      });

      row.appendChild(top);
      row.appendChild(meta);
      row.appendChild(analyticsLine);
      row.appendChild(actions);
      scheduleInventoryList.appendChild(row);
    });
  }

  async function loadScheduleInventory(options = {}) {
    if (scheduleInventoryStatus) scheduleInventoryStatus.textContent = 'Loading schedules…';
    try {
      await fetchScheduleInventory();
      await renderScheduleInventory();
      if (scheduleInventoryStatus) scheduleInventoryStatus.textContent = `Last refresh: ${new Date().toLocaleTimeString()}`;
      if (options.toast) showToast('Schedules refreshed.');
    } catch (err) {
      if (scheduleInventoryStatus) scheduleInventoryStatus.textContent = err.message || 'Failed to load schedules';
      if (options.toast) showToast(`Schedule refresh failed: ${err.message || 'unknown error'}`);
      if (scheduleInventoryList) scheduleInventoryList.innerHTML = '';
    }
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
    const agentTrace = agentTraceApi.extractAgentTrace
      ? agentTraceApi.extractAgentTrace(notification)
      : (notification.agent_trace || notification.agentTrace || null);
    const rawMessage = {
      ...notification,
      metadata: notification.metadata || (notification.raw && notification.raw.metadata) || {},
      context: notification.context || {},
    };
    const workflow = workflowUiApi.extractWorkflow ? workflowUiApi.extractWorkflow(rawMessage) : null;
    return {
      message_id: msgId,
      session_id: notification.session_id || notification.sessionId,
      created_at: notification.created_at || notification.createdAt,
      correlation_id: notification.correlation_id || notification.correlationId,
      severity: notification.severity || 'info',
      title: notification.title || 'New message from Orion',
      preview_text: notification.body_text || notification.preview_text || notification.previewText || '',
      full_text: notification.full_text || notification.fullText || notification.body_text || '',
      agent_trace: agentTrace,
      workflow: workflow,
      status: (notification.status || 'unread').toLowerCase(),
      silent: Boolean(notification.silent),
      raw: rawMessage,
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
        if (item.workflow) summary.appendChild(renderBadgeRow(buildWorkflowMetaBadges(item.workflow)));
        summary.appendChild(preview);

        const body = document.createElement('div');
        body.className = 'mt-2 space-y-2';

        const bodyText = document.createElement('div');
        bodyText.className = 'text-[11px] text-gray-300 whitespace-pre-wrap';
        bodyText.textContent = item.full_text || item.preview_text || '';

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
        const workflowPanel = createWorkflowPanel(item.workflow, {
          status: item.status,
          onRunAgain: async (workflow) => submitExplicitChatText(workflow.rerun_prompt),
        });
        if (workflowPanel) body.appendChild(workflowPanel);
        const tracePanel = createAgentTracePanel(item.agent_trace, { correlationId: item.correlation_id });
        if (tracePanel) body.appendChild(tracePanel);
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

  function normalizeTextForPreview(text) {
    if (!text) return '';
    const trimmed = String(text).trim();
    if (!trimmed) return '';
    const half = Math.floor(trimmed.length / 2);
    if (trimmed.length % 2 === 0) {
      const firstHalf = trimmed.slice(0, half);
      const secondHalf = trimmed.slice(half);
      if (firstHalf === secondHalf) return firstHalf.trim();
    }
    const blocks = trimmed.split(/\n\s*\n/).map((block) => block.trim()).filter(Boolean);
    if (blocks.length >= 2 && blocks[0] === blocks[1]) return blocks[0];
    return trimmed;
  }

  function truncate(text, maxChars = 220) {
    if (!text) return '';
    const value = String(text);
    return value.length > maxChars ? `${value.slice(0, maxChars - 1)}…` : value;
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

      const normalizedText = normalizeTextForPreview(n.body_text || n.preview_text || '');
      const previewText = truncate(normalizedText);
      const preview = document.createElement('div');
      preview.className = 'text-[11px] text-gray-300 line-clamp-3';
      preview.textContent = previewText;

      let details = null;
      if (normalizedText.length > previewText.length) {
        details = document.createElement('details');
        details.className = 'text-[11px] text-gray-300';
        const summary = document.createElement('summary');
        summary.className = 'cursor-pointer text-gray-400';
        summary.textContent = 'Expand';
        const bodyText = document.createElement('div');
        bodyText.className = 'mt-1 whitespace-pre-wrap';
        bodyText.textContent = normalizedText;
        details.appendChild(summary);
        details.appendChild(bodyText);
        details.addEventListener('toggle', () => {
          summary.textContent = details.open ? 'Collapse' : 'Expand';
        });
      }

      item.appendChild(header);
      item.appendChild(meta);
      item.appendChild(preview);
      if (details) item.appendChild(details);
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
      const resp = await fetch(`${API_BASE_URL}/api/attention/${attentionId}/ack`, {
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
      const resp = await fetch(`${API_BASE_URL}/api/chat/message/${messageId}/receipt`, {
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
    const meta = arguments.length > 3 && arguments[3] && typeof arguments[3] === 'object' ? arguments[3] : {};
    const headerRow = document.createElement('div');
    headerRow.className = 'mb-1 flex items-center justify-between gap-3';
    const header = document.createElement('p');
    header.className = `font-bold ${color}`;
    header.textContent = sender;
    headerRow.appendChild(header);
    if (sender === 'Orion') {
      const actionRow = document.createElement('div');
      actionRow.className = 'flex items-center gap-2';
      if (
        socialInspectionApi.shouldShowSocialInspection
        && socialInspectionApi.shouldShowSocialInspection(meta.routingDebug)
        && typeof window.syncSocialInspectionFromRouteDebug === 'function'
        && typeof window.openSocialInspectionModal === 'function'
      ) {
        const inspectionButton = document.createElement('button');
        inspectionButton.className = 'rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 text-[10px] font-semibold text-emerald-200 hover:bg-emerald-500/20';
        inspectionButton.type = 'button';
        inspectionButton.textContent = 'Social Inspect';
        inspectionButton.addEventListener('click', () => {
          window.syncSocialInspectionFromRouteDebug(meta.routingDebug);
          window.openSocialInspectionModal({
            routeDebug: meta.routingDebug,
            liveSnapshot: meta.routingDebug && meta.routingDebug.social_inspection,
            memorySnapshot: null,
            loadingMemory: false,
            error: '',
          });
        });
        actionRow.appendChild(inspectionButton);
      }
      if (actionRow.childNodes.length) {
        headerRow.appendChild(actionRow);
      }
    }
    const body = document.createElement('p');
    body.className = `${colorClass} whitespace-pre-wrap`;
    body.textContent = text || "";
    div.className = "mb-2 border-b border-gray-800/50 pb-2 last:border-0";
    if (sender === 'Orion') {
      const autonomyMeta = { ...meta, replyText: text || '' };
      updateAgentTraceDebugPanel(meta.agentTrace, meta);
      updateAutonomyDebugPanel(meta.autonomySummary || meta.autonomy_summary, meta.autonomyDebug || meta.autonomy_debug, autonomyMeta);
      const actionRow = document.createElement('div');
      actionRow.className = 'flex items-center gap-2';
      if (agentTraceApi.shouldShowAgentTrace && agentTraceApi.shouldShowAgentTrace(meta.agentTrace)) {
        const traceButton = document.createElement('button');
        const toolCount = Number(meta.agentTrace && meta.agentTrace.tool_call_count || 0);
        traceButton.className = 'rounded-full border border-indigo-500/40 bg-indigo-500/10 px-2 py-1 text-[10px] font-semibold text-indigo-200 hover:bg-indigo-500/20';
        traceButton.type = 'button';
        traceButton.textContent = toolCount > 0 ? `Agent: ${toolCount} tools` : 'Agent Trace';
        traceButton.addEventListener('click', () => openAgentTraceModal(meta.agentTrace, meta));
        actionRow.appendChild(traceButton);
      }
      if (socialInspectionApi.shouldShowSocialInspection && socialInspectionApi.shouldShowSocialInspection(meta.routingDebug)) {
        const inspectionButton = document.createElement('button');
        inspectionButton.className = 'rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 text-[10px] font-semibold text-emerald-200 hover:bg-emerald-500/20';
        inspectionButton.type = 'button';
        inspectionButton.textContent = 'Social Inspect';
        inspectionButton.addEventListener('click', () => {
          syncSocialInspectionFromRouteDebug(meta.routingDebug);
          openSocialInspectionModal({
            routeDebug: meta.routingDebug,
            liveSnapshot: meta.routingDebug.social_inspection,
            memorySnapshot: latestSocialInspectionState ? latestSocialInspectionState.memorySnapshot : null,
            loadingMemory: false,
            error: '',
          });
        });
        actionRow.appendChild(inspectionButton);
      }
      if (actionRow.childNodes.length) {
        headerRow.appendChild(actionRow);
      }
    }
    div.appendChild(headerRow);
    const workflowPanel = sender === 'Orion' ? createWorkflowPanel(meta.workflow, {
      onRunAgain: async (workflow) => submitExplicitChatText(workflow.rerun_prompt),
    }) : null;
    if (workflowPanel) div.appendChild(workflowPanel);
    div.appendChild(body);
    if (sender === 'Orion') {
      const autonomyPanel = createAutonomyPanel(
        meta.autonomySummary || meta.autonomy_summary,
        meta.autonomyDebug || meta.autonomy_debug,
        { ...meta, replyText: text || '' },
      );
      if (autonomyPanel) div.appendChild(autonomyPanel);
      const tracePanel = createAgentTracePanel(meta.agentTrace, meta);
      if (tracePanel) div.appendChild(tracePanel);
      const metacogPanel = createMetacogTracePanel(meta.metacogTraces || meta.metacog_traces || []);
      if (metacogPanel) div.appendChild(metacogPanel);
    }
    conversationDiv.appendChild(div);
    conversationDiv.scrollTop = conversationDiv.scrollHeight;
  }

  function closeAgentTraceModal() {
    if (!agentTraceModal) return;
    agentTraceModal.classList.add('hidden');
    agentTraceModal.setAttribute('aria-hidden', 'true');
  }

  function setAgentTraceEmptyState(isEmpty) {
    if (agentTraceEmptyState) agentTraceEmptyState.classList.toggle('hidden', !isEmpty);
    if (agentTraceContent) agentTraceContent.classList.toggle('hidden', !!isEmpty);
  }

  function buildAgentTraceOverviewNode(summary) {
    const root = document.createElement('div');
    root.className = 'grid gap-3 md:grid-cols-3 xl:grid-cols-6';
    const cards = [
      ['Status', summary.status || '--'],
      ['Duration', agentTraceApi.formatDuration ? agentTraceApi.formatDuration(summary.duration_ms) : '--'],
      ['Steps', summary.step_count ?? 0],
      ['Tool calls', summary.tool_call_count ?? 0],
      ['Unique tools', summary.unique_tool_count ?? 0],
      ['Families', Array.isArray(summary.unique_tool_families) && summary.unique_tool_families.length ? summary.unique_tool_families.join(', ') : '--'],
    ];
    cards.forEach(([label, value]) => {
      const card = document.createElement('div');
      card.className = 'rounded-xl border border-gray-700 bg-gray-800/50 p-3';
      const title = document.createElement('div');
      title.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      title.textContent = label;
      const val = document.createElement('div');
      val.className = 'mt-2 text-sm font-semibold text-gray-100';
      val.textContent = String(value ?? '--');
      card.appendChild(title);
      card.appendChild(val);
      root.appendChild(card);
    });
    return root;
  }

  function buildAgentTraceToolGroupsNode(summary) {
    const root = document.createElement('div');
    root.className = 'space-y-3';
    const groups = agentTraceApi.groupToolsByFamily ? agentTraceApi.groupToolsByFamily(summary.tools) : [];
    if (!groups.length) {
      const empty = document.createElement('div');
      empty.className = 'rounded-xl border border-dashed border-gray-700 bg-gray-900/50 p-3 text-xs text-gray-500';
      empty.textContent = 'No tool usage was captured for this trace.';
      root.appendChild(empty);
      return root;
    }
    groups.forEach((group) => {
      const wrap = document.createElement('div');
      wrap.className = 'rounded-xl border border-gray-700 bg-gray-900/40 p-3';

      const header = document.createElement('div');
      header.className = 'flex items-center justify-between gap-2';
      const title = document.createElement('div');
      title.className = 'text-sm font-semibold text-gray-100 capitalize';
      title.textContent = group.family || 'unknown';
      const meta = document.createElement('div');
      meta.className = 'text-[11px] text-gray-400';
      meta.textContent = `${group.count} call(s)`;
      header.appendChild(title);
      header.appendChild(meta);

      const list = document.createElement('div');
      list.className = 'mt-3 flex flex-wrap gap-2';
      (group.tools || []).forEach((tool) => {
        const chip = document.createElement('div');
        chip.className = 'rounded-lg border border-gray-700 bg-gray-800 px-2 py-1 text-[11px] text-gray-200';
        const durationLabel = agentTraceApi.formatDuration ? agentTraceApi.formatDuration(tool.duration_ms) : '--';
        chip.textContent = `${tool.tool_id} · ${tool.count} · ${durationLabel}`;
        list.appendChild(chip);
      });

      wrap.appendChild(header);
      wrap.appendChild(list);
      root.appendChild(wrap);
    });
    return root;
  }

  function buildAgentTraceTimelineNode(summary) {
    const wrapper = document.createElement('div');
    wrapper.className = 'overflow-hidden rounded-2xl border border-gray-700 bg-gray-900/40';
    const scroll = document.createElement('div');
    scroll.className = 'overflow-x-auto';
    const table = document.createElement('table');
    table.className = 'min-w-full divide-y divide-gray-800 text-left text-xs';
    table.innerHTML = `
      <thead class="bg-gray-900/70 text-gray-400">
        <tr>
          <th class="px-3 py-2 font-medium">#</th>
          <th class="px-3 py-2 font-medium">Event</th>
          <th class="px-3 py-2 font-medium">Tool</th>
          <th class="px-3 py-2 font-medium">Family</th>
          <th class="px-3 py-2 font-medium">Action</th>
          <th class="px-3 py-2 font-medium">Effect</th>
          <th class="px-3 py-2 font-medium">Status</th>
          <th class="px-3 py-2 font-medium">Duration</th>
          <th class="px-3 py-2 font-medium">Summary</th>
        </tr>
      </thead>
    `;
    const tbody = document.createElement('tbody');
    tbody.className = 'divide-y divide-gray-800 bg-gray-950/40';
    const rows = agentTraceApi.buildTimelineRows ? agentTraceApi.buildTimelineRows(summary) : [];
    if (!rows.length) {
      const row = document.createElement('tr');
      row.innerHTML = '<td class="px-3 py-3 text-gray-500" colspan="9">No timeline events were normalized for this trace.</td>';
      tbody.appendChild(row);
    } else {
      rows.forEach((entry) => {
        const row = document.createElement('tr');
        row.className = 'align-top';
        const cells = [
          entry.index,
          entry.event_type,
          entry.tool_id,
          entry.tool_family,
          entry.action_kind,
          entry.effect_kind,
          entry.status,
          entry.duration_label,
          entry.summary,
        ];
        cells.forEach((value, idx) => {
          const cell = document.createElement('td');
          cell.className = `px-3 py-2 ${idx === 8 ? 'max-w-md whitespace-pre-wrap text-gray-300' : 'whitespace-nowrap'}`;
          cell.textContent = String(value ?? '--');
          row.appendChild(cell);
        });
        tbody.appendChild(row);
      });
    }
    table.appendChild(tbody);
    scroll.appendChild(table);
    wrapper.appendChild(scroll);
    return wrapper;
  }

  function buildAgentTraceRawPayloadsNode(summary) {
    const details = document.createElement('details');
    details.className = 'rounded-xl border border-gray-700 bg-gray-900/40';
    const summaryRow = document.createElement('summary');
    summaryRow.className = 'cursor-pointer px-3 py-2 text-xs uppercase tracking-wide text-gray-400';
    summaryRow.textContent = 'Raw payloads';

    const body = document.createElement('div');
    body.className = 'space-y-3 border-t border-gray-800 px-3 py-3';

    const summaryBlock = document.createElement('div');
    const summaryTitle = document.createElement('div');
    summaryTitle.className = 'mb-1 text-[10px] uppercase tracking-wide text-gray-500';
    summaryTitle.textContent = 'Summary';
    const summaryPre = document.createElement('pre');
    summaryPre.className = 'overflow-x-auto rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-[11px] text-gray-200';
    summaryPre.textContent = JSON.stringify(summary, null, 2);
    summaryBlock.appendChild(summaryTitle);
    summaryBlock.appendChild(summaryPre);

    const rawBlock = document.createElement('div');
    const rawTitle = document.createElement('div');
    rawTitle.className = 'mb-1 text-[10px] uppercase tracking-wide text-gray-500';
    rawTitle.textContent = 'Raw';
    const rawPre = document.createElement('pre');
    rawPre.className = 'overflow-x-auto rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-[11px] text-gray-200';
    rawPre.textContent = JSON.stringify((summary && summary.raw) || {}, null, 2);
    rawBlock.appendChild(rawTitle);
    rawBlock.appendChild(rawPre);

    body.appendChild(summaryBlock);
    body.appendChild(rawBlock);
    details.appendChild(summaryRow);
    details.appendChild(body);
    return details;
  }

  function populateAgentTraceModal(summary) {
    if (!agentTraceOverview || !agentTraceToolGroups || !agentTraceTimelineBody) return;
    agentTraceOverview.innerHTML = '';
    Array.from(buildAgentTraceOverviewNode(summary).children).forEach((child) => agentTraceOverview.appendChild(child));
    if (agentTraceSummary) {
      agentTraceSummary.textContent = summary.summary_text || 'No deterministic summary available.';
    }
    agentTraceToolGroups.innerHTML = '';
    Array.from(buildAgentTraceToolGroupsNode(summary).children).forEach((child) => agentTraceToolGroups.appendChild(child));
    agentTraceTimelineBody.innerHTML = '';
    const timeline = buildAgentTraceTimelineNode(summary);
    const replacementBody = timeline.querySelector('tbody');
    if (replacementBody) {
      Array.from(replacementBody.children).forEach((row) => agentTraceTimelineBody.appendChild(row));
    }
    if (agentTraceRawSummary) {
      agentTraceRawSummary.textContent = JSON.stringify(summary, null, 2);
    }
    if (agentTraceRawPayloads) {
      agentTraceRawPayloads.textContent = JSON.stringify((summary && summary.raw) || {}, null, 2);
    }
  }

  function createAgentTracePanel(summary, meta = {}) {
    if (!agentTraceApi.shouldShowAgentTrace || !agentTraceApi.shouldShowAgentTrace(summary)) return null;

    const panel = document.createElement('div');
    panel.className = 'mt-3 rounded-xl border border-gray-700 bg-gray-800/40 p-3 space-y-2';

    const header = document.createElement('div');
    header.className = 'flex items-center justify-between gap-2';

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'flex-1 text-left text-xs text-gray-300 hover:text-white';
    toggle.innerHTML = '<span class="uppercase tracking-wide">Agent Trace</span>';

    const headerActions = document.createElement('div');
    headerActions.className = 'flex items-center gap-2';

    const expand = document.createElement('button');
    expand.type = 'button';
    expand.className = 'text-[10px] text-indigo-300 hover:text-indigo-200';
    expand.textContent = 'Open modal';
    expand.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openAgentTraceModal(summary, meta);
    });

    const caret = document.createElement('span');
    caret.className = 'text-gray-500';
    caret.textContent = '▾';

    headerActions.appendChild(expand);
    headerActions.appendChild(caret);
    header.appendChild(toggle);
    header.appendChild(headerActions);

    const body = document.createElement('div');
    body.className = 'hidden space-y-4 text-xs text-gray-300';

    const metaLine = document.createElement('div');
    metaLine.className = 'text-[10px] text-gray-500';
    const corr = summary.corr_id || meta.correlationId || '--';
    metaLine.textContent = `corr ${corr} · status ${summary.status || '--'} · ${summary.step_count || 0} steps`;

    const summaryBlock = document.createElement('div');
    const summaryTitle = document.createElement('div');
    summaryTitle.className = 'mb-1 text-gray-500';
    summaryTitle.textContent = 'Deterministic summary';
    const summaryText = document.createElement('div');
    summaryText.className = 'rounded-xl border border-gray-700 bg-gray-900/50 p-3 text-sm text-gray-200';
    summaryText.textContent = summary.summary_text || 'No deterministic summary available.';
    summaryBlock.appendChild(summaryTitle);
    summaryBlock.appendChild(summaryText);

    const toolsBlock = document.createElement('div');
    const toolsTitle = document.createElement('div');
    toolsTitle.className = 'mb-1 text-gray-500';
    toolsTitle.textContent = 'Grouped tool usage';
    toolsBlock.appendChild(toolsTitle);
    toolsBlock.appendChild(buildAgentTraceToolGroupsNode(summary));

    const timelineBlock = document.createElement('div');
    const timelineTitle = document.createElement('div');
    timelineTitle.className = 'mb-1 text-gray-500';
    timelineTitle.textContent = 'Timeline';
    timelineBlock.appendChild(timelineTitle);
    timelineBlock.appendChild(buildAgentTraceTimelineNode(summary));

    body.appendChild(metaLine);
    body.appendChild(buildAgentTraceOverviewNode(summary));
    body.appendChild(summaryBlock);
    body.appendChild(toolsBlock);
    body.appendChild(timelineBlock);
    body.appendChild(buildAgentTraceRawPayloadsNode(summary));

    toggle.addEventListener('click', () => {
      const nextHidden = !body.classList.contains('hidden');
      body.classList.toggle('hidden', nextHidden);
      caret.textContent = nextHidden ? '▾' : '▴';
    });

    panel.appendChild(header);
    panel.appendChild(body);
    return panel;
  }

  function createAutonomyPanel(summary, debug, meta = {}) {
    const model = normalizeAutonomyModel(summary, debug, meta);
    if (!model || !shouldRenderAutonomyInline(model)) return null;

    const panel = document.createElement('div');
    panel.className = 'mt-3 rounded-xl border border-violet-500/30 bg-violet-500/10 p-3 space-y-2';

    const headerRow = document.createElement('div');
    headerRow.className = 'flex flex-wrap items-center justify-between gap-2';
    const title = document.createElement('div');
    title.className = 'text-xs uppercase tracking-wide font-semibold text-violet-200';
    title.textContent = 'Autonomy';
    headerRow.appendChild(title);

    const badges = document.createElement('div');
    badges.className = 'flex flex-wrap gap-1';
    [
      `backend:${model.backend}`,
      `availability:${model.availability.available}/${model.availability.subjects}`,
      'proposal-only',
    ].forEach((value) => {
      const badge = document.createElement('span');
      badge.className = 'rounded-full border border-violet-400/40 bg-violet-500/15 px-2 py-0.5 text-[10px] text-violet-100';
      badge.textContent = value;
      badges.appendChild(badge);
    });
    headerRow.appendChild(badges);
    panel.appendChild(headerRow);

    [
      ['dominant drive', model.dominantDrive || '--'],
      ['top drives', model.topDrives.length ? model.topDrives.join(', ') : '--'],
      ['top tensions', model.tensions.length ? model.tensions.join(', ') : '--'],
      ['proposal headlines', model.proposals.length ? model.proposals.join(' · ') : '--'],
      ['alignment', model.alignment.alignment_note],
    ].forEach(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'text-xs text-violet-100/90';
      const prefix = document.createElement('span');
      prefix.className = 'uppercase tracking-wide text-[10px] text-violet-300 mr-1';
      prefix.textContent = `${label}:`;
      row.appendChild(prefix);
      row.appendChild(document.createTextNode(` ${value}`));
      panel.appendChild(row);
    });

    const debugButton = document.createElement('button');
    debugButton.type = 'button';
    debugButton.className = 'rounded-full border border-violet-400/40 bg-violet-500/20 px-2 py-1 text-[10px] font-semibold text-violet-100 hover:bg-violet-500/30';
    debugButton.textContent = 'Open debug';
    debugButton.addEventListener('click', () => {
      if (autonomyDebugPanel) autonomyDebugPanel.classList.remove('hidden');
      if (autonomyDebugBody) autonomyDebugBody.classList.remove('hidden');
      if (autonomyDebugCaret) autonomyDebugCaret.textContent = '▴';
    });
    panel.appendChild(debugButton);
    return panel;
  }

  function createMetacogTracePanel(traces) {
    const list = Array.isArray(traces) ? traces.filter((item) => item && typeof item === 'object') : [];
    if (!list.length) return null;
    const top = list[0];

    const panel = document.createElement('div');
    panel.className = 'mt-3 rounded-xl border border-purple-700/60 bg-purple-900/10 p-3 space-y-2';

    const header = document.createElement('div');
    header.className = 'text-xs uppercase tracking-wide text-purple-300';
    header.textContent = 'METACOG TRACE';

    const summary = document.createElement('div');
    summary.className = 'text-xs text-gray-200 whitespace-pre-wrap';
    const raw = String(top.content || '').trim();
    summary.textContent = raw.length > 280 ? `${raw.slice(0, 280)}…` : raw || 'No trace content.';

    const meta = document.createElement('div');
    meta.className = 'text-[10px] text-purple-200/80';
    const md = top.metadata && typeof top.metadata === 'object' ? top.metadata : {};
    const reasoningDepth = md.reasoning_depth != null ? md.reasoning_depth : '--';
    const tokenCount = top.token_count != null ? top.token_count : '--';
    meta.textContent = `role ${top.trace_role || '--'} · tokens ${tokenCount} · reasoning depth ${reasoningDepth}`;

    panel.appendChild(header);
    panel.appendChild(summary);
    panel.appendChild(meta);
    return panel;
  }

  function openAgentTraceModal(summary, meta = {}) {
    if (!agentTraceModal) return;
    if (!agentTraceApi.shouldShowAgentTrace || !agentTraceApi.shouldShowAgentTrace(summary)) {
      setAgentTraceEmptyState(true);
      if (agentTraceModalMeta) {
        agentTraceModalMeta.textContent = meta.correlationId ? `corr ${meta.correlationId}` : 'No trace data available.';
      }
      agentTraceModal.classList.remove('hidden');
      agentTraceModal.setAttribute('aria-hidden', 'false');
      return;
    }
    setAgentTraceEmptyState(false);
    if (agentTraceModalMeta) {
      const corr = summary.corr_id || meta.correlationId || '--';
      agentTraceModalMeta.textContent = `corr ${corr} · status ${summary.status || '--'} · ${summary.step_count || 0} steps`;
    }
    populateAgentTraceModal(summary);
    agentTraceModal.classList.remove('hidden');
    agentTraceModal.setAttribute('aria-hidden', 'false');
  }

  async function loadNotifications() {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/notifications?limit=50`);
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
      const profileResp = await fetch(`${API_BASE_URL}/api/notify/recipients/${RECIPIENT_GROUP}`);
      if (profileResp.ok) {
        const profile = await profileResp.json();
        if (notifyDisplayName) notifyDisplayName.value = profile.display_name || '';
        if (notifyTimezone) notifyTimezone.value = profile.timezone || '';
        if (notifyQuietEnabled) notifyQuietEnabled.checked = Boolean(profile.quiet_hours_enabled);
        if (notifyQuietStart) notifyQuietStart.value = profile.quiet_start_local || '22:00';
        if (notifyQuietEnd) notifyQuietEnd.value = profile.quiet_end_local || '07:00';
      }
      const prefsResp = await fetch(`${API_BASE_URL}/api/notify/recipients/${RECIPIENT_GROUP}/preferences`);
      if (prefsResp.ok) {
        const prefs = await prefsResp.json();
        if (Array.isArray(prefs)) applyPreferenceRows(prefs);
      }
      setSettingsStatus('Loaded');
    } catch (err) {
      setSettingsStatus('Failed to load settings', true);
      console.warn('Failed to load notify settings', err);
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
      await fetch(`${API_BASE_URL}/api/notify/recipients/${RECIPIENT_GROUP}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profilePayload),
      });

      const preferencesPayload = { preferences: readPreferenceRows() };
      const prefResp = await fetch(`${API_BASE_URL}/api/notify/recipients/${RECIPIENT_GROUP}/preferences`, {
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
      const resp = await fetch(`${API_BASE_URL}/api/chat/messages?limit=50&status=${statusParam}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data)) {
        chatMessages = data
          .map((item) => normalizeChatMessage(item))
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
      const resp = await fetch(`${API_BASE_URL}/api/attention?status=pending&limit=50`);
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

  if (clearButton && conversationDiv) {
    clearButton.addEventListener('click', () => {
      conversationDiv.innerHTML = '';
      clearAgentTraceDebugPanel();
      clearAutonomyDebugPanel();
    });
  }
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
  if (agentTraceDebugToggle) {
    agentTraceDebugToggle.addEventListener('click', toggleAgentTraceDebugPanel);
  }
  if (autonomyDebugToggle) {
    autonomyDebugToggle.addEventListener('click', toggleAutonomyDebugPanel);
  }
  ensureAutonomyModalRootOnBody();
  if (autonomyDebugOpenModal) {
    autonomyDebugOpenModal.addEventListener('click', openAutonomyDebugModal);
  }
  if (autonomyDebugModalClose) {
    autonomyDebugModalClose.addEventListener('click', closeAutonomyDebugModal);
  }
  if (autonomyDebugModalBackdrop) {
    autonomyDebugModalBackdrop.addEventListener('click', closeAutonomyDebugModal);
  }
  if (autonomyDebugModalRoot) {
    autonomyDebugModalRoot.addEventListener('click', (event) => {
      if (event.target === autonomyDebugModalRoot) closeAutonomyDebugModal();
    });
  }
  if (autonomyDebugModalDialog) {
    autonomyDebugModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  if (socialInspectionOpen) {
    socialInspectionOpen.addEventListener('click', () => openSocialInspectionModal());
  }
  if (socialInspectionModalClose) {
    socialInspectionModalClose.addEventListener('click', closeSocialInspectionModal);
  }
  if (socialInspectionModal) {
    socialInspectionModal.addEventListener('click', (event) => {
      if (event.target === socialInspectionModal) closeSocialInspectionModal();
    });
  }
  if (agentTraceModalClose) {
    agentTraceModalClose.addEventListener('click', closeAgentTraceModal);
  }
  if (workflowModalClose) {
    workflowModalClose.addEventListener('click', closeWorkflowModal);
  }
  if (scheduleRefreshButton) {
    scheduleRefreshButton.addEventListener('click', () => loadScheduleInventory({ toast: true }));
  }
  if (scheduleFilter) {
    scheduleFilter.addEventListener('change', () => renderScheduleInventory());
  }
  if (scheduleModalClose) {
    scheduleModalClose.addEventListener('click', closeScheduleModal);
  }
  if (scheduleModal) {
    scheduleModal.addEventListener('click', (event) => {
      if (event.target === scheduleModal) closeScheduleModal();
    });
  }
  if (scheduleEditModalClose) {
    scheduleEditModalClose.addEventListener('click', closeScheduleEdit);
  }
  if (scheduleEditModal) {
    scheduleEditModal.addEventListener('click', (event) => {
      if (event.target === scheduleEditModal) closeScheduleEdit();
    });
  }
  if (scheduleEditSave) {
    scheduleEditSave.addEventListener('click', async () => {
      if (!selectedSchedule) return;
      const patch = {};
      if (scheduleEditCadence && scheduleEditCadence.value) patch.cadence = scheduleEditCadence.value;
      if (scheduleEditNotify && scheduleEditNotify.value) patch.notify_on = scheduleEditNotify.value;
      if (scheduleEditHour && scheduleEditHour.value !== '') patch.hour_local = Number(scheduleEditHour.value);
      if (scheduleEditMinute && scheduleEditMinute.value !== '') patch.minute_local = Number(scheduleEditMinute.value);
      if (selectedSchedule && Number.isFinite(selectedSchedule.revision)) patch.expected_revision = selectedSchedule.revision;
      if (scheduleEditStatus) scheduleEditStatus.textContent = `Saving (rev ${selectedSchedule.revision})…`;
      try {
        await performScheduleAction(selectedSchedule.schedule_id, 'update', patch);
        if (scheduleEditStatus) scheduleEditStatus.textContent = 'Saved.';
        showToast('Schedule update saved.');
        closeScheduleEdit();
        await loadScheduleInventory();
      } catch (err) {
        const msg = String(err.message || 'Update failed');
        const code = String(err.code || '');
        const isConflict = code === 'schedule_revision_conflict';
        if (scheduleEditStatus) scheduleEditStatus.textContent = isConflict ? `${msg} Refresh and retry.` : msg;
        showToast(isConflict ? 'Schedule changed elsewhere. Refresh and retry.' : msg);
      }
    });
  }
  if (agentTraceModal) {
    agentTraceModal.addEventListener('click', (event) => {
      if (event.target === agentTraceModal) closeAgentTraceModal();
    });
  }
  if (workflowModal) {
    workflowModal.addEventListener('click', (event) => {
      if (event.target === workflowModal) closeWorkflowModal();
    });
  }
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && socialInspectionModal && !socialInspectionModal.classList.contains('hidden')) {
      closeSocialInspectionModal();
      return;
    }
    if (event.key === 'Escape' && agentTraceModal && !agentTraceModal.classList.contains('hidden')) {
      closeAgentTraceModal();
      return;
    }
    if (event.key === 'Escape' && workflowModal && !workflowModal.classList.contains('hidden')) {
      closeWorkflowModal();
      return;
    }
    if (event.key === 'Escape' && autonomyDebugModalRoot && !autonomyDebugModalRoot.classList.contains('hidden')) {
      closeAutonomyDebugModal();
      return;
    }
    if (event.key === 'Escape' && scheduleModal && !scheduleModal.classList.contains('hidden')) {
      closeScheduleModal();
      return;
    }
    if (event.key === 'Escape' && scheduleEditModal && !scheduleEditModal.classList.contains('hidden')) {
      closeScheduleEdit();
    }
  });

  renderSocialInspectionState(null);
  loadScheduleInventory();

  // --- WebSocket ---
  function setupWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}${URL_PREFIX}/ws`;
    
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
          if (d.llm_response) {
            appendMessage('Orion', d.llm_response, 'text-white', {
              agentTrace: d.agent_trace,
              metacogTraces: d.metacog_traces,
              correlationId: d.correlation_id,
              routingDebug: d.routing_debug,
              workflow: d.workflow,
              autonomySummary: d.autonomy_summary,
              autonomyDebug: d.autonomy_debug,
              autonomyStatePreview: d.autonomy_state_preview,
              autonomyBackend: d.autonomy_backend,
              autonomySelectedSubject: d.autonomy_selected_subject,
            });
            updateMemoryPanelFromResponse(d);
            syncSocialInspectionFromRouteDebug(d.routing_debug);
          }
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

  async function submitExplicitChatText(text) {
    const value = String(text || '').trim();
    if (!value) return;
    appendMessage('You', value);
    if (chatInput) chatInput.value = '';

    const recallMode = recallModeSelect ? recallModeSelect.value : "auto";
    const recallProfile = recallProfileSelect ? recallProfileSelect.value : "auto";
    const payload = {
       text_input: value,
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

        fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({messages:[{role:'user', content:value}], ...payload})
        })
        .then(r => r.json())
        .then(d => {
            if(d.text) {
              appendMessage('Orion', d.text, 'text-white', {
                agentTrace: d.agent_trace,
                metacogTraces: d.metacog_traces,
                correlationId: d.correlation_id,
                routingDebug: d.routing_debug,
                workflow: d.workflow,
                autonomySummary: d.autonomy_summary,
                autonomyDebug: d.autonomy_debug,
                autonomyStatePreview: d.autonomy_state_preview,
                autonomyBackend: d.autonomy_backend,
                autonomySelectedSubject: d.autonomy_selected_subject,
              });
              syncSocialInspectionFromRouteDebug(d.routing_debug);
            } else if(d.error) appendMessage('System', d.error, 'text-red-400');
            updateMemoryPanelFromResponse(d);
        })
        .catch(e => appendMessage('System', "HTTP Failed: " + e.message, 'text-red-400'));
    }
  }

  async function sendTextMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    await submitExplicitChatText(text);
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
       const r = await fetch(`${API_BASE_URL}/api/session`, {headers: sid ? {'X-Orion-Session-Id': sid} : {}});
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
      const result = await topicFoundryFetch("/ready");
      const checks = result?.checks || {};
      formatStatusBadge(tsStatusBadge, result.ok, result.ok ? "Healthy" : "Degraded");
      formatStatusBadge(tsStatusPg, checks.pg?.ok, checks.pg?.ok ? "ok" : "fail");
      formatStatusBadge(tsStatusEmbedding, checks.embedding?.ok, checks.embedding?.ok ? "ok" : "fail");
      formatStatusBadge(tsStatusModelDir, checks.model_dir?.ok, checks.model_dir?.ok ? "ok" : "fail");
      if (tsStatusDetail) {
        tsStatusDetail.textContent = `PG: ${checks.pg?.detail || "--"} · Embedding: ${checks.embedding?.detail || "--"} · Model dir: ${checks.model_dir?.detail || "--"}`;
      }
      setLoading(tsStatusLoading, false);
    } catch (err) {
      formatStatusBadge(tsStatusBadge, false, "Unreachable");
      formatStatusBadge(tsStatusPg, null, "--");
      formatStatusBadge(tsStatusEmbedding, null, "--");
      formatStatusBadge(tsStatusModelDir, null, "--");
      renderError(tsStatusDetail, err, "Failed to read /ready.");
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

  function applyCapabilityDefaults(defaults = {}) {
    if (tsModelEmbeddingUrl && !tsModelEmbeddingUrl.value) {
      tsModelEmbeddingUrl.value = defaults.embedding_source_url || "";
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
      const result = await topicFoundryFetch("/capabilities");
      topicStudioCapabilities = result;
      const modes = result.segmentation_modes_supported || [];
      renderSegmentationModes(modes, Boolean(result.llm_enabled));
      applyCapabilityDefaults(result.defaults || {});
      setLoading(tsStatusLoading, false);
    } catch (err) {
      const fallbackModes = ["time_gap", "semantic", "hybrid"];
      renderSegmentationModes(fallbackModes, false);
      if (tsCapabilitiesWarning) {
        tsCapabilitiesWarning.textContent = `Capabilities unavailable. Falling back to safe defaults. ${err.status ? `status ${err.status}` : ""} ${err.body || err.message || ""}`.trim();
        tsCapabilitiesWarning.classList.remove("hidden");
      }
      setLoading(tsStatusLoading, false);
    }
  }

  async function refreshTopicStudio() {
    if (topicFoundryBaseLabel) {
      topicFoundryBaseLabel.textContent = TOPIC_FOUNDRY_PROXY_BASE;
    }
    await refreshTopicStudioCapabilities();
    await refreshTopicStudioStatus();
    try {
      const datasetsResponse = await topicFoundryFetch("/datasets");
      topicStudioDatasets = datasetsResponse?.datasets || [];
      renderDatasetOptions();
      renderConversationDatasetOptions();
    } catch (err) {
      console.warn("[TopicStudio] Failed to load datasets", err);
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
    } catch (err) {
      console.warn("[TopicStudio] Failed to load runs", err);
      if (tsRunsWarning) {
        tsRunsWarning.textContent = `Failed to load runs. Enter a run id manually. ${err.status ? `status ${err.status}` : ""} ${err.body || err.message || ""}`.trim();
        tsRunsWarning.classList.remove("hidden");
      }
    }
    setTopicStudioSubview(resolveTopicStudioSubview());
  }

  applyTopicStudioState();
  bindTopicStudioPersistence();
  if (tsUsePreviewSpec) {
    tsUsePreviewSpec.disabled = true;
  }
  setTopicStudioSubview(resolveTopicStudioSubview());

  if (hubTabButton && topicStudioTabButton) {
    hubTabButton.addEventListener("click", () => {
      setActiveTab("hub");
      history.replaceState(null, "", "#hub");
    });
    topicStudioTabButton.addEventListener("click", () => {
      setActiveTab("topic-studio");
      history.replaceState(null, "", "#topic-studio");
      refreshTopicStudio();
    });
    if (window.location.hash === "#topic-studio") {
      setActiveTab("topic-studio");
      refreshTopicStudio();
    } else {
      setActiveTab("hub");
    }
  }

  if (tsDatasetSelect) {
    tsDatasetSelect.addEventListener("change", () => {
      const selected = topicStudioDatasets.find((dataset) => dataset.dataset_id === tsDatasetSelect.value);
      if (selected) {
        populateDatasetForm(selected);
      }
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

  if (tsCreateDataset) {
    tsCreateDataset.addEventListener("click", async () => {
      try {
        const payload = buildDatasetSpec();
        const response = await topicFoundryFetch("/datasets", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        showToast("Dataset created.");
        await refreshTopicStudio();
        if (response?.dataset_id && tsDatasetSelect) {
          tsDatasetSelect.value = response.dataset_id;
        }
      } catch (err) {
        showToast("Failed to create dataset.");
      }
    });
  }

  if (tsPreviewDataset) {
    tsPreviewDataset.addEventListener("click", async () => {
      try {
        setLoading(tsPreviewLoading, true);
        clearPreview();
        const payload = {
          dataset: buildDatasetSpec(),
          windowing: buildWindowingSpec(),
          start_at: parseDateInput(tsStartAt?.value),
          end_at: parseDateInput(tsEndAt?.value),
          limit: 200,
        };
        await executePreview(payload);
      } catch (err) {
        renderError(tsPreviewError, err, "Failed to preview dataset.");
        showToast("Failed to preview dataset.");
        setWarning(tsPreviewWarning, null);
        setLoading(tsPreviewLoading, false);
      }
    });
  }

  if (tsCreateModel) {
    tsCreateModel.addEventListener("click", async () => {
      if (!tsDatasetSelect?.value) {
        showToast("Select a dataset before creating a model.");
        return;
      }
      try {
        const payload = {
          name: tsModelName?.value?.trim() || "",
          version: tsModelVersion?.value?.trim() || "",
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
        await topicFoundryFetch("/models", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        showToast("Model created.");
        await refreshTopicStudio();
      } catch (err) {
        showToast("Failed to create model.");
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
      try {
        setLoading(tsRunLoading, true);
        const result = await topicFoundryFetch("/runs/train", {
          method: "POST",
          body: JSON.stringify({
            model_id: model.model_id,
            dataset_id: model.dataset_id,
            start_at: parseDateInput(tsStartAt?.value),
            end_at: parseDateInput(tsEndAt?.value),
          }),
        });
        if (tsRunId) tsRunId.value = result.run_id;
        if (tsRunError) tsRunError.textContent = "--";
        renderRunStatus(result);
        startRunPolling(result.run_id);
        await refreshTopicStudio();
      } catch (err) {
        renderError(tsRunError, err);
        showToast("Failed to start training run.");
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
        renderError(tsRunError, err);
        showToast("Failed to poll run status.");
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
        const result = await topicFoundryFetch(`/runs/${tsRunId.value}/enrich`, {
          method: "POST",
          body: JSON.stringify(payload),
        });
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
        renderError(tsEnrichStatus, err);
        showToast("Failed to start enrichment.");
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
      row.innerHTML = '<td class="py-3 text-gray-500" colspan="9">No segments found.</td>';
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
      const bounds = startAt === "--" && endAt === "--" ? "--" : `${startAt} → ${endAt}`;
      const snippet = truncate(segment.snippet || "", 200) || "--";
      const rowIdsCount = segment.row_ids_count ?? "--";
      row.innerHTML = `
        <td class="py-2 pr-3 text-indigo-300 cursor-pointer" data-segment-id="${segment.segment_id}">${segment.segment_id}</td>
        <td class="py-2 pr-3">${segment.size ?? "--"}</td>
        <td class="py-2 pr-3">${rowIdsCount}</td>
        <td class="py-2 pr-3">${segment.title || segment.label || "--"}</td>
        <td class="py-2 pr-3">${aspectChips}</td>
        <td class="py-2 pr-3">${bounds}</td>
        <td class="py-2 pr-3">${snippet}</td>
        <td class="py-2 pr-3">${sentiment.friction ?? "--"}</td>
        <td class="py-2 pr-3">${sentiment.valence ?? "--"}</td>
      `;
      row.querySelector("[data-segment-id]")?.addEventListener("click", async () => {
        try {
          topicStudioSelectedSegmentId = segment.segment_id;
          const detail = await topicFoundryFetch(`/segments/${segment.segment_id}`);
          const meaning = detail.meaning || {};
          const provenance = detail.provenance || {};
          const rowIds = Array.isArray(provenance.row_ids) ? provenance.row_ids.length : provenance.row_ids_count;
          const snippet = detail.snippet || detail.text || provenance.snippet || provenance.text || null;
          const detailPayload = {
            segment_id: detail.segment_id,
            title: detail.title || detail.label || null,
            meaning: {
              intent: meaning.intent || null,
              outcome: meaning.outcome || null,
              questions: meaning.questions || null,
              next_steps: meaning.next_steps || null,
            },
            provenance: {
              row_ids_count: rowIds ?? null,
              start_at: detail.start_at || provenance.start_at || null,
              end_at: detail.end_at || provenance.end_at || null,
            },
            snippet: snippet || null,
          };
          if (tsSegmentDetail) {
            tsSegmentDetail.textContent = JSON.stringify(detailPayload, null, 2);
          }
        } catch (err) {
          renderError(tsSegmentDetail, err);
        }
      });
      tsSegmentsTableBody.appendChild(row);
    });
  }

  async function showSegmentDetailFromId(segmentId) {
    if (!segmentId) return;
    try {
      const detail = await topicFoundryFetch(`/segments/${segmentId}`);
      const meaning = detail.meaning || {};
      const provenance = detail.provenance || {};
      const rowIds = Array.isArray(provenance.row_ids) ? provenance.row_ids.length : provenance.row_ids_count;
      const snippet = detail.snippet || detail.text || provenance.snippet || provenance.text || null;
      const detailPayload = {
        segment_id: detail.segment_id,
        title: detail.title || detail.label || null,
        meaning: {
          intent: meaning.intent || null,
          outcome: meaning.outcome || null,
          questions: meaning.questions || null,
          next_steps: meaning.next_steps || null,
        },
        provenance: {
          row_ids_count: rowIds ?? null,
          start_at: detail.start_at || provenance.start_at || null,
          end_at: detail.end_at || provenance.end_at || null,
        },
        snippet: snippet || null,
      };
      if (tsSegmentDetail) {
        tsSegmentDetail.textContent = JSON.stringify(detailPayload, null, 2);
      }
      setTopicStudioSubview("runs");
    } catch (err) {
      renderError(tsSegmentDetail, err);
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
        <td class="py-2 pr-3">${topic.count ?? "--"}</td>
        <td class="py-2 pr-3">${pct}</td>
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
      row.innerHTML = `
        <td class="py-2 pr-3">${segment.segment_id ?? "--"}</td>
        <td class="py-2 pr-3">${segment.size ?? "--"}</td>
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
      topicStudioEventsPage = response.items || [];
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
      topicStudioKgEdgesPage = response.items || [];
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
      const response = await topicFoundryFetch(`/topics?${params.toString()}`);
      const items = response.items || response.topics || [];
      renderTopicsTable(items);
      if (tsTopicsStatus) {
        tsTopicsStatus.textContent = `Loaded ${items.length} topics.`;
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
      const segments = response.items || response.segments || [];
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
    try {
      setLoading(tsSegmentsLoading, true);
      if (tsSegmentsError) tsSegmentsError.textContent = "--";
      const queryKey = `${tsSegmentsRunId.value}|${tsSegmentsEnrichment?.value || ""}|${tsSegmentsAspect?.value || ""}|${topicStudioSegmentsLimit}`;
      if (topicStudioSegmentsQueryKey !== queryKey) {
        topicStudioSegmentsQueryKey = queryKey;
        resetSegmentsPaging();
      }
      const sortValue = tsSegmentsSort?.value || "time_desc";
      let sortBy = "created_at";
      let sortDir = "desc";
      if (sortValue === "friction_desc") {
        sortBy = "friction";
      } else if (sortValue === "size_desc") {
        sortBy = "size";
      } else if (sortValue === "time_asc") {
        sortDir = "asc";
      }
      const params = new URLSearchParams({
        run_id: tsSegmentsRunId.value,
        include_snippet: "true",
        include_bounds: "true",
        limit: String(topicStudioSegmentsLimit),
        offset: String(topicStudioSegmentsOffset),
        format: "wrapped",
        sort_by: sortBy,
        sort_dir: sortDir,
      });
      if (tsSegmentsEnrichment?.value) {
        params.set("has_enrichment", tsSegmentsEnrichment.value);
      }
      if (tsSegmentsAspect?.value) {
        params.set("aspect", tsSegmentsAspect.value);
      }
      const query = tsSegmentsSearch?.value?.trim();
      if (query) {
        params.set("q", query);
      }
      const { payload, headers } = await topicFoundryFetchWithHeaders(`/segments?${params.toString()}`);
      const items = payload.items || payload.segments || payload;
      topicStudioSegmentsPage = Array.isArray(items) ? items : [];
      const totalValue = Number(payload.total);
      if (Number.isFinite(totalValue)) {
        topicStudioSegmentsTotal = totalValue;
      } else {
        const headerTotal = Number(headers.get("X-Total-Count"));
        topicStudioSegmentsTotal = Number.isFinite(headerTotal) ? headerTotal : null;
      }
      const filtered = applySegmentsClientFilters(topicStudioSegmentsPage);
      topicStudioSegmentsDisplayed = filtered;
      renderSegmentsTable(filtered);
      renderSegmentsFacets(topicStudioSegmentsLastFacets);
      topicStudioLastSubview = "runs";
      saveTopicStudioState();
      updateSegmentsRange();
      setLoading(tsSegmentsLoading, false);
      refreshSegmentFacets();
    } catch (err) {
      renderError(tsSegmentsError, err);
      showToast("Failed to load segments.");
      setLoading(tsSegmentsLoading, false);
    } finally {
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
      topicStudioSegmentsLastFacets = facets;
      renderSegmentsFacets(facets);
    } catch (err) {
      console.warn("[TopicStudio] Failed to load segment facets", err);
    }
  }

  if (tsLoadSegments) {
    tsLoadSegments.addEventListener("click", loadSegments);
  }

  if (tsSegmentsRefresh) {
    tsSegmentsRefresh.addEventListener("click", () => {
      loadSegments();
      refreshSegmentFacets();
    });
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
