// services/orion-hub/static/js/app.js

// ───────────────────────────────────────────────────────────────
// Global State
// ───────────────────────────────────────────────────────────────
function resolveHubApiBaseUrl() {
  const cfg = window.__HUB_CFG__ || {};
  const override = String(cfg.apiBaseOverride || '').trim();
  if (override) {
    return override.replace(/\/+$/, '');
  }
  // Hub REST routes are mounted at /api on the app root (not /{first-path-segment}/api).
  return String(window.location.origin || '').replace(/\/+$/, '');
}

function resolveHubWebSocketUrl() {
  const cfg = window.__HUB_CFG__ || {};
  const override = String(cfg.wsBaseOverride || '').trim();
  if (override) {
    const normalized = override.replace(/\/+$/, '');
    return normalized.endsWith('/ws') ? normalized : `${normalized}/ws`;
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws`;
}

const URL_PREFIX = '';
const API_BASE_URL = resolveHubApiBaseUrl();
const HUB_WEBSOCKET_URL = resolveHubWebSocketUrl();
const VISION_EDGE_BASE = "https://athena.tail348bbe.ts.net/vision-edge";

let socket;
let wsReadyResolve = null;
let wsReadyPromise = null;
let recordingStream = null;
/** @type {{ source: MediaStreamAudioSourceNode, processor: ScriptProcessorNode, gain: GainNode, chunks: Float32Array[], capturing: boolean, flushing: boolean } | null} */
let pcmCapture = null;
/** @type {{ generation: number, stopRequested: boolean, starting: boolean }} */
let voiceCapture = { generation: 0, stopRequested: false, starting: false };
// Client float peak warn threshold (metadata only); server gate is STT_NEAR_SILENT_PEAK_INT16 in whisper stt.py.
const VOICE_CLIENT_PEAK_MIN = 0.00025;
const VOICE_MIN_CAPTURE_DURATION_SEC = 0.15;
const VOICE_PCM_FLUSH_MS = 150;
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
let modeVerbOverride = null;
/** Hub quick lane: 'fast' = light prep; 'stance' = full brain/stance prep (slower, richer). */
let chatQuickVariant = 'fast';
let selectedLlmRoute = localStorage.getItem('orion_llm_route') || 'chat';
let llmRouteCatalog = { default_route: 'chat', routes: [] };
let orionSessionId = localStorage.getItem('orion_sid') || null;
let browserClientId = localStorage.getItem('orion_browser_client_id') || null;
let presenceContext = null;
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

function notificationBatchHydrateThreshold() {
  const el = document.getElementById("hubNotificationsPanel");
  const raw = el?.dataset?.notificationBatchThreshold;
  const n = raw !== undefined && raw !== "" ? parseInt(String(raw), 10) : NaN;
  return Number.isFinite(n) && n >= 1 ? n : 5;
}
let notificationsInitialHydrateDone = false;
const ATTENTION_EVENT_KIND = "orion.chat.attention";
const CHAT_MESSAGE_EVENT_KIND = "orion.chat.message";
const RECIPIENT_GROUP = "juniper_primary";
let latestSocialInspectionState = null;
const socialInspectionCache = new Map();
let workflowSchedules = [];
let selectedSchedule = null;
const submittedFeedbackTargets = new Set();

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
  const chatInputExpandButton = document.getElementById('chatInputExpandButton');
  const chatInputExpandModalRoot = document.getElementById('chatInputExpandModalRoot');
  const chatInputExpandModalBackdrop = document.getElementById('chatInputExpandModalBackdrop');
  const chatInputExpandModalDialog = document.getElementById('chatInputExpandModalDialog');
  const chatInputExpandModalClose = document.getElementById('chatInputExpandModalClose');
  const chatInputExpandTextarea = document.getElementById('chatInputExpandTextarea');
  const chatInputExpandModalApply = document.getElementById('chatInputExpandModalApply');
  const chatInputExpandModalSend = document.getElementById('chatInputExpandModalSend');
  const sendButton = document.getElementById('sendButton');
  const skillRunnerSelect = document.getElementById('skillRunnerSelect');
  const skillRunnerRunBtn = document.getElementById('skillRunnerRunBtn');
  const skillRunnerInsertBtn = document.getElementById('skillRunnerInsertBtn');
  const textToSpeechToggle = document.getElementById('textToSpeechToggle');
  const recallToggle = document.getElementById('recallToggle');
  const recallRequiredToggle = document.getElementById('recallRequiredToggle');
  const noWriteToggle = document.getElementById('noWriteToggle');
  const presenceOpenButton = document.getElementById('presenceOpenButton');
  const presenceStatusChip = document.getElementById('presenceStatusChip');
  const presenceModalRoot = document.getElementById('presenceModalRoot');
  const presenceModalBackdrop = document.getElementById('presenceModalBackdrop');
  const presenceModalClose = document.getElementById('presenceModalClose');
  const presenceRequestorName = document.getElementById('presenceRequestorName');
  const presenceAudienceMode = document.getElementById('presenceAudienceMode');
  const presencePresetSolo = document.getElementById('presencePresetSolo');
  const presencePresetKids = document.getElementById('presencePresetKids');
  const presencePresetSpouse = document.getElementById('presencePresetSpouse');
  const presencePresetFamily = document.getElementById('presencePresetFamily');
  const presencePresetGuest = document.getElementById('presencePresetGuest');
  const presenceAddCompanionButton = document.getElementById('presenceAddCompanionButton');
  const presenceCompanionRows = document.getElementById('presenceCompanionRows');
  const presenceSessionOnlyToggle = document.getElementById('presenceSessionOnlyToggle');
  const presenceSaveButton = document.getElementById('presenceSaveButton');
  const presenceClearButton = document.getElementById('presenceClearButton');
  const recallModeSelect = document.getElementById('recallModeSelect');
  const recallProfileSelect = document.getElementById('recallProfileSelect');
  const runtimeDebugPanelToggle = document.getElementById('runtimeDebugPanelToggle');
  const runtimeDebugPanelBody = document.getElementById('runtimeDebugPanelBody');
  const runtimeDebugPanelCaret = document.getElementById('runtimeDebugPanelCaret');
  const memoryPanelToggle = document.getElementById('memoryPanelToggle');
  const memoryPanelCaret = document.getElementById('memoryPanelCaret');
  const memoryPanelBody = document.getElementById('memoryPanelBody');
  const memoryUsedValue = document.getElementById('memoryUsedValue');
  const recallCountValue = document.getElementById('recallCountValue');
  const backendCountsValue = document.getElementById('backendCountsValue');
  const memoryDigestPre = document.getElementById('memoryDigestPre');
  const memoryDebugOpenModal = document.getElementById('memoryDebugOpenModal');
  const memoryDebugModalRoot = document.getElementById('memoryDebugModalRoot');
  const memoryDebugModalBackdrop = document.getElementById('memoryDebugModalBackdrop');
  const memoryDebugModalDialog = document.getElementById('memoryDebugModalDialog');
  const memoryDebugModalClose = document.getElementById('memoryDebugModalClose');
  const memoryDebugModalMeta = document.getElementById('memoryDebugModalMeta');
  const memoryDebugModalBody = document.getElementById('memoryDebugModalBody');
  const mindRunsModal = document.getElementById('mindRunsModal');
  const mindRunsModalClose = document.getElementById('mindRunsModalClose');
  const mindRunsModalMeta = document.getElementById('mindRunsModalMeta');
  const mindRunsModalStatus = document.getElementById('mindRunsModalStatus');
  const mindRunsModalList = document.getElementById('mindRunsModalList');
  const mindRunsModalDetails = document.getElementById('mindRunsModalDetails');
  const agentTraceDebugPanel = document.getElementById('agentTraceDebugPanel');
  const agentTraceDebugToggle = document.getElementById('agentTraceDebugToggle');
  const agentTraceDebugOpenModal = document.getElementById('agentTraceDebugOpenModal');
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
  const autonomyGoalArchiveDryRun = document.getElementById('autonomyGoalArchiveDryRun');
  const autonomyGoalArchiveApply = document.getElementById('autonomyGoalArchiveApply');
  const autonomyGoalArchiveStatus = document.getElementById('autonomyGoalArchiveStatus');
  const chatStanceDebugPanel = document.getElementById('chatStanceDebugPanel');
  const chatStanceDebugToggle = document.getElementById('chatStanceDebugToggle');
  const chatStanceDebugCaret = document.getElementById('chatStanceDebugCaret');
  const chatStanceDebugBody = document.getElementById('chatStanceDebugBody');
  const chatStanceDebugMeta = document.getElementById('chatStanceDebugMeta');
  const chatStanceDebugOverview = document.getElementById('chatStanceDebugOverview');
  const chatStanceDebugLineage = document.getElementById('chatStanceDebugLineage');
  const chatStanceDebugRaw = document.getElementById('chatStanceDebugRaw');
  const chatStanceDebugOpenModal = document.getElementById('chatStanceDebugOpenModal');
  const chatStanceDebugModalRoot = document.getElementById('chatStanceDebugModalRoot');
  const chatStanceDebugModalBackdrop = document.getElementById('chatStanceDebugModalBackdrop');
  const chatStanceDebugModalDialog = document.getElementById('chatStanceDebugModalDialog');
  const chatStanceDebugModalClose = document.getElementById('chatStanceDebugModalClose');
  const chatStanceDebugModalMeta = document.getElementById('chatStanceDebugModalMeta');
  const chatStanceDebugModalBody = document.getElementById('chatStanceDebugModalBody');
  const substrateReviewDebugPanel = document.getElementById('substrateReviewDebugPanel');
  const substrateReviewDebugToggle = document.getElementById('substrateReviewDebugToggle');
  const substrateReviewDebugCaret = document.getElementById('substrateReviewDebugCaret');
  const substrateReviewDebugBody = document.getElementById('substrateReviewDebugBody');
  const substrateReviewDebugMeta = document.getElementById('substrateReviewDebugMeta');
  const substrateReviewDebugOverview = document.getElementById('substrateReviewDebugOverview');

  function ensureSkillRunnerWorkflowOptions() {
    if (!skillRunnerSelect) return;
    const workflows = [
      { workflowId: 'dream_cycle', prompt: 'Run your dream cycle.', label: 'Workflow · Run your dream cycle.' },
      { workflowId: 'journal_pass', prompt: 'Do a journal pass.', label: 'Workflow · Do a journal pass.' },
      { workflowId: 'self_review', prompt: 'Run a self review.', label: 'Workflow · Run a self review.' },
      { workflowId: 'concept_induction_pass', prompt: 'Run concept induction.', label: 'Workflow · Run concept induction.' },
    ];
    const existingByWorkflowId = new Set(
      Array.from(skillRunnerSelect.options || [])
        .map((opt) => String(opt.dataset.workflowId || '').trim())
        .filter(Boolean)
    );
    if (workflows.every((wf) => existingByWorkflowId.has(wf.workflowId))) {
      return;
    }
    let workflowGroup = Array.from(skillRunnerSelect.querySelectorAll('optgroup')).find(
      (group) => String(group.label || '').trim().toLowerCase() === 'cognitive workflows'
    );
    if (!workflowGroup) {
      workflowGroup = document.createElement('optgroup');
      workflowGroup.label = 'Cognitive workflows';
      skillRunnerSelect.appendChild(workflowGroup);
    }
    workflows.forEach((wf) => {
      if (existingByWorkflowId.has(wf.workflowId)) return;
      const option = document.createElement('option');
      option.value = wf.prompt;
      option.dataset.workflowId = wf.workflowId;
      option.textContent = wf.label;
      workflowGroup.appendChild(option);
    });
  }
  ensureSkillRunnerWorkflowOptions();
  const selfExperimentsDebugPanel = document.getElementById('selfExperimentsDebugPanel');
  const selfExperimentsDebugToggle = document.getElementById('selfExperimentsDebugToggle');
  const selfExperimentsDebugCaret = document.getElementById('selfExperimentsDebugCaret');
  const selfExperimentsDebugBody = document.getElementById('selfExperimentsDebugBody');
  const selfExperimentsDebugMeta = document.getElementById('selfExperimentsDebugMeta');
  const selfExperimentsDebugOverview = document.getElementById('selfExperimentsDebugOverview');
  const selfExperimentsDebugRaw = document.getElementById('selfExperimentsDebugRaw');
  const selfExperimentsDebugOpenModal = document.getElementById('selfExperimentsDebugOpenModal');
  const selfExperimentsDebugRefresh = document.getElementById('selfExperimentsDebugRefresh');
  const selfExperimentsFilterCorrelation = document.getElementById('selfExperimentsFilterCorrelation');
  const selfExperimentsFilterDate = document.getElementById('selfExperimentsFilterDate');
  const selfExperimentsFilterSkill = document.getElementById('selfExperimentsFilterSkill');
  const selfExperimentsApplyFilters = document.getElementById('selfExperimentsApplyFilters');
  const selfExperimentsTriggerPulse = document.getElementById('selfExperimentsTriggerPulse');
  const selfExperimentsTriggerMetacog = document.getElementById('selfExperimentsTriggerMetacog');
  const selfExperimentsActionStatus = document.getElementById('selfExperimentsActionStatus');
  const selfExperimentsModalRoot = document.getElementById('selfExperimentsModalRoot');
  const selfExperimentsModalBackdrop = document.getElementById('selfExperimentsModalBackdrop');
  const selfExperimentsModalDialog = document.getElementById('selfExperimentsModalDialog');
  const selfExperimentsModalMeta = document.getElementById('selfExperimentsModalMeta');
  const selfExperimentsModalRefresh = document.getElementById('selfExperimentsModalRefresh');
  const selfExperimentsModalClose = document.getElementById('selfExperimentsModalClose');
  const selfExperimentsModalSummary = document.getElementById('selfExperimentsModalSummary');
  const selfExperimentsModalRuns = document.getElementById('selfExperimentsModalRuns');
  const selfExperimentsModalProvenanceTable = document.getElementById('selfExperimentsModalProvenanceTable');
  const selfExperimentsModalRaw = document.getElementById('selfExperimentsModalRaw');
  const autonomyReadinessPanel = document.getElementById('autonomyReadinessPanel');
  const autonomyReadinessToggle = document.getElementById('autonomyReadinessToggle');
  const autonomyReadinessCaret = document.getElementById('autonomyReadinessCaret');
  const autonomyReadinessBody = document.getElementById('autonomyReadinessBody');
  const autonomyReadinessMeta = document.getElementById('autonomyReadinessMeta');
  const autonomyReadinessOverview = document.getElementById('autonomyReadinessOverview');
  const autonomyReadinessWarnings = document.getElementById('autonomyReadinessWarnings');
  const recallCanaryPanel = document.getElementById('recallCanaryPanel');
  const recallCanaryOpenModal = document.getElementById('recallCanaryOpenModal');
  const recallCanaryToggle = document.getElementById('recallCanaryToggle');
  const recallCanaryCaret = document.getElementById('recallCanaryCaret');
  const recallCanaryBody = document.getElementById('recallCanaryBody');
  const recallCanaryModalRoot = document.getElementById('recallCanaryModalRoot');
  const recallCanaryModalBackdrop = document.getElementById('recallCanaryModalBackdrop');
  const recallCanaryModalDialog = document.getElementById('recallCanaryModalDialog');
  const recallCanaryModalMeta = document.getElementById('recallCanaryModalMeta');
  const recallCanaryModalRefresh = document.getElementById('recallCanaryModalRefresh');
  const recallCanaryModalClose = document.getElementById('recallCanaryModalClose');
  const recallCanaryModalStatusMeta = document.getElementById('recallCanaryModalStatusMeta');
  const recallCanaryModalSummary = document.getElementById('recallCanaryModalSummary');
  const recallCanaryModalActionStatus = document.getElementById('recallCanaryModalActionStatus');
  const recallCanaryProfileSelect = document.getElementById('recallCanaryProfileSelect');
  const recallCanaryProfileEmptyState = document.getElementById('recallCanaryProfileEmptyState');
  const recallCanarySafetyBadges = document.getElementById('recallCanarySafetyBadges');
  const recallCanaryQueryInput = document.getElementById('recallCanaryQueryInput');
  const recallCanaryRunButton = document.getElementById('recallCanaryRunButton');
  const recallCanaryStatusMeta = document.getElementById('recallCanaryStatusMeta');
  const recallCanarySummary = document.getElementById('recallCanarySummary');
  const recallCanaryLatestResult = document.getElementById('recallCanaryLatestResult');
  const recallCanaryRawResponse = document.getElementById('recallCanaryRawResponse');
  const recallCanaryJudgmentSelect = document.getElementById('recallCanaryJudgmentSelect');
  const recallCanaryOperatorNote = document.getElementById('recallCanaryOperatorNote');
  const recallCanaryRecordJudgmentButton = document.getElementById('recallCanaryRecordJudgmentButton');
  const recallCanaryCreateReviewArtifactButton = document.getElementById('recallCanaryCreateReviewArtifactButton');
  const cognitiveReviewPanel = document.getElementById('cognitiveReviewPanel');
  const cognitiveReviewOpenModal = document.getElementById('cognitiveReviewOpenModal');
  const cognitiveReviewStatusMeta = document.getElementById('cognitiveReviewStatusMeta');
  const cognitiveReviewStatusSummary = document.getElementById('cognitiveReviewStatusSummary');
  const cognitiveProposalIdInput = document.getElementById('cognitiveProposalIdInput');
  const cognitiveReviewRationaleInput = document.getElementById('cognitiveReviewRationaleInput');
  const cognitiveReviewAcceptDraftButton = document.getElementById('cognitiveReviewAcceptDraftButton');
  const cognitiveReviewRejectButton = document.getElementById('cognitiveReviewRejectButton');
  const cognitiveReviewArchiveButton = document.getElementById('cognitiveReviewArchiveButton');
  const cognitiveReviewSupersedeButton = document.getElementById('cognitiveReviewSupersedeButton');
  const cognitiveDraftList = document.getElementById('cognitiveDraftList');
  const cognitiveStanceNoteList = document.getElementById('cognitiveStanceNoteList');
  const cognitiveReviewModalRoot = document.getElementById('cognitiveReviewModalRoot');
  const cognitiveReviewModalBackdrop = document.getElementById('cognitiveReviewModalBackdrop');
  const cognitiveReviewModalDialog = document.getElementById('cognitiveReviewModalDialog');
  const cognitiveReviewModalMeta = document.getElementById('cognitiveReviewModalMeta');
  const cognitiveReviewModalClose = document.getElementById('cognitiveReviewModalClose');
  const cognitiveReviewModalRefresh = document.getElementById('cognitiveReviewModalRefresh');
  const cognitiveReviewModalStatusMeta = document.getElementById('cognitiveReviewModalStatusMeta');
  const cognitiveReviewModalStatusSummary = document.getElementById('cognitiveReviewModalStatusSummary');
  const cognitiveReviewModalProposalIdInput = document.getElementById('cognitiveReviewModalProposalIdInput');
  const cognitiveReviewModalRationaleInput = document.getElementById('cognitiveReviewModalRationaleInput');
  const cognitiveReviewModalAcceptDraftButton = document.getElementById('cognitiveReviewModalAcceptDraftButton');
  const cognitiveReviewModalRejectButton = document.getElementById('cognitiveReviewModalRejectButton');
  const cognitiveReviewModalArchiveButton = document.getElementById('cognitiveReviewModalArchiveButton');
  const cognitiveReviewModalSupersedeButton = document.getElementById('cognitiveReviewModalSupersedeButton');
  const cognitiveReviewModalDraftList = document.getElementById('cognitiveReviewModalDraftList');
  const cognitiveReviewModalStanceNoteList = document.getElementById('cognitiveReviewModalStanceNoteList');
  const autonomyConstitutionOpenModal = document.getElementById('autonomyConstitutionOpenModal');
  const autonomyConstitutionModalRoot = document.getElementById('autonomyConstitutionModalRoot');
  const autonomyConstitutionModalBackdrop = document.getElementById('autonomyConstitutionModalBackdrop');
  const autonomyConstitutionModalDialog = document.getElementById('autonomyConstitutionModalDialog');
  const autonomyConstitutionModalMeta = document.getElementById('autonomyConstitutionModalMeta');
  const autonomyConstitutionModalRefresh = document.getElementById('autonomyConstitutionModalRefresh');
  const autonomyConstitutionModalClose = document.getElementById('autonomyConstitutionModalClose');
  const autonomyConstitutionModalSummary = document.getElementById('autonomyConstitutionModalSummary');
  const autonomyConstitutionModalInvariants = document.getElementById('autonomyConstitutionModalInvariants');
  const autonomyConstitutionModalSurfaces = document.getElementById('autonomyConstitutionModalSurfaces');
  const substrateReviewDebugOpenModal = document.getElementById('substrateReviewDebugOpenModal');
  const substrateReviewModalRoot = document.getElementById('substrateReviewModalRoot');
  const substrateReviewModalBackdrop = document.getElementById('substrateReviewModalBackdrop');
  const substrateReviewModalDialog = document.getElementById('substrateReviewModalDialog');
  const substrateReviewModalMeta = document.getElementById('substrateReviewModalMeta');
  const substrateReviewModalClose = document.getElementById('substrateReviewModalClose');
  const substrateReviewModalQuickStatus = document.getElementById('substrateReviewModalQuickStatus');
  const substrateReviewActionRefresh = document.getElementById('substrateReviewActionRefresh');
  const substrateReviewActionExecuteOnce = document.getElementById('substrateReviewActionExecuteOnce');
  const substrateReviewActionExecuteFollowup = document.getElementById('substrateReviewActionExecuteFollowup');
  const substrateReviewActionSmokeCheck = document.getElementById('substrateReviewActionSmokeCheck');
  const substrateReviewActionStatus = document.getElementById('substrateReviewActionStatus');
  const substrateReviewResultSummary = document.getElementById('substrateReviewResultSummary');
  const substrateReviewResultRaw = document.getElementById('substrateReviewResultRaw');
  const notificationList = document.getElementById('notificationList');
  const notificationFilter = document.getElementById('notificationFilter');
  const attentionList = document.getElementById('attentionList');
  const attentionCount = document.getElementById('attentionCount');
  const messagesToggle = document.getElementById('messagesToggle');
  const messagesCaret = document.getElementById('messagesCaret');
  const messagesBody = document.getElementById('messagesBody');
  const messageList = document.getElementById('messageList');
  const messageFilter = document.getElementById('messageFilter');
  const worldPulseToggle = document.getElementById('worldPulseToggle');
  const worldPulseCaret = document.getElementById('worldPulseCaret');
  const worldPulseBody = document.getElementById('worldPulseBody');
  const worldPulseStatus = document.getElementById('worldPulseStatus');
  const worldPulseSummary = document.getElementById('worldPulseSummary');
  const worldPulseDetails = document.getElementById('worldPulseDetails');
  const worldPulseRunButton = document.getElementById('worldPulseRunButton');
  const worldPulseFixtureRunEnabled = Boolean(window.__HUB_CFG__?.worldPulseFixtureRunEnabled);
  const WORLD_PULSE_REQUIRED_SECTIONS = ['us_politics', 'global_politics', 'local_politics'];
  const WORLD_PULSE_RECOMMENDED_SECTIONS = [
    'ai_technology',
    'science_climate_energy',
    'healthcare_mental_health',
    'security_infrastructure_software',
    'hardware_compute_gpu',
    'local_conditions',
  ];
  const toastContainer = document.getElementById('toastContainer');
  const agentTraceApi = window.OrionAgentTrace || {};
  const socialInspectionApi = window.OrionSocialInspection || {};
  const workflowUiApi = window.OrionWorkflowUI || {};
  const scheduleUiApi = window.OrionWorkflowScheduleUI || {};
  const thoughtProcessApi = window.OrionThoughtProcess || {};
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
  const responseFeedbackModal = document.getElementById('responseFeedbackModal');
  const responseFeedbackModalClose = document.getElementById('responseFeedbackModalClose');
  const responseFeedbackCancel = document.getElementById('responseFeedbackCancel');
  const responseFeedbackSubmit = document.getElementById('responseFeedbackSubmit');
  const responseFeedbackTitle = document.getElementById('responseFeedbackTitle');
  const responseFeedbackMeta = document.getElementById('responseFeedbackMeta');
  const responseFeedbackCategoryList = document.getElementById('responseFeedbackCategoryList');
  const responseFeedbackNotes = document.getElementById('responseFeedbackNotes');
  const responseFeedbackStatus = document.getElementById('responseFeedbackStatus');
  const memoryGraphBridgeModal = document.getElementById('memoryGraphBridgeModal');
  let lastMemoryDebugModel = null;
  let lastAgentTraceSummary = null;
  let lastAgentTraceMeta = {};
  let lastChatStanceDebug = null;
  let lastSubstrateReviewStatus = null;
  let lastSubstrateReviewAction = null;
  let lastAutonomyReadinessSnapshot = null;
  let lastRecallCanaryRunId = null;
  let lastRecallCanaryResponse = null;
  let lastRecallCanarySelectedProfile = null;
  let memoryGraphBridgeTurnsCache = [];
  let memoryGraphBridgeAnchorDiv = null;
  const MEMORY_GRAPH_BRIDGE_MAX_TURNS_CAP = 80;
  const MEMORY_GRAPH_BRIDGE_MAX_TURNS_DEFAULT = 40;
  function memoryGraphSuggestFetchTimeoutMs() {
    const ui = window.OrionMemoryGraphDraftUI || {};
    if (typeof ui.resolveMemoryGraphSuggestFetchTimeoutMs === 'function') {
      return ui.resolveMemoryGraphSuggestFetchTimeoutMs();
    }
    const raw = Number((window.__HUB_CFG__ || {}).memoryGraphSuggestFetchTimeoutMs);
    return Number.isFinite(raw) && raw > 0 ? raw : 205000;
  }
  const MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS = 12000;
  const MEMORY_GRAPH_SUGGEST_INPUT_PER_TURN_CHARS = 1800;
  const LS_MEMORY_GRAPH_BRIDGE_MAX_TURNS = 'orion_memory_graph_bridge_max_turns';
  let memoryGraphBridgeDraftViz = null;
  let memoryGraphBridgeDraftForm = null;
  let organSignalsGraphCtl = null;
  const RECALL_CANARY_PROFILE_STORAGE_KEY = 'orion_recall_canary_profile_v1';

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

  const toastMessage = document.getElementById("toastMessage");

  // Topic Studio
  const hubTabButton = document.getElementById("hubTabButton");
  const topicStudioTabButton = document.getElementById("topicStudioTabButton");
  const serviceLogsTabButton = document.getElementById("serviceLogsTabButton");
  const substrateLegacyTabButton = document.getElementById("substratePageLink");
  const substrateTabButton =
    document.getElementById("substrateTabButton") || substrateLegacyTabButton;
  const memoryTabButton = document.getElementById("memoryTabButton");
  const memoryPanel = document.getElementById("memory");
  const mindTabButton = document.getElementById("mindTabButton");
  const mindPanel = document.getElementById("mind");
  const forgeTabButton = document.getElementById("forgeTabButton");
  const forgePanel = document.getElementById("forge");
  const forgeRefreshButton = document.getElementById("forgeRefreshButton");
  const forgeStatus = document.getElementById("forgeStatus");
  const forgeStatusSources = document.getElementById("forgeStatusSources");
  const forgeStatusClaims = document.getElementById("forgeStatusClaims");
  const forgeStatusAccepted = document.getElementById("forgeStatusAccepted");
  const forgeStatusDisputed = document.getElementById("forgeStatusDisputed");
  const forgeStatusStale = document.getElementById("forgeStatusStale");
  const forgeStatusSpecs = document.getElementById("forgeStatusSpecs");
  const forgeStatusExecutionReady = document.getElementById("forgeStatusExecutionReady");
  const forgeStatusPendingReviews = document.getElementById("forgeStatusPendingReviews");
  const forgeStatusContextPacks = document.getElementById("forgeStatusContextPacks");
  const forgeHealthBadge = document.getElementById("forgeHealthBadge");
  const forgeWarningsList = document.getElementById("forgeWarningsList");
  const forgeSuggestedAction = document.getElementById("forgeSuggestedAction");
  const forgeSearchInput = document.getElementById("forgeSearchInput");
  const forgeSearchButton = document.getElementById("forgeSearchButton");
  const forgeSearchResults = document.getElementById("forgeSearchResults");
  const forgeClaimsList = document.getElementById("forgeClaimsList");
  const forgeSpecsStatusFilter = document.getElementById("forgeSpecsStatusFilter");
  const forgeSpecsList = document.getElementById("forgeSpecsList");
  const forgeReviewsList = document.getElementById("forgeReviewsList");
  const forgeCompileTask = document.getElementById("forgeCompileTask");
  const forgeCompileTarget = document.getElementById("forgeCompileTarget");
  const forgeCompileSpecsList = document.getElementById("forgeCompileSpecsList");
  const forgeIncludeDisputed = document.getElementById("forgeIncludeDisputed");
  const forgeIncludeStale = document.getElementById("forgeIncludeStale");
  const forgeWriteFile = document.getElementById("forgeWriteFile");
  const forgeCompileButton = document.getElementById("forgeCompileButton");
  const forgeCompileResult = document.getElementById("forgeCompileResult");
  const forgeCompilePath = document.getElementById("forgeCompilePath");
  const forgeCompileIncludedSpecs = document.getElementById("forgeCompileIncludedSpecs");
  const forgeCompileIncludedClaims = document.getElementById("forgeCompileIncludedClaims");
  const forgeCompileExcludedClaims = document.getElementById("forgeCompileExcludedClaims");
  const forgeCompileWarnings = document.getElementById("forgeCompileWarnings");
  const forgeCompileContentPreview = document.getElementById("forgeCompileContentPreview");
  const forgeDebugStatus = document.getElementById("forgeDebugStatus");
  const forgeDebugSearch = document.getElementById("forgeDebugSearch");
  const forgeDebugCompile = document.getElementById("forgeDebugCompile");
  const forgeSourcePath = document.getElementById("forgeSourcePath");
  const forgeSourceId = document.getElementById("forgeSourceId");
  const forgeSourceKind = document.getElementById("forgeSourceKind");
  const forgeSourceDryRun = document.getElementById("forgeSourceDryRun");
  const forgeSourceWriteReview = document.getElementById("forgeSourceWriteReview");
  const forgeSourceIngestButton = document.getElementById("forgeSourceIngestButton");
  const forgeSourceIngestResult = document.getElementById("forgeSourceIngestResult");
  const forgeSourceIngestStatus = document.getElementById("forgeSourceIngestStatus");
  const forgeSourceIngestSourceId = document.getElementById("forgeSourceIngestSourceId");
  const forgeSourceIngestSourcePath = document.getElementById("forgeSourceIngestSourcePath");
  const forgeSourceIngestReviewPath = document.getElementById("forgeSourceIngestReviewPath");
  const forgeSourceIngestClaimsCount = document.getElementById("forgeSourceIngestClaimsCount");
  const forgeSourceIngestClaimsList = document.getElementById("forgeSourceIngestClaimsList");
  const forgeSourceIngestAffectedSpecs = document.getElementById("forgeSourceIngestAffectedSpecs");
  const forgeSourceIngestWarnings = document.getElementById("forgeSourceIngestWarnings");
  const forgeSourceIngestContentPreview = document.getElementById("forgeSourceIngestContentPreview");
  const forgeSourceIngestError = document.getElementById("forgeSourceIngestError");
  const forgeDebugSourceIngest = document.getElementById("forgeDebugSourceIngest");
  const signalsTabButton = document.getElementById("signalsTabButton");
  const signalsPanel = document.getElementById("signals");
  const mindHoursInput = document.getElementById("mindHoursInput");
  const mindRefreshButton = document.getElementById("mindRefreshButton");
  const mindFilterOk = document.getElementById("mindFilterOk");
  const mindFilterTrigger = document.getElementById("mindFilterTrigger");
  const mindFilterErrorCode = document.getElementById("mindFilterErrorCode");
  const mindFilterRouterProfileId = document.getElementById("mindFilterRouterProfileId");
  const mindDefaultOnSendToggle = document.getElementById("mindDefaultOnSendToggle");
  const mindStatus = document.getElementById("mindStatus");
  const mindSummaryTotal = document.getElementById("mindSummaryTotal");
  const mindSummaryOk = document.getElementById("mindSummaryOk");
  const mindSummaryFailed = document.getElementById("mindSummaryFailed");
  const mindRunsTableBody = document.getElementById("mindRunsTableBody");
  const hubTabPanel = document.getElementById("hub") || document.getElementById("hubTabPanel");
  const topicStudioPanel =
    document.getElementById("topic-studio") || document.getElementById("topicStudioPanel");
  const serviceLogsPanel = document.getElementById("service-logs");
  const substratePanel = document.getElementById("substrate");
  const substratePanelFrame = document.getElementById("substratePanelFrame");
  const substratePanelRefresh = document.getElementById("substratePanelRefresh");
  const substrateAtlasTabButton = document.getElementById("substrateAtlasTabButton");
  const substrateAtlasPanel = document.getElementById("substrate-atlas");
  const substrateAtlasPanelFrame = document.getElementById("substrateAtlasPanelFrame");
  const substrateAtlasPanelRefresh = document.getElementById("substrateAtlasPanelRefresh");
  const pressureAnalyticsTabButton = document.getElementById("pressureAnalyticsTabButton");
  const pressurePanel = document.getElementById("pressure");
  const collapseMirrorTabButton = document.getElementById("collapseMirrorTabButton");
  const collapseMirrorPanel = document.getElementById("collapse-mirror");
  const pressureAnalyticsFrame = document.getElementById("pressureAnalyticsFrame");
  const pressureAnalyticsRefresh = document.getElementById("pressureAnalyticsRefresh");
  const substrateLatticeTabButton = document.getElementById("substrateLatticeTabButton");
  const substrateLatticePanelEl = document.getElementById("substrate-lattice");
  const substrateLatticeFrame = document.getElementById("substrateLatticeFrame");
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
  const KNOWLEDGE_PROXY_BASE = `${API_BASE_URL}/api/knowledge`;
  const TOPIC_STUDIO_STATE_KEY = "topic_studio_state_v1";
  const MIN_PREVIEW_DOCS = 20;

  function styleTabButton(button, isActive) {
    if (!button) return;
    button.classList.toggle("bg-indigo-600", isActive);
    button.classList.toggle("text-white", isActive);
    button.classList.toggle("border-indigo-500", isActive);
    button.classList.toggle("bg-gray-800", !isActive);
    button.classList.toggle("text-gray-200", !isActive);
    button.classList.toggle("border-gray-700", !isActive);
  }

  function setActiveTab(tabKey) {
    if (
      !hubTabPanel ||
      !topicStudioPanel ||
      !serviceLogsPanel ||
      !substratePanel ||
      !hubTabButton ||
      !topicStudioTabButton ||
      !serviceLogsTabButton ||
      !substrateTabButton
    ) {
      return;
    }
    let effectiveTab = tabKey;
    if (tabKey === "memory" && !memoryPanel) {
      effectiveTab = "hub";
    }
    if (tabKey === "pressure" && !pressurePanel) {
      effectiveTab = "hub";
    }
    if (tabKey === "mind" && !mindPanel) {
      effectiveTab = "hub";
    }
    if (tabKey === "signals" && !signalsPanel) {
      effectiveTab = "hub";
    }
    if (tabKey === "forge" && !forgePanel) {
      effectiveTab = "hub";
    }
    if (tabKey === "substrate-atlas" && !substrateAtlasPanel) {
      effectiveTab = "hub";
    }
    if (tabKey === "substrate-lattice" && !substrateLatticePanelEl) {
      effectiveTab = "hub";
    }
    if (tabKey === "collapse-mirror" && !collapseMirrorPanel) {
      effectiveTab = "hub";
    }
    const isHub = effectiveTab === "hub";
    const isTopicStudio = effectiveTab === "topic-studio";
    const isServiceLogs = effectiveTab === "service-logs";
    const isSubstrate = effectiveTab === "substrate";
    const isSubstrateAtlas = effectiveTab === "substrate-atlas";
    const isMemory = effectiveTab === "memory";
    const isPressure = effectiveTab === "pressure";
    const isSubstrateLattice = effectiveTab === "substrate-lattice";
    const isMind = effectiveTab === "mind";
    const isSignals = effectiveTab === "signals";
    const isForge = effectiveTab === "forge";
    const isCollapseMirror = effectiveTab === "collapse-mirror";
    hubTabPanel.classList.toggle("hidden", !isHub);
    topicStudioPanel.classList.toggle("hidden", !isTopicStudio);
    serviceLogsPanel.classList.toggle("hidden", !isServiceLogs);
    substratePanel.classList.toggle("hidden", !isSubstrate);
    if (substrateAtlasPanel) {
      substrateAtlasPanel.classList.toggle("hidden", !isSubstrateAtlas);
      if (isSubstrateAtlas && substrateAtlasPanelFrame) {
        const pingAtlasFrame = () => {
          try {
            const atlasWin = substrateAtlasPanelFrame.contentWindow;
            if (atlasWin && atlasWin.OrionSubstrateAtlas && typeof atlasWin.OrionSubstrateAtlas.activate === "function") {
              atlasWin.OrionSubstrateAtlas.activate();
            }
          } catch {
            /* iframe not ready */
          }
        };
        setTimeout(pingAtlasFrame, 150);
      }
    }
    if (memoryPanel) {
      memoryPanel.classList.toggle("hidden", !isMemory);
      if (isMemory) {
        window.dispatchEvent(new Event("orion-hub-memory-tab-activated"));
      }
    }
    if (mindPanel) {
      mindPanel.classList.toggle("hidden", !isMind);
      if (isMind) {
        refreshMindRuns();
      }
    }
    if (signalsPanel) {
      signalsPanel.classList.toggle("hidden", !isSignals);
      if (isSignals && window.OrionOrganSignalsGraphUI) {
        const ctl = ensureOrganSignalsGraph();
        if (ctl && typeof ctl.refresh === "function") {
          ctl.refresh();
        }
      }
    }
    if (pressurePanel) {
      pressurePanel.classList.toggle("hidden", !isPressure);
    }
    if (substrateLatticePanelEl) {
      substrateLatticePanelEl.classList.toggle("hidden", !isSubstrateLattice);
    }
    if (substrateLatticeFrame && isSubstrateLattice) {
      const currentSrc = substrateLatticeFrame.getAttribute("src");
      substrateLatticeFrame.setAttribute("src", currentSrc);
    }
    if (forgePanel) {
      forgePanel.classList.toggle("hidden", !isForge);
      if (isForge) {
        refreshForgeTab();
      }
    }
    if (collapseMirrorPanel) {
      collapseMirrorPanel.classList.toggle("hidden", !isCollapseMirror);
    }
    styleTabButton(hubTabButton, isHub);
    styleTabButton(topicStudioTabButton, isTopicStudio);
    styleTabButton(serviceLogsTabButton, isServiceLogs);
    styleTabButton(substrateTabButton, isSubstrate);
    if (substrateAtlasTabButton) {
      styleTabButton(substrateAtlasTabButton, isSubstrateAtlas);
    }
    if (memoryTabButton) {
      styleTabButton(memoryTabButton, isMemory);
    }
    if (mindTabButton) {
      styleTabButton(mindTabButton, isMind);
    }
    if (signalsTabButton) {
      styleTabButton(signalsTabButton, isSignals);
    }
    if (pressureAnalyticsTabButton) {
      styleTabButton(pressureAnalyticsTabButton, isPressure);
    }
    if (substrateLatticeTabButton) {
      styleTabButton(substrateLatticeTabButton, isSubstrateLattice);
    }
    if (forgeTabButton) {
      styleTabButton(forgeTabButton, isForge);
    }
    if (collapseMirrorTabButton) {
      styleTabButton(collapseMirrorTabButton, isCollapseMirror);
    }
  }

  function formatMindTs(value) {
    if (!value) return "—";
    try {
      return new Date(value).toLocaleString();
    } catch {
      return String(value);
    }
  }

  const MIND_PREFS_STORAGE_KEY = "orion.hub.mind.prefs.v1";
  let mindModalLastFocus = null;
  let mindFocusTrapHandler = null;
  let latestMindRows = [];

  function loadMindPrefs() {
    try {
      const raw = window.localStorage.getItem(MIND_PREFS_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }

  function saveMindPrefs(next) {
    try {
      window.localStorage.setItem(MIND_PREFS_STORAGE_KEY, JSON.stringify(next || {}));
    } catch {
      // Ignore storage failures.
    }
  }

  function applyMindPrefsToControls() {
    const prefs = loadMindPrefs();
    if (mindHoursInput && Number(prefs.hours) > 0) {
      mindHoursInput.value = String(prefs.hours);
    }
    if (mindFilterOk && typeof prefs.ok === "string") {
      mindFilterOk.value = prefs.ok;
    }
    if (mindFilterTrigger && typeof prefs.trigger === "string") {
      mindFilterTrigger.value = prefs.trigger;
    }
    if (mindFilterErrorCode && typeof prefs.error_code === "string") {
      mindFilterErrorCode.value = prefs.error_code;
    }
    if (mindFilterRouterProfileId && typeof prefs.router_profile_id === "string") {
      mindFilterRouterProfileId.value = prefs.router_profile_id;
    }
    if (mindDefaultOnSendToggle) {
      mindDefaultOnSendToggle.checked = prefs.default_mind_on_send === true;
    }
  }

  function persistMindPrefsFromControls() {
    const prefs = {
      hours: Math.max(1, Number(mindHoursInput?.value || 24)),
      ok: String(mindFilterOk?.value || ""),
      trigger: String(mindFilterTrigger?.value || "").trim(),
      error_code: String(mindFilterErrorCode?.value || "").trim(),
      router_profile_id: String(mindFilterRouterProfileId?.value || "").trim(),
      default_mind_on_send: Boolean(mindDefaultOnSendToggle && mindDefaultOnSendToggle.checked),
    };
    saveMindPrefs(prefs);
    return prefs;
  }

  function mindCorrelationFromMeta(meta) {
    if (!meta || typeof meta !== "object") return "";
    const raw = meta.raw && typeof meta.raw === "object" ? meta.raw : null;
    return String(
      meta.correlationId
      || meta.correlation_id
      || meta.turnId
      || meta.turn_id
      || (raw ? raw.correlation_id : "")
      || "",
    ).trim();
  }

  function resolveMindRunsEmptyStatus(turnMeta) {
    const routing = turnMeta && typeof turnMeta.routingDebug === "object"
      ? turnMeta.routingDebug
      : (turnMeta && typeof turnMeta.routing_debug === "object" ? turnMeta.routing_debug : {});
    const raw = turnMeta && typeof turnMeta.raw === "object" ? turnMeta.raw : {};
    const rawMeta = raw && typeof raw.metadata === "object" ? raw.metadata : {};
    if (rawMeta.mind_artifact_persist_failed) {
      return "Mind ran but artifact publish failed.";
    }
    if (rawMeta.mind_invocation_failed) {
      return "Mind was requested but invocation failed.";
    }
    const mindRequested = rawMeta.mind_requested !== undefined && rawMeta.mind_requested !== null
      ? rawMeta.mind_requested
      : routing.mind_requested;
    if (mindRequested === false) {
      const skipReason = rawMeta.mind_skip_reason ? ` (${rawMeta.mind_skip_reason})` : "";
      return `Mind was not requested for this turn.${skipReason}`;
    }
    return "No Mind runs for this correlation yet.";
  }

  function mindSessionIdFromMeta(turnMeta) {
    if (!turnMeta || typeof turnMeta !== "object") return null;
    const sid = turnMeta.sessionId || turnMeta.session_id;
    return sid ? String(sid).trim() : null;
  }

  function mindRunsApiHeaders(turnMeta = null) {
    const headers = {};
    if (orionSessionId) headers["X-Orion-Session-Id"] = orionSessionId;
    return headers;
  }

  function mindRunsApiParams(correlationId, turnMeta = null) {
    const params = new URLSearchParams({ correlation_id: String(correlationId), limit: "200" });
    const contextSessionId = mindSessionIdFromMeta(turnMeta);
    if (contextSessionId && contextSessionId !== orionSessionId) {
      params.set("context_session_id", contextSessionId);
    }
    return params;
  }

  function mindRunsEmptyDiagnostics(correlationId, turnMeta = null) {
    const contextSessionId = mindSessionIdFromMeta(turnMeta);
    const lines = [
      `Correlation: ${String(correlationId)}`,
      `Active session: ${orionSessionId || "(none)"}`,
    ];
    if (contextSessionId) lines.push(`Message session: ${contextSessionId}`);
    if (contextSessionId && orionSessionId && contextSessionId !== orionSessionId) {
      lines.push("Session mismatch: Mind runs may be stored under the message session.");
    }
    return lines.join("\n");
  }

  function formatMindRunsApiError(status, detailText) {
    const detail = String(detailText || "").trim();
    if (status === 503) {
      if (/postgres_pool_unavailable|mind_store_unavailable/.test(detail)) {
        return "Mind run store is unavailable. Postgres (RECALL_PG_DSN) is down or unreachable — check orion-athena-sql-db and /mnt/postgres disk space, then restart Hub.";
      }
      if (/mind_schema_missing/.test(detail)) {
        return "Mind tables are missing. Restart Hub after Postgres is up.";
      }
      return "Mind run store temporarily unavailable.";
    }
    return detail || (status ? `HTTP ${status}` : "Request failed");
  }

  function openMindRunsModal(correlationId, triggerEl = null, turnMeta = null) {
    if (!mindRunsModal || !correlationId) return;
    mindModalLastFocus = triggerEl && typeof triggerEl.focus === "function" ? triggerEl : document.activeElement;
    mindRunsModal.classList.remove("hidden");
    mindRunsModal.setAttribute("aria-hidden", "false");
    if (mindRunsModalMeta) {
      mindRunsModalMeta.textContent = mindRunsEmptyDiagnostics(correlationId, turnMeta);
    }
    if (mindRunsModalDetails) {
      mindRunsModalDetails.textContent = "Loading runs…";
    }
    enableMindModalFocusTrap();
    refreshMindRunsForCorrelation(correlationId, turnMeta);
    syncDebugModalScrollLock();
  }

  function closeMindRunsModal() {
    if (!mindRunsModal) return;
    disableMindModalFocusTrap();
    mindRunsModal.classList.add("hidden");
    mindRunsModal.setAttribute("aria-hidden", "true");
    syncDebugModalScrollLock();
    if (mindModalLastFocus && typeof mindModalLastFocus.focus === "function") {
      mindModalLastFocus.focus();
    }
  }

  /**
   * Hub DB / transport sometimes returns jsonb columns as JSON strings; normalize to a plain value
   * so we pretty-print objects instead of one escaped string line.
   */
  function parseMindJsonbField(value) {
    if (value == null) return null;
    if (typeof value === "string") {
      const t = value.trim();
      if (!t) return null;
      try {
        return JSON.parse(t);
      } catch {
        return value;
      }
    }
    return value;
  }

  function mindJsonPrettyObject(parsed) {
    if (parsed == null) return {};
    if (typeof parsed === "object" && parsed !== null) return parsed;
    return { _unparsed: String(parsed) };
  }

  function mindRequestOverviewHtml(reqParsed) {
    if (!reqParsed || typeof reqParsed !== "object") return "";
    const rows = [];
    const push = (label, val) => {
      if (val == null) return;
      const s = String(val).trim();
      if (!s) return;
      rows.push(
        `<div class="flex flex-col gap-0.5 border-b border-gray-800/60 pb-2 last:border-0 last:pb-0"><div class="text-[10px] uppercase tracking-wide text-gray-500">${escapeHtml(label)}</div><div class="break-words font-mono text-[11px] text-gray-200">${escapeHtml(s)}</div></div>`,
      );
    };
    push("Mode", reqParsed.mode);
    push("Verb", reqParsed.verb);
    push("Session", reqParsed.session_id);
    push("Correlation", reqParsed.correlation_id);
    if (!rows.length) return "";
    return `<div class="mb-3 space-y-2 rounded border border-gray-800 bg-gray-950/60 p-3"><div class="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Request</div><div class="mt-2 space-y-2">${rows.join("")}</div></div>`;
  }

  function mindResultOverviewHtml(resultParsed) {
    if (!resultParsed || typeof resultParsed !== "object") return "";
    const brief = resultParsed.brief;
    const decision = resultParsed.decision;
    const rows = [];
    const push = (label, val) => {
      if (val == null) return;
      const s = String(val).trim();
      if (!s) return;
      rows.push(
        `<div class="flex flex-col gap-0.5 border-b border-gray-800/60 pb-2 last:border-0 last:pb-0"><div class="text-[10px] uppercase tracking-wide text-gray-500">${escapeHtml(label)}</div><div class="text-[11px] text-gray-200">${escapeHtml(s)}</div></div>`,
      );
    };
    if (brief && typeof brief === "object") {
      const sp = brief.stance_payload;
      if (sp && typeof sp === "object") {
        push("User intent", sp.user_intent);
        push("Stance", sp.stance_summary);
        push("Strategy", sp.answer_strategy);
        push("Frame", sp.conversation_frame);
      }
      if (typeof brief.summary_one_paragraph === "string" && brief.summary_one_paragraph.trim()) {
        push("Mind summary", brief.summary_one_paragraph.trim());
      }
    }
    if (decision && typeof decision === "object") {
      push("Route", decision.route_kind);
      push("Mode binding", decision.mode_binding);
      if (Array.isArray(decision.allowed_verbs) && decision.allowed_verbs.length) {
        push("Allowed verbs", decision.allowed_verbs.join(", "));
      }
      if (decision.mode_suggestion) push("Mode suggestion", decision.mode_suggestion);
    }
    if (!rows.length) return "";
    return `<div class="mb-3 space-y-2 rounded border border-gray-800 bg-gray-950/60 p-3"><div class="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Stance & routing</div><div class="mt-2 space-y-2">${rows.join("")}</div></div>`;
  }

  function renderMindRunDetails(run) {
    if (!mindRunsModalDetails) return;
    if (!run || typeof run !== "object") {
      mindRunsModalDetails.textContent = "No run details.";
      return;
    }
    const reqParsed = parseMindJsonbField(run.request_summary_jsonb);
    const resultParsed = parseMindJsonbField(run.result_jsonb);
    const reqPretty = mindJsonPrettyObject(reqParsed);
    const resultPretty = mindJsonPrettyObject(resultParsed);
    const runForRaw = { ...run, request_summary_jsonb: reqPretty, result_jsonb: resultPretty };
    const parts = [];
    const safeRunId = escapeHtml(run.mind_run_id || "—");
    const safeStatus = escapeHtml(run.ok ? "ok" : "failed");
    const safeCreated = escapeHtml(formatMindTs(run.created_at_utc));
    parts.push(`<div class="text-[11px] text-gray-400">Run ${safeRunId} · ${safeStatus} · ${safeCreated}</div>`);
    if (globalThis.OrionMindProvenance && typeof globalThis.OrionMindProvenance.renderMindProvenanceSections === "function") {
      parts.push(globalThis.OrionMindProvenance.renderMindProvenanceSections(run));
    }
    parts.push(mindRequestOverviewHtml(reqParsed));
    parts.push(mindResultOverviewHtml(resultParsed));
    parts.push(`<details class="rounded border border-gray-800 bg-gray-950/40 p-2"><summary class="cursor-pointer text-[11px] text-gray-300">Decision / brief (JSON)</summary><pre class="mt-2 whitespace-pre-wrap break-words font-mono text-[11px] text-gray-200">${escapeHtml(JSON.stringify(reqPretty, null, 2))}</pre></details>`);
    parts.push(`<details class="rounded border border-gray-800 bg-gray-950/40 p-2"><summary class="cursor-pointer text-[11px] text-gray-300">Trajectory / result (JSON)</summary><pre class="mt-2 whitespace-pre-wrap break-words font-mono text-[11px] text-gray-200">${escapeHtml(JSON.stringify(resultPretty, null, 2))}</pre></details>`);
    parts.push(`<details class="rounded border border-gray-800 bg-gray-950/40 p-2"><summary class="cursor-pointer text-[11px] text-gray-300">Raw run payload</summary><pre class="mt-2 whitespace-pre-wrap break-words font-mono text-[11px] text-gray-200">${escapeHtml(JSON.stringify(runForRaw, null, 2))}</pre></details>`);
    mindRunsModalDetails.innerHTML = parts.join("");
  }

  async function fetchMindRunDetail(mindRunId, turnMeta = null) {
    const headers = mindRunsApiHeaders(turnMeta);
    const params = new URLSearchParams();
    const contextSessionId = mindSessionIdFromMeta(turnMeta);
    if (contextSessionId && contextSessionId !== orionSessionId) {
      params.set("context_session_id", contextSessionId);
    }
    const qs = params.toString();
    const response = await fetch(
      `${API_BASE_URL}/api/mind/runs/${encodeURIComponent(mindRunId)}${qs ? `?${qs}` : ""}`,
      { headers },
    );
    if (!response.ok) {
      const detail = await response.text();
      const err = new Error(formatMindRunsApiError(response.status, detail || "(no body)"));
      err.status = response.status;
      throw err;
    }
    return await response.json();
  }

  async function refreshMindRunsForCorrelation(correlationId, turnMeta = null) {
    if (!mindRunsModalList || !mindRunsModalStatus) return;
    if (!correlationId) {
      mindRunsModalStatus.textContent = "No correlation id available for this message.";
      mindRunsModalList.innerHTML = "";
      return;
    }
    mindRunsModalStatus.textContent = "Loading runs...";
    mindRunsModalList.innerHTML = "";
    try {
      const params = mindRunsApiParams(correlationId, turnMeta);
      const headers = mindRunsApiHeaders(turnMeta);
      const response = await fetch(`${API_BASE_URL}/api/mind/runs?${params.toString()}`, { headers });
      if (!response.ok) {
        const detail = await response.text();
        const err = new Error(formatMindRunsApiError(response.status, detail || "(no body)"));
        err.status = response.status;
        throw err;
      }
      const rows = await response.json();
      if (!Array.isArray(rows) || rows.length === 0) {
        const emptyStatus = resolveMindRunsEmptyStatus(turnMeta);
        mindRunsModalStatus.textContent = emptyStatus;
        mindRunsModalList.innerHTML = '<div class="text-xs text-gray-500">No runs yet.</div>';
        if (mindRunsModalDetails) {
          mindRunsModalDetails.textContent = `${emptyStatus}\n\n${mindRunsEmptyDiagnostics(correlationId, turnMeta)}`;
        }
        if (mindRunsModalMeta) {
          mindRunsModalMeta.textContent = mindRunsEmptyDiagnostics(correlationId, turnMeta);
        }
        return;
      }
      mindRunsModalStatus.textContent = `Loaded ${rows.length} run(s).`;
      rows.forEach((row) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "w-full rounded border border-gray-800 bg-gray-950/40 px-3 py-2 text-left hover:bg-gray-900/70";
        const runIdDiv = document.createElement("div");
        runIdDiv.className = "text-[11px] text-gray-200 font-mono";
        runIdDiv.textContent = row.mind_run_id || "—";
        btn.appendChild(runIdDiv);
        const summaryDiv = document.createElement("div");
        summaryDiv.className = "mt-1 text-[11px] text-gray-400";
        const summaryParts = [
          formatMindTs(row.created_at_utc),
          row.ok ? "ok" : "failed",
          row.trigger || "—",
        ];
        if (row.error_code) summaryParts.push(row.error_code);
        summaryDiv.textContent = summaryParts.join(" · ");
        btn.appendChild(summaryDiv);
        btn.addEventListener("click", async () => {
          if (!row.mind_run_id) {
            if (mindRunsModalStatus) mindRunsModalStatus.textContent = "Run row missing id.";
            return;
          }
          if (mindRunsModalStatus) mindRunsModalStatus.textContent = `Loading run ${row.mind_run_id}...`;
          try {
            const detail = await fetchMindRunDetail(row.mind_run_id, turnMeta);
            renderMindRunDetails(detail);
            if (mindRunsModalStatus) mindRunsModalStatus.textContent = `Loaded run ${row.mind_run_id}.`;
          } catch (err) {
            if (mindRunsModalStatus) mindRunsModalStatus.textContent = `Failed to load run detail: ${String(err.message || err)}`;
          }
        });
        mindRunsModalList.appendChild(btn);
      });
      const firstRun = rows[0];
      if (firstRun && firstRun.mind_run_id) {
        if (mindRunsModalStatus) mindRunsModalStatus.textContent = `Loading run ${firstRun.mind_run_id}...`;
        try {
          const detail = await fetchMindRunDetail(firstRun.mind_run_id, turnMeta);
          renderMindRunDetails(detail);
          if (mindRunsModalStatus) mindRunsModalStatus.textContent = `Loaded ${rows.length} run(s).`;
        } catch (err) {
          renderMindRunDetails(null);
          if (mindRunsModalStatus) {
            mindRunsModalStatus.textContent = `Loaded ${rows.length} run(s); failed initial detail load: ${String(err.message || err)}`;
          }
        }
      } else {
        renderMindRunDetails(null);
      }
    } catch (err) {
      mindRunsModalStatus.textContent = `Failed to load runs: ${String(err.message || err)}`;
      mindRunsModalList.innerHTML = '<div class="text-xs text-rose-300">Could not load runs.</div>';
      if (mindRunsModalDetails) {
        mindRunsModalDetails.textContent = `${String(err.message || err)}\n\n${mindRunsEmptyDiagnostics(correlationId, turnMeta)}`;
      }
    }
  }

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function getMindModalTabbables() {
    if (!mindRunsModal) return [];
    const selectors = [
      "button",
      "[href]",
      "input",
      "select",
      "textarea",
      '[tabindex]:not([tabindex="-1"])',
    ];
    return Array.from(mindRunsModal.querySelectorAll(selectors.join(","))).filter((el) => {
      if (!(el instanceof HTMLElement)) return false;
      if (el.hasAttribute("disabled")) return false;
      return !el.classList.contains("hidden");
    });
  }

  function enableMindModalFocusTrap() {
    if (!mindRunsModal) return;
    const closeBtn = mindRunsModalClose instanceof HTMLElement ? mindRunsModalClose : null;
    (closeBtn || getMindModalTabbables()[0])?.focus?.();
    if (mindFocusTrapHandler) return;
    mindFocusTrapHandler = (event) => {
      if (event.key !== "Tab" || !mindRunsModal || mindRunsModal.classList.contains("hidden")) return;
      const tabbables = getMindModalTabbables();
      if (!tabbables.length) return;
      const first = tabbables[0];
      const last = tabbables[tabbables.length - 1];
      const active = document.activeElement;
      if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", mindFocusTrapHandler);
  }

  function disableMindModalFocusTrap() {
    if (!mindFocusTrapHandler) return;
    document.removeEventListener("keydown", mindFocusTrapHandler);
    mindFocusTrapHandler = null;
  }

  function renderMindRows(items) {
    if (!mindRunsTableBody) return;
    latestMindRows = Array.isArray(items) ? items : [];
    if (!Array.isArray(items) || items.length === 0) {
      mindRunsTableBody.innerHTML = '<tr><td colspan="7" class="px-3 py-3 text-gray-500">No Mind runs in this window.</td></tr>';
      return;
    }
    mindRunsTableBody.textContent = "";
    items.forEach((row) => {
      const tr = document.createElement("tr");
      tr.className = "border-t border-gray-800";

      const createdTd = document.createElement("td");
      createdTd.className = "px-3 py-2 whitespace-nowrap";
      createdTd.textContent = formatMindTs(row.created_at_utc);
      tr.appendChild(createdTd);

      const okTd = document.createElement("td");
      okTd.className = `px-3 py-2 ${row.ok ? "text-emerald-200" : "text-rose-200"}`;
      okTd.textContent = row.ok ? "yes" : "no";
      tr.appendChild(okTd);

      const triggerTd = document.createElement("td");
      triggerTd.className = "px-3 py-2";
      triggerTd.textContent = row.trigger || "—";
      tr.appendChild(triggerTd);

      const errorTd = document.createElement("td");
      errorTd.className = "px-3 py-2";
      errorTd.textContent = row.error_code || "—";
      tr.appendChild(errorTd);

      const routerTd = document.createElement("td");
      routerTd.className = "px-3 py-2";
      routerTd.textContent = row.router_profile_id || "—";
      tr.appendChild(routerTd);

      const corrTd = document.createElement("td");
      corrTd.className = "px-3 py-2 font-mono text-[11px]";
      corrTd.textContent = row.correlation_id || "—";
      tr.appendChild(corrTd);

      const runTd = document.createElement("td");
      runTd.className = "px-3 py-2 font-mono text-[11px]";
      const openButton = document.createElement("button");
      openButton.type = "button";
      openButton.className = "mind-run-open text-indigo-300 hover:text-indigo-100";
      openButton.textContent = row.mind_run_id || "—";
      openButton.addEventListener("click", (event) => {
        const corr = String(row.correlation_id || "").trim();
        if (!corr) {
          if (mindStatus) mindStatus.textContent = "Mind data unavailable for this row.";
          return;
        }
        openMindRunsModal(corr, event.currentTarget);
      });
      runTd.appendChild(openButton);
      tr.appendChild(runTd);

      mindRunsTableBody.appendChild(tr);
    });
  }

  async function refreshMindRuns() {
    if (!mindPanel || !mindStatus) return;
    const prefs = persistMindPrefsFromControls();
    const hours = Math.max(1, Number(prefs.hours || 24));
    mindStatus.textContent = "Loading Mind runs...";
    try {
      const params = new URLSearchParams({ hours: String(hours), limit: "200" });
      if (prefs.ok === "true") params.set("ok", "true");
      else if (prefs.ok === "false") params.set("ok", "false");
      if (prefs.trigger) params.set("trigger", prefs.trigger);
      if (prefs.error_code) params.set("error_code", prefs.error_code);
      if (prefs.router_profile_id) params.set("router_profile_id", prefs.router_profile_id);
      const headers = orionSessionId ? { 'X-Orion-Session-Id': orionSessionId } : {};
      const response = await fetch(`${API_BASE_URL}/api/mind/runs/recent?${params.toString()}`, { headers });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(formatMindRunsApiError(response.status, detail || `status ${response.status}`));
      }
      const payload = await response.json();
      renderMindRows(payload.items || []);
      const aggregates = payload.aggregates || {};
      if (mindSummaryTotal) mindSummaryTotal.textContent = String(aggregates.total_runs || 0);
      if (mindSummaryOk) mindSummaryOk.textContent = String(aggregates.ok_count || 0);
      if (mindSummaryFailed) mindSummaryFailed.textContent = String(aggregates.failed_count || 0);
      const itemCount = Array.isArray(payload.items) ? payload.items.length : 0;
      if (itemCount === 0 && payload.diagnostics && payload.diagnostics.hint) {
        mindStatus.textContent = `${payload.diagnostics.hint} (session ${payload.diagnostics.session_id || orionSessionId || "?"})`;
      } else {
        mindStatus.textContent = `Loaded ${itemCount} run(s).`;
      }
    } catch (error) {
      renderMindRows([]);
      mindStatus.textContent = `Failed to load Mind runs: ${error.message || error}`;
    }
  }

  function applyHashToTab() {
    const h = window.location.hash;
    if (h === "#topic-studio") {
      setActiveTab("topic-studio");
      refreshTopicStudio();
    } else if (h === "#service-logs") {
      setActiveTab("service-logs");
    } else if (h === "#substrate") {
      setActiveTab("substrate");
    } else if (h === "#substrate-atlas" && substrateAtlasPanel && substrateAtlasTabButton) {
      setActiveTab("substrate-atlas");
    } else if (h === "#pressure" && pressurePanel && pressureAnalyticsTabButton) {
      setActiveTab("pressure");
    } else if (h === "#substrate-lattice" && substrateLatticePanelEl && substrateLatticeTabButton) {
      setActiveTab("substrate-lattice");
    } else if (h === "#memory" && memoryPanel && memoryTabButton) {
      setActiveTab("memory");
    } else if (h === "#mind" && mindPanel && mindTabButton) {
      setActiveTab("mind");
    } else if (h === "#signals" && signalsPanel && signalsTabButton) {
      setActiveTab("signals");
    } else if (h === "#forge" && forgePanel && forgeTabButton) {
      setActiveTab("forge");
    } else if (h === "#collapse-mirror" && collapseMirrorPanel && collapseMirrorTabButton) {
      setActiveTab("collapse-mirror");
    } else {
      if (
        h === "#pressure"
        || h === "#substrate-lattice"
        || h === "#memory"
        || h === "#mind"
        || h === "#signals"
        || h === "#forge"
        || h === "#substrate-atlas"
        || h === "#collapse-mirror"
      ) {
        history.replaceState(null, "", "#hub");
      }
      setActiveTab("hub");
    }
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

  let forgeClaimsFilter = "all";
  let forgeClaimsCache = [];
  let forgeSpecsCache = [];
  let forgeStatusCache = null;
  let forgeSearchCache = null;
  let forgeCompileCache = null;
  let forgeSourceIngestCache = null;

  async function knowledgeForgeFetch(path, options = {}) {
    const response = await fetch(`${KNOWLEDGE_PROXY_BASE}${path}`, {
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
      } catch {
        payload = payloadText;
      }
    }
    if (!response.ok) {
      const detail =
        typeof payload === "object" && payload && payload.detail
          ? String(payload.detail)
          : payloadText || response.statusText || `Request failed (${response.status})`;
      const error = new Error(detail);
      error.status = response.status;
      error.body = payload;
      throw error;
    }
    if (response.status === 204) return null;
    return payload;
  }

  function forgeBadgeClass(status) {
    const normalized = String(status || "").toLowerCase();
    if (normalized === "accepted" || normalized === "execution_ready" || normalized === "reviewed") {
      return "border-emerald-700/60 bg-emerald-950/40 text-emerald-200";
    }
    if (normalized === "disputed" || normalized === "draft") {
      return "border-amber-700/60 bg-amber-950/40 text-amber-200";
    }
    if (normalized === "stale") {
      return "border-rose-700/60 bg-rose-950/40 text-rose-200";
    }
    return "border-gray-600 bg-gray-800 text-gray-300";
  }

  function forgeRenderBadge(label, status) {
    const cls = forgeBadgeClass(status);
    return `<span class="inline-flex px-1.5 py-0.5 rounded border text-[10px] uppercase tracking-wide ${cls}">${label}</span>`;
  }

  function forgeSetDebugPre(target, value) {
    if (!target) return;
    target.textContent = value == null ? "" : JSON.stringify(value, null, 2);
  }

  function forgeRenderStatusStrip(status) {
    if (!status) return;
    const set = (el, val) => {
      if (el) el.textContent = String(val ?? "—");
    };
    set(forgeStatusSources, status.source_count);
    set(forgeStatusClaims, status.claim_count);
    set(forgeStatusAccepted, status.accepted_claim_count);
    set(forgeStatusDisputed, status.disputed_claim_count);
    set(forgeStatusStale, status.stale_claim_count);
    set(forgeStatusSpecs, status.spec_count);
    set(forgeStatusExecutionReady, status.execution_ready_spec_count);
    set(forgeStatusPendingReviews, status.pending_review_count);
    set(forgeStatusContextPacks, status.context_pack_count);
    forgeSetDebugPre(forgeDebugStatus, status);
  }

  function forgeRenderTakeaway(status) {
    if (!forgeHealthBadge || !forgeSuggestedAction) return;
    const warnings = Array.isArray(status?.warnings) ? status.warnings : [];
    const enabled = status?.enabled !== false;
    const healthy = enabled && warnings.length === 0;
    forgeHealthBadge.textContent = healthy ? "Healthy" : "Degraded";
    forgeHealthBadge.className = healthy
      ? "text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full border border-emerald-600/60 bg-emerald-950/40 text-emerald-200"
      : "text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full border border-amber-600/60 bg-amber-950/40 text-amber-200";
    if (forgeWarningsList) {
      if (warnings.length) {
        forgeWarningsList.classList.remove("hidden");
        forgeWarningsList.innerHTML = warnings.map((w) => `<li>${w}</li>`).join("");
      } else {
        forgeWarningsList.classList.add("hidden");
        forgeWarningsList.innerHTML = "";
      }
    }
    if (!enabled) {
      forgeSuggestedAction.textContent =
        "Knowledge Forge is disabled on the hub proxy. Set KNOWLEDGE_FORGE_BASE_URL and ensure orion-knowledge-forge is reachable.";
    } else if ((status?.pending_review_count || 0) > 0) {
      forgeSuggestedAction.textContent = "Review pending patches in the corpus before compiling execution packs.";
    } else if ((status?.disputed_claim_count || 0) > 0) {
      forgeSuggestedAction.textContent = "Resolve disputed claims or compile with include disputed only when intentional.";
    } else if ((status?.execution_ready_spec_count || 0) === 0) {
      forgeSuggestedAction.textContent = "Promote specs to reviewed or execution_ready, then compile a context pack.";
    } else if (warnings.length) {
      forgeSuggestedAction.textContent = "Fix corpus warnings (malformed YAML, dangling refs) then refresh.";
    } else {
      forgeSuggestedAction.textContent = "Select execution-ready specs and compile a context pack for your target agent.";
    }
  }

  function forgeRenderClaimsList() {
    if (!forgeClaimsList) return;
    const filter = forgeClaimsFilter;
    const rows = forgeClaimsCache.filter((row) => {
      if (filter === "all") return true;
      return String(row.status || "").toLowerCase() === filter;
    });
    if (!rows.length) {
      forgeClaimsList.innerHTML = '<p class="text-gray-500">No claims match this filter.</p>';
      return;
    }
    forgeClaimsList.innerHTML = rows
      .map((row) => {
        const preview = String(row.statement || "").slice(0, 160);
        const refs = Array.isArray(row.source_refs) ? row.source_refs.length : 0;
        const used = Array.isArray(row.used_by) ? row.used_by.length : 0;
        return `<article class="rounded border border-gray-800 bg-gray-900/40 p-2">
          <div class="flex flex-wrap items-center gap-2 mb-1">
            <span class="font-mono text-[10px] text-gray-500">${row.claim_id || "—"}</span>
            ${forgeRenderBadge(row.status || "unknown", row.status)}
          </div>
          <p class="text-gray-200">${preview}${String(row.statement || "").length > 160 ? "…" : ""}</p>
          <p class="text-[10px] text-gray-500 mt-1">source_refs: ${refs} · used_by: ${used}</p>
        </article>`;
      })
      .join("");
  }

  function forgeStyleClaimsFilterButtons() {
    document.querySelectorAll(".forge-claims-filter").forEach((btn) => {
      const active = btn.dataset.forgeClaimsFilter === forgeClaimsFilter;
      btn.classList.toggle("border-indigo-500", active);
      btn.classList.toggle("bg-indigo-600/30", active);
      btn.classList.toggle("text-indigo-100", active);
      btn.classList.toggle("border-gray-600", !active);
      btn.classList.toggle("bg-gray-800", !active);
      btn.classList.toggle("text-gray-300", !active);
    });
  }

  function forgeRenderSpecsList() {
    if (!forgeSpecsList) return;
    const statusFilter = (forgeSpecsStatusFilter?.value || "").trim().toLowerCase();
    const rows = forgeSpecsCache.filter((row) => {
      if (!statusFilter) return true;
      return String(row.status || "").toLowerCase() === statusFilter;
    });
    if (!rows.length) {
      forgeSpecsList.innerHTML = '<p class="text-gray-500">No specs match this filter.</p>';
      return;
    }
    forgeSpecsList.innerHTML = rows
      .map((row) => {
        const claims = Array.isArray(row.source_claims) ? row.source_claims.length : 0;
        return `<article class="rounded border border-gray-800 bg-gray-900/40 p-2">
          <div class="flex flex-wrap items-center gap-2 mb-1">
            <span class="font-semibold text-gray-100">${row.title || row.spec_id || "—"}</span>
            ${forgeRenderBadge(row.status || "unknown", row.status)}
          </div>
          <p class="text-[10px] text-gray-500 font-mono">${row.spec_id || ""} · ${row.component || "—"}</p>
          <p class="text-[10px] text-gray-500 mt-1">source_claims: ${claims}</p>
        </article>`;
      })
      .join("");
  }

  function forgeRenderCompileSpecCheckboxes() {
    if (!forgeCompileSpecsList) return;
    const eligible = forgeSpecsCache.filter((row) => {
      const st = String(row.status || "").toLowerCase();
      return st === "reviewed" || st === "execution_ready";
    });
    if (!eligible.length) {
      forgeCompileSpecsList.innerHTML =
        '<p class="text-gray-500">No reviewed or execution_ready specs loaded.</p>';
      return;
    }
    forgeCompileSpecsList.innerHTML = eligible
      .map(
        (row) => `<label class="inline-flex items-start gap-2 text-gray-300">
          <input type="checkbox" class="forge-compile-spec mt-0.5 rounded border border-gray-600 bg-gray-800" value="${row.spec_id}" />
          <span><span class="font-medium text-gray-100">${row.title || row.spec_id}</span>
          <span class="text-[10px] text-gray-500 block">${row.spec_id} · ${row.status}</span></span>
        </label>`
      )
      .join("");
  }

  function forgeRenderReviewsList(reviews) {
    if (!forgeReviewsList) return;
    const rows = Array.isArray(reviews) ? reviews : [];
    if (!rows.length) {
      forgeReviewsList.innerHTML = '<p class="text-gray-500">No pending reviews.</p>';
      return;
    }
    forgeReviewsList.innerHTML = rows
      .map(
        (row) => `<article class="rounded border border-gray-800 bg-gray-900/40 p-2 flex flex-wrap gap-2 justify-between">
          <span class="font-mono text-[10px] text-gray-400 break-all">${row.path || "—"}</span>
          <span class="text-[10px] uppercase tracking-wide text-amber-200">${row.action || "—"}</span>
        </article>`
      )
      .join("");
  }

  function forgeRenderSearchResults(hits) {
    if (!forgeSearchResults) return;
    const rows = Array.isArray(hits) ? hits : [];
    if (!rows.length) {
      forgeSearchResults.innerHTML = '<p class="text-gray-500">No hits.</p>';
      return;
    }
    forgeSearchResults.innerHTML = rows
      .map(
        (row) => `<article class="rounded border border-gray-800 bg-gray-900/40 p-2 flex flex-wrap items-center gap-2 justify-between">
          <div>
            <span class="font-medium text-gray-100">${row.label || row.id || "—"}</span>
            <span class="font-mono text-[10px] text-gray-500 ml-2">${row.id || ""}</span>
          </div>
          <div class="flex gap-1">${forgeRenderBadge(row.kind || "hit", row.kind)}${
          row.status ? forgeRenderBadge(row.status, row.status) : ""
        }</div>
        </article>`
      )
      .join("");
  }

  function forgeRenderCompileResult(body) {
    if (!forgeCompileResult) return;
    if (!body) {
      forgeCompileResult.classList.add("hidden");
      return;
    }
    forgeCompileResult.classList.remove("hidden");
    if (forgeCompilePath) forgeCompilePath.textContent = body.path || "(not written)";
    if (forgeCompileIncludedSpecs) {
      forgeCompileIncludedSpecs.textContent = (body.included_specs || []).join(", ") || "—";
    }
    if (forgeCompileIncludedClaims) {
      forgeCompileIncludedClaims.textContent = (body.included_claims || []).join(", ") || "—";
    }
    if (forgeCompileExcludedClaims) {
      forgeCompileExcludedClaims.textContent = (body.excluded_claims || []).join(", ") || "—";
    }
    const warnings = Array.isArray(body.warnings) ? body.warnings : [];
    if (forgeCompileWarnings) {
      if (warnings.length) {
        forgeCompileWarnings.classList.remove("hidden");
        forgeCompileWarnings.innerHTML = warnings.map((w) => `<li>${w}</li>`).join("");
      } else {
        forgeCompileWarnings.classList.add("hidden");
        forgeCompileWarnings.innerHTML = "";
      }
    }
    if (forgeCompileContentPreview) {
      forgeCompileContentPreview.textContent = body.content || "";
    }
    forgeSetDebugPre(forgeDebugCompile, body);
  }

  async function refreshForgeTab() {
    if (!forgePanel) return;
    if (forgeStatus) forgeStatus.textContent = "Loading Knowledge Forge…";
    try {
      const [status, claims, specs, reviews] = await Promise.all([
        knowledgeForgeFetch("/status"),
        knowledgeForgeFetch("/claims"),
        knowledgeForgeFetch("/specs"),
        knowledgeForgeFetch("/reviews/pending"),
      ]);
      forgeStatusCache = status;
      forgeClaimsCache = Array.isArray(claims) ? claims : [];
      forgeSpecsCache = Array.isArray(specs) ? specs : [];
      forgeRenderStatusStrip(status);
      forgeRenderTakeaway(status);
      forgeRenderClaimsList();
      forgeStyleClaimsFilterButtons();
      forgeRenderSpecsList();
      forgeRenderCompileSpecCheckboxes();
      forgeRenderReviewsList(reviews);
      if (forgeStatus) {
        forgeStatus.textContent = `Loaded corpus · write ${status?.write_enabled ? "on" : "off"}`;
      }
    } catch (error) {
      forgeStatusCache = null;
      forgeClaimsCache = [];
      forgeSpecsCache = [];
      if (forgeStatus) {
        const msg = error?.message || String(error);
        forgeStatus.textContent =
          error?.status === 503 || /disabled|unreachable|not configured/i.test(msg)
            ? `Knowledge Forge unavailable: ${msg}`
            : `Failed to load Knowledge Forge: ${msg}`;
      }
      if (forgeHealthBadge) {
        forgeHealthBadge.textContent = "Unavailable";
        forgeHealthBadge.className =
          "text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full border border-rose-600/60 bg-rose-950/40 text-rose-200";
      }
      if (forgeSuggestedAction) {
        forgeSuggestedAction.textContent =
          "Check KNOWLEDGE_FORGE_BASE_URL, start orion-knowledge-forge, and refresh.";
      }
      if (forgeClaimsList) forgeClaimsList.innerHTML = "";
      if (forgeSpecsList) forgeSpecsList.innerHTML = "";
      if (forgeReviewsList) forgeReviewsList.innerHTML = "";
      if (forgeCompileSpecsList) forgeCompileSpecsList.innerHTML = "";
    }
  }

  async function runForgeSearch() {
    const q = (forgeSearchInput?.value || "").trim();
    if (!q) {
      if (forgeSearchResults) forgeSearchResults.innerHTML = '<p class="text-gray-500">Enter a query.</p>';
      return;
    }
    if (forgeStatus) forgeStatus.textContent = "Searching…";
    try {
      const hits = await knowledgeForgeFetch(`/search?${new URLSearchParams({ q })}`);
      forgeSearchCache = hits;
      forgeRenderSearchResults(hits);
      forgeSetDebugPre(forgeDebugSearch, hits);
      if (forgeStatus) forgeStatus.textContent = `Search returned ${Array.isArray(hits) ? hits.length : 0} hit(s).`;
    } catch (error) {
      forgeSearchCache = null;
      forgeSetDebugPre(forgeDebugSearch, { error: error?.message || String(error) });
      if (forgeSearchResults) {
        forgeSearchResults.innerHTML = `<p class="text-rose-300">Search failed: ${error?.message || error}</p>`;
      }
      if (forgeStatus) forgeStatus.textContent = `Search failed: ${error?.message || error}`;
    }
  }

  function forgeHideSourceIngestError() {
    if (forgeSourceIngestError) {
      forgeSourceIngestError.classList.add("hidden");
      forgeSourceIngestError.textContent = "";
    }
  }

  function forgeShowSourceIngestError(message) {
    if (forgeSourceIngestError) {
      forgeSourceIngestError.textContent = message;
      forgeSourceIngestError.classList.remove("hidden");
    }
    if (forgeSourceIngestResult) forgeSourceIngestResult.classList.add("hidden");
  }

  function forgeRenderSourceIngestResult(body) {
    if (!forgeSourceIngestResult) return;
    if (!body) {
      forgeSourceIngestResult.classList.add("hidden");
      return;
    }
    forgeSourceIngestResult.classList.remove("hidden");
    if (forgeSourceIngestStatus) forgeSourceIngestStatus.textContent = body.status || "—";
    if (forgeSourceIngestSourceId) forgeSourceIngestSourceId.textContent = body.source_id || "—";
    if (forgeSourceIngestSourcePath) {
      forgeSourceIngestSourcePath.textContent = body.source_path || "(not written)";
    }
    if (forgeSourceIngestReviewPath) {
      forgeSourceIngestReviewPath.textContent = body.review_path || "(not written)";
    }
    const claims = Array.isArray(body.proposed_claims) ? body.proposed_claims : [];
    if (forgeSourceIngestClaimsCount) {
      forgeSourceIngestClaimsCount.textContent = String(claims.length);
    }
    if (forgeSourceIngestClaimsList) {
      if (claims.length) {
        forgeSourceIngestClaimsList.classList.remove("hidden");
        forgeSourceIngestClaimsList.innerHTML = claims.map((c) => `<li>${escapeHtml(c)}</li>`).join("");
      } else {
        forgeSourceIngestClaimsList.classList.add("hidden");
        forgeSourceIngestClaimsList.innerHTML = "";
      }
    }
    const specs = Array.isArray(body.possibly_affected_specs) ? body.possibly_affected_specs : [];
    if (forgeSourceIngestAffectedSpecs) {
      forgeSourceIngestAffectedSpecs.textContent = specs.length ? specs.join(", ") : "—";
    }
    const warnings = Array.isArray(body.warnings) ? body.warnings : [];
    if (forgeSourceIngestWarnings) {
      if (warnings.length) {
        forgeSourceIngestWarnings.classList.remove("hidden");
        forgeSourceIngestWarnings.innerHTML = warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
      } else {
        forgeSourceIngestWarnings.classList.add("hidden");
        forgeSourceIngestWarnings.innerHTML = "";
      }
    }
    if (forgeSourceIngestContentPreview) {
      forgeSourceIngestContentPreview.textContent = body.content || "";
    }
    forgeSetDebugPre(forgeDebugSourceIngest, body);
  }

  function forgeValidateSourceIngestInputs() {
    const path = (forgeSourcePath?.value || "").trim();
    const sourceId = (forgeSourceId?.value || "").trim();
    if (!path) {
      return { ok: false, message: "Enter a source path." };
    }
    if (!sourceId) {
      return { ok: false, message: "Enter a source ID (e.g. source:my-design-doc)." };
    }
    if (!/^source:[a-z0-9][a-z0-9._-]*$/i.test(sourceId)) {
      return {
        ok: false,
        message: "Invalid source ID. Use format source:slug (letters, numbers, dots, hyphens, underscores).",
      };
    }
    return { ok: true, path, sourceId };
  }

  async function runForgeSourceIngest() {
    forgeHideSourceIngestError();
    const validation = forgeValidateSourceIngestInputs();
    if (!validation.ok) {
      forgeShowSourceIngestError(validation.message);
      showToastText(validation.message);
      return;
    }
    const payload = {
      path: validation.path,
      source_id: validation.sourceId,
      kind: forgeSourceKind?.value || "design_doc",
      dry_run: Boolean(forgeSourceDryRun?.checked),
      write_review: Boolean(forgeSourceWriteReview?.checked),
    };
    if (forgeSourceIngestButton) forgeSourceIngestButton.disabled = true;
    if (forgeStatus) forgeStatus.textContent = "Ingesting source…";
    try {
      const body = await knowledgeForgeFetch("/sources/ingest", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      forgeSourceIngestCache = body;
      forgeRenderSourceIngestResult(body);
      const claimCount = Array.isArray(body?.proposed_claims) ? body.proposed_claims.length : 0;
      if (forgeStatus) {
        forgeStatus.textContent = `Source ingest ${body?.status || "done"} · ${claimCount} proposed claim(s).`;
      }
      showToastText(`Source ingest ${body?.status || "complete"}.`);
      if (payload.write_review && !payload.dry_run) {
        await refreshForgeTab();
      }
    } catch (error) {
      forgeSourceIngestCache = { error: error?.message || String(error), body: error?.body };
      forgeSetDebugPre(forgeDebugSourceIngest, forgeSourceIngestCache);
      forgeRenderSourceIngestResult(null);
      const msg = error?.message || String(error);
      const status = error?.status;
      let display = msg;
      if (status === 403 || /write.*disabled/i.test(msg)) {
        display = `Write disabled: ${msg}. Enable KNOWLEDGE_FORGE_WRITE_ENABLED on the Forge service.`;
      } else if (status === 502 || status === 503 || /unreachable|not configured/i.test(msg)) {
        display = `Knowledge Forge unavailable: ${msg}`;
      } else if (status === 422 || status === 400) {
        display = `Validation error: ${msg}`;
      }
      forgeShowSourceIngestError(display);
      if (forgeStatus) forgeStatus.textContent = `Source ingest failed: ${msg}`;
      showToastText(`Source ingest failed: ${msg}`);
    } finally {
      if (forgeSourceIngestButton) forgeSourceIngestButton.disabled = false;
    }
  }

  async function runForgeCompile() {
    const task = (forgeCompileTask?.value || "").trim();
    if (!task) {
      showToastText("Enter a task for the context pack.");
      return;
    }
    const specIds = Array.from(
      document.querySelectorAll(".forge-compile-spec:checked")
    ).map((el) => el.value);
    const payload = {
      task,
      target: forgeCompileTarget?.value || "cursor",
      spec_ids: specIds,
      claim_ids: [],
      include_disputed: Boolean(forgeIncludeDisputed?.checked),
      include_stale: Boolean(forgeIncludeStale?.checked),
      write_file: Boolean(forgeWriteFile?.checked),
    };
    if (forgeCompileButton) forgeCompileButton.disabled = true;
    if (forgeStatus) forgeStatus.textContent = "Compiling context pack…";
    try {
      const body = await knowledgeForgeFetch("/context-packs/compile", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      forgeCompileCache = body;
      forgeRenderCompileResult(body);
      if (forgeStatus) forgeStatus.textContent = "Context pack compiled.";
      showToastText("Context pack compiled.");
    } catch (error) {
      forgeCompileCache = { error: error?.message || String(error) };
      forgeSetDebugPre(forgeDebugCompile, forgeCompileCache);
      forgeRenderCompileResult(null);
      if (forgeStatus) forgeStatus.textContent = `Compile failed: ${error?.message || error}`;
      showToastText(`Compile failed: ${error?.message || error}`);
    } finally {
      if (forgeCompileButton) forgeCompileButton.disabled = false;
    }
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

  function applyDebugTextLayout(node) {
    if (!node) return;
    node.style.whiteSpace = 'pre-wrap';
    node.style.overflowWrap = 'anywhere';
    node.style.wordBreak = 'break-word';
  }

  function toPrettyText(value) {
    if (typeof value === 'string') return value;
    if (value === null || value === undefined) return '--';
    try {
      return JSON.stringify(value, null, 2);
    } catch (_error) {
      return String(value);
    }
  }

  function summarizeInlineText(value, maxChars = 220) {
    const raw = String(value || '').trim();
    if (!raw) return '--';
    return raw.length > maxChars ? `${raw.slice(0, maxChars - 1)}…` : raw;
  }

  function collectRecallEntries(recallDebug) {
    if (!recallDebug || typeof recallDebug !== 'object') return [];
    const debugLayer = recallDebug.debug && typeof recallDebug.debug === 'object' ? recallDebug.debug : null;
    const candidates = [
      recallDebug.results,
      recallDebug.recall_results,
      recallDebug.entries,
      recallDebug.items,
      recallDebug.memories,
      debugLayer && debugLayer.results,
      debugLayer && debugLayer.recall_results,
      debugLayer && debugLayer.entries,
      debugLayer && debugLayer.items,
      debugLayer && debugLayer.memories,
    ];
    const found = candidates.find((entry) => Array.isArray(entry));
    return Array.isArray(found) ? found : [];
  }

  function normalizeMemoryDebugModel(data) {
    const recallDebug = data && typeof data.recall_debug === 'object' ? data.recall_debug : null;
    const recallCount = recallDebug && typeof recallDebug.count === 'number' ? recallDebug.count : null;
    const routingDebug = data && typeof data.routing_debug === 'object' ? data.routing_debug : null;
    const decisionDbg = recallDebug && recallDebug.decision && typeof recallDebug.decision === 'object'
      ? recallDebug.decision
      : null;
    const backendCounts = (recallDebug && recallDebug.backend_counts)
      || (decisionDbg && decisionDbg.backend_counts)
      || (recallDebug && recallDebug.debug && recallDebug.debug.backend_counts)
      || null;
    const recallProfileResolved = (recallDebug && (recallDebug.profile || recallDebug.profile_selected))
      || (decisionDbg && (decisionDbg.profile || (decisionDbg.recall_debug && decisionDbg.recall_debug.profile_selected)))
      || (routingDebug && routingDebug.recall_profile)
      || null;
    const memoryUsed = typeof data.memory_used === 'boolean'
      ? data.memory_used
      : (typeof recallCount === 'number' ? recallCount > 0 : false);
    const memoryDigest = data.memory_digest || (recallDebug && recallDebug.memory_digest) || '';
    const recallEntries = collectRecallEntries(recallDebug);

    return {
      memoryUsed,
      recallCount,
      backendCounts,
      recallProfileResolved,
      memoryDigest,
      routingDebug,
      recallEntries,
      recallDebug,
      raw: {
        memory_used: memoryUsed,
        memory_digest: memoryDigest,
        recall_debug: recallDebug || {},
        routing_debug: routingDebug || {},
      },
    };
  }

  function buildMemoryDebugRecallEntryNode(entry, index) {
    const details = document.createElement('details');
    details.className = 'rounded-xl border border-gray-700 bg-gray-900/40';

    const summary = document.createElement('summary');
    summary.className = 'cursor-pointer list-none px-3 py-2 text-xs text-gray-200';
    const source = entry && typeof entry === 'object' ? (entry.source || entry.backend || entry.kind || 'unknown') : 'scalar';
    const score = entry && typeof entry === 'object' && entry.score != null ? `score ${entry.score}` : '';
    const title = entry && typeof entry === 'object'
      ? (entry.title || entry.id || entry.uri || entry.source_ref || '')
      : '';
    const stableLabel = `Entry ${index + 1}`;
    summary.textContent = `${stableLabel} · ${source}${score ? ` · ${score}` : ''}${title ? ` · ${title}` : ''}`;
    details.dataset.recallEntry = title || `${source}-${index + 1}`;

    const body = document.createElement('div');
    body.className = 'border-t border-gray-800 px-3 py-3 space-y-2';

    const snippet = entry && typeof entry === 'object'
      ? (entry.snippet || entry.text || entry.content || entry.rendered || '')
      : '';
    if (snippet) {
      const snippetLabel = document.createElement('div');
      snippetLabel.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      snippetLabel.textContent = 'Snippet';
      const snippetPre = document.createElement('pre');
      snippetPre.className = 'whitespace-pre-wrap break-words rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-[11px] leading-5 text-gray-200 max-h-52 overflow-y-auto';
      snippetPre.textContent = String(snippet);
      applyDebugTextLayout(snippetPre);
      body.appendChild(snippetLabel);
      body.appendChild(snippetPre);
    }

    const rawLabel = document.createElement('div');
    rawLabel.className = 'text-[10px] uppercase tracking-wide text-gray-500';
    rawLabel.textContent = 'Raw entry';
    const rawPre = document.createElement('pre');
    rawPre.className = 'whitespace-pre-wrap break-words rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-[11px] leading-5 text-gray-200 max-h-72 overflow-y-auto';
    rawPre.textContent = toPrettyText(entry);
    applyDebugTextLayout(rawPre);
    body.appendChild(rawLabel);
    body.appendChild(rawPre);

    details.appendChild(summary);
    details.appendChild(body);
    return details;
  }

  function renderMemoryDebugModal(model) {
    if (!memoryDebugModalMeta || !memoryDebugModalBody) return;
    const safeModel = model || normalizeMemoryDebugModel({});
    const backendLabel = safeModel.backendCounts ? toPrettyText(safeModel.backendCounts).replace(/\s+/g, ' ') : '--';
    const profileLabel = safeModel.recallProfileResolved ? String(safeModel.recallProfileResolved) : '--';
    memoryDebugModalMeta.textContent = `memory ${safeModel.memoryUsed ? 'used' : 'unused'} · recall ${safeModel.recallCount ?? '--'} · profile ${profileLabel} · backends ${backendLabel}`;

    memoryDebugModalBody.innerHTML = '';
    const summaryGrid = document.createElement('div');
    summaryGrid.className = 'grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4';
    [
      ['memory used', safeModel.memoryUsed ? 'true' : 'false'],
      ['recall count', safeModel.recallCount ?? '--'],
      ['recall profile (resolved)', safeModel.recallProfileResolved ?? '--'],
      ['backend counts', toPrettyText(safeModel.backendCounts)],
      ['recall entries', safeModel.recallEntries.length],
    ].forEach(([label, value]) => {
      const card = document.createElement('div');
      card.className = 'rounded-xl border border-gray-800 bg-gray-900/50 px-3 py-2';
      const key = document.createElement('div');
      key.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      key.textContent = label;
      const val = document.createElement('pre');
      val.className = 'mt-1 whitespace-pre-wrap break-words text-[11px] text-gray-100';
      val.textContent = String(value);
      applyDebugTextLayout(val);
      card.appendChild(key);
      card.appendChild(val);
      summaryGrid.appendChild(card);
    });
    memoryDebugModalBody.appendChild(summaryGrid);

    const sections = [
      ['Memory digest', safeModel.memoryDigest],
      ['Outbound routing debug', safeModel.routingDebug],
      ['Recall debug payload', safeModel.recallDebug],
      ['Raw payload', safeModel.raw],
    ];
    sections.forEach(([label, value]) => {
      const block = document.createElement('section');
      block.className = 'rounded-xl border border-gray-700 bg-gray-900/40 p-3';
      const title = document.createElement('div');
      title.className = 'text-[10px] uppercase tracking-wide text-gray-500 mb-2';
      title.textContent = label;
      const pre = document.createElement('pre');
      pre.className = 'whitespace-pre-wrap break-words rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-[11px] leading-5 text-gray-200 max-h-80 overflow-y-auto';
      pre.textContent = toPrettyText(value);
      applyDebugTextLayout(pre);
      block.appendChild(title);
      block.appendChild(pre);
      memoryDebugModalBody.appendChild(block);
    });

    const entriesSection = document.createElement('section');
    entriesSection.className = 'rounded-xl border border-gray-700 bg-gray-900/40 p-3 space-y-2';
    const entriesTitle = document.createElement('div');
    entriesTitle.className = 'text-[10px] uppercase tracking-wide text-gray-500';
    entriesTitle.textContent = `Recall entries (${safeModel.recallEntries.length})`;
    entriesSection.appendChild(entriesTitle);
    if (!safeModel.recallEntries.length) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-400';
      empty.textContent = 'No explicit recall entries were provided in recall_debug.';
      entriesSection.appendChild(empty);
    } else {
      safeModel.recallEntries.forEach((entry, index) => {
        entriesSection.appendChild(buildMemoryDebugRecallEntryNode(entry, index));
      });
    }
    memoryDebugModalBody.appendChild(entriesSection);
  }

  function updateMemoryPanelFromResponse(data) {
    if (!memoryUsedValue || !recallCountValue || !backendCountsValue || !memoryDigestPre) return;
    const model = normalizeMemoryDebugModel(data);
    lastMemoryDebugModel = model;

    memoryUsedValue.textContent = model.memoryUsed ? 'true' : 'false';
    recallCountValue.textContent = typeof model.recallCount === 'number' ? model.recallCount : '--';
    backendCountsValue.textContent = model.backendCounts ? JSON.stringify(model.backendCounts, null, 2) : '--';
    memoryDigestPre.textContent = summarizeInlineText(model.memoryDigest);
    updateRoutingDebugPanel(data);
    applyDebugTextLayout(backendCountsValue);
    applyDebugTextLayout(memoryDigestPre);
    applyDebugTextLayout(outboundRoutingDebug);
    renderMemoryDebugModal(model);
  }

  function clearMemoryDebugPanel() {
    if (memoryUsedValue) memoryUsedValue.textContent = '--';
    if (recallCountValue) recallCountValue.textContent = '--';
    if (backendCountsValue) backendCountsValue.textContent = '--';
    if (memoryDigestPre) memoryDigestPre.textContent = '--';
    if (outboundRoutingDebug) outboundRoutingDebug.textContent = '--';
    lastMemoryDebugModel = normalizeMemoryDebugModel({});
    renderMemoryDebugModal(lastMemoryDebugModel);
  }

  function ensureMemoryDebugModalRootOnBody() {
    if (!memoryDebugModalRoot || !document.body) return;
    if (memoryDebugModalRoot.parentElement !== document.body) {
      document.body.appendChild(memoryDebugModalRoot);
    }
  }

  function isModalVisible(el) {
    return !!(el && !el.classList.contains('hidden'));
  }

  function syncDebugModalScrollLock() {
    if (!document.body) return;
    const shouldLock = isModalVisible(memoryDebugModalRoot)
      || isModalVisible(autonomyDebugModalRoot)
      || isModalVisible(chatStanceDebugModalRoot)
      || isModalVisible(mindRunsModal)
      || isModalVisible(chatInputExpandModalRoot)
      || isModalVisible(substrateReviewModalRoot)
      || isModalVisible(cognitiveReviewModalRoot)
      || isModalVisible(autonomyConstitutionModalRoot)
      || isModalVisible(agentTraceModal);
    document.body.classList.toggle('overflow-hidden', shouldLock);
  }

  function ensureChatInputExpandModalRootOnBody() {
    if (!chatInputExpandModalRoot || !document.body) return;
    if (chatInputExpandModalRoot.parentElement !== document.body) {
      document.body.appendChild(chatInputExpandModalRoot);
    }
  }

  function syncChatExpandTextareaFromInput() {
    if (!chatInput || !chatInputExpandTextarea) return;
    chatInputExpandTextarea.value = chatInput.value || '';
  }

  function syncChatInputFromExpandTextarea() {
    if (!chatInput || !chatInputExpandTextarea) return;
    chatInput.value = chatInputExpandTextarea.value || '';
  }

  function openChatInputExpandModal() {
    if (!chatInputExpandModalRoot || !chatInputExpandTextarea) return;
    ensureChatInputExpandModalRootOnBody();
    syncChatExpandTextareaFromInput();
    chatInputExpandModalRoot.style.position = 'fixed';
    chatInputExpandModalRoot.style.inset = '0';
    chatInputExpandModalRoot.style.zIndex = '2147483646';
    if (chatInputExpandModalBackdrop) {
      chatInputExpandModalBackdrop.style.position = 'fixed';
      chatInputExpandModalBackdrop.style.inset = '0';
      chatInputExpandModalBackdrop.style.zIndex = '2147483646';
    }
    if (chatInputExpandModalDialog) {
      chatInputExpandModalDialog.style.position = 'fixed';
      chatInputExpandModalDialog.style.zIndex = '2147483647';
    }
    chatInputExpandModalRoot.classList.remove('hidden');
    chatInputExpandModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
    chatInputExpandTextarea.focus();
    chatInputExpandTextarea.setSelectionRange(
      chatInputExpandTextarea.value.length,
      chatInputExpandTextarea.value.length,
    );
  }

  function closeChatInputExpandModal(opts = {}) {
    if (!chatInputExpandModalRoot) return;
    const applyToInput = opts && Object.prototype.hasOwnProperty.call(opts, 'applyToInput')
      ? Boolean(opts.applyToInput)
      : true;
    const focusInput = opts && Object.prototype.hasOwnProperty.call(opts, 'focusInput')
      ? Boolean(opts.focusInput)
      : true;
    if (applyToInput) syncChatInputFromExpandTextarea();
    chatInputExpandModalRoot.classList.add('hidden');
    chatInputExpandModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
    if (focusInput && chatInput) chatInput.focus();
  }

  async function sendExpandedChatMessage() {
    if (!chatInputExpandTextarea) return;
    const value = String(chatInputExpandTextarea.value || '').trim();
    if (!value) return;
    syncChatInputFromExpandTextarea();
    closeChatInputExpandModal({ applyToInput: false, focusInput: false });
    await submitExplicitChatText(value);
  }

  function openMemoryDebugModal() {
    if (!memoryDebugModalRoot) return;
    closeAutonomyDebugModal();
    ensureMemoryDebugModalRootOnBody();
    memoryDebugModalRoot.style.position = 'fixed';
    memoryDebugModalRoot.style.inset = '0';
    memoryDebugModalRoot.style.zIndex = '2147483646';
    if (memoryDebugModalBackdrop) {
      memoryDebugModalBackdrop.style.position = 'fixed';
      memoryDebugModalBackdrop.style.inset = '0';
      memoryDebugModalBackdrop.style.zIndex = '2147483646';
    }
    if (memoryDebugModalDialog) {
      memoryDebugModalDialog.style.position = 'fixed';
      memoryDebugModalDialog.style.zIndex = '2147483647';
    }
    renderMemoryDebugModal(lastMemoryDebugModel || normalizeMemoryDebugModel({}));
    memoryDebugModalRoot.classList.remove('hidden');
    memoryDebugModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
  }

  function closeMemoryDebugModal() {
    if (!memoryDebugModalRoot) return;
    memoryDebugModalRoot.classList.add('hidden');
    memoryDebugModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
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
    lastAgentTraceSummary = null;
    lastAgentTraceMeta = {};
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
    lastAgentTraceSummary = summary;
    lastAgentTraceMeta = meta || {};

    agentTraceDebugPanel.classList.remove('hidden');
    if (agentTraceDebugMeta) {
      const corr = summary.corr_id || meta.correlationId || '--';
      agentTraceDebugMeta.textContent = `corr ${corr} · status ${summary.status || '--'} · ${summary.step_count || 0} steps`;
    }

    agentTraceDebugOverview.innerHTML = '';
    Array.from(buildAgentTraceOverviewNode(summary).children).forEach((child) => agentTraceDebugOverview.appendChild(child));

    agentTraceDebugSummary.textContent = summary.summary_text || 'No deterministic summary available.';

    const grouped = agentTraceApi.groupToolsByFamily ? agentTraceApi.groupToolsByFamily(summary.tools) : [];
    const timelineRows = agentTraceApi.buildTimelineRows ? agentTraceApi.buildTimelineRows(summary) : [];
    const compactRows = [
      ['Tool families', grouped.length],
      ['Total tool calls', summary.tool_call_count ?? summary.tools_count ?? '--'],
      ['Timeline events', timelineRows.length],
      ['Status', summary.status || '--'],
    ];
    agentTraceDebugToolGroups.innerHTML = '';
    compactRows.forEach(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'flex items-center justify-between rounded-lg border border-gray-700 bg-gray-900/50 px-3 py-2';
      const key = document.createElement('span');
      key.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      key.textContent = String(label);
      const val = document.createElement('span');
      val.className = 'text-xs text-gray-200';
      val.textContent = String(value ?? '--');
      row.appendChild(key);
      row.appendChild(val);
      agentTraceDebugToolGroups.appendChild(row);
    });

    agentTraceDebugTimeline.innerHTML = '';
    agentTraceDebugTimeline.innerHTML = '<div class="text-[11px] text-gray-400">Timeline details moved to modal for full inspection.</div>';
    agentTraceDebugRaw.innerHTML = '';
    agentTraceDebugRaw.innerHTML = '<div class="text-[11px] text-gray-400">Raw payload inspection is available in the modal.</div>';
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
    if (key === 'relational_stability' || key === 'relational') return 'relationally steady';
    if (key === 'capability_expansion') return 'forward-building';
    if (key === 'predictive_mastery' || key === 'predictive') return 'clarifying / forecasting';
    // Graph-native dominant drives: no keyword gate (avoid permanent "not clearly visible").
    if (key === 'autonomy') return null;
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
      { label: 'clarifying / forecasting', patterns: ['likely', 'if we', 'next step', 'expect', 'forecast', 'prediction'] },
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
    let alignmentNote = 'no fixed keyword gate for this dominant drive';
    if (model && model.stateQuality && String(model.stateQuality).startsWith('degraded')) {
      alignmentNote = model.stanceMode === 'proposal_only'
        ? 'proposal-only because selected subject drives were unavailable'
        : (model.degradedReason || 'autonomy state degraded');
    } else if (expectedPosture === 'neutral') {
      alignmentNote = visibleCues.length
        ? 'informal stylistic cues present (neutral drive)'
        : 'no strong posture expected';
    } else if (expectedPosture) {
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

  function formatAutonomyFieldLabel(model, field) {
    const degraded = model && model.stateQuality && String(model.stateQuality).startsWith('degraded');
    const reason = String((model && model.degradedReason) || '').trim();
    const contextNote = String((model && model.contextNote) || '').trim();
    const selectedSubject = String((model && model.selectedSubject) || '').trim().toLowerCase();
    const usingRelationshipFallback = selectedSubject === 'relationship' && contextNote.toLowerCase().includes('orion drives unavailable');
    if (field === 'dominantDrive') {
      if (model && model.dominantDrive) return model.dominantDrive;
      if (degraded && usingRelationshipFallback) return 'unavailable — Orion drives skipped or timed out (see context note)';
      if (degraded) return `unavailable — ${reason || 'selected subject drives unavailable'}`;
      return '--';
    }
    if (field === 'topDrives') {
      if (model && model.topDrives && model.topDrives.length) return model.topDrives.join(', ');
      if (degraded) return 'unavailable — drives facet failed';
      return '--';
    }
    if (field === 'tensions') {
      if (model && model.tensions && model.tensions.length) return model.tensions.join(', ');
      if (degraded) return 'unavailable — drive competition requires drives';
      return '--';
    }
    if (field === 'expectedPosture') {
      if (model && model.alignment && model.alignment.expected_posture) return model.alignment.expected_posture;
      if (degraded && model && model.stanceMode === 'proposal_only') return 'proposal-only (drives unavailable)';
      if (degraded) return `unavailable — ${reason || 'selected subject drives unavailable'}`;
      return '--';
    }
    return '--';
  }

  function hubAutonomySubjectDisplayMode() {
    const raw = String(window.__HUB_AUTONOMY_SUBJECT_DISPLAY__ || 'two').trim().toLowerCase();
    return raw === 'three' ? 'three' : 'two';
  }

  function autonomyAvailabilityRowsForDisplay(safeDebug, mode) {
    const rows = safeDebug
      ? Object.entries(safeDebug).filter(([subject, value]) => !String(subject).startsWith('_') && value && typeof value === 'object')
      : [];
    if (mode !== 'two') return rows;
    return rows.filter(([subject]) => {
      const key = String(subject || '').trim().toLowerCase();
      return key === 'orion' || key === 'relationship';
    });
  }

  function formatAutonomyAvailabilityLine(model, mode) {
    if (mode === 'two') {
      return `availability: orion + relationship (orion↔juniper) · ${model.availability.available}/${model.availability.subjects} available · ${model.availability.degraded} degraded`;
    }
    return `availability: ${model.availability.available}/${model.availability.subjects} available · ${model.availability.degraded} degraded`;
  }

  function normalizeAutonomyModel(summary, debug, meta = {}) {
    const safeSummary = summary && typeof summary === 'object' ? summary : null;
    const safeDebug = debug && typeof debug === 'object' ? debug : null;
    const safePreview = meta && meta.autonomyStatePreview && typeof meta.autonomyStatePreview === 'object'
      ? meta.autonomyStatePreview
      : (meta && meta.autonomy_state_preview && typeof meta.autonomy_state_preview === 'object' ? meta.autonomy_state_preview : null);
    const safeV2Preview = meta && meta.autonomyStateV2Preview && typeof meta.autonomyStateV2Preview === 'object'
      ? meta.autonomyStateV2Preview
      : (meta && meta.autonomy_state_v2_preview && typeof meta.autonomy_state_v2_preview === 'object' ? meta.autonomy_state_v2_preview : null);
    const safeDelta = meta && meta.autonomyStateDelta && typeof meta.autonomyStateDelta === 'object'
      ? meta.autonomyStateDelta
      : (meta && meta.autonomy_state_delta && typeof meta.autonomy_state_delta === 'object' ? meta.autonomy_state_delta : null);
    const safeChatStanceDebug = meta && meta.chatStanceDebug && typeof meta.chatStanceDebug === 'object'
      ? meta.chatStanceDebug
      : (meta && meta.chat_stance_debug && typeof meta.chat_stance_debug === 'object' ? meta.chat_stance_debug : null);
    const topDrives = Array.isArray((safeSummary && safeSummary.top_drives) || (safePreview && safePreview.top_drives))
      ? ((safeSummary && safeSummary.top_drives) || (safePreview && safePreview.top_drives)).map((v) => String(v || '').trim()).filter(Boolean).slice(0, 3)
      : [];
    const tensions = Array.isArray((safeSummary && safeSummary.active_tensions) || (safePreview && safePreview.active_tensions))
      ? ((safeSummary && safeSummary.active_tensions) || (safePreview && safePreview.active_tensions)).map((v) => String(v || '').trim()).filter(Boolean).slice(0, 3)
      : [];
    const proposalHeadlines = Array.isArray((safeSummary && safeSummary.proposal_headlines) || (safePreview && safePreview.proposal_headlines))
      ? ((safeSummary && safeSummary.proposal_headlines) || (safePreview && safePreview.proposal_headlines)).map((v) => String(v || '').trim()).filter(Boolean).slice(0, 3)
      : [];
    const goalsPresent = Boolean(
      meta.autonomyGoalsPresent
      || meta.autonomy_goals_present
      || (safeSummary && safeSummary.goals_present)
      || proposalHeadlines.length
    );
    const displayMode = hubAutonomySubjectDisplayMode();
    const availabilityRows = autonomyAvailabilityRowsForDisplay(safeDebug, displayMode);
    const availableCount = availabilityRows.filter(([, value]) => value.availability === 'available').length;
    const degradedCount = availabilityRows.filter(([, value]) => value.availability === 'degraded').length;
    const unavailableCount = availabilityRows.filter(([, value]) => value.availability === 'unavailable').length;
    const subjectCount = displayMode === 'two' ? 2 : availabilityRows.length;
    // Keep this exact fallback expression stable for UI contract tests:
    // (safeSummary && safeSummary.dominant_drive) || (safePreview && safePreview.dominant_drive)
    const dominantDrive = String(
      (safeSummary && safeSummary.dominant_drive)
        || (safePreview && safePreview.dominant_drive)
        || ''
    ).trim();
    const runtimeRepositoryStatus = (safeDebug && safeDebug._runtime && safeDebug._runtime.repository_status && typeof safeDebug._runtime.repository_status === 'object')
      ? safeDebug._runtime.repository_status
      : {};
    const repositoryStatus = (meta.autonomyRepositoryStatus && typeof meta.autonomyRepositoryStatus === 'object')
      ? meta.autonomyRepositoryStatus
      : ((meta.autonomy_repository_status && typeof meta.autonomy_repository_status === 'object') ? meta.autonomy_repository_status : runtimeRepositoryStatus);
    const executionMode = String((meta.autonomyExecutionMode != null && meta.autonomyExecutionMode)
      || meta.autonomy_execution_mode || '').trim();
    const goalLineageRaw = (meta.autonomyGoalLineage && typeof meta.autonomyGoalLineage === 'object')
      ? meta.autonomyGoalLineage
      : ((meta.autonomy_goal_lineage && typeof meta.autonomy_goal_lineage === 'object')
        ? meta.autonomy_goal_lineage
        : (safePreview && safePreview.goal_lineage) || null);
    const driveCompetition = (safeSummary && safeSummary.drive_competition && typeof safeSummary.drive_competition === 'object')
      ? safeSummary.drive_competition
      : ((safePreview && safePreview.drive_competition && typeof safePreview.drive_competition === 'object')
        ? safePreview.drive_competition
        : null);
    const runtimeMeta = (safeDebug && safeDebug._runtime && typeof safeDebug._runtime === 'object') ? safeDebug._runtime : {};
    const stateQuality = String((safeSummary && safeSummary.state_quality) || runtimeMeta.state_quality || '').trim();
    const stanceMode = String((safeSummary && safeSummary.stance_mode) || runtimeMeta.stance_mode || '').trim();
    const degradedReason = String((safeSummary && safeSummary.degraded_reason) || runtimeMeta.degraded_reason || '').trim();
    const contextNote = String((safeSummary && safeSummary.context_note) || runtimeMeta.context_note || '').trim();
    const facetHealth = (safeSummary && safeSummary.facet_health && typeof safeSummary.facet_health === 'object')
      ? safeSummary.facet_health
      : ((runtimeMeta.facet_health && typeof runtimeMeta.facet_health === 'object') ? runtimeMeta.facet_health : {});
    const hasSemanticSignal = !!(dominantDrive || topDrives.length || tensions.length || proposalHeadlines.length
      || (driveCompetition && (driveCompetition.top_drive || driveCompetition.runner_drive))
      || (stateQuality && stateQuality.startsWith('degraded')));
    const hasDebugSignal = !!(safeDebug && typeof safeDebug === 'object' && Object.keys(safeDebug).length);
    const hasPreviewSignal = !!(safePreview && (
      safePreview.dominant_drive
      || (Array.isArray(safePreview.top_drives) && safePreview.top_drives.length)
      || (Array.isArray(safePreview.active_tensions) && safePreview.active_tensions.length)
      || (Array.isArray(safePreview.proposal_headlines) && safePreview.proposal_headlines.length)
      || (safePreview.drive_competition && (safePreview.drive_competition.top_drive || safePreview.drive_competition.runner_drive))
      || (safePreview.goal_lineage && Object.keys(safePreview.goal_lineage).length)
    ));
    const hasV2PreviewSignal = !!(safeV2Preview && Object.keys(safeV2Preview).length);
    const hasDeltaSignal = !!(safeDelta && Object.keys(safeDelta).length);
    const hasChatStanceDebugSignal = !!(safeChatStanceDebug && Object.keys(safeChatStanceDebug).length);
    const hasLineageMeta = !!(executionMode || (goalLineageRaw && Object.keys(goalLineageRaw).length));
    const hasAnySignal = !!(
      hasSemanticSignal
      || hasDebugSignal
      || hasPreviewSignal
      || hasV2PreviewSignal
      || hasDeltaSignal
      || hasChatStanceDebugSignal
      || String((safeSummary && safeSummary.stance_hint) || '').trim()
      || hasLineageMeta
    );
    if (!hasAnySignal) return null;

    return {
      dominantDrive,
      topDrives,
      tensions,
      driveCompetition,
      proposals: proposalHeadlines,
      proposalHeadlines,
      goalsPresent,
      stanceHint: String((safeSummary && safeSummary.stance_hint) || '').trim(),
      hasSemanticSignal,
      hasDebugSignal,
      hasPreviewSignal,
      hasV2PreviewSignal,
      hasDeltaSignal,
      hasChatStanceDebugSignal,
      executionMode,
      goalLineage: goalLineageRaw,
      backend: String((meta.autonomyBackend || (safeDebug && safeDebug._runtime && safeDebug._runtime.backend) || meta.backend || '')).trim() || '--',
      selectedSubject: String((meta.autonomySelectedSubject || (safeDebug && safeDebug._runtime && safeDebug._runtime.selected_subject) || meta.selectedSubject || '')).trim() || '--',
      stateQuality: stateQuality || 'empty',
      stanceMode,
      degradedReason,
      contextNote,
      facetHealth,
      availability: {
        available: availableCount,
        degraded: degradedCount,
        unavailable: unavailableCount,
        subjects: subjectCount,
      },
      fallback: unavailableCount > 0 ? 'yes' : 'no',
      repositoryStatus: {
        source_available: !!repositoryStatus.source_available,
        source_path: String(repositoryStatus.source_path || '').trim() || '--',
      },
      alignment: computeAutonomyAlignment(
        { dominantDrive, stateQuality, stanceMode, degradedReason, contextNote, selectedSubject },
        meta.replyText || meta.reply_text || '',
      ),
      raw: {
        summary: safeSummary || {},
        debug: safeDebug || {},
        state_preview: safePreview || {},
        state_v2_preview: safeV2Preview || {},
        state_delta: safeDelta || {},
        chat_stance_debug: safeChatStanceDebug || {},
        runtime: {
          backend: String((meta.autonomyBackend || (safeDebug && safeDebug._runtime && safeDebug._runtime.backend) || meta.backend || '')).trim() || '--',
          selected_subject: String((meta.autonomySelectedSubject || (safeDebug && safeDebug._runtime && safeDebug._runtime.selected_subject) || meta.selectedSubject || '')).trim() || '--',
          state_quality: stateQuality || 'empty',
          stance_mode: stanceMode,
          degraded_reason: degradedReason,
          context_note: contextNote,
          facet_health: facetHealth,
          repository_status: {
            source_available: !!repositoryStatus.source_available,
            source_path: String(repositoryStatus.source_path || '').trim() || '--',
          },
        },
      },
    };
  }

  function shouldRenderAutonomyInline(model) {
    if (!model || typeof model !== 'object') return false;
    const dc = model.driveCompetition;
    const hasDc = dc && typeof dc === 'object' && (dc.top_drive || dc.runner_drive);
    const degraded = model.stateQuality && String(model.stateQuality).startsWith('degraded');
    const hasDebug = Boolean(model.hasDebugSignal);
    const hasStanceHint = Boolean(String(model.stanceHint || '').trim());
    const hasLineage = Boolean(model.executionMode || (model.goalLineage && Object.keys(model.goalLineage).length));
    return !!(
      model.dominantDrive
      || (model.topDrives || []).length
      || (model.tensions || []).length
      || (model.proposals || []).length
      || (model.proposalHeadlines || []).length
      || hasDc
      || degraded
      || hasDebug
      || hasStanceHint
      || model.hasPreviewSignal
      || model.hasV2PreviewSignal
      || model.hasDeltaSignal
      || model.hasChatStanceDebugSignal
      || hasLineage
    );
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
      autonomyDebugMeta.textContent = `backend ${model.backend} · selected ${model.selectedSubject} · runtime+semantic`;
    }

    const executionModeDisplay = model.executionMode === 'proposal_only'
      ? 'none (legacy proposal_only)'
      : model.executionMode;
    const displayMode = hubAutonomySubjectDisplayMode();
    const juniperNote = 'n/a — dyadic scope uses relationship';
    const juniperDebugEntry = debug && typeof debug === 'object' ? debug.juniper : null;
    const juniperAvailabilityEmpty = juniperDebugEntry && typeof juniperDebugEntry === 'object'
      && (!String(juniperDebugEntry.availability || '').trim()
        || ['unavailable', 'empty'].includes(String(juniperDebugEntry.availability || '').trim()));

    autonomyDebugOverview.innerHTML = '';
    [
      `autonomy state: ${model.stateQuality || 'unknown'}`,
      `backend: ${model.backend}`,
      `selected subject: ${model.selectedSubject}`,
      ...(model.degradedReason ? [`reason: ${model.degradedReason}`] : []),
      ...(model.contextNote ? [`context note: ${model.contextNote}`] : []),
      ...(model.stanceMode ? [`stance mode: ${model.stanceMode}`] : []),
      ...(model.goalsPresent ? [`goals: ${model.proposalHeadlines.length} active`] : []),
      formatAutonomyAvailabilityLine(model, displayMode),
      ...(displayMode === 'three' && juniperAvailabilityEmpty ? [`juniper: ${juniperNote}`] : []),
      `repository source: ${model.repositoryStatus.source_available ? 'available' : 'unavailable'}`,
      `repository path: ${model.repositoryStatus.source_path}`,
      `fallback: ${model.fallback}`,
      ...(executionModeDisplay ? [`execution: ${executionModeDisplay}`] : []),
      ...(model.goalLineage && Object.keys(model.goalLineage).length
        ? [`goal lineage: ${JSON.stringify(model.goalLineage)}`]
        : []),
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      autonomyDebugOverview.appendChild(row);
    });

    autonomyDebugState.innerHTML = '';
    [
      `dominant drive: ${formatAutonomyFieldLabel(model, 'dominantDrive')}`,
      `top drives: ${formatAutonomyFieldLabel(model, 'topDrives')}`,
      `top tensions: ${formatAutonomyFieldLabel(model, 'tensions')}`,
      ...(Object.keys(model.facetHealth || {}).length
        ? [`facet health: ${Object.entries(model.facetHealth).map(([k, v]) => `${k}=${v}`).join(', ')}`]
        : []),
      ...(model.driveCompetition && model.driveCompetition.top_drive && model.driveCompetition.runner_drive
        ? [`competing pressures: ${model.driveCompetition.top_drive} ${Number(model.driveCompetition.pressure_top).toFixed(2)} vs ${model.driveCompetition.runner_drive} ${Number(model.driveCompetition.pressure_runner).toFixed(2)} (spread ${Number(model.driveCompetition.spread).toFixed(2)})`]
        : []),
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      autonomyDebugState.appendChild(row);
    });

    let proposalsTitle = document.getElementById('autonomyDebugProposalsTitle');
    if (!proposalsTitle && autonomyDebugProposals && autonomyDebugProposals.parentElement) {
      proposalsTitle = document.createElement('div');
      proposalsTitle.id = 'autonomyDebugProposalsTitle';
      proposalsTitle.className = 'text-xs uppercase tracking-wide text-amber-400/80 mb-1';
      autonomyDebugProposals.parentElement.insertBefore(proposalsTitle, autonomyDebugProposals);
    }
    if (proposalsTitle) {
      proposalsTitle.textContent = model.stateQuality === 'healthy'
        ? 'Active goals (non-executing)'
        : 'Proposals (proposal-only)';
    }

    autonomyDebugProposals.innerHTML = '';
    if (model.proposalHeadlines.length) {
      model.proposalHeadlines.forEach((proposal) => {
        const row = document.createElement('div');
        row.className = 'rounded-lg border border-amber-500/30 bg-amber-500/5 px-2 py-1';
        row.textContent = proposal;
        autonomyDebugProposals.appendChild(row);
      });
    } else {
      autonomyDebugProposals.textContent = '--';
    }

    autonomyDebugAlignment.innerHTML = '';
    [
      `expected posture: ${formatAutonomyFieldLabel(model, 'expectedPosture')}`,
      `visible cues: ${model.alignment && model.alignment.visible_cues && model.alignment.visible_cues.length ? model.alignment.visible_cues.join(', ') : '--'}`,
      `alignment note: ${(model.alignment && model.alignment.alignment_note) || '--'}`,
      ...(model.stanceMode === 'proposal_only' && model.degradedReason
        ? [`stance: proposal-only because ${model.degradedReason.toLowerCase()}`]
        : []),
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      autonomyDebugAlignment.appendChild(row);
    });

    autonomyDebugRaw.textContent = safeHubJsonStringify(model.raw);
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
    closeMemoryDebugModal();
    ensureAutonomyModalRootOnBody();
    autonomyDebugModalRoot.style.position = 'fixed';
    autonomyDebugModalRoot.style.inset = '0';
    autonomyDebugModalRoot.style.zIndex = '2147483646';
    if (autonomyDebugModalBackdrop) {
      autonomyDebugModalBackdrop.style.position = 'fixed';
      autonomyDebugModalBackdrop.style.inset = '0';
      autonomyDebugModalBackdrop.style.zIndex = '2147483646';
    }
    if (autonomyDebugModalDialog) {
      autonomyDebugModalDialog.style.position = 'fixed';
      autonomyDebugModalDialog.style.zIndex = '2147483647';
    }
    if (autonomyDebugModalMeta) autonomyDebugModalMeta.textContent = autonomyDebugMeta ? autonomyDebugMeta.textContent : '--';
    if (autonomyDebugModalBody) autonomyDebugModalBody.innerHTML = autonomyDebugBody ? autonomyDebugBody.innerHTML : '--';
    autonomyDebugModalRoot.classList.remove('hidden');
    autonomyDebugModalRoot.setAttribute('aria-hidden', 'false');
    if (document.body) document.body.classList.add('overflow-hidden');
    syncDebugModalScrollLock();
  }

  function closeAutonomyDebugModal() {
    if (!autonomyDebugModalRoot) return;
    autonomyDebugModalRoot.classList.add('hidden');
    autonomyDebugModalRoot.setAttribute('aria-hidden', 'true');
    if (document.body) document.body.classList.remove('overflow-hidden');
    syncDebugModalScrollLock();
  }

  function clearChatStanceDebugPanel() {
    lastChatStanceDebug = null;
    if (chatStanceDebugBody) chatStanceDebugBody.classList.add('hidden');
    if (chatStanceDebugCaret) chatStanceDebugCaret.textContent = '▾';
    if (chatStanceDebugMeta) chatStanceDebugMeta.textContent = 'No chat stance debug payload on this turn.';
    if (chatStanceDebugOverview) chatStanceDebugOverview.textContent = 'No chat stance debug payload on this turn.';
    if (chatStanceDebugLineage) chatStanceDebugLineage.textContent = '--';
    if (chatStanceDebugRaw) chatStanceDebugRaw.textContent = 'No chat stance debug payload on this turn.';
    if (chatStanceDebugModalMeta) chatStanceDebugModalMeta.textContent = 'No chat stance debug payload on this turn.';
    if (chatStanceDebugModalBody) chatStanceDebugModalBody.textContent = 'No chat stance debug payload on this turn.';
  }

  function buildChatStanceSection(title, value) {
    const section = document.createElement('section');
    section.className = 'rounded-xl border border-gray-700 bg-gray-900/50 p-3 space-y-2';
    const heading = document.createElement('div');
    heading.className = 'text-[10px] uppercase tracking-wide text-gray-500';
    heading.textContent = title;
    section.appendChild(heading);
    const pre = document.createElement('pre');
    pre.className = 'whitespace-pre-wrap break-words text-[11px] text-gray-200';
    pre.textContent = JSON.stringify(value || {}, null, 2);
    section.appendChild(pre);
    return section;
  }

  function updateChatStanceDebugPanel(payload) {
    if (!chatStanceDebugPanel) return;
    const model = payload && typeof payload === 'object' ? payload : null;
    if (!model) {
      clearChatStanceDebugPanel();
      return;
    }
    lastChatStanceDebug = model;
    const overview = model.overview && typeof model.overview === 'object' ? model.overview : {};
    const lineage = Array.isArray(model.lineage_summary) ? model.lineage_summary : [];
    if (chatStanceDebugMeta) {
      chatStanceDebugMeta.textContent = `categories ${Array.isArray(overview.categories_present) ? overview.categories_present.length : 0} · fallback ${overview.fallback_invoked ? 'yes' : 'no'} · quality modified ${overview.quality_enforcement_modified ? 'yes' : 'no'}`;
    }
    if (chatStanceDebugOverview) {
      chatStanceDebugOverview.innerHTML = '';
      [
        `categories present: ${(overview.categories_present || []).join(', ') || '--'}`,
        `fallback invoked: ${overview.fallback_invoked ? 'yes' : 'no'}`,
        `normalized applied: ${overview.normalized_applied ? 'yes' : 'no'}`,
        `quality enforcement modified: ${overview.quality_enforcement_modified ? 'yes' : 'no'}`,
        `semantic fallback: ${overview.semantic_fallback ? 'yes' : 'no'}`,
      ].forEach((line) => {
        const row = document.createElement('div');
        row.textContent = line;
        chatStanceDebugOverview.appendChild(row);
      });
    }
    if (chatStanceDebugLineage) {
      chatStanceDebugLineage.innerHTML = '';
      if (!lineage.length) {
        chatStanceDebugLineage.textContent = '--';
      } else {
        lineage.forEach((line) => {
          const row = document.createElement('div');
          row.textContent = String(line || '--');
          chatStanceDebugLineage.appendChild(row);
        });
      }
    }
    if (chatStanceDebugRaw) chatStanceDebugRaw.textContent = safeHubJsonStringify(model);
    if (chatStanceDebugModalMeta) chatStanceDebugModalMeta.textContent = chatStanceDebugMeta ? chatStanceDebugMeta.textContent : '--';
    if (chatStanceDebugModalBody) {
      chatStanceDebugModalBody.innerHTML = '';
      if (!model || Object.keys(model).length === 0) {
        chatStanceDebugModalBody.textContent = 'No chat stance debug payload on this turn.';
      } else {
        chatStanceDebugModalBody.appendChild(buildChatStanceSection('Overview', model.overview || {}));
        chatStanceDebugModalBody.appendChild(buildChatStanceSection('Source Inputs by Category', model.source_inputs || {}));
        chatStanceDebugModalBody.appendChild(buildChatStanceSection('Attention / Curiosity', (model.source_inputs && model.source_inputs.attention_frame) || {}));
        chatStanceDebugModalBody.appendChild(
          buildChatStanceSection(
            'Journal PageIndex',
            (model.source_inputs && model.source_inputs.journal_pageindex) || {}
          )
        );
        chatStanceDebugModalBody.appendChild(buildChatStanceSection('Synthesized Brief', model.synthesized_brief || {}));
        chatStanceDebugModalBody.appendChild(buildChatStanceSection('Enforcement / Fallback', model.enforcement || {}));
        chatStanceDebugModalBody.appendChild(buildChatStanceSection('Final Prompt Contract', model.final_prompt_contract || {}));
        chatStanceDebugModalBody.appendChild(buildChatStanceSection('Raw compact JSON', model.raw || model));
      }
    }
  }

  function toggleChatStanceDebugPanel() {
    if (!chatStanceDebugBody) return;
    const nextHidden = !chatStanceDebugBody.classList.contains('hidden');
    chatStanceDebugBody.classList.toggle('hidden', nextHidden);
    if (chatStanceDebugCaret) chatStanceDebugCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function ensureChatStanceModalRootOnBody() {
    if (!chatStanceDebugModalRoot || !document.body) return;
    if (chatStanceDebugModalRoot.parentElement !== document.body) {
      document.body.appendChild(chatStanceDebugModalRoot);
    }
  }

  function openChatStanceDebugModal() {
    if (!chatStanceDebugModalRoot) return;
    closeAutonomyDebugModal();
    closeMemoryDebugModal();
    ensureChatStanceModalRootOnBody();
    chatStanceDebugModalRoot.style.position = 'fixed';
    chatStanceDebugModalRoot.style.inset = '0';
    chatStanceDebugModalRoot.style.zIndex = '2147483646';
    if (chatStanceDebugModalBackdrop) {
      chatStanceDebugModalBackdrop.style.position = 'fixed';
      chatStanceDebugModalBackdrop.style.inset = '0';
      chatStanceDebugModalBackdrop.style.zIndex = '2147483646';
    }
    if (chatStanceDebugModalDialog) {
      chatStanceDebugModalDialog.style.position = 'fixed';
      chatStanceDebugModalDialog.style.zIndex = '2147483647';
    }
    updateChatStanceDebugPanel(lastChatStanceDebug);
    chatStanceDebugModalRoot.classList.remove('hidden');
    chatStanceDebugModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
  }

  function closeChatStanceDebugModal() {
    if (!chatStanceDebugModalRoot) return;
    chatStanceDebugModalRoot.classList.add('hidden');
    chatStanceDebugModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
  }

  async function substrateReviewFetch(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload && payload.detail;
      const message = (typeof detail === 'string')
        ? detail
        : (detail && typeof detail.message === 'string')
          ? detail.message
          : payload.error || `HTTP ${response.status}`;
      throw new Error(String(message));
    }
    return payload;
  }

  function clearSubstrateReviewDebugPanel() {
    if (substrateReviewDebugBody) substrateReviewDebugBody.classList.add('hidden');
    if (substrateReviewDebugCaret) substrateReviewDebugCaret.textContent = '▾';
    if (substrateReviewDebugMeta) substrateReviewDebugMeta.textContent = 'No substrate review runtime status loaded.';
    if (substrateReviewDebugOverview) substrateReviewDebugOverview.textContent = 'No substrate review runtime status loaded.';
    if (substrateReviewModalQuickStatus) substrateReviewModalQuickStatus.innerHTML = '';
    if (substrateReviewResultSummary) substrateReviewResultSummary.textContent = 'No action run yet.';
    if (substrateReviewResultRaw) substrateReviewResultRaw.textContent = '--';
    if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Idle.';
    lastSubstrateReviewStatus = null;
    lastSubstrateReviewAction = null;
  }

  function renderSubstrateQuickStatus(statusPayload) {
    if (!substrateReviewModalQuickStatus) return;
    const summary = statusPayload && statusPayload.summary ? statusPayload.summary : {};
    const source = statusPayload && statusPayload.source ? statusPayload.source : {};
    const control = source.control_plane || {};
    const semantic = source.semantic || {};
    const rows = [
      ['Queue count', summary.queue_count ?? '--'],
      ['Due count', summary.due_count ?? '--'],
      ['Next due', summary.next_due_at || '--'],
      ['Last outcome', (summary.last_execution && summary.last_execution.execution_outcome) || '--'],
      ['Telemetry count', summary.telemetry_count ?? '--'],
      ['Policy profile', summary.policy_active_profile_id || '--'],
      ['Control posture', `${control.posture || '--'} (${control.kind || '--'})`],
      ['Semantic posture', `${semantic.posture || '--'} (${semantic.kind || '--'})`],
    ];
    substrateReviewModalQuickStatus.innerHTML = '';
    rows.forEach(([label, value]) => {
      const card = document.createElement('div');
      card.className = 'rounded-xl border border-gray-800 bg-gray-900/60 px-3 py-2';
      const k = document.createElement('div');
      k.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      k.textContent = label;
      const v = document.createElement('div');
      v.className = 'mt-1 text-[11px] text-gray-100 break-words';
      v.textContent = String(value ?? '--');
      card.appendChild(k);
      card.appendChild(v);
      substrateReviewModalQuickStatus.appendChild(card);
    });
  }

  function updateSubstrateReviewDebugPanel(statusPayload) {
    if (!substrateReviewDebugPanel || !substrateReviewDebugMeta || !substrateReviewDebugOverview) return;
    const summary = statusPayload && statusPayload.summary ? statusPayload.summary : {};
    const source = statusPayload && statusPayload.source ? statusPayload.source : {};
    const control = source.control_plane || {};
    const semantic = source.semantic || {};
    lastSubstrateReviewStatus = statusPayload || null;
    substrateReviewDebugMeta.textContent = `queue ${summary.queue_count ?? '--'} · due ${summary.due_count ?? '--'} · control ${control.posture || '--'} · semantic ${semantic.posture || '--'}`;
    substrateReviewDebugOverview.innerHTML = '';
    [
      `last outcome: ${(summary.last_execution && summary.last_execution.execution_outcome) || '--'}`,
      `next eligible: ${(summary.next_item && summary.next_item.queue_item_id) || '--'}`,
      `policy profile: ${summary.policy_active_profile_id || '--'}`,
      `telemetry count: ${summary.telemetry_count ?? '--'}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      substrateReviewDebugOverview.appendChild(row);
    });
    if (substrateReviewModalMeta) substrateReviewModalMeta.textContent = substrateReviewDebugMeta.textContent;
    renderSubstrateQuickStatus(statusPayload);
  }

  function toggleSubstrateReviewDebugPanel() {
    if (!substrateReviewDebugBody) return;
    const nextHidden = !substrateReviewDebugBody.classList.contains('hidden');
    substrateReviewDebugBody.classList.toggle('hidden', nextHidden);
    if (substrateReviewDebugCaret) substrateReviewDebugCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function clearSelfExperimentsDebugPanel() {
    if (selfExperimentsDebugBody) selfExperimentsDebugBody.classList.add('hidden');
    if (selfExperimentsDebugCaret) selfExperimentsDebugCaret.textContent = '▾';
    if (selfExperimentsDebugMeta) selfExperimentsDebugMeta.textContent = 'No self-experiments status loaded.';
    if (selfExperimentsDebugOverview) selfExperimentsDebugOverview.textContent = 'No self-experiments status loaded.';
    if (selfExperimentsDebugRaw) selfExperimentsDebugRaw.textContent = '--';
  }

  function toggleSelfExperimentsDebugPanel() {
    if (!selfExperimentsDebugBody) return;
    const nextHidden = !selfExperimentsDebugBody.classList.contains('hidden');
    selfExperimentsDebugBody.classList.toggle('hidden', nextHidden);
    if (selfExperimentsDebugCaret) selfExperimentsDebugCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function updateSelfExperimentsDebugPanel(payload) {
    if (!selfExperimentsDebugPanel || !selfExperimentsDebugMeta || !selfExperimentsDebugOverview) return;
    const summary = payload && payload.summary ? payload.summary : {};
    const statusCounts = summary.status_counts || {};
    const sourceCounts = summary.source_counts || {};
    selfExperimentsDebugMeta.textContent = `total ${summary.total ?? 0} · validated ${statusCounts.validated ?? 0} · rejected ${statusCounts.rejected ?? 0}`;
    selfExperimentsDebugOverview.innerHTML = '';
    [
      `status counts: ${JSON.stringify(statusCounts)}`,
      `source counts: ${JSON.stringify(sourceCounts)}`,
      `latest generated_at: ${(payload && payload.generated_at) || '--'}`,
      `latest item: ${((payload && payload.items && payload.items[0] && payload.items[0].skill_id) || '--')} · status ${((payload && payload.items && payload.items[0] && payload.items[0].status) || '--')}`,
      `latest hint: ${((payload && payload.items && payload.items[0] && payload.items[0].fix_hint) || '--')}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.textContent = line;
      selfExperimentsDebugOverview.appendChild(row);
    });
    if (selfExperimentsDebugRaw) {
      selfExperimentsDebugRaw.textContent = JSON.stringify(payload || {}, null, 2);
    }
    if (selfExperimentsModalMeta) {
      selfExperimentsModalMeta.textContent = selfExperimentsDebugMeta ? selfExperimentsDebugMeta.textContent : 'Self experiments runtime';
    }
    if (selfExperimentsModalSummary) {
      selfExperimentsModalSummary.innerHTML = '';
      [
        `total=${summary.total ?? 0}`,
        `validated=${statusCounts.validated ?? 0}`,
        `rejected=${statusCounts.rejected ?? 0}`,
        `sources=${JSON.stringify(sourceCounts)}`,
      ].forEach((line) => {
        const row = document.createElement('div');
        row.className = 'autonomy-readiness-row';
        row.textContent = line;
        selfExperimentsModalSummary.appendChild(row);
      });
    }
    if (selfExperimentsModalRuns) {
      const rows = Array.isArray(payload && payload.items) ? payload.items : [];
      selfExperimentsModalRuns.innerHTML = '';
      if (!rows.length) {
        selfExperimentsModalRuns.textContent = 'No runs for current filters.';
      } else {
        rows.slice(0, 12).forEach((item) => {
          const block = document.createElement('div');
          block.className = 'rounded-xl border border-gray-700 bg-gray-900/50 p-3 space-y-1';
          const provenance = item && item.provenance ? item.provenance : {};
          const header = document.createElement('div');
          header.className = 'text-[11px] text-gray-100';
          header.textContent = `${item.status || '--'} · ${item.skill_id || '--'} · ${provenance.source || '--'}`;
          const line1 = document.createElement('div');
          line1.className = 'text-[10px] text-gray-400';
          line1.textContent = `date=${provenance.date || '--'} corr=${provenance.correlation_id || '--'} id=${item.experiment_id || '--'}`;
          const line2 = document.createElement('div');
          line2.className = 'text-[10px] text-amber-200';
          line2.textContent = `reason=${item.reason || 'none'} · fix=${item.fix_hint || '--'}`;
          block.appendChild(header);
          block.appendChild(line1);
          block.appendChild(line2);
          selfExperimentsModalRuns.appendChild(block);
        });
      }
    }
    if (selfExperimentsModalProvenanceTable) {
      const rows = Array.isArray(payload && payload.items) ? payload.items : [];
      selfExperimentsModalProvenanceTable.innerHTML = '';
      if (!rows.length) {
        selfExperimentsModalProvenanceTable.innerHTML = '<tr><td colspan="9" class="px-2 py-3 text-gray-500">No provenance rows for current filters.</td></tr>';
      } else {
        rows.slice(0, 30).forEach((item) => {
          const provenance = item && item.provenance ? item.provenance : {};
          const tr = document.createElement('tr');
          tr.className = 'border-t border-gray-800 align-top';
          const cells = [
            item.created_at_utc || '--',
            item.status || '--',
            provenance.source || '--',
            item.skill_id || '--',
            provenance.correlation_id || '--',
            provenance.date || '--',
            provenance.node || '--',
            item.reason || 'none',
            item.fix_hint || '--',
          ];
          cells.forEach((value) => {
            const td = document.createElement('td');
            td.className = 'px-2 py-2 whitespace-nowrap';
            td.textContent = String(value);
            tr.appendChild(td);
          });
          selfExperimentsModalProvenanceTable.appendChild(tr);
        });
      }
    }
    if (selfExperimentsModalRaw) {
      selfExperimentsModalRaw.textContent = JSON.stringify(payload || {}, null, 2);
    }
  }

  async function refreshSelfExperimentsDebugStatus() {
    const params = new URLSearchParams();
    if (selfExperimentsFilterCorrelation && selfExperimentsFilterCorrelation.value.trim()) {
      params.set('correlation_id', selfExperimentsFilterCorrelation.value.trim());
    }
    if (selfExperimentsFilterDate && selfExperimentsFilterDate.value.trim()) {
      params.set('date', selfExperimentsFilterDate.value.trim());
    }
    if (selfExperimentsFilterSkill && selfExperimentsFilterSkill.value.trim()) {
      params.set('skill_id', selfExperimentsFilterSkill.value.trim());
    }
    const query = params.toString() ? `?${params.toString()}` : '';
    const payload = await substrateReviewFetch(`/api/debug/self-experiments/status${query}`);
    updateSelfExperimentsDebugPanel(payload || {});
    return payload;
  }

  async function runAutonomyGoalArchive(dryRun) {
    const label = dryRun ? 'dry-run' : 'apply';
    if (autonomyGoalArchiveStatus) autonomyGoalArchiveStatus.textContent = `Running goal archive (${label})...`;
    try {
      const payload = await substrateReviewFetch('/api/debug/autonomy/goal-archive', {
        method: 'POST',
        body: JSON.stringify({ dry_run: dryRun }),
      });
      const summaries = Array.isArray(payload && payload.summaries) ? payload.summaries : [];
      const lines = summaries.map((row) => {
        const subject = row.subject || '--';
        if (row.error) return `${subject}: error=${row.error}`;
        return `${subject}: candidates=${row.candidates ?? 0} applied=${row.applied ?? 0} scanned=${row.rows_scanned ?? 0}`;
      });
      if (autonomyGoalArchiveStatus) {
        autonomyGoalArchiveStatus.textContent = lines.length
          ? `${label} ok=${payload.ok ? 'yes' : 'no'} · ${lines.join(' · ')}`
          : `${label} finished (no summaries)`;
      }
      return payload;
    } catch (err) {
      if (autonomyGoalArchiveStatus) {
        autonomyGoalArchiveStatus.textContent = `Goal archive ${label} failed: ${err && err.message ? err.message : err}`;
      }
      throw err;
    }
  }

  async function triggerSelfExperimentsDaily(action) {
    if (selfExperimentsActionStatus) selfExperimentsActionStatus.textContent = `Triggering ${action}...`;
    const dateValue = selfExperimentsFilterDate && selfExperimentsFilterDate.value.trim()
      ? selfExperimentsFilterDate.value.trim()
      : '';
    const payload = await substrateReviewFetch('/api/debug/self-experiments/trigger-daily', {
      method: 'POST',
      body: JSON.stringify({ action, date: dateValue || null }),
    });
    if (selfExperimentsActionStatus) {
      selfExperimentsActionStatus.textContent = `Triggered ${action} · correlation_id=${payload.correlation_id || '--'}`;
    }
    if (selfExperimentsFilterCorrelation && payload.correlation_id) {
      selfExperimentsFilterCorrelation.value = String(payload.correlation_id);
    }
    await refreshSelfExperimentsDebugStatus();
    return payload;
  }

  function ensureSelfExperimentsModalRootOnBody() {
    if (!selfExperimentsModalRoot || !document.body) return;
    if (selfExperimentsModalRoot.parentElement !== document.body) {
      document.body.appendChild(selfExperimentsModalRoot);
    }
  }

  function openSelfExperimentsModal() {
    if (!selfExperimentsModalRoot) return;
    closeAutonomyDebugModal();
    closeMemoryDebugModal();
    closeSubstrateReviewModal();
    ensureSelfExperimentsModalRootOnBody();
    selfExperimentsModalRoot.style.position = 'fixed';
    selfExperimentsModalRoot.style.inset = '0';
    selfExperimentsModalRoot.style.zIndex = '2147483646';
    if (selfExperimentsModalBackdrop) {
      selfExperimentsModalBackdrop.style.position = 'fixed';
      selfExperimentsModalBackdrop.style.inset = '0';
      selfExperimentsModalBackdrop.style.zIndex = '2147483646';
    }
    if (selfExperimentsModalDialog) {
      selfExperimentsModalDialog.style.position = 'fixed';
      selfExperimentsModalDialog.style.zIndex = '2147483647';
    }
    selfExperimentsModalRoot.classList.remove('hidden');
    selfExperimentsModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
  }

  function closeSelfExperimentsModal() {
    if (!selfExperimentsModalRoot) return;
    selfExperimentsModalRoot.classList.add('hidden');
    selfExperimentsModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
  }

  function ensureSubstrateReviewModalRootOnBody() {
    if (!substrateReviewModalRoot || !document.body) return;
    if (substrateReviewModalRoot.parentElement !== document.body) {
      document.body.appendChild(substrateReviewModalRoot);
    }
  }

  function openSubstrateReviewModal() {
    if (!substrateReviewModalRoot) return;
    closeAutonomyDebugModal();
    closeMemoryDebugModal();
    ensureSubstrateReviewModalRootOnBody();
    substrateReviewModalRoot.style.position = 'fixed';
    substrateReviewModalRoot.style.inset = '0';
    substrateReviewModalRoot.style.zIndex = '2147483646';
    if (substrateReviewModalBackdrop) {
      substrateReviewModalBackdrop.style.position = 'fixed';
      substrateReviewModalBackdrop.style.inset = '0';
      substrateReviewModalBackdrop.style.zIndex = '2147483646';
    }
    if (substrateReviewModalDialog) {
      substrateReviewModalDialog.style.position = 'fixed';
      substrateReviewModalDialog.style.zIndex = '2147483647';
    }
    substrateReviewModalRoot.classList.remove('hidden');
    substrateReviewModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
  }

  function closeSubstrateReviewModal() {
    if (!substrateReviewModalRoot) return;
    substrateReviewModalRoot.classList.add('hidden');
    substrateReviewModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
  }

  function renderSubstrateActionResult(resultPayload) {
    if (substrateReviewResultRaw) substrateReviewResultRaw.textContent = JSON.stringify(resultPayload || {}, null, 2);
    if (!substrateReviewResultSummary) return;
    const result = resultPayload && resultPayload.result ? resultPayload.result : {};
    const source = resultPayload && resultPayload.source ? resultPayload.source : {};
    const control = source.control_plane || {};
    const semantic = source.semantic || {};
    substrateReviewResultSummary.innerHTML = '';
    [
      `outcome: ${result.outcome || '--'}`,
      `selected queue item: ${result.selected_queue_item_id || '--'}`,
      `frontier follow-up invoked: ${result.frontier_followup_invoked ? 'yes' : 'no'}`,
      `control posture: ${control.posture || '--'} (${control.kind || '--'})`,
      `semantic posture: ${semantic.posture || '--'} (${semantic.kind || '--'})`,
      `executed at: ${result.executed_at || '--'}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.className = 'rounded-lg border border-gray-800 bg-gray-900/60 px-3 py-2';
      row.textContent = line;
      substrateReviewResultSummary.appendChild(row);
    });
  }

  async function refreshSubstrateReviewStatus() {
    const payload = await substrateReviewFetch('/api/substrate/review-runtime/status');
    updateSubstrateReviewDebugPanel(payload);
    return payload;
  }

  function clearAutonomyReadinessPanel() {
    if (autonomyReadinessBody) autonomyReadinessBody.classList.add('hidden');
    if (autonomyReadinessCaret) autonomyReadinessCaret.textContent = '▾';
    if (autonomyReadinessMeta) autonomyReadinessMeta.textContent = 'No autonomy readiness snapshot loaded.';
    if (autonomyReadinessOverview) autonomyReadinessOverview.textContent = 'No data yet.';
    if (autonomyReadinessWarnings) autonomyReadinessWarnings.textContent = 'No warnings.';
  }

  function toggleAutonomyReadinessPanel() {
    if (!autonomyReadinessBody) return;
    const nextHidden = !autonomyReadinessBody.classList.contains('hidden');
    autonomyReadinessBody.classList.toggle('hidden', nextHidden);
    if (autonomyReadinessCaret) autonomyReadinessCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function updateAutonomyReadinessPanel(snapshot) {
    if (!autonomyReadinessPanel || !autonomyReadinessMeta || !autonomyReadinessOverview) return;
    lastAutonomyReadinessSnapshot = snapshot || null;
    const overall = snapshot && snapshot.overall ? snapshot.overall : {};
    const scheduler = snapshot && snapshot.scheduler ? snapshot.scheduler : {};
    const surfaces = snapshot && snapshot.surfaces ? snapshot.surfaces : {};
    const recall = snapshot && snapshot.recall ? snapshot.recall : {};
    const cognitive = snapshot && snapshot.cognitive ? snapshot.cognitive : {};
    const pressure = snapshot && snapshot.pressure ? snapshot.pressure : {};
    const recent = snapshot && snapshot.recent_activity ? snapshot.recent_activity : {};
    const warnings = Array.isArray(snapshot && snapshot.warnings) ? snapshot.warnings : [];
    const liveCount = Array.isArray(surfaces.live) ? surfaces.live.length : 0;
    const shadowCount = Array.isArray(surfaces.shadow) ? surfaces.shadow.length : 0;
    const proposalOnlyCount = Array.isArray(surfaces.proposal_only) ? surfaces.proposal_only.length : 0;
    const blockedCount = Array.isArray(surfaces.blocked) ? surfaces.blocked.length : 0;
    const recallReadiness = (recall.readiness && recall.readiness.recommendation) || 'unavailable';
    const manualCanary = recall.manual_canary || {};
    const pressureTop = Array.isArray(pressure.top_pressure_keys) && pressure.top_pressure_keys.length
      ? pressure.top_pressure_keys.slice(0, 2).map((row) => `${row.key || '--'}:${row.count ?? '--'}`).join(', ')
      : 'No data yet';
    const recentApplies = Array.isArray(recent.applies) ? recent.applies.length : 0;
    const recentRollbacks = Array.isArray(recent.rollbacks) ? recent.rollbacks.length : 0;
    autonomyReadinessMeta.textContent = `schema ${(snapshot && snapshot.schema_version) || '--'} · generated ${(snapshot && snapshot.generated_at) || '--'}`;
    autonomyReadinessOverview.innerHTML = '';
    [
      `overall: ${overall.summary || 'Unavailable'}`,
      `safe next: ${overall.safe_next_action || 'Unavailable'}`,
      `scheduler: enabled=${scheduler.enabled ? 'yes' : 'no'} proposals=${scheduler.proposal_enabled ? 'yes' : 'no'} apply=${scheduler.apply_enabled ? 'yes' : 'no'}`,
      `surfaces: live=${liveCount} shadow=${shadowCount} proposal-only=${proposalOnlyCount} blocked=${blockedCount}`,
      `recall: production=${recall.production_mode || 'v1'} live_apply=${recall.live_apply_enabled ? 'true' : 'false'} readiness=${recallReadiness}`,
      `recall manual canary: runs=${manualCanary.run_count ?? 0} review_artifacts=${manualCanary.review_artifact_count ?? 0} recommended=${manualCanary.recommended_canary_action || '--'}`,
      `cognitive: live_apply=${cognitive.live_apply_enabled ? 'true' : 'false'} proposal_states=${JSON.stringify(cognitive.counts_by_state || {})}`,
      `pressure: ${pressureTop}`,
      `activity: applies=${recentApplies} rollbacks=${recentRollbacks}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.className = 'autonomy-readiness-row';
      row.textContent = line;
      autonomyReadinessOverview.appendChild(row);
    });
    if (autonomyReadinessWarnings) {
      autonomyReadinessWarnings.textContent = warnings.length ? warnings.join(' | ') : 'No warnings.';
    }
  }

  async function refreshAutonomyReadinessPanel() {
    const payload = await substrateReviewFetch('/api/substrate/autonomy-readiness');
    updateAutonomyReadinessPanel(payload || {});
    return payload;
  }

  function selectedRecallCanaryFailureModes() {
    return Array.from(document.querySelectorAll('.recall-canary-failure-mode:checked')).map((el) => el.value);
  }

  function setRecallCanaryActionStatus(message) {
    const text = String(message || '').trim();
    if (recallCanaryModalActionStatus) {
      recallCanaryModalActionStatus.textContent = text || 'Ready.';
    }
    if (recallCanaryStatusMeta && text) {
      recallCanaryStatusMeta.textContent = text;
    }
  }

  function toggleRecallCanaryPanel() {
    if (!recallCanaryBody) return;
    const nextHidden = !recallCanaryBody.classList.contains('hidden');
    recallCanaryBody.classList.toggle('hidden', nextHidden);
    if (recallCanaryCaret) recallCanaryCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function toggleMessagesPanel() {
    if (!messagesBody) return;
    const nextHidden = !messagesBody.classList.contains('hidden');
    messagesBody.classList.toggle('hidden', nextHidden);
    if (messagesCaret) messagesCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function toggleWorldPulsePanel() {
    if (!worldPulseBody) return;
    const nextHidden = !worldPulseBody.classList.contains('hidden');
    worldPulseBody.classList.toggle('hidden', nextHidden);
    if (worldPulseCaret) worldPulseCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function toRecallCanaryError(err) {
    const raw = String((err && err.message) || err || 'Unknown recall canary error');
    const lower = raw.toLowerCase();
    if (lower.includes('mutation_operator_token_not_configured')) return 'Hub operator token guard is not configured on backend.';
    if (lower.includes('operator_guard_rejected') || lower.includes('forbidden')) return 'Operator token is invalid or not authorized.';
    if (lower.includes('invalid_recall_canary_profile_id')) return 'Selected recall profile is invalid. Refresh profiles and retry.';
    if (lower.includes('failed to fetch') || lower.includes('networkerror') || lower.includes('503')) return 'Recall canary backend is unavailable.';
    if (lower.includes('operator token is required')) return 'Operator token is required.';
    return raw;
  }

  function renderRecallCanaryLatestResult(payload) {
    const data = payload && payload.data ? payload.data : {};
    const selectedProfile = data.selected_profile || {};
    if (recallCanaryLatestResult) {
      const lines = [
        `canary_run_id: ${data.canary_run_id || '--'}`,
        `selected_profile: ${selectedProfile.profile_id || '--'} (${selectedProfile.label || selectedProfile.profile_id || '--'} · ${selectedProfile.status || '--'})`,
        `production_recall_mode: ${data.production_recall_mode || 'v1'}`,
        `recall_live_apply_enabled: ${String(data.recall_live_apply_enabled === true)}`,
        `v1_summary: ${JSON.stringify(data.v1_summary || {})}`,
        `v2_summary: ${JSON.stringify(data.v2_summary || {})}`,
        `comparison: ${JSON.stringify(data.comparison || data.comparison_summary || {})}`,
      ];
      recallCanaryLatestResult.innerHTML = '';
      lines.forEach((line) => {
        const row = document.createElement('div');
        row.className = 'autonomy-readiness-row';
        row.textContent = line;
        recallCanaryLatestResult.appendChild(row);
      });
    }
    if (recallCanaryRawResponse) {
      recallCanaryRawResponse.textContent = JSON.stringify(payload || {}, null, 2);
    }
  }

  function hydrateRecallCanaryProfileSelect(data = {}) {
    if (!recallCanaryProfileSelect) return;
    const profiles = Array.isArray(data.available_profiles) ? data.available_profiles : [];
    const defaultProfileId = data.default_canary_profile_id ? String(data.default_canary_profile_id) : '';
    const storedProfileId = localStorage.getItem(RECALL_CANARY_PROFILE_STORAGE_KEY) || '';
    recallCanaryProfileSelect.innerHTML = '';
    profiles.forEach((profile) => {
      const option = document.createElement('option');
      const profileId = String(profile.profile_id || '');
      const label = String(profile.label || profileId || 'Unnamed profile');
      const status = String(profile.status || 'shadow_canary_review_only');
      option.value = profileId;
      option.textContent = `${label} — ${status}`;
      recallCanaryProfileSelect.appendChild(option);
    });
    let selectedValue = '';
    if (storedProfileId && profiles.some((profile) => String(profile.profile_id) === storedProfileId)) {
      selectedValue = storedProfileId;
    } else if (defaultProfileId && profiles.some((profile) => String(profile.profile_id) === defaultProfileId)) {
      selectedValue = defaultProfileId;
    } else if (profiles.length > 0) {
      selectedValue = String(profiles[0].profile_id || '');
    }
    if (selectedValue) {
      recallCanaryProfileSelect.value = selectedValue;
      localStorage.setItem(RECALL_CANARY_PROFILE_STORAGE_KEY, selectedValue);
      lastRecallCanarySelectedProfile = profiles.find((profile) => String(profile.profile_id) === selectedValue) || null;
    } else {
      lastRecallCanarySelectedProfile = null;
    }
    if (recallCanaryProfileEmptyState) {
      recallCanaryProfileEmptyState.classList.toggle('hidden', profiles.length > 0);
    }
    if (recallCanaryRunButton) {
      const disabled = !(profiles.length > 0 && !!selectedValue);
      recallCanaryRunButton.disabled = disabled;
      recallCanaryRunButton.classList.toggle('opacity-50', disabled);
      recallCanaryRunButton.classList.toggle('cursor-not-allowed', disabled);
    }
    if (recallCanarySafetyBadges) {
      recallCanarySafetyBadges.textContent = 'Production recall remains V1 · Selected profile is canary/shadow only · No production promotion';
    }
    if (recallCanaryModalActionStatus && profiles.length === 0) {
      recallCanaryModalActionStatus.textContent = 'No recall profiles available for canary testing.';
    }
  }

  function renderRecallCanaryStatus(payload) {
    if (!recallCanaryPanel || !recallCanarySummary || !recallCanaryStatusMeta) return;
    const data = payload && payload.data ? payload.data : {};
    const judgmentCounts = data.judgment_counts || {};
    const recommended = ((lastAutonomyReadinessSnapshot && lastAutonomyReadinessSnapshot.recall && lastAutonomyReadinessSnapshot.recall.manual_canary && lastAutonomyReadinessSnapshot.recall.manual_canary.recommended_canary_action) || '--');
    recallCanaryStatusMeta.textContent = `runs=${data.run_count ?? 0} · review_artifacts=${data.review_artifact_count ?? 0} · recommended=${recommended} · production_recall_mode=${data.production_recall_mode || 'v1'} · recall_live_apply_enabled=${String(data.recall_live_apply_enabled === true)}`;
    hydrateRecallCanaryProfileSelect(data);
    recallCanarySummary.innerHTML = '';
    [
      `judgments: v2_better=${judgmentCounts.v2_better ?? 0}, v1_better=${judgmentCounts.v1_better ?? 0}, tie=${judgmentCounts.tie ?? 0}, both_bad=${judgmentCounts.both_bad ?? 0}, inconclusive=${judgmentCounts.inconclusive ?? 0}`,
      `failure modes: ${JSON.stringify(data.failure_mode_counts || {})}`,
      `last review artifact: ${data.last_review_artifact_at || '--'}`,
      `selected profile: ${lastRecallCanarySelectedProfile ? `${lastRecallCanarySelectedProfile.label || lastRecallCanarySelectedProfile.profile_id} (${lastRecallCanarySelectedProfile.status || 'shadow_canary_review_only'})` : '--'}`,
    ].forEach((line) => {
      const row = document.createElement('div');
      row.className = 'autonomy-readiness-row';
      row.textContent = line;
      recallCanarySummary.appendChild(row);
    });
    if (recallCanaryModalMeta) {
      recallCanaryModalMeta.textContent = 'Manual canary only. No production promotion.';
    }
    if (recallCanaryModalStatusMeta) {
      recallCanaryModalStatusMeta.textContent = recallCanaryStatusMeta.textContent;
    }
    if (recallCanaryModalSummary) {
      recallCanaryModalSummary.innerHTML = recallCanarySummary.innerHTML;
    }
    if (!lastRecallCanaryResponse && recallCanaryRawResponse) {
      recallCanaryRawResponse.textContent = 'No response yet.';
    }
  }

  async function refreshRecallCanaryStatus() {
    const payload = await substrateReviewFetch('/api/substrate/recall-canary/status');
    renderRecallCanaryStatus(payload);
    return payload;
  }

  function ensureRecallCanaryModalRootOnBody() {
    if (!recallCanaryModalRoot || !document.body) return;
    if (recallCanaryModalRoot.parentElement !== document.body) {
      document.body.appendChild(recallCanaryModalRoot);
    }
  }

  async function refreshRecallCanaryModal() {
    const payload = await refreshRecallCanaryStatus();
    return payload;
  }

  function openRecallCanaryModal() {
    if (!recallCanaryModalRoot) return;
    closeMemoryDebugModal();
    closeAutonomyDebugModal();
    closeChatStanceDebugModal();
    closeSubstrateReviewModal();
    closeCognitiveReviewModal();
    closeAutonomyConstitutionModal();
    ensureRecallCanaryModalRootOnBody();
    recallCanaryModalRoot.style.position = 'fixed';
    recallCanaryModalRoot.style.inset = '0';
    recallCanaryModalRoot.style.zIndex = '2147483646';
    if (recallCanaryModalBackdrop) {
      recallCanaryModalBackdrop.style.position = 'fixed';
      recallCanaryModalBackdrop.style.inset = '0';
      recallCanaryModalBackdrop.style.zIndex = '2147483646';
    }
    if (recallCanaryModalDialog) {
      recallCanaryModalDialog.style.position = 'fixed';
      recallCanaryModalDialog.style.zIndex = '2147483647';
    }
    refreshRecallCanaryModal().catch((err) => {
      if (recallCanaryModalStatusMeta) recallCanaryModalStatusMeta.textContent = `Recall canary unavailable: ${toRecallCanaryError(err)}`;
    });
    recallCanaryModalRoot.classList.remove('hidden');
    recallCanaryModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
  }

  function closeRecallCanaryModal() {
    if (!recallCanaryModalRoot) return;
    recallCanaryModalRoot.classList.add('hidden');
    recallCanaryModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
  }

  async function refreshCognitiveReviewPanelInto({
    statusMetaEl,
    statusSummaryEl,
    draftListEl,
    stanceNoteListEl,
  }) {
    if (!statusMetaEl || !statusSummaryEl) return;
    const status = await substrateReviewFetch('/api/substrate/cognitive-proposals/status');
    const proposals = await substrateReviewFetch('/api/substrate/cognitive-proposals?limit=8');
    const statusData = (status && status.data) || {};
    const posture = statusData.review_posture || {};
    const proposalRows = ((proposals && proposals.data && proposals.data.recent_cognitive_proposals) || []).slice(0, 8);
    statusMetaEl.textContent = `live_apply_enabled=${String(statusData.live_apply_enabled === true)} · recommended=${String(posture.recommended_action || 'monitor')}`;
    statusSummaryEl.innerHTML = '';
    const lines = [
      `pending_review=${Number(posture.pending_review_count || 0)}`,
      `active_drafts=${Number(posture.active_draft_count || 0)}`,
      `active_stance_notes=${Number(posture.active_stance_note_count || 0)}`,
      proposalRows.length ? `proposals=${proposalRows.map((row) => row.proposal_id).join(', ')}` : 'No cognitive proposals yet',
    ];
    lines.forEach((line) => {
      const row = document.createElement('div');
      row.className = 'autonomy-readiness-row';
      row.textContent = line;
      statusSummaryEl.appendChild(row);
    });
    if (draftListEl) {
      const drafts = await substrateReviewFetch('/api/substrate/cognitive-drafts?limit=8');
      const rows = ((drafts && drafts.data && drafts.data.drafts) || []).slice(0, 8);
      draftListEl.textContent = rows.length
        ? rows.map((row) => `${row.draft_id} · ${row.state} · ${row.proposal_class}`).join('\n')
        : 'No accepted drafts yet.';
    }
    if (stanceNoteListEl) {
      const notes = await substrateReviewFetch('/api/substrate/cognitive-stance-notes?limit=8');
      const rows = ((notes && notes.data && notes.data.stance_notes) || []).slice(0, 8);
      stanceNoteListEl.textContent = rows.length
        ? rows.map((row) => `${row.stance_note_id} · ${row.status} · ${row.visibility}`).join('\n')
        : 'No active stance notes yet.';
    }
  }

  async function refreshCognitiveReviewModal() {
    await refreshCognitiveReviewPanelInto({
      statusMetaEl: cognitiveReviewModalStatusMeta,
      statusSummaryEl: cognitiveReviewModalStatusSummary,
      draftListEl: cognitiveReviewModalDraftList,
      stanceNoteListEl: cognitiveReviewModalStanceNoteList,
    });
    console.log('event=cognitive_review_modal_data_requested source=ui');
  }

  async function submitCognitiveProposalReview(decision, source = 'modal') {
    const proposalId = cognitiveReviewModalProposalIdInput && cognitiveReviewModalProposalIdInput.value
      ? cognitiveReviewModalProposalIdInput.value.trim()
      : '';
    if (!proposalId) throw new Error('proposal_id is required');
    await substrateReviewFetch(`/api/substrate/cognitive-proposals/${encodeURIComponent(proposalId)}/review`, {
      method: 'POST',
      body: JSON.stringify({
        decision,
        rationale: cognitiveReviewModalRationaleInput ? cognitiveReviewModalRationaleInput.value || '' : '',
      }),
    });
    await refreshCognitiveReviewModal();
    await refreshAutonomyReadinessPanel();
  }

  function ensureCognitiveReviewModalRootOnBody() {
    if (!cognitiveReviewModalRoot || !document.body) return;
    if (cognitiveReviewModalRoot.parentElement !== document.body) {
      document.body.appendChild(cognitiveReviewModalRoot);
    }
  }

  function openCognitiveReviewModal() {
    if (!cognitiveReviewModalRoot) return;
    closeMemoryDebugModal();
    closeAutonomyDebugModal();
    closeChatStanceDebugModal();
    closeSubstrateReviewModal();
    ensureCognitiveReviewModalRootOnBody();
    cognitiveReviewModalRoot.style.position = 'fixed';
    cognitiveReviewModalRoot.style.inset = '0';
    cognitiveReviewModalRoot.style.zIndex = '2147483646';
    if (cognitiveReviewModalBackdrop) {
      cognitiveReviewModalBackdrop.style.position = 'fixed';
      cognitiveReviewModalBackdrop.style.inset = '0';
      cognitiveReviewModalBackdrop.style.zIndex = '2147483646';
    }
    if (cognitiveReviewModalDialog) {
      cognitiveReviewModalDialog.style.position = 'fixed';
      cognitiveReviewModalDialog.style.zIndex = '2147483647';
    }
    if (cognitiveReviewModalMeta) {
      cognitiveReviewModalMeta.textContent = 'Review/draft/context only. No live cognitive apply. No identity/policy/prompt rewrite.';
    }
    refreshCognitiveReviewModal().catch((err) => {
      if (cognitiveReviewModalStatusMeta) cognitiveReviewModalStatusMeta.textContent = `Cognitive review unavailable: ${String(err.message || err)}`;
    });
    cognitiveReviewModalRoot.classList.remove('hidden');
    cognitiveReviewModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
  }

  function closeCognitiveReviewModal() {
    if (!cognitiveReviewModalRoot) return;
    cognitiveReviewModalRoot.classList.add('hidden');
    cognitiveReviewModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
  }

  function ensureAutonomyConstitutionModalRootOnBody() {
    if (!autonomyConstitutionModalRoot || !document.body) return;
    if (autonomyConstitutionModalRoot.parentElement !== document.body) {
      document.body.appendChild(autonomyConstitutionModalRoot);
    }
  }

  function renderAutonomyConstitution(payload) {
    const data = payload || {};
    const summary = data.summary || {};
    const surfaces = Array.isArray(data.surfaces) ? data.surfaces : [];
    const invariants = Array.isArray(data.safety_invariants) ? data.safety_invariants : [];
    const warnings = Array.isArray(data.warnings) ? data.warnings : [];
    if (autonomyConstitutionModalMeta) {
      autonomyConstitutionModalMeta.textContent = `schema=${data.schema_version || '--'} loaded_at=${data.loaded_at || '--'} source=${data.source || '--'}`;
    }
    if (autonomyConstitutionModalSummary) {
      autonomyConstitutionModalSummary.textContent = [
        `live_apply_surfaces=${JSON.stringify(summary.live_apply_surfaces || [])}`,
        `blocked_surfaces=${JSON.stringify(summary.blocked_surfaces || [])}`,
        `protected_surfaces=${JSON.stringify(summary.protected_surfaces || [])}`,
        `human_required_surfaces=${JSON.stringify(summary.human_required_surfaces || [])}`,
      ].join(' · ');
    }
    if (autonomyConstitutionModalInvariants) {
      autonomyConstitutionModalInvariants.textContent = invariants.length
        ? invariants.map((row) => `- ${String(row)}`).join('\n')
        : 'No policy surfaces loaded.';
    }
    if (autonomyConstitutionModalSurfaces) {
      autonomyConstitutionModalSurfaces.textContent = surfaces.length
        ? surfaces.map((row) => `${row.surface} | category=${row.category} | status=${row.status} | propose=${row.propose} | trial=${row.trial} | apply=${row.apply} | rollback=${row.rollback} | human_required=${String(row.human_required)} | forbidden=${JSON.stringify(row.forbidden || [])}`).join('\n')
        : 'No policy surfaces loaded.';
      if (warnings.length) {
        autonomyConstitutionModalSurfaces.textContent += `\n\nwarnings=${warnings.join(' | ')}`;
      }
    }
  }

  async function refreshAutonomyConstitutionModal() {
    const payload = await substrateReviewFetch('/api/substrate/autonomy-constitution');
    renderAutonomyConstitution(payload || {});
    return payload;
  }

  function openAutonomyConstitutionModal() {
    if (!autonomyConstitutionModalRoot) return;
    closeMemoryDebugModal();
    closeAutonomyDebugModal();
    closeChatStanceDebugModal();
    closeSubstrateReviewModal();
    closeCognitiveReviewModal();
    ensureAutonomyConstitutionModalRootOnBody();
    autonomyConstitutionModalRoot.style.position = 'fixed';
    autonomyConstitutionModalRoot.style.inset = '0';
    autonomyConstitutionModalRoot.style.zIndex = '2147483646';
    if (autonomyConstitutionModalBackdrop) {
      autonomyConstitutionModalBackdrop.style.position = 'fixed';
      autonomyConstitutionModalBackdrop.style.inset = '0';
      autonomyConstitutionModalBackdrop.style.zIndex = '2147483646';
    }
    if (autonomyConstitutionModalDialog) {
      autonomyConstitutionModalDialog.style.position = 'fixed';
      autonomyConstitutionModalDialog.style.zIndex = '2147483647';
    }
    refreshAutonomyConstitutionModal().catch((err) => {
      if (autonomyConstitutionModalMeta) autonomyConstitutionModalMeta.textContent = `Autonomy constitution unavailable: ${String(err.message || err)}`;
      if (autonomyConstitutionModalSummary) autonomyConstitutionModalSummary.textContent = 'Autonomy constitution unavailable';
    });
    autonomyConstitutionModalRoot.classList.remove('hidden');
    autonomyConstitutionModalRoot.setAttribute('aria-hidden', 'false');
    syncDebugModalScrollLock();
  }

  function closeAutonomyConstitutionModal() {
    if (!autonomyConstitutionModalRoot) return;
    autonomyConstitutionModalRoot.classList.add('hidden');
    autonomyConstitutionModalRoot.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
  }

  async function runRecallCanaryQuery() {
    const queryText = (recallCanaryQueryInput && recallCanaryQueryInput.value ? recallCanaryQueryInput.value.trim() : '');
    if (!queryText) throw new Error('Canary query text is required');
    const profileId = (recallCanaryProfileSelect && recallCanaryProfileSelect.value ? recallCanaryProfileSelect.value.trim() : '');
    if (!profileId) throw new Error('Recall profile is required');
    setRecallCanaryActionStatus('Running canary query...');
    const payload = await substrateReviewFetch('/api/substrate/recall-canary/query', {
      method: 'POST',
      body: JSON.stringify({ query_text: queryText, profile_id: profileId }),
    });
    lastRecallCanaryRunId = payload && payload.data ? payload.data.canary_run_id : null;
    lastRecallCanaryResponse = payload || null;
    lastRecallCanarySelectedProfile = payload && payload.data ? (payload.data.selected_profile || null) : null;
    localStorage.setItem(RECALL_CANARY_PROFILE_STORAGE_KEY, profileId);
    renderRecallCanaryLatestResult(payload || {});
    setRecallCanaryActionStatus(`Canary run complete: ${lastRecallCanaryRunId || '--'}`);
    await refreshRecallCanaryStatus();
    return payload;
  }

  async function recordRecallCanaryJudgment() {
    if (!lastRecallCanaryRunId) throw new Error('No canary run available');
    const selectedRecallCanaryJudgment = recallCanaryJudgmentSelect && recallCanaryJudgmentSelect.value
      ? String(recallCanaryJudgmentSelect.value)
      : '';
    if (!selectedRecallCanaryJudgment) throw new Error('Select a judgment');
    const payload = await substrateReviewFetch(`/api/substrate/recall-canary/runs/${encodeURIComponent(lastRecallCanaryRunId)}/judgment`, {
      method: 'POST',
      body: JSON.stringify({
        judgment: selectedRecallCanaryJudgment,
        failure_modes: selectedRecallCanaryFailureModes(),
        operator_note: recallCanaryOperatorNote ? recallCanaryOperatorNote.value : '',
        should_emit_pressure: true,
        should_mark_review_candidate: false,
      }),
    });
    setRecallCanaryActionStatus(`Judgment submitted for run ${lastRecallCanaryRunId}.`);
    await refreshRecallCanaryStatus();
    await refreshAutonomyReadinessPanel();
    return payload;
  }

  async function createRecallCanaryReviewArtifact() {
    if (!lastRecallCanaryRunId) throw new Error('No canary run available');
    const payload = await substrateReviewFetch(`/api/substrate/recall-canary/runs/${encodeURIComponent(lastRecallCanaryRunId)}/create-review-artifact`, {
      method: 'POST',
      body: JSON.stringify({
        review_type: 'production_candidate_evidence',
        include_comparison_summary: true,
        include_operator_judgment: true,
        operator_note: recallCanaryOperatorNote ? recallCanaryOperatorNote.value : '',
      }),
    });
    const artifactId = payload && payload.data ? (payload.data.review_artifact_id || payload.data.artifact_id || '--') : '--';
    setRecallCanaryActionStatus(`Review artifact result: ${artifactId}`);
    await refreshRecallCanaryStatus();
    await refreshAutonomyReadinessPanel();
    return payload;
  }

  async function runSubstrateReviewExecuteOnce() {
    if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Running execute-once…';
    const payload = await substrateReviewFetch('/api/substrate/review-runtime/execute-once', { method: 'POST', body: JSON.stringify({}) });
    lastSubstrateReviewAction = payload;
    renderSubstrateActionResult(payload);
    await refreshSubstrateReviewStatus();
    if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Execute-once complete.';
    return payload;
  }

  async function runSubstrateReviewExecuteOnceWithFollowup() {
    if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Running execute-once with explicit follow-up…';
    const payload = await substrateReviewFetch('/api/substrate/review-runtime/execute-once-followup', { method: 'POST', body: JSON.stringify({}) });
    lastSubstrateReviewAction = payload;
    renderSubstrateActionResult(payload);
    await refreshSubstrateReviewStatus();
    if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Execute-once + follow-up complete.';
    return payload;
  }

  async function runSubstrateReviewSmokeCheck() {
    if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Running smoke check…';
    const payload = await substrateReviewFetch('/api/substrate/review-runtime/smoke-check', { method: 'POST', body: JSON.stringify({}) });
    lastSubstrateReviewAction = payload;
    if (substrateReviewResultRaw) substrateReviewResultRaw.textContent = JSON.stringify(payload || {}, null, 2);
    if (substrateReviewResultSummary) {
      const checks = payload.checks || {};
      substrateReviewResultSummary.innerHTML = [
        `queue available: ${checks.queue_available ? 'yes' : 'no'}`,
        `runtime eligible: ${checks.runtime_eligible ? 'yes' : 'no'}`,
        `semantic available: ${checks.semantic_available ? 'yes' : 'no'}`,
        `control plane available: ${checks.control_plane_available ? 'yes' : 'no'}`,
      ].map((line) => `<div class="rounded-lg border border-gray-800 bg-gray-900/60 px-3 py-2">${line}</div>`).join('');
    }
    await refreshSubstrateReviewStatus();
    if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Smoke check complete.';
    return payload;
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
    const nextHidden = !memoryPanelBody.classList.contains('hidden');
    memoryPanelBody.classList.toggle('hidden', nextHidden);
    if (memoryPanelCaret) memoryPanelCaret.textContent = nextHidden ? '▾' : '▴';
  }

  function toggleRuntimeDebugPanel() {
    if (!runtimeDebugPanelBody) return;
    const nextHidden = !runtimeDebugPanelBody.classList.contains('hidden');
    runtimeDebugPanelBody.classList.toggle('hidden', nextHidden);
    if (runtimeDebugPanelCaret) runtimeDebugPanelCaret.textContent = nextHidden ? '▾' : '▴';
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
    const effectiveVerbs = modeVerbOverride ? [modeVerbOverride] : selectedVerbs;
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
        raw: data.raw,
        workflow: data.workflow,
        correlationId: data.correlation_id,
        substrateEffectSummary: data.substrate_effect_summary || null,
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

  function sortedNotificationsForTray(list) {
    return [...list].sort((a, b) => {
      const ra = Date.parse(a.received_at || a.created_at || 0) || 0;
      const rb = Date.parse(b.received_at || b.created_at || 0) || 0;
      return rb - ra;
    });
  }

  function renderNotifications() {
    if (!notificationList) return;
    const filter = notificationFilter ? notificationFilter.value : 'all';
    notificationList.innerHTML = '';
    const filtered = sortedNotificationsForTray(notifications).filter(
      (n) => filter === 'all' || (n.severity || '').toLowerCase() === filter
    );

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
      meta.title =
        'Notifications reflect server recent events; refresh reloads the Hub cache, not full history.';
      const createdAt = n.created_at ? new Date(n.created_at).toLocaleString() : null;
      const receivedAt = n.received_at ? new Date(n.received_at).toLocaleString() : null;
      const timeParts = [];
      if (createdAt) timeParts.push(`Created ${createdAt}`);
      if (receivedAt) timeParts.push(`Received ${receivedAt}`);
      const timeLine = timeParts.length ? timeParts.join(' · ') : '--';
      meta.textContent = `${timeLine} • ${n.event_kind || 'event'} • ${n.source_service || 'unknown'}`;

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

  function normalizeRecallProfileDisplay() {
    if (!recallProfileSelect) return;
    const options = Array.from(recallProfileSelect.options || []);
    if (!options.some((opt) => (opt.value || '').trim().toLowerCase() === 'recall.v1')) {
      const opt = document.createElement('option');
      opt.value = 'recall.v1';
      opt.textContent = 'recall.v1';
      recallProfileSelect.insertBefore(opt, recallProfileSelect.options[1] || null);
    }
    options.forEach((opt) => {
      if ((opt.value || '').trim().toLowerCase() === 'chat.general.v1') {
        opt.textContent = 'chat.general.v1';
      }
    });
    const laneApi = (typeof globalThis !== 'undefined' ? globalThis : window).OrionHubGroundedSmallLane;
    if (laneApi && typeof laneApi.syncRecallProfileForLane === 'function') {
      const lane =
        (typeof document !== 'undefined' && document.body && document.body.dataset.orionChatLane) ||
        'grounded_small';
      laneApi.syncRecallProfileForLane(lane);
    } else if (!recallProfileSelect.value || recallProfileSelect.value === 'auto') {
      recallProfileSelect.value = 'assist.light.v1';
    }
  }

  function extractResponseInspectSections(meta = {}) {
    const sections = [];
    const addSection = (key, label, value) => {
      if (value === null || value === undefined) return;
      if (typeof value === 'string' && !value.trim()) return;
      if (Array.isArray(value) && !value.length) return;
      if (typeof value === 'object' && !Array.isArray(value) && !Object.keys(value).length) return;
      sections.push({ key, label, value });
    };
    const thought = thoughtProcessApi.selectThoughtProcess
      ? thoughtProcessApi.selectThoughtProcess(meta)
      : { text: null, source: null, metadata: {} };
    sections.push({
      key: 'thought_process',
      label: 'Thought process',
      value: thought,
      render: renderThoughtProcessSection,
    });
    addSection('reasoning', 'Reasoning', meta.reasoning || meta.reasoning_trace || meta.reasoningTrace);
    addSection('situation', 'Situation', meta.situation_brief || meta.situationBrief);
    addSection('presence', 'Presence', meta.presence_context || meta.presenceContext);
    addSection('conversation_phase', 'Conversation phase', meta.temporal_phase || (meta.situation_brief || {}).conversation_phase);
    addSection('time_place', 'Time/place', { time: (meta.situation_brief || {}).time, place: (meta.situation_brief || {}).place });
    addSection('weather', 'Weather/environment', (meta.situation_brief || {}).environment);
    addSection('agenda', 'Agenda', (meta.situation_brief || {}).agenda);
    addSection('lab', 'Lab context', (meta.situation_brief || {}).lab);
    addSection('surface', 'Surface', (meta.situation_brief || {}).surface);
    addSection('affordances', 'Affordances', meta.situation_affordances || (meta.situation_brief || {}).affordances);
    addSection('provider_diagnostics', 'Provider diagnostics', (meta.situation_brief || {}).diagnostics);
    addSection('situation_fragment', 'Prompt fragment injected', meta.situation_prompt_fragment || meta.situationPromptFragment);
    addSection('recall', 'Recall', meta.recallDebug || meta.recall_debug || meta.memoryDigest || meta.memory_digest);
    addSection('agent_trace', 'Agent Trace', meta.agentTrace || meta.agent_trace);
    addSection('routing', 'Routing', meta.routingDebug || meta.routing_debug);
    addSection('raw', 'Raw cortex', meta.raw);
    return sections;
  }

  function formatInspectValue(value) {
    if (typeof value === 'string') return value;
    try {
      return JSON.stringify(value, null, 2);
    } catch (_error) {
      return String(value);
    }
  }

  function createThoughtChip(label, value) {
    if (!value) return null;
    const chip = document.createElement('span');
    chip.className = 'rounded-full border border-gray-700 bg-gray-800/80 px-2 py-0.5 text-[10px] text-gray-300';
    chip.textContent = `${label}: ${value}`;
    return chip;
  }

  function appendExecutionStepsPanel(parent, meta = {}) {
    if (!parent || !thoughtProcessApi.mountExecutionStepsPanel) return;
    thoughtProcessApi.mountExecutionStepsPanel(parent, {
      meta,
      apiBaseUrl: API_BASE_URL,
      debug: false,
    });
  }

  function renderThoughtProcessSection(thoughtState = {}, meta = {}) {
    const state = thoughtState && typeof thoughtState === 'object' ? thoughtState : {};
    const metadata = state.metadata && typeof state.metadata === 'object' ? state.metadata : {};
    const root = document.createElement('div');
    root.className = 'space-y-2';

    const card = document.createElement('section');
    card.className = 'rounded-lg border border-violet-500/30 bg-violet-500/5 p-2';

    const header = document.createElement('div');
    header.className = 'flex items-center justify-between gap-2';
    const title = document.createElement('div');
    title.className = 'text-[10px] uppercase tracking-wide text-violet-200';
    title.textContent = 'Thought process';
    header.appendChild(title);

    const copyButton = document.createElement('button');
    copyButton.type = 'button';
    copyButton.className = 'rounded-md border border-violet-400/50 bg-violet-500/15 px-2 py-1 text-[10px] font-semibold text-violet-100 hover:bg-violet-500/25 disabled:cursor-not-allowed disabled:opacity-50';
    copyButton.textContent = 'Copy';
    copyButton.disabled = !state.text;
    copyButton.addEventListener('click', () => {
      if (!state.text) return;
      copyText(state.text, 'Thought process copied.');
    });
    header.appendChild(copyButton);
    card.appendChild(header);

    const chipRow = document.createElement('div');
    chipRow.className = 'mt-2 flex flex-wrap items-center gap-1.5';
    const chips = [
      createThoughtChip('source', state.source),
      createThoughtChip('mode', metadata.mode),
      createThoughtChip('model', metadata.model),
      createThoughtChip('provider', metadata.provider),
      createThoughtChip('tokens', metadata.token_count),
      createThoughtChip('chars', metadata.char_count),
    ].filter(Boolean);
    chips.forEach((chip) => chipRow.appendChild(chip));
    if (chips.length) card.appendChild(chipRow);

    const body = document.createElement('pre');
    body.className = 'mt-2 rounded-md border border-violet-500/20 bg-gray-950/60 p-2 text-[11px] leading-5 text-gray-100 whitespace-pre-wrap break-words max-h-72 overflow-y-auto';
    if (state.text) {
      body.textContent = state.text;
    } else {
      body.className = 'mt-2 rounded-md border border-gray-700 bg-gray-900/50 p-2 text-[11px] leading-5 text-gray-400';
      body.textContent = 'No model thought process was captured for this response.';
    }
    card.appendChild(body);
    root.appendChild(card);
    appendExecutionStepsPanel(root, meta);
    return root;
  }

  function buildResponseInspectPanel(meta = {}) {
    const sections = extractResponseInspectSections(meta);
    if (!sections.length) return null;
    const panel = document.createElement('div');
    panel.className = 'mt-2 rounded-xl border border-gray-700 bg-gray-900/40 p-2 hidden space-y-2';

    const tabRow = document.createElement('div');
    tabRow.className = 'flex flex-wrap items-center gap-1';
    const content = document.createElement('div');
    content.className = 'rounded-lg border border-gray-700 bg-gray-950/50 p-2 text-[11px] text-gray-200';
    let activeKey = sections[0].key;

    const renderContent = () => {
      const activeSection = sections.find((section) => section.key === activeKey) || sections[0];
      content.innerHTML = '';
      if (typeof activeSection.render === 'function') {
        content.appendChild(activeSection.render(activeSection.value, meta));
      } else {
        const pre = document.createElement('pre');
        pre.className = 'whitespace-pre-wrap break-words text-[10px] leading-5 text-gray-200';
        pre.textContent = formatInspectValue(activeSection.value);
        content.appendChild(pre);
      }
      tabRow.querySelectorAll('button').forEach((btn) => {
        const selected = btn.dataset.inspectTab === activeKey;
        btn.className = selected
          ? 'rounded-md border border-indigo-400/60 bg-indigo-500/20 px-2 py-1 text-[10px] font-semibold text-indigo-100'
          : 'rounded-md border border-gray-700 bg-gray-800/70 px-2 py-1 text-[10px] text-gray-300 hover:bg-gray-700';
      });
    };

    sections.forEach((section) => {
      const tab = document.createElement('button');
      tab.type = 'button';
      tab.dataset.inspectTab = section.key;
      tab.textContent = section.label;
      tab.addEventListener('click', () => {
        activeKey = section.key;
        renderContent();
      });
      tabRow.appendChild(tab);
    });
    renderContent();
    panel.appendChild(tabRow);
    panel.appendChild(content);
    return panel;
  }

  const RESPONSE_FEEDBACK_CATEGORY_OPTIONS_FALLBACK = {
    up: [
      { value: 'helpful_actionable', label: 'Helpful / actionable' },
      { value: 'well_grounded', label: 'Well grounded' },
      { value: 'good_recall_continuity', label: 'Good recall / continuity' },
      { value: 'right_depth', label: 'Right depth' },
      { value: 'good_tone', label: 'Good tone' },
      { value: 'strong_implementation_detail', label: 'Strong implementation detail' },
      { value: 'good_structure_easy_to_use', label: 'Good structure / easy to use' },
      { value: 'good_judgment', label: 'Good judgment' },
      { value: 'other', label: 'Other' },
    ],
    down: [
      { value: 'made_up_facts', label: 'Made up facts' },
      { value: 'fabricated_recall_memory', label: 'Fabricated recall / memory' },
      { value: 'missed_relevant_context', label: 'Missed relevant context' },
      { value: 'lost_conversation_continuity', label: 'Lost conversation continuity' },
      { value: 'contradicted_earlier_messages', label: 'Contradicted earlier messages' },
      { value: 'did_not_distinguish_fact_vs_inference', label: "Didn't distinguish fact vs inference" },
      { value: 'overconfident_false_certainty', label: 'Overconfident / false certainty' },
      { value: 'too_surface_level', label: 'Too surface-level' },
      { value: 'too_abstract', label: 'Too abstract' },
      { value: 'not_actionable', label: 'Not actionable' },
      { value: 'incomplete_truncated', label: 'Incomplete / truncated' },
      { value: 'did_not_answer_directly', label: "Didn't answer directly" },
      { value: 'missed_edge_cases', label: 'Missed edge cases' },
      { value: 'incorrect_tone', label: 'Incorrect tone' },
      { value: 'too_boilerplate_generic', label: 'Too boilerplate / generic' },
      { value: 'too_guarded_sanitized', label: 'Too guarded / sanitized' },
      { value: 'poor_attunement', label: 'Poor attunement' },
      { value: 'ignored_instructions', label: 'Ignored instructions' },
      { value: 'asked_unnecessary_follow_up', label: 'Asked unnecessary follow-up' },
      { value: 'poor_structure_hard_to_scan', label: 'Poor structure / hard to scan' },
      { value: 'wrong_tool_wrong_routing_wrong_mode', label: 'Wrong tool / routing / mode' },
      { value: 'should_have_probed_more_about_stated_topics', label: 'Should have probed more' },
      { value: 'other', label: 'Other' },
    ],
  };
  let responseFeedbackCategoryOptions = RESPONSE_FEEDBACK_CATEGORY_OPTIONS_FALLBACK;
  let responseFeedbackDraft = null;

  async function loadResponseFeedbackOptions() {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/chat/response-feedback/options`);
      if (!resp.ok) return;
      const data = await resp.json();
      const categories = data && typeof data === 'object' ? data.categories : null;
      if (
        categories
        && typeof categories === 'object'
        && Array.isArray(categories.up)
        && Array.isArray(categories.down)
      ) {
        responseFeedbackCategoryOptions = categories;
      }
    } catch (_err) {
      // keep fallback options
    }
  }

  function feedbackTargetKey(meta = {}) {
    const linkage = resolveFeedbackLinkage(meta);
    const turnId = linkage ? String(linkage.targetTurnId || linkage.targetCorrelationId || linkage.targetMessageId || '').trim() : '';
    const sessionId = String(orionSessionId || '').trim();
    if (!turnId || !sessionId) return null;
    return `${sessionId}:${turnId}`;
  }

  function resolveFeedbackLinkage(meta = {}) {
    const explicitTurnId = String(meta.turnId || '').trim();
    const explicitMessageId = String(meta.messageId || '').trim();
    const explicitCorrelationId = String(meta.correlationId || '').trim();

    let linkageStrategy = 'explicit_ids';
    let targetTurnId = explicitTurnId || null;
    let targetMessageId = explicitMessageId || null;
    let targetCorrelationId = explicitCorrelationId || null;

    if (!targetTurnId && targetCorrelationId) {
      targetTurnId = targetCorrelationId;
      linkageStrategy = 'derived_turn_id_from_correlation_id';
    }
    if (!targetMessageId && targetTurnId) {
      targetMessageId = `${targetTurnId}:assistant`;
      linkageStrategy = linkageStrategy === 'explicit_ids'
        ? 'derived_message_id_from_turn_id'
        : `${linkageStrategy}+derived_message_id_from_turn_id`;
    }
    if (!targetCorrelationId && targetTurnId) {
      targetCorrelationId = targetTurnId;
      linkageStrategy = linkageStrategy === 'explicit_ids'
        ? 'derived_correlation_id_from_turn_id'
        : `${linkageStrategy}+derived_correlation_id_from_turn_id`;
    }

    if (!targetTurnId && !targetMessageId && !targetCorrelationId) return null;
    return { targetTurnId, targetMessageId, targetCorrelationId, linkageStrategy };
  }

  // Single id for data-turn-id and suggest evidence. Explicit meta ids win over linkage-derived :assistant suffixes (user turns must not inherit assistant message ids).
  function canonicalTurnIdForMemoryGraph(meta = {}) {
    const explicitTurn = String(meta.turnId || meta.turn_id || '').trim();
    if (explicitTurn) return explicitTurn;
    const explicitMsg = String(meta.messageId || meta.message_id || '').trim();
    if (explicitMsg) return explicitMsg;
    const linkage = resolveFeedbackLinkage(meta);
    const id =
      (linkage && linkage.targetMessageId)
      || linkage?.targetTurnId
      || linkage?.targetCorrelationId
      || '';
    const s = String(id || '').trim();
    return s || null;
  }

  function buildMemoryGraphSuggestUserContent(turns) {
    const lines = [];
    lines.push('Structured transcript evidence for memory graph extraction (do not invent turns).');
    lines.push('');
    let remaining = MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS;
    let clippedTurns = 0;
    let omittedTurns = 0;
    for (let i = 0; i < turns.length; i += 1) {
      const t = turns[i];
      const id = t.turnId;
      const role = t.role || 'unknown';
      lines.push(`--- turn ${i + 1} id=${id} role=${role} ---`);
      if (remaining <= 0) {
        lines.push('[omitted: input budget exhausted]');
        lines.push('');
        omittedTurns += 1;
        continue;
      }
      const raw = String(t.text || '');
      const perTurn = raw.length > MEMORY_GRAPH_SUGGEST_INPUT_PER_TURN_CHARS
        ? `${raw.slice(0, MEMORY_GRAPH_SUGGEST_INPUT_PER_TURN_CHARS)}…`
        : raw;
      if (perTurn.length < raw.length) clippedTurns += 1;
      const capped = perTurn.length > remaining ? `${perTurn.slice(0, Math.max(0, remaining - 1))}…` : perTurn;
      if (capped.length < perTurn.length) clippedTurns += 1;
      lines.push(capped);
      lines.push('');
      remaining -= capped.length;
    }
    if (clippedTurns > 0 || omittedTurns > 0) {
      lines.push(
        `[Input clipped for reliability: clipped_turns=${clippedTurns}, omitted_turns=${omittedTurns}, max_total_chars=${MEMORY_GRAPH_SUGGEST_INPUT_TOTAL_CHARS}, max_per_turn_chars=${MEMORY_GRAPH_SUGGEST_INPUT_PER_TURN_CHARS}]`
      );
    }
    lines.push('Emit utterance_ids matching the ids above; fill utterance_text_by_id with excerpts.');
    return lines.join('\n');
  }

  function closeResponseFeedbackModal() {
    if (!responseFeedbackModal) return;
    responseFeedbackModal.classList.add('hidden');
    responseFeedbackModal.setAttribute('aria-hidden', 'true');
    responseFeedbackDraft = null;
  }

  function renderResponseFeedbackCategoryOptions(value) {
    if (!responseFeedbackCategoryList) return;
    responseFeedbackCategoryList.innerHTML = '';
    const options = responseFeedbackCategoryOptions[value] || [];
    options.forEach((item) => {
      const label = document.createElement('label');
      label.className = 'inline-flex items-center gap-2 rounded-full border border-gray-700 bg-gray-900/70 px-3 py-1 text-xs text-gray-200';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = item.value;
      checkbox.className = 'h-3 w-3 accent-indigo-400';
      checkbox.checked = !!(responseFeedbackDraft && responseFeedbackDraft.categories.has(item.value));
      checkbox.addEventListener('change', () => {
        if (!responseFeedbackDraft) return;
        if (checkbox.checked) responseFeedbackDraft.categories.add(item.value);
        else responseFeedbackDraft.categories.delete(item.value);
      });
      const text = document.createElement('span');
      text.textContent = item.label;
      label.appendChild(checkbox);
      label.appendChild(text);
      responseFeedbackCategoryList.appendChild(label);
    });
  }

  function openResponseFeedbackModal(value, meta = {}, responseText = '') {
    if (!responseFeedbackModal || !responseFeedbackTitle) return;
    const linkage = resolveFeedbackLinkage(meta);
    if (!linkage) return;
    const key = feedbackTargetKey(meta);
    if (key && submittedFeedbackTargets.has(key)) return;
    responseFeedbackDraft = {
      feedbackId: (window.crypto && typeof window.crypto.randomUUID === 'function') ? window.crypto.randomUUID() : `feedback:${Date.now()}:${Math.random()}`,
      feedbackValue: value,
      categories: new Set(),
      targetTurnId: linkage.targetTurnId,
      targetMessageId: linkage.targetMessageId,
      targetCorrelationId: linkage.targetCorrelationId,
      sessionId: String(orionSessionId || ''),
      userId: '',
      source: 'hub_ui',
      responsePreview: String(responseText || '').slice(0, 140),
      uiContext: {
        mode: meta.routingDebug && meta.routingDebug.mode ? meta.routingDebug.mode : null,
        trace_verb: meta.routingDebug && meta.routingDebug.verb ? meta.routingDebug.verb : null,
        linkage_strategy: linkage.linkageStrategy,
      },
    };
    responseFeedbackTitle.textContent = value === 'up' ? 'What was good about this response?' : 'What went wrong with this response?';
    if (responseFeedbackMeta) {
      const turnLabel = String(linkage.targetTurnId || linkage.targetCorrelationId || linkage.targetMessageId || '').slice(0, 16);
      responseFeedbackMeta.textContent = `Turn ${turnLabel} · ${value === 'up' ? 'thumbs up' : 'thumbs down'}`;
    }
    if (responseFeedbackNotes) responseFeedbackNotes.value = '';
    if (responseFeedbackStatus) responseFeedbackStatus.textContent = '';
    renderResponseFeedbackCategoryOptions(value);
    responseFeedbackModal.classList.remove('hidden');
    responseFeedbackModal.setAttribute('aria-hidden', 'false');
  }

  async function submitResponseFeedback() {
    if (!responseFeedbackDraft) return;
    if (!responseFeedbackSubmit) return;
    if (!responseFeedbackDraft.feedbackValue || !['up', 'down'].includes(responseFeedbackDraft.feedbackValue)) return;
    responseFeedbackSubmit.disabled = true;
    responseFeedbackSubmit.textContent = 'Submitting…';
    if (responseFeedbackStatus) responseFeedbackStatus.textContent = '';
    try {
      const payload = {
        feedback_id: responseFeedbackDraft.feedbackId,
        target_turn_id: responseFeedbackDraft.targetTurnId,
        target_message_id: responseFeedbackDraft.targetMessageId,
        target_correlation_id: responseFeedbackDraft.targetCorrelationId,
        session_id: responseFeedbackDraft.sessionId || null,
        user_id: responseFeedbackDraft.userId || null,
        feedback_value: responseFeedbackDraft.feedbackValue,
        categories: Array.from(responseFeedbackDraft.categories),
        free_text: responseFeedbackNotes ? String(responseFeedbackNotes.value || '').trim().slice(0, 2000) : null,
        source: responseFeedbackDraft.source,
        ui_context: responseFeedbackDraft.uiContext,
      };
      const resp = await fetch(`${API_BASE_URL}/api/chat/response-feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `HTTP ${resp.status}`);
      }
      const key = feedbackTargetKey({
        turnId: responseFeedbackDraft.targetTurnId,
        correlationId: responseFeedbackDraft.targetCorrelationId,
      });
      if (key) submittedFeedbackTargets.add(key);
      if (responseFeedbackStatus) responseFeedbackStatus.textContent = 'Thanks — feedback saved.';
      closeResponseFeedbackModal();
    } catch (err) {
      if (responseFeedbackStatus) responseFeedbackStatus.textContent = `Could not save feedback: ${String(err.message || err)}`;
    } finally {
      responseFeedbackSubmit.disabled = false;
      responseFeedbackSubmit.textContent = 'Submit feedback';
    }
  }

  function hubCoalesceAssistantText(primary, meta) {
    const a = String(primary || '').trim();
    if (!meta || typeof meta !== 'object') return primary || '';
    if (meta.workflowMetadataOnly || meta.workflow_metadata_only) return primary || '';
    const raw = meta.raw;
    if (!raw || typeof raw !== 'object') return primary || '';
    let b = String(raw.final_text || '').trim();
    if (!b && raw.cortex_result && typeof raw.cortex_result === 'object') {
      b = String(raw.cortex_result.final_text || '').trim();
    }
    if (b.length > a.length) return b;
    return a || b;
  }

  function memoryGraphCorrelationBase(meta) {
    if (!meta || typeof meta !== 'object') return '';
    const raw = meta.raw;
    return String(
      meta.correlationId
      || meta.correlation_id
      || meta.turnId
      || meta.turn_id
      || (raw && typeof raw === 'object' ? raw.correlation_id : null)
      || '',
    ).trim();
  }

  /** Pair the latest You-line with this Orion turn (WS/HTTP never sent user meta). */
  function backfillLatestUserTurnIdForGraph(threadRoot, correlationId) {
    const base = String(correlationId || '').trim();
    if (!threadRoot || !base) return;
    let el = threadRoot.lastElementChild;
    while (el) {
      const role = el.dataset && el.dataset.role;
      if (role === 'user') {
        if (!el.dataset.turnId) el.dataset.turnId = `${base}:user`;
        return;
      }
      if (role === 'assistant' && el.dataset.turnId) return;
      el = el.previousElementSibling;
    }
  }

  function contentLooksLikeGatewayFailureBlurb(c) {
    const s = String(c || '').trim();
    if (!s) return false;
    if (s.startsWith('[Error:')) return true;
    const head = s.length > 600 ? `${s.slice(0, 600)}…` : s;
    return /\b(timed out|timeout)\b/i.test(head);
  }

  function extractCortexStepErrorHint(raw) {
    if (!raw || typeof raw !== 'object') return '';
    const topErr = raw.error != null ? String(raw.error).trim() : '';
    const steps = raw.steps;
    const parts = [];
    if (topErr) parts.push(topErr);
    if (!Array.isArray(steps)) {
      return parts.filter(Boolean).join(' · ') || '';
    }
    steps.forEach((step) => {
      if (!step || typeof step !== 'object') return;
      if (step.error != null) parts.push(String(step.error));
      const res = step.result;
      if (!res || typeof res !== 'object') return;
      Object.keys(res).forEach((svcKey) => {
        const block = res[svcKey];
        if (!block || typeof block !== 'object') return;
        const be = block.error;
        if (be != null) {
          if (typeof be === 'string' && be.trim()) parts.push(be.trim());
          else if (typeof be === 'object' && typeof be.message === 'string' && be.message.trim()) {
            parts.push(be.message.trim());
          }
        }
        const c = String(block.content || '').trim();
        if (contentLooksLikeGatewayFailureBlurb(c)) parts.push(c);
      });
    });
    const merged = parts.filter(Boolean);
    if (!merged.length) return '';
    return merged.filter((v, i, a) => a.indexOf(v) === i).join(' · ');
  }

  function appendMessage(sender, text, colorClass = 'text-white') {
    if (!conversationDiv) return;
    const meta = arguments.length > 3 && arguments[3] && typeof arguments[3] === 'object' ? arguments[3] : {};
    if (orionSessionId && !meta.sessionId && !meta.session_id) {
      meta.sessionId = orionSessionId;
    }
    if (sender === 'Orion') {
      const corr = memoryGraphCorrelationBase(meta);
      if (corr) backfillLatestUserTurnIdForGraph(conversationDiv, corr);
    }
    const div = document.createElement('div');
    const color = sender === 'You' ? 'text-blue-300' : 'text-green-300';
    const turnIdForGraph = canonicalTurnIdForMemoryGraph(meta);
    if (turnIdForGraph) div.dataset.turnId = turnIdForGraph;
    else if (sender === 'You') {
      const uuid = (window.crypto && typeof window.crypto.randomUUID === 'function')
        ? window.crypto.randomUUID()
        : `u${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
      div.dataset.turnId = `hub-utterance:${uuid}`;
    }
    div.dataset.role = sender === 'Orion' ? 'assistant' : (sender === 'You' ? 'user' : 'system');
    const displayText = sender === 'Orion' ? hubCoalesceAssistantText(text, meta) : (text || '');
    const workflowOnlyTurn = Boolean(
      sender === 'Orion'
      && (meta.workflowMetadataOnly || meta.workflow_metadata_only)
      && (!String(displayText || '').trim())
    );
    let inspectPanel = null;
    let feedbackRow = null;
    const headerRow = document.createElement('div');
    headerRow.className = 'mb-1 flex items-center justify-between gap-3';
    const header = document.createElement('p');
    header.className = `font-bold ${color}`;
    header.textContent = sender;
    headerRow.appendChild(header);
    const body = document.createElement('p');
    body.className = `${colorClass} whitespace-pre-wrap`;
    body.textContent = displayText;
    div.className = "mb-2 border-b border-gray-800/50 pb-2 last:border-0";
    div.appendChild(headerRow);
    const workflowPanel = sender === 'Orion' ? createWorkflowPanel(meta.workflow, {
      onRunAgain: async (workflow) => submitExplicitChatText(workflow.rerun_prompt),
    }) : null;
    if (workflowPanel) div.appendChild(workflowPanel);
    if (!workflowOnlyTurn) div.appendChild(body);
    conversationDiv.appendChild(div);
    conversationDiv.scrollTop = conversationDiv.scrollHeight;

    if (sender === 'Orion') {
      const autonomyMeta = { ...meta, replyText: displayText };
      try {
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
      const memGraphTurnId = canonicalTurnIdForMemoryGraph(meta);
      if (memGraphTurnId) {
        const graphBtn = document.createElement('button');
        graphBtn.type = 'button';
        graphBtn.className = 'rounded-full border border-violet-500/40 bg-violet-500/10 px-2 py-1 text-[10px] font-semibold text-violet-200 hover:bg-violet-500/20';
        graphBtn.textContent = 'Memory graph';
        graphBtn.addEventListener('click', () => openMemoryGraphBridgeModal(div));
        actionRow.appendChild(graphBtn);
      }
      const mindCorrelationId = mindCorrelationFromMeta(meta);
      if (mindCorrelationId) {
        const mindButton = document.createElement('button');
        mindButton.type = 'button';
        mindButton.className = 'rounded-full border border-indigo-500/40 bg-indigo-500/10 px-2 py-1 text-[10px] font-semibold text-indigo-200 hover:bg-indigo-500/20';
        mindButton.textContent = 'Mind';
        mindButton.addEventListener('click', () => openMindRunsModal(mindCorrelationId, mindButton, meta));
        actionRow.appendChild(mindButton);
      } else {
        const disabledMindButton = document.createElement('button');
        disabledMindButton.type = 'button';
        disabledMindButton.className = 'rounded-full border border-gray-700 bg-gray-800 px-2 py-1 text-[10px] font-semibold text-gray-400';
        disabledMindButton.textContent = 'Mind';
        disabledMindButton.disabled = true;
        disabledMindButton.title = 'Mind data unavailable for this message';
        actionRow.appendChild(disabledMindButton);
      }
      // Substrate Effect chip — only renders when summary is present on meta.
      const substrateSummary = meta.substrateEffectSummary || meta.substrate_effect_summary;
      if (substrateSummary && window.SubstrateEffectUI && typeof window.SubstrateEffectUI.renderChip === 'function') {
        const chip = window.SubstrateEffectUI.renderChip(substrateSummary);
        if (chip) actionRow.appendChild(chip);
      }
      inspectPanel = buildResponseInspectPanel(meta);
      if (inspectPanel) {
        const inspectButton = document.createElement('button');
        inspectButton.className = 'rounded-full border border-blue-500/40 bg-blue-500/10 px-2 py-1 text-[10px] font-semibold text-blue-200 hover:bg-blue-500/20';
        inspectButton.type = 'button';
        inspectButton.textContent = 'Inspect';
        inspectButton.addEventListener('click', () => {
          const nextHidden = inspectPanel.classList.contains('hidden');
          inspectPanel.classList.toggle('hidden', !nextHidden);
          inspectButton.textContent = nextHidden ? 'Hide Inspect' : 'Inspect';
        });
        actionRow.appendChild(inspectButton);
      }
      if (actionRow.childNodes.length) {
        headerRow.appendChild(actionRow);
      }
      const targetKey = feedbackTargetKey(meta);
      feedbackRow = document.createElement('div');
      feedbackRow.className = 'mt-2 flex items-center gap-2';
      const thumbsUp = document.createElement('button');
      thumbsUp.type = 'button';
      thumbsUp.className = 'rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 text-[10px] font-semibold text-emerald-200 hover:bg-emerald-500/20';
      thumbsUp.textContent = '👍';
      const thumbsDown = document.createElement('button');
      thumbsDown.type = 'button';
      thumbsDown.className = 'rounded-full border border-rose-500/40 bg-rose-500/10 px-2 py-1 text-[10px] font-semibold text-rose-200 hover:bg-rose-500/20';
      thumbsDown.textContent = '👎';
      const ack = document.createElement('span');
      ack.className = 'text-[10px] text-gray-400';
      if (targetKey && submittedFeedbackTargets.has(targetKey)) {
        thumbsUp.disabled = true;
        thumbsDown.disabled = true;
        ack.textContent = 'Feedback saved';
      }
      thumbsUp.addEventListener('click', () => openResponseFeedbackModal('up', meta, displayText));
      thumbsDown.addEventListener('click', () => openResponseFeedbackModal('down', meta, displayText));
      feedbackRow.appendChild(thumbsUp);
      feedbackRow.appendChild(thumbsDown);
      feedbackRow.appendChild(ack);
      if (feedbackRow) div.appendChild(feedbackRow);
      if (inspectPanel) div.appendChild(inspectPanel);
      const autonomyPanel = createAutonomyPanel(
        meta.autonomySummary || meta.autonomy_summary,
        meta.autonomyDebug || meta.autonomy_debug,
        { ...meta, replyText: displayText },
      );
      if (autonomyPanel) div.appendChild(autonomyPanel);
      const tracePanel = createAgentTracePanel(meta.agentTrace, meta);
      if (tracePanel) div.appendChild(tracePanel);
      const metacogPanel = createMetacogTracePanel(meta.metacogTraces || meta.metacog_traces || []);
      if (metacogPanel) div.appendChild(metacogPanel);
      appendExecutionStepsPanel(div, meta);
      } catch (err) {
        console.warn('[Hub] Orion message chrome failed (body still shown)', err);
      }
      try {
        updateAgentTraceDebugPanel(meta.agentTrace, meta);
      } catch (err) {
        console.warn('[Hub] Agent trace debug panel failed', err);
      }
      try {
        updateAutonomyDebugPanel(
          meta.autonomySummary || meta.autonomy_summary,
          meta.autonomyDebug || meta.autonomy_debug,
          autonomyMeta,
        );
      } catch (err) {
        console.warn('[Hub] Autonomy debug panel failed', err);
      }
      try {
        updateChatStanceDebugPanel(meta.chatStanceDebug || meta.chat_stance_debug);
      } catch (err) {
        console.warn('[Hub] Chat stance debug panel failed', err);
      }
    }
  }

  function collectConversationTurnsUpTo(anchorEl, maxTurns) {
    if (!conversationDiv || !anchorEl) return { turns: [], skippedWithoutId: 0 };
    const out = [];
    let skippedWithoutId = 0;
    let el = anchorEl;
    while (el && out.length < maxTurns) {
      if (el.parentElement === conversationDiv) {
        if (el.dataset && el.dataset.turnId) {
          const body = el.querySelector('p.whitespace-pre-wrap');
          out.push({
            turnId: el.dataset.turnId,
            role: el.dataset.role || 'unknown',
            text: body ? body.textContent : '',
          });
        } else {
          skippedWithoutId += 1;
        }
      }
      el = el.previousElementSibling;
    }
    return { turns: out.reverse(), skippedWithoutId };
  }

  function readBridgeMaxTurnsStored() {
    const raw = localStorage.getItem(LS_MEMORY_GRAPH_BRIDGE_MAX_TURNS);
    let n = raw ? parseInt(raw, 10) : MEMORY_GRAPH_BRIDGE_MAX_TURNS_DEFAULT;
    if (!Number.isFinite(n)) n = MEMORY_GRAPH_BRIDGE_MAX_TURNS_DEFAULT;
    return Math.min(MEMORY_GRAPH_BRIDGE_MAX_TURNS_CAP, Math.max(1, n));
  }

  function ensureMemoryGraphBridgeDraftViz() {
    if (!window.OrionMemoryGraphDraftUI) return null;
    if (memoryGraphBridgeDraftViz) return memoryGraphBridgeDraftViz;
    const ta = document.getElementById('memoryGraphBridgeDraft');
    const cy = document.getElementById('memoryGraphBridgeCyHost');
    const det = document.getElementById('memoryGraphBridgeDetail');
    const ban = document.getElementById('memoryGraphBridgeParseBanner');
    const formHost = document.getElementById('memoryGraphBridgeFormHost');
    if (!ta || !cy || !det) return null;
    memoryGraphBridgeDraftViz = window.OrionMemoryGraphDraftUI.attach({
      draftTextarea: ta,
      cyHost: cy,
      detailHost: det,
      bannerEl: ban,
      onDraftJsonChange: () => {
        if (memoryGraphBridgeDraftForm && typeof memoryGraphBridgeDraftForm.refresh === 'function') {
          memoryGraphBridgeDraftForm.refresh();
        }
      },
    });
    if (formHost && window.OrionMemoryGraphDraftForm && !memoryGraphBridgeDraftForm) {
      memoryGraphBridgeDraftForm = window.OrionMemoryGraphDraftForm.attachFormEditor({
        draftTextarea: ta,
        formHost,
        onDraftChange: () => {
          const viz = memoryGraphBridgeDraftViz || ensureMemoryGraphBridgeDraftViz();
          if (viz && typeof viz.refresh === 'function') viz.refresh();
        },
      });
    }
    return memoryGraphBridgeDraftViz;
  }

  function flushMemoryGraphBridgeDraftForm() {
    if (memoryGraphBridgeDraftForm && typeof memoryGraphBridgeDraftForm.flushToTextarea === 'function') {
      memoryGraphBridgeDraftForm.flushToTextarea();
    }
  }

  function ensureOrganSignalsGraph() {
    if (!window.OrionOrganSignalsGraphUI) {
      return null;
    }
    if (organSignalsGraphCtl) {
      return organSignalsGraphCtl;
    }
    const cyHost = document.getElementById("organSignalsCyHost");
    if (!cyHost) {
      return null;
    }
    organSignalsGraphCtl = window.OrionOrganSignalsGraphUI.attach({
      apiBaseUrl: API_BASE_URL,
      cyHost,
      statusEl: document.getElementById("organSignalsStatus"),
      detailEl: document.getElementById("organSignalsDetail"),
      refreshBtn: document.getElementById("organSignalsRefreshBtn"),
      autoRefreshCheckbox: document.getElementById("organSignalsAutoRefresh"),
      layerFilterEl: document.getElementById("organ-signals-layer-filter"),
    });
    return organSignalsGraphCtl;
  }

  function openOrganSignalsForCorrelation(correlationId) {
    const corr = String(correlationId || '').trim();
    if (!corr) return;
    const url = new URL(window.location.href);
    url.searchParams.set('correlation_id', corr);
    url.hash = '#signals';
    history.pushState(null, '', `${url.pathname}${url.search}${url.hash}`);
    setActiveTab('signals');
    const ctl = ensureOrganSignalsGraph();
    if (ctl && typeof ctl.refresh === 'function') {
      ctl.refresh();
    }
  }
  window.OrionHubOpenOrganSignals = openOrganSignalsForCorrelation;

  function closeMemoryGraphBridgeModal() {
    if (!memoryGraphBridgeModal) return;
    memoryGraphBridgeAnchorDiv = null;
    memoryGraphBridgeModal.classList.add('hidden');
    memoryGraphBridgeModal.setAttribute('aria-hidden', 'true');
    const hintsEl = document.getElementById('memoryGraphBridgeChainHints');
    if (hintsEl) {
      hintsEl.textContent = '';
      hintsEl.classList.add('hidden');
    }
  }

  function rebuildMemoryGraphBridgeTurnList() {
    const list = document.getElementById('memoryGraphBridgeTurnList');
    const statusEl = document.getElementById('memoryGraphBridgeStatus');
    const hintsEl = document.getElementById('memoryGraphBridgeChainHints');
    if (!list || !memoryGraphBridgeAnchorDiv) return;
    const maxTurnsInputEl = document.getElementById('memoryGraphBridgeMaxTurns');
    let maxTurns = parseInt(maxTurnsInputEl && maxTurnsInputEl.value ? maxTurnsInputEl.value : '', 10);
    if (!Number.isFinite(maxTurns)) maxTurns = readBridgeMaxTurnsStored();
    maxTurns = Math.min(MEMORY_GRAPH_BRIDGE_MAX_TURNS_CAP, Math.max(1, maxTurns));
    const { turns, skippedWithoutId } = collectConversationTurnsUpTo(memoryGraphBridgeAnchorDiv, maxTurns);
    memoryGraphBridgeTurnsCache = turns;
    list.innerHTML = '';
    if (statusEl) statusEl.textContent = '';
    if (hintsEl) {
      hintsEl.textContent = '';
      hintsEl.classList.add('hidden');
    }
    if (!turns.length) {
      if (statusEl) statusEl.textContent = 'No turns with stable ids found in this thread yet.';
      if (skippedWithoutId > 0 && hintsEl) {
        hintsEl.textContent = `${skippedWithoutId} older message(s) have no stable turn id (often user turns). They are omitted from the chain until linkage meta is present.`;
        hintsEl.classList.remove('hidden');
      }
      const viz0 = ensureMemoryGraphBridgeDraftViz();
      if (viz0 && viz0.refresh) requestAnimationFrame(() => viz0.refresh());
      return;
    }
    const lastIdx = turns.length - 1;
    let priorUserIdx = -1;
    for (let j = lastIdx - 1; j >= 0; j -= 1) {
      if (turns[j].role === 'user') {
        priorUserIdx = j;
        break;
      }
    }
    const hasUserTurnInChain = turns.some((t) => t.role === 'user');
    if (hintsEl) {
      const hintParts = [];
      if (skippedWithoutId > 0) {
        hintParts.push(`${skippedWithoutId} older message(s) have no stable turn id and were skipped when building this chain.`);
      }
      if (!hasUserTurnInChain) {
        hintParts.push('No user turn with a stable id appears in this chain — select turns manually, or ensure chat passes turn ids for user messages.');
      } else if (priorUserIdx < 0) {
        hintParts.push('No user turn directly before this reply appears in the chain (ids may start mid-thread). Adjust checkboxes as needed.');
      }
      if (hintParts.length) {
        hintsEl.textContent = hintParts.join(' ');
        hintsEl.classList.remove('hidden');
      }
    }
    turns.forEach((t, i) => {
      const row = document.createElement('label');
      row.className = 'flex gap-2 items-start border border-gray-800 rounded p-2';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'mt-0.5 accent-violet-400';
      cb.dataset.turnIndex = String(i);
      cb.checked = i === lastIdx || i === priorUserIdx;
      const cap = document.createElement('div');
      cap.className = 'min-w-0 flex-1';
      const roleLine = document.createElement('div');
      roleLine.className = 'text-[10px] text-gray-500 font-mono';
      roleLine.textContent = `${t.role} · ${t.turnId}`;
      const snippet = document.createElement('div');
      snippet.className = 'text-gray-200 mt-1 break-words';
      snippet.textContent = (t.text || '').slice(0, 600) + ((t.text || '').length > 600 ? '…' : '');
      cap.appendChild(roleLine);
      cap.appendChild(snippet);
      row.appendChild(cb);
      row.appendChild(cap);
      list.appendChild(row);
    });
    const viz = ensureMemoryGraphBridgeDraftViz();
    if (viz && viz.refresh) requestAnimationFrame(() => viz.refresh());
  }

  function openMemoryGraphBridgeModal(anchorDiv) {
    const draftTa = document.getElementById('memoryGraphBridgeDraft');
    const closeForFocus = document.getElementById('memoryGraphBridgeModalClose');
    if (!memoryGraphBridgeModal || !document.getElementById('memoryGraphBridgeTurnList')) return;
    memoryGraphBridgeAnchorDiv = anchorDiv;
    const maxTurnsInputEl = document.getElementById('memoryGraphBridgeMaxTurns');
    if (maxTurnsInputEl) maxTurnsInputEl.value = String(readBridgeMaxTurnsStored());
    rebuildMemoryGraphBridgeTurnList();
    if (memoryGraphBridgeTurnsCache.length > 0 && draftTa) draftTa.value = '';
    renderMemoryGraphBridgeDiagnostics(null, null);
    memoryGraphBridgeModal.classList.remove('hidden');
    memoryGraphBridgeModal.setAttribute('aria-hidden', 'false');
    if (closeForFocus && typeof closeForFocus.focus === 'function') {
      closeForFocus.focus({ preventScroll: true });
    }
  }

  function formatMemoryGraphBridgeDiagnosticsBlock(coalesce, apiData) {
    const diag = coalesce && coalesce.diagnostics ? { ...coalesce.diagnostics } : {};
    if (apiData && typeof apiData === 'object') {
      const apiAttempts = Array.isArray(apiData.attempts)
        ? apiData.attempts
        : Array.isArray(apiData.suggest_attempts)
          ? apiData.suggest_attempts
          : [];
      if (apiAttempts.length && (!Array.isArray(diag.attempts) || !diag.attempts.length)) {
        diag.attempts = apiAttempts;
      }
      if (apiData.route_used != null && diag.route_used == null) diag.route_used = apiData.route_used;
      if (apiData.error != null && !diag.api_error) diag.api_error = apiData.error;
    }
    const lines = [];
    if (diag.route_used != null) lines.push(`route_used: ${diag.route_used}`);
    if (Array.isArray(diag.attempts) && diag.attempts.length) {
      lines.push('attempts:');
      diag.attempts.forEach((a, i) => {
        if (!a || typeof a !== 'object') return;
        const route = a.route != null ? a.route : a.route_used;
        const phase = a.phase != null ? a.phase : '';
        const err =
          a.error_summary != null
            ? a.error_summary
            : a.error != null
              ? a.error
              : '';
        lines.push(`  [${i}] route=${route || '?'} phase=${phase || '?'} error=${err || '—'}`);
        if (Array.isArray(a.validation_errors) && a.validation_errors.length) {
          lines.push(`      validation_errors: ${a.validation_errors.join(', ')}`);
        }
        if (a.raw_output_preview) lines.push(`      raw_output_preview: ${String(a.raw_output_preview).slice(0, 400)}`);
      });
    }
    if (Array.isArray(diag.validation_errors) && diag.validation_errors.length) {
      lines.push(`validation_errors: ${diag.validation_errors.join(', ')}`);
    }
    if (diag.api_error) lines.push(`api_error: ${diag.api_error}`);
    if (diag.prose_preview) lines.push(`prose_preview: ${String(diag.prose_preview).slice(0, 400)}`);
    if (diag.diagnostic_raw) lines.push(`diagnostic_raw: ${String(diag.diagnostic_raw).slice(0, 600)}`);
    if (coalesce && coalesce.error) lines.push(`coalesce_error: ${coalesce.error}`);
    if (apiData && apiData.ok === false && apiData.error) lines.push(`api_response_error: ${apiData.error}`);
    if (apiData && apiData.fallback_draft) lines.push('fallback_draft: true');
    return lines.join('\n');
  }

  function renderMemoryGraphBridgeDiagnostics(coalesce, apiData) {
    const diagEl = document.getElementById('memoryGraphBridgeDiagnostics');
    const copyBtn = document.getElementById('memoryGraphBridgeCopyDiagnostics');
    const text = formatMemoryGraphBridgeDiagnosticsBlock(coalesce, apiData);
    const show = Boolean(text && text.trim());
    if (diagEl) {
      if (show) {
        diagEl.textContent = text;
        diagEl.classList.remove('hidden');
      } else {
        diagEl.textContent = '';
        diagEl.classList.add('hidden');
      }
    }
    if (copyBtn) {
      copyBtn.classList.toggle('hidden', !show);
      copyBtn.dataset.diagnosticsText = show ? text : '';
    }
  }

  function setupMemoryGraphBridgeModal() {
    const closeBtn = document.getElementById('memoryGraphBridgeModalClose');
    const suggestBtn = document.getElementById('memoryGraphBridgeSuggest');
    const toMemBtn = document.getElementById('memoryGraphBridgeToMemory');
    const draftTa = document.getElementById('memoryGraphBridgeDraft');
    const statusEl = document.getElementById('memoryGraphBridgeStatus');
    const list = document.getElementById('memoryGraphBridgeTurnList');
    const copyDiagBtn = document.getElementById('memoryGraphBridgeCopyDiagnostics');
    if (!memoryGraphBridgeModal || !closeBtn) return;
    const maxTurnsInput = document.getElementById('memoryGraphBridgeMaxTurns');
    if (maxTurnsInput) {
      maxTurnsInput.value = String(readBridgeMaxTurnsStored());
      function syncBridgeMaxTurnsFromInput(opts) {
        const strict = Boolean(opts && opts.strict);
        const raw = String(maxTurnsInput.value || '').trim();
        if (raw === '' && !strict) return;
        let v = parseInt(raw, 10);
        if (!Number.isFinite(v)) {
          if (!strict) return;
          v = MEMORY_GRAPH_BRIDGE_MAX_TURNS_DEFAULT;
        }
        v = Math.min(MEMORY_GRAPH_BRIDGE_MAX_TURNS_CAP, Math.max(1, v));
        maxTurnsInput.value = String(v);
        localStorage.setItem(LS_MEMORY_GRAPH_BRIDGE_MAX_TURNS, String(v));
        rebuildMemoryGraphBridgeTurnList();
      }
      maxTurnsInput.addEventListener('input', () => syncBridgeMaxTurnsFromInput({ strict: false }));
      maxTurnsInput.addEventListener('change', () => syncBridgeMaxTurnsFromInput({ strict: true }));
    }
    const selKBtn = document.getElementById('memoryGraphBridgeSelectLastKBtn');
    const selKIn = document.getElementById('memoryGraphBridgeSelectLastKInput');
    if (selKBtn && selKIn) {
      selKBtn.addEventListener('click', () => {
        const turnList = document.getElementById('memoryGraphBridgeTurnList');
        if (!turnList) return;
        let k = parseInt(selKIn.value, 10);
        const boxes = turnList.querySelectorAll('input[type="checkbox"][data-turn-index]');
        const n = boxes.length;
        if (!n) return;
        if (!Number.isFinite(k) || k < 1) k = n;
        k = Math.min(k, n);
        boxes.forEach((cb, i) => {
          cb.checked = i >= n - k;
        });
      });
    }
    ensureMemoryGraphBridgeDraftViz();
    function close() {
      closeMemoryGraphBridgeModal();
    }
    closeBtn.addEventListener('click', close);
    memoryGraphBridgeModal.addEventListener('click', (e) => {
      if (e.target === memoryGraphBridgeModal) close();
    });
    if (copyDiagBtn) {
      copyDiagBtn.addEventListener('click', async () => {
        const text = copyDiagBtn.dataset.diagnosticsText || '';
        if (!text.trim()) return;
        try {
          await navigator.clipboard.writeText(text);
          showToast('Diagnostics copied.');
        } catch (err) {
          showToast(String(err.message || err));
        }
      });
    }
    if (suggestBtn) {
      suggestBtn.addEventListener('click', async () => {
        if (!list || !draftTa) return;
        if (suggestBtn.disabled) return;
        const boxes = list.querySelectorAll('input[type="checkbox"][data-turn-index]');
        const selected = [];
        boxes.forEach((cb) => {
          if (cb.checked) {
            const idx = Number(cb.dataset.turnIndex);
            const row = memoryGraphBridgeTurnsCache[idx];
            if (row && row.turnId && String(row.turnId).trim()) selected.push(row);
          }
        });
        if (!selected.length) {
          if (statusEl) statusEl.textContent = 'Select at least one turn with a stable id.';
          return;
        }
        const prevLabel = suggestBtn.textContent;
        suggestBtn.disabled = true;
        suggestBtn.textContent = 'Working…';
        if (statusEl) statusEl.textContent = 'Requesting draft…';
        const content = buildMemoryGraphSuggestUserContent(selected);
        const selectedTurnIds = selected
          .map((row) => (row && row.turnId ? String(row.turnId).trim() : ''))
          .filter(Boolean);
        const payload = {
          mode: 'brain',
          verbs: ['memory_graph_suggest'],
          messages: [{ role: 'user', content }],
          use_recall: false,
          no_write: true,
          diagnostic: true,
          options: { diagnostic: true },
        };
        try {
          const headers = {
            'Content-Type': 'application/json',
            ...(orionSessionId ? { 'X-Orion-Session-Id': orionSessionId } : {}),
          };
          const controller = typeof AbortController === 'function' ? new AbortController() : null;
          const timerId = controller
            ? setTimeout(() => {
              try {
                controller.abort();
              } catch (_) {
                /* ignore */
              }
            }, memoryGraphSuggestFetchTimeoutMs())
            : null;
          let res;
          try {
            res = await fetch(`${API_BASE_URL}/api/memory/graph/suggest`, {
              method: 'POST',
              headers,
              body: JSON.stringify(payload),
              ...(controller ? { signal: controller.signal } : {}),
            });
          } finally {
            if (timerId != null) clearTimeout(timerId);
          }
          const text = await res.text();
          let data = null;
          try {
            data = text ? JSON.parse(text) : null;
          } catch {
            data = { raw: text };
          }
          const ui = window.OrionMemoryGraphDraftUI || {};
          const utteranceTextById = {};
          selected.forEach((row) => {
            const id = row && row.turnId ? String(row.turnId).trim() : '';
            if (id) utteranceTextById[id] = String(row.text || '').trim();
          });
          const emptyDraftFn =
            typeof ui.emptySuggestDraft === 'function'
              ? ui.emptySuggestDraft
              : typeof ui.emptyValidSuggestDraft === 'function'
                ? ui.emptyValidSuggestDraft
                : null;
          const emptyDraft = emptyDraftFn
            ? emptyDraftFn({ utteranceIds: selectedTurnIds, utteranceTextById })
            : {
                ontology_version: 'orionmem-2026-05',
                utterance_ids: selectedTurnIds,
                entities: [],
                situations: [],
                edges: [],
                dispositions: [],
                utterance_text_by_id: utteranceTextById,
              };
          if (!res.ok) {
            draftTa.value = JSON.stringify(emptyDraft, null, 2);
            const vDraftErr = ensureMemoryGraphBridgeDraftViz();
            if (vDraftErr && vDraftErr.refresh) vDraftErr.refresh();
            const failCoalesce = { error: 'memory_graph_suggest_failed', diagnostics: { api_error: data && data.error } };
            renderMemoryGraphBridgeDiagnostics(failCoalesce, data);
            if (statusEl) {
              statusEl.textContent =
                'Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics.';
              const diagBits =
                typeof data === 'object' && data
                  ? [data.error, data.validation_errors].filter(Boolean)
                  : [text];
              if (diagBits.length) statusEl.textContent += ` ${String(diagBits)}`;
            }
            return;
          }
          const coalesceFn =
            typeof ui.coalesceMemoryGraphSuggestEnvelope === 'function'
              ? ui.coalesceMemoryGraphSuggestEnvelope
              : null;
          const coalesce = coalesceFn
            ? coalesceFn(data, { utteranceIds: selectedTurnIds, utteranceTextById })
            : null;
          const out =
            coalesce && typeof coalesce.draftText === 'string' && coalesce.draftText.trim()
              ? coalesce.draftText
              : JSON.stringify(emptyDraft, null, 2);
          draftTa.value = out;
          const vDraft = ensureMemoryGraphBridgeDraftViz();
          if (vDraft && vDraft.refresh) vDraft.refresh();
          if (memoryGraphBridgeDraftForm && memoryGraphBridgeDraftForm.refresh) {
            memoryGraphBridgeDraftForm.refresh();
          }
          renderMemoryGraphBridgeDiagnostics(coalesce, data);
          if (statusEl) {
            const statusFn =
              typeof ui.formatSuggestCoalesceUserStatus === 'function'
                ? ui.formatSuggestCoalesceUserStatus
                : null;
            let line = statusFn
              ? statusFn(coalesce)
              : coalesce && coalesce.error
                ? 'Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics.'
                : 'Loaded validated role-grounded SuggestDraftV1 JSON.';
            const diag = coalesce && coalesce.diagnostics ? coalesce.diagnostics : {};
            if (coalesce && coalesce.error && diag.route_used != null) {
              line += ` · route=${diag.route_used}`;
            }
            if (
              coalesce &&
              coalesce.error &&
              Array.isArray(diag.validation_errors) &&
              diag.validation_errors.length
            ) {
              line += ` · validation_errors=${diag.validation_errors.join(',')}`;
            }
            statusEl.textContent = line;
          }
        } catch (err) {
          if (statusEl) {
            if (err && err.name === 'AbortError') {
              const fetchSec = Math.round(memoryGraphSuggestFetchTimeoutMs() / 1000);
              statusEl.textContent = `Suggest timed out after ${fetchSec}s (browser fetch limit; Hub may still be running Quick then Brain). Retry when the gateway is responsive, or set MEMORY_GRAPH_SUGGEST_CLIENT_FETCH_TIMEOUT_MS higher. Reduce selected turns only if the prompt was clipped.`;
            } else {
              statusEl.textContent = String(err.message || err);
            }
          }
        } finally {
          suggestBtn.disabled = false;
          suggestBtn.textContent = prevLabel;
        }
      });
    }
    if (toMemBtn) {
      toMemBtn.addEventListener('click', () => {
        flushMemoryGraphBridgeDraftForm();
        const raw = document.getElementById('memoryGraphBridgeDraft')?.value || '';
        if (!String(raw).trim()) {
          showToast('Add or generate a draft first (run Suggest draft), then continue.');
          return;
        }
        sessionStorage.setItem('orion_memory_graph_draft_import', raw);
        if (memoryGraphBridgeDraftForm && typeof memoryGraphBridgeDraftForm.buildCardProjectionPayload === 'function') {
          sessionStorage.setItem(
            'orion_memory_graph_card_defaults_import',
            JSON.stringify(memoryGraphBridgeDraftForm.buildCardProjectionPayload()),
          );
        }
        if (memoryTabButton) {
          memoryTabButton.click();
        } else {
          setActiveTab('memory');
          history.replaceState(null, '', '#memory');
        }
        window.dispatchEvent(new CustomEvent('orion-hub-memory-graph-draft-import', { detail: { source: 'bridge' } }));
        closeMemoryGraphBridgeModal();
      });
    }
  }

  function closeAgentTraceModal() {
    if (!agentTraceModal) return;
    agentTraceModal.classList.add('hidden');
    agentTraceModal.setAttribute('aria-hidden', 'true');
    syncDebugModalScrollLock();
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
      `selected:${model.selectedSubject}`,
      `state:${model.stateQuality || 'unknown'}`,
      `availability:${model.availability.available}/${model.availability.subjects}`,
      `repo:${model.repositoryStatus.source_available ? 'up' : 'down'}`,
    ].forEach((value) => {
      const badge = document.createElement('span');
      badge.className = 'rounded-full border border-violet-400/40 bg-violet-500/15 px-2 py-0.5 text-[10px] text-violet-100';
      badge.textContent = value;
      badges.appendChild(badge);
    });
    headerRow.appendChild(badges);
    panel.appendChild(headerRow);

    if (!model.hasSemanticSignal && (model.hasDebugSignal || model.hasPreviewSignal || model.hasV2PreviewSignal || model.hasChatStanceDebugSignal)) {
      const sparseNote = document.createElement('div');
      sparseNote.className = 'text-xs text-violet-100/90';
      sparseNote.textContent = 'Autonomy observed, semantic summary unavailable';
      panel.appendChild(sparseNote);
    }

    [
      ['dominant drive', formatAutonomyFieldLabel(model, 'dominantDrive')],
      ['top drives', formatAutonomyFieldLabel(model, 'topDrives')],
      ['top tensions', formatAutonomyFieldLabel(model, 'tensions')],
      ...(model.degradedReason ? [['degraded reason', model.degradedReason]] : []),
      ...(model.contextNote ? [['context note', model.contextNote]] : []),
      ...(model.driveCompetition && model.driveCompetition.top_drive && model.driveCompetition.runner_drive
        ? [['competing pressures', `${model.driveCompetition.top_drive} ${Number(model.driveCompetition.pressure_top).toFixed(2)} vs ${model.driveCompetition.runner_drive} ${Number(model.driveCompetition.pressure_runner).toFixed(2)} (spread ${Number(model.driveCompetition.spread).toFixed(2)})`]]
        : []),
      ['proposal headlines', model.proposalHeadlines.length ? model.proposalHeadlines.join('; ') : '--'],
      ['alignment', (model.alignment && model.alignment.alignment_note) || '--'],
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
      syncDebugModalScrollLock();
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
    syncDebugModalScrollLock();
  }

  async function loadNotifications() {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/notifications?limit=50`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data)) {
        const firstHydrate = !notificationsInitialHydrateDone;
        notificationsInitialHydrateDone = true;
        notifications = data;
        renderNotifications();
        if (firstHydrate && data.length >= notificationBatchHydrateThreshold()) {
          showToastText(`${data.length} notifications loaded (see tray)`);
        }
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

  function createWorldPulseBlock(title, value) {
    if (value === null || value === undefined || value === '' || (Array.isArray(value) && !value.length)) return null;
    const wrap = document.createElement('div');
    wrap.className = 'rounded border border-gray-800 bg-gray-950/40 p-2';
    const label = document.createElement('div');
    label.className = 'text-[10px] uppercase tracking-wide text-gray-500';
    label.textContent = title;
    wrap.appendChild(label);
    if (Array.isArray(value)) {
      const list = document.createElement('ul');
      list.className = 'mt-1 list-disc pl-4 space-y-0.5';
      value.forEach((entry) => {
        if (entry === null || entry === undefined || entry === '') return;
        const li = document.createElement('li');
        li.textContent = String(entry);
        list.appendChild(li);
      });
      if (list.children.length) wrap.appendChild(list);
      return list.children.length ? wrap : null;
    }
    const body = document.createElement('div');
    body.className = 'mt-1 text-gray-200 whitespace-pre-wrap';
    body.textContent = String(value);
    wrap.appendChild(body);
    return wrap;
  }

  function worldPulseSourceCount(item) {
    return Array.isArray(item?.source_ids) ? item.source_ids.length : null;
  }

  function normalizeWorldPulsePayload(raw) {
    const payload = raw && typeof raw === 'object' ? raw : {};
    if (payload.digest && payload.run) {
      return {
        run: payload.run || {},
        digest: payload.digest || {},
        source: 'latest',
      };
    }
    const structured = payload.structured_payload && typeof payload.structured_payload === 'object'
      ? payload.structured_payload
      : {};
    const digestFromMessage = structured && Object.keys(structured).length
      ? structured
      : {
          run_id: payload.run_id,
          date: payload.date,
          title: payload.title || 'Daily World Pulse',
          executive_summary: payload.executive_summary || '',
          items: Array.isArray(payload.cards) ? payload.cards : [],
          things_worth_reading: Array.isArray(payload.worth_reading) ? payload.worth_reading : [],
          things_worth_watching: Array.isArray(payload.worth_watching) ? payload.worth_watching : [],
        };
    return {
      run: {
        run_id: payload.run_id || digestFromMessage.run_id || 'unknown',
        status: payload.status || 'published',
      },
      digest: digestFromMessage,
      source: 'message',
    };
  }

  function computeEffectiveWorldPulseMetrics(model) {
    const digest = model?.digest && typeof model.digest === 'object' ? model.digest : {};
    const run = model?.run && typeof model.run === 'object' ? model.run : {};
    const items = Array.isArray(digest.items) ? digest.items : [];
    const rollups = Array.isArray(digest.section_rollups) ? digest.section_rollups : [];
    const sectionCoverage = digest.section_coverage && typeof digest.section_coverage === 'object'
      ? { ...digest.section_coverage }
      : {};

    const articleIds = new Set();
    const sourceIds = new Set(Array.isArray(digest.source_ids) ? digest.source_ids.filter(Boolean) : []);
    const inferredDigestCounts = {};
    items.forEach((item) => {
      (Array.isArray(item?.article_ids) ? item.article_ids : []).forEach((id) => {
        if (id) articleIds.add(id);
      });
      (Array.isArray(item?.source_ids) ? item.source_ids : []).forEach((id) => {
        if (id) sourceIds.add(id);
      });
      const category = item?.category;
      if (!category) return;
      inferredDigestCounts[category] = (inferredDigestCounts[category] || 0) + 1;
      if (!sectionCoverage[category]) {
        sectionCoverage[category] = {
          status: 'covered',
          articles_accepted: Array.isArray(item?.article_ids) ? item.article_ids.length : 0,
          digest_items: 1,
        };
      }
    });

    const rollupBySection = {};
    rollups.forEach((r) => {
      if (!r || !r.section) return;
      rollupBySection[r.section] = r;
      if (!sectionCoverage[r.section]) {
        sectionCoverage[r.section] = {
          status: r.status || 'missing',
          articles_accepted: Number(r.article_count || 0),
          digest_items: Number(r.digest_item_count || 0),
          cluster_count: Number(r.cluster_count || 0),
        };
      }
    });

    const rollupArticleSum = rollups.reduce((sum, r) => sum + Number((r && r.article_count) || 0), 0);
    const rollupClusterSum = rollups.reduce((sum, r) => sum + Number((r && r.cluster_count) || 0), 0);
    const acceptedArticleCount = Number.isFinite(Number(digest.accepted_article_count)) && Number(digest.accepted_article_count) > 0
      ? Number(digest.accepted_article_count)
      : (Number.isFinite(Number(run.articles_accepted)) && Number(run.articles_accepted) > 0
          ? Number(run.articles_accepted)
          : (articleIds.size || rollupArticleSum || 0));
    const articleClusterCount = Number.isFinite(Number(digest.article_cluster_count)) && Number(digest.article_cluster_count) > 0
      ? Number(digest.article_cluster_count)
      : (Number.isFinite(Number(run?.metrics?.article_clusters)) && Number(run.metrics.article_clusters) > 0
          ? Number(run.metrics.article_clusters)
          : (rollupClusterSum || 0));
    const maxDigestItemsTotal = Number.isFinite(Number(digest.max_digest_items_total)) && Number(digest.max_digest_items_total) > 0
      ? Number(digest.max_digest_items_total)
      : Math.max(12, items.length);

    const derivedCoverageStatus = (() => {
      if (digest.coverage_status) return digest.coverage_status;
      if (!Object.keys(sectionCoverage).length) return 'unknown';
      const requiredCovered = WORLD_PULSE_REQUIRED_SECTIONS.every((section) => {
        const row = sectionCoverage[section] || {};
        const status = row.status || (inferredDigestCounts[section] > 0 ? 'covered' : 'missing');
        return status === 'covered';
      });
      const anyCovered = Object.values(sectionCoverage).some((row) => row && row.status === 'covered');
      if (requiredCovered) return 'complete';
      return anyCovered ? 'partial' : 'empty';
    })();

    return {
      acceptedArticleCount,
      articleClusterCount,
      maxDigestItemsTotal,
      coverageStatus: derivedCoverageStatus,
      sectionCoverage,
      sourceIds: Array.from(sourceIds),
      items,
      digest,
      run,
      rollupBySection,
    };
  }

  function renderWorldPulseDetails(model) {
    if (!worldPulseDetails) return;
    worldPulseDetails.innerHTML = '';
    const metrics = computeEffectiveWorldPulseMetrics(model);
    const digest = metrics.digest;
    const run = metrics.run;
    if (!digest || !Object.keys(digest).length) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-gray-500';
      empty.textContent = 'No digest payload available.';
      worldPulseDetails.appendChild(empty);
      return;
    }

    const coverage = metrics.sectionCoverage;
    const coverageBox = document.createElement('div');
    coverageBox.className = 'rounded border border-gray-800 bg-gray-950/40 p-2 space-y-2';
    const coverageTitle = document.createElement('div');
    coverageTitle.className = 'text-[10px] uppercase tracking-wide text-gray-500';
    coverageTitle.textContent = 'Section Coverage';
    coverageBox.appendChild(coverageTitle);
    const sectionGroups = [
      { label: 'Required', sections: WORLD_PULSE_REQUIRED_SECTIONS },
      { label: 'Recommended', sections: WORLD_PULSE_RECOMMENDED_SECTIONS },
    ];
    sectionGroups.forEach((group) => {
      const groupTitle = document.createElement('div');
      groupTitle.className = 'text-[11px] font-semibold text-gray-300';
      groupTitle.textContent = group.label;
      coverageBox.appendChild(groupTitle);
      const list = document.createElement('ul');
      list.className = 'space-y-1';
      group.sections.forEach((section) => {
        const c = coverage[section] || {};
        const fallbackRollup = metrics.rollupBySection[section] || {};
        const inferredDigest = metrics.items.filter((item) => item?.category === section).length;
        const li = document.createElement('li');
        const status = c.status || fallbackRollup.status || (inferredDigest > 0 ? 'covered' : 'missing');
        const articlesAccepted = c.articles_accepted ?? fallbackRollup.article_count ?? 0;
        const digestItems = c.digest_items ?? fallbackRollup.digest_item_count ?? inferredDigest;
        const clusters = c.cluster_count ?? fallbackRollup.cluster_count;
        li.className = 'text-xs text-gray-300';
        li.textContent = `${section}: ${status} · articles=${articlesAccepted} · digest=${digestItems}${clusters !== undefined ? ` · clusters=${clusters}` : ''}`;
        list.appendChild(li);
      });
      coverageBox.appendChild(list);
    });
    worldPulseDetails.appendChild(coverageBox);

    if (Array.isArray(digest.items) && digest.items.length) {
      const cardsWrap = document.createElement('div');
      cardsWrap.className = 'space-y-2';
      const cardsTitle = document.createElement('div');
      cardsTitle.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      cardsTitle.textContent = `Digest Cards (${digest.items.length})`;
      cardsWrap.appendChild(cardsTitle);
      digest.items.forEach((item) => {
        const details = document.createElement('details');
        details.className = 'rounded border border-gray-800 bg-gray-950/40 p-2';
        const summary = document.createElement('summary');
        summary.className = 'cursor-pointer text-xs text-gray-200';
        const srcCount = worldPulseSourceCount(item);
        const articleCount = Array.isArray(item?.article_ids) ? item.article_ids.length : null;
        const badgeBits = [
          item?.category || 'unknown',
          item?.confidence !== undefined ? `conf=${item.confidence}` : null,
          item?.volatility ? `vol=${item.volatility}` : null,
          srcCount !== null ? `sources=${srcCount}` : null,
          articleCount !== null ? `articles=${articleCount}` : null,
        ].filter(Boolean);
        summary.textContent = `${item?.title || 'Untitled'} · ${badgeBits.join(' · ')}`;
        details.appendChild(summary);
        const blocks = [
          createWorldPulseBlock('Summary', item?.summary),
          createWorldPulseBlock('Why It Matters', item?.why_it_matters),
          createWorldPulseBlock('What Changed', item?.what_changed),
          createWorldPulseBlock('Context', item?.context_bullets),
          createWorldPulseBlock('By The Numbers', item?.by_the_numbers),
          createWorldPulseBlock("What They're Saying", item?.what_theyre_saying),
          createWorldPulseBlock('Caveats', item?.caveats),
          createWorldPulseBlock("Orion's Read", item?.orion_read),
          createWorldPulseBlock('What To Watch', item?.what_to_watch),
          createWorldPulseBlock('Worth Reading', item?.worth_reading),
          createWorldPulseBlock('Sources', item?.source_ids),
          createWorldPulseBlock('Article IDs', item?.article_ids),
        ];
        blocks.filter(Boolean).forEach((b) => details.appendChild(b));
        cardsWrap.appendChild(details);
      });
      worldPulseDetails.appendChild(cardsWrap);
    }

    const worthReading = digest.things_worth_reading || digest.worth_reading || [];
    const worthWatching = digest.things_worth_watching || digest.worth_watching || [];
    const worthReadingSection = document.createElement('details');
    worthReadingSection.className = 'rounded border border-gray-800 bg-gray-950/40 p-2';
    worthReadingSection.open = true;
    worthReadingSection.innerHTML = `<summary class="cursor-pointer text-xs text-gray-200">Worth Reading (${worthReading.length})</summary>`;
    if (worthReading.length) {
      worthReading.forEach((w) => {
        const row = document.createElement('div');
        row.className = 'mt-2 border-t border-gray-800 pt-2';
        const line = [w.title, w.source_id ? `source=${w.source_id}` : null, w.reading_type ? `type=${w.reading_type}` : null, w.trust_tier !== undefined ? `trust=${w.trust_tier}` : null].filter(Boolean).join(' · ');
        const itemBlock = createWorldPulseBlock('Item', line);
        const reasonBlock = createWorldPulseBlock('Reason', w.reason_selected);
        if (itemBlock) row.appendChild(itemBlock);
        if (reasonBlock) row.appendChild(reasonBlock);
        if (w.url) {
          const link = document.createElement('a');
          link.href = w.url;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.className = 'text-indigo-300 hover:underline';
          link.textContent = w.url;
          const linkWrap = document.createElement('div');
          linkWrap.className = 'mt-1';
          linkWrap.appendChild(link);
          row.appendChild(linkWrap);
        }
        worthReadingSection.appendChild(row);
      });
    } else {
      const none = document.createElement('div');
      none.className = 'mt-2 text-gray-500';
      none.textContent = 'No worth-reading items.';
      worthReadingSection.appendChild(none);
    }
    worldPulseDetails.appendChild(worthReadingSection);

    const worthWatchingSection = document.createElement('details');
    worthWatchingSection.className = 'rounded border border-gray-800 bg-gray-950/40 p-2';
    worthWatchingSection.open = true;
    worthWatchingSection.innerHTML = `<summary class="cursor-pointer text-xs text-gray-200">Things Worth Watching (${worthWatching.length})</summary>`;
    if (worthWatching.length) {
      worthWatching.forEach((w) => {
        const row = document.createElement('div');
        row.className = 'mt-2 border-t border-gray-800 pt-2';
        const line = [w.title || w.topic_id || 'watch item', w.category ? `category=${w.category}` : null, w.confidence !== undefined ? `conf=${w.confidence}` : null, w.volatility ? `vol=${w.volatility}` : null].filter(Boolean).join(' · ');
        const itemBlock = createWorldPulseBlock('Item', line);
        const reasonBlock = createWorldPulseBlock('Reason', w.reason);
        const conditionBlock = createWorldPulseBlock('Watch Condition', w.watch_condition);
        const recheckBlock = createWorldPulseBlock('Recheck After', w.recheck_after);
        if (itemBlock) row.appendChild(itemBlock);
        if (reasonBlock) row.appendChild(reasonBlock);
        if (conditionBlock) row.appendChild(conditionBlock);
        if (recheckBlock) row.appendChild(recheckBlock);
        worthWatchingSection.appendChild(row);
      });
    } else {
      const none = document.createElement('div');
      none.className = 'mt-2 text-gray-500';
      none.textContent = 'No worth-watching items.';
      worthWatchingSection.appendChild(none);
    }
    worldPulseDetails.appendChild(worthWatchingSection);

    if (Array.isArray(digest.section_rollups) && digest.section_rollups.length) {
      const rollups = document.createElement('div');
      rollups.className = 'space-y-2';
      const title = document.createElement('div');
      title.className = 'text-[10px] uppercase tracking-wide text-gray-500';
      title.textContent = 'Section Rollups';
      rollups.appendChild(title);
      digest.section_rollups.forEach((r) => {
        const det = document.createElement('details');
        det.className = 'rounded border border-gray-800 bg-gray-950/40 p-2';
        det.innerHTML = `<summary class="cursor-pointer text-xs text-gray-200">${r.section} · ${r.status} · articles=${r.article_count ?? 0} · clusters=${r.cluster_count ?? 0} · digest=${r.digest_item_count ?? 0}</summary>`;
        [createWorldPulseBlock('Summary', r.summary), createWorldPulseBlock('Source Notes', r.source_notes), createWorldPulseBlock('Confidence', r.confidence)].filter(Boolean).forEach((b) => det.appendChild(b));
        rollups.appendChild(det);
      });
      worldPulseDetails.appendChild(rollups);
    }

    const evidence = document.createElement('details');
    evidence.className = 'rounded border border-gray-800 bg-gray-950/40 p-2';
    evidence.innerHTML = '<summary class="cursor-pointer text-xs text-gray-200">Evidence / Sources</summary>';
    [
      createWorldPulseBlock('Accepted Article Count', metrics.acceptedArticleCount),
      createWorldPulseBlock('Article Cluster Count', metrics.articleClusterCount),
      createWorldPulseBlock('Source Notes', digest.source_notes),
      createWorldPulseBlock('Source IDs', (Array.isArray(digest.source_ids) && digest.source_ids.length) ? digest.source_ids : metrics.sourceIds),
    ].filter(Boolean).forEach((b) => evidence.appendChild(b));
    worldPulseDetails.appendChild(evidence);

    const debug = document.createElement('details');
    debug.className = 'rounded border border-gray-800 bg-gray-950/40 p-2';
    debug.innerHTML = '<summary class="cursor-pointer text-xs text-gray-400">Debug JSON</summary>';
    const pre = document.createElement('pre');
    pre.className = 'mt-2 max-h-60 overflow-auto whitespace-pre-wrap text-[10px] text-gray-400';
    pre.textContent = JSON.stringify({ run, digest, source: model.source }, null, 2);
    debug.appendChild(pre);
    worldPulseDetails.appendChild(debug);
  }

  async function loadWorldPulseLatest() {
    if (!worldPulseStatus || !worldPulseSummary) return;
    try {
      const resp = await fetch(`${API_BASE_URL}/api/world-pulse/latest`);
      if (!resp.ok) {
        worldPulseStatus.textContent = resp.status === 502
          ? 'World Pulse service unavailable.'
          : 'No world pulse run available.';
        worldPulseSummary.textContent = '';
        if (worldPulseDetails) worldPulseDetails.innerHTML = '';
        return;
      }
      const data = normalizeWorldPulsePayload(await resp.json());
      const metrics = computeEffectiveWorldPulseMetrics(data);
      const digest = metrics.digest;
      const run = metrics.run;
      const generated = digest?.generated_at || digest?.date || run?.date || '--';
      worldPulseStatus.textContent = `Run ${run?.run_id || 'unknown'} • ${run?.status || 'unknown'} • coverage ${metrics.coverageStatus || 'unknown'}`;
      worldPulseSummary.textContent = [
        `${digest?.title || 'Daily World Pulse'}`,
        `${digest?.executive_summary || ''}`,
        `Generated: ${generated}`,
        `Evidence: ${metrics.acceptedArticleCount} accepted articles / ${metrics.articleClusterCount} clusters`,
        `Curated cards: ${(digest?.items || []).length} (cap: ${metrics.maxDigestItemsTotal ?? '--'})`,
      ].filter(Boolean).join('\n');
      renderWorldPulseDetails(data);
    } catch (err) {
      worldPulseStatus.textContent = 'Failed to load world pulse.';
      worldPulseSummary.textContent = '';
      if (worldPulseDetails) worldPulseDetails.innerHTML = '';
      console.warn('Failed to load world pulse latest', err);
    }
  }

  async function triggerWorldPulseRun() {
    if (!worldPulseStatus) return;
    worldPulseStatus.textContent = 'Starting world pulse run...';
    try {
      const resp = await fetch(`${API_BASE_URL}/api/world-pulse/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dry_run: true,
          requested_by: 'hub',
          ...(worldPulseFixtureRunEnabled ? { fixtures: true } : {}),
        }),
      });
      if (!resp.ok) {
        worldPulseStatus.textContent = 'World pulse run failed.';
        return;
      }
      await loadWorldPulseLatest();
    } catch (err) {
      worldPulseStatus.textContent = 'World pulse run failed.';
      console.warn('Failed to trigger world pulse run', err);
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

  // --- 3. Event Listeners ---

  if (recordButton) {
    recordButton.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      try {
        recordButton.setPointerCapture(e.pointerId);
      } catch (_) {
        /* ignore */
      }
      startRecording();
    });
    recordButton.addEventListener('pointerup', (e) => {
      e.preventDefault();
      try {
        if (recordButton.hasPointerCapture(e.pointerId)) {
          recordButton.releasePointerCapture(e.pointerId);
        }
      } catch (_) {
        /* ignore */
      }
      stopRecording();
    });
    recordButton.addEventListener('pointercancel', () => stopRecording());
  }

  if (sendButton) sendButton.addEventListener('click', sendTextMessage);
  if (chatInput) {
    chatInput.addEventListener('input', () => {
      if (!chatInputExpandModalRoot || chatInputExpandModalRoot.classList.contains('hidden')) return;
      syncChatExpandTextareaFromInput();
    });
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
          e.preventDefault(); 
          sendTextMessage();
      }
    });
  }
  if (chatInputExpandButton) {
    chatInputExpandButton.addEventListener('click', () => openChatInputExpandModal());
  }
  if (chatInputExpandTextarea) {
    chatInputExpandTextarea.addEventListener('input', syncChatInputFromExpandTextarea);
    chatInputExpandTextarea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        sendExpandedChatMessage();
      }
    });
  }
  if (chatInputExpandModalClose) {
    chatInputExpandModalClose.addEventListener('click', () => closeChatInputExpandModal());
  }
  if (chatInputExpandModalApply) {
    chatInputExpandModalApply.addEventListener('click', () => closeChatInputExpandModal());
  }
  if (chatInputExpandModalSend) {
    chatInputExpandModalSend.addEventListener('click', () => {
      sendExpandedChatMessage();
    });
  }
  if (chatInputExpandModalBackdrop) {
    chatInputExpandModalBackdrop.addEventListener('click', () => closeChatInputExpandModal());
  }
  if (chatInputExpandModalRoot) {
    chatInputExpandModalRoot.addEventListener('click', (event) => {
      if (event.target === chatInputExpandModalRoot) closeChatInputExpandModal();
    });
  }
  if (chatInputExpandModalDialog) {
    chatInputExpandModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }

  function getSkillRunnerSelection() {
    if (!skillRunnerSelect || !skillRunnerSelect.value) {
      return { prompt: '', workflowId: null };
    }
    const selectedOption = skillRunnerSelect.options[skillRunnerSelect.selectedIndex];
    const workflowId = selectedOption ? String(selectedOption.dataset.workflowId || '').trim() : '';
    return {
      prompt: String(skillRunnerSelect.value),
      workflowId: workflowId || null,
    };
  }
  function getSkillRunnerLaneOptions({ workflowId } = {}) {
    // Catalogue prompts: deterministic lane (catalogue skills.* only; hub ignores chat UI mode for this send).
    if (!workflowId) {
      return {
        verbs: [],
        skillRunnerOrigin: true,
        skillRunnerLane: 'deterministic',
      };
    }
    const laneMode = String(currentMode || 'brain').toLowerCase();
    const verbOverrideRaw = modeVerbOverride ? String(modeVerbOverride).trim() : '';
    const verbOverride = verbOverrideRaw.toLowerCase();
    if (verbOverride === 'chat_quick') {
      return {
        mode: 'brain',
        verbs: ['chat_quick'],
        skillRunnerOrigin: true,
        skillRunnerLane: 'quick',
        options: { chat_quick_full_stance: chatQuickVariant === 'stance' },
      };
    }
    if (verbOverride === 'chat_kids_story') {
      return {
        mode: 'brain',
        verbs: ['chat_kids_story'],
        skillRunnerOrigin: true,
        skillRunnerLane: 'quick',
      };
    }
    if (laneMode === 'agent') {
      return {
        mode: 'agent',
        verbs: [],
        skillRunnerOrigin: true,
        skillRunnerLane: 'agent',
      };
    }
    return {
      mode: 'brain',
      verbs: [],
      skillRunnerOrigin: true,
      skillRunnerLane: 'brain',
    };
  }
  if (skillRunnerInsertBtn && skillRunnerSelect && chatInput) {
    skillRunnerInsertBtn.addEventListener('click', () => {
      const { prompt: promptText } = getSkillRunnerSelection();
      if (!promptText) return;
      chatInput.value = promptText;
      chatInput.focus();
    });
  }
  if (skillRunnerRunBtn && skillRunnerSelect) {
    skillRunnerRunBtn.addEventListener('click', async () => {
      const { prompt: promptText, workflowId } = getSkillRunnerSelection();
      if (!promptText) return;
      const runOptions = getSkillRunnerLaneOptions({ workflowId });
      if (workflowId) {
        runOptions.workflowRequestOverride = { workflow_id: workflowId };
      }
      await submitExplicitChatText(promptText, runOptions);
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
      clearMemoryDebugPanel();
      clearAgentTraceDebugPanel();
      clearAutonomyDebugPanel();
      clearChatStanceDebugPanel();
      clearSubstrateReviewDebugPanel();
      clearSelfExperimentsDebugPanel();
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
  function closeQuickModeMenu() {
    const menu = document.getElementById('quickModeMenu');
    const chev = document.getElementById('quickModeMenuBtn');
    if (menu) menu.classList.add('hidden');
    if (chev) chev.setAttribute('aria-expanded', 'false');
  }
  function syncQuickMainButtonLabel() {
    const b = document.getElementById('quickModeBtn');
    if (!b) return;
    if (chatQuickVariant === 'stance') {
      b.textContent = 'Quick+';
      b.title = 'Quick + stance (full brain prep)';
    } else {
      b.textContent = 'Quick';
      b.title = 'Quick (fast lane)';
    }
  }
  function applyModeButtonSelection(selectedBtn) {
    modeButtons.forEach((b) => {
      b.classList.remove('bg-indigo-600', 'text-white', 'mode-btn-active');
      if (b.classList.contains('mode-btn-quick')) {
        b.classList.add('mode-btn-quick');
      } else {
        b.classList.add('bg-gray-700', 'text-gray-200');
      }
    });
    const qChevron = document.getElementById('quickModeMenuBtn');
    if (qChevron) {
      qChevron.classList.remove('bg-indigo-600', 'text-white');
      qChevron.classList.add('bg-indigo-500/30', 'text-indigo-100');
    }
    closeQuickModeMenu();
    if (selectedBtn) {
      selectedBtn.classList.add('mode-btn-active', 'text-white');
      selectedBtn.classList.remove('bg-gray-700', 'text-gray-200');
      if (selectedBtn.id === 'quickModeBtn' && qChevron) {
        qChevron.classList.add('bg-indigo-600', 'text-white');
        qChevron.classList.remove('bg-indigo-500/30', 'text-indigo-100');
      }
    }
  }
  modeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      currentMode = btn.dataset.mode || 'brain';
      modeVerbOverride = btn.dataset.verbOverride || null;
      applyModeButtonSelection(btn);
      const modeLabel = modeVerbOverride ? `${currentMode} (${modeVerbOverride})` : currentMode;
      updateStatus(`Switched to ${modeLabel} mode.`);
    });
  });
  const quickMenuBtn = document.getElementById('quickModeMenuBtn');
  const quickModeMenu = document.getElementById('quickModeMenu');
  if (quickMenuBtn && quickModeMenu) {
    quickMenuBtn.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      quickModeMenu.classList.toggle('hidden');
      quickMenuBtn.setAttribute('aria-expanded', String(!quickModeMenu.classList.contains('hidden')));
    });
    document.querySelectorAll('.quick-variant-item').forEach((item) => {
      item.addEventListener('click', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const v = String(item.getAttribute('data-quick-variant') || '').trim();
        if (v !== 'fast' && v !== 'stance') return;
        chatQuickVariant = v;
        currentMode = 'brain';
        modeVerbOverride = 'chat_quick';
        syncQuickMainButtonLabel();
        const qMain = document.getElementById('quickModeBtn');
        applyModeButtonSelection(qMain);
        updateStatus(v === 'stance' ? 'Quick + stance (full prep).' : 'Quick (fast lane).');
      });
    });
    document.addEventListener('mousedown', (e) => {
      const g = document.getElementById('quickModeGroup');
      if (!g || !quickModeMenu || quickModeMenu.classList.contains('hidden')) return;
      if (g.contains(e.target)) return;
      closeQuickModeMenu();
    });
  }
  const defaultModeButton = Array.from(modeButtons).find((btn) =>
    (btn.dataset.mode || 'brain') === currentMode
    && (btn.dataset.verbOverride || null) === modeVerbOverride
  )
    || Array.from(modeButtons).find((btn) => (btn.dataset.mode || 'brain') === currentMode && !btn.dataset.verbOverride)
    || modeButtons[0];
  applyModeButtonSelection(defaultModeButton);
  syncQuickMainButtonLabel();

  function applyLlmRouteButtonSelection(routeId) {
    const rid = String(routeId || 'chat').toLowerCase();
    document.querySelectorAll('.llm-route-btn').forEach((btn) => {
      const btnRoute = String(btn.dataset.llmRoute || 'chat').toLowerCase();
      btn.classList.remove('bg-indigo-600', 'text-white');
      btn.classList.add('bg-gray-700', 'text-gray-200', 'hover:bg-gray-600');
      if (btnRoute === rid) {
        btn.classList.add('bg-indigo-600', 'text-white');
        btn.classList.remove('bg-gray-700', 'text-gray-200', 'hover:bg-gray-600');
      }
    });
    const entry = (llmRouteCatalog.routes || []).find((r) => String(r.id || '').toLowerCase() === rid);
    const metaEl = document.getElementById('llmRouteMeta');
    if (metaEl) {
      if (!entry) metaEl.textContent = '—';
      else {
        metaEl.textContent = `${entry.served_by || '—'} · ${entry.backend || '—'} · ${entry.status || 'unknown'}`;
        metaEl.title = `served_by=${entry.served_by || '—'} backend=${entry.backend || '—'} status=${entry.status || 'unknown'}`;
      }
    }
  }

  function routeStatusIsDown(routeId) {
    const rid = String(routeId || '').toLowerCase();
    const entry = (llmRouteCatalog.routes || []).find((r) => String(r.id || '').toLowerCase() === rid);
    if (!entry) return false;
    return entry.status === 'down' || entry.status === 'not_configured';
  }

  async function confirmDownRouteOrProceed(routeId) {
    const rid = String(routeId || 'chat').toLowerCase();
    if (!routeStatusIsDown(rid)) return rid;
    const entry = (llmRouteCatalog.routes || []).find((r) => String(r.id || '').toLowerCase() === rid) || { id: rid };
    const detail = `${entry.served_by || '—'} / ${entry.backend || '—'} / ${entry.status || 'down'}`;
    const useChat = window.confirm(
      `Route "${rid}" is unavailable (${detail}).\n\nOK = Use chat\nCancel = more options`
    );
    if (useChat) return 'chat';
    const tryAnyway = window.confirm(`Try "${rid}" anyway? Cancel aborts the send.`);
    if (tryAnyway) return rid;
    return null;
  }

  async function loadLlmRouteCatalog() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/llm-routes`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      llmRouteCatalog = await res.json();
      const known = new Set((llmRouteCatalog.routes || []).map((r) => String(r.id || '').toLowerCase()));
      if (!known.has(String(selectedLlmRoute || '').toLowerCase())) {
        selectedLlmRoute = String(llmRouteCatalog.default_route || 'chat').toLowerCase();
      }
    } catch (err) {
      console.warn('[LLM routes] catalog load failed', err);
    }
    applyLlmRouteButtonSelection(selectedLlmRoute || 'chat');
  }

  document.querySelectorAll('.llm-route-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const routeId = String(btn.dataset.llmRoute || 'chat').toLowerCase();
      selectedLlmRoute = routeId;
      localStorage.setItem('orion_llm_route', routeId);
      applyLlmRouteButtonSelection(routeId);
      updateStatus(`LLM route: ${routeId}`);
    });
  });
  loadLlmRouteCatalog();

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
  applyMindPrefsToControls();
  if (mindRefreshButton) {
    mindRefreshButton.addEventListener("click", () => {
      refreshMindRuns();
    });
  }
  [mindHoursInput, mindFilterOk, mindFilterTrigger, mindFilterErrorCode, mindFilterRouterProfileId].forEach((el) => {
    if (!el) return;
    el.addEventListener("change", () => {
      persistMindPrefsFromControls();
      if (window.location.hash === "#mind") refreshMindRuns();
    });
  });
  if (mindDefaultOnSendToggle) {
    mindDefaultOnSendToggle.addEventListener("change", () => {
      persistMindPrefsFromControls();
    });
  }
  if (mindRunsModalClose) {
    mindRunsModalClose.addEventListener("click", closeMindRunsModal);
  }
  if (mindRunsModal) {
    mindRunsModal.addEventListener("click", (event) => {
      if (event.target === mindRunsModal) closeMindRunsModal();
    });
  }
  ensureMemoryDebugModalRootOnBody();
  if (memoryDebugOpenModal) {
    memoryDebugOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openMemoryDebugModal();
    });
  }
  if (memoryDebugModalClose) {
    memoryDebugModalClose.addEventListener('click', closeMemoryDebugModal);
  }
  if (memoryDebugModalBackdrop) {
    memoryDebugModalBackdrop.addEventListener('click', closeMemoryDebugModal);
  }
  if (memoryDebugModalRoot) {
    memoryDebugModalRoot.addEventListener('click', (event) => {
      if (event.target === memoryDebugModalRoot) closeMemoryDebugModal();
    });
  }
  if (memoryDebugModalDialog) {
    memoryDebugModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  if (runtimeDebugPanelToggle) {
    runtimeDebugPanelToggle.addEventListener('click', toggleRuntimeDebugPanel);
  }
  if (agentTraceDebugToggle) {
    agentTraceDebugToggle.addEventListener('click', toggleAgentTraceDebugPanel);
  }
  if (agentTraceDebugOpenModal) {
    agentTraceDebugOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openAgentTraceModal(lastAgentTraceSummary, lastAgentTraceMeta);
    });
  }
  if (autonomyDebugToggle) {
    autonomyDebugToggle.addEventListener('click', toggleAutonomyDebugPanel);
  }
  if (autonomyGoalArchiveDryRun) {
    autonomyGoalArchiveDryRun.addEventListener('click', () => {
      runAutonomyGoalArchive(true).catch((err) => {
        if (typeof showToast === 'function') showToast(`Goal archive failed: ${err && err.message ? err.message : err}`, 'error');
      });
    });
  }
  if (autonomyGoalArchiveApply) {
    autonomyGoalArchiveApply.addEventListener('click', () => {
      if (!window.confirm('Archive stale autonomy goals for orion + relationship?')) return;
      runAutonomyGoalArchive(false).catch((err) => {
        if (typeof showToast === 'function') showToast(`Goal archive failed: ${err && err.message ? err.message : err}`, 'error');
      });
    });
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
  if (chatStanceDebugToggle) {
    chatStanceDebugToggle.addEventListener('click', toggleChatStanceDebugPanel);
  }
  ensureChatStanceModalRootOnBody();
  if (chatStanceDebugOpenModal) {
    chatStanceDebugOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openChatStanceDebugModal();
    });
  }
  if (chatStanceDebugModalClose) {
    chatStanceDebugModalClose.addEventListener('click', closeChatStanceDebugModal);
  }
  if (chatStanceDebugModalBackdrop) {
    chatStanceDebugModalBackdrop.addEventListener('click', closeChatStanceDebugModal);
  }
  if (chatStanceDebugModalRoot) {
    chatStanceDebugModalRoot.addEventListener('click', (event) => {
      if (event.target === chatStanceDebugModalRoot) closeChatStanceDebugModal();
    });
  }
  if (chatStanceDebugModalDialog) {
    chatStanceDebugModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  ensureChatInputExpandModalRootOnBody();
  if (substrateReviewDebugToggle) {
    substrateReviewDebugToggle.addEventListener('click', toggleSubstrateReviewDebugPanel);
  }
  if (selfExperimentsDebugToggle) {
    selfExperimentsDebugToggle.addEventListener('click', toggleSelfExperimentsDebugPanel);
  }
  if (selfExperimentsDebugRefresh) {
    selfExperimentsDebugRefresh.addEventListener('click', async () => {
      try {
        await refreshSelfExperimentsDebugStatus();
      } catch (err) {
        if (selfExperimentsDebugMeta) selfExperimentsDebugMeta.textContent = `Self experiments unavailable: ${String(err.message || err)}`;
      }
    });
  }
  ensureSelfExperimentsModalRootOnBody();
  if (selfExperimentsDebugOpenModal) {
    selfExperimentsDebugOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openSelfExperimentsModal();
    });
  }
  if (selfExperimentsModalClose) {
    selfExperimentsModalClose.addEventListener('click', closeSelfExperimentsModal);
  }
  if (selfExperimentsModalRefresh) {
    selfExperimentsModalRefresh.addEventListener('click', async () => {
      try {
        await refreshSelfExperimentsDebugStatus();
      } catch (err) {
        if (selfExperimentsModalMeta) selfExperimentsModalMeta.textContent = `Self experiments unavailable: ${String(err.message || err)}`;
      }
    });
  }
  if (selfExperimentsModalBackdrop) {
    selfExperimentsModalBackdrop.addEventListener('click', closeSelfExperimentsModal);
  }
  if (selfExperimentsModalRoot) {
    selfExperimentsModalRoot.addEventListener('click', (event) => {
      if (event.target === selfExperimentsModalRoot) closeSelfExperimentsModal();
    });
  }
  if (selfExperimentsModalDialog) {
    selfExperimentsModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  if (selfExperimentsApplyFilters) {
    selfExperimentsApplyFilters.addEventListener('click', async () => {
      try {
        await refreshSelfExperimentsDebugStatus();
      } catch (err) {
        if (selfExperimentsDebugMeta) selfExperimentsDebugMeta.textContent = `Self experiments unavailable: ${String(err.message || err)}`;
      }
    });
  }
  if (selfExperimentsTriggerPulse) {
    selfExperimentsTriggerPulse.addEventListener('click', async () => {
      try {
        await triggerSelfExperimentsDaily('pulse');
      } catch (err) {
        if (selfExperimentsActionStatus) selfExperimentsActionStatus.textContent = `Trigger failed: ${String(err.message || err)}`;
      }
    });
  }
  if (selfExperimentsTriggerMetacog) {
    selfExperimentsTriggerMetacog.addEventListener('click', async () => {
      try {
        await triggerSelfExperimentsDaily('metacog');
      } catch (err) {
        if (selfExperimentsActionStatus) selfExperimentsActionStatus.textContent = `Trigger failed: ${String(err.message || err)}`;
      }
    });
  }
  if (autonomyReadinessToggle) {
    autonomyReadinessToggle.addEventListener('click', toggleAutonomyReadinessPanel);
  }
  if (recallCanaryRunButton) {
    recallCanaryRunButton.addEventListener('click', async () => {
      try {
        await runRecallCanaryQuery();
      } catch (err) {
        setRecallCanaryActionStatus(`Canary query failed: ${toRecallCanaryError(err)}`);
      }
    });
  }
  if (recallCanaryProfileSelect) {
    recallCanaryProfileSelect.addEventListener('change', () => {
      const selectedValue = recallCanaryProfileSelect.value ? String(recallCanaryProfileSelect.value) : '';
      if (!selectedValue) {
        lastRecallCanarySelectedProfile = null;
        localStorage.removeItem(RECALL_CANARY_PROFILE_STORAGE_KEY);
        if (recallCanaryRunButton) {
          recallCanaryRunButton.disabled = true;
          recallCanaryRunButton.classList.add('opacity-50', 'cursor-not-allowed');
        }
        return;
      }
      localStorage.setItem(RECALL_CANARY_PROFILE_STORAGE_KEY, selectedValue);
      if (recallCanaryRunButton) {
        recallCanaryRunButton.disabled = false;
        recallCanaryRunButton.classList.remove('opacity-50', 'cursor-not-allowed');
      }
    });
  }
  if (recallCanaryRecordJudgmentButton) {
    recallCanaryRecordJudgmentButton.addEventListener('click', async () => {
      try {
        await recordRecallCanaryJudgment();
      } catch (err) {
        setRecallCanaryActionStatus(`Record judgment failed: ${toRecallCanaryError(err)}`);
      }
    });
  }
  if (recallCanaryCreateReviewArtifactButton) {
    recallCanaryCreateReviewArtifactButton.addEventListener('click', async () => {
      try {
        await createRecallCanaryReviewArtifact();
      } catch (err) {
        setRecallCanaryActionStatus(`Create review artifact failed: ${toRecallCanaryError(err)}`);
      }
    });
  }
  ensureRecallCanaryModalRootOnBody();
  if (recallCanaryToggle) {
    recallCanaryToggle.addEventListener('click', toggleRecallCanaryPanel);
  }
  if (recallCanaryOpenModal) {
    recallCanaryOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openRecallCanaryModal();
    });
  }
  if (recallCanaryModalClose) {
    recallCanaryModalClose.addEventListener('click', closeRecallCanaryModal);
  }
  if (recallCanaryModalRefresh) {
    recallCanaryModalRefresh.addEventListener('click', async () => {
      try {
        await refreshRecallCanaryModal();
      } catch (err) {
        if (recallCanaryModalStatusMeta) recallCanaryModalStatusMeta.textContent = `Refresh failed: ${toRecallCanaryError(err)}`;
      }
    });
  }
  if (recallCanaryModalBackdrop) {
    recallCanaryModalBackdrop.addEventListener('click', closeRecallCanaryModal);
  }
  if (recallCanaryModalRoot) {
    recallCanaryModalRoot.addEventListener('click', (event) => {
      if (event.target === recallCanaryModalRoot) closeRecallCanaryModal();
    });
  }
  if (recallCanaryModalDialog) {
    recallCanaryModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  ensureCognitiveReviewModalRootOnBody();
  if (cognitiveReviewOpenModal) {
    cognitiveReviewOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openCognitiveReviewModal();
    });
  }
  if (cognitiveReviewModalClose) {
    cognitiveReviewModalClose.addEventListener('click', closeCognitiveReviewModal);
  }
  if (cognitiveReviewModalRefresh) {
    cognitiveReviewModalRefresh.addEventListener('click', async () => {
      try {
        await refreshCognitiveReviewModal();
      } catch (err) {
        if (cognitiveReviewModalStatusMeta) cognitiveReviewModalStatusMeta.textContent = `Refresh failed: ${String(err.message || err)}`;
      }
    });
  }
  if (cognitiveReviewModalBackdrop) {
    cognitiveReviewModalBackdrop.addEventListener('click', closeCognitiveReviewModal);
  }
  if (cognitiveReviewModalRoot) {
    cognitiveReviewModalRoot.addEventListener('click', (event) => {
      if (event.target === cognitiveReviewModalRoot) closeCognitiveReviewModal();
    });
  }
  if (cognitiveReviewModalDialog) {
    cognitiveReviewModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  if (cognitiveReviewModalAcceptDraftButton) {
    cognitiveReviewModalAcceptDraftButton.addEventListener('click', async () => {
      try {
        await submitCognitiveProposalReview('accept_as_draft', 'modal');
      } catch (err) {
        if (cognitiveReviewModalStatusMeta) cognitiveReviewModalStatusMeta.textContent = `Accept draft failed: ${String(err.message || err)}`;
      }
    });
  }
  if (cognitiveReviewModalRejectButton) {
    cognitiveReviewModalRejectButton.addEventListener('click', async () => {
      try {
        await submitCognitiveProposalReview('reject', 'modal');
      } catch (err) {
        if (cognitiveReviewModalStatusMeta) cognitiveReviewModalStatusMeta.textContent = `Reject failed: ${String(err.message || err)}`;
      }
    });
  }
  if (cognitiveReviewModalArchiveButton) {
    cognitiveReviewModalArchiveButton.addEventListener('click', async () => {
      try {
        await submitCognitiveProposalReview('archive', 'modal');
      } catch (err) {
        if (cognitiveReviewModalStatusMeta) cognitiveReviewModalStatusMeta.textContent = `Archive failed: ${String(err.message || err)}`;
      }
    });
  }
  if (cognitiveReviewModalSupersedeButton) {
    cognitiveReviewModalSupersedeButton.addEventListener('click', async () => {
      try {
        await submitCognitiveProposalReview('supersede', 'modal');
      } catch (err) {
        if (cognitiveReviewModalStatusMeta) cognitiveReviewModalStatusMeta.textContent = `Supersede failed: ${String(err.message || err)}`;
      }
    });
  }
  ensureAutonomyConstitutionModalRootOnBody();
  if (autonomyConstitutionOpenModal) {
    autonomyConstitutionOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openAutonomyConstitutionModal();
    });
  }
  if (autonomyConstitutionModalClose) {
    autonomyConstitutionModalClose.addEventListener('click', closeAutonomyConstitutionModal);
  }
  if (autonomyConstitutionModalRefresh) {
    autonomyConstitutionModalRefresh.addEventListener('click', async () => {
      try {
        await refreshAutonomyConstitutionModal();
      } catch (err) {
        if (autonomyConstitutionModalMeta) autonomyConstitutionModalMeta.textContent = `Refresh failed: ${String(err.message || err)}`;
      }
    });
  }
  if (autonomyConstitutionModalBackdrop) {
    autonomyConstitutionModalBackdrop.addEventListener('click', closeAutonomyConstitutionModal);
  }
  if (autonomyConstitutionModalRoot) {
    autonomyConstitutionModalRoot.addEventListener('click', (event) => {
      if (event.target === autonomyConstitutionModalRoot) closeAutonomyConstitutionModal();
    });
  }
  if (autonomyConstitutionModalDialog) {
    autonomyConstitutionModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  ensureSubstrateReviewModalRootOnBody();
  if (substrateReviewDebugOpenModal) {
    substrateReviewDebugOpenModal.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openSubstrateReviewModal();
    });
  }
  if (substrateReviewModalClose) {
    substrateReviewModalClose.addEventListener('click', closeSubstrateReviewModal);
  }
  if (substrateReviewModalBackdrop) {
    substrateReviewModalBackdrop.addEventListener('click', closeSubstrateReviewModal);
  }
  if (substrateReviewModalRoot) {
    substrateReviewModalRoot.addEventListener('click', (event) => {
      if (event.target === substrateReviewModalRoot) closeSubstrateReviewModal();
    });
  }
  if (substrateReviewModalDialog) {
    substrateReviewModalDialog.addEventListener('click', (event) => event.stopPropagation());
  }
  if (substrateReviewActionRefresh) {
    substrateReviewActionRefresh.addEventListener('click', async () => {
      try {
        if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Refreshing…';
        await refreshSubstrateReviewStatus();
        if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = 'Status refreshed.';
      } catch (err) {
        if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = `Refresh failed: ${String(err.message || err)}`;
      }
    });
  }
  if (substrateReviewActionExecuteOnce) {
    substrateReviewActionExecuteOnce.addEventListener('click', async () => {
      try {
        await runSubstrateReviewExecuteOnce();
      } catch (err) {
        if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = `Execute failed: ${String(err.message || err)}`;
      }
    });
  }
  if (substrateReviewActionExecuteFollowup) {
    substrateReviewActionExecuteFollowup.addEventListener('click', async () => {
      try {
        await runSubstrateReviewExecuteOnceWithFollowup();
      } catch (err) {
        if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = `Execute follow-up failed: ${String(err.message || err)}`;
      }
    });
  }
  if (substrateReviewActionSmokeCheck) {
    substrateReviewActionSmokeCheck.addEventListener('click', async () => {
      try {
        await runSubstrateReviewSmokeCheck();
      } catch (err) {
        if (substrateReviewActionStatus) substrateReviewActionStatus.textContent = `Smoke check failed: ${String(err.message || err)}`;
      }
    });
  }
  if (socialInspectionOpen) {
    socialInspectionOpen.addEventListener('click', () => openSocialInspectionModal());
  }
  if (socialInspectionModalClose) {
    socialInspectionModalClose.addEventListener('click', closeSocialInspectionModal);
  }
  if (responseFeedbackModalClose) {
    responseFeedbackModalClose.addEventListener('click', closeResponseFeedbackModal);
  }
  if (responseFeedbackCancel) {
    responseFeedbackCancel.addEventListener('click', closeResponseFeedbackModal);
  }
  if (responseFeedbackSubmit) {
    responseFeedbackSubmit.addEventListener('click', submitResponseFeedback);
  }
  if (responseFeedbackModal) {
    responseFeedbackModal.addEventListener('click', (event) => {
      if (event.target === responseFeedbackModal) closeResponseFeedbackModal();
    });
  }
  setupMemoryGraphBridgeModal();
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
    if (event.key === 'Escape' && memoryGraphBridgeModal && !memoryGraphBridgeModal.classList.contains('hidden')) {
      closeMemoryGraphBridgeModal();
      return;
    }
    if (event.key === 'Escape' && memoryDebugModalRoot && !memoryDebugModalRoot.classList.contains('hidden')) {
      closeMemoryDebugModal();
      return;
    }
    if (event.key === 'Escape' && socialInspectionModal && !socialInspectionModal.classList.contains('hidden')) {
      closeSocialInspectionModal();
      return;
    }
    if (event.key === 'Escape' && mindRunsModal && !mindRunsModal.classList.contains('hidden')) {
      closeMindRunsModal();
      return;
    }
    if (event.key === 'Escape' && responseFeedbackModal && !responseFeedbackModal.classList.contains('hidden')) {
      closeResponseFeedbackModal();
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
    if (event.key === 'Escape' && chatStanceDebugModalRoot && !chatStanceDebugModalRoot.classList.contains('hidden')) {
      closeChatStanceDebugModal();
      return;
    }
    if (event.key === 'Escape' && chatInputExpandModalRoot && !chatInputExpandModalRoot.classList.contains('hidden')) {
      closeChatInputExpandModal();
      return;
    }
    if (event.key === 'Escape' && substrateReviewModalRoot && !substrateReviewModalRoot.classList.contains('hidden')) {
      closeSubstrateReviewModal();
      return;
    }
    if (event.key === 'Escape' && selfExperimentsModalRoot && !selfExperimentsModalRoot.classList.contains('hidden')) {
      closeSelfExperimentsModal();
      return;
    }
    if (event.key === 'Escape' && cognitiveReviewModalRoot && !cognitiveReviewModalRoot.classList.contains('hidden')) {
      closeCognitiveReviewModal();
      return;
    }
    if (event.key === 'Escape' && autonomyConstitutionModalRoot && !autonomyConstitutionModalRoot.classList.contains('hidden')) {
      closeAutonomyConstitutionModal();
      return;
    }
    if (event.key === 'Escape' && recallCanaryModalRoot && !recallCanaryModalRoot.classList.contains('hidden')) {
      closeRecallCanaryModal();
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

  if (textToSpeechToggle) {
    textToSpeechToggle.checked = false;
  }
  normalizeRecallProfileDisplay();
  clearMemoryDebugPanel();
  clearChatStanceDebugPanel();
  clearSubstrateReviewDebugPanel();
  clearSelfExperimentsDebugPanel();
  clearAutonomyReadinessPanel();
  refreshSubstrateReviewStatus().catch((err) => {
    if (substrateReviewDebugMeta) substrateReviewDebugMeta.textContent = `Substrate review status unavailable: ${String(err.message || err)}`;
  });
  refreshSelfExperimentsDebugStatus().catch((err) => {
    if (selfExperimentsDebugMeta) selfExperimentsDebugMeta.textContent = `Self experiments unavailable: ${String(err.message || err)}`;
  });
  refreshAutonomyReadinessPanel().catch((err) => {
    if (autonomyReadinessMeta) autonomyReadinessMeta.textContent = `Autonomy readiness unavailable: ${String(err.message || err)}`;
  });
  refreshRecallCanaryStatus().catch((err) => {
    if (recallCanaryStatusMeta) recallCanaryStatusMeta.textContent = `Recall canary status unavailable: ${String(err.message || err)}`;
  });
  renderSocialInspectionState(null);
  loadResponseFeedbackOptions();
  loadScheduleInventory();

  // --- WebSocket ---
  function safeHubJsonStringify(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch (err) {
      return `{"error":"${String((err && err.message) || err).replace(/"/g, "'")}"}`;
    }
  }

  function resolveAssistantDisplayText(d) {
    if (!d || typeof d !== 'object') return '';
    if (d.mode === 'agent' && d.operator_summary && typeof d.operator_summary === 'object') {
      const op = d.operator_summary;
      const dbg = d.routing_debug && typeof d.routing_debug === 'object' ? d.routing_debug : {};
      const synthesis = dbg.model_synthesis_used ? 'used'
        : (dbg.synthesis_fallback_used || String(dbg.synthesis_fallback_reason || '').startsWith('synthesis') ? 'fallback' : 'skipped');
      const lines = [
        'Agent run complete',
        `Mode: ${op.agent_mode || dbg.context_exec_mode || 'unknown'}`,
        `Route: ${op.route_used || dbg.route_used || dbg.llm_profile || 'chat'}`,
        `Synthesis: ${synthesis}`,
        `Result: ${op.summary || ''}`,
      ];
      if (op.proposal_id) {
        lines.push(`Proposal: ${op.proposal_id} ${op.proposal_status || 'pending_review'}`);
        lines.push('Open Pending Decisions to review.');
      }
      lines.push('Mutation: none');
      return lines.join('\n');
    }
    const top = String(d.llm_response ?? d.text ?? '').trim();
    const raw = d.raw && typeof d.raw === 'object' ? d.raw : {};
    const nested = String(raw.final_text ?? '').trim();
    const cortexNested = raw.cortex_result && typeof raw.cortex_result === 'object'
      ? String(raw.cortex_result.final_text ?? '').trim()
      : '';
    let pick = nested.length > top.length ? nested : top;
    if (cortexNested.length > pick.length) pick = cortexNested;
    if (pick) return pick;
    const meta = raw.metadata && typeof raw.metadata === 'object' ? raw.metadata : {};
    const sr = meta.skill_result != null ? meta.skill_result : meta.skillResult;
    if (sr !== undefined && sr !== null) {
      try {
        return typeof sr === 'string' ? String(sr).trim() : JSON.stringify(sr);
      } catch (_) {
        return String(sr);
      }
    }
    return '';
  }

  function shouldAppendOrionWsPayload(d) {
    if (!d || typeof d !== 'object') return false;
    if (d.tts_error) return false;
    // TTS playback follow-up may carry assistant text for logging; never spawn a second bubble.
    if (d.audio_response && !d.llm_response) return Boolean(d.workflow);
    if (d.workflow) return true;
    if (resolveAssistantDisplayText(d)) return true;
    if (d.error) return false;
    return false;
  }

  function beginWsReadyWait() {
    wsReadyPromise = new Promise((resolve) => {
      wsReadyResolve = resolve;
    });
  }

  async function waitForWebSocketOpen(timeoutMs = 5000) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      return true;
    }
    if (!socket || socket.readyState === WebSocket.CLOSED) {
      setupWebSocket();
    }
    if (!wsReadyPromise) {
      beginWsReadyWait();
    }
    const timeout = new Promise((resolve) => {
      setTimeout(() => resolve(false), Math.max(0, timeoutMs));
    });
    const opened = await Promise.race([
      wsReadyPromise.then(() => socket && socket.readyState === WebSocket.OPEN),
      timeout,
    ]);
    return Boolean(opened);
  }

  function setupWebSocket() {
    const wsUrl = HUB_WEBSOCKET_URL;
    console.log(`[WS] Connecting to ${wsUrl}...`);
    if (socket && socket.readyState === WebSocket.OPEN) {
      return;
    }
    if (socket && socket.readyState === WebSocket.CONNECTING) {
      return;
    }
    if (socket) {
      try {
        socket.onopen = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.onmessage = null;
        socket.close();
      } catch (_err) {
        // ignore close errors during reconnect
      }
    }
    beginWsReadyWait();
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("[WS] Connected");
        window.__orionWsFallbackWarned = false;
        if (typeof wsReadyResolve === 'function') {
          wsReadyResolve(true);
          wsReadyResolve = null;
        }
        updateStatus('Connected.');
    };

    socket.onmessage = (e) => {
      try {
          const d = JSON.parse(e.data);
          const displayText = resolveAssistantDisplayText(d);
          if (d.transcript && !d.is_text_input) appendMessage('You', d.transcript);
          if (shouldAppendOrionWsPayload(d)) {
            appendMessage('Orion', displayText || '', 'text-white', {
              raw: d.raw,
              reasoning: d.reasoning,
              reasoningTrace: d.reasoning_trace,
              reasoningContent: d.reasoning_content,
              inlineThinkContent: d.inline_think_content,
              thinkingSource: d.thinking_source,
              agentTrace: d.agent_trace,
              metacogTraces: d.metacog_traces,
              mode: d.mode,
              model: d.model || (d.raw && d.raw.metadata ? d.raw.metadata.model : null),
              provider: d.provider || (d.raw && d.raw.metadata ? d.raw.metadata.provider : null),
              correlationId: d.correlation_id,
              trace_linkage: d.trace_linkage,
              turnId: d.turn_id || d.turnId || d.correlation_id,
              messageId: d.message_id || d.messageId || null,
              routingDebug: d.routing_debug,
              recallDebug: d.recall_debug,
              memoryDigest: d.memory_digest,
              workflow: d.workflow,
              workflowMetadataOnly: d.workflow_metadata_only,
              autonomySummary: d.autonomy_summary,
              autonomyDebug: d.autonomy_debug,
              autonomyStatePreview: d.autonomy_state_preview,
              autonomyExecutionMode: d.autonomy_execution_mode,
              autonomyGoalsPresent: d.autonomy_goals_present,
              autonomyGoalLineage: d.autonomy_goal_lineage,
              autonomyBackend: d.autonomy_backend,
              autonomySelectedSubject: d.autonomy_selected_subject,
              autonomyRepositoryStatus: d.autonomy_repository_status,
              autonomyStateV2Preview: d.autonomy_state_v2_preview,
              autonomyStateDelta: d.autonomy_state_delta,
              chatStanceDebug: d.chat_stance_debug,
              situationBrief: d.situation_brief,
              situationPromptFragment: d.situation_prompt_fragment,
              presenceContext: d.presence_context,
              temporalPhase: d.temporal_phase,
              situationAffordances: d.situation_affordances,
              substrateEffectSummary: d.substrate_effect_summary || null,
            });
            updateMemoryPanelFromResponse(d);
            syncSocialInspectionFromRouteDebug(d.routing_debug);
          }
          if (d.state) { orionState = d.state; updateStatusBasedOnState(); }
          if (d.tts_debug) {
            console.info('[tts] debug', d.tts_debug);
          }
          if (d.audio_response) {
            console.info('[tts] audio_response received', {
              audio_b64_len: d.audio_response.length,
              tts_meta: d.tts_meta || null,
            });
            audioQueue.push({ audio_b64: d.audio_response, meta: d.tts_meta || null });
            processAudioQueue();
          }
          if (d.tts_error) appendMessage('System', `TTS warning: ${d.tts_error}`, 'text-yellow-400');
          if (d.error) {
            appendMessage('System', `Error: ${d.error}`, 'text-red-400');
            if (d.audio_debug) {
              console.info('[voice] audio_debug', d.audio_debug);
              const sttDbg = d.audio_debug.stt;
              if (sttDbg && sttDbg.peak != null && sttDbg.peak_threshold != null) {
                updateStatus(
                  `Voice error — peak=${sttDbg.peak} threshold=${sttDbg.peak_threshold}`,
                );
              }
            }
          }
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

  function ensureBrowserClientId() {
    if (browserClientId) return browserClientId;
    browserClientId = `browser_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem('orion_browser_client_id', browserClientId);
    return browserClientId;
  }

  function audienceChipLabel(mode) {
    switch (mode) {
      case 'kid_present': return 'Kid present';
      case 'family': return 'Family';
      case 'spouse_present': return 'Spouse present';
      case 'guest_present': return 'Guest present';
      case 'mixed_group': return 'Mixed group';
      default: return 'Solo';
    }
  }

  function syncPresenceChip() {
    if (!presenceStatusChip) return;
    const mode = presenceContext && presenceContext.audience_mode ? presenceContext.audience_mode : 'solo';
    presenceStatusChip.textContent = audienceChipLabel(mode);
  }

  function buildPresencePreset(mode) {
    const base = {
      kind: 'presence.context.v1',
      requestor: {
        display_name: (presenceRequestorName && presenceRequestorName.value ? presenceRequestorName.value : 'Juniper'),
        relationship_to_orion: 'primary_operator',
        source: 'hub_manual',
        confidence: 'medium',
      },
      companions: [],
      audience_mode: mode,
      source: 'hub_manual',
      persist_to_memory: false,
      privacy_mode: (presenceSessionOnlyToggle && presenceSessionOnlyToggle.checked) ? 'session_only' : 'persist_allowed',
    };
    if (mode === 'kid_present') base.companions = [{ display_name: 'Kid', relationship: 'child', role: 'listener', age_band: 'child' }];
    if (mode === 'spouse_present') base.companions = [{ display_name: 'Spouse', relationship: 'spouse', role: 'participant', age_band: 'adult' }];
    if (mode === 'family') base.companions = [
      { display_name: 'Spouse', relationship: 'spouse', role: 'participant', age_band: 'adult' },
      { display_name: 'Kid', relationship: 'child', role: 'listener', age_band: 'child' },
    ];
    return base;
  }

  function renderPresenceCompanions(companions = []) {
    if (!presenceCompanionRows) return;
    presenceCompanionRows.innerHTML = '';
    companions.forEach((item, idx) => {
      const row = document.createElement('div');
      row.className = 'grid grid-cols-1 gap-1 rounded border border-gray-700 bg-gray-900/40 p-2 md:grid-cols-6';
      row.innerHTML = `
        <input data-presence-field="display_name" data-presence-index="${idx}" class="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px]" placeholder="Name" value="${item.display_name || ''}" />
        <input data-presence-field="relationship" data-presence-index="${idx}" class="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px]" placeholder="relationship" value="${item.relationship || 'other'}" />
        <input data-presence-field="role" data-presence-index="${idx}" class="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px]" placeholder="role" value="${item.role || 'nearby'}" />
        <input data-presence-field="age_band" data-presence-index="${idx}" class="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px]" placeholder="age_band" value="${item.age_band || 'unknown'}" />
        <input data-presence-field="context_note" data-presence-index="${idx}" class="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px]" placeholder="context note" value="${item.context_note || ''}" />
        <button data-presence-remove="${idx}" class="rounded border border-rose-500/50 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200" type="button">Remove</button>
      `;
      presenceCompanionRows.appendChild(row);
    });
  }

  function collectPresenceCompanions() {
    if (!presenceCompanionRows) return [];
    const rows = {};
    presenceCompanionRows.querySelectorAll('input[data-presence-index]').forEach((el) => {
      const idx = String(el.dataset.presenceIndex || '');
      const field = String(el.dataset.presenceField || '');
      if (!rows[idx]) rows[idx] = {};
      rows[idx][field] = String(el.value || '').trim();
    });
    return Object.values(rows).filter((row) => row.display_name);
  }

  async function loadPresenceContext() {
    ensureBrowserClientId();
    try {
      const headers = {};
      if (orionSessionId) headers['X-Orion-Session-Id'] = orionSessionId;
      const resp = await fetch(`${API_BASE_URL}/api/presence`, { headers });
      if (!resp.ok) return;
      presenceContext = await resp.json();
      if (presenceAudienceMode && presenceContext && presenceContext.audience_mode) {
        presenceAudienceMode.value = presenceContext.audience_mode;
      }
      if (presenceRequestorName && presenceContext && presenceContext.requestor) {
        presenceRequestorName.value = presenceContext.requestor.display_name || 'Juniper';
      }
      renderPresenceCompanions((presenceContext && presenceContext.companions) || []);
      syncPresenceChip();
    } catch (err) {
      console.warn('[Presence] load failed', err);
      if (presenceStatusChip) presenceStatusChip.textContent = 'Presence unavailable';
    }
  }

  async function savePresenceContext(payload) {
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (orionSessionId) headers['X-Orion-Session-Id'] = orionSessionId;
      const resp = await fetch(`${API_BASE_URL}/api/presence`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ ...payload, browser_client_id: ensureBrowserClientId() }),
      });
      if (!resp.ok) return;
      presenceContext = await resp.json();
      syncPresenceChip();
    } catch (err) {
      console.warn('[Presence] save failed', err);
      if (presenceStatusChip) presenceStatusChip.textContent = 'Presence unavailable';
    }
  }
  if (presenceOpenButton && presenceModalRoot) {
    presenceOpenButton.addEventListener('click', () => {
      presenceModalRoot.classList.remove('hidden');
      presenceModalRoot.setAttribute('aria-hidden', 'false');
    });
  }
  if (presenceModalClose && presenceModalRoot) {
    presenceModalClose.addEventListener('click', () => {
      presenceModalRoot.classList.add('hidden');
      presenceModalRoot.setAttribute('aria-hidden', 'true');
    });
  }
  if (presenceModalBackdrop && presenceModalRoot) {
    presenceModalBackdrop.addEventListener('click', () => {
      presenceModalRoot.classList.add('hidden');
      presenceModalRoot.setAttribute('aria-hidden', 'true');
    });
  }
  if (presencePresetSolo) presencePresetSolo.addEventListener('click', () => { if (presenceAudienceMode) presenceAudienceMode.value = 'solo'; renderPresenceCompanions([]); });
  if (presencePresetKids) presencePresetKids.addEventListener('click', () => { if (presenceAudienceMode) presenceAudienceMode.value = 'kid_present'; renderPresenceCompanions(buildPresencePreset('kid_present').companions); });
  if (presencePresetSpouse) presencePresetSpouse.addEventListener('click', () => { if (presenceAudienceMode) presenceAudienceMode.value = 'spouse_present'; renderPresenceCompanions(buildPresencePreset('spouse_present').companions); });
  if (presencePresetFamily) presencePresetFamily.addEventListener('click', () => { if (presenceAudienceMode) presenceAudienceMode.value = 'family'; renderPresenceCompanions(buildPresencePreset('family').companions); });
  if (presencePresetGuest) presencePresetGuest.addEventListener('click', () => { if (presenceAudienceMode) presenceAudienceMode.value = 'guest_present'; renderPresenceCompanions([{display_name:'Guest', relationship:'guest', role:'participant', age_band:'adult'}]); });
  if (presenceAddCompanionButton) {
    presenceAddCompanionButton.addEventListener('click', () => {
      const current = collectPresenceCompanions();
      current.push({ display_name: `Companion ${current.length + 1}`, relationship: 'other', role: 'nearby', age_band: 'unknown' });
      renderPresenceCompanions(current);
    });
  }
  if (presenceCompanionRows) {
    presenceCompanionRows.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const removeIdx = target.getAttribute('data-presence-remove');
      if (removeIdx === null) return;
      const current = collectPresenceCompanions();
      current.splice(Number(removeIdx), 1);
      renderPresenceCompanions(current);
    });
  }
  if (presenceSaveButton) {
    presenceSaveButton.addEventListener('click', async () => {
      const mode = presenceAudienceMode ? presenceAudienceMode.value : 'solo';
      const payload = buildPresencePreset(mode);
      payload.companions = collectPresenceCompanions();
      await savePresenceContext(payload);
      if (presenceModalRoot) presenceModalRoot.classList.add('hidden');
    });
  }
  if (presenceClearButton) {
    presenceClearButton.addEventListener('click', async () => {
      const headers = {};
      if (orionSessionId) headers['X-Orion-Session-Id'] = orionSessionId;
      await fetch(`${API_BASE_URL}/api/presence`, { method: 'DELETE', headers });
      await loadPresenceContext();
    });
  }
  async function submitExplicitChatText(text, opts = {}) {
    const value = String(text || '').trim();
    if (!value) return;

    let effectiveRoute = String(
      (opts && opts.llm_route) || selectedLlmRoute || llmRouteCatalog.default_route || 'chat'
    ).toLowerCase();
    if (!(opts && opts.skipRouteDownCheck)) {
      const confirmedRoute = await confirmDownRouteOrProceed(effectiveRoute);
      if (confirmedRoute === null) {
        updateStatus('Send cancelled (route unavailable).');
        return;
      }
      effectiveRoute = confirmedRoute;
      if (effectiveRoute !== selectedLlmRoute) {
        selectedLlmRoute = effectiveRoute;
        localStorage.setItem('orion_llm_route', effectiveRoute);
        applyLlmRouteButtonSelection(effectiveRoute);
      }
    }

    appendMessage('You', value);
    if (chatInput) chatInput.value = '';

    const recallMode = recallModeSelect ? recallModeSelect.value : "auto";
    const recallProfile = recallProfileSelect ? recallProfileSelect.value : "auto";
    const forceAgentPath = Boolean(opts && opts.forceAgentPath);
    const explicitVerbs = Array.isArray(opts && opts.verbs)
      ? opts.verbs.map((verb) => String(verb || '').trim()).filter(Boolean)
      : null;
    const effectiveVerbs = forceAgentPath
      ? []
      : (explicitVerbs !== null ? explicitVerbs : (modeVerbOverride ? [modeVerbOverride] : selectedVerbs));
    const isChatQuickSend =
      Array.isArray(effectiveVerbs) &&
      effectiveVerbs.length === 1 &&
      String(effectiveVerbs[0] || '').trim().toLowerCase() === 'chat_quick';
    const requestMode = forceAgentPath
      ? 'agent'
      : (opts && opts.mode ? String(opts.mode) : currentMode);
    const omitChatUiMode =
      Boolean(opts && opts.skillRunnerOrigin && String(opts.skillRunnerLane || '').toLowerCase() === 'deterministic');
    const payload = {
       text_input: value,
       session_id: orionSessionId,
       browser_client_id: ensureBrowserClientId(),
       disable_tts: textToSpeechToggle ? !textToSpeechToggle.checked : false,
       no_write: noWriteToggle ? noWriteToggle.checked : false,
       use_recall: omitChatUiMode ? false : (recallToggle ? recallToggle.checked : true),
       recall_mode: omitChatUiMode ? null : (recallMode !== "auto" ? recallMode : null),
       recall_profile: omitChatUiMode ? null : (recallProfile !== "auto" ? recallProfile : null),
       recall_profile_explicit: omitChatUiMode
         ? false
         : (recallProfile !== "auto"),
       recall_required: omitChatUiMode ? false : (recallRequiredToggle ? recallRequiredToggle.checked : false),
       packs: omitChatUiMode ? [] : selectedPacks,
       verbs: effectiveVerbs,
       skill_runner_origin: Boolean(opts && opts.skillRunnerOrigin),
       skill_runner_lane: opts && opts.skillRunnerLane ? String(opts.skillRunnerLane) : null,
       presence_context: presenceContext,
       surface_context: { surface: 'hub_desktop', input_modality: 'typed' },
       llm_route: effectiveRoute,
    };
    if (!omitChatUiMode) {
      payload.mode = requestMode;
    }
    const optFromOpts = opts && opts.options && typeof opts.options === 'object' ? { ...opts.options } : {};
    if (isChatQuickSend) {
      payload.options = { ...optFromOpts, chat_quick_full_stance: chatQuickVariant === 'stance' };
    } else if (Object.keys(optFromOpts).length > 0) {
      payload.options = optFromOpts;
    }
    if (mindDefaultOnSendToggle && mindDefaultOnSendToggle.checked) {
      payload.context = payload.context && typeof payload.context === "object" ? payload.context : {};
      payload.context.metadata = payload.context.metadata && typeof payload.context.metadata === "object"
        ? payload.context.metadata
        : {};
      payload.context.metadata.mind_enabled = true;
    }
    if (opts && opts.workflowRequestOverride && typeof opts.workflowRequestOverride === 'object') {
      payload.workflow_request_override = opts.workflowRequestOverride;
    }

    let wsOpen = socket && socket.readyState === WebSocket.OPEN;
    if (!wsOpen) {
      updateStatus('Connecting...');
      wsOpen = await waitForWebSocketOpen(5000);
    }

    if (wsOpen && socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(payload));
        updateStatus('Sent...');
    } else {
        if (!window.__orionWsFallbackWarned) {
          appendMessage(
            'System',
            'WebSocket not connected — using HTTP. Replies may take longer; check status bar for Connected.',
            'text-yellow-400'
          );
          window.__orionWsFallbackWarned = true;
        }
        updateStatus('Processing (HTTP)...');

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
        .then(async (r) => {
          const body = await r.json().catch(() => ({}));
          if (!r.ok) {
            const detail = body && (body.detail || body.error) ? String(body.detail || body.error) : r.statusText;
            throw new Error(detail || `HTTP ${r.status}`);
          }
          return body;
        })
        .then(d => {
            const displayText = resolveAssistantDisplayText(d);
            if (shouldAppendOrionWsPayload(d)) {
              appendMessage('Orion', displayText || '', 'text-white', {
                raw: d.raw,
                reasoning: d.reasoning,
                reasoningTrace: d.reasoning_trace,
                reasoningContent: d.reasoning_content,
                inlineThinkContent: d.inline_think_content,
                thinkingSource: d.thinking_source,
                agentTrace: d.agent_trace,
                metacogTraces: d.metacog_traces,
                mode: d.mode,
                model: d.model || (d.raw && d.raw.metadata ? d.raw.metadata.model : null),
                provider: d.provider || (d.raw && d.raw.metadata ? d.raw.metadata.provider : null),
                correlationId: d.correlation_id,
                trace_linkage: d.trace_linkage,
                turnId: d.turn_id || d.turnId || d.correlation_id,
                messageId: d.message_id || d.messageId || null,
                routingDebug: d.routing_debug,
                recallDebug: d.recall_debug,
                memoryDigest: d.memory_digest,
                workflow: d.workflow,
                workflowMetadataOnly: d.workflow_metadata_only,
                autonomySummary: d.autonomy_summary,
                autonomyDebug: d.autonomy_debug,
                autonomyStatePreview: d.autonomy_state_preview,
                autonomyExecutionMode: d.autonomy_execution_mode,
                autonomyGoalsPresent: d.autonomy_goals_present,
                autonomyGoalLineage: d.autonomy_goal_lineage,
                autonomyBackend: d.autonomy_backend,
                autonomySelectedSubject: d.autonomy_selected_subject,
                autonomyRepositoryStatus: d.autonomy_repository_status,
                autonomyStateV2Preview: d.autonomy_state_v2_preview,
                autonomyStateDelta: d.autonomy_state_delta,
                chatStanceDebug: d.chat_stance_debug,
                situationBrief: d.situation_brief,
                situationPromptFragment: d.situation_prompt_fragment,
                presenceContext: d.presence_context,
                temporalPhase: d.temporal_phase,
                situationAffordances: d.situation_affordances,
                substrateEffectSummary: d.substrate_effect_summary || null,
              });
              syncSocialInspectionFromRouteDebug(d.routing_debug);
            } else if (d.error) {
              appendMessage('System', d.error, 'text-red-400');
            } else {
              appendMessage(
                'System',
                'HTTP completed but no assistant text was returned (empty or blocked response).',
                'text-yellow-400'
              );
            }
            updateMemoryPanelFromResponse(d);
            updateStatusBasedOnState();
        })
        .catch(e => {
          appendMessage('System', `HTTP failed: ${e.message}`, 'text-red-400');
          updateStatus('HTTP error.');
        });
    }
  }

  async function sendTextMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    await submitExplicitChatText(text);
  }

  // --- Audio ---
  function _uint8ToBase64(bytes) {
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  }

  function _encodeWavPcm16(audioBuffer) {
    const channel = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    const numSamples = channel.length;
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numSamples * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i += 1) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);
    let offset = 44;
    for (let i = 0; i < numSamples; i += 1) {
      const sample = Math.max(-1, Math.min(1, channel[i]));
      const int16 = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
      view.setInt16(offset, int16, true);
      offset += 2;
    }
    return new Uint8Array(buffer);
  }

  function _mergePcmChunks(chunks) {
    let total = 0;
    for (const chunk of chunks) {
      total += chunk.length;
    }
    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  function _measureFloatPeak(pcm) {
    let peak = 0;
    for (let i = 0; i < pcm.length; i += 1) {
      peak = Math.max(peak, Math.abs(pcm[i]));
    }
    return peak;
  }

  /** PCM float32 → 16 kHz mono WAV base64 with capture telemetry (warn on low peak, do not block send). */
  async function _pcmToWavBase64(pcm, sampleRate, chunkCount) {
    const sourceSampleRate = sampleRate || 48000;
    const sourcePeak = _measureFloatPeak(pcm);
    const ctx = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    const buffer = ctx.createBuffer(1, pcm.length, sourceSampleRate);
    buffer.getChannelData(0).set(pcm);
    const targetRate = 16000;
    const frames = Math.max(1, Math.ceil(buffer.duration * targetRate));
    const offline = new OfflineAudioContext(1, frames, targetRate);
    const src = offline.createBufferSource();
    src.buffer = buffer;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    const channel = rendered.getChannelData(0);
    let peak = 0;
    let sumSq = 0;
    for (let i = 0; i < channel.length; i += 1) {
      const v = channel[i];
      peak = Math.max(peak, Math.abs(v));
      sumSq += v * v;
    }
    const rms = channel.length ? Math.sqrt(sumSq / channel.length) : 0;
    const durationSec = rendered.duration;
    let clientGate = 'passed';
    if (pcm.length === 0 || channel.length === 0) {
      clientGate = 'empty';
    } else if (peak < VOICE_CLIENT_PEAK_MIN) {
      clientGate = 'low_peak_warn';
    }
    const audioB64 = _uint8ToBase64(_encodeWavPcm16(rendered));
    return {
      audio_b64: audioB64,
      metadata: {
        source_sample_rate: sourceSampleRate,
        target_sample_rate: targetRate,
        duration_sec: durationSec,
        peak,
        rms,
        pcm_samples: pcm.length,
        chunk_count: chunkCount,
        source_peak: sourcePeak,
        client_peak_threshold: VOICE_CLIENT_PEAK_MIN,
        client_gate: clientGate,
      },
    };
  }

  function _teardownPcmCapture() {
    if (!pcmCapture) {
      return;
    }
    pcmCapture.capturing = false;
    pcmCapture.flushing = false;
    try {
      pcmCapture.processor.disconnect();
      pcmCapture.source.disconnect();
      pcmCapture.gain.disconnect();
    } catch (_) {
      /* ignore */
    }
    pcmCapture = null;
  }

  async function _finalizePcmCapture() {
    const capture = pcmCapture;
    _teardownPcmCapture();
    recordButton.classList.remove('pulse');
    if (recordingStream) {
      recordingStream.getTracks().forEach(t => t.stop());
      recordingStream = null;
    }
    if (!capture || !capture.chunks.length) {
      updateStatus('No audio chunks captured — browser did not deliver mic frames.');
      return;
    }
    const captureSampleRate = capture.sampleRate || audioContext.sampleRate || 48000;
    const pcm = _mergePcmChunks(capture.chunks);
    const durationSec = pcm.length / captureSampleRate;
    console.info(
      '[voice] chunk_count=%d source_sample_rate=%d pcm_samples=%d duration=%ss',
      capture.chunks.length,
      captureSampleRate,
      pcm.length,
      durationSec.toFixed(2),
    );
    if (durationSec < VOICE_MIN_CAPTURE_DURATION_SEC) {
      updateStatus('Recording too short — hold the mic button longer.');
      return;
    }
    let encoded;
    try {
      encoded = await _pcmToWavBase64(pcm, captureSampleRate, capture.chunks.length);
    } catch (err) {
      console.error('[voice] encode failed', err);
      updateStatus('Could not process recording — try again.');
      return;
    }
    const audioB64 = encoded.audio_b64;
    const audioMeta = encoded.metadata;
    console.info(
      '[voice] peak=%s rms=%s target_sample_rate=%d duration=%ss client_gate=%s',
      audioMeta.peak.toFixed(6),
      audioMeta.rms.toFixed(6),
      audioMeta.target_sample_rate,
      audioMeta.duration_sec.toFixed(2),
      audioMeta.client_gate,
    );
    if (audioMeta.client_gate === 'empty' || (audioMeta.source_peak === 0 && audioMeta.peak === 0)) {
      updateStatus(
        'No microphone signal in capture — check OS input device, browser mic permission, and input level.',
      );
      console.warn('[voice] silent capture source_peak=0 chunk_count=%d', audioMeta.chunk_count);
      return;
    }
    if (audioMeta.client_gate === 'low_peak_warn') {
      updateStatus('Low mic level, sending anyway...');
    }
    const voiceRoute = await confirmDownRouteOrProceed(
      selectedLlmRoute || llmRouteCatalog.default_route || 'chat'
    );
    if (voiceRoute === null) {
      updateStatus('Voice send cancelled (route unavailable).');
      return;
    }
    if (voiceRoute !== selectedLlmRoute) {
      selectedLlmRoute = voiceRoute;
      localStorage.setItem('orion_llm_route', voiceRoute);
      applyLlmRouteButtonSelection(voiceRoute);
    }
    try {
      let wsOpen = socket && socket.readyState === WebSocket.OPEN;
      if (!wsOpen) {
        updateStatus('Connecting...');
        wsOpen = await waitForWebSocketOpen(5000);
      }
      if (wsOpen && socket && socket.readyState === WebSocket.OPEN) {
        const audioPayload = {
          audio: audioB64,
          audio_format: 'wav',
          client_audio_meta: audioMeta,
          mode: currentMode,
          session_id: orionSessionId,
          browser_client_id: ensureBrowserClientId(),
          disable_tts: textToSpeechToggle ? !textToSpeechToggle.checked : false,
          no_write: noWriteToggle ? noWriteToggle.checked : false,
          use_recall: recallToggle ? recallToggle.checked : true,
          recall_mode: recallModeSelect && recallModeSelect.value !== 'auto' ? recallModeSelect.value : null,
          recall_profile: recallProfileSelect && recallProfileSelect.value !== 'auto' ? recallProfileSelect.value : null,
          recall_profile_explicit:
            recallProfileSelect && recallProfileSelect.value !== 'auto',
          recall_required: recallRequiredToggle ? recallRequiredToggle.checked : false,
          presence_context: presenceContext,
          surface_context: { surface: 'hub_desktop', input_modality: 'spoken' },
          llm_route: voiceRoute,
        };
        const audioLaneVerbs = modeVerbOverride ? [modeVerbOverride] : selectedVerbs;
        const audioIsChatQuick =
          Array.isArray(audioLaneVerbs) &&
          audioLaneVerbs.length === 1 &&
          String(audioLaneVerbs[0] || '').trim().toLowerCase() === 'chat_quick';
        if (audioIsChatQuick) {
          audioPayload.verbs = ['chat_quick'];
          audioPayload.options = { chat_quick_full_stance: chatQuickVariant === 'stance' };
        } else if (
          Array.isArray(audioLaneVerbs) &&
          audioLaneVerbs.length === 1 &&
          String(audioLaneVerbs[0] || '').trim().toLowerCase() === 'chat_kids_story'
        ) {
          audioPayload.verbs = ['chat_kids_story'];
        }
        socket.send(JSON.stringify(audioPayload));
        console.info('[voice] sent audio payload bytes=%d', audioB64.length);
        updateStatus(
          `Processing audio... peak=${audioMeta.peak.toFixed(5)} duration=${audioMeta.duration_sec.toFixed(2)}s`,
        );
      } else {
        updateStatus('Offline. Cannot send audio.');
      }
    } catch (err) {
      console.error('[voice] send failed', err);
      updateStatus('Failed to send audio.');
    }
  }

  function _finishVoiceStop() {
    if (!pcmCapture || !pcmCapture.capturing) {
      recordButton.classList.remove('pulse');
      return;
    }
    pcmCapture.flushing = true;
    // Keep capturing=true so ScriptProcessor keeps delivering buffers during flush.
    setTimeout(() => {
      if (pcmCapture) {
        pcmCapture.capturing = false;
      }
      _finalizePcmCapture();
    }, VOICE_PCM_FLUSH_MS);
  }

  async function startRecording() {
    if (pcmCapture && pcmCapture.capturing) {
      return;
    }
    const captureGen = ++voiceCapture.generation;
    voiceCapture.stopRequested = false;
    voiceCapture.starting = true;
    try {
      if (recordingStream) {
        recordingStream.getTracks().forEach(t => t.stop());
        recordingStream = null;
      }
      _teardownPcmCapture();
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });
      if (captureGen !== voiceCapture.generation || voiceCapture.stopRequested) {
        stream.getTracks().forEach(t => t.stop());
        if (voiceCapture.stopRequested) {
          updateStatus('Mic not ready in time — hold the button a moment longer.');
        }
        return;
      }
      recordingStream = stream;
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const gain = audioContext.createGain();
      gain.gain.value = 0;
      pcmCapture = {
        source,
        processor,
        gain,
        chunks: [],
        capturing: true,
        flushing: false,
        sampleRate: audioContext.sampleRate,
      };
      processor.onaudioprocess = e => {
        if (!pcmCapture || !pcmCapture.capturing) {
          return;
        }
        const input = e.inputBuffer.getChannelData(0);
        const output = e.outputBuffer.getChannelData(0);
        // Required in Chromium: without copying input→output, inputBuffer is often silent.
        output.set(input);
        pcmCapture.chunks.push(new Float32Array(input));
      };
      source.connect(processor);
      processor.connect(gain);
      gain.connect(audioContext.destination);
      updateStatus('Recording...');
      recordButton.classList.add('pulse');
      if (voiceCapture.stopRequested) {
        _finishVoiceStop();
      }
    } catch (e) {
      console.error(e);
      updateStatus('Mic Access Denied');
    } finally {
      if (captureGen === voiceCapture.generation) {
        voiceCapture.starting = false;
      }
    }
  }

  function stopRecording() {
    voiceCapture.stopRequested = true;
    if (!pcmCapture || !pcmCapture.capturing) {
      recordButton.classList.remove('pulse');
      if (voiceCapture.starting) {
        updateStatus('Mic not ready in time — hold the button a moment longer.');
      }
      return;
    }
    _finishVoiceStop();
  }

  function processAudioQueue() {
    if (isPlayingAudio || !audioQueue.length) return;
    isPlayingAudio = true;
    const item = audioQueue.shift();
    playAudio(item);
  }

  async function playAudio(item) {
    const audioB64 = typeof item === 'string' ? item : item.audio_b64;
    const meta = typeof item === 'object' && item !== null ? item.meta : null;
    try {
        if (!audioB64) {
          throw new Error('empty audio payload');
        }
        if (!audioContext) {
          audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        const beforeState = audioContext.state;
        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }
        console.info('[tts] playback decode start', {
          beforeState,
          afterState: audioContext.state,
          audio_b64_len: audioB64.length,
          meta,
        });
        const bin = atob(audioB64);
        const arr = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i += 1) arr[i] = bin.charCodeAt(i);
        const buf = await audioContext.decodeAudioData(arr.buffer.slice(0));
        console.info('[tts] decoded', { duration: buf.duration, sampleRate: buf.sampleRate });
        const src = audioContext.createBufferSource();
        src.buffer = buf;
        const gain = audioContext.createGain();
        gain.gain.value = 1.0;
        src.connect(gain);
        gain.connect(audioContext.destination);

        analyser = audioContext.createAnalyser();
        src.connect(analyser);
        drawVisualizer();

        src.start(0);
        console.info('[tts] playback started');
        currentAudioSource = src;
        src.onended = () => {
          console.info('[tts] playback ended');
          isPlayingAudio = false;
          cancelAnimationFrame(animationFrameId);
          if (canvasCtx) canvasCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
          processAudioQueue();
        };
    } catch (e) {
        console.error('[tts] playback failed', e);
        appendMessage('System', `TTS playback failed: ${e.message || e}`, 'text-yellow-400');
        updateStatus('TTS playback failed.');
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

  // Connect WebSocket immediately so chat is not blocked on slow initSession/library loads.
  setupWebSocket();

  (async () => {
      // 1. Session
      await initSession();
      await loadPresenceContext();
      // 2. Library (Safe)
      await loadCognitionLibrary();
      // 4. UI
      if (document.body && document.body.dataset.toastSeconds) {
        const parsed = parseInt(document.body.dataset.toastSeconds, 10);
        if (!Number.isNaN(parsed)) notificationToastSeconds = parsed;
      }
      await loadNotifications();
      await loadChatMessages();
      await loadPendingAttention();
      await loadWorldPulseLatest();
      if (messagesToggle) {
        messagesToggle.addEventListener('click', toggleMessagesPanel);
      }
      if (messageFilter) {
        messageFilter.addEventListener('click', (event) => event.stopPropagation());
      }
      if (notificationFilter) {
        notificationFilter.addEventListener('change', renderNotifications);
      }
      if (messageFilter) {
        messageFilter.addEventListener('change', () => {
          renderChatMessages();
          loadChatMessages();
        });
      }
      if (worldPulseToggle) {
        worldPulseToggle.addEventListener('click', toggleWorldPulsePanel);
      }
      if (worldPulseRunButton) {
        worldPulseRunButton.addEventListener('click', (event) => {
          event.stopPropagation();
          triggerWorldPulseRun();
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

  if (hubTabButton && topicStudioTabButton && serviceLogsTabButton && substrateTabButton) {
    hubTabButton.addEventListener("click", () => {
      setActiveTab("hub");
      history.replaceState(null, "", "#hub");
    });
    topicStudioTabButton.addEventListener("click", () => {
      setActiveTab("topic-studio");
      history.replaceState(null, "", "#topic-studio");
      refreshTopicStudio();
    });
    serviceLogsTabButton.addEventListener("click", () => {
      setActiveTab("service-logs");
      history.replaceState(null, "", "#service-logs");
    });
    substrateTabButton.addEventListener("click", (event) => {
      event.preventDefault();
      setActiveTab("substrate");
      history.replaceState(null, "", "#substrate");
    });
    if (substrateAtlasTabButton && substrateAtlasPanel) {
      substrateAtlasTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("substrate-atlas");
        history.replaceState(null, "", "#substrate-atlas");
      });
    }
    if (memoryTabButton) {
      memoryTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("memory");
        history.replaceState(null, "", "#memory");
      });
    }
    if (mindTabButton && mindPanel) {
      mindTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("mind");
        history.replaceState(null, "", "#mind");
      });
    }
    if (signalsTabButton && signalsPanel) {
      signalsTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("signals");
        history.replaceState(null, "", "#signals");
      });
    }
    if (pressureAnalyticsTabButton && pressurePanel) {
      pressureAnalyticsTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("pressure");
        history.replaceState(null, "", "#pressure");
      });
    }
    if (substrateLatticeTabButton && substrateLatticePanelEl) {
      substrateLatticeTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("substrate-lattice");
        history.replaceState(null, "", "#substrate-lattice");
      });
    }
    if (forgeTabButton && forgePanel) {
      forgeTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("forge");
        history.replaceState(null, "", "#forge");
      });
    }
    if (collapseMirrorTabButton && collapseMirrorPanel) {
      collapseMirrorTabButton.addEventListener("click", (event) => {
        event.preventDefault();
        setActiveTab("collapse-mirror");
        history.replaceState(null, "", "#collapse-mirror");
      });
    }
    applyHashToTab();
    window.addEventListener("hashchange", () => {
      applyHashToTab();
    });
  }

  if (substratePanelRefresh && substratePanelFrame) {
    substratePanelRefresh.addEventListener("click", () => {
      substratePanelFrame.contentWindow?.location.reload();
    });
  }

  if (substrateAtlasPanelRefresh && substrateAtlasPanelFrame) {
    substrateAtlasPanelRefresh.addEventListener("click", () => {
      try {
        substrateAtlasPanelFrame.contentWindow?.location.reload();
      } catch {
        /* ignore */
      }
    });
  }

  if (pressureAnalyticsRefresh && pressureAnalyticsFrame) {
    pressureAnalyticsRefresh.addEventListener("click", () => {
      try {
        pressureAnalyticsFrame.contentWindow?.location.reload();
      } catch {
        /* ignore */
      }
    });
  }

  if (forgeRefreshButton) {
    forgeRefreshButton.addEventListener("click", () => refreshForgeTab());
  }
  if (forgeSearchButton) {
    forgeSearchButton.addEventListener("click", () => runForgeSearch());
  }
  if (forgeSearchInput) {
    forgeSearchInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        runForgeSearch();
      }
    });
  }
  document.querySelectorAll(".forge-claims-filter").forEach((btn) => {
    btn.addEventListener("click", () => {
      forgeClaimsFilter = btn.dataset.forgeClaimsFilter || "all";
      forgeStyleClaimsFilterButtons();
      forgeRenderClaimsList();
    });
  });
  if (forgeSpecsStatusFilter) {
    forgeSpecsStatusFilter.addEventListener("change", () => forgeRenderSpecsList());
  }
  if (forgeCompileButton) {
    forgeCompileButton.addEventListener("click", () => runForgeCompile());
  }
  if (forgeSourceIngestButton) {
    forgeSourceIngestButton.addEventListener("click", () => runForgeSourceIngest());
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

  // ─────────────────────────────────────────────────────────────────────
  // Social Room Inspection — bridge-turn polling
  // Polls /api/social-room/inspection/latest every 8 s so the Social
  // Inspection panel updates for turns routed through the bridge (which
  // bypass the Hub WebSocket used for direct UI turns).
  // ─────────────────────────────────────────────────────────────────────
  let _socialInspectionPollLastStoredAt = null;
  async function _pollSocialRoomInspection() {
    if (!socialInspectionApi.shouldShowSocialInspection) return;
    try {
      const res = await fetch('/api/social-room/inspection/latest');
      if (!res.ok) return;
      const data = await res.json();
      if (!data || !data.routing_debug || !data.stored_at) return;
      if (data.stored_at === _socialInspectionPollLastStoredAt) return;
      _socialInspectionPollLastStoredAt = data.stored_at;
      syncSocialInspectionFromRouteDebug(data.routing_debug);
    } catch (_) {
      // network hiccup — ignore, will retry next interval
    }
  }
  setInterval(_pollSocialRoomInspection, 8000);

  if (/[?&]hub_e2e=1(?:&|$)/.test(location.search)) {
    window.__ORION_HUB_E2E__ = {
      seedMemoryGraphTurns(turns) {
        if (!conversationDiv) return;
        conversationDiv.innerHTML = '';
        (turns || []).forEach((t) => {
          if (!t || typeof t !== 'object') return;
          appendMessage(
            t.sender || 'You',
            t.text || '',
            t.colorClass || 'text-white',
            t.meta && typeof t.meta === 'object' ? t.meta : {},
          );
        });
      },
      openMemoryGraphBridgeForAssistantTurn(turnId) {
        const id = String(turnId || '').trim();
        if (!id || !conversationDiv) throw new Error('turnId required');
        const div = conversationDiv.querySelector(`[data-turn-id="${CSS.escape(id)}"]`);
        if (!div) throw new Error(`assistant turn not found: ${id}`);
        openMemoryGraphBridgeModal(div);
      },
    };
  }

});

