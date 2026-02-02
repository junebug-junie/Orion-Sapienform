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
const NOTIFICATION_MAX = 200;
let notificationToastSeconds = 8;
const ATTENTION_EVENT_KIND = "orion.chat.attention";
const CHAT_MESSAGE_EVENT_KIND = "orion.chat.message";
const RECIPIENT_GROUP = "juniper_primary";

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
    if (!notification || !notification.message_id) return null;
    return {
      message_id: notification.message_id,
      session_id: notification.session_id,
      created_at: notification.created_at,
      severity: notification.severity || 'info',
      title: notification.title || 'New message from Orion',
      preview_text: notification.body_text || '',
      status: 'unread',
      silent: Boolean(notification.silent),
    };
  }

  function upsertChatMessage(item) {
    if (!item || !item.message_id) return;
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
    const filtered =
      filter === 'all' ? chatMessages : chatMessages.filter((m) => m.status === 'unread');

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
        const card = document.createElement('div');
        card.className = 'bg-gray-900/60 border border-gray-700 rounded-lg p-2 space-y-2';

        const title = document.createElement('div');
        title.className = 'text-gray-100 font-semibold text-xs';
        title.textContent = item.title || 'Message';

        const body = document.createElement('div');
        body.className = 'text-[11px] text-gray-300 whitespace-pre-wrap';
        body.textContent = item.preview_text || '';

        const actions = document.createElement('div');
        actions.className = 'flex items-center gap-2 text-[10px]';

        const openBtn = document.createElement('button');
        openBtn.className = 'px-2 py-1 rounded bg-indigo-600/80 hover:bg-indigo-500 text-white';
        openBtn.textContent = 'Open chat';
        openBtn.addEventListener('click', () => {
          setSessionId(item.session_id);
          focusChatInput();
          handleChatMessageReceipt(item.message_id, item.session_id, 'opened');
        });

        const dismissBtn = document.createElement('button');
        dismissBtn.className = 'px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200';
        dismissBtn.textContent = 'Dismiss';
        dismissBtn.addEventListener('click', () => {
          handleChatMessageReceipt(item.message_id, item.session_id, 'dismissed');
        });

        actions.appendChild(openBtn);
        actions.appendChild(dismissBtn);

        card.appendChild(title);
        card.appendChild(body);
        card.appendChild(actions);
        messageList.appendChild(card);
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
      item.className = 'bg-gray-900/60 border border-gray-700 rounded-lg p-2 space-y-1';

      const header = document.createElement('div');
      header.className = 'flex items-center justify-between gap-2';

      const title = document.createElement('div');
      title.className = 'text-gray-100 font-semibold text-xs';
      title.textContent = n.title || 'Notification';

      const badge = document.createElement('span');
      badge.className = `text-[10px] px-2 py-0.5 rounded-full uppercase ${severityBadgeClass(n.severity)}`;
      badge.textContent = (n.severity || 'info').toUpperCase();

      header.appendChild(title);
      header.appendChild(badge);

      const meta = document.createElement('div');
      meta.className = 'text-[10px] text-gray-400';
      const createdAt = n.created_at ? new Date(n.created_at).toLocaleString() : '--';
      meta.textContent = `${createdAt} • ${n.event_kind || 'event'} • ${n.source_service || 'unknown'}`;

      const body = document.createElement('details');
      body.className = 'text-[11px] text-gray-300';
      const summary = document.createElement('summary');
      summary.className = 'cursor-pointer text-gray-400';
      summary.textContent = 'Details';
      const bodyText = document.createElement('div');
      bodyText.className = 'mt-1 whitespace-pre-wrap';
      bodyText.textContent = n.body_text || '';
      body.appendChild(summary);
      body.appendChild(bodyText);

      item.appendChild(header);
      item.appendChild(meta);
      item.appendChild(body);
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

  function showToast(notification) {
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
        chatMessages = data.map((item) => ({
          message_id: item.message_id,
          session_id: item.session_id,
          created_at: item.created_at,
          severity: item.severity || 'info',
          title: item.title || 'New message from Orion',
          preview_text: item.preview_text || '',
          status: item.status || 'unread',
          silent: false,
        }));
        renderChatMessages();
        chatMessages.forEach((item) => {
          if (item.status === 'unread' && !seenMessageIds.has(item.message_id)) {
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

  if (memoryPanelToggle) {
    memoryPanelToggle.addEventListener('click', toggleMemoryPanel);
  }

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
        fetch(`${API_BASE_URL}/api/chat`, {
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

  if (bioNodeSelect) {
    bioNodeSelect.addEventListener("change", () => {
      selectedBiometricsNode = bioNodeSelect.value || "cluster";
      if (lastBiometricsPayload) {
        updateBiometricsPanel(lastBiometricsPayload);
      }
    });
  }

});
