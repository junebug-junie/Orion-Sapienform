// --- Element References ---
const recordButton = document.getElementById('recordButton');
const interruptButton = document.getElementById('interruptButton');
const statusDiv = document.getElementById('status');
const conversationDiv = document.getElementById('conversation');
const speedControl = document.getElementById('speedControl');
const speedValue = document.getElementById('speedValue');
const tempControl = document.getElementById('tempControl');
const tempValue = document.getElementById('tempValue');
const styleControl = document.getElementById('styleControl');
const colorControl = document.getElementById('colorControl');
const visualizerCanvas = document.getElementById('visualizer');
const canvasCtx = visualizerCanvas ? visualizerCanvas.getContext('2d') : null;
const visualizerContainer = document.getElementById('visualizerContainer');
const stateVisualizerCanvas = document.getElementById('stateVisualizer');
const stateCtx = stateVisualizerCanvas ? stateVisualizerCanvas.getContext('2d') : null;
const stateVisualizerContainer = document.getElementById('stateVisualizerContainer');
const contextControl = document.getElementById('contextControl');
const contextValue = document.getElementById('contextValue');
const instructionsText = document.getElementById('instructions');
const clearButton = document.getElementById('clearButton');
const copyButton = document.getElementById('copyButton');

// Settings panel
const settingsToggle = document.getElementById('settingsToggle');
const settingsPanel = document.getElementById('settingsPanel');
const settingsClose = document.getElementById('settingsClose');

// Vision elements
const visionPopoutButton = document.getElementById('visionPopoutButton');
const visionDockedContainer = document.getElementById('visionDockedContainer');
const visionFloatingContainer = document.getElementById('visionFloatingContainer');
const visionCloseFloatingButton = document.getElementById('visionCloseFloating');

// Collapse Mirror elements
const collapseModeGuided = document.getElementById('collapseModeGuided');
const collapseModeRaw = document.getElementById('collapseModeRaw');
const collapseGuidedSection = document.getElementById('collapseGuidedSection');
const collapseRawSection = document.getElementById('collapseRawSection');
const collapseStatus = document.getElementById('collapseStatus');
const collapseTooltipToggle = document.getElementById('collapseTooltipToggle');
const collapseTooltip = document.getElementById('collapseTooltip');

// These are declared here but assigned inside DOMContentLoaded
let chatInput, sendButton, textToSpeechToggle, councilButton;
const API_BASE_URL = window.location.origin;

// --- State Management ---
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
let liveMicAnalyser;
let liveMicSource;
let particles = [];
let baseParticleCount = 200;
let visionIsFloating = false;

// Orion Hub session id (from /api/session)
let orionSessionId = null;

// ───────────────────────────────────────────────────────────────
// Session Initialization
// ───────────────────────────────────────────────────────────────
async function initSession() {
  try {
    const headers = {};
    const stored = localStorage.getItem('orion_session_id');
    if (stored) {
      headers['X-Orion-Session-Id'] = stored;
    }

    const res = await fetch(`${API_BASE_URL}/api/session`, {
      method: 'GET',
      headers,
    });

    if (!res.ok) {
      console.warn('Failed to init Orion session:', res.status);
      return;
    }

    const data = await res.json();
    if (data.session_id) {
      orionSessionId = data.session_id;
      localStorage.setItem('orion_session_id', orionSessionId);
      console.log('Orion session initialized:', orionSessionId);
    } else {
      console.warn('No session_id in /api/session response', data);
    }
  } catch (err) {
    console.error('Error during initSession:', err);
  }
}

// --- Event Listeners (for elements that exist on page load) ---
if (tempControl && tempValue) {
  tempControl.addEventListener('input', () => {
    tempValue.textContent = parseFloat(tempControl.value).toFixed(1);
  });
}

if (contextControl && contextValue) {
  contextControl.addEventListener('input', () => {
    contextValue.textContent = contextControl.value;
  });
}

if (speedControl && speedValue) {
  speedControl.addEventListener('input', () => {
    const actualSpeed = parseFloat(speedControl.value);
    const minSpeed = 0.97, maxSpeed = 1.2;
    const normalizedValue = (actualSpeed - minSpeed) / (maxSpeed - minSpeed);
    speedValue.textContent = normalizedValue.toFixed(2);
    if (currentAudioSource) currentAudioSource.playbackRate.value = actualSpeed;
  });
}

if (clearButton && conversationDiv) {
  clearButton.addEventListener('click', () => {
    conversationDiv.innerHTML = '';
  });
}

if (copyButton && conversationDiv) {
  copyButton.addEventListener('click', () => {
    const conversationText = conversationDiv.innerText;
    const textArea = document.createElement("textarea");
    textArea.value = conversationText;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    textArea.remove();
    updateStatus('Conversation copied!');
    setTimeout(() => updateStatusBasedOnState(), 2000);
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

// Settings panel open/close (gear now toggles)
if (settingsToggle && settingsPanel) {
  settingsToggle.addEventListener('click', () => {
    const isHidden = settingsPanel.classList.contains('translate-x-full');
    if (isHidden) {
      settingsPanel.classList.remove('translate-x-full');
      settingsPanel.classList.add('translate-x-0');
    } else {
      settingsPanel.classList.add('translate-x-full');
      settingsPanel.classList.remove('translate-x-0');
    }
  });
}

if (settingsClose && settingsPanel) {
  settingsClose.addEventListener('click', () => {
    settingsPanel.classList.add('translate-x-full');
    settingsPanel.classList.remove('translate-x-0');
  });
}

// Vision popout / dock
function updateVisionUi() {
  if (!visionDockedContainer || !visionFloatingContainer || !visionPopoutButton) return;
  if (visionIsFloating) {
    visionFloatingContainer.classList.remove('hidden');
    visionDockedContainer.classList.add('opacity-30');
    visionPopoutButton.textContent = 'Dock Viewer';
  } else {
    visionFloatingContainer.classList.add('hidden');
    visionDockedContainer.classList.remove('opacity-30');
    visionPopoutButton.textContent = 'Pop Out';
  }
}

if (visionPopoutButton) {
  visionPopoutButton.addEventListener('click', () => {
    visionIsFloating = !visionIsFloating;
    updateVisionUi();
  });
}

if (visionCloseFloatingButton) {
  visionCloseFloatingButton.addEventListener('click', () => {
    visionIsFloating = false;
    updateVisionUi();
  });
}

// Collapse Mirror tabs
function setCollapseMode(mode) {
  if (!collapseModeGuided || !collapseModeRaw || !collapseGuidedSection || !collapseRawSection) return;

  if (mode === 'guided') {
    collapseGuidedSection.classList.remove('hidden');
    collapseRawSection.classList.add('hidden');
    collapseModeGuided.classList.add('bg-gray-700', 'text-gray-100');
    collapseModeGuided.classList.remove('text-gray-300');
    collapseModeRaw.classList.remove('bg-gray-700', 'text-gray-100');
    collapseModeRaw.classList.add('text-gray-300');
  } else {
    collapseGuidedSection.classList.add('hidden');
    collapseRawSection.classList.remove('hidden');
    collapseModeRaw.classList.add('bg-gray-700', 'text-gray-100');
    collapseModeRaw.classList.remove('text-gray-300');
    collapseModeGuided.classList.remove('bg-gray-700', 'text-gray-100');
    collapseModeGuided.classList.add('text-gray-300');
  }
}

if (collapseModeGuided && collapseModeRaw) {
  collapseModeGuided.addEventListener('click', () => setCollapseMode('guided'));
  collapseModeRaw.addEventListener('click', () => setCollapseMode('raw'));
  // default
  setCollapseMode('guided');
}

// Collapse tooltip (the little ? button)
if (collapseTooltipToggle && collapseTooltip) {
  collapseTooltipToggle.addEventListener('click', (event) => {
    event.stopPropagation();
    collapseTooltip.classList.toggle('hidden');
  });

  // Click outside tooltip closes it
  document.addEventListener('click', (event) => {
    if (!collapseTooltip.classList.contains('hidden')) {
      const clickedInside =
        collapseTooltip.contains(event.target) ||
        event.target === collapseTooltipToggle;
      if (!clickedInside) {
        collapseTooltip.classList.add('hidden');
      }
    }
  });
}

// --- Core Functions ---
function updateStatus(newStatus) {
  if (statusDiv) statusDiv.textContent = newStatus;
}

function updateStatusBasedOnState() {
  if (!statusDiv) return;
  if (orionState === 'idle') updateStatus('Ready. Press the button to speak or type a message.');
  else if (orionState === 'speaking') updateStatus('Playing response...');
  else if (orionState === 'processing') updateStatus('Processing...');
}

function setupWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  socket = new WebSocket(wsUrl);

  socket.onopen = () => updateStatus('Connection established. Ready to interact.');
  socket.onclose = () => updateStatus('Connection lost. Please refresh.');

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.transcript) {
      if (!data.is_text_input) appendMessage('You', data.transcript);
    } else if (data.llm_response) {
      appendMessage('Orion', data.llm_response);
      if (data.tokens) {
        baseParticleCount = 200 + Math.min(data.tokens, 200);
        createParticles();
      }
    } else if (data.audio_response) {
      audioQueue.push(data.audio_response);
      processAudioQueue();
    } else if (data.state) {
      orionState = data.state;
      updateStatusBasedOnState();
    } else if (data.error) {
      const errorMessage = `Error: ${data.error}`;
      updateStatus(errorMessage);
      appendMessage('System', errorMessage, 'text-red-400');
      setTimeout(() => updateStatusBasedOnState(), 3000);
    }
  };
}

function processAudioQueue() {
  if (isPlayingAudio || audioQueue.length === 0) return;
  isPlayingAudio = true;
  const base64String = audioQueue.shift();
  playAudio(base64String);
}

async function playAudio(base64String) {
  try {
    if (currentAudioSource) currentAudioSource.stop();
    cancelAnimationFrame(animationFrameId);

    const binaryString = window.atob(base64String);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);

    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    source.playbackRate.value = parseFloat(speedControl.value);
    source.start(0);
    currentAudioSource = source;

    updateStatus('Playing response...');
    interruptButton && interruptButton.classList.remove('hidden');
    drawVisualizer();

    source.onended = () => {
      currentAudioSource = null;
      cancelAnimationFrame(animationFrameId);
      if (visualizerCanvas && canvasCtx) {
        canvasCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
      }
      isPlayingAudio = false;
      processAudioQueue();
      if (audioQueue.length === 0) {
        updateStatusBasedOnState();
        interruptButton && interruptButton.classList.add('hidden');
      }
    };
  } catch (error) {
    console.error("Failed to play audio:", error);
    updateStatus('Error playing audio response.');
    isPlayingAudio = false;
  }
}

function drawVisualizer() {
  if (!analyser || !visualizerCanvas || !canvasCtx) return;
  animationFrameId = requestAnimationFrame(drawVisualizer);
  const selectedStyle = styleControl.value;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  canvasCtx.fillStyle = 'rgb(31, 41, 55)';
  canvasCtx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
  if (selectedStyle === 'bars') {
    analyser.getByteFrequencyData(dataArray);
    drawBars(dataArray, bufferLength);
  } else if (selectedStyle === 'waveform') {
    analyser.getByteTimeDomainData(dataArray);
    drawWaveform(dataArray, bufferLength);
  }
}

function drawBars(dataArray, bufferLength) {
  const barWidth = (visualizerCanvas.width / bufferLength) * 1.5;
  let x = 0;
  const colorScheme = colorControl.value;
  for (let i = 0; i < bufferLength; i++) {
    const barHeight = dataArray[i] / 2;
    let r, g, b;
    if (colorScheme === 'orion') {
      r = barHeight + 50 * (i / bufferLength);
      g = 100 * (i / bufferLength);
      b = 150;
    } else if (colorScheme === 'retro') {
      r = 50;
      g = barHeight + 100 * (i / bufferLength);
      b = 50;
    } else {
      r = 150 * (i / bufferLength);
      g = barHeight;
      b = 200;
    }
    canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
    canvasCtx.fillRect(x, visualizerCanvas.height - barHeight, barWidth, barHeight);
    x += barWidth + 1;
  }
}

function drawWaveform(dataArray, bufferLength) {
  canvasCtx.lineWidth = 2;
  const colorScheme = colorControl.value;
  if (colorScheme === 'orion') canvasCtx.strokeStyle = 'rgb(147, 197, 253)';
  else if (colorScheme === 'retro') canvasCtx.strokeStyle = 'rgb(74, 222, 128)';
  else if (colorScheme === 'vaporwave') canvasCtx.strokeStyle = 'rgb(244, 114, 182)';
  canvasCtx.beginPath();
  const currentPlaybackRate = currentAudioSource ? currentAudioSource.playbackRate.value : 1.0;
  const sliceWidth = (visualizerCanvas.width * 1.0 / bufferLength) * currentPlaybackRate;
  let x = 0;
  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = v * visualizerCanvas.height / 2;
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
    x += sliceWidth;
  }
  canvasCtx.lineTo(Math.min(x, visualizerCanvas.width), visualizerCanvas.height / 2);
  canvasCtx.stroke();
}

// Preserve paragraphs & chat order
function appendMessage(sender, text, extraClass = 'text-white') {
  if (!conversationDiv) return;
  const senderClass =
    sender === 'You'
      ? 'font-semibold text-blue-300'
      : sender === 'Orion'
        ? 'font-semibold text-green-300'
        : sender === 'Council'
          ? 'font-semibold text-indigo-300'
          : 'font-semibold text-gray-300';

  const messageElement = document.createElement('div');
  messageElement.className = 'space-y-1';
  messageElement.innerHTML = `
    <p class="${senderClass}">${sender}</p>
    <p class="message-text ${extraClass}">${text}</p>
  `;
  conversationDiv.appendChild(messageElement);
  conversationDiv.scrollTop = conversationDiv.scrollHeight;
}

function sendTextMessage() {
  if (!chatInput) return;
  const text = chatInput.value.trim();
  if (!text || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  interruptButton && interruptButton.click();
  appendMessage('You', text);
  updateStatus('Processing...');
  const payload = {
    text_input: text,
    disable_tts: textToSpeechToggle ? !textToSpeechToggle.checked : false,
    temperature: parseFloat(tempControl.value),
    context_length: parseInt(contextControl.value),
    instructions: instructionsText ? instructionsText.value : "",
    session_id: orionSessionId || null,
    user_id: "local-user",
  };
  socket.send(JSON.stringify(payload));
  chatInput.value = '';
}

// --- Council HTTP path (Option B) ---
async function sendCouncilMessage() {
  if (!chatInput) return;
  const text = chatInput.value.trim();
  if (!text) {
    return;
  }

  // Show user message labeled for council
  appendMessage('You', text + '  (routed to Council)', 'text-white');
  updateStatus('Consulting Council...');

  const payload = {
    mode: 'council',
    temperature: parseFloat(tempControl.value),
    use_recall: true,
    messages: [
      { role: 'user', content: text }
    ],
  };

  const headers = {
    'Content-Type': 'application/json',
  };
  if (orionSessionId) {
    headers['X-Orion-Session-Id'] = orionSessionId;
  }

  try {
    const res = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errorText = await res.text().catch(() => '');
      const msg = `Council error (${res.status}) ${errorText}`;
      appendMessage('System', msg, 'text-red-400');
      updateStatus('Council error');
      return;
    }

    const data = await res.json();
    const councilText =
      (data && data.text) ||
      (data && data.raw && data.raw.text) ||
      '';

    if (councilText) {
      appendMessage('Council', councilText, 'text-indigo-200');
    } else {
      appendMessage('Council', '(no response)', 'text-gray-400');
    }

    chatInput.value = '';
    updateStatusBasedOnState();
  } catch (err) {
    const msg = `Council request failed: ${err.message}`;
    appendMessage('System', msg, 'text-red-400');
    updateStatus('Council error');
  }
}

// Recording helpers
async function startRecording() {
  if (interruptButton) interruptButton.click();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    liveMicSource = audioContext.createMediaStreamSource(stream);
    liveMicAnalyser = audioContext.createAnalyser();
    liveMicAnalyser.fftSize = 256;
    liveMicSource.connect(liveMicAnalyser);
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };
    mediaRecorder.onstop = async () => {
      if (socket && socket.readyState === WebSocket.OPEN && audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
          const payload = {
            audio: reader.result.split(',')[1],
            temperature: parseFloat(tempControl.value),
            context_length: parseInt(contextControl.value),
            instructions: instructionsText ? instructionsText.value : "",
            disable_tts: false,
            session_id: orionSessionId || null,
            user_id: "local-user",
          };
          socket.send(JSON.stringify(payload));
        };
      }
      stream.getTracks().forEach(track => track.stop());
      if (liveMicSource) liveMicSource.disconnect();
      liveMicSource = null;
    };
    audioChunks = [];
    mediaRecorder.start();
    updateStatus('Recording... Release to stop.');
    recordButton && recordButton.classList.add('pulse');
  } catch (err) {
    console.error('Error accessing microphone:', err);
    updateStatus('Error: Microphone access denied.');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    updateStatus('Processing...');
    recordButton && recordButton.classList.remove('pulse');
  }
}

// Canvas sizes
function setAllCanvasSizes() {
  if (visualizerContainer && visualizerCanvas) {
    visualizerCanvas.width = visualizerContainer.clientWidth;
    visualizerCanvas.height = visualizerContainer.clientHeight;
  }
  if (stateVisualizerContainer && stateVisualizerCanvas) {
    stateVisualizerCanvas.width = stateVisualizerContainer.clientWidth;
    stateVisualizerCanvas.height = stateVisualizerContainer.clientHeight;
  }
}

// Orion state particles
function createParticles(extra = 0) {
  if (!stateVisualizerCanvas || !stateCtx) return;
  particles = [];
  const total = baseParticleCount + extra;
  for (let i = 0; i < total; i++) {
    particles.push({
      x: Math.random() * stateVisualizerCanvas.width,
      y: Math.random() * stateVisualizerCanvas.height,
      radius: Math.random() * 2 + 1,
      baseVx: (Math.random() - 0.5) * 3.5,
      baseVy: (Math.random() - 0.5) * 4.5,
      vx: 0,
      vy: 0,
      color: `rgba(147, 197, 253, ${Math.random() * 0.5 + 0.3})`
    });
  }
}

function drawOrionState() {
  if (!stateCtx || !stateVisualizerCanvas) return;
  stateCtx.clearRect(0, 0, stateVisualizerCanvas.width, stateVisualizerCanvas.height);
  let speedBoost = 1.0;
  if (orionState === 'processing') speedBoost = 6.5;
  else if (orionState === 'speaking') speedBoost = 3.5;
  particles.forEach((p, index) => {
    p.vx = p.baseVx * speedBoost;
    p.vy = p.baseVy * speedBoost;
    p.x += p.vx;
    p.y += p.vy;
    if (p.x < 0) p.x = stateVisualizerCanvas.width;
    if (p.x > stateVisualizerCanvas.width) p.x = 0;
    if (p.y < 0) p.y = stateVisualizerCanvas.height;
    if (p.y > stateVisualizerCanvas.height) p.y = 0;
    stateCtx.beginPath();
    stateCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
    stateCtx.fillStyle = p.color;
    stateCtx.fill();
    for (let j = index + 1; j < particles.length; j++) {
      const other = particles[j];
      const dist = Math.hypot(p.x - other.x, p.y - other.y);
      if (dist < 70) {
        stateCtx.beginPath();
        stateCtx.moveTo(p.x, p.y);
        stateCtx.lineTo(other.x, other.y);
        stateCtx.strokeStyle = `rgba(255, 255, 255, ${0.4 - dist / 100})`;
        stateCtx.stroke();
      }
    }
  });
  requestAnimationFrame(drawOrionState);
}

// Collapse: load default JSON for Raw editor
async function loadCollapseTemplate() {
  const collapseInput = document.getElementById("collapseInput");
  if (!collapseInput) return;
  try {
    const res = await fetch(`${API_BASE_URL}/schema/collapse`);
    await res.json();
    const defaults = {
      observer: "DEMO",
      trigger: "UI test submission",
      observer_state: ["neutral"],
      field_resonance: "baseline",
      type: "test",
      emergent_entity: "demo_entity",
      summary: "This is a test collapse entry to validate the pipeline.",
      mantra: "hold the mirror",
      causal_echo: "none",
      environment: "dev"
    };
    collapseInput.value = JSON.stringify(defaults, null, 2);
  } catch (err) {
    console.error("❌ Failed to load collapse schema", err);
    collapseInput.value = '{"observer":"DEMO","trigger":"fallback"}';
  }
}

// Raw JSON submit
function setupCollapseForm() {
  const submitBtn = document.getElementById("collapseSubmit");
  const input = document.getElementById("collapseInput");
  if (!submitBtn || !input || !collapseStatus) return;

  submitBtn.addEventListener("click", async () => {
    let payload;
    try {
      payload = JSON.parse(input.value);
    } catch (err) {
      collapseStatus.textContent = "❌ Invalid JSON";
      collapseStatus.classList.remove("hidden", "text-green-400");
      collapseStatus.classList.add("text-red-400");
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/submit-collapse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        collapseStatus.textContent = "✅ Submitted successfully";
        collapseStatus.classList.remove("hidden", "text-red-400");
        collapseStatus.classList.add("text-green-400");
      } else {
        collapseStatus.textContent = "❌ Submission failed";
        if (data.error) collapseStatus.textContent += `: ${data.error}`;
        collapseStatus.classList.remove("hidden", "text-green-400");
        collapseStatus.classList.add("text-red-400");
      }
    } catch (err) {
      collapseStatus.textContent = `❌ Error: ${err.message}`;
      collapseStatus.classList.remove("hidden", "text-green-400");
      collapseStatus.classList.add("text-red-400");
    }
    setTimeout(() => collapseStatus.classList.add("hidden"), 3000);
  });
}

// Guided form submit
function setupGuidedCollapseForm() {
  const guidedBtn = document.getElementById("collapseGuidedSubmit");
  if (!guidedBtn || !collapseStatus) return;

  guidedBtn.addEventListener("click", async () => {
    const observer = (document.getElementById("collapseObserverInput")?.value || "DEMO").trim();
    const trigger = (document.getElementById("collapseTriggerInput")?.value || "").trim();
    const observerStateRaw = (document.getElementById("collapseObserverStateInput")?.value || "").trim();
    const fieldResonance = (document.getElementById("collapseFieldResonanceInput")?.value || "").trim();
    const type = (document.getElementById("collapseTypeInput")?.value || "").trim();
    const emergentEntity = (document.getElementById("collapseEntityInput")?.value || "").trim();
    const summary = (document.getElementById("collapseSummaryInput")?.value || "").trim();
    const mantra = (document.getElementById("collapseMantraInput")?.value || "").trim();
    const causalEcho = (document.getElementById("collapseCausalEchoInput")?.value || "").trim();
    const environment = (document.getElementById("collapseEnvironmentInput")?.value || "").trim();

    const observer_state = observerStateRaw
      ? observerStateRaw.split(',').map(s => s.trim()).filter(Boolean)
      : [];

    const payload = {
      observer,
      trigger,
      observer_state,
      field_resonance: fieldResonance,
      type,
      emergent_entity: emergentEntity,
      summary,
      mantra,
      causal_echo: causalEcho,
      environment
    };

    try {
      const res = await fetch(`${API_BASE_URL}/submit-collapse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        collapseStatus.textContent = "✅ Guided collapse submitted";
        collapseStatus.classList.remove("hidden", "text-red-400");
        collapseStatus.classList.add("text-green-400");
      } else {
        collapseStatus.textContent = "❌ Guided submission failed";
        if (data.error) collapseStatus.textContent += `: ${data.error}`;
        collapseStatus.classList.remove("hidden", "text-green-400");
        collapseStatus.classList.add("text-red-400");
      }
    } catch (err) {
      collapseStatus.textContent = `❌ Error: ${err.message}`;
      collapseStatus.classList.remove("hidden", "text-green-400");
      collapseStatus.classList.add("text-red-400");
    }
    setTimeout(() => collapseStatus.classList.add("hidden"), 3000);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadCollapseTemplate();
  setupCollapseForm();
  setupGuidedCollapseForm();

  chatInput = document.getElementById('chatInput');
  sendButton = document.getElementById('sendButton');
  textToSpeechToggle = document.getElementById('textToSpeechToggle');
  councilButton = document.getElementById('councilButton');

  if (sendButton) {
    sendButton.addEventListener('click', sendTextMessage);
  }

  if (councilButton) {
    councilButton.addEventListener('click', sendCouncilMessage);
  }

  if (chatInput) {
    chatInput.addEventListener('keydown', (event) => {
      // Shift+Enter → newline (let browser insert line break)
      if (event.key === 'Enter' && event.shiftKey) {
        return;
      }
      // Enter (without Shift) → send to Brain (WS)
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendTextMessage();
      }
    });
  }
});

// --- Final Setup ---
if (recordButton) {
  recordButton.addEventListener('mousedown', startRecording);
  recordButton.addEventListener('mouseup', stopRecording);
  recordButton.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startRecording();
  });
  recordButton.addEventListener('touchend', stopRecording);
}

window.addEventListener('load', () => {
  (async () => {
    await initSession();
    setupWebSocket();
    setAllCanvasSizes();
    if (stateVisualizerCanvas && stateVisualizerCanvas.width > 0) {
      createParticles();
      drawOrionState();
    }
  })();
});

window.addEventListener('resize', setAllCanvasSizes);
