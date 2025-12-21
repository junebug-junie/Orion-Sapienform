// ───────────────────────────────────────────────────────────────
// Global State (Keep these outside so they persist)
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
let liveMicAnalyser;
let liveMicSource;
let particles = [];
let baseParticleCount = 200;
let visionIsFloating = false;
let currentMode = "brain";
let selectedPacks = [];
let orionSessionId = null;

// ───────────────────────────────────────────────────────────────
// Main Initialization (Waits for HTML to load)
// ───────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  console.log("[Main] DOM Content Loaded - Initializing UI...");

  // --- 1. Element References (Scoped locally) ---
  const recordButton = document.getElementById('recordButton');
  const interruptButton = document.getElementById('interruptButton');
  const statusDiv = document.getElementById('status');
  const conversationDiv = document.getElementById('conversation');
  const chatInput = document.getElementById('chatInput');
  const sendButton = document.getElementById('sendButton');
  const textToSpeechToggle = document.getElementById('textToSpeechToggle');
  
  // Controls
  const speedControl = document.getElementById('speedControl');
  const speedValue = document.getElementById('speedValue');
  const tempControl = document.getElementById('tempControl');
  const tempValue = document.getElementById('tempValue');
  const styleControl = document.getElementById('styleControl');
  const colorControl = document.getElementById('colorControl');
  const contextControl = document.getElementById('contextControl');
  const contextValue = document.getElementById('contextValue');
  const clearButton = document.getElementById('clearButton');
  const copyButton = document.getElementById('copyButton');

  // Settings Panel
  const settingsToggle = document.getElementById('settingsToggle');
  const settingsPanel = document.getElementById('settingsPanel');
  const settingsClose = document.getElementById('settingsClose');

  // Visualizers
  const visualizerCanvas = document.getElementById('visualizer');
  const canvasCtx = visualizerCanvas ? visualizerCanvas.getContext('2d') : null;
  const visualizerContainer = document.getElementById('visualizerContainer');
  const stateVisualizerCanvas = document.getElementById('stateVisualizer');
  const stateCtx = stateVisualizerCanvas ? stateVisualizerCanvas.getContext('2d') : null;
  const stateVisualizerContainer = document.getElementById('stateVisualizerContainer');

  // Vision
  const visionPopoutButton = document.getElementById("visionPopoutButton");
  const visionDockedContainer = document.getElementById("visionDockedContainer");
  const visionFloatingContainer = document.getElementById("visionFloatingContainer");
  const visionCloseFloatingButton = document.getElementById("visionCloseFloating");
  const visionSourceSelect = document.getElementById("visionSource");

  // Collapse Mirror
  const collapseModeGuided = document.getElementById('collapseModeGuided');
  const collapseModeRaw = document.getElementById('collapseModeRaw');
  const collapseGuidedSection = document.getElementById('collapseGuidedSection');
  const collapseRawSection = document.getElementById('collapseRawSection');
  const collapseStatus = document.getElementById('collapseStatus');
  const collapseTooltipToggle = document.getElementById('collapseTooltipToggle');
  const collapseTooltip = document.getElementById('collapseTooltip');

  // --- 2. Event Listeners ---

  // Recording
  if (recordButton) {
    recordButton.addEventListener('mousedown', startRecording);
    recordButton.addEventListener('mouseup', stopRecording);
    recordButton.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    recordButton.addEventListener('touchend', stopRecording);
  }

  // Send Message
  if (sendButton) {
    sendButton.addEventListener('click', sendTextMessage);
  }
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendTextMessage();
    });
  }

  // Interrupt
  if (interruptButton) {
    interruptButton.addEventListener('click', () => {
      if (currentAudioSource) currentAudioSource.stop();
      audioQueue = [];
      isPlayingAudio = false;
      updateStatusBasedOnState();
      interruptButton.classList.add('hidden');
    });
  }

  // Controls UI Updates
  if (tempControl && tempValue) {
    tempControl.addEventListener('input', () => tempValue.textContent = parseFloat(tempControl.value).toFixed(1));
  }
  if (contextControl && contextValue) {
    contextControl.addEventListener('input', () => contextValue.textContent = contextControl.value);
  }
  if (speedControl && speedValue) {
    speedControl.addEventListener('input', () => {
      const actualSpeed = parseFloat(speedControl.value);
      const minSpeed = 0.97, maxSpeed = 1.2;
      speedValue.textContent = ((actualSpeed - minSpeed) / (maxSpeed - minSpeed)).toFixed(2);
      if (currentAudioSource) currentAudioSource.playbackRate.value = actualSpeed;
    });
  }

  // Chat Utils
  if (clearButton && conversationDiv) {
    clearButton.addEventListener('click', () => conversationDiv.innerHTML = '');
  }
  if (copyButton && conversationDiv) {
    copyButton.addEventListener('click', () => {
      const text = conversationDiv.innerText;
      navigator.clipboard.writeText(text).then(() => {
        updateStatus('Conversation copied!');
        setTimeout(() => updateStatusBasedOnState(), 2000);
      });
    });
  }

  // Settings Panel Toggle
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

  // Mode Toggles
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

      if (currentMode === 'brain') updateStatus('Brain mode: Press mic to speak or type.');
      else if (currentMode === 'agentic') updateStatus('Agentic mode: Type to run tools.');
      else updateStatus('Council mode: Consult the swarm.');
    });
  });

  // Pack Selection (Agentic)
  const packButtons = document.querySelectorAll('.pack-btn');
  function updatePackButtonStyles() {
    packButtons.forEach((btn) => {
      const packName = btn.dataset.pack;
      if (selectedPacks.includes(packName)) {
        btn.classList.remove('bg-gray-700', 'text-gray-200');
        btn.classList.add('bg-emerald-600', 'text-white');
      } else {
        btn.classList.remove('bg-emerald-600', 'text-white');
        btn.classList.add('bg-gray-700', 'text-gray-200');
      }
    });
  }
  
  packButtons.forEach((btn) => {
    const packName = btn.dataset.pack;
    if (btn.dataset.default === 'true' && !selectedPacks.includes(packName)) {
      selectedPacks.push(packName);
    }
    btn.addEventListener('click', () => {
      if (!packName) return;
      const idx = selectedPacks.indexOf(packName);
      if (idx >= 0) selectedPacks.splice(idx, 1);
      else selectedPacks.push(packName);
      updatePackButtonStyles();
    });
  });
  updatePackButtonStyles();

  // Vision UI Logic
  function updateVisionUi() {
    if (!visionDockedContainer || !visionFloatingContainer) return;
    const value = visionSourceSelect ? visionSourceSelect.value : "";
    let endpoint = null;
    if (value === "gopro-1") endpoint = VISION_EDGE_BASE + "/stream.mjpg";
    else if (value === "simulated") endpoint = "/static/img/vision-simulated.gif";

    const updateImg = (container) => {
      let img = container.querySelector("img[data-role='vision-feed']");
      if (!img) {
        img = document.createElement("img");
        img.dataset.role = "vision-feed";
        img.className = "w-full h-full object-cover";
        container.appendChild(img);
      }
      img.src = endpoint + "?ts=" + Date.now();
      const ph = container.querySelector("#visionPlaceholder");
      if (ph) ph.remove();
    };

    if (!endpoint) {
      visionFloatingContainer.classList.add("hidden");
      visionDockedContainer.classList.remove("opacity-30");
      if (visionPopoutButton) visionPopoutButton.textContent = "Pop Out";
      return;
    }

    updateImg(visionDockedContainer);
    if (visionIsFloating) {
      visionFloatingContainer.classList.remove("hidden");
      visionDockedContainer.classList.add("opacity-30");
      if (visionPopoutButton) visionPopoutButton.textContent = "Dock";
      const target = visionFloatingContainer.querySelector(".flex-1") || visionFloatingContainer;
      updateImg(target);
    } else {
      visionFloatingContainer.classList.add("hidden");
      visionDockedContainer.classList.remove("opacity-30");
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
  updateVisionUi();

  // Collapse UI Logic
  function setCollapseMode(mode) {
    if (!collapseModeGuided || !collapseModeRaw) return;
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

  if (collapseModeGuided) collapseModeGuided.addEventListener('click', () => setCollapseMode('guided'));
  if (collapseModeRaw) collapseModeRaw.addEventListener('click', () => setCollapseMode('raw'));
  setCollapseMode('guided');

  // Tooltip
  if (collapseTooltipToggle) {
    collapseTooltipToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      collapseTooltip.classList.toggle('hidden');
    });
    document.addEventListener('click', (e) => {
      if (collapseTooltip && !collapseTooltip.classList.contains('hidden') && !collapseTooltip.contains(e.target) && e.target !== collapseTooltipToggle) {
        collapseTooltip.classList.add('hidden');
      }
    });
  }

  // Load Schema
  async function loadCollapseTemplate() {
    const input = document.getElementById("collapseInput");
    if (!input) return;
    try {
      const res = await fetch(`${API_BASE_URL}/schema/collapse`);
      await res.json(); 
      input.value = JSON.stringify({
        observer: "DEMO",
        trigger: "UI test",
        observer_state: ["neutral"],
        type: "test",
        summary: "Test entry",
        environment: "dev"
      }, null, 2);
    } catch (e) { console.error(e); }
  }
  loadCollapseTemplate();

  // Submit Logic
  const handleCollapseSubmit = async (payload) => {
    if (!collapseStatus) return;
    try {
      const res = await fetch(`${API_BASE_URL}/submit-collapse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok && data.success) {
        collapseStatus.textContent = "✅ Submitted";
        collapseStatus.className = "text-sm text-green-400";
      } else {
        collapseStatus.textContent = `❌ Failed: ${data.error || 'Unknown'}`;
        collapseStatus.className = "text-sm text-red-400";
      }
    } catch (err) {
      collapseStatus.textContent = `❌ Error: ${err.message}`;
      collapseStatus.className = "text-sm text-red-400";
    }
    collapseStatus.classList.remove('hidden');
    setTimeout(() => collapseStatus.classList.add('hidden'), 3000);
  };

  const collapseSubmit = document.getElementById("collapseSubmit");
  if (collapseSubmit) {
    collapseSubmit.addEventListener("click", () => {
      const input = document.getElementById("collapseInput");
      try {
        handleCollapseSubmit(JSON.parse(input.value));
      } catch (e) { alert("Invalid JSON"); }
    });
  }

  const collapseGuidedSubmit = document.getElementById("collapseGuidedSubmit");
  if (collapseGuidedSubmit) {
    collapseGuidedSubmit.addEventListener("click", () => {
      const getVal = (id) => (document.getElementById(id)?.value || "").trim();
      const payload = {
        observer: getVal("collapseObserverInput") || "DEMO",
        trigger: getVal("collapseTriggerInput"),
        observer_state: getVal("collapseObserverStateInput").split(',').map(s => s.trim()).filter(Boolean),
        field_resonance: getVal("collapseFieldResonanceInput"),
        type: getVal("collapseTypeInput"),
        emergent_entity: getVal("collapseEntityInput"),
        summary: getVal("collapseSummaryInput"),
        mantra: getVal("collapseMantraInput"),
        causal_echo: getVal("collapseCausalEchoInput"),
        environment: getVal("collapseEnvironmentInput")
      };
      handleCollapseSubmit(payload);
    });
  }

  // --- 3. Start Core Services ---
  (async () => {
    await initSession();
    setupWebSocket();
    setAllCanvasSizes();
    if (stateVisualizerCanvas && stateVisualizerCanvas.width > 0) {
      createParticles();
      drawOrionState();
    }
  })();

  // --- 4. Helpers that need scope access ---
  
  function updateStatus(msg) { if (statusDiv) statusDiv.textContent = msg; }
  
  function updateStatusBasedOnState() {
    if (orionState === 'idle') updateStatus('Ready.');
    else if (orionState === 'speaking') updateStatus('Speaking...');
    else if (orionState === 'processing') updateStatus('Processing...');
  }

  function appendMessage(sender, text, colorClass = 'text-white') {
    if (!conversationDiv) return;
    const div = document.createElement('div');
    const color = sender === 'You' ? 'text-blue-300' : 'text-green-300';
    div.innerHTML = `<p class="font-bold ${color}">${sender}</p><p class="${colorClass}">${text}</p>`;
    div.className = "mb-2";
    conversationDiv.appendChild(div);
    conversationDiv.scrollTop = conversationDiv.scrollHeight;
  }

  function sendTextMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    appendMessage('You', text);
    chatInput.value = '';
    
    if (currentMode === 'brain') {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
          text_input: text,
          mode: 'brain',
          session_id: orionSessionId,
          disable_tts: textToSpeechToggle ? !textToSpeechToggle.checked : false
        }));
        updateStatus('Sent to Brain...');
      } else {
        appendMessage('System', 'WebSocket disconnected', 'text-red-400');
      }
    } else {
      // Agentic/Council via HTTP
      const label = currentMode === 'agentic' ? 'Agentic' : 'Council';
      updateStatus(`${label} thinking...`);
      fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Orion-Session-Id': orionSessionId || '' 
        },
        body: JSON.stringify({ 
          mode: currentMode, 
          messages: [{role: 'user', content: text}],
          packs: selectedPacks
        })
      })
      .then(r => r.json())
      .then(d => {
        const reply = d.text || (d.raw && d.raw.text) || JSON.stringify(d);
        appendMessage(label, reply, 'text-indigo-200');
        updateStatus('Ready.');
      })
      .catch(e => {
        appendMessage('System', e.message, 'text-red-400');
        updateStatus('Error.');
      });
    }
  }

  // Recording Logic
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
               mode: 'brain',
               session_id: orionSessionId
             }));
             updateStatus('Audio sent.');
           }
        };
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
      updateStatus('Recording...');
      recordButton.classList.add('pulse');
    } catch (e) { console.error(e); updateStatus('Mic Error'); }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      recordButton.classList.remove('pulse');
    }
  }

  // WebSocket Setup
  function setupWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${proto}//${window.location.host}/ws`);
    socket.onopen = () => updateStatus('Connected.');
    socket.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.transcript && !d.is_text_input) appendMessage('You', d.transcript);
      if (d.llm_response) appendMessage('Orion', d.llm_response);
      if (d.state) { orionState = d.state; updateStatusBasedOnState(); }
      if (d.audio_response) { audioQueue.push(d.audio_response); processAudioQueue(); }
    };
  }

  // Audio Playback
  function processAudioQueue() {
    if (isPlayingAudio || !audioQueue.length) return;
    isPlayingAudio = true;
    playAudio(audioQueue.shift());
  }

  async function playAudio(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for(let i=0; i<bin.length; i++) arr[i] = bin.charCodeAt(i);
    const buf = await audioContext.decodeAudioData(arr.buffer);
    const src = audioContext.createBufferSource();
    src.buffer = buf;
    const gain = audioContext.createGain();
    src.connect(gain);
    gain.connect(audioContext.destination);
    
    // Viz
    analyser = audioContext.createAnalyser();
    src.connect(analyser);
    drawVisualizer();

    src.start(0);
    currentAudioSource = src;
    src.onended = () => {
      isPlayingAudio = false;
      cancelAnimationFrame(animationFrameId);
      processAudioQueue();
    };
  }
  
  // Viz Loop
  function drawVisualizer() {
    if (!analyser || !canvasCtx) return;
    const bufLen = analyser.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyser.getByteFrequencyData(data);
    canvasCtx.fillStyle = '#1f2937';
    canvasCtx.fillRect(0,0,visualizerCanvas.width, visualizerCanvas.height);
    const barW = (visualizerCanvas.width / bufLen) * 2.5;
    let x = 0;
    for(let i=0; i<bufLen; i++) {
      const h = data[i] / 2;
      canvasCtx.fillStyle = `rgb(${h+50}, 100, 150)`;
      canvasCtx.fillRect(x, visualizerCanvas.height - h, barW, h);
      x += barW + 1;
    }
    animationFrameId = requestAnimationFrame(drawVisualizer);
  }

  // Particles
  function createParticles() {
    particles = Array.from({length: 100}, () => ({
      x: Math.random()*stateVisualizerCanvas.width,
      y: Math.random()*stateVisualizerCanvas.height,
      vx: (Math.random()-0.5)*2,
      vy: (Math.random()-0.5)*2
    }));
  }
  
  function drawOrionState() {
    if(!stateCtx) return;
    stateCtx.clearRect(0,0,stateVisualizerCanvas.width, stateVisualizerCanvas.height);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if(p.x<0||p.x>stateVisualizerCanvas.width) p.vx*=-1;
      if(p.y<0||p.y>stateVisualizerCanvas.height) p.vy*=-1;
      stateCtx.fillStyle = 'rgba(147, 197, 253, 0.5)';
      stateCtx.beginPath();
      stateCtx.arc(p.x, p.y, 2, 0, Math.PI*2);
      stateCtx.fill();
    });
    requestAnimationFrame(drawOrionState);
  }

  // --- Session Init ---
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

  // Canvas Resize
  function setAllCanvasSizes() {
    if (visualizerCanvas) {
      visualizerCanvas.width = visualizerContainer.clientWidth;
      visualizerCanvas.height = visualizerContainer.clientHeight;
    }
    if (stateVisualizerCanvas) {
      stateVisualizerCanvas.width = stateVisualizerContainer.clientWidth;
      stateVisualizerCanvas.height = stateVisualizerContainer.clientHeight;
    }
  }
  window.addEventListener('resize', setAllCanvasSizes);

});
