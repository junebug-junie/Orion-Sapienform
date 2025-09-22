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
const canvasCtx = visualizerCanvas.getContext('2d');
const visualizerContainer = document.getElementById('visualizerContainer');
const stateVisualizerCanvas = document.getElementById('stateVisualizer');
const stateCtx = stateVisualizerCanvas.getContext('2d');
const stateVisualizerContainer = document.getElementById('stateVisualizerContainer');
const contextControl = document.getElementById('contextControl');
const contextValue = document.getElementById('contextValue');
const instructionsText = document.getElementById('instructions');
const clearButton = document.getElementById('clearButton');
const copyButton = document.getElementById('copyButton');

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

// --- Event Listeners ---
tempControl.addEventListener('input', () => {
    tempValue.textContent = parseFloat(tempControl.value).toFixed(1);
});
contextControl.addEventListener('input', () => {
    contextValue.textContent = contextControl.value;
});
speedControl.addEventListener('input', () => {
    const actualSpeed = parseFloat(speedControl.value);
    const minSpeed = 0.97; const maxSpeed = 1.2;
    const normalizedValue = (actualSpeed - minSpeed) / (maxSpeed - minSpeed);
    speedValue.textContent = normalizedValue.toFixed(2);
    if (currentAudioSource) currentAudioSource.playbackRate.value = actualSpeed;
});
clearButton.addEventListener('click', () => {
    conversationDiv.innerHTML = '';
});
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
interruptButton.addEventListener('click', () => {
    if (currentAudioSource) currentAudioSource.stop();
    audioQueue = [];
    isPlayingAudio = false;
    updateStatus('Ready. Press the button to speak.');
    interruptButton.classList.add('hidden');
});

// --- Core Functions ---
function updateStatus(newStatus) {
    statusDiv.textContent = newStatus;
}
function updateStatusBasedOnState() {
    if (orionState === 'idle') updateStatus('Ready. Press the button to speak.');
    else if (orionState === 'speaking') updateStatus('Playing response...');
}

function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        updateStatus('Connection established. Hold button to speak.');
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.transcript) {
            appendMessage('You', data.transcript);
        }
        else if (data.llm_response) {
            appendMessage('Assistant', data.llm_response);

            // 🔥 Adjust density based on token count
            if (data.tokens) {
                baseParticleCount = 200 + Math.min(data.tokens, 200); // cap at +200
                createParticles();
            }
        }
        else if (data.audio_response) {
            audioQueue.push(data.audio_response);
            processAudioQueue();
        }
        else if (data.state) {
            // 🔥 Update Orion state (processing/speaking/idle)
            orionState = data.state;
            updateStatus(`State: ${orionState}`);
        }
        else if (data.error) {
            const errorMessage = `Error: ${data.error}`;
            updateStatus(errorMessage);
            appendMessage('System', errorMessage, 'text-red-400');
            setTimeout(() => updateStatus('Ready. Press the button to speak.'), 3000);
        }
    };

    socket.onclose = () => {
        updateStatus('Connection lost. Please refresh.');
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
        interruptButton.classList.remove('hidden');
        drawVisualizer();
        source.onended = () => {
            currentAudioSource = null;
            cancelAnimationFrame(animationFrameId);
            canvasCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
            isPlayingAudio = false;
            processAudioQueue();
            if (audioQueue.length === 0) {
                updateStatus('Ready. Press the button to speak.');
                interruptButton.classList.add('hidden');
            }
        };
    } catch (error) {
        console.error("Failed to play audio:", error);
        updateStatus('Error playing audio response.');
        isPlayingAudio = false;
    }
}

function drawVisualizer() {
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
        if (colorScheme === 'orion') { r = barHeight + 50 * (i/bufferLength); g = 100 * (i/bufferLength); b = 150; } 
        else if (colorScheme === 'retro') { r = 50; g = barHeight + 100 * (i/bufferLength); b = 50; } 
        else { r = 150 * (i/bufferLength); g = barHeight; b = 200; }
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
    else canvasCtx.strokeStyle = 'rgb(244, 114, 182)';
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

function appendMessage(sender, text) {
    const senderClass = sender === 'You' ? 'font-semibold text-blue-300' : 'font-semibold text-green-300';
    const messageElement = document.createElement('div');
    messageElement.innerHTML = `<p class="${senderClass}">${sender}</p><p class="text-white">${text}</p>`;
    conversationDiv.appendChild(messageElement);
    conversationDiv.scrollTop = conversationDiv.scrollHeight;
}

async function startRecording() {
    interruptButton.click();
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
                        instructions: instructionsText.value
                    };
                    socket.send(JSON.stringify(payload));
                }
            }
            stream.getTracks().forEach(track => track.stop());
            if (liveMicSource) liveMicSource.disconnect();
            liveMicSource = null;
        };
        audioChunks = [];
        mediaRecorder.start();
        updateStatus('Recording... Release to stop.');
        recordButton.classList.add('pulse');
    } catch (err) {
        console.error('Error accessing microphone:', err);
        updateStatus('Error: Microphone access denied.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        updateStatus('Processing...');
        recordButton.classList.remove('pulse');
    }
}

function setAllCanvasSizes() {
    visualizerCanvas.width = visualizerContainer.clientWidth;
    visualizerCanvas.height = visualizerContainer.clientHeight;
    stateVisualizerCanvas.width = stateVisualizerContainer.clientWidth;
    stateVisualizerCanvas.height = stateVisualizerContainer.clientHeight;
}

function createParticles(extra = 0) {
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

        // draw neuron
        stateCtx.beginPath();
        stateCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        stateCtx.fillStyle = p.color;
        stateCtx.fill();

        // draw connections
        for (let j = index + 1; j < particles.length; j++) {
            const other = particles[j];
            const dist = Math.hypot(p.x - other.x, p.y - other.y);
            if (dist < 70) {
                stateCtx.beginPath();
                stateCtx.moveTo(p.x, p.y);
                stateCtx.lineTo(other.x, other.y);
                stateCtx.strokeStyle = `rgba(255, 255, 255, ${0.4 - dist/100})`;
                stateCtx.stroke();
            }
        }
    });

    requestAnimationFrame(drawOrionState);
}

// --- Final Setup ---
recordButton.addEventListener('mousedown', startRecording);
recordButton.addEventListener('mouseup', stopRecording);
recordButton.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
recordButton.addEventListener('touchend', stopRecording);
window.addEventListener('load', () => {
    setupWebSocket();
    setAllCanvasSizes();
    if (stateVisualizerCanvas.width > 0) {
        createParticles();
        drawOrionState();
    }
});
window.addEventListener('resize', setAllCanvasSizes);
