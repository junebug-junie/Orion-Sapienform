// Read-only Cabinet instrument. Pure source adapter/mapping also run under Node.
(function (root) {
  'use strict';
  const HOLD_MS = 5000;
  const PRESET = Object.freeze({version: 'cabinet-live-1', smoothingSec: 0.15, deadband: 0,
    historyGapSec: 90, voices: Object.freeze([
      Object.freeze({voice: 'magnetic', key: 'cabinet_magnetic_ut', channel: 'magnetic', unit: 'µT', lo: 20, hi: 60, soundLo: 110, soundHi: 440, gain: 0.12, approved: true}),
      Object.freeze({voice: 'lidar', key: 'cabinet_lidar_mm', channel: 'lidar', unit: 'mm', lo: 0, hi: 4000, soundLo: 0.5, soundHi: 4, gain: 0.10, approved: false}),
      Object.freeze({voice: 'vibration', key: 'cabinet_vibration_g', channel: 'imu', unit: '|g−1|', lo: 0, hi: 0.2, onset: 0.10, rearm: 0.06, gain: 0.10, approved: false})
    ])});
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const object = x => x && typeof x === 'object' && !Array.isArray(x);
  function timestamp(s) {
    if (typeof s !== 'string') return NaN;
    // Python treats naive source timestamps as UTC.
    return Date.parse(/(?:Z|[+-]\d\d:\d\d)$/i.test(s) ? s : s + 'Z');
  }
  function frameFromPayload(payload, now = Date.now(), preset = PRESET) {
    return {version: 1, mode: 'live', readings: preset.voices.map(def => {
      const reading = {key: def.key, value: null, unit: def.unit, observedAt: null, ageSec: null, sourceId: null, status: 'missing'};
      if (!payload || payload.ok !== true || !finite(payload.age_sec) || payload.age_sec < 0 || payload.age_sec > 5 || payload.snapshot?.status !== 'ok') return reading;
      // Match newest non-stale channel selection, including stable first-source ties.
      let winner = null;
      for (const [id, source] of Object.entries(payload.sources || {})) {
        const snap = source?.snapshot;
        if (!object(snap?.frame?.[def.channel])) continue;
        const at = timestamp(snap.received_at);
        const age = now - at;
        if (['stale', 'error', 'missing'].includes(String(snap.status || '').toLowerCase()) || !finite(at) || age < 0 || age > HOLD_MS) {
          reading.status = 'stale'; continue;
        }
        if (!winner || at > winner.at) winner = {id, snap, at};
      }
      if (!winner) return reading;
      const block = winner.snap.frame[def.channel];
      let value;
      if (def.voice === 'magnetic') value = block.magnitude_ut;
      if (def.voice === 'lidar' && block.status === 0) value = block.distance_mm;
      if (def.voice === 'vibration' && [block.accel_x, block.accel_y, block.accel_z].every(finite)) {
        value = Math.abs(Math.hypot(block.accel_x, block.accel_y, block.accel_z) / 9.80665 - 1);
      }
      return {...reading, value: finite(value) ? value : null, observedAt: winner.snap.received_at,
        sourceId: winner.id, ageSec: (now - winner.at) / 1000, status: finite(value) && value >= 0 ? 'ok' : 'invalid'};
    })};
  }
  function mapFrame(frame, preset = PRESET) {
    return preset.voices.map(def => {
      const r = frame.readings.find(r => r.key === def.key);
      const target = {voice: def.voice, enabled: false, gain: 0, reason: r?.status || 'missing', normalized: null, clipped: false};
      if (![def.lo, def.hi, def.gain].every(finite) || def.hi <= def.lo || def.gain < 0 || def.gain > 0.2) return {...target, reason: 'invalid preset'};
      if (!r || r.status !== 'ok' || !finite(r.value)) return target;
      if (r.unit !== def.unit) return {...target, reason: 'invalid unit'};
      const u = Math.max(0, Math.min(1, (r.value - def.lo) / (def.hi - def.lo)));
      target.normalized = u;
      target.clipped = r.value < def.lo || r.value > def.hi;
      if (!def.approved) return {...target, reason: 'calibration pending — voice disabled'};
      if (def.voice !== 'vibration' && (![def.soundLo, def.soundHi].every(finite) || def.soundLo <= 0 || def.soundHi < def.soundLo)) return {...target, reason: 'invalid sound bounds'};
      Object.assign(target, {enabled: true, gain: def.gain, reason: target.clipped ? 'clipped to fixed bounds' : 'mapped level'});
      if (def.voice === 'magnetic') target.frequencyHz = def.soundLo * (def.soundHi / def.soundLo) ** u;
      if (def.voice === 'lidar') target.pulseHz = def.soundLo + (def.soundHi - def.soundLo) * (1 - u);
      return target;
    });
  }
  class Onset {
    constructor() { this.reset(); }
    reset() { this.at = null; this.source = null; this.armed = false; }
    update(r, def) {
      if (!r || r.status !== 'ok' || !finite(r.value)) { this.reset(); return false; }
      const at = timestamp(r.observedAt);
      if (!finite(at)) { this.reset(); return false; }
      if (this.source !== r.sourceId || this.at === null || at < this.at || at - this.at > HOLD_MS) {
        this.source = r.sourceId; this.at = at; this.armed = r.value <= def.rearm; return false;
      }
      if (at === this.at) return false;
      this.at = at;
      if (r.value <= def.rearm) this.armed = true;
      if (r.value >= def.onset && this.armed) { this.armed = false; return true; }
      return false;
    }
  }
  // Dedicated context: never touches app.js TTS or its queue/context.
  class Instrument {
    constructor({contextFactory, now = Date.now, preset = PRESET, onChange = () => {}} = {}) {
      this.contextFactory = contextFactory || (() => new (root.AudioContext || root.webkitAudioContext)());
      this.now = now; this.preset = preset; this.onChange = onChange; this.context = null;
      this.armed = false; this.generation = 0; this.lastPoll = null; this.payload = null;
      this.muted = new Set(); this.solo = null; this.onset = new Onset(); this.nodes = new Set();
      this.targets = []; this.trace = null; this.nextPulse = 0; this.reason = 'Stopped';
    }
    receive(payload) {
      this.payload = payload; this.lastPoll = this.now(); this.update();
    }
    fail() { this.payload = null; this.stop('Poll failed — press Listen to retry'); }
    update() {
      const frame = frameFromPayload(this.payload, this.now(), this.preset);
      this.frame = frame;
      this.targets = mapFrame(frame, this.preset).map(t => this.muted.has(t.voice) || (this.solo && this.solo !== t.voice) ? {...t, enabled: false, gain: 0, reason: 'muted / solo'} : t);
      const def = this.preset.voices.find(v => v.voice === 'vibration');
      const vibration = this.targets.find(t => t.voice === 'vibration');
      const hit = this.onset.update(vibration?.enabled ? frame.readings.find(r => r.key === def.key) : null, def);
      if (this.armed && this.context?.state === 'running') {
        const t = this.context.currentTime;
        const magnetic = this.targets.find(v => v.voice === 'magnetic');
        this.ramp(this.toneGain.gain, magnetic?.enabled ? magnetic.gain : 0, t);
        if (magnetic?.enabled) this.ramp(this.tone.frequency, magnetic.frequencyHz, t);
        if (hit) this.pulse(t, 90, vibration.gain, 0.12, 'vibration');
        for (const n of this.nodes) {
          if (!this.targets.find(v => v.voice === n.voice)?.enabled) this.ramp(n.gain.gain, 0, t);
        }
      }
      if (this.trace && this.trace.length < 600) this.trace.push({mappingVersion: this.preset.version, capturedAt: new Date(this.now()).toISOString(), frame, targets: this.targets.map(t => ({...t})), armed: this.armed, onset: hit});
      this.onChange();
    }
    ramp(param, value, at) {
      if (param.cancelAndHoldAtTime) param.cancelAndHoldAtTime(at);
      else { param.cancelScheduledValues(at); param.setValueAtTime(param.value, at); }
      param.linearRampToValueAtTime(value, at + this.preset.smoothingSec);
    }
    async listen() {
      if (this.armed || this.starting) return;
      this.starting = true;
      const generation = ++this.generation;
      let context;
      try {
        context = this.contextFactory(); this.context = context;
        await context.resume();
        if (generation !== this.generation) { await context.close(); return; }
        if (context.state !== 'running') throw new Error('Audio suspended; press Listen again');
        this.master = context.createGain(); this.master.gain.value = 0; this.master.connect(context.destination);
        this.tone = context.createOscillator(); this.toneGain = context.createGain(); this.toneGain.gain.value = 0;
        this.tone.connect(this.toneGain); this.toneGain.connect(this.master); this.tone.start();
        this.onset.reset(); this.armed = true; this.reason = 'Listening'; this.nextPulse = context.currentTime;
        this.ramp(this.master.gain, 0.35, context.currentTime);
        this.update(); this.timer = setInterval(() => this.tick(), 25);
        this.tick();
      } catch (err) {
        if (generation === this.generation) this.stop(err.message || 'Audio unavailable');
      } finally { if (generation === this.generation) this.starting = false; }
    }
    tick() {
      if (!this.armed) return;
      if (this.lastPoll === null || this.now() - this.lastPoll >= HOLD_MS) { this.stop('Fresh poll timed out'); return; }
      if (this.context.state !== 'running') { this.stop('Audio suspended — press Listen to retry'); return; }
      // Enforce each source's own timestamp between polls, not only fetch age.
      if (this.frame.readings.some(r => r.status === 'ok' && this.now() - timestamp(r.observedAt) > HOLD_MS)) this.update();
      if (!mapFrame(this.frame, this.preset).some(t => t.enabled)) { this.stop('No fresh calibrated voices'); return; }
      const lidar = this.targets.find(t => t.voice === 'lidar');
      const t = this.context.currentTime;
      if (lidar?.enabled && this.nextPulse <= t + 0.05) {
        this.pulse(Math.max(t, this.nextPulse), 330, lidar.gain, 0.08, 'lidar');
        this.nextPulse = Math.max(t, this.nextPulse) + 1 / lidar.pulseHz;
      } else if (!lidar?.enabled) this.nextPulse = t;
    }
    pulse(at, frequency, level, duration, voice) {
      const context = this.context, oscillator = context.createOscillator(), gain = context.createGain();
      oscillator.frequency.value = frequency;
      oscillator.connect(gain); gain.connect(this.master);
      gain.gain.setValueAtTime(0, at); gain.gain.linearRampToValueAtTime(level, at + 0.008);
      gain.gain.linearRampToValueAtTime(0, at + duration);
      const node = {oscillator, gain, voice}; this.nodes.add(node);
      oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); this.nodes.delete(node); };
      oscillator.start(at); oscillator.stop(at + duration + 0.01);
    }
    stop(reason = 'Stopped') {
      ++this.generation; this.starting = false; this.armed = false; this.reason = reason;
      clearInterval(this.timer); this.timer = null; this.onset.reset();
      const context = this.context, master = this.master, tone = this.tone;
      this.context = null; this.master = null; this.tone = null;
      if (context) {
        const at = context.currentTime;
        if (master) this.ramp(master.gain, 0, at);
        if (tone) { tone.stop(at + 0.2); tone.onended = () => tone.disconnect(); }
        for (const n of this.nodes) n.oscillator.stop(at + 0.2);
        this.nodes.clear();
        setTimeout(() => { if (master) master.disconnect(); context.close().catch(() => {}); }, 220);
      }
      this.onChange();
    }
    mute(voice, muted) { muted ? this.muted.add(voice) : this.muted.delete(voice); this.update(); }
    setSolo(voice) { this.solo = this.solo === voice ? null : voice; this.update(); }
    startTrace() { this.trace = []; this.update(); }
    takeTrace() { const data = {preset: this.preset, records: this.trace || []}; this.trace = null; return data; }
  }
  function mount() {
    const host = document.getElementById('cabinetSonification');
    if (!host) return;
    const status = document.getElementById('cabinetSoundStatus'), debug = document.getElementById('cabinetSoundDebug');
    const instrument = new Instrument({onChange: () => {
      status.textContent = instrument.reason + (instrument.trace ? ` · recording ${instrument.trace.length}/600 frames` : '');
      debug.textContent = JSON.stringify({mappingVersion: PRESET.version, readings: instrument.frame?.readings,
        bounds: PRESET.voices, targets: instrument.targets}, null, 2);
      host.querySelectorAll('[data-sound-solo]').forEach(b => b.setAttribute('aria-pressed', String(instrument.solo === b.dataset.soundSolo)));
    }});
    root.OrionCabinetSonification = instrument;
    document.getElementById('cabinetSoundListen').onclick = () => instrument.listen();
    document.getElementById('cabinetSoundStop').onclick = () => instrument.stop();
    document.getElementById('cabinetSoundRecord').onclick = () => instrument.startTrace();
    document.getElementById('cabinetSoundDownload').onclick = () => {
      const blob = new Blob([JSON.stringify(instrument.takeTrace(), null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob), a = document.createElement('a');
      a.href = url; a.download = 'cabinet-sonification-trace.json'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000); instrument.onChange();
    };
    host.querySelectorAll('[data-sound-mute]').forEach(input => input.onchange = () => instrument.mute(input.dataset.soundMute, input.checked));
    host.querySelectorAll('[data-sound-solo]').forEach(button => button.onclick = () => instrument.setSolo(button.dataset.soundSolo));
    document.addEventListener('visibilitychange', () => { if (document.hidden) instrument.stop('Page hidden'); });
    root.addEventListener('pagehide', () => instrument.stop('Page hidden'));
    // Ancestor modal can be hidden without calling cabinet.deactivate().
    const visible = () => !document.hidden && host.getClientRects().length > 0 && getComputedStyle(host).visibility !== 'hidden';
    const observer = new MutationObserver(() => { if ((instrument.armed || instrument.starting) && !visible()) instrument.stop('Cabinet hidden'); });
    for (let parent = host; parent; parent = parent.parentElement) observer.observe(parent, {attributes: true, attributeFilter: ['class', 'style', 'hidden']});
    instrument.onChange();
  }
  const api = {PRESET, HOLD_MS, frameFromPayload, mapFrame, Onset, Instrument};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document !== 'undefined') {
    root.OrionCabinetSound = api;
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, {once: true}); else mount();
  }
})(typeof window === 'undefined' ? globalThis : window);
