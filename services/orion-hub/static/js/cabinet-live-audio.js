// Live cabinet mic player for the chat card's "Cabinet audio" button.
//
// GET /api/cabinet/ambient/live streams raw S16_LE PCM tailed from the host
// ambient reader's slot ring (scripts/orion_ambient_audio_reader.py). Each
// network chunk becomes an AudioBuffer scheduled back-to-back on the shared
// AudioContext, through an AnalyserNode that app.js draws on the same
// waveform canvas as Orion's TTS voice.
(function (root) {
  'use strict';

  const LIVE_URL = '/api/cabinet/ambient/live';
  // Head start over currentTime for each (re)start of the schedule, so network
  // jitter between 0.5s chunks does not click into silence.
  const JITTER_SEC = 0.35;

  // Little-endian int16 bytes -> float32 [-1, 1), whole frames only. Returns
  // the unused partial-frame tail so the next network chunk can complete it.
  function pcm16ToFloat32(bytes, channels) {
    const frameBytes = 2 * (channels || 1);
    const usable = bytes.length - (bytes.length % frameBytes);
    const out = new Float32Array(usable / 2);
    const view = new DataView(bytes.buffer, bytes.byteOffset, usable);
    for (let i = 0; i < out.length; i += 1) out[i] = view.getInt16(i * 2, true) / 32768;
    return { samples: out, tail: bytes.subarray(usable) };
  }

  function concatBytes(a, b) {
    if (!a || !a.length) return b;
    const out = new Uint8Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  // opts.getContext(): the page's AudioContext
  // opts.onAnalyser(analyser | null): live analyser to draw, null when stopped
  // opts.onStatus(state, text): 'connecting' | 'live' | 'stopped' | 'error'
  // opts.fetchImpl: injectable fetch (tests)
  function createCabinetLiveAudio(opts) {
    const fetchImpl = opts.fetchImpl || ((url, init) => root.fetch(url, init));
    let run = null;

    function status(state, text) {
      if (opts.onStatus) opts.onStatus(state, text || '');
    }

    function teardown(current) {
      if (run !== current) return;
      run = null;
      try { current.gain.disconnect(); } catch (_) { /* already gone */ }
      try { current.analyser.disconnect(); } catch (_) { /* already gone */ }
      if (opts.onAnalyser) opts.onAnalyser(null);
    }

    function schedule(current, samples, sampleRate, channels) {
      const ctx = current.ctx;
      const frames = Math.floor(samples.length / channels);
      if (!frames) return;
      const buffer = ctx.createBuffer(channels, frames, sampleRate);
      for (let c = 0; c < channels; c += 1) {
        const data = buffer.getChannelData(c);
        for (let i = 0; i < frames; i += 1) data[i] = samples[i * channels + c];
      }
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(current.gain);
      // Fell behind (stall, background tab): restart the schedule with a fresh
      // head start instead of playing a backlog late.
      if (current.nextTime < ctx.currentTime) current.nextTime = ctx.currentTime + JITTER_SEC;
      src.start(current.nextTime);
      current.nextTime += buffer.duration;
    }

    async function start() {
      if (run) return;
      const ctx = opts.getContext();
      const current = {
        ctx,
        abort: new AbortController(),
        gain: ctx.createGain(),
        analyser: ctx.createAnalyser(),
        nextTime: 0,
      };
      run = current;
      current.gain.connect(current.analyser);
      current.analyser.connect(ctx.destination);
      status('connecting', 'connecting…');
      try {
        if (ctx.state === 'suspended') await ctx.resume();
        const resp = await fetchImpl(LIVE_URL, { signal: current.abort.signal, cache: 'no-store' });
        if (!resp.ok) {
          let detail = `HTTP ${resp.status}`;
          try { detail = (await resp.json()).detail || detail; } catch (_) { /* not json */ }
          throw new Error(detail);
        }
        const sampleRate = Number(resp.headers.get('X-Orion-Sample-Rate')) || 16000;
        const channels = Number(resp.headers.get('X-Orion-Channels')) || 1;
        if (opts.onAnalyser) opts.onAnalyser(current.analyser);
        status('live', 'live');
        const reader = resp.body.getReader();
        let carry = null;
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          if (run !== current) return;
          const { samples, tail } = pcm16ToFloat32(concatBytes(carry, value), channels);
          carry = tail.length ? tail.slice() : null;
          schedule(current, samples, sampleRate, channels);
        }
        if (run === current) {
          teardown(current);
          status('error', 'cabinet audio went quiet (reader stalled)');
        }
      } catch (err) {
        if (run !== current) return; // stop() aborted the fetch on purpose
        teardown(current);
        status('error', String((err && err.message) || err));
      }
    }

    function stop() {
      const current = run;
      if (!current) return;
      teardown(current);
      current.abort.abort();
      status('stopped', '');
    }

    return {
      start,
      stop,
      isActive: () => run !== null,
    };
  }

  const api = { LIVE_URL, JITTER_SEC, pcm16ToFloat32, createCabinetLiveAudio };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.OrionCabinetLiveAudio = api;
})(typeof window === 'undefined' ? globalThis : window);
