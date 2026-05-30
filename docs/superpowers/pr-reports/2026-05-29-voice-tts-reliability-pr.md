# PR: Hub voice ingress + STT silence gate + TTS playback reliability

**Branch:** `fix/voice-input-audio-telemetry`  
**Base:** `main`  
**Repo:** [junebug-junie/Orion-Sapienform](https://github.com/junebug-junie/Orion-Sapienform)

## Summary

This PR fixes the full spoken-input and spoken-output path in Orion Hub:

1. **Voice capture** — stop falsely rejecting low-gain mic audio; fix Chromium `ScriptProcessor` silence bug; thread `client_audio_meta` through to STT.
2. **STT ingress** — lower/configurable int16 silence gate (default 50); always canonicalize WAV via ffmpeg; rich `silence_gate` metadata and actionable Hub errors (`audio_debug`).
3. **TTS playback** — instrument Hub → bus → whisper-tts → browser; resume `AudioContext` before decode; handle `system.error` from TTS worker.
4. **UI** — prevent duplicate Orion chat bubbles when TTS audio arrives (TTS message must not use `text` as a second `llm_response`).

**Scope:** Mic capture, STT ingress, TTS egress, and Hub WebSocket UX only. No Whisper model swap, no TTS engine swap, no bus channel redesign.

## Commits (8)

| SHA | Message |
|-----|---------|
| `ec13949d` | fix(hub): send low-gain mic audio with client telemetry |
| `5dfe6010` | fix(whisper-tts): lower STT silence gate and canonicalize WAV ingress |
| `b50ca34a` | fix(hub): voice STT errors with audio_debug and client meta logging |
| `5c0c8a02` | fix: address voice PR review — errors helper, STT worker meta |
| `e60e3441` | docs: add voice input audio telemetry PR report |
| `0bd1d6d4` | fix(hub): ScriptProcessor mic capture was recording silence |
| `8c033400` | fix(hub): TTS pipeline instrumentation and browser playback |
| `be556564` | fix(hub): stop TTS audio_response from duplicating chat bubble |

## Problems solved

| Symptom | Root cause | Fix |
|---------|------------|-----|
| "No microphone signal — check permissions" | Client threw on `peak < 0.003`; STT rejected `peak < 200` | Warn-but-send; threshold 50; ffmpeg canonicalization |
| STT `peak=0` with real mic | ScriptProcessor never copied input→output; echo cancellation | `output.set(input)`; disable echoCancellation; block send if `source_peak=0` |
| TTS checked, no sound | `AudioContext` suspended after mic capture | `audioContext.resume()` before `decodeAudioData` |
| Duplicate Orion replies | TTS WS message had `text` → UI treated as second `llm_response` | `tts_source_text` + skip append for `audio_response`-only payloads |
| Opaque failures | Thin logging | `voice.tts.decision`, `[voice]`/`[tts]` console, `audio_debug`, worker/engine logs |

## Files changed (20 files, +1013 / −115)

### Orion Hub

| File | Change |
|------|--------|
| `services/orion-hub/static/js/app.js` | Voice: warn-but-send, `client_audio_meta`, capture `sampleRate`, ScriptProcessor fix, block silent send. TTS: `[tts]` logs, `AudioContext.resume()`, queue objects, no duplicate bubble on `audio_response`. |
| `services/orion-hub/scripts/websocket_handler.py` | STT: log/forward `client_audio_meta`, `audio_debug` on errors/timeouts. TTS: `voice.tts.decision`, `tts_debug`, richer `run_tts_remote` payload (`tts_source_text`, `tts_meta`, `state: speaking`). |
| `services/orion-hub/scripts/voice_stt_errors.py` | **New** — STT empty-transcript message classification; sanitize `client_audio_meta`. |
| `services/orion-hub/scripts/bus_clients/tts_client.py` | Raise `ValueError` on bus `system.error` (parity with STT). |
| `services/orion-hub/README.md` | Voice debugging section. |
| `services/orion-hub/tests/test_voice_stt_errors.py` | **New** — error classification tests. |
| `services/orion-hub/tests/test_tts_client_errors.py` | **New** — TTS `system.error` test. |
| `services/orion-hub/tests/test_voice_tts_hang_guards.py` | TTS speaking payload, JS harness asserts, duplicate-guard assert. |

### Orion Whisper-TTS

| File | Change |
|------|--------|
| `services/orion-whisper-tts/app/stt.py` | `STT_NEAR_SILENT_PEAK_INT16` via env/settings (default 50); always ffmpeg → 16 kHz mono s16; metadata (`silence_gate`, `input_peak`, …); client override when browser reports signal. |
| `services/orion-whisper-tts/app/stt_worker.py` | Log/merge `client_audio_meta` from STT options. |
| `services/orion-whisper-tts/app/tts.py` | XTTS load/synthesis logging; actionable synthesis failures. |
| `services/orion-whisper-tts/app/tts_worker.py` | Detailed request/reply logging. |
| `services/orion-whisper-tts/app/settings.py` | `stt_near_silent_peak_int16`. |
| `services/orion-whisper-tts/.env_example` | `STT_NEAR_SILENT_PEAK_INT16=50`. |
| `services/orion-whisper-tts/docker-compose.yml` | Env passthrough. |
| `services/orion-whisper-tts/README.md` | STT gate + manual verification. |
| `services/orion-whisper-tts/tests/test_stt_engine.py` | Low-amplitude, env threshold, ffmpeg, client override tests. |
| `services/orion-whisper-tts/tests/test_tts_engine_settings.py` | Settings/compose assertions. |

**Local only (gitignored):** `services/orion-whisper-tts/.env` — sync `STT_NEAR_SILENT_PEAK_INT16=50`.

## Configuration

```bash
# services/orion-whisper-tts/.env
STT_NEAR_SILENT_PEAK_INT16=50
```

Restart whisper-tts after change. Client peak warn threshold is in `app.js` (`VOICE_CLIENT_PEAK_MIN = 0.00025`, warn-only).

## Architecture (unchanged channels)

```
Browser PCM → WS audio + client_audio_meta
    → Hub STT RPC (orion:stt:intake) → whisper STT worker
    → transcript → Cortex chat
    → Hub voice.tts.decision → TTS RPC (orion:tts:intake) → XTTS worker
    → WS audio_response → browser decode/play
```

## Manual verification

### 1. Hard refresh Hub

Ctrl+Shift+R so `app.js` updates load (critical after each frontend fix).

### 2. Voice capture (browser console)

After mic release:

```
[voice] chunk_count=… source_sample_rate=…
[voice] peak=… rms=… client_gate=…
[voice] sent audio payload bytes=…
```

On STT failure:

```
[voice] audio_debug { client: …, stt: … }
```

### 3. Hub logs (voice + TTS)

```bash
docker compose logs -f orion-hub orion-whisper-tts | grep -E 'voice\.(ws|stt|tts)|\[STT\]|\[TTS\]|Processing TTS|Sent TTS|system\.error|audio_response'
```

Expected happy path:

- `voice.ws.audio_received` with `client_meta=…`
- `voice.stt.done` with `transcript_len > 0`
- `voice.tts.decision … will_tts=true disable_tts=false`
- `voice.tts.start` → `voice.tts.done audio_b64_len=…` (large)
- Worker: `Processing TTS request` → `Sent TTS reply audio_b64_len=…`

### 4. TTS playback (browser console)

```
[tts] audio_response received { audio_b64_len: …, tts_meta: … }
[tts] playback decode start { beforeState: "suspended", afterState: "running", … }
[tts] decoded { duration: …, sampleRate: … }
[tts] playback started
[tts] playback ended
```

If TTS skipped while checkbox on:

```
[tts] debug { stage: "hub_decision", will_tts: false, … }
```

### 5. Debugging split

| Observation | Layer |
|-------------|--------|
| Low peak in browser, no `voice.ws.audio_received` | Frontend blocked (should not happen) |
| Hub receives audio, `silence_gate=rejected` | Tune `STT_NEAR_SILENT_PEAK_INT16` / mic gain |
| STT peak OK, empty transcript | Whisper / no-speech |
| `will_tts=false` + `[tts] debug` | Hub gate (`workflow_metadata_only`, no `tts_client`) |
| `will_tts=true`, no worker `Processing TTS` | Bus / channel mismatch |
| Worker `FAILED to synthesize` | XTTS speaker / `TTS_DEFAULT_SPEAKER` / `speaker_wav` |
| `voice.tts.done` with audio, no `[tts] playback started` | Browser cache / playback |
| Two identical Orion bubbles | Fixed — refresh; TTS must not re-append via `text` |

## Test plan

### Automated (run in repo venv)

```bash
cd /mnt/scripts/Orion-Sapienform
.venv/bin/pytest \
  services/orion-whisper-tts/tests/test_stt_engine.py \
  services/orion-whisper-tts/tests/test_tts_engine_settings.py \
  services/orion-hub/tests/test_voice_stt_errors.py \
  services/orion-hub/tests/test_voice_tts_hang_guards.py \
  services/orion-hub/tests/test_tts_client_errors.py \
  -v

python3 -m compileall \
  services/orion-hub/scripts \
  services/orion-whisper-tts/app \
  orion/schemas
```

### Manual (operator)

- [ ] Record quiet speech — sends anyway; status shows peak/duration
- [ ] Record normal speech — transcript appears; no permission blame on low gain
- [ ] Chat with TTS checked — one Orion bubble only; audible TTS
- [ ] Empty/near-silent recording — `audio_debug` in WS error, not generic permission text
- [ ] `docker compose restart whisper-tts` after `.env` sync

## Acceptance criteria

- [x] Low-gain real audio reaches Hub/STT (not blocked by browser peak gate)
- [x] UI shows useful peak/duration/chunk debug; `client_audio_meta` on WS audio payload
- [x] Hub STT errors include `audio_debug` with client + server metadata
- [x] `STT_NEAR_SILENT_PEAK_INT16` configurable; default 50
- [x] STT metadata includes `silence_gate`, `peak_threshold`, input/wav bytes
- [x] TTS: `voice.tts.decision` logging; `system.error` surfaced in Hub client
- [x] TTS: browser resumes AudioContext and logs playback lifecycle
- [x] No duplicate Orion message on TTS audio delivery
- [x] Text-only chat path unchanged aside from TTS instrumentation
- [x] Unit tests pass

## Out of scope

- Whisper model change
- TTS backend swap (Coqui/XTTS v2 remains)
- MediaRecorder migration
- Bus channel / schema registry changes (uses existing `STTRequestPayload.options` and `TTSResultPayload`)
