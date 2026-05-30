# PR: Voice input audio telemetry and silence-gate reliability

**Branch:** `fix/voice-input-audio-telemetry`  
**Base:** `main`  
**Repo:** junebug-junie/Orion-Sapienform

## Summary

Fixes false “no microphone signal” failures by stopping the browser from blocking low-gain PCM, lowering and configuring the STT int16 silence gate (default 50, was 200), canonicalizing all ingress WAV through ffmpeg, and threading `client_audio_meta` / `audio_debug` through Hub → STT for operator-visible peak, duration, and chunk diagnostics.

## Problem

- Hub `app.js` threw on `peak < 0.003` before sending audio to the WebSocket.
- STT hard-rejected `peak < 200` before Whisper.
- Errors blamed browser permissions without server/client peak context.

## Files changed

| File | Change |
|------|--------|
| `services/orion-hub/static/js/app.js` | Warn-but-send; `client_audio_meta`; capture `sampleRate`; console telemetry |
| `services/orion-hub/scripts/voice_stt_errors.py` | Error classification + meta sanitization (new) |
| `services/orion-hub/scripts/websocket_handler.py` | Log/forward client meta; `audio_debug` on STT errors |
| `services/orion-hub/README.md` | Voice debugging section |
| `services/orion-hub/tests/test_voice_stt_errors.py` | Error message + sanitize tests (new) |
| `services/orion-hub/tests/test_voice_tts_hang_guards.py` | Assert `audio_debug` / `client_audio_meta` in JS |
| `services/orion-whisper-tts/app/stt.py` | Configurable threshold; ffmpeg canonical s16 16kHz mono; rich metadata |
| `services/orion-whisper-tts/app/stt_worker.py` | Log/merge `client_audio_meta` from STT options |
| `services/orion-whisper-tts/app/settings.py` | `stt_near_silent_peak_int16` |
| `services/orion-whisper-tts/.env_example` | `STT_NEAR_SILENT_PEAK_INT16=50` |
| `services/orion-whisper-tts/.env` | Same (local sync, gitignored) |
| `services/orion-whisper-tts/docker-compose.yml` | Env passthrough |
| `services/orion-whisper-tts/README.md` | STT gate + manual verification |
| `services/orion-whisper-tts/tests/test_stt_engine.py` | Low-amplitude, env threshold, ffmpeg integration |

## Configuration

Add to `services/orion-whisper-tts/.env` (and restart container):

```bash
STT_NEAR_SILENT_PEAK_INT16=50
```

## Manual verification

### Browser console (after mic release)

```
[voice] chunk_count=… source_sample_rate=…
[voice] peak=… rms=… client_gate=…
[voice] sent audio payload bytes=…
```

On empty transcript:

```
[voice] audio_debug { client: …, stt: … }
```

### Hub logs

```bash
docker compose logs -f orion-hub | grep -E 'voice\.ws\.audio_received|voice\.stt'
```

Expect `client_meta=…` when the browser sends telemetry.

### STT logs

```bash
docker compose logs -f orion-whisper-tts | grep -E '\[STT\]|Sent STT result|client_audio_meta'
```

### Debugging split

| Symptom | Layer |
|---------|--------|
| Low peak in browser, no `voice.ws.audio_received` | Frontend still blocked (should not happen after this PR) |
| Hub receives audio, STT `peak` low / `silence_gate=rejected` | Tune `STT_NEAR_SILENT_PEAK_INT16` or mic gain |
| STT peak healthy, empty transcript | Whisper / no-speech threshold |

## Test plan

- [x] `pytest services/orion-whisper-tts/tests/test_stt_engine.py services/orion-whisper-tts/tests/test_tts_engine_settings.py -v`
- [x] `pytest services/orion-hub/tests/test_voice_stt_errors.py -v`
- [x] `python3 -m compileall services/orion-hub/scripts/websocket_handler.py services/orion-whisper-tts/app`
- [ ] Hold mic button in Hub UI — status shows peak/duration; low gain sends anyway
- [ ] Confirm `audio_debug` in WS error payload when STT returns empty
- [ ] Restart `orion-whisper-tts` after `.env` change

## Acceptance criteria

1. Low-gain real audio is sent to Hub (not blocked by browser peak gate).
2. UI status shows peak/duration/chunk debug.
3. Hub error payloads include `audio_debug`.
4. `STT_NEAR_SILENT_PEAK_INT16` configurable; default 50.
5. STT metadata includes `silence_gate`, `peak_threshold`, input/wav bytes.
6. Text chat and TTS unchanged.
7. New/updated unit tests pass.
