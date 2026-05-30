# PR: TTS pipeline instrumentation and browser playback fix

**Branch:** `fix/voice-input-audio-telemetry` (or dedicated TTS branch)  
**Base:** `main`

## Summary

End-to-end TTS visibility from Hub decision → bus RPC → whisper-tts worker → browser playback, plus `AudioContext.resume()` before decode so checked TTS actually plays.

## Files changed

| File | Change |
|------|--------|
| `services/orion-hub/scripts/websocket_handler.py` | `voice.tts.decision` log; `tts_debug` WS payload; richer `run_tts_remote` queue message |
| `services/orion-hub/scripts/bus_clients/tts_client.py` | Raise on `system.error` replies |
| `services/orion-hub/static/js/app.js` | `[tts]` console logs; queue objects; resume AudioContext before playback |
| `services/orion-whisper-tts/app/tts_worker.py` | Request/reply detail logging |
| `services/orion-whisper-tts/app/tts.py` | XTTS load/synthesis logging; actionable failures |
| `services/orion-hub/tests/test_voice_tts_hang_guards.py` | Speaking payload + JS harness asserts |
| `services/orion-hub/tests/test_tts_client_errors.py` | `system.error` handling test |

## Manual verification

```bash
docker compose logs -f orion-hub orion-whisper-tts | grep -E 'voice\.tts|TTS|tts|synthesize|system.error|audio_response'
```

### Expected paths

| Observation | Fix layer |
|-------------|-----------|
| `will_tts=false` in `voice.tts.decision` | Hub decision (`workflow_metadata_only`, missing `tts_client`, empty text) |
| `will_tts=true`, `voice.tts.start`, no worker `Processing TTS` | Bus channel / `TTS_REQUEST_CHANNEL` vs `channel_tts_intake` |
| Worker `FAILED to synthesize` | XTTS speaker / `TTS_DEFAULT_SPEAKER` / `speaker_wav` path |
| `voice.tts.done audio_b64_len>0`, no browser sound | Hard-refresh Hub; check `[tts] playback started` and `AudioContext` state |

### Browser console

After response with TTS enabled:

```
[tts] audio_response received { audio_b64_len: …, tts_meta: … }
[tts] playback decode start { beforeState: "suspended", afterState: "running", … }
[tts] decoded { duration: …, sampleRate: … }
[tts] playback started
[tts] playback ended
```

If Hub skips TTS but checkbox is on:

```
[tts] debug { stage: "hub_decision", will_tts: false, … }
```

## Test plan

- [x] `pytest services/orion-hub/tests/test_voice_tts_hang_guards.py services/orion-hub/tests/test_tts_client_errors.py -v`
- [ ] Hub chat with TTS checked → `voice.tts.decision will_tts=true`
- [ ] Worker → `Sent TTS reply audio_b64_len=N` with N>0
- [ ] Browser → `[tts] playback started` and audible output
