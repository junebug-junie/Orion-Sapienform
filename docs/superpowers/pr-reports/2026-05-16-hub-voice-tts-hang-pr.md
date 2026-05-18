# PR: Hub voice/TTS hang guards

**Branch:** `fix/hub-voice-tts-hang`  
**Worktree:** `.worktrees/hub-voice-tts-hang`

## Summary

Voice recordings could appear to hang because spoken WebSocket payloads omitted `disable_tts` (TTS always ran when the checkbox was off), STT/TTS RPC used the 400s general `TIMEOUT_SEC`, and Coqui TTS synthesis had no worker-level timeout. This patch aligns spoken sends with typed chat, adds layered timeouts and stage logs, surfaces non-fatal `tts_error` in the UI, and makes whisper-tts respect `TTS_USE_GPU` with a single default model (`vits`).

## Root cause

| Layer | Finding |
|-------|---------|
| **Frontend** | `submitExplicitChatText` sent `disable_tts`; `startRecording()` `audioPayload` did not. |
| **Hub WS** | `disable_tts` defaulted to `False`; TTS scheduled after `llm_response` via `run_tts_remote` with only error logging. |
| **Hub RPC** | `TTSClient` used `TIMEOUT_SEC=400` for STT and TTS. |
| **whisper-tts** | `run_in_executor` for STT/TTS with no `asyncio.wait_for`; `TTSEngine` hardcoded `gpu=True`; model default mismatch (`vits` in `tts.py` vs `tacotron2-DDC` in settings/compose). |

## Commits

| Commit | Description |
|--------|-------------|
| `79133657` | TTS toggle on audio, timeouts, stage logs, `tts_error` queue, whisper-tts GPU/model/timeouts, tests |
| `02570427` | Reset `state: idle` on STT/chat errors; `voice.chat.error` logging |

## Files changed

| File | Change |
|------|--------|
| `services/orion-hub/static/js/app.js` | `disable_tts` on audio payload; yellow `tts_error` warning in `onmessage` |
| `services/orion-hub/scripts/websocket_handler.py` | Stage logs; `asyncio.wait_for` STT/TTS; `run_tts_remote` → `tts_error`; idle state on errors |
| `services/orion-hub/scripts/bus_clients/tts_client.py` | `HUB_STT_TIMEOUT_SEC` / `HUB_TTS_TIMEOUT_SEC` for bus RPC |
| `services/orion-hub/app/settings.py` | New timeout settings (120s STT, 180s TTS) |
| `services/orion-hub/.env_example` | Document hub voice timeouts |
| `services/orion-hub/docker-compose.yml` | Pass `HUB_STT_TIMEOUT_SEC`, `HUB_TTS_TIMEOUT_SEC` |
| `services/orion-hub/tests/test_voice_tts_hang_guards.py` | `run_tts_remote` error/timeout + static JS/settings checks |
| `services/orion-whisper-tts/app/tts.py` | Use `settings.tts_model_name` / `settings.tts_use_gpu` |
| `services/orion-whisper-tts/app/settings.py` | Default `vits`; `WHISPER_TTS_*_TIMEOUT_SEC` |
| `services/orion-whisper-tts/app/stt_worker.py` | `asyncio.wait_for` on STT executor |
| `services/orion-whisper-tts/app/tts_worker.py` | `asyncio.wait_for` on TTS executor |
| `services/orion-whisper-tts/docker-compose.yml` | `vits` default; whisper timeout env |
| `services/orion-whisper-tts/.env_example` | Whisper worker timeouts |
| `services/orion-whisper-tts/tests/test_tts_engine_settings.py` | Settings defaults + wiring asserts |

**Operator note:** Copy new keys from `.env_example` into local (gitignored) `.env`:

- `services/orion-hub/.env` — `HUB_STT_TIMEOUT_SEC`, `HUB_TTS_TIMEOUT_SEC`
- `services/orion-whisper-tts/.env` — `WHISPER_TTS_STT_TIMEOUT_SEC`, `WHISPER_TTS_SYNTH_TIMEOUT_SEC`

## Timeout layering

| Stage | Hub (bus RPC) | whisper-tts (executor) |
|-------|---------------|-------------------------|
| STT | `HUB_STT_TIMEOUT_SEC` = 120 | `WHISPER_TTS_STT_TIMEOUT_SEC` = 90 |
| TTS | `HUB_TTS_TIMEOUT_SEC` = 180 | `WHISPER_TTS_SYNTH_TIMEOUT_SEC` = 120 |

Hub caps exceed worker caps so failures surface at the worker first when synthesis is slow, and the Hub still bounds end-to-end wait.

## Stage logs (grep-friendly)

- `voice.ws.audio_received`
- `voice.stt.start` / `voice.stt.done` / `voice.stt.error`
- `voice.chat.start` / `voice.chat.done` / `voice.chat.error`
- `voice.tts.start` / `voice.tts.done` / `voice.tts.error`

## Tests

```bash
cd .worktrees/hub-voice-tts-hang
PYTHONPATH=. /path/to/Orion-Sapienform/venv/bin/python -m pytest \
  services/orion-hub/tests/test_voice_tts_hang_guards.py \
  services/orion-whisper-tts/tests/test_tts_engine_settings.py -q --tb=short
```

**Result:** 7 passed (4 hub async/static, 3 whisper static/settings).

```bash
python -m compileall services/orion-hub/scripts/websocket_handler.py \
  services/orion-hub/scripts/bus_clients/tts_client.py \
  services/orion-hub/app/settings.py \
  services/orion-whisper-tts/app -q
```

**Result:** exit 0.

## Acceptance

| Criterion | Status |
|-----------|--------|
| Spoken input honors TTS checkbox | Met — `disable_tts` on audio WS payload + server gate |
| No STT/TTS call hangs indefinitely | Met — hub RPC + worker `wait_for` (was 400s shared timeout) |
| Transcript + text visible if TTS fails | Met — `transcript` + `llm_response` before async TTS; `tts_error` warning |
| Logs reveal last completed stage | Met — `voice.*` stage lines |

## Live verification

**UNVERIFIED** — no live Hub + whisper-tts stack exercise in this session (unit/static tests only).

### Suggested smoke (operator)

1. Hub with TTS checkbox **off** → record voice → confirm no `voice.tts.start` in hub logs and no audio playback.
2. TTS **on** → record → transcript + Orion text appear; audio plays if synth succeeds.
3. Force slow/failed TTS (e.g. stop whisper-tts mid-request) → yellow “TTS warning” in chat; text still visible; `state` returns idle.

## Remaining risks

- Default compose model switched to `vits`; deployments that relied on tacotron2 default should set `TTS_MODEL_NAME` explicitly.
- Executor threads are not cancelled after worker timeout; repeated timeouts may still load the default thread pool (acceptable for this patch).
- Full WebSocket voice E2E not covered by automated tests.

## Test plan (PR)

- [ ] Spoken send with TTS disabled → no TTS RPC / no playback
- [ ] Spoken send with TTS enabled → STT → chat → audio (happy path)
- [ ] TTS failure/timeout → transcript + `llm_response` + yellow warning; UI not stuck on “Processing…”
- [ ] Hub logs show last `voice.*` stage before failure
- [ ] After deploy, confirm `HUB_*` and `WHISPER_TTS_*` timeout env vars in running containers
