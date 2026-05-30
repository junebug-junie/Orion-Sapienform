# PR: feat(whisper-tts) — XTTS-v2 voice-aware TTS

**Branch:** `feat/whisper-tts-xtts-v2`  
**Base:** `main`

## Summary

Upgrades `services/orion-whisper-tts` from single-speaker VITS to Coqui **XTTS-v2** as the default model, wiring `voice_id`, `language`, and `options` from the bus through `TTSEngine.synthesize_to_b64()` → `TTSOutput` with synthesis metadata on legacy and typed replies. Adds reference-voice support via `speaker_wav`, Docker voice-profile mount, smoke script, and README bus examples. Backend seam (`TTS_BACKEND`) is ready for future HTTP backends without touching the worker.

## Files changed

| File | Change |
|------|--------|
| `docs/superpowers/plans/2026-05-29-orion-whisper-tts-xtts-v2.md` | Implementation plan |
| `services/orion-whisper-tts/app/settings.py` | New TTS env fields; XTTS-v2 defaults |
| `services/orion-whisper-tts/app/tts.py` | `TTSOutput`, `resolve_synthesis_plan`, `CoquiBackend`, `TTSEngine` |
| `services/orion-whisper-tts/app/tts_worker.py` | Voice-aware synthesis + metadata replies |
| `services/orion-whisper-tts/.env_example` | XTTS defaults, speaker modes, voice mount host dir |
| `services/orion-whisper-tts/docker-compose.yml` | Env passthrough + `/models/voices` volume |
| `services/orion-whisper-tts/README.md` | Runbook, smoke, bus payload examples |
| `services/orion-whisper-tts/scripts/smoke_xtts.py` | Container smoke synthesis |
| `services/orion-whisper-tts/tests/test_tts_voice_resolution.py` | Voice resolution + path containment |
| `services/orion-whisper-tts/tests/test_tts_worker_replies.py` | Typed reply metadata |
| `services/orion-whisper-tts/tests/test_tts_engine_settings.py` | Updated defaults / compose guards |

**Not changed:** `orion/schemas/tts.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml` (already correct).

## Commands

### Rebuild and run

```bash
cd services/orion-whisper-tts
cp .env_example .env   # set ORION_BUS_URL, GPU as needed
export PROJECT=orion
docker compose build whisper-tts
docker compose up -d whisper-tts
docker compose logs whisper-tts | tail -n 80
```

### Unit tests (host with deps)

```bash
cd services/orion-whisper-tts
python3 -m pytest tests/ -v
python3 -m compileall app
```

### Smoke (inside container; requires GPU + model cache)

```bash
docker compose exec whisper-tts python3 scripts/smoke_xtts.py
docker compose exec whisper-tts ls -la /tmp/orion_xtts_smoke.wav
```

### Bus legacy test (redis-cli)

```bash
TRACE=$(uuidgen)
REPLY="orion:tts:result:manual-${TRACE}"
redis-cli -u "$ORION_BUS_URL" SUBSCRIBE "$REPLY" &
sleep 1
redis-cli -u "$ORION_BUS_URL" PUBLISH orion:tts:intake \
  '{"kind":"legacy.message","payload":{"text":"Hello legacy","response_channel":"'"$REPLY"'","trace_id":"'"$TRACE"'"}}'
```

### Typed bus test (Hub)

Use `TTSClient.speak(TTSRequestPayload(...))` from `services/orion-hub/scripts/bus_clients/tts_client.py` with full payload (see README example with `options.speaker_wav`).

## Test plan

- [x] `python3 -m compileall services/orion-whisper-tts/app orion/schemas`
- [ ] `python3 -m pytest services/orion-whisper-tts/tests/ -v` (CI / venv with pydantic + pytest)
- [ ] Container build on GPU host
- [ ] Startup logs: backend, model, gpu
- [ ] Smoke WAV at `/tmp/orion_xtts_smoke.wav`
- [ ] Legacy reply: `audio_b64`, `mime_type`, `metadata`
- [ ] Typed reply: `metadata.speaker_wav_used` when wav provided

## Known risks

- **First XTTS load** is slow and downloads large weights into the Coqui cache volume.
- **GPU RAM** — XTTS-v2 needs a capable NVIDIA GPU; startup fails loudly if CUDA/torch mismatch.
- **Voice required** — XTTS needs `TTS_DEFAULT_SPEAKER`, `TTS_DEFAULT_SPEAKER_WAV`, or per-request voice fields; compose defaults to `Ana Florence`.
- **Breaking default** — Deployments pinned to `ljspeech/vits` must set `TTS_MODEL_NAME` explicitly.
- **speaker_wav paths** must live under `TTS_VOICE_PROFILE_DIR` (path traversal blocked).

## Diff summary

- ~400 lines production code + tests; plan doc included in branch for agentic execution trace.
- Legacy replies gain optional `metadata`; typed replies populate `duration_sec` and `metadata`.
- No new external services; `TTS_BACKEND=coqui` only.
