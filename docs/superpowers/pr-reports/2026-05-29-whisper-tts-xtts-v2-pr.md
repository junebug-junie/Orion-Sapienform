# PR: feat(whisper-tts) — XTTS-v2 voice-aware TTS

**Branch:** `feat/whisper-tts-xtts-v2`  
**Base:** `main`

## Summary

Upgrades `services/orion-whisper-tts` from single-speaker VITS to Coqui **XTTS-v2** as the default model, wiring `voice_id`, `language`, and `options` from the bus through `TTSEngine.synthesize_to_b64()` → `TTSOutput` with synthesis metadata on legacy and typed replies. Adds reference-voice support via `speaker_wav`, Docker voice-profile mount, smoke/download scripts, and README bus examples. Backend seam (`TTS_BACKEND`) is ready for future HTTP backends without touching the worker.

Also fixes Hub hold-to-talk **STT**: Web Audio PCM → 16 kHz WAV upload (replaces broken WebM path), silent-audio detection, typed `stt.transcribe.result` envelopes, and clearer WS errors for silent mic vs empty Whisper output.

**Telemetry:** XTTS-v2 weights (~1.8 GB) pre-downloaded to `/mnt/telemetry/models/coqui/tts/tts_models--multilingual--multi-dataset--xtts_v2`.

## Files changed

| File | Change |
|------|--------|
| `docs/superpowers/plans/2026-05-29-orion-whisper-tts-xtts-v2.md` | Implementation plan |
| `services/orion-whisper-tts/app/settings.py` | New TTS env fields; XTTS-v2 defaults |
| `services/orion-whisper-tts/app/tts.py` | `TTSOutput`, voice resolution, `CoquiBackend`, `TTSEngine`, torch.load shim |
| `services/orion-whisper-tts/app/tts_worker.py` | Voice-aware synthesis + metadata replies |
| `services/orion-whisper-tts/app/main.py` | Startup TTS config logging |
| `services/orion-whisper-tts/.env_example` | XTTS defaults, `COQUI_TOS_AGREED`, voice mount |
| `services/orion-whisper-tts/docker-compose.yml` | Env passthrough, voices volume, `COQUI_TOS_AGREED` |
| `services/orion-whisper-tts/Dockerfile` | Pin `transformers<4.46` after torch install |
| `services/orion-whisper-tts/requirements.txt` | `transformers>=4.33,<4.46` for XTTS + TTS 0.22 |
| `services/orion-whisper-tts/README.md` | Runbook, pre-download, smoke, bus examples |
| `services/orion-whisper-tts/scripts/smoke_xtts.py` | Container smoke synthesis |
| `services/orion-whisper-tts/scripts/download_xtts_model.py` | Pre-download model to telemetry cache |
| `services/orion-whisper-tts/tests/test_tts_voice_resolution.py` | Voice resolution + path containment |
| `services/orion-whisper-tts/tests/test_tts_worker_replies.py` | Typed + legacy reply metadata |
| `services/orion-whisper-tts/tests/test_tts_engine_settings.py` | Updated defaults / compose guards |
| `services/orion-whisper-tts/app/stt.py` | ffmpeg WebM/WAV, level measurement, Whisper short-utterance tuning |
| `services/orion-whisper-tts/app/stt_worker.py` | Typed STT result + `system.error` envelopes |
| `services/orion-whisper-tts/tests/test_stt_engine.py` | STT level + near-silent unit tests |
| `services/orion-hub/static/js/app.js` | Web Audio PCM capture → WAV for STT |
| `services/orion-hub/scripts/websocket_handler.py` | STT errors, `audio_bytes` logging |
| `services/orion-hub/scripts/bus_clients/tts_client.py` | `system.error` STT handling |

**Not changed:** `orion/schemas/tts.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`.

## Commands

### Rebuild and run (required after merge — image must include transformers pin)

```bash
cd services/orion-whisper-tts
cp .env_example .env
export PROJECT=orion
docker compose build whisper-tts
docker compose up -d whisper-tts
docker compose logs whisper-tts | tail -n 80
```

### Pre-download (already done on this host; idempotent)

```bash
COQUI_TOS_AGREED=1 docker compose run --rm whisper-tts python3 scripts/download_xtts_model.py
```

### Smoke

```bash
docker compose exec whisper-tts python3 scripts/smoke_xtts.py
```

### Unit tests

```bash
cd services/orion-whisper-tts && python3 -m pytest tests/ -v
```

## Test plan

- [x] `python3 -m compileall services/orion-whisper-tts/app orion/schemas`
- [x] XTTS-v2 weights on `/mnt/telemetry/models/coqui/tts` (~1.8 GB)
- [x] Model load verified in container (transformers pin + `weights_only=False` shim)
- [ ] `python3 -m pytest services/orion-whisper-tts/tests/ -v` in CI/venv
- [ ] `docker compose build` + service restart on GPU host
- [ ] Smoke WAV at `/tmp/orion_xtts_smoke.wav`
- [ ] Legacy + typed bus requests with metadata
- [ ] Hub hold-to-talk: console `[voice] peak amplitude` above ~0.003 when speaking; short utterance transcribes
- [ ] Deploy **hub + whisper-tts together** from this branch (hard refresh Hub)

## Known risks

- **Rebuild required** — existing images have `transformers` 5.x; Dockerfile now pins `<4.46`.
- **PyTorch 2.8** — Coqui checkpoints need `weights_only=False` (shim in `app/tts.py`).
- **GPU RAM** — XTTS-v2 needs a capable NVIDIA GPU.
- **Breaking default** — was `ljspeech/vits`; set `TTS_MODEL_NAME` explicitly to stay on VITS.
- **Voice required** — compose defaults `TTS_DEFAULT_SPEAKER=Ana Florence` for text-only callers.

## Diff summary

- Voice-aware bus path end-to-end with synthesis metadata.
- Path confinement for `speaker_wav` under `TTS_VOICE_PROFILE_DIR`.
- No new external services; `TTS_BACKEND=coqui` only.
