# Orion Whisper TTS

The **Whisper TTS** service provides Text-to-Speech (Coqui XTTS-v2 by default) and Speech-to-Text (Whisper) on the Orion bus.

## Contracts

### Consumed Channels

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:tts:intake` | `CHANNEL_TTS_INTAKE` | `tts.synthesize.request` | TTS requests. |
| `orion:stt:intake` | (see STT settings) | `stt.transcribe.request` | STT requests. |

### Published Channels

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| (Caller-defined) | (via `reply_to` / `response_channel`) | `tts.synthesize.result` | Audio + metadata. |

### Environment Variables

Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TTS_BACKEND` | `coqui` | Backend selector (`coqui` today). |
| `TTS_MODEL_NAME` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui model id. |
| `TTS_USE_GPU` | `true` | Pass GPU flag to Coqui. |
| `TTS_DEFAULT_LANGUAGE` | `en` | Default language code. |
| `TTS_DEFAULT_SPEAKER` | (empty) | Built-in XTTS speaker name. |
| `TTS_DEFAULT_SPEAKER_WAV` | (empty) | Reference `.wav` under `TTS_VOICE_PROFILE_DIR`. |
| `TTS_SPLIT_SENTENCES` | `true` | XTTS sentence splitting. |
| `TTS_VOICE_PROFILE_DIR` | `/models/voices` | Voice profile mount inside container. |
| `CHANNEL_TTS_INTAKE` | `orion:tts:intake` | Input channel. |

## Running

### Docker Compose (GPU host)

```bash
cd services/orion-whisper-tts
cp .env_example .env   # edit ORION_BUS_URL, speaker mode
docker compose build whisper-tts
docker compose up -d whisper-tts
docker compose logs -f whisper-tts
```

On startup you should see **TTS configured** logs (`backend`, `model`, `gpu`, defaults). Coqui model load logs appear on the first synthesis request (`[TTS] Loading coqui model=...`).

Mount reference voices on the host at `TTS_VOICE_PROFILE_HOST_DIR` (default `/mnt/telemetry/models/coqui/voices`) → `/models/voices`.

### List Coqui XTTS speakers (inside container)

```bash
docker compose exec whisper-tts python3 -c "
from TTS.api import TTS
t = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=True)
print(getattr(t, 'speakers', None) or 'no speakers attr')
"
```

### Smoke test (inside container)

Set either `TTS_DEFAULT_SPEAKER=Ana Florence` or place `orion_reference.wav` under `/models/voices` and set `TTS_DEFAULT_SPEAKER_WAV=/models/voices/orion_reference.wav`.

```bash
docker compose exec whisper-tts python3 scripts/smoke_xtts.py
docker compose exec whisper-tts ls -la /tmp/orion_xtts_smoke.wav
```

## Bus payload examples

### 1. Simple text-only (legacy)

```json
{
  "text": "Hello Juniper.",
  "response_channel": "orion:tts:result:my-trace-1",
  "trace_id": "my-trace-1"
}
```

Wrap as `legacy.message` on `orion:tts:intake` if not using typed envelopes.

### 2. Typed request with `voice_id` (built-in speaker)

```json
{
  "text": "Hello Juniper. This is Orion with the upgraded voice.",
  "voice_id": "Ana Florence",
  "language": "en"
}
```

Kind: `tts.synthesize.request`, `reply_to`: `orion:tts:result:<uuid>`.

### 3. Reference voice via `options.speaker_wav`

```json
{
  "text": "Hello Juniper. This is Orion with the upgraded voice.",
  "voice_id": "orion_reference.wav",
  "language": "en",
  "options": {
    "speaker_wav": "/models/voices/orion_reference.wav",
    "split_sentences": true
  }
}
```

### 4. Language only

```json
{
  "text": "Bonjour.",
  "language": "fr",
  "voice_id": "Ana Florence"
}
```

## Replies

**Legacy:** `{ "trace_id", "audio_b64", "mime_type", "metadata" }`

**Typed:** `tts.synthesize.result` with `TTSResultPayload` (`audio_b64`, `content_type`, `duration_sec`, `metadata`).

Metadata includes `backend`, `model_name`, `language`, `voice_id`, `speaker`, `speaker_wav_used`, `split_sentences`, `synthesis_ms`.

## Local unit tests

```bash
cd services/orion-whisper-tts
python -m pytest tests/ -v
python -m compileall app
```
