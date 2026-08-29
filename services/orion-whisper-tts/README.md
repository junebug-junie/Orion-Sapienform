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
| `TTS_DEFAULT_SPEAKER` | `Ana Florence` | Built-in XTTS speaker name -- fallback only while `TTS_DEFAULT_SPEAKER_WAV` is set (see below). |
| `TTS_DEFAULT_SPEAKER_WAV` | `/models/voices/orion_reference.wav` | Reference `.wav` under `TTS_VOICE_PROFILE_DIR` -- takes precedence over `TTS_DEFAULT_SPEAKER` whenever set. Live default since 2026-08-27 (cloned voice); the `.wav` itself is a host asset, not checked into git. |
| `TTS_SPLIT_SENTENCES` | `true` | XTTS sentence splitting. |
| `TTS_VOICE_PROFILE_DIR` | `/models/voices` | Voice profile mount inside container. |
| `CHANNEL_TTS_INTAKE` | `orion:tts:intake` | Input channel. |
| `CUDA_WATCHDOG_ENABLED` | `true` | Self-restart on detected CUDA staleness. See "GPU liveness" below. |
| `CUDA_WATCHDOG_POLL_SEC` | `30` | Watchdog check cadence. |
| `CUDA_WATCHDOG_FAILURE_THRESHOLD` | `2` | Consecutive failed checks required before restarting. |

## GPU liveness

Real incident, 2026-08-26: this container's Coqui TTS backend crashed on
its first real request with `CUDA is not available on this machine`, while
STT (same container, same GPU) kept working. Root cause: a
docker+nvidia-container-toolkit staleness quirk (`nvidia-smi` inside the
container returned `Failed to initialize NVML: Unknown Error`), most likely
triggered by something at the host level (another GPU container being
rebuilt/restarted, a driver/persistenced reload) -- not anything this
service's own code does. STT survived it because `openai-whisper` falls
back to CPU (`stt.py`: `"cuda" if torch.cuda.is_available() else "cpu"`)
and its already-running model had likely established a CUDA context before
the staleness set in; Coqui's `TTS` library hard-asserts CUDA with no such
fallback. A plain container restart fixed it immediately.

One mechanism enforces this, at both moments a single check cannot cover
alone -- unified deliberately, after a review (2026-08-26) caught that two
separate hard-fail paths regressed the exact resilience this closes:

- **`_require_cuda_or_die()`** (`app/main.py`), called once at startup when
  `TTS_USE_GPU=true`. Existed as dead code for an unknown period -- defined,
  documented with a comment saying to call it during startup, never actually
  called -- until this patch wired it in. **Advisory only**: logs
  `CRITICAL` and lets `startup()` continue -- it does NOT raise. An earlier
  draft of this patch let it raise, which crashed the whole process (bus,
  `listener_task`, `stt_task`, all of it) before any of them started --
  taking STT down too, on a component that does not need CUDA at all and is
  exactly what survived the real incident. Boot-broken must degrade the
  same way mid-uptime-broken does, not worse.
- **The CUDA watchdog** (`app/cuda_watchdog.py`), a background task started
  alongside the existing heartbeat loop -- the single real enforcement path
  for BOTH "broken at boot" and "broken mid-uptime". Polls
  `torch.cuda.is_available()` (off the event loop, under a timeout -- a
  genuine NVML wedge can hang rather than fast-return, and an in-line
  synchronous call would freeze `heartbeat_loop`/`listener_task`/`stt_task`
  right along with itself) every `CUDA_WATCHDOG_POLL_SEC`. On
  `CUDA_WATCHDOG_FAILURE_THRESHOLD` consecutive failures (debounced against
  a single transient hiccup; a hang counts as a failure, a raised exception
  from the check itself does not), it sends itself `SIGTERM` -- not
  `os._exit()`, so `shutdown()` still runs cleanly first -- and
  `restart: unless-stopped` brings the container back with a fresh device
  mapping. A GPU broken from the very first boot simply fails its first
  check almost immediately and restarts on the normal threshold.

Neither closes the underlying host/driver quirk; that is outside this
service's code entirely. What they do is turn a silent, multi-hour "TTS is
just broken and nobody noticed" outage into a self-healing, few-second one.
Both are gated on `TTS_USE_GPU` -- a deliberate CPU-mode deployment is not
forced to have a GPU by either.

`GET /health` reports `cuda_available` (a fresh, direct check, `null` in
CPU mode) and `cuda_watchdog_enabled` -- an operator no longer has to wait
for a log line to see CUDA state mid-outage.

**Known, deliberately deferred risk**: the watchdog has no cross-restart
rate limit. Each restart is a brand-new process with fresh in-memory state,
so a genuinely PERMANENT GPU failure (not the transient staleness this was
built for) would restart the container roughly every
`poll_sec * failure_threshold` seconds indefinitely, with no backoff.
`services/orion-mesh-guardian` already has exactly this kind of
`cooldown_sec`/`max_attempts_per_hour` remediation state machine
(`config/mesh_remediation_roster.yaml`) -- whisper-tts is not registered
there. Not fixed here: mesh-guardian is an external, cross-restart-persistent
supervisor and doing this properly means registering with it (or exposing
CUDA state through a probe it can reach), not reinventing a second,
in-process, un-persisted rate limiter. Left as a named follow-up rather
than silently uncovered.

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

### Pre-download XTTS-v2 to the telemetry cache

Coqui stores weights under `~/.local/share/tts` in the container, which maps to `/mnt/telemetry/models/coqui/tts` on the host. Run once after build (before first bus request):

```bash
mkdir -p /mnt/telemetry/models/coqui/tts
export PROJECT=orion
COQUI_TOS_AGREED=1 docker compose run --rm whisper-tts python3 scripts/download_xtts_model.py
```

One-liner (same thing):

```bash
mkdir -p /mnt/telemetry/models/coqui/tts && cd services/orion-whisper-tts && PROJECT=orion COQUI_TOS_AGREED=1 docker compose run --rm whisper-tts python3 scripts/download_xtts_model.py
```

On the host without Docker (writes directly to telemetry):

```bash
mkdir -p /mnt/telemetry/models/coqui/tts
TTS_HOME=/mnt/telemetry/models/coqui/tts TTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2 python3 scripts/download_xtts_model.py
```

After download, expect a folder like `tts_models--multilingual--multi-dataset--xtts_v2` under that cache path.

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

## STT silence gate

`STT_NEAR_SILENT_PEAK_INT16` (default `50`) controls when incoming audio is treated as
near-silent and Whisper is skipped. Browser capture telemetry is sent separately via Hub
(`client_audio_meta` on the WebSocket audio payload).

## Local unit tests

```bash
cd services/orion-whisper-tts
python -m pytest tests/ -v
python -m compileall app
```

### Voice ingress manual check (Hub + browser)

1. Record from Hub mic button; browser console should log `[voice] chunk_count=`, `peak=`, `rms=`, and `sent audio payload`.
2. Hub: `docker compose logs -f orion-hub | grep -E 'voice\.ws\.audio_received|voice\.stt'`
3. STT: `docker compose logs -f orion-whisper-tts | grep -E '\[STT\]|Sent STT result'`

If the browser warns on low peak but Hub never logs `voice.ws.audio_received`, the client still blocked send. If Hub receives audio but STT `peak` is low, tune `STT_NEAR_SILENT_PEAK_INT16` or check mic gain.
