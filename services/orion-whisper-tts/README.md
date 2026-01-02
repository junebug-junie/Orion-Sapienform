# Orion Whisper TTS

The **Whisper TTS** service provides Text-to-Speech capabilities, synthesizing audio from text payloads on the bus.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:tts:intake` | `CHANNEL_TTS_INTAKE` | `tts.synthesize.request` | TTS requests. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| (Caller-defined) | (via `reply_to`) | `tts.synthesize.result` | Audio data (or URL). |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_TTS_INTAKE` | `orion:tts:intake` | Input channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-whisper-tts
```
