# Orion Whisper/TTS Service

This service provides Text-to-Speech (TTS) capabilities using the Titanium contract stack.

## Architecture

*   **Bus:** Uses `orion.core.bus.async_service.OrionBusAsync`.
*   **Protocol:** Redis Pub/Sub with Typed Envelopes (`orion.schemas`).
*   **Engine:** Coqui TTS (synchronous, runs in thread pool).

## Interfaces

### 1. Typed Envelope (Preferred)

*   **Request Channel:** `orion:tts:intake` (configurable via `CHANNEL_TTS_INTAKE`)
*   **Kind:** `tts.synthesize.request`
*   **Payload:** `TTSRequestPayload` (see `orion/schemas/tts.py`)
    *   `text` (string, required)
    *   `voice_id` (string, optional)
    *   `language` (string, optional)
*   **Response:**
    *   **Kind:** `tts.synthesize.result`
    *   **Payload:** `TTSResultPayload`
        *   `audio_b64` (base64 encoded wav)
        *   `content_type` (e.g. "audio/wav")

### 2. Legacy/Generic

*   **Request Channel:** `orion:tts:intake`
*   **Payload:**
    ```json
    {
      "text": "Hello world",
      "response_channel": "orion:reply:..."
    }
    ```
*   **Response:**
    *   Typed Envelope with `tts.synthesize.result` sent to `response_channel`.

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ORION_BUS_URL` | Redis URL | `redis://localhost:6379/0` |
| `CHANNEL_TTS_INTAKE` | Intake channel | `orion:tts:intake` |
| `TTS_MODEL_NAME` | TTS Model path/name | `tts_models/en/ljspeech/tacotron2-DDC` |
| `TTS_USE_GPU` | Enable GPU usage | `true` |

## Development

Run sanity check:
```bash
python -m compileall services/orion-whisper-tts
```
