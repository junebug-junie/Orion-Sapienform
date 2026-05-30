#!/usr/bin/env python3
"""Synthesize a short XTTS smoke WAV (run inside the whisper-tts container)."""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

# Container layout: /app = service root (app/, scripts/)
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.settings import Settings  # noqa: E402
from app.tts import TTSEngine  # noqa: E402


def main() -> int:
    out = Path(os.getenv("SMOKE_OUTPUT", "/tmp/orion_xtts_smoke.wav"))
    text = os.getenv(
        "SMOKE_TEXT",
        "Hello Juniper. This is Orion testing the upgraded voice engine.",
    )
    cfg = Settings()
    options = None
    speaker_wav = os.getenv("TTS_DEFAULT_SPEAKER_WAV") or cfg.tts_default_speaker_wav
    if speaker_wav:
        options = {
            "speaker_wav": speaker_wav,
            "split_sentences": cfg.tts_split_sentences,
        }

    voice_id = os.getenv("TTS_DEFAULT_SPEAKER") or cfg.tts_default_speaker
    language = os.getenv("TTS_DEFAULT_LANGUAGE", cfg.tts_default_language)

    engine = TTSEngine()
    result = engine.synthesize_to_b64(
        text,
        voice_id=voice_id,
        language=language,
        options=options,
    )
    out.write_bytes(base64.b64decode(result.audio_b64))
    print(f"Wrote {out} metadata={result.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
