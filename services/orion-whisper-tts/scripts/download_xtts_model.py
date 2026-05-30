#!/usr/bin/env python3
"""Pre-download the configured Coqui TTS model into the cache (telemetry volume when mounted)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.settings import Settings  # noqa: E402


def main() -> int:
    # Non-interactive CPML acceptance (see TTS.utils.manage.ModelManager.tos_agreed)
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    cfg = Settings()
    cache_dir = os.environ.get("TTS_HOME") or str(Path.home() / ".local/share" / "tts")
    print(f"TTS_HOME/cache={cache_dir}")
    print(f"Downloading model={cfg.tts_model_name} gpu={cfg.tts_use_gpu}")

    from app.tts import _ensure_torch_load_compat  # noqa: E402

    _ensure_torch_load_compat()
    from TTS.api import TTS  # noqa: E402

    TTS(cfg.tts_model_name, gpu=cfg.tts_use_gpu)
    print(f"Done. Weights should be under: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
