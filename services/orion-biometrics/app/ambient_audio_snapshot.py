from __future__ import annotations

"""Load host-local ambient audio snapshot into BiometricsSampleV1.

The loader lives in `orion.telemetry.ambient_audio` so the situation brief
reads the mic file with the exact same validation and staleness rules.
"""

from orion.telemetry.ambient_audio import load_ambient_audio_snapshot

__all__ = ["load_ambient_audio_snapshot"]
