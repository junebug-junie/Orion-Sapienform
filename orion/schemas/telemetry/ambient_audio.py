from __future__ import annotations

"""Host-local transport contract: Athena cabinet ambient audio snapshot.

This is NOT a bus payload and is deliberately not registered in
`orion/schemas/registry.py`. It validates the JSON the host ALSA ambient
audio reader (`scripts/orion_ambient_audio_reader.py`) atomically writes to
`/run/orion-audio/latest.json`. `services/orion-biometrics` reads that file
(bind-mounted read-only) and folds validated levels into
`BiometricsSampleV1.ambient_audio` -- see that model's docstring for the
raw-vs-normalized boundary this schema sits on the raw side of.

On-disk shape (v1), refreshed ~1 Hz:

    {
      "schema": "orion.ambient_audio.v1",
      "status": "ok",
      "received_at": "2026-08-25T05:00:00.123Z",
      "device": "plughw:CARD=CMTECK,DEV=0",
      "window_sec": 0.5,
      "sample_rate": 16000,
      "channels": 1,
      "rms": 412.3,
      "peak": 1820
    }

`status` is one of: ok | stale | error | missing. Failed captures must not
overwrite the last good snapshot on disk; biometrics additionally marks the
sample field stale when age exceeds `AMBIENT_AUDIO_STALE_AFTER_SEC`.

Units are raw PCM magnitudes (RMS float, peak int16 abs max) -- this model
does not normalize or judge. See `orion/telemetry/ambient_audio.py` for the
raw->measurements/pressures step.

Live-listen ring (also host-local, not a bus payload): the same reader
writes every PCM window to `/run/orion-audio/live/slot-NN.pcm`, slot
`seq % AMBIENT_AUDIO_LIVE_RING_SLOTS`. Each file is
`AMBIENT_AUDIO_LIVE_SLOT_HEADER` (seq u64, sample_rate u32, channels u16,
reserved u16) followed by raw S16_LE PCM. Hub tails the ring by seq and
streams it to an operator's browser (`GET /api/cabinet/ambient/live`).
"""

import struct
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

AMBIENT_AUDIO_SCHEMA_V1 = "orion.ambient_audio.v1"

# 16 slots x 0.5s window = 8s of history: enough for a reader that lags a few
# polls, small enough that nothing resembling a recording sits on disk.
AMBIENT_AUDIO_LIVE_RING_SLOTS = 16
AMBIENT_AUDIO_LIVE_SLOT_HEADER = struct.Struct("<QIHH")


def ambient_audio_live_slot_name(seq: int) -> str:
    return f"slot-{seq % AMBIENT_AUDIO_LIVE_RING_SLOTS:02d}.pcm"


class AmbientAudioSnapshotV1(BaseModel):
    """One validated snapshot from the host ambient audio reader."""

    model_config = ConfigDict(extra="ignore")

    schema_: str = Field(alias="schema")
    status: str
    received_at: str
    device: str
    window_sec: float
    sample_rate: int
    channels: int
    rms: float
    peak: int
    error: Optional[str] = None
