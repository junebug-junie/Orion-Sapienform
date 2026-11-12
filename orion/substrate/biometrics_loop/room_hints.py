"""Which node-biometrics pressure hints describe the ROOM, not the machine.

athena carries the cabinet sensor board (Nano ESP32: climate, particulate, EM, UV,
vibration, proximity) and the ambient microphone. Their readings land on athena's
``pressure_hints`` next to its CPU/GPU/memory/thermal hints, because athena is the
host that reads them. They measure the room around the machines (a person walking
past, a door, a fan in the hallway), not Orion's machine body.

Juniper approved (2026-11-12) taking them out of the machine-body reading. This
module is the one owner of that list. Two generic consumers read node hints as
"how is the body":

- ``orion/substrate/prediction_error.py::biometrics_prediction_error`` (the
  ``node:substrate.biometrics`` surprise number; definition v3).
- ``orion/substrate/biometrics_loop/pressure_organ.py`` (node pressure detected /
  reinforced / decayed, and the primary hint kind).

Both call :func:`machine_body_hints`. There is no kill switch: a room reading in the
body mean is the thing being retired, so there is nothing to fall back to.

The keys stay on the projection. Named consumers still read them on purpose:
the field digester's dedicated ``cabinet_*`` channels
(``services/orion-field-digester/app/ingest/state_deltas.py``), the ambient spike
consumer (``orion/substrate/cabinet_ambient_spike_consumer.py``) and the Hub
cabinet panels. The cabinet's own attention source is ``node:substrate.cabinet``.
"""

from __future__ import annotations

from typing import Any, Mapping

# grammar semantic_role -> pressure_hints key. Read by grammar_extract.py to write
# the hint; read here to keep it out of the machine-body reading.
CABINET_ROLE_TO_HINT: dict[str, str] = {
    "cabinet_climate_activity_signal": "cabinet_climate_activity",
    "cabinet_particulate_activity_signal": "cabinet_particulate_activity",
    "cabinet_em_activity_signal": "cabinet_em_activity",
    "cabinet_uv_activity_signal": "cabinet_uv_activity",
    "cabinet_vibration_activity_signal": "cabinet_vibration_activity",
    "cabinet_proximity_activity_signal": "cabinet_proximity_activity",
    "cabinet_sensor_staleness_signal": "cabinet_sensor_staleness",
    "cabinet_ambient_audio_activity_signal": "cabinet_ambient_audio_activity",
    "cabinet_ambient_audio_staleness_signal": "cabinet_ambient_audio_staleness",
    "cabinet_ambient_audio_spike_signal": "cabinet_ambient_audio_activity",
}

ROOM_HINT_KEYS: frozenset[str] = frozenset(CABINET_ROLE_TO_HINT.values())


def is_room_hint(key: str) -> bool:
    """True for a hint that reads the room. Any ``cabinet_`` key counts, so a new
    cabinet sensor added to the extractor cannot slip into the body reading."""
    return key in ROOM_HINT_KEYS or key.startswith("cabinet_")


def machine_body_hints(hints: Mapping[str, Any] | None) -> dict[str, Any]:
    """``hints`` without the room keys."""
    return {k: v for k, v in (hints or {}).items() if not is_room_hint(k)}
