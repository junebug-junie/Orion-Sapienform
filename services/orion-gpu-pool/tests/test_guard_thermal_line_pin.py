"""Pin the gpu2 swap guard's thermal decision table (body economy Patch B).

The guard refuses a gpu2 model load at `hot` and when the room is unknown; Juniper has
not decided whether to split or price that, so only where its line comes from may change.
The line (32.0 C, re-arm 30.5 C) is owned by orion/autonomy/thermal_gate.py and reaches
the guard through orion.autonomy.cabinet_heat.read_cabinet_heat. The expected reasons
are written out by hand: a moved line or a guard that stops reading the owner fails here.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from app.guards import GuardReader

CAB = "http://cab/latest"
T0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, temp, age):
        self.status_code, self._body = 200, {"snapshot": {"frame": {"environment": {"temp_c": temp}}}, "age_sec": age}

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _Client:
    def __init__(self):
        self.next = None

    async def get(self, url):
        if isinstance(self.next, Exception):
            raise self.next
        return _Resp(*self.next)


# (seconds since T0, reading or error) -> guard reason (None = clear to load gpu2).
TABLE = [
    (0, ConnectionError(), "unavailable:ConnectionError"),  # unknown at start: refuse
    (30, (31.9, 5.0), None),                                # elevated: allowed
    (60, (32.0, 5.0), "hot:room_at_32.0c"),                 # trip is >=
    (90, (31.0, 5.0), "hot:room_at_31.0c"),                 # under trip, over re-arm: still hot
    (120, ConnectionError(), "hot:room_at_31.0c"),          # one failed read inside grace holds
    (150, (30.6, 5.0), "hot:room_at_30.6c"),
    (180, (30.5, 5.0), None),                               # re-arm releases
    (210, (28.0, 5.0), None),
    (240, (34.0, 5.0), "hot:room_at_34.0c"),
    (900, ConnectionError(), "unavailable:ConnectionError"),  # past grace: unknown, refuse
]


def test_decision_table_is_pinned():
    clock = {"now": T0}
    reader = GuardReader(cabinet_url=CAB, clock=lambda: clock["now"])
    client = _Client()
    got = []
    for sec, reading, _ in TABLE:
        clock["now"] = T0 + timedelta(seconds=sec)
        client.next = reading
        got.append(asyncio.run(reader.read(client))["thermal"])
    assert got == [want for _, _, want in TABLE]
