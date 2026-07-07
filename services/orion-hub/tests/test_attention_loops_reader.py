from datetime import datetime, timezone, timedelta

import scripts.attention_loops_store as store


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Conn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *a, **k):
        return _Result(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return _Conn(self._rows)


def test_load_pending_loops_filters_and_builds(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [
        {  # old enough + has description -> surfaces
            "theme_key": "t1", "loop_id": "open-loop-1", "salience": 0.8,
            "features": {"evidence_strength": 0.9}, "description": "reactor drift",
            "created_at": now - timedelta(seconds=600),
            "recurrence_count": 3, "first_seen": now - timedelta(seconds=600),
        },
        {  # too new -> filtered out by SURFACE_MIN_AGE_SEC
            "theme_key": "t2", "loop_id": "open-loop-2", "salience": 0.9,
            "features": {}, "description": "",
            "created_at": now - timedelta(seconds=10),
            "recurrence_count": 1, "first_seen": now - timedelta(seconds=10),
        },
    ]

    monkeypatch.setattr(store, "_engine", lambda: _Engine(rows))
    out = store.load_pending_loops()
    assert len(out) == 1
    loop, first_seen, recurrence, narrative = out[0]
    assert loop.id == "open-loop-1"
    assert loop.description == "reactor drift"  # description used, not theme_key
    assert loop.salience_features == {"evidence_strength": 0.9}
    assert recurrence == 3


def test_load_pending_loops_falls_back_to_theme_key(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [{
        "theme_key": "t-fallback", "loop_id": "open-loop-9", "salience": 0.7,
        "features": "{}", "description": "",  # empty desc + string features json
        "created_at": now - timedelta(seconds=600),
        "recurrence_count": 1, "first_seen": now - timedelta(seconds=600),
    }]

    monkeypatch.setattr(store, "_engine", lambda: _Engine(rows))
    out = store.load_pending_loops()
    assert len(out) == 1
    assert out[0][0].description == "t-fallback"  # fell back to theme_key
