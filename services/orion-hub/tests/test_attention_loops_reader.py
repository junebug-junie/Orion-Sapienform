from datetime import datetime, timezone, timedelta

import scripts.attention_loops_store as store


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


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


def test_latest_salience_for_theme_dict_features(monkeypatch):
    rows = [{"salience": 0.75, "features": {"evidence_strength": 0.9}}]
    monkeypatch.setattr(store, "_engine", lambda: _Engine(rows))
    salience, features = store.latest_salience_for_theme("t1")
    assert salience == 0.75
    assert features == {"evidence_strength": 0.9}


def test_latest_salience_for_theme_string_features(monkeypatch):
    rows = [{"salience": 0.5, "features": '{"recurrence": 0.4}'}]
    monkeypatch.setattr(store, "_engine", lambda: _Engine(rows))
    salience, features = store.latest_salience_for_theme("t2")
    assert salience == 0.5
    assert features == {"recurrence": 0.4}


def test_latest_salience_for_theme_no_row(monkeypatch):
    monkeypatch.setattr(store, "_engine", lambda: _Engine([]))
    assert store.latest_salience_for_theme("missing") == (0.0, {})


# --- SURFACE_MIN_SALIENCE recalibration regression coverage -----------------
#
# Confirmed live 2026-07-30/31 (`scripts/analysis/measure_attention_surface_
# threshold_calibration.py`): the real attention_salience_trace table's max
# score EVER recorded is 0.490 -- structurally below the old 0.5 gate, every
# single time, since the gate was introduced 2026-07-07. The rows below use
# the exact real per-theme "latest score" shape measured live on 2026-07-31
# (6 distinct theme_keys, scores 0.356-0.458, all < 0.5), not synthetic
# round numbers, so this test replays reality rather than asserting the fix
# in prose.
#
# `_Conn`/`_Engine` above (shared by the rest of this file) don't apply the
# `min_sal`/`limit` bind params -- they just hand back whatever row list was
# constructed, because `load_pending_loops()`'s salience filter lives in the
# SQL WHERE clause, not in Python. A mock that ignores those params would
# silently pass this test regardless of what SURFACE_MIN_SALIENCE is set to
# -- exactly the kind of test that would NOT have caught the original bug.
# `_FilteringConn` instead re-implements that one clause so these tests
# actually exercise the real gate.


class _FilteringConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _stmt, params=None):
        params = params or {}
        min_sal = params.get("min_sal")
        rows = self._rows
        if min_sal is not None:
            rows = [r for r in rows if float(r["salience"]) >= min_sal]
        return _Result(rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FilteringEngine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return _FilteringConn(self._rows)


_REAL_LATEST_SCORES_2026_07_31 = {
    "open-loop-e11d318a9036": 0.458,
    "open-loop-3be13c7644a0": 0.413,
    "open-loop-8f8766373aba": 0.385,
    "open-loop-1c822db7221d": 0.384,
    "open-loop-733aad95961b": 0.366,
    "open-loop-9d84d08cddf5": 0.356,
}


def _real_shaped_rows(now):
    return [
        {
            "theme_key": theme_key, "loop_id": theme_key, "salience": salience,
            "features": {}, "description": f"loop {theme_key}",
            "created_at": now - timedelta(days=1),
            "recurrence_count": 10, "first_seen": now - timedelta(days=6),
        }
        for theme_key, salience in _REAL_LATEST_SCORES_2026_07_31.items()
    ]


def test_old_threshold_would_block_every_real_row(monkeypatch):
    """Documents the bug directly: at the OLD SURFACE_MIN_SALIENCE=0.5, not a
    single real historical score would ever have surfaced -- this is exactly
    why attention_loop_outcome had zero rows, ever.
    """
    now = datetime.now(timezone.utc)
    rows = _real_shaped_rows(now)
    monkeypatch.setattr(store, "SURFACE_MIN_SALIENCE", 0.5)
    monkeypatch.setattr(store, "_engine", lambda: _FilteringEngine(rows))
    assert store.load_pending_loops() == []
    assert all(r["salience"] < 0.5 for r in rows)


def test_recalibrated_threshold_surfaces_real_historical_loops(monkeypatch):
    """Proves the recalibration actually unblocks the gate: at the new
    SURFACE_MIN_SALIENCE=0.40, the two highest real historical scores
    surface, matching the live measurement script's "2/6 real themes" result
    for this exact snapshot -- not just any surfacing, the measured amount.
    """
    now = datetime.now(timezone.utc)
    rows = _real_shaped_rows(now)
    assert store.SURFACE_MIN_SALIENCE == 0.40
    monkeypatch.setattr(store, "_engine", lambda: _FilteringEngine(rows))
    out = store.load_pending_loops()
    surfaced_ids = {loop.id for loop, *_ in out}
    assert surfaced_ids == {"open-loop-e11d318a9036", "open-loop-3be13c7644a0"}


def test_recalibrated_threshold_still_filters_low_scores(monkeypatch):
    """The recalibration is a real gate, not a rubber stamp -- the four
    lowest real historical scores (0.356-0.385) stay below 0.40 and must not
    surface.
    """
    now = datetime.now(timezone.utc)
    rows = _real_shaped_rows(now)
    monkeypatch.setattr(store, "_engine", lambda: _FilteringEngine(rows))
    out = store.load_pending_loops()
    surfaced_ids = {loop.id for loop, *_ in out}
    for theme_key, salience in _REAL_LATEST_SCORES_2026_07_31.items():
        if salience < 0.40:
            assert theme_key not in surfaced_ids


def test_surface_min_age_sec_unchanged_and_still_enforced(monkeypatch):
    """Companion age gate (measured live 2026-07-31 to NOT be currently
    binding on real data) is left untouched by this patch -- confirm it
    still filters a too-new row even at the new salience threshold.
    """
    assert store.SURFACE_MIN_AGE_SEC == 300.0
    now = datetime.now(timezone.utc)
    rows = [{
        "theme_key": "brand-new", "loop_id": "open-loop-new", "salience": 0.9,
        "features": {}, "description": "",
        "created_at": now - timedelta(seconds=10),
        "recurrence_count": 1, "first_seen": now - timedelta(seconds=10),
    }]
    monkeypatch.setattr(store, "_engine", lambda: _FilteringEngine(rows))
    assert store.load_pending_loops() == []
