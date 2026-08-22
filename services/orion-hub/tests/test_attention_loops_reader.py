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
        self.executed_sql = []  # captured for SQL-text assertions

    def execute(self, clause, *a, **k):
        self.executed_sql.append(str(clause))
        return _Result(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, rows):
        self._rows = rows
        self.last_conn = None

    def connect(self):
        self.last_conn = _Conn(self._rows)
        return self.last_conn


def test_load_pending_loops_filters_and_builds(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [
        {  # old enough + has description -> surfaces
            "theme_key": "t1", "loop_id": "open-loop-1", "salience": 0.8,
            "features": {"evidence_strength": 0.9}, "description": "reactor drift",
            "why_it_matters": "you flagged it as urgent", "target_type": "anomaly",
            "scope": "chat",
            "created_at": now - timedelta(seconds=600),
            "recurrence_count": 3, "first_seen": now - timedelta(seconds=600),
        },
        {  # too new -> filtered out by SURFACE_MIN_AGE_SEC
            "theme_key": "t2", "loop_id": "open-loop-2", "salience": 0.9,
            "features": {}, "description": "",
            "why_it_matters": "", "target_type": "other", "scope": "chat",
            "created_at": now - timedelta(seconds=10),
            "recurrence_count": 1, "first_seen": now - timedelta(seconds=10),
        },
    ]

    engine = _Engine(rows)
    monkeypatch.setattr(store, "_engine", lambda: engine)
    out = store.load_pending_loops()
    assert len(out) == 1
    loop, first_seen, recurrence, narrative, scope = out[0]
    assert loop.id == "open-loop-1"
    assert loop.description == "reactor drift"  # description used, not theme_key
    assert loop.why_it_matters == "you flagged it as urgent"
    assert loop.target_type == "anomaly"
    assert loop.salience_features == {"evidence_strength": 0.9}
    assert recurrence == 3
    assert scope == "chat"


def test_load_pending_loops_query_excludes_already_verdicted_evidence(monkeypatch):
    # Regression, confirmed live 2026-08-22: substrate_reverie_refractory is
    # only a 24h cooldown -- ~22 loops Juniper resolved/dismissed on 2026-08-20
    # silently reappeared once it lapsed, because nothing checked
    # attention_loop_outcome directly. Can't exercise the real WHERE-clause
    # semantics through this mock (Postgres evaluates it, not Python), so this
    # asserts the query TEXT itself carries the exclusion -- a guard against
    # someone deleting the clause, not a full behavioral proof (see the PR
    # report / live before-after verification for that).
    engine = _Engine([])
    monkeypatch.setattr(store, "_engine", lambda: engine)
    store.load_pending_loops()
    sql = engine.last_conn.executed_sql[0]
    assert "attention_loop_outcome" in sql
    assert "resolved" in sql and "dismissed" in sql and "decayed_unattended" in sql
    assert "o.created_at >= t.created_at" in sql


def test_load_pending_loops_falls_back_to_theme_key(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [{
        "theme_key": "t-fallback", "loop_id": "open-loop-9", "salience": 0.7,
        "features": "{}", "description": "",  # empty desc + string features json
        "why_it_matters": "", "target_type": "other", "scope": "reverie",
        "created_at": now - timedelta(seconds=600),
        "recurrence_count": 1, "first_seen": now - timedelta(seconds=600),
    }]

    monkeypatch.setattr(store, "_engine", lambda: _Engine(rows))
    out = store.load_pending_loops()
    assert len(out) == 1
    assert out[0][0].description == "t-fallback"  # fell back to theme_key
    assert out[0][4] == "reverie"


def test_load_pending_loops_defaults_target_type_when_row_carries_an_invalid_value(monkeypatch):
    # Guards card_kind_for_scope's caller against a malformed/pre-migration row
    # taking the whole panel down on a Pydantic Literal mismatch (see
    # attention_loops_store.py::_safe_target_type).
    now = datetime.now(timezone.utc)
    rows = [{
        "theme_key": "t-bad", "loop_id": "open-loop-bad", "salience": 0.7,
        "features": {}, "description": "d",
        "why_it_matters": "", "target_type": "not_a_real_type", "scope": "chat",
        "created_at": now - timedelta(seconds=600),
        "recurrence_count": 1, "first_seen": now - timedelta(seconds=600),
    }]
    monkeypatch.setattr(store, "_engine", lambda: _Engine(rows))
    out = store.load_pending_loops()
    assert out[0][0].target_type == "other"


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


def test_latest_trace_for_theme_reads_scope_in_the_same_query(monkeypatch):
    rows = [{"salience": 0.8, "features": {"evidence_strength": 0.5}, "scope": "reverie"}]
    monkeypatch.setattr(store, "_engine", lambda: _Engine(rows))
    trace = store.latest_trace_for_theme("t1")
    assert trace == {"salience": 0.8, "features": {"evidence_strength": 0.5}, "scope": "reverie"}


def test_latest_trace_for_theme_no_row_defaults_to_chat_scope(monkeypatch):
    # 'chat' (permissive), NOT 'unknown' -- regression from a second review
    # pass: this is called from _close() on a loop_id the Hub panel already
    # showed the user moments earlier, so a miss here is virtually always a
    # transient hiccup, not evidence the loop is chronic_pressure. Defaulting
    # to the restrictive branch would falsely block a legitimate human
    # Resolve/Dismiss click on a DB blip -- preserves
    # latest_salience_for_theme's original "closing a loop never fails"
    # contract. A genuinely-read non-'chat' scope (see the dict/string-features
    # tests above) is the only thing that should ever route to chronic_pressure.
    monkeypatch.setattr(store, "_engine", lambda: _Engine([]))
    assert store.latest_trace_for_theme("missing") == {"salience": 0.0, "features": {}, "scope": "chat"}


class _BoomEngine:
    def connect(self):
        raise RuntimeError("db unreachable")


def test_latest_trace_for_theme_db_failure_defaults_to_chat_scope(monkeypatch):
    monkeypatch.setattr(store, "_engine", lambda: _BoomEngine())
    assert store.latest_trace_for_theme("t1") == {"salience": 0.0, "features": {}, "scope": "chat"}


def test_card_kind_for_scope_allowlists_only_chat():
    assert store.card_kind_for_scope("chat") == "resolvable"
    assert store.card_kind_for_scope("reverie") == "chronic_pressure"
    # Schema-documented but not-yet-produced third scope value -- must default
    # to the safe branch, not silently fall through to resolvable.
    assert store.card_kind_for_scope("broadcast") == "chronic_pressure"
    assert store.card_kind_for_scope("unknown") == "chronic_pressure"
    assert store.card_kind_for_scope("") == "chronic_pressure"
