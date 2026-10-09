"""Pin: the reverie visual seed stays dead for auto-saved memories (memory redesign Stage 0A).

Stage 0A starts writing an `auto_activate` history row for every memory the
formation policy saves on its own. The reverie visual seed
(`load_latest_memory_crystallization`) treats "has a history row with
op='approve'" as "someone actually reviewed this". If it ever widened to "has
any history row", every auto-saved chat line ("sup", "Run github compactor.")
would start seeding generated images. The spec keeps the seed dead until
Stage 2, so this runs the real SQL against a real (SQLite) database.
"""

from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")


def _engine_with_rows(rows):
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE memory_crystallizations ("
                " crystallization_id TEXT PRIMARY KEY, status TEXT, summary TEXT, created_at TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE memory_crystallization_history ("
                " crystallization_id TEXT, op TEXT, actor TEXT)"
            )
        )
        for cid, summary, created, ops in rows:
            conn.execute(
                text(
                    "INSERT INTO memory_crystallizations VALUES (:c, 'active', :s, :t)"
                ),
                {"c": cid, "s": summary, "t": created},
            )
            for op, actor in ops:
                conn.execute(
                    text("INSERT INTO memory_crystallization_history VALUES (:c, :o, :a)"),
                    {"c": cid, "o": op, "a": actor},
                )
    return engine


def _load(monkeypatch, engine):
    import app.store as store

    monkeypatch.setattr(store, "_get_engine", lambda: engine)
    return store.load_latest_memory_crystallization()


def test_auto_activate_history_does_not_seed_reverie(monkeypatch):
    engine = _engine_with_rows(
        [("auto-1", "Run github compactor.", "2026-10-01", [("auto_activate", "system:formation_policy")])]
    )
    assert _load(monkeypatch, engine) is None


def test_only_a_real_approval_seeds_reverie(monkeypatch):
    engine = _engine_with_rows(
        [
            ("approved-1", "An approved memory", "2026-09-01", [("approve", "orion_journal")]),
            # Newer, but auto-saved: must not win.
            ("auto-1", "sup", "2026-10-01", [("auto_activate", "system:formation_policy")]),
        ]
    )
    assert _load(monkeypatch, engine) == "An approved memory"
