"""Round-trip tests for the AutonomyStateV2 Postgres store.

Uses an on-disk SQLite database as a stand-in for the shared Postgres store,
mirroring `orion/autonomy/tests/test_action_outcomes_sql.py`'s convention.
Both `load_autonomy_state_v2` and `save_autonomy_state_v2` must fail open
(never raise) since they sit on the hot chat-turn path.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from orion.autonomy import state_store
from orion.autonomy.models import AutonomyStateV2
from orion.autonomy.state_store import load_autonomy_state_v2, save_autonomy_state_v2


def _make_db(path) -> str:
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE autonomy_state_v2 (
                subject TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    engine.dispose()
    return url


def _make_state(subject: str = "orion", dominant_drive: str | None = "curiosity") -> AutonomyStateV2:
    return AutonomyStateV2(
        subject=subject,
        model_layer="cortex",
        entity_id="orion-prime",
        source="reducer",
        dominant_drive=dominant_drive,
        active_drives=["curiosity", "coherence"],
        unknowns=["whether juniper is asleep"],
    )


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    state_store._ENGINE_CACHE.clear()
    yield
    state_store._ENGINE_CACHE.clear()


def test_load_returns_none_when_db_url_unset(monkeypatch) -> None:
    monkeypatch.delenv("ORION_AUTONOMY_STATE_DB_URL", raising=False)
    assert load_autonomy_state_v2(subject="orion") is None


def test_save_is_noop_when_db_url_unset(monkeypatch) -> None:
    monkeypatch.delenv("ORION_AUTONOMY_STATE_DB_URL", raising=False)
    # Must not raise even though there's nowhere to write.
    save_autonomy_state_v2(subject="orion", state=_make_state())


def test_round_trip_save_then_load(tmp_path, monkeypatch) -> None:
    url = _make_db(tmp_path / "autonomy_state.db")
    monkeypatch.setenv("ORION_AUTONOMY_STATE_DB_URL", url)

    state = _make_state(subject="orion", dominant_drive="curiosity")
    save_autonomy_state_v2(subject="orion", state=state)

    loaded = load_autonomy_state_v2(subject="orion")
    assert loaded is not None
    assert loaded.subject == "orion"
    assert loaded.dominant_drive == "curiosity"
    assert loaded.active_drives == ["curiosity", "coherence"]
    assert loaded.unknowns == ["whether juniper is asleep"]
    assert loaded.schema_version == "autonomy.state.v2"


def test_save_upserts_existing_subject(tmp_path, monkeypatch) -> None:
    url = _make_db(tmp_path / "autonomy_state.db")
    monkeypatch.setenv("ORION_AUTONOMY_STATE_DB_URL", url)

    save_autonomy_state_v2(subject="orion", state=_make_state(dominant_drive="curiosity"))
    save_autonomy_state_v2(subject="orion", state=_make_state(dominant_drive="coherence"))

    loaded = load_autonomy_state_v2(subject="orion")
    assert loaded is not None
    assert loaded.dominant_drive == "coherence"

    engine = create_engine(url)
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM autonomy_state_v2")).scalar()
    engine.dispose()
    assert count == 1


def test_round_trip_is_subject_scoped(tmp_path, monkeypatch) -> None:
    url = _make_db(tmp_path / "autonomy_state.db")
    monkeypatch.setenv("ORION_AUTONOMY_STATE_DB_URL", url)

    save_autonomy_state_v2(subject="orion", state=_make_state(subject="orion", dominant_drive="curiosity"))
    save_autonomy_state_v2(subject="juniper", state=_make_state(subject="juniper", dominant_drive="coherence"))

    assert load_autonomy_state_v2(subject="orion").dominant_drive == "curiosity"
    assert load_autonomy_state_v2(subject="juniper").dominant_drive == "coherence"


def test_load_returns_none_for_missing_subject(tmp_path, monkeypatch) -> None:
    url = _make_db(tmp_path / "autonomy_state.db")
    monkeypatch.setenv("ORION_AUTONOMY_STATE_DB_URL", url)

    assert load_autonomy_state_v2(subject="nobody") is None


def test_load_fails_open_on_connection_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_STATE_DB_URL", f"sqlite:///{tmp_path / 'missing_table.db'}")
    # DB exists but has no autonomy_state_v2 table -> read raises internally -> None.
    assert load_autonomy_state_v2(subject="orion") is None


def test_save_fails_open_on_connection_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_STATE_DB_URL", f"sqlite:///{tmp_path / 'missing_table.db'}")
    # DB exists but has no autonomy_state_v2 table -> write raises internally -> swallowed.
    save_autonomy_state_v2(subject="orion", state=_make_state())
