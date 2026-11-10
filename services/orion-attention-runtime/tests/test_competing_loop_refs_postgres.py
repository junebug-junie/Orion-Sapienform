"""The competition read's SQL filter, against an explicitly supplied disposable Postgres only.

Review finding 2026-11-10: the filter kept only ``node:%`` refs, so a world
goal (``world:chat``, ``world:reading``) could never win goal precedence over a
body node the competition held. The unit test fakes the row, so only a real
query can pin the filter.
"""
import json
import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

from app.store import AttentionRuntimeStore


@pytest.fixture
def store():
    uri = os.environ.get("FOCUS_RUN_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("FOCUS_RUN_TEST_POSTGRES_URI must name a disposable test DB")
    engine = create_engine(uri)
    schema = "competing_refs_test_" + uuid4().hex
    with engine.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped = create_engine(uri, connect_args={"options": f"-c search_path={schema}"})
    with scoped.begin() as c:
        # Only the two columns the reader touches.
        c.execute(text(
            "CREATE TABLE substrate_attention_broadcast_projection "
            "(projection_json JSONB NOT NULL, generated_at TIMESTAMPTZ NOT NULL)"
        ))
    instance = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
    instance._engine = scoped
    yield instance
    scoped.dispose()
    engine.dispose()


def test_competition_read_keeps_world_refs(store):
    loops = [
        {"source_refs": ["world:chat", "node:substrate.execution", "turn:abc"]},
        {"source_refs": ["world:reading", "concept:x"]},
    ]
    with store._engine.begin() as c:
        c.execute(
            text("INSERT INTO substrate_attention_broadcast_projection VALUES (CAST(:p AS JSONB), now())"),
            {"p": json.dumps({"frame": {"open_loops": loops}})},
        )
    assert store.load_competing_loop_refs(max_age_sec=120.0) == {
        "world:chat", "world:reading", "node:substrate.execution"
    }
