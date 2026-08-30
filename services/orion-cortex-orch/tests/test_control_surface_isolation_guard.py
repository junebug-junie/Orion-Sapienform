"""The suite must be structurally incapable of writing the live control surface.

`test_auto_router.py` sets `routing.chat_reflective_lane_threshold` to exercise the
routing gate, through a module-global store that resolves its database from the
ambient environment. `Dockerfile` COPYs `tests/` into the image, and the container
has both SQLAlchemy and a production `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL`, so
`docker compose exec cortex-orch python -m pytest tests/` would write Orion's live
routing threshold and never restore it.

Production has already been written this way once, from the hub suite:
`value=0.5, actor="scheduler_seed"`, 4,925 updates on the row.
"""

from __future__ import annotations

import os

from orion.substrate import mutation_control_surface


def test_ambient_database_url_cannot_reach_the_control_surface(monkeypatch):
    """The autouse fixture in conftest.py must survive a test that sets the env
    itself -- otherwise it only protects against the ambient case."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@production-would-be-here:5432/conjourney")
    store = mutation_control_surface.control_surface_store()
    assert store.postgres_url is None
    assert store.source_kind() == "sqlite"


def test_writes_from_this_suite_land_in_the_isolated_store():
    mutation_control_surface.set_chat_reflective_lane_threshold(
        value=0.95, actor="test_control_surface_isolation_guard"
    )
    store = mutation_control_surface.control_surface_store()
    assert store.source_kind() == "sqlite"
    assert mutation_control_surface.get_chat_reflective_lane_threshold() == 0.95


def test_no_production_postgres_env_is_visible_to_this_suite():
    for key in ("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "SUBSTRATE_POLICY_POSTGRES_URL", "DATABASE_URL"):
        assert not os.getenv(key), f"{key} is visible to the test suite; the control surface is not isolated"
