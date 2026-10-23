"""Act downstream use (app/act_downstream_use.py): pure contract checks. SQL behaviour is pinned
against a real Postgres in tests/test_act_downstream_use_pg.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

MIGRATION = REPO / "services" / "orion-sql-db" / "manual_migration_act_downstream_use_v1.sql"


def test_arc_source_tables_match_the_temporal_self_sources() -> None:
    # A direct ref only links if its table can be arc evidence; a new arc source must show up here.
    from app.act_downstream_use import ARC_SOURCE_TABLES

    src = (REPO / "orion" / "temporal_self" / "sources.py").read_text()
    assert set(ARC_SOURCE_TABLES) == set(re.findall(r'table="([a-z_]+)"', src))


def test_every_explicit_path_ends_in_an_arc_source_table() -> None:
    from app.act_downstream_use import ARC_SOURCE_TABLES, ROLLUP_SQL

    targets = set(re.findall(r"'([a-z_]+):' \|\|", str(ROLLUP_SQL)))
    assert targets == {"reverie_visual_chain", "chat_history_log", "retro_play_sessions"}
    assert targets <= set(ARC_SOURCE_TABLES)


def test_outreach_is_not_given_a_path_its_rows_can_never_take() -> None:
    # Temporal Self binds Orion's own chat rows as context, never as arc evidence: outreach has no
    # link path, and must read no_link_path (NULL), never 0% used.
    from app.act_downstream_use import PATH_ACTORS, ROLLUP_SQL

    assert "hub.outreach" not in PATH_ACTORS and "hub.outreach" not in str(ROLLUP_SQL)


def test_model_and_migration_declare_the_same_columns() -> None:
    from app.models.act_downstream_use import ActDownstreamUseSQL

    sql = MIGRATION.read_text()
    body = sql.split("CREATE TABLE IF NOT EXISTS act_downstream_use (")[1].split(");")[0]
    cols = {line.split()[0] for line in body.strip().splitlines() if line.strip()}
    assert cols == set(ActDownstreamUseSQL.__table__.columns.keys())


def test_migration_fits_the_post_pull_auto_apply_allowlist() -> None:
    from orion.sql_migration_drift import auto_apply_refusals

    sql = MIGRATION.read_text()
    assert sql.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer"
    assert auto_apply_refusals(sql, set()) == []


def test_loop_is_on_by_default_and_registered() -> None:
    from app.settings import Settings

    assert Settings.model_fields["act_downstream_use_interval_sec"].default > 0
    env = (SERVICE / ".env_example").read_text()
    assert re.search(r"^ACT_DOWNSTREAM_USE_INTERVAL_SEC=900$", env, re.M)
    assert "act_downstream_use_loop(settings)" in (SERVICE / "app" / "main.py").read_text()
