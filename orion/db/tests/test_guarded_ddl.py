"""GuardedDDL skips only statements the catalog shows are already satisfied (2026-10-31)."""
from __future__ import annotations

from orion.db.guarded_ddl import COLUMNS_SQL, INDEXES_SQL, TABLES_SQL, GuardedDDL

CATALOG = {
    TABLES_SQL: [("chat_message",), ("world_pulse_read_seed",)],
    COLUMNS_SQL: [("chat_message", "trace_id"), ("chat_message", "memory_tier"),
                  ("world_pulse_read_seed", "attempts")],
    INDEXES_SQL: [("idx_chat_message_corr_id",)],
}


def guard():
    ran = []
    return GuardedDDL(lambda sql, *a: ran.append(sql), lambda q: CATALOG[q]), ran


def test_present_columns_indexes_and_tables_are_skipped():
    ddl, ran = guard()
    assert not ddl("ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS trace_id TEXT;")
    assert not ddl("ALTER TABLE IF EXISTS public.chat_message\n  ADD COLUMN IF NOT EXISTS memory_tier TEXT;")
    assert not ddl("CREATE INDEX IF NOT EXISTS idx_chat_message_corr_id ON chat_message (correlation_id);")
    assert not ddl("\n CREATE TABLE IF NOT EXISTS chat_message (\n id TEXT PRIMARY KEY\n);\n")
    assert ran == [] and (ddl.ran, ddl.skipped) == (0, 4)


def test_missing_ones_run():
    ddl, ran = guard()
    assert ddl("ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS memory_status TEXT;")
    assert ddl("CREATE UNIQUE INDEX IF NOT EXISTS uq_new ON chat_message (trace_id);")
    assert ddl("CREATE TABLE IF NOT EXISTS brand_new (id TEXT);")
    assert len(ran) == 3


def test_multi_clause_add_is_skipped_only_when_every_column_is_there():
    ddl, ran = guard()
    assert not ddl("ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS trace_id TEXT,\n"
                   "    ADD COLUMN IF NOT EXISTS memory_tier TEXT;")
    assert ddl("ALTER TABLE world_pulse_read_seed\n    ADD COLUMN IF NOT EXISTS attempts int NOT NULL DEFAULT 0,\n"
               "    ADD COLUMN IF NOT EXISTS stage2_attempts int NOT NULL DEFAULT 0;")


def test_other_alter_shapes_always_run():
    ddl, ran = guard()
    assert ddl("ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS trace_id TEXT, DROP COLUMN old;")
    assert ddl("ALTER TABLE world_pulse_read_seed ADD CONSTRAINT k CHECK (kind IN ('a', 'b'));")
    assert ddl("ALTER TABLE chat_message ADD COLUMN trace_id TEXT;")   # no IF NOT EXISTS: let it fail as before
    assert ddl("UPDATE chat_message SET trace_id = '' WHERE trace_id IS NULL;")
    # a satisfied first statement must never hide a second one
    assert ddl("ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS trace_id TEXT; UPDATE chat_message SET trace_id = '';")
    assert ddl("-- why\nCREATE INDEX IF NOT EXISTS idx_chat_message_corr_id ON chat_message (correlation_id);")


def test_a_table_missing_from_the_snapshot_or_made_this_run_is_left_to_postgres():
    ddl, ran = guard()
    assert ddl("ALTER TABLE IF EXISTS gone ADD COLUMN IF NOT EXISTS x TEXT;")   # lock-free in Postgres anyway
    assert ddl("ALTER TABLE gone ADD COLUMN IF NOT EXISTS x TEXT;")
    ddl("CREATE TABLE IF NOT EXISTS fresh (id TEXT);")
    assert ddl("ALTER TABLE IF EXISTS fresh ADD COLUMN IF NOT EXISTS x TEXT;")
    assert ddl("CREATE INDEX IF NOT EXISTS ix_fresh ON fresh (id);") and not ddl("CREATE INDEX IF NOT EXISTS ix_fresh ON fresh (id);")


def test_quoted_names_keep_their_case():
    catalog = {TABLES_SQL: [("Foo",), ("foo",)], COLUMNS_SQL: [("Foo", "Bar")], INDEXES_SQL: [("Ix",)]}
    ddl = GuardedDDL(lambda *a: None, lambda q: catalog[q])
    assert not ddl('ALTER TABLE "Foo" ADD COLUMN IF NOT EXISTS "Bar" TEXT;')
    assert ddl("ALTER TABLE foo ADD COLUMN IF NOT EXISTS bar TEXT;")      # unquoted folds to foo/bar: missing
    assert not ddl('CREATE INDEX IF NOT EXISTS "Ix" ON "Foo" ("Bar");') and ddl("CREATE INDEX IF NOT EXISTS ix ON foo (id);")
