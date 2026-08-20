"""The migration drift gate's parser, which is the half that can be wrong silently.

`services/orion-sql-db/*.sql` is 78 hand-applied files with no migration table, no version
stamp, and no way to tell an applied migration from an unapplied one except by looking. The
gate closes that. On its first live run it found `manual_migration_substrate_reverie_thought_
expectation.sql` had never been applied -- 4 columns and an index missing -- while
`services/orion-thought` shipped code using all of them and ran with
ORION_REVERIE_EXPECTATION_SCORING_ENABLED=true. That table had not accepted a write in 39
hours.

These tests are DB-free on purpose: the parser is what decides which objects to look for, and
a parser that under-reports produces a green gate that means nothing. Every test below is
about the parser being wrong in the direction that looks fine.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_sql_migrations_applied",
    REPO_ROOT / "scripts" / "check_sql_migrations_applied.py",
)
gate = importlib.util.module_from_spec(_spec)
# Registered BEFORE exec_module: @dataclass resolves its own annotations through
# sys.modules[cls.__module__], which is None for a module loaded by spec alone. Without this
# line the import fails with a bare AttributeError inside dataclasses.
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)


class TestTheParserFindsRealStatements:
    def test_create_index_with_every_optional_clause(self):
        sql = """
        create unique index concurrently if not exists idx_a on only public.t (c);
        create index idx_b on t (c);
        """
        wanted, _ = gate.parse_migration(sql)
        assert ("index", "idx_a", "t") in wanted
        assert ("index", "idx_b", "t") in wanted

    def test_create_table_and_add_column(self):
        sql = """
        create table if not exists foo (id text);
        alter table foo add column if not exists bar timestamptz;
        alter table public.foo add column baz text;
        """
        wanted, _ = gate.parse_migration(sql)
        assert ("table", "foo", None) in wanted
        assert ("column", "bar", "foo") in wanted
        assert ("column", "baz", "foo") in wanted

    def test_the_same_column_name_on_two_tables_stays_distinct(self):
        """The live-state set is keyed "table.column". Dropping the table qualifier makes
        `foo.status` satisfy a check for `bar.status` -- a missing column reported as applied,
        which is the exact direction of error this gate exists to prevent. Mutation-confirmed
        that nothing else in this file caught it."""
        sql = ("alter table foo add column if not exists status text;\n"
               "alter table bar add column if not exists status text;")
        wanted, _ = gate.parse_migration(sql)
        assert ("column", "status", "foo") in wanted
        assert ("column", "status", "bar") in wanted

    def test_a_column_present_on_a_different_table_does_not_satisfy_the_check(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(gate, "MIGRATION_DIR", tmp_path)
        monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
        f = tmp_path / "m.sql"
        f.write_text("alter table bar add column if not exists status text;")
        # Live database has foo.status but NOT bar.status.
        r = gate.check([f], {}, {"foo", "bar"}, {"foo.status"})[0]
        assert r.status == "MISSING", r.findings

    def test_duplicate_declarations_collapse(self):
        sql = "create index if not exists idx_a on t (c);\ncreate index if not exists idx_a on t (c);"
        wanted, _ = gate.parse_migration(sql)
        assert [w for w in wanted if w[0] == "index"] == [("index", "idx_a", "t")]


class TestTheParserIgnoresProse:
    """These files carry very long `--` headers that discuss the SQL in English. A parser
    that reads its own documentation invents objects that were never meant to exist, and the
    gate goes permanently red on fiction."""

    def test_a_line_comment_mentioning_create_index_is_not_a_statement(self):
        sql = """
        -- This migration used to CREATE INDEX idx_ghost ON t (c) but we removed it.
        -- Apply with: create index concurrently if not exists idx_also_ghost on t (c);
        create index if not exists idx_real on t (c);
        """
        wanted, _ = gate.parse_migration(sql)
        names = {n for k, n, _ in wanted if k == "index"}
        assert names == {"idx_real"}, names

    def test_a_block_comment_is_ignored_too(self):
        sql = "/* create table if not exists ghost (id text); */\ncreate table if not exists real_t (id text);"
        wanted, _ = gate.parse_migration(sql)
        assert {n for k, n, _ in wanted if k == "table"} == {"real_t"}

    def test_a_commented_out_add_column_is_not_counted(self):
        sql = "-- alter table t add column if not exists ghost text;\nalter table t add column if not exists real_c text;"
        wanted, _ = gate.parse_migration(sql)
        assert {n for k, n, _ in wanted if k == "column"} == {"real_c"}


class TestUncheckableStatementsAreCountedNotSwallowed:
    """A file full of backfills must not read as 'applied'. UNKNOWN is not a pass."""

    def test_backfills_and_drops_are_counted(self):
        sql = """
        update t set c = 1 where c is null;
        insert into t (id) values ('x');
        delete from t where id = 'y';
        drop index if exists idx_old;
        alter table t set (autovacuum_vacuum_scale_factor = 0.05);
        """
        wanted, unchecked = gate.parse_migration(sql)
        assert wanted == []
        assert unchecked >= 5, unchecked

    def test_a_file_with_nothing_checkable_reports_unknown_not_applied(self):
        r = gate.FileReport(path="x.sql", findings=[], unchecked=3)
        assert r.status == "UNKNOWN"
        assert r.status != "APPLIED"


class TestStatusRules:
    def test_a_present_but_invalid_index_is_a_failure_not_a_pass(self, tmp_path, monkeypatch):
        """An interrupted CREATE INDEX CONCURRENTLY leaves indisvalid=false. IF NOT EXISTS
        will not rebuild it and the planner will not use it -- so re-running the migration is
        a silent no-op and queries quietly seq-scan. Reporting that as applied would be worse
        than having no gate.

        Drives the real check() against a live-state map, NOT a hand-built Finding. An
        earlier version of this test asserted on FileReport.status given a Finding someone had
        already labelled "invalid", which tests the arithmetic and not the lifecycle --
        mutation-confirmed: replacing check()'s `elif not indexes[name]` with `elif False`
        left the whole suite green.
        """
        monkeypatch.setattr(gate, "MIGRATION_DIR", tmp_path)
        monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
        f = tmp_path / "m.sql"
        f.write_text("create index concurrently if not exists idx_a on t (c);")

        valid = gate.check([f], {"idx_a": True}, set(), set())[0]
        assert valid.status == "APPLIED"

        invalid = gate.check([f], {"idx_a": False}, set(), set())[0]
        assert invalid.status == "INVALID"
        assert invalid.findings[0].status == "invalid"
        assert "indisvalid" in invalid.findings[0].detail

        absent = gate.check([f], {}, set(), set())[0]
        assert absent.status == "MISSING"

    def test_missing_beats_applied_within_one_file(self):
        r = gate.FileReport(path="x.sql", findings=[
            gate.Finding("table", "t", None, "applied"),
            gate.Finding("index", "idx_a", "t", "missing")])
        assert r.status == "MISSING"

    def test_invalid_beats_missing(self):
        r = gate.FileReport(path="x.sql", findings=[
            gate.Finding("index", "idx_a", "t", "missing"),
            gate.Finding("index", "idx_b", "t", "invalid")])
        assert r.status == "INVALID"


class TestTheEscapeHatchesMustBeWrittenIntoTheFile:
    """A permanently-red gate is a gate people learn to ignore. But an exception that lives in
    a commit message is not an exception, it is drift with an alibi."""

    def test_superseded_turns_missing_into_expected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "MIGRATION_DIR", tmp_path)
        (tmp_path / "v2.sql").write_text("drop index if exists idx_a;")
        f = tmp_path / "v1.sql"
        f.write_text("-- ORION-MIGRATION-SUPERSEDED-BY: v2.sql\ncreate index if not exists idx_a on t (c);")
        monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
        r = gate.check([f], {}, set(), set())[0]
        assert r.status == "SUPERSEDED"
        assert r.superseded_by == "v2.sql"

    def test_a_supersede_marker_pointing_at_nothing_is_itself_drift(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "MIGRATION_DIR", tmp_path)
        monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
        f = tmp_path / "v1.sql"
        f.write_text("-- ORION-MIGRATION-SUPERSEDED-BY: does_not_exist.sql\ncreate index if not exists idx_a on t (c);")
        r = gate.check([f], {}, set(), set())[0]
        assert r.status == "MISSING"
        assert "does not exist" in (r.marker_error or "")

    def test_not_a_migration_skips_the_file_entirely(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "MIGRATION_DIR", tmp_path)
        monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
        f = tmp_path / "dump.sql"
        f.write_text("-- ORION-MIGRATION-NOT-A-MIGRATION: pg_dump output\ncreate index ix_x on t (c);")
        r = gate.check([f], {}, set(), set())[0]
        assert r.status == "SKIPPED"
        assert r.findings == []


class TestAgainstTheRealMigrationDirectory:
    """The parser must actually work on the real corpus, not just synthetic input."""

    def test_every_migration_file_parses_without_raising(self):
        paths = sorted((REPO_ROOT / "services" / "orion-sql-db").glob("*.sql"))
        assert len(paths) > 50, f"only found {len(paths)} migrations -- wrong directory?"
        for p in paths:
            gate.parse_migration(p.read_text(errors="replace"))

    def test_the_corpus_still_declares_a_meaningful_number_of_objects(self):
        """Guards the failure mode where a regex change makes the gate silently blind: it
        would still exit 0, having checked nothing."""
        total = 0
        for p in sorted((REPO_ROOT / "services" / "orion-sql-db").glob("*.sql")):
            wanted, _ = gate.parse_migration(p.read_text(errors="replace"))
            total += len(wanted)
        assert total > 100, f"parser found only {total} declarative objects across the corpus"

    def test_the_reverie_expectation_migration_declares_what_it_should(self):
        """The migration whose absence this gate found on its first run."""
        p = (REPO_ROOT / "services" / "orion-sql-db"
             / "manual_migration_substrate_reverie_thought_expectation.sql")
        wanted, _ = gate.parse_migration(p.read_text())
        cols = {n for k, n, _ in wanted if k == "column"}
        assert cols == {"expectation", "expectation_checkable_by",
                        "expectation_verdict", "expectation_scored_at"}, cols
        assert ("index", "idx_substrate_reverie_thought_expectation_pending",
                "substrate_reverie_thought") in wanted
