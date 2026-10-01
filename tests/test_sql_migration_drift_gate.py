"""Gate tests for orion/sql_migration_drift.py: did every merged hand-applied migration reach
the live database?

The incident replays use the REAL migration files from services/orion-sql-db/ and a live state
built by applying the real corpus, then removing exactly what was missing in production:
- PR #2424: hardware_watch_incident table never created -> orion-hardware-watch crash-looped 13x.
- PR #2400: substrate_node_prediction_error_baseline.last_value_observed_at never added ->
  attention logged a warning and silently ran degraded.
- PR #2400: manual_migration_chat_projection_pe_baseline_v3_reset.sql is a data reset; the schema
  cannot show whether it ran, so it must say "verify manually", never RED and never "applied".

Earlier history these keep honest: the first version of this gate (regex over comment-stripped
text) found manual_migration_substrate_reverie_thought_expectation.sql unapplied on its first run.

DB-free and git-free on purpose (CI runs them in the static-gates job).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion import sql_migration_drift as d  # noqa: E402

MIG_DIR = REPO_ROOT / "services" / "orion-sql-db"
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=200)


def mf(name, text, added=OLD, changed=None):
    return d.MigrationFile(name, text, added, changed or added)


def live(tables=(), columns=(), indexes=None, sequences=()):
    return d.LiveState(set(tables), set(columns), dict(indexes or {}), set(sequences))


def creates(sql):
    return [(e.kind, e.name, e.table) for e in d.parse_migration(sql).effects if e.op == "create"]


def real_corpus(recent=frozenset(), recent_at=NOW - timedelta(days=1)):
    """Every real migration, all 'added' at the same instant (replay ties break by name), with
    the named files marked as changed recently."""
    files = []
    for p in sorted(MIG_DIR.glob("*.sql")):
        changed = recent_at if p.name in recent else OLD
        files.append(d.MigrationFile(p.name, p.read_text(errors="replace"), OLD, changed))
    return files


ALL_RECENT = frozenset(p.name for p in MIG_DIR.glob("*.sql"))


def fully_applied(files) -> d.LiveState:
    """Live state after applying every file in replay order -- an independent mini-replay, so
    the incident tests subtract from 'everything applied' rather than from a hand-picked set."""
    state = {k: set() for k in d.KINDS}
    for f in sorted(files, key=lambda f: (f.added_at, f.name)):
        if d._NOT_A_MIGRATION.search(f.text):
            continue
        for e in d.parse_migration(f.text).effects:
            if e.op == "create":
                state[e.kind].add(e.name)
            else:
                state[e.kind].discard(e.name)
                if e.kind == "table":
                    state["column"] = {c for c in state["column"] if not c.startswith(e.name + ".")}
    return live(state["table"], state["column"], {i: True for i in state["index"]}, state["sequence"])


def by_name(report, name):
    return next(f for f in report.files if f.name == name)


# ----------------------------------------------------------------- incident replays


class TestIncidentReplays:
    HW = "manual_migration_hardware_watch_v1.sql"
    PE = "manual_migration_node_prediction_error_baseline_v3_last_value_observed_at.sql"
    RESET = "manual_migration_chat_projection_pe_baseline_v3_reset.sql"

    def test_pr2424_hardware_watch_table_missing_is_red_and_names_the_file(self):
        files = real_corpus(recent={self.HW})
        state = fully_applied(files)
        assert "hardware_watch_incident" in state.tables  # the parser does see the table
        state.tables.discard("hardware_watch_incident")
        for name in [i for i in state.indexes if i.startswith("hardware_watch_incident")]:
            del state.indexes[name]

        report = d.evaluate(files, state, now=NOW)

        assert report.red
        assert report.red_keys() == [f"migration:{self.HW}"]
        hw = by_name(report, self.HW)
        assert hw.status == "MISSING"
        assert ("table", "hardware_watch_incident") in {(p.kind, p.name) for p in hw.problems}
        text = "\n".join(d.alert_lines(report))
        assert self.HW in text
        assert f"psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/{self.HW}" in text

    def test_pr2400_last_value_observed_at_column_missing_is_red(self):
        files = real_corpus(recent={self.PE})
        state = fully_applied(files)
        col = "substrate_node_prediction_error_baseline.last_value_observed_at"
        assert col in state.columns
        state.columns.discard(col)

        report = d.evaluate(files, state, now=NOW)

        assert report.red_keys() == [f"migration:{self.PE}"]
        pe = by_name(report, self.PE)
        assert [(p.kind, p.name, p.status) for p in pe.problems] == [("column", col, "missing")]

    def test_pr2400_data_only_reset_is_verify_manually_not_red_not_applied(self):
        files = real_corpus(recent={self.RESET})
        report = d.evaluate(files, fully_applied(files), now=NOW)
        reset = by_name(report, self.RESET)
        assert reset.status == "DATA"
        assert not reset.red
        assert reset in report.verify_manually()
        assert not report.red

    def test_the_same_missing_table_is_not_red_outside_the_window(self):
        """Ancient migrations for retired tables must not page anyone."""
        files = real_corpus(recent=set())  # hardware_watch changed 200 days ago in this replay
        state = fully_applied(files)
        state.tables.discard("hardware_watch_incident")
        report = d.evaluate(files, state, now=NOW, window_days=30)
        hw = by_name(report, self.HW)
        assert hw.status == "MISSING" and not hw.in_window and not hw.red
        assert not report.red
        # ...but an all-time run (window 0/None) still reports it.
        assert d.evaluate(files, state, now=NOW, window_days=None).red

    def test_the_fully_applied_real_corpus_is_green(self):
        files = real_corpus(recent=ALL_RECENT)
        report = d.evaluate(files, fully_applied(files), now=NOW)
        assert not report.red, [f.summary() for f in report.red_files]


class TestLaterDropsExplainAbsence:
    """The four false alarms the old gate raised against live Postgres on 2026-10-01: objects
    deliberately dropped by a later migration (GPU pool stage 5.6, the action-outcome index
    swap). A permanently-red gate is a gate people learn to ignore."""

    def test_gpu_pool_stage5_drop_makes_legacy_tables_superseded_not_missing(self):
        files = real_corpus(recent=ALL_RECENT)
        state = fully_applied(files)
        assert "durable_gateway_permits" not in state.tables
        report = d.evaluate(files, state, now=NOW)
        for name in ("manual_migration_gateway_capacity_v1.sql", "manual_migration_gpu2_elastic_v1.sql"):
            r = by_name(report, name)
            assert r.status == "SUPERSEDED", (name, r.status, r.problems)
            assert any("gpu_pool_stage5_drop_legacy_tables" in i for i in r.info)
        adm = by_name(report, "manual_migration_durable_resource_admission_v1.sql")
        assert adm.status == "APPLIED"  # durable_admission_runs/events stay live
        assert any("dropped later" in i for i in adm.info)

    def test_action_outcome_index_swapped_by_control_arm_is_not_missing(self):
        files = real_corpus(recent=ALL_RECENT)
        state = fully_applied(files)
        assert "substrate_action_outcomes_dispatch_signal_uidx" not in state.indexes
        report = d.evaluate(files, state, now=NOW)
        assert by_name(report, "manual_migration_action_outcome_ledger.sql").status == "APPLIED"

    def test_drop_ordering_follows_commit_order_not_file_name(self):
        """A drop that was committed BEFORE the create does not explain the create's absence."""
        create = mf("b_create.sql", "create table t (id int);", added=NOW - timedelta(days=2))
        drop_before = mf("a_drop.sql", "drop table if exists t;", added=NOW - timedelta(days=5))
        r = d.evaluate([create, drop_before], live(), now=NOW)
        assert by_name(r, "b_create.sql").status == "MISSING"
        drop_after = mf("a_drop.sql", "drop table if exists t;", added=NOW - timedelta(days=1))
        r = d.evaluate([create, drop_after], live(), now=NOW)
        assert by_name(r, "b_create.sql").status == "SUPERSEDED"
        assert not r.red

    def test_a_drop_that_did_not_run_is_red_on_the_dropping_file(self):
        create = mf("v1.sql", "create table t (id int);", added=NOW - timedelta(days=9))
        drop = mf("v2.sql", "drop table if exists t;", added=NOW - timedelta(days=1))
        r = d.evaluate([create, drop], live(tables={"t"}), now=NOW)
        assert r.red_keys() == ["migration:v2.sql"]
        assert by_name(r, "v2.sql").problems[0].status == "drop_not_applied"

    def test_dropping_a_table_takes_its_columns_and_indexes_with_it(self):
        v1 = mf("v1.sql", "create table t (id int); alter table t add column c int; create index t_c on t (c);",
                added=NOW - timedelta(days=9))
        v2 = mf("v2.sql", "drop table t cascade;", added=NOW - timedelta(days=1))
        r = d.evaluate([v1, v2], live(), now=NOW)
        assert not r.red
        assert by_name(r, "v1.sql").status == "SUPERSEDED"

    def test_drop_then_recreate_in_one_file_expects_the_object(self):
        f = mf("m.sql", "drop index if exists ix; create index ix on t (c);", changed=NOW)
        assert d.evaluate([f], live(), now=NOW).red
        assert not d.evaluate([f], live(indexes={"ix": True}), now=NOW).red


# ------------------------------------------------------------------------- tokenizer


class TestTheParserIgnoresProseAndLiterals:
    def test_line_and_block_comments_are_not_statements(self):
        sql = """
        -- This migration used to CREATE INDEX idx_ghost ON t (c) but we removed it.
        /* create table if not exists ghost (id text); /* nested */ still comment */
        create index if not exists idx_real on t (c);
        """
        assert creates(sql) == [("index", "idx_real", "t")]

    def test_a_commented_out_add_column_is_not_counted(self):
        sql = "-- alter table t add column if not exists ghost text;\nalter table t add column if not exists real_c text;"
        assert creates(sql) == [("column", "t.real_c", "t")]

    def test_string_literals_cannot_declare_objects_or_split_statements(self):
        sql = """insert into notes (body) values ('create table ghost (id int); -- not a comment');
                 comment on table t is 'alter table t add column ghost2 int';
                 create table real_t (id int);"""
        pm = d.parse_migration(sql)
        assert [(e.kind, e.name) for e in pm.effects] == [("table", "real_t")]
        assert pm.data_statements == 1

    def test_escaped_quotes_and_e_strings(self):
        sql = r"""update t set v = 'it''s; create table ghost (x int)';
                  update t set v = E'a\'; create table ghost2 (x int)';
                  create table real_t (id int);"""
        pm = d.parse_migration(sql)
        assert [e.name for e in pm.effects] == ["real_t"]
        assert pm.data_statements == 2

    def test_quoted_identifiers_are_unquoted(self):
        assert creates('create table if not exists "my_t" (id int); alter table "my_t" add column "c" int;') == [
            ("table", "my_t", None), ("column", "my_t.c", "my_t")]

    def test_psql_meta_commands_are_skipped(self):
        sql = "\\set ON_ERROR_STOP on\ncreate table t (id int);"
        assert creates(sql) == [("table", "t", None)]

    def test_unterminated_input_does_not_raise(self):
        for sql in ("create table t (id int", "select 'oops", "do $$ begin", "/* never closed", '"x'):
            d.parse_migration(sql)


class TestTheParserFindsRealStatements:
    def test_create_index_with_every_optional_clause(self):
        sql = """create unique index concurrently if not exists idx_a on only public.t (c);
                 create index idx_b on t (c);"""
        assert creates(sql) == [("index", "idx_a", "t"), ("index", "idx_b", "t")]

    def test_multi_action_alter_table_yields_every_column(self):
        """The old regex gate saw only the FIRST column of a multi-ADD ALTER -- e.g.
        manual_migration_general_reading_v1.sql declares several; it checked one."""
        sql = """alter table t add column if not exists a numeric(10, 2) default 0,
                                add column if not exists b text,
                                add c jsonb,
                                add constraint t_pk primary key (a);"""
        assert creates(sql) == [("column", "t.a", "t"), ("column", "t.b", "t"), ("column", "t.c", "t")]

    def test_the_real_general_reading_migration_declares_all_its_columns(self):
        pm = d.parse_migration((MIG_DIR / "manual_migration_general_reading_v1.sql").read_text())
        cols = {e.name for e in pm.effects if e.kind == "column"}
        assert len(cols) >= 5, cols

    def test_same_column_name_on_two_tables_stays_distinct(self):
        f = mf("m.sql", "alter table bar add column if not exists status text;", changed=NOW)
        r = d.evaluate([f], live(tables={"foo", "bar"}, columns={"foo.status"}), now=NOW)
        assert r.red

    def test_temp_tables_are_not_expected_to_persist(self):
        assert creates("create temp table scratch (id int); create temporary table s2 (id int);") == []

    def test_rename_column_is_drop_plus_create(self):
        pm = d.parse_migration("alter table t rename column old_c to new_c;")
        assert [(e.op, e.name) for e in pm.effects] == [("drop", "t.old_c"), ("create", "t.new_c")]

    def test_do_block_effects_are_conditional_and_never_red(self):
        sql = """do $$
        begin
            if to_regclass('vision_events') is not null then
                alter table vision_events add column if not exists stream_id text;
                create index if not exists ve_idx on vision_events (stream_id);
            end if;
        end $$;"""
        pm = d.parse_migration(sql)
        assert {(e.kind, e.name, e.conditional) for e in pm.effects} == {
            ("column", "vision_events.stream_id", True), ("index", "ve_idx", True)}
        r = d.evaluate([mf("m.sql", sql, changed=NOW)], live(), now=NOW)
        assert not r.red
        assert any("DO block" in i for i in r.files[0].info)

    def test_function_bodies_do_not_count_as_migration_time_objects(self):
        sql = """create or replace function f() returns void language plpgsql as $fn$
                 begin create table ghost (id int); end $fn$;"""
        assert creates(sql) == []

    def test_sequences_are_checked(self):
        f = mf("m.sql", "create sequence if not exists s1;", changed=NOW)
        assert d.evaluate([f], live(), now=NOW).red
        assert not d.evaluate([f], live(sequences={"s1"}), now=NOW).red


# ---------------------------------------------------------------------- status rules


class TestStatusRules:
    def test_present_but_invalid_index_is_a_failure(self):
        f = mf("m.sql", "create index concurrently if not exists idx_a on t (c);", changed=NOW)
        assert d.evaluate([f], live(indexes={"idx_a": True}), now=NOW).files[0].status == "APPLIED"
        bad = d.evaluate([f], live(indexes={"idx_a": False}), now=NOW)
        assert bad.files[0].status == "INVALID" and bad.red
        assert "indisvalid" in bad.files[0].problems[0].detail
        assert d.evaluate([f], live(), now=NOW).files[0].status == "MISSING"

    def test_data_only_is_data_and_settings_only_is_unknown(self):
        data = mf("data.sql", "begin; update t set c = 1; commit;", changed=NOW)
        params = mf("params.sql", "alter table t set (autovacuum_vacuum_scale_factor = 0.05);", changed=NOW)
        r = d.evaluate([data, params], live(), now=NOW)
        assert by_name(r, "data.sql").status == "DATA"
        assert by_name(r, "params.sql").status == "UNKNOWN"
        assert not r.red

    def test_green_keys_cover_only_in_window_files(self):
        new = mf("new.sql", "create table a (id int);", changed=NOW)
        old = mf("old.sql", "create table b (id int);")
        r = d.evaluate([new, old], live(tables={"a", "b"}), now=NOW)
        assert r.green_keys() == ["migration:new.sql"]

    def test_an_uncommitted_file_is_treated_as_new(self, tmp_path):
        (tmp_path / d.MIGRATION_SUBDIR).mkdir(parents=True)
        (tmp_path / d.MIGRATION_SUBDIR / "fresh.sql").write_text("create table t (id int);")
        files = d.load_files(tmp_path, {}, now=NOW)
        assert files[0].changed_at == NOW
        assert d.evaluate(files, live(), now=NOW).red

    def test_header_named_script_is_offered_as_the_apply_path(self):
        sql = "-- PRODUCTION: run scripts/gpu_pool_stage5_snapshot_and_drop.sh, not this file.\ncreate table t (id int);"
        r = d.evaluate([mf("m.sql", sql, changed=NOW)], live(), now=NOW)
        assert "scripts/gpu_pool_stage5_snapshot_and_drop.sh" in r.files[0].apply_command()


class TestEscapeHatchesLiveInTheFile:
    def test_superseded_marker(self):
        v2 = mf("v2.sql", "select 1;")
        v1 = mf("v1.sql", "-- ORION-MIGRATION-SUPERSEDED-BY: v2.sql\ncreate index if not exists idx_a on t (c);", changed=NOW)
        r = d.evaluate([v1, v2], live(), now=NOW)
        assert by_name(r, "v1.sql").status == "SUPERSEDED" and not r.red

    def test_superseded_marker_pointing_at_nothing_is_drift(self):
        v1 = mf("v1.sql", "-- ORION-MIGRATION-SUPERSEDED-BY: nope.sql\ncreate index idx_a on t (c);", changed=NOW)
        r = d.evaluate([v1], live(indexes={"idx_a": True}), now=NOW)
        assert r.red and "does not exist" in r.files[0].marker_error

    def test_not_a_migration_is_skipped_and_not_replayed(self):
        dump = mf("dump.sql", "-- ORION-MIGRATION-NOT-A-MIGRATION: pg_dump output\ncreate table ghost (id int);", changed=NOW)
        r = d.evaluate([dump], live(), now=NOW)
        assert r.files[0].status == "SKIPPED" and not r.red

    def test_absent_ok_needs_a_reason_and_then_silences_only_that_object(self):
        sql = ("-- ORION-MIGRATION-ABSENT-OK: t_old retired with the v1 reader, see PR #1\n"
               "create table t_old (id int); create table t_new (id int);")
        r = d.evaluate([mf("m.sql", sql, changed=NOW)], live(), now=NOW)
        assert [p.name for p in r.files[0].problems] == ["t_new"]
        bare = mf("m.sql", "-- ORION-MIGRATION-ABSENT-OK: t_old\ncreate table t_old (id int);", changed=NOW)
        r = d.evaluate([bare], live(), now=NOW)
        assert r.red and "reason" in r.files[0].marker_error


# ------------------------------------------------------------------ real corpus


class TestAgainstTheRealMigrationDirectory:
    def test_every_real_migration_parses_without_raising_and_declares_sane_names(self):
        paths = sorted(MIG_DIR.glob("*.sql"))
        assert len(paths) > 100, f"only found {len(paths)} migrations -- wrong directory?"
        ident = re.compile(r"^[a-z_][a-z0-9_$]*(\.[a-z_][a-z0-9_$]*)?$")
        for p in paths:
            pm = d.parse_migration(p.read_text(errors="replace"))
            for e in pm.effects:
                assert e.kind in d.KINDS, (p.name, e)
                assert ident.match(e.name), (p.name, e)
                assert e.name.split(".")[-1] not in d._NOT_A_COLUMN, (p.name, e)
                if e.kind == "column":
                    assert e.name.split(".")[0] == e.table, (p.name, e)

    def test_the_corpus_still_declares_a_meaningful_number_of_objects(self):
        """A regex change that makes the gate silently blind would still exit 0."""
        total = sum(len(d.parse_migration(p.read_text(errors="replace")).effects)
                    for p in MIG_DIR.glob("*.sql"))
        assert total > 300, total

    def test_every_real_migration_has_a_verdict(self):
        """No file falls through: each one is schema-checked, data, settings-only or skipped."""
        files = real_corpus()
        report = d.evaluate(files, fully_applied(files), now=NOW, window_days=None)
        assert {f.status for f in report.files} <= {"APPLIED", "DATA", "UNKNOWN", "SUPERSEDED", "SKIPPED", "CONDITIONAL"}
        assert len(report.files) == len(files)

    def test_the_reverie_expectation_migration_declares_what_it_should(self):
        """The migration whose absence the first version of this gate found."""
        pm = d.parse_migration((MIG_DIR / "manual_migration_substrate_reverie_thought_expectation.sql").read_text())
        cols = {e.name.split(".")[1] for e in pm.effects if e.kind == "column"}
        assert cols == {"expectation", "expectation_checkable_by", "expectation_verdict", "expectation_scored_at"}
        assert ("index", "idx_substrate_reverie_thought_expectation_pending", "substrate_reverie_thought") in [
            (e.kind, e.name, e.table) for e in pm.effects]


def test_cli_script_imports_and_uses_the_module():
    spec = importlib.util.spec_from_file_location(
        "check_sql_migrations_applied", REPO_ROOT / "scripts" / "check_sql_migrations_applied.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.drift is d


def test_commit_times_reads_first_and_last_commit():
    try:
        times = d.commit_times(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - no git / odd checkout
        pytest.skip(f"git history unavailable: {exc}")
    if "manual_migration_hardware_watch_v1.sql" not in times:
        pytest.skip("shallow history")
    first, last = times["manual_migration_hardware_watch_v1.sql"]
    assert first <= last
