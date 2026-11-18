"""The blind-arm guard (scripts/sql/2026-11-18_blind_experiment_arms_orion_readonly.sql, issue #3458) against a REAL
Postgres: once applied, Orion's read-only role cannot read a blinded experiment arm by any SQL it can write.

The scratch database gets the real temporal_self_event table (its create migration plus every later
``ALTER TABLE temporal_self_event ADD COLUMN`` in services/orion-sql-db), a logged-in ``orion_readonly`` role with the
grants the live role had before the guard, one row per arm, then the guard file through psql exactly as the apply
command runs it. Every check connects AS orion_readonly, so the role's own search_path is what resolves names.

Database: BLIND_ARM_GUARD_TEST_POSTGRES_URI (CI: a throwaway service container), else a throwaway cluster started
with the local initdb/pg_ctl in tmp_path, else skipped. Never the live database.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args
from urllib.parse import unquote, urlparse

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from orion.curiosity.value import ARM_SELF_INQUIRY, OFFER_ARMS  # noqa: E402
from orion.schemas.dream_cycle import HypothesisArm  # noqa: E402
from orion.temporal_self import sources as S  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts/sql/2026-11-18_blind_experiment_arms_orion_readonly.sql"
MIG_DIR = REPO_ROOT / "services/orion-sql-db"
DREAM_ARMS = get_args(HypothesisArm)
RO_PASSWORD = "test-only-ro"
AT = datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server(tmp_path):
    """dict(host, port, user, password, dbname) of a throwaway server's maintenance database."""
    uri = os.environ.get("BLIND_ARM_GUARD_TEST_POSTGRES_URI", "").strip()
    if uri:
        u = urlparse(uri)
        yield dict(host=u.hostname or "localhost", port=u.port or 5432, user=unquote(u.username or "postgres"),
                   password=unquote(u.password or ""), dbname=u.path.lstrip("/") or "postgres")
        return
    # Live is Postgres 15; prefer it, else the newest (numeric, not text, order).
    bins = sorted(glob.glob("/usr/lib/postgresql/*/bin/initdb"), key=lambda b: (Path(b).parts[-3] != "15",
                                                                              -int(Path(b).parts[-3])))
    if not bins or not shutil.which("psql"):
        if os.environ.get("CI"):
            pytest.fail("CI must provide BLIND_ARM_GUARD_TEST_POSTGRES_URI; a skip here hides the only real-DB proof")
        pytest.skip("no BLIND_ARM_GUARD_TEST_POSTGRES_URI and no local initdb/psql")
    bindir = Path(bins[-1]).parent
    data, sock = tmp_path / "pgdata", tmp_path / "sock"
    sock.mkdir()
    subprocess.run([str(bindir / "initdb"), "-D", str(data), "-U", "postgres", "-A", "trust"],
                   check=True, capture_output=True)
    port = _free_port()
    subprocess.run([str(bindir / "pg_ctl"), "-D", str(data), "-w", "-l", str(tmp_path / "pg.log"),
                    "-o", f"-k {sock} -p {port} -c listen_addresses=''", "start"],
                   check=True, capture_output=True)
    try:
        yield dict(host=str(sock), port=port, user="postgres", password="", dbname="postgres")
    finally:
        subprocess.run([str(bindir / "pg_ctl"), "-D", str(data), "-m", "immediate", "stop"], capture_output=True)


def _exec(conn_args: dict, sql: str, params=None):
    conn = psycopg2.connect(**conn_args)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() if cur.description else None
    finally:
        conn.close()


def _psql_file(admin: dict, path: Path | None = None, sql: str | None = None) -> subprocess.CompletedProcess:
    """psql -v ON_ERROR_STOP=1, the same flags as the guard file's apply command."""
    env = {**os.environ, "PGPASSWORD": admin["password"]}
    cmd = ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-h", admin["host"], "-p", str(admin["port"]),
           "-U", admin["user"], "-d", admin["dbname"]]
    return subprocess.run(cmd + (["-f", str(path)] if path else []), input=sql, env=env, capture_output=True,
                          text=True, check=True)


def _added_columns_sql() -> str:
    """Every later ADD COLUMN on temporal_self_event, so a new column without a view update fails this test."""
    out = []
    pat = r"(?im)^\s*ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:public\.)?temporal_self_event\s+ADD\b[^;]*;"
    for f in sorted(MIG_DIR.glob("manual_migration_*.sql")):
        out += [m.group(0) for m in re.finditer(pat, f.read_text())]
    return "\n".join(out)


def _stored(e) -> tuple[str, str, dict]:
    """(event_id, source_kind, payload_json) as orion-durable-runs stores it (temporal_self_store.stored_payload)."""
    return e.event_id, e.source_kind, dict(e.payload, ended_at=e.ended_at.isoformat() if e.ended_at else None,
                                           privacy_class=e.privacy_class)


def _producer_rows() -> list[tuple[str, str, dict]]:
    """One event per arm, built by the real producers, so a renamed or added arm key reaches this test."""
    out = []
    for arm in (*OFFER_ARMS, ARM_SELF_INQUIRY):
        out.append(_stored(S.curiosity_run({"run_id": f"cur-{arm}", "turn_started_at": AT, "decided_at": AT,
                                            "arm": arm, "offered": [], "completed_at": AT, "turn_ok": True,
                                            "n_tested": 2, "n_moved": 1, "n_formed": 0})))
    for arm in DREAM_ARMS:
        out.append(_stored(S.dream_hypothesis({"hypothesis_id": f"hyp-{arm}", "cycle_id": "c1", "arm": arm,
                                               "created_at": AT, "expires_at": None, "offered_at": None,
                                               "offered_run_id": None})))
    return out


def _eid(kind: str, ref: str) -> str:
    return f"{kind}:{ref}"


@pytest.fixture()
def db(server):
    """(admin, readonly) connection dicts for a fresh database in the leaked state the live role had on 2026-11-18."""
    name = f"blind_arm_{uuid.uuid4().hex[:10]}"
    _exec(server, f"CREATE DATABASE {name}")
    admin = {**server, "dbname": name}
    _exec(admin, f"""DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orion_readonly') THEN
          CREATE ROLE orion_readonly LOGIN PASSWORD '{RO_PASSWORD}';
        ELSE ALTER ROLE orion_readonly LOGIN PASSWORD '{RO_PASSWORD}'; END IF; END $$""")
    _exec(admin, "ALTER ROLE orion_readonly RESET search_path")
    _psql_file(admin, MIG_DIR / "manual_migration_temporal_self_event_v1.sql")
    _psql_file(admin, sql=_added_columns_sql())
    _exec(admin, """
        CREATE TABLE curiosity_offer_decisions (run_id text PRIMARY KEY, arm text NOT NULL);
        CREATE TABLE dream_hypothesis (hypothesis_id text PRIMARY KEY, arm text NOT NULL, ref_a text, ref_b text);
        CREATE TABLE dream_cycle (cycle_id text PRIMARY KEY, cycle_json jsonb);
        GRANT SELECT ON temporal_self_event, curiosity_offer_decisions, dream_hypothesis, dream_cycle TO orion_readonly;
    """)
    rows = _producer_rows()
    rows.append(("chat-1", "chat_turn", {"speaker": "juniper"}))
    for event_id, kind, payload in rows:
        _exec(admin, "INSERT INTO temporal_self_event (event_id, day_id, occurred_at, source_kind, label, payload_json)"
                     " VALUES (%s, '2026-11-18', now(), %s, 'x', %s::jsonb)", (event_id, kind, _json(payload, default=str)))
    _exec(admin, "INSERT INTO curiosity_offer_decisions VALUES ('cur-a', %s)", (OFFER_ARMS[0],))
    _exec(admin, "INSERT INTO dream_hypothesis VALUES ('hyp-a', %s, 'r1', 'r2')", (DREAM_ARMS[0],))
    readonly = {**admin, "user": "orion_readonly", "password": RO_PASSWORD}
    try:
        yield admin, readonly
    finally:
        _exec(server, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def _json(obj, **kw) -> str:
    import json

    return json.dumps(obj, sort_keys=True, **kw)


def _payloads(readonly: dict) -> dict[str, dict]:
    return {r[0]: r[1] for r in _exec(readonly, "SELECT event_id, payload_json FROM temporal_self_event")}


def _refused(readonly: dict, sql: str) -> bool:
    try:
        _exec(readonly, sql)
    except psycopg2.errors.InsufficientPrivilege:
        return True
    return False


def test_before_the_guard_the_role_reads_the_arm(db):
    """The harness can see the leak, so the next test's silence means the guard closed it."""
    _, readonly = db
    assert _payloads(readonly)[_eid("curiosity_run", f"cur-{OFFER_ARMS[1]}")]["arm"] == OFFER_ARMS[1]
    assert _exec(readonly, "SELECT arm FROM curiosity_offer_decisions")


def test_after_the_guard_no_sql_as_orion_readonly_reads_a_blinded_arm(db):
    admin, readonly = db
    _exec(admin, "ANALYZE public.temporal_self_event")
    assert _exec(readonly, "SELECT attname FROM pg_stats WHERE tablename = 'temporal_self_event' "
                           "AND attname = 'payload_json'")  # before: the planner's sample values are readable too
    _psql_file(admin, GUARD)
    _psql_file(admin, GUARD)  # idempotent: a second apply is a no-op, not an error

    payloads = _payloads(readonly)
    cur = {a: payloads[_eid("curiosity_run", f"cur-{a}")] for a in OFFER_ARMS}
    hyp = {a: payloads[_eid("dream_hypothesis", f"hyp-{a}")] for a in DREAM_ARMS}
    # The arms are indistinguishable to Orion; the rest of the payload is untouched.
    assert len({_json(p) for p in cur.values()}) == 1 and len({_json(p) for p in hyp.values()}) == 1
    assert "n_moved" in cur[OFFER_ARMS[0]] and "offered_at" in hyp[DREAM_ARMS[0]]
    # self_inquiry is a line, not an offer arm, and the now already names it.
    assert payloads[_eid("curiosity_run", f"cur-{ARM_SELF_INQUIRY}")]["arm"] == ARM_SELF_INQUIRY
    assert payloads["chat-1"] == {"speaker": "juniper"}

    # No leaf value anywhere in these payloads is an arm name, whatever key a producer puts it under.
    leaves = _exec(readonly, "SELECT v #>> '{}' FROM temporal_self_event, jsonb_path_query(payload_json, "
                             "'strict $.**') v WHERE source_kind IN ('curiosity_run', 'dream_hypothesis') "
                             "AND event_id <> %s AND jsonb_typeof(v) = 'string'",
                   (_eid("curiosity_run", f"cur-{ARM_SELF_INQUIRY}"),))
    assert not {r[0] for r in leaves} & set(OFFER_ARMS + DREAM_ARMS)
    # The planner's statistics follow the column privilege: none for a table the role cannot read.
    assert _exec(readonly, "SELECT attname FROM pg_stats WHERE tablename = 'temporal_self_event'") == []

    for sql in ("SELECT payload_json FROM public.temporal_self_event",
                "SELECT arm FROM curiosity_offer_decisions", "SELECT arm FROM public.curiosity_offer_decisions",
                "SELECT arm FROM dream_hypothesis", "SELECT cycle_json FROM dream_cycle"):
        assert _refused(readonly, sql), sql


def test_the_view_has_every_column_of_the_table(db):
    """A new temporal_self_event column needs the view (and the guard file) updated; this fails until it is.
    Compared as sets: the scratch table gets its ADD COLUMNs in file-name order, not the live apply order."""
    admin, readonly = db
    _psql_file(admin, GUARD)
    cols = "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = " \
           "'temporal_self_event' ORDER BY ordinal_position"
    assert sorted(_exec(admin, cols, ("orion_scope",))) == sorted(_exec(admin, cols, ("public",)))
    assert _exec(readonly, "SELECT count(*) FROM temporal_self_event") == _exec(admin, "SELECT count(*) FROM "
                                                                                       "public.temporal_self_event")


def _lift_block(heading: str) -> str:
    """The SQL of one LIFT block in the guard file's comments (indented lines after ``heading``)."""
    lines = GUARD.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if heading in ln) + 1
    body = []
    for ln in lines[start:]:
        if not ln.startswith("--     "):
            if body:
                break
            continue
        body.append(ln[len("--     "):])
    return "\n".join(body)


def test_the_documented_lift_steps_work(db):
    admin, readonly = db
    _psql_file(admin, GUARD)

    _psql_file(admin, sql=_lift_block("Curiosity stopped, dream still blind"))
    payloads = _payloads(readonly)
    assert payloads[_eid("curiosity_run", f"cur-{OFFER_ARMS[0]}")]["arm"] == OFFER_ARMS[0]
    assert all("arm" not in payloads[_eid("dream_hypothesis", f"hyp-{a}")] for a in DREAM_ARMS)

    _psql_file(admin, sql=_lift_block("Both stopped, full undo"))
    assert _exec(readonly, "SELECT payload_json->>'arm' FROM temporal_self_event WHERE event_id = %s",
                 (_eid("dream_hypothesis", f"hyp-{DREAM_ARMS[0]}"),)) == [(DREAM_ARMS[0],)]
    assert not _refused(readonly, "SELECT 1 FROM public.temporal_self_event")
