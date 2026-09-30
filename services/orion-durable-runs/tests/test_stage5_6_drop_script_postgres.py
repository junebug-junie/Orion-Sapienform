"""scripts/gpu_pool_stage5_snapshot_and_drop.sh against a real, disposable Postgres.

GPU pool stage 5.6 ships (does not run) the drop of the four dead legacy tables. This drives the real
script end to end through a fake ``docker`` on PATH that forwards ``docker exec <c> psql|pg_dump`` to
the local client binaries on a fresh database per test. Needs ORION_ADMISSION_TEST_DSN (like every
Postgres test here) and a pg_dump at least as new as the server; skipped otherwise.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gpu_pool_stage5_snapshot_and_drop.sh"
SQL = REPO / "services" / "orion-sql-db"
DSN = os.environ.get("ORION_ADMISSION_TEST_DSN")
LEGACY = ("durable_gateway_permits", "durable_resource_leases", "durable_resource_demands", "durable_elastic_slot")
CUTOFF = "2026-09-29 22:45:00+00"

pytestmark = pytest.mark.skipif(not DSN or not shutil.which("psql"), reason="needs ORION_ADMISSION_TEST_DSN and psql")


def _bin(name: str) -> str:
    """The newest installed client (a pg_dump older than the server refuses to run)."""
    found = sorted(glob.glob(f"/usr/lib/postgresql/*/bin/{name}"), key=lambda p: int(p.split("/")[4]))
    return found[-1] if found else (shutil.which(name) or name)


def _psql(dsn: str, sql: str) -> str:
    return subprocess.run([_bin("psql"), dsn, "-v", "ON_ERROR_STOP=1", "-X", "-At", "-c", sql],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def db(tmp_path):
    name = f"stage5_drop_{uuid.uuid4().hex[:8]}"
    _psql(DSN, f"CREATE DATABASE {name}")
    dsn = re.sub(r"/[^/?]+(\?|$)", f"/{name}\\1", DSN, count=1)
    probe = subprocess.run([_bin("pg_dump"), "--schema-only", dsn], capture_output=True, text=True)
    if probe.returncode != 0:
        _psql(DSN, f"DROP DATABASE {name}")
        pytest.skip(f"pg_dump unusable against this server: {probe.stderr.strip()[:200]}")
    for f in ("manual_migration_durable_resource_admission_v1.sql", "manual_migration_gateway_capacity_v1.sql",
              "manual_migration_gpu2_elastic_v1.sql"):
        subprocess.run([_bin("psql"), dsn, "-v", "ON_ERROR_STOP=1", "-q", "-f", str(SQL / f)], check=True,
                       capture_output=True)
    _psql(dsn, """
      INSERT INTO durable_admission_runs(run_id, request) VALUES ('r1', '{}'), ('r2', '{}');
      INSERT INTO durable_resource_events(entry_id, run_id, event, payload) VALUES ('e1', 'r1', 'run.started', '{}');
      INSERT INTO durable_resource_demands(demand_id, run_id, requirement, status, created_at)
        VALUES ('d1', 'r1', '{}', 'granted', '2026-09-25 00:00+00');
      INSERT INTO durable_resource_leases(lease_id, demand_id, run_id, resource_key, lane, backend_key, granted_at,
          expires_at, heartbeat_at, status)
        VALUES ('l1', 'd1', 'r1', 'k', 'agent', 'b', '2026-09-25 00:00+00', '2026-09-25 00:05+00',
                '2026-09-25 00:01+00', 'released');
      INSERT INTO durable_gateway_permits(request_id, permit_id, request, correlation_id, lane, backend_key,
          max_inflight, deadline_at, granted_at, heartbeat_at, expires_at, status)
        SELECT 'q'||g, 'p'||g, jsonb_build_object('note', E'multi\\nline\\ttab'), 'c', 'diffusion', 'b', 1,
               '2026-09-29 21:00+00', '2026-09-29 21:00+00', '2026-09-29 21:00+00', '2026-09-29 21:01+00', 'released'
        FROM generate_series(1, 25) g;
    """)
    shim = tmp_path / "bin"
    shim.mkdir()
    # docker exec [-i] <container> psql|pg_dump -U <u> -d <db> ARGS -> <client> <dsn> ARGS
    (shim / "docker").write_text(f"""#!/usr/bin/env bash
[[ "$1" == exec ]] || {{ echo "shim: only exec" >&2; exit 97; }}
shift; [[ "$1" == -i ]] && shift; shift
cmd="$1"; shift
args=()
while [[ $# -gt 0 ]]; do case "$1" in -U|-d) shift 2 ;; *) args+=("$1"); shift ;; esac; done
case "$cmd" in
  psql) exec {_bin('psql')} "{dsn}" "${{args[@]}}" ;;
  pg_dump) "{_bin('pg_dump')}" "{dsn}" "${{args[@]}}"; rc=$?
           [[ -n "${{SHIM_AFTER_DUMP_SQL:-}}" ]] && {_bin('psql')} "{dsn}" -q -c "$SHIM_AFTER_DUMP_SQL" >/dev/null
           exit $rc ;;
  *) echo "shim: $cmd not allowed" >&2; exit 98 ;;
esac
""")
    (shim / "docker").chmod(0o755)
    yield {"dsn": dsn, "path": f"{shim}:{os.environ['PATH']}", "out": tmp_path / "out"}
    _psql(DSN, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def run(db, *args, **env):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PATH": db["path"], "OUT": str(db["out"]), "SQL_CONTAINER": "fake",
                               **env})


def legacy_present(db) -> list[str]:
    return [t for t in LEGACY if _psql(db["dsn"], f"SELECT to_regclass('public.{t}') IS NOT NULL") == "t"]


def test_snapshot_only_verifies_the_dump_and_drops_nothing(db):
    res = run(db, "--cutoff", CUTOFF)
    assert res.returncode == 0, res.stderr
    verify = (db["out"] / "dump_verify.csv").read_text().splitlines()
    assert verify == ["durable_gateway_permits,25,25", "durable_resource_leases,1,1",
                      "durable_resource_demands,1,1", "durable_elastic_slot,0,0"]
    assert legacy_present(db) == list(LEGACY)


def test_drop_removes_the_four_tables_keeps_the_registry_and_reruns_clean(db):
    res = run(db, "--cutoff", CUTOFF, "--drop")
    assert res.returncode == 0, res.stderr
    assert legacy_present(db) == []
    assert _psql(db["dsn"], "SELECT to_regclass('public.durable_resource_fencing_generation') IS NULL") == "t"
    assert _psql(db["dsn"], "SELECT (SELECT count(*) FROM durable_admission_runs)||','||"
                            "(SELECT count(*) FROM durable_resource_events)") == "2,1"
    assert "durable_gateway_permits,25,dropped" in (db["out"] / "before_after.csv").read_text()
    again = run(db, "--cutoff", CUTOFF, "--drop")
    assert again.returncode == 0 and "already_dropped" in (db["out"] / "progress.log").read_text()


def test_write_after_the_cutoff_refuses_before_any_dump(db):
    res = run(db, "--cutoff", "2026-09-29 20:00:00+00", "--drop")
    assert res.returncode == 2 and "after the cutoff" in res.stderr
    assert not (db["out"] / "legacy_tables.sql.gz").exists() and legacy_present(db) == list(LEGACY)


def test_an_active_permit_refuses(db):
    _psql(db["dsn"], "UPDATE durable_gateway_permits SET status='active' WHERE request_id='q1'")
    res = run(db, "--cutoff", CUTOFF, "--drop")
    assert res.returncode == 2 and "still active/pending" in res.stderr and legacy_present(db) == list(LEGACY)


def test_a_row_written_between_dump_and_drop_rolls_the_drop_back(db):
    late = ("INSERT INTO durable_resource_demands(demand_id, run_id, requirement, status, created_at) "
            "VALUES ('d9', 'r2', '{}', 'withdrawn', '2026-09-25 03:00+00')")
    res = run(db, "--cutoff", CUTOFF, "--drop", SHIM_AFTER_DUMP_SQL=late)
    assert res.returncode == 2 and "rolled back" in res.stderr
    assert legacy_present(db) == list(LEGACY)


def test_over_the_row_line_needs_the_explicit_flag(db):
    res = run(db, "--cutoff", CUTOFF, "--drop", ROW_LIMIT_OVERRIDE="10")
    assert res.returncode == 2 and "--accept-large-snapshot" in res.stderr and legacy_present(db) == list(LEGACY)
    ok = run(db, "--cutoff", CUTOFF, "--accept-large-snapshot", ROW_LIMIT_OVERRIDE="10")
    assert ok.returncode == 0, ok.stderr


@pytest.mark.parametrize("cutoff", ["2026-09-29'::timestamptz; SELECT 1; --", "2026-09-29 22:45:00",
                                    "yesterday", "2026-09-29 22:45:00+00; DROP TABLE durable_admission_runs"])
def test_a_cutoff_that_is_not_a_plain_zoned_timestamp_is_rejected_before_any_sql(db, cutoff):
    res = run(db, "--cutoff", cutoff, "--drop")
    assert res.returncode == 64
    assert not (db["out"] / "progress.log").exists() and legacy_present(db) == list(LEGACY)
