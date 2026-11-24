"""scripts/memory_own_thought_settle.py (issue #3840): the one-time own-thought pass.

Pure helpers always run; the end-to-end dry-run / apply / rollback runs against a disposable Postgres
(ORION_MEMORY_EPISODE_TEST_DATABASE_URL), same harness as test_confirmation_pg. Synthetic rows only.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("memory_own_thought_settle",
                                               ROOT / "scripts" / "memory_own_thought_settle.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def test_csv_and_report_are_private_and_carry_no_memory_text(tmp_path):
    out = S.ensure_out_dir(tmp_path / "run")
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    snap = [{"memory_id": "m1", "voice": "orion_thought", "channel": "chat", "stakes": "high", "status": "active",
             "confirmation_state": "pending_confirmation", "confirmation_loop_id": "memory-confirm-m1",
             "updated_at": None, "cards": "a1:open", "answer_waiting": False,
             "statement": "SECRET TEXT"}]
    S.write_csv(out / "snapshot.csv", S.SNAPSHOT_FIELDS, snap)
    assert stat.S_IMODE((out / "snapshot.csv").stat().st_mode) == 0o600
    assert "SECRET" not in (out / "snapshot.csv").read_text()
    assert "statement" not in S.SNAPSHOT_FIELDS and "statement" not in S.TARGET_SQL
    rows = S.before_after_rows(snap, {"m1": {"confirmation_state": "own_thought", "cards": "a1:withdrawn"}})
    assert rows == [{"memory_id": "m1", "channel": "chat", "state_before": "pending_confirmation",
                     "state_after": "own_thought", "cards_before": "a1:open", "cards_after": "a1:withdrawn"}]
    report = S.render_report(mode="dry-run", snapshot=snap, moved=0, skipped_waiting=0, cards_withdrawn=1,
                             before_counts=[], after_counts=[], pending_before=[], pending_after=[], anomalies=[])
    assert "Would move to own_thought: 1" in report and "SECRET" not in report


def test_script_runs_the_tick_s_own_statement():
    from orion.memory.episode import confirmation as c

    assert S.SETTLE_OWN_THOUGHTS_SQL is c.SETTLE_OWN_THOUGHTS_SQL


ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")


@pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")
def test_dry_run_apply_rollback_end_to_end(tmp_path):
    from orion.memory.episode import confirmation as c
    from orion.memory.episode.tests import test_confirmation_pg as P

    async def body(pool):
        stuck = await P._memory(pool, voice="orion_thought")
        backfilled = await P._memory(pool, voice="orion_thought", state="not_asked",
                                     created=P.T0 + timedelta(minutes=1))
        hers = await P._memory(pool, voice="juniper_said", created=P.T0 + timedelta(minutes=2))
        loop = c.loop_id_for(stuck)
        far = P.T0 + timedelta(days=3650)  # the card is still live when rollback reopens it
        await pool.execute(
            "INSERT INTO orion_ask (ask_id, question, status, created_at, expires_at, source_kind, source_ref) "
            "VALUES ($1, 'q', 'open', $2, $3, $4, $5)", c.ask_id_for(loop), P.T0, far, c.SOURCE_KIND, loop)
        await pool.execute("UPDATE episode_memory SET confirmation_loop_id = $2 WHERE memory_id = $1",
                           uuid.UUID(stuck), loop)
        name = await pool.fetchval("SELECT current_database()")
        dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
        out = tmp_path / "own"

        async def states():
            return {str(r["memory_id"]): r["confirmation_state"] for r in await pool.fetch(
                "SELECT memory_id, confirmation_state FROM episode_memory")}

        before = await states()
        assert S.main(["--dsn", dsn, "--out", str(out)]) == 0                       # dry run: no writes
        assert await states() == before and (out / "snapshot_dry_run.csv").exists()
        assert "Would move to own_thought: 2" in (out / "report.md").read_text()

        assert S.main(["--dsn", dsn, "--out", str(out), "--apply"]) == 0
        after = await states()
        assert (after[stuck], after[backfilled], after[hers]) == ("own_thought", "own_thought", "pending_confirmation")
        assert await pool.fetchval("SELECT status FROM orion_ask WHERE ask_id = $1", c.ask_id_for(loop)) == "withdrawn"
        ba = S.read_csv(out / "before_after.csv")
        assert {(r["state_before"], r["state_after"]) for r in ba} == {
            ("pending_confirmation", "own_thought"), ("not_asked", "own_thought")}
        with pytest.raises(SystemExit):                                              # never clobber the snapshot
            S.main(["--dsn", dsn, "--out", str(out), "--apply"])

        assert S.main(["--dsn", dsn, "--out", str(out), "--rollback"]) == 0
        assert await states() == before
        assert await pool.fetchval("SELECT status FROM orion_ask WHERE ask_id = $1", c.ask_id_for(loop)) == "open"
        log = (out / "progress.log").read_text()
        assert "rollback (restored 2, cards reopened 1)" in log

    P.run(body)
