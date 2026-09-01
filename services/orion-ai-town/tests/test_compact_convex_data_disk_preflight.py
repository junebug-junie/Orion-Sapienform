"""Behavioural tests for compact_convex_data.sh's step 0 disk preflight.

Regression for a live incident (2026-08-31): the script wrote an 11GB
db.sqlite3 backup into /tmp on a filesystem without 11GB free, filled the root
disk, and died at step 5b with `npm error nospc` -- but only AFTER step 4 had
renamed the live database aside and step 5 had started the backend on a fresh
empty one. AI Town served an empty world for 33 hours while its container
reported "healthy". Nothing in the script looked at df.

These tests actually RUN the script rather than grepping it for the word "df".
A structural assertion cannot tell the difference between a preflight that
aborts and one that computes a number and ignores it -- and the whole failure
being guarded against is a check that exists but does not stop anything.

Docker/Convex are stubbed: the script is copied into a temp tree so its
BASH_SOURCE-derived ROOT points at a fake service layout, and a fake `docker`
on PATH reports whatever database size the test wants.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

_SERVICE = Path(__file__).resolve().parents[1]
_SCRIPT = _SERVICE / "scripts" / "compact_convex_data.sh"

_FAKE_DOCKER = """#!/usr/bin/env bash
args="$*"
case "$args" in
  *"exec -T backend stat"*) echo "${FAKE_DB_SIZE:-1024}" ;;
  *"ps -a -q backend"*)     echo "fakecid" ;;
  *"ps -q backend"*)        echo "fakecid" ;;
  *inspect*)                echo "fakevol" ;;
  *)                        exit 0 ;;
esac
"""

# With FAKE_NPX_OK unset this fails, so the script stops right after the
# preflight lets it through -- enough to observe that it got past step 0. With
# FAKE_NPX_OK=1 it fakes a successful export so a test can drive the script all
# the way past step 4, into the state the 2026-08-31 outage left behind.
_FAKE_NPX = """#!/usr/bin/env bash
if [ "${FAKE_NPX_OK:-0}" != "1" ]; then
  echo "fake npx: $*" >&2
  exit 1
fi
case "$*" in
  *"convex export"*)
    out=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--path" ]; then out="$a"; fi
      prev="$a"
    done
    head -c 2048 /dev/zero > "$out"
    ;;
  *"convex env list"*) echo "LLM_MODEL=quick_background" ;;
  *) : ;;
esac
exit 0
"""

# Always fails, so step 5's health loop times out deterministically instead of
# depending on whether anything happens to be listening on :3210.
_FAKE_CURL = """#!/usr/bin/env bash
[ "${FAKE_CURL_OK:-0}" = "1" ] && exit 0
exit 1
"""

# Lets a test drive the preflight arithmetic to any scenario. Without it the
# suite can only ever see this host's real free space, which is why the
# shared-filesystem bug below was invisible to the first version of these tests.
# Mirrors real df by failing on a path that does not exist -- otherwise the
# "unreadable $HOME" branch is unreachable and its test passes vacuously.
_FAKE_DF = """#!/usr/bin/env bash
target="${!#}"
[ -e "$target" ] || exit 1
echo "Filesystem 1B-blocks Used Available Capacity Mounted"
echo "fake 1000000000000 0 ${FAKE_FREE_BYTES:-999000000000} 1% /"
"""

# The shared-vs-separate filesystem branch keys off `stat -c%d`. On this host
# (and on circe) /tmp, $HOME and / are genuinely one device, so without a stub
# the separate-filesystem branch cannot be exercised at all. Only -c%d is faked;
# everything else defers to the real stat, which the script uses for file sizes.
_FAKE_STAT = """#!/usr/bin/env bash
if [ "$1" = "-c%d" ]; then
  [ -e "$2" ] || exit 1
  case "$2" in
    */jobs/*) echo "${FAKE_JOB_DEV:-1}" ;;
    *)        echo "${FAKE_HOME_DEV:-1}" ;;
  esac
  exit 0
fi
exec /usr/bin/stat "$@"
"""


def _stage(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build a fake service tree + stub PATH; return (script, env)."""
    root = tmp_path / "service"
    (root / "scripts").mkdir(parents=True)
    (root / "upstream" / "convex").mkdir(parents=True)

    script = root / "scripts" / "compact_convex_data.sh"
    shutil.copy2(_SCRIPT, script)

    binp = tmp_path / "bin"
    binp.mkdir()
    for name, body in (("docker", _FAKE_DOCKER), ("npx", _FAKE_NPX),
                       ("curl", _FAKE_CURL), ("df", _FAKE_DF),
                       ("stat", _FAKE_STAT)):
        p = binp / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)

    jobs = tmp_path / "jobs"
    jobs.mkdir()

    env = dict(os.environ)
    env["PATH"] = f"{binp}:{env['PATH']}"
    env["AITOWN_COMPACT_JOB_DIR_BASE"] = str(jobs)
    return script, env


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), "--force"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_preflight_aborts_when_job_dir_cannot_hold_the_backup(tmp_path):
    """A database larger than the disk must stop the run before step 1."""
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = str(10**15)  # 1 PB -- exceeds any real filesystem
    env["FAKE_JOB_DEV"] = "1"
    env["FAKE_HOME_DEV"] = "2"  # separate filesystems -> job-dir-specific message

    r = _run(script, env)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "insufficient space for the step 3 backup" in out, out
    # the critical property: it stopped BEFORE anything destructive or even
    # before the export, not partway through
    assert "step 1/7" not in out, out


def test_preflight_allows_a_run_that_fits(tmp_path):
    """Positive control: the gate must not block a database that fits.

    Without this, a preflight that always aborted would pass the test above.
    """
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"

    r = _run(script, env)

    out = r.stdout + r.stderr
    assert "step 0/7: disk preflight" in out, out
    assert "insufficient space" not in out, out
    assert "step 1/7" in out, out  # got past the gate


def test_preflight_runs_before_the_export(tmp_path):
    """Ordering is the point: steps 1-3 are recoverable, past step 4 is not."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert text.index("step 0/7: disk preflight") < text.index("npx convex export")


def test_job_dir_base_is_overridable(tmp_path):
    """The incident's root cause was /tmp sharing a disk with the OS."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "AITOWN_COMPACT_JOB_DIR_BASE" in text
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"
    r = _run(script, env)
    assert str(tmp_path / "jobs") in (r.stdout + r.stderr)


def test_failure_past_step_4_points_at_the_resume_script(tmp_path):
    """The 2026-08-31 failure had no recovery hint, and re-running looked safe.

    Drives the script past step 4 (live DB renamed aside, backend restarted on
    an empty one) and then fails the health wait, which is the same shape as the
    real incident. Any failure from that point must say "do not re-run" and name
    the resume script.
    """
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"
    env["FAKE_NPX_OK"] = "1"
    env["AITOWN_COMPACT_HEALTH_TIMEOUT_SEC"] = "1"

    r = _run(script, env)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "renamed to db.sqlite3.pre-compact-" in out, out  # got past step 4
    assert "Do NOT re-run this script" in out, out
    assert "resume_compact_convex_data.sh" in out, out


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_past_step_4_still_warns_and_does_not_exit_zero(tmp_path, sig):
    """Ctrl-C on a hung step is the most likely operator action, and it was silent.

    `$?` inside an EXIT trap is the status of the last COMPLETED command, not the
    signal. Verified on this host: with `trap on_exit EXIT` alone, SIGINT/TERM/HUP
    all entered the trap with rc=0, so a status-gated warning never printed -- and
    SIGINT additionally made the script exit 0, so cron would log the run as a
    success while the town sat empty.
    """
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"
    env["FAKE_NPX_OK"] = "1"
    env["AITOWN_COMPACT_HEALTH_TIMEOUT_SEC"] = "60"  # park in the health loop

    p = subprocess.Popen(["bash", str(script), "--force"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    try:
        jobs = tmp_path / "jobs"
        deadline = time.time() + 30
        while time.time() < deadline:
            logs = list(jobs.glob("*/progress.log"))
            if logs and "step 5/7" in logs[0].read_text(encoding="utf-8"):
                break
            time.sleep(0.2)
        else:
            p.kill()
            pytest.fail("script never reached step 5")

        p.send_signal(sig)
        out, _ = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            p.kill()

    assert p.returncode != 0, f"signal {sig} produced exit 0\n{out}"
    assert "Do NOT re-run this script" in out, out
    assert "resume_compact_convex_data.sh" in out, out


def test_completed_run_does_not_warn(tmp_path):
    """Control: the warning gates on COMPLETED, so a clean run must stay quiet."""
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"
    env["FAKE_NPX_OK"] = "1"
    env["FAKE_CURL_OK"] = "1"

    r = _run(script, env)
    out = r.stdout + r.stderr
    if "report written" in out:  # only meaningful if the run actually completed
        assert "Do NOT re-run this script" not in out, out


def test_no_resume_hint_when_the_run_fails_before_step_4(tmp_path):
    """Control: before the point of no return, a re-run IS the right move.

    Without this, a hint printed unconditionally would pass the test above.
    """
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"  # FAKE_NPX_OK unset -> step 1 export fails

    r = _run(script, env)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "renamed to db.sqlite3.pre-compact-" not in out, out
    assert "Do NOT re-run this script" not in out, out


def test_shared_filesystem_requirements_are_summed(tmp_path):
    """THE regression. Checking the two demands separately does not close it.

    On the incident host -- and, verified 2026-09-01, on circe today -- /tmp,
    $HOME and / are one filesystem. Free space that satisfies the step 3 backup
    AND separately satisfies the step 5b cache can still be too small for both,
    because they draw on the same pool. The first version of this patch checked
    them independently and would have let the identical failure through with the
    preflight logging green.

    Free space here is set to BACKUP_NEED + 1GiB: enough for the backup alone,
    and enough for the 2GiB cache alone, but not for the two together.
    """
    db = 10 * 1024**3
    backup_need = db + db // 20 + 536870912
    home_need = 2 * 1024**3

    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = str(db)
    env["FAKE_FREE_BYTES"] = str(backup_need + 1024**3)
    env["HOME"] = str(tmp_path)  # same filesystem as the job dir

    assert backup_need + 1024**3 >= backup_need          # backup alone: fits
    assert backup_need + 1024**3 >= home_need            # cache alone: fits
    assert backup_need + 1024**3 < backup_need + home_need  # together: does not

    r = _run(script, env)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "requirements summed" in out, out
    assert "insufficient space for the compaction" in out, out
    assert "step 1/7" not in out, out


def test_shared_filesystem_with_room_for_both_proceeds(tmp_path):
    """Positive control for the summing branch."""
    db = 10 * 1024**3
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = str(db)
    env["FAKE_FREE_BYTES"] = str(db * 3)
    env["HOME"] = str(tmp_path)

    r = _run(script, env)
    out = r.stdout + r.stderr
    assert "step 1/7" in out, out


def test_home_check_fails_closed_when_unreadable(tmp_path):
    """`npm error nospc` under $HOME is what actually killed the live run.

    This check guards a step that runs AFTER the database has been reset, so
    "could not measure" must not read as "fine".
    """
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"
    env["HOME"] = str(tmp_path / "definitely-not-here")
    env["FAKE_HOME_DEV"] = "2"

    r = _run(script, env)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "could not read free space under" in out, out
    assert "step 1/7" not in out, out


def test_unreadable_home_can_be_overridden(tmp_path):
    """Fail closed, but with a documented door."""
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = "1024"
    env["HOME"] = str(tmp_path / "definitely-not-here")
    env["FAKE_HOME_DEV"] = "2"
    env["AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE"] = "1"

    r = _run(script, env)
    out = r.stdout + r.stderr
    assert "proceeding on explicit override" in out, out
    assert "step 1/7" in out, out


def test_skip_raw_backup_relaxes_the_gate_and_skips_step_3(tmp_path):
    """Without a door, the gate blocks the only thing that shrinks the DB.

    Once the database passes ~48% of its filesystem a full-copy-sized gate
    refuses every run -- including the one that would fix it.
    """
    db = 10 * 1024**3
    script, env = _stage(tmp_path)
    env["FAKE_DB_SIZE"] = str(db)
    env["FAKE_FREE_BYTES"] = str(4 * 1024**3)  # far too small for a full copy
    env["HOME"] = str(tmp_path)

    blocked = _run(script, env)
    assert blocked.returncode != 0
    assert "insufficient space" in (blocked.stdout + blocked.stderr)

    env["AITOWN_COMPACT_SKIP_RAW_BACKUP"] = "1"
    env["FAKE_NPX_OK"] = "1"
    env["AITOWN_COMPACT_HEALTH_TIMEOUT_SEC"] = "1"
    r = _run(script, env)

    out = r.stdout + r.stderr
    assert "step 1/7" in out, out
    assert "step 3/7: SKIPPED" in out, out
