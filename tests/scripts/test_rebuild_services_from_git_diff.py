"""Behavioral coverage for service-specific post-merge rebuild commands."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rebuild_services_from_git_diff.sh"

# Fake classifier: answers --plan from $FAKE_PLAN and
# --running-profile-services from $FAKE_PROFILES (both "\t"-separated lines).
_FAKE_CLASSIFIER = """\
import os, sys
args = sys.argv[1:]
if "--running-profile-services" in args:
    svc = args[args.index("--running-profile-services") + 1]
    skip = {args[i + 1] for i, a in enumerate(args) if a == "--skip-profile"}
    for line in os.environ.get("FAKE_PROFILES", "").split(";"):
        if not line:
            continue
        owner, profile, name = line.split(",")
        if owner == svc and profile not in skip:
            print(profile + "\\t" + name)
    raise SystemExit(0)
if "--pool-held-services" in args:
    svc = args[args.index("--pool-held-services") + 1]
    # FAKE_POOL answers the first call, FAKE_POOL_AFTER (when set) every later one:
    # the script re-reads ownership after the build.
    marker = os.environ["FAKE_POOL_CALLS"]
    later = os.path.exists(marker)
    open(marker, "a").close()
    if os.environ.get("FAKE_POOL_RC"):
        raise SystemExit(int(os.environ["FAKE_POOL_RC"]))
    plan = os.environ.get("FAKE_POOL_AFTER") if later and "FAKE_POOL_AFTER" in os.environ else os.environ.get("FAKE_POOL", "")
    for line in plan.split(";"):
        if not line:
            continue
        owner, kind, name = line.split(",")
        if owner == svc:
            print(kind + "\\t" + name)
    raise SystemExit(0)
if "--migrations" in args:
    for line in os.environ.get("FAKE_MIGRATIONS", "").split(";"):
        if line:
            print(line.replace(",", "\t", 1))
    raise SystemExit(0)
for line in os.environ.get("FAKE_PLAN", "").split(";"):
    if line:
        print(line.replace(",", "\\t"))
"""


def _run(
    tmp_path: Path,
    *,
    plan: str,
    profiles: str = "",
    pool: str = "",
    pool_after: str | None = None,
    pool_rc: int | None = None,
    build_rc: int = 0,
    extra_args: list[str] | None = None,
    safe_build_reads_stdin: bool = False,
    safe_build_rc: int = 0,
    migrations: str = "",
    checker: str | None = None,
    applier: str | None = None,
    safe_build_body: str | None = None,
):
    fake_root = tmp_path / "repo"
    fake_scripts = fake_root / "scripts"
    fake_scripts.mkdir(parents=True)

    script = fake_scripts / SCRIPT.name
    shutil.copy2(SCRIPT, script)
    (fake_scripts / "rebuild_affected_services.py").write_text(_FAKE_CLASSIFIER)
    if checker is not None:
        (fake_scripts / "check_sql_migrations_applied.py").write_text(checker)
    if applier is not None:
        (fake_scripts / "apply_post_pull_migrations.py").write_text(applier)

    capture = tmp_path / "safe-build-args.txt"
    safe_build = fake_scripts / "safe_docker_build.sh"
    safe_build.write_text(
        "#!/bin/sh\n"
        + ("cat >/dev/null\n" if safe_build_reads_stdin else "")
        + "printf '%s\\n' \"$*\" >> \"$REBUILD_TEST_CAPTURE\"\n"
        + 'case "$*" in *" build") exit "${FAKE_BUILD_RC:-0}";; esac\n'
        + (safe_build_body or "")
        + f"exit {safe_build_rc}\n"
    )
    safe_build.chmod(0o755)

    env = os.environ.copy()
    env["REBUILD_TEST_CAPTURE"] = str(capture)
    env["FAKE_PLAN"] = plan
    env["FAKE_PROFILES"] = profiles
    env["FAKE_POOL"] = pool
    env["FAKE_POOL_CALLS"] = str(tmp_path / "pool-calls")
    env.pop("FAKE_POOL_AFTER", None)
    env.pop("FAKE_POOL_RC", None)
    if pool_after is not None:
        env["FAKE_POOL_AFTER"] = pool_after
    if pool_rc is not None:
        env["FAKE_POOL_RC"] = str(pool_rc)
    env["FAKE_BUILD_RC"] = str(build_rc)
    env["FAKE_MIGRATIONS"] = migrations
    # Never append to the operator's real ~/.orion/post_pull_rebuild.log.
    env["ORION_REBUILD_LOG"] = str(tmp_path / "post_pull_rebuild.log")
    env.pop("ORION_POST_MERGE_REBUILD", None)
    completed = subprocess.run(
        [str(script), *(extra_args or [])],
        cwd=fake_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = capture.read_text().splitlines() if capture.exists() else []
    return completed, calls


def test_analytics_rebuild_selects_analytics_compose_profile(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-analytics,build",
        # Running analytics-profile services are already covered by the
        # whole-profile `up`; they must not get a second per-service pass.
        profiles="orion-analytics,analytics,lightdash",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-analytics --profile analytics up -d --build"]


def test_running_profile_service_is_rebuilt_with_its_profile(tmp_path: Path) -> None:
    """orion-vision-retina-walkway (profile walkway) was never rebuilt on pull."""
    completed, calls = _run(
        tmp_path,
        plan="orion-vision-retina,build",
        profiles="orion-vision-retina,walkway,orion-vision-retina-walkway",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [
        "orion-vision-retina up -d --build",
        "orion-vision-retina --profile walkway up -d --build orion-vision-retina-walkway",
    ]


def test_no_running_profile_service_means_plain_up_only(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, plan="orion-vision-retina,build", profiles="")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-vision-retina up -d --build"]


def test_recreate_plan_adds_force_recreate_to_every_pass(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-vision-frame-router,recreate;orion-sql-writer,build",
        profiles="orion-vision-frame-router,extra,router-b",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [
        "orion-vision-frame-router up -d --build --force-recreate",
        "orion-vision-frame-router --profile extra up -d --build --force-recreate router-b",
        "orion-sql-writer up -d --build",
    ]


def test_dry_run_reports_profile_services_without_building(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-vision-retina,recreate",
        profiles="orion-vision-retina,walkway,orion-vision-retina-walkway",
        extra_args=["--dry-run"],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == []
    assert "would rebuild: orion-vision-retina (--force-recreate)" in completed.stdout
    assert "--profile walkway orion-vision-retina-walkway" in completed.stdout


def test_build_step_reading_stdin_cannot_swallow_later_profile_services(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-vision-retina,build",
        profiles="orion-vision-retina,walkway,svc-a;orion-vision-retina,walkway,svc-b",
        safe_build_reads_stdin=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [
        "orion-vision-retina up -d --build",
        "orion-vision-retina --profile walkway up -d --build svc-a",
        "orion-vision-retina --profile walkway up -d --build svc-b",
    ]


# 2026-10-16 04:02 UTC: a circe pull ran `up -d --build` on orion-diffusion-host while the GPU
# pool had it stopped (agent-gpu2 held gpu2). Both ended up running, the lane controller refused
# every load/unload (seat_and_evicted_both_running) and no picture rendered for 15 hours.
def test_pool_evicted_service_is_built_but_never_started(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-diffusion-host,build",
        pool="orion-diffusion-host,hold,diffusion-host",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-diffusion-host build"]
    assert "GPU pool holds stopped: diffusion-host" in completed.stdout


def test_pool_hold_still_brings_up_the_stacks_other_services(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-diffusion-host,recreate",
        pool="orion-diffusion-host,hold,diffusion-host;orion-diffusion-host,start,sidecar",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [
        "orion-diffusion-host build",
        "orion-diffusion-host up -d --no-build --no-deps --force-recreate sidecar",
    ]


def test_running_pool_target_is_recreated_after_the_build(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path, plan="orion-diffusion-host,build", pool="orion-diffusion-host,running,diffusion-host"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [
        "orion-diffusion-host build",
        "orion-diffusion-host up -d --no-build --no-deps diffusion-host",
    ]


def test_target_evicted_during_the_build_is_not_restarted(tmp_path: Path) -> None:
    # Running at the first look; the pool stopped it while the image built.
    completed, calls = _run(
        tmp_path,
        plan="orion-diffusion-host,build",
        pool="orion-diffusion-host,running,diffusion-host",
        pool_after="orion-diffusion-host,hold,diffusion-host",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-diffusion-host build"]


def test_failed_build_starts_nothing_and_reports_failed(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-diffusion-host,build",
        pool="orion-diffusion-host,running,diffusion-host;orion-diffusion-host,start,sidecar",
        build_rc=1,
    )
    assert completed.returncode == 1
    assert calls == ["orion-diffusion-host build"]
    assert "FAILED orion-diffusion-host" in completed.stderr


def test_unreadable_pool_ownership_skips_the_stack(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, plan="orion-diffusion-host,build;orion-sql-writer,build", pool_rc=3)
    assert completed.returncode == 1
    # Every stack is skipped, never plain-up'd: the helper failing is not "the pool owns nothing".
    assert calls == []
    assert "cannot read GPU pool ownership" in completed.stderr


def test_stack_without_pool_targets_keeps_the_plain_up(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, plan="orion-sql-writer,build", pool="")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-sql-writer up -d --build"]


def test_dry_run_reports_pool_hold(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-diffusion-host,build",
        pool="orion-diffusion-host,hold,diffusion-host",
        extra_args=["--dry-run"],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == []
    assert "would hold stopped (GPU pool evicted it): orion-diffusion-host diffusion-host" in completed.stdout


# 2026-10-15: a migration-only pull (PR #2698) recreated orion-athena-sql-db.
# The classifier now rebuilds nothing for it; the script must instead name the
# migration, its live state and the apply command, and still build nothing.
_MIG = "services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql"

_CHECKER_REPORT = """\\
import sys
args = sys.argv[1:]
assert "--post-pull-report" in args, args
for rel in args[args.index("--post-pull-report") + 1:]:
    print("  - " + rel + " (required by: orion-sql-writer, orion-hub)")
    print("      live state: NOT APPLIED -- the services above stay refused by safe_docker_build.sh until it is")
    print("      apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney "
          "-v ON_ERROR_STOP=1 < " + rel)
"""

_CHECKER_CRASH = """\\
raise SystemExit(3)
"""

_ROLLBACK = "services/orion-sql-db/manual_migration_vision_confirm_asks_v1_rollback.sql"


def test_migration_only_pull_builds_nothing_and_names_the_apply_command(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, plan="", migrations=_MIG, checker=_CHECKER_REPORT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == []
    out = completed.stdout
    assert f"  - {_MIG} (required by: orion-sql-writer, orion-hub)" in out
    assert "live state: NOT APPLIED" in out
    assert "apply: docker exec -i orion-athena-sql-db psql" in out
    assert "no affected services to rebuild" in out


def test_failed_checker_reports_unknown_and_never_offers_a_rollback(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-hub,build",
        migrations=f"{_MIG};{_ROLLBACK}",
        checker=_CHECKER_CRASH,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-hub up -d --build"]
    out = completed.stdout
    assert "live state: UNKNOWN" in out
    assert f"orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < {_MIG}" in out
    assert f"< {_ROLLBACK}" not in out
    assert "rollback file: never applied on deploy" in out


def test_no_migrations_prints_no_migration_block(tmp_path: Path) -> None:
    completed, _calls = _run(tmp_path, plan="orion-hub,build")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "hand-applied SQL migration" not in completed.stdout


# 2026-10-15 (Juniper): apply safe migrations inside the pull, BEFORE the rebuilds, so the deploy
# gate does not refuse their consumers. The applier writes to the same capture file as the fake
# safe_docker_build.sh, so the order of the two is visible.
_APPLIER = """\
import os, sys
with open(os.environ["REBUILD_TEST_CAPTURE"], "a") as fh:
    fh.write("APPLY " + " ".join(sys.argv[1:]) + "\\n")
print("post-pull migrations: fake applier ran")
raise SystemExit(1)  # a refusal/failure must never stop the rebuilds
"""


def test_migrations_are_applied_before_any_rebuild(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path, plan="orion-hub,build;orion-sql-writer,build", migrations=_MIG,
        checker=_CHECKER_REPORT, applier=_APPLIER,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [f"APPLY --defer-if-backup {_MIG}", "orion-hub up -d --build",
                     "orion-sql-writer up -d --build"]
    out = completed.stdout
    # applier output, then the read-only report of what is live afterwards
    assert out.index("fake applier ran") < out.index("live state:")


def test_dry_run_passes_dry_run_to_the_applier(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path, plan="orion-hub,build", migrations=_MIG, checker=_CHECKER_REPORT,
        applier=_APPLIER, extra_args=["--dry-run"],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [f"APPLY --dry-run {_MIG}"]


# 2026-10-17 ~03:56 UTC: the nightly pg_dump held locks, the first pass hit lock_timeout, and hub +
# sql-writer stayed refused until the file was applied by hand. Now the first pass defers (exit 3),
# every rebuild runs, then the second pass waits for the backup and applies, and only the consumers
# the deploy gate refused are rebuilt again. The fake gate refuses orion-hub until the fake applier
# has "applied" (marker file).
_DEFERRING_APPLIER = """\
import os, sys
args = sys.argv[1:]
cap = os.environ["REBUILD_TEST_CAPTURE"]
if "--required-by" in args:
    print(os.environ.get("FAKE_REQUIRED_BY", "orion-hub"))
    raise SystemExit(0)
with open(cap, "a") as fh:
    fh.write("APPLY " + " ".join(args) + "\\n")
if "--defer-if-backup" in args:
    print("post-pull migrations: a backup is running; DEFERRED")
    raise SystemExit(3)
rc = int(os.environ.get("FAKE_SECOND_APPLY_RC", "0"))
if rc == 0:
    open(cap + ".applied", "w").close()
raise SystemExit(rc)
"""

_GATE = ('case "$*" in orion-hub*up*) [ -f "$REBUILD_TEST_CAPTURE.applied" ] || '
         '{ echo "refused: migration not applied" >&2; exit 1; } ;; esac\n')


def test_backup_defers_the_apply_until_after_the_other_rebuilds_then_retries_the_refused(
        tmp_path: Path, monkeypatch) -> None:
    completed, calls = _run(
        tmp_path, plan="orion-hub,build;orion-planner,build", migrations=_MIG,
        checker=_CHECKER_REPORT, applier=_DEFERRING_APPLIER, safe_build_body=_GATE,
    )
    assert calls == [
        f"APPLY --defer-if-backup {_MIG}",
        "orion-hub up -d --build",          # refused by the gate: migration deferred
        "orion-planner up -d --build",      # not a consumer: deploys without waiting
        f"APPLY {_MIG}",                    # the wait-and-apply pass, after every rebuild
        "orion-hub up -d --build",          # the refused consumer, again
    ], completed.stdout + completed.stderr
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "rebuilding the services the deploy gate refused: orion-hub" in completed.stdout
    events = [ln.split(" ", 4)[3] for ln in _log_lines(tmp_path)]
    assert "DEFERRED-APPLY" in events and "RETRY-AFTER-MIGRATION" in events
    assert _log_lines(tmp_path)[-1].endswith("DONE rebuilt=2 failed=")


def test_deferred_apply_that_cannot_run_leaves_the_consumer_refused(tmp_path: Path, monkeypatch) -> None:
    """Exit 2 = not allowed here / live state UNKNOWN: nothing was applied, nothing is retried."""
    monkeypatch.setenv("FAKE_SECOND_APPLY_RC", "2")
    completed, calls = _run(
        tmp_path, plan="orion-hub,build", migrations=_MIG, checker=_CHECKER_REPORT,
        applier=_DEFERRING_APPLIER, safe_build_body=_GATE,
    )
    assert calls == [f"APPLY --defer-if-backup {_MIG}", "orion-hub up -d --build", f"APPLY {_MIG}"]
    assert completed.returncode != 0
    assert "deferred migration apply did not finish (exit 2)" in completed.stderr
    assert "git pull will not return until the backup finishes" in completed.stderr


def test_partly_failed_second_pass_still_retries_the_consumers(tmp_path: Path, monkeypatch) -> None:
    """Exit 1: some file applied, another did not. Retry the refused consumers anyway; the deploy
    gate refuses again any whose file is still missing (here the gate marker exists: it landed)."""
    monkeypatch.setenv("FAKE_SECOND_APPLY_RC", "1")
    (tmp_path / "safe-build-args.txt.applied").write_text("")  # hub's own file did land
    completed, calls = _run(
        tmp_path, plan="orion-hub,build", migrations=_MIG, checker=_CHECKER_REPORT,
        applier=_DEFERRING_APPLIER,
        safe_build_body='case "$*" in orion-hub*up*) [ -f "$REBUILD_TEST_CAPTURE.phase2" ] || '
                        '{ touch "$REBUILD_TEST_CAPTURE.phase2"; exit 1; } ;; esac\n',
    )
    assert calls == [f"APPLY --defer-if-backup {_MIG}", "orion-hub up -d --build", f"APPLY {_MIG}",
                     "orion-hub up -d --build"], completed.stdout + completed.stderr
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_only_consumers_of_the_deferred_file_are_retried(tmp_path: Path, monkeypatch) -> None:
    """A service that failed for another reason is not rebuilt a second time."""
    monkeypatch.setenv("FAKE_REQUIRED_BY", "orion-sql-writer")
    completed, calls = _run(
        tmp_path, plan="orion-hub,build", migrations=_MIG, checker=_CHECKER_REPORT,
        applier=_DEFERRING_APPLIER, safe_build_body=_GATE,
    )
    assert calls == [f"APPLY --defer-if-backup {_MIG}", "orion-hub up -d --build", f"APPLY {_MIG}"]


def test_migration_only_pull_still_runs_the_deferred_apply(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, plan="", migrations=_MIG, checker=_CHECKER_REPORT,
                            applier=_DEFERRING_APPLIER)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [f"APPLY --defer-if-backup {_MIG}", f"APPLY {_MIG}"]


def test_dry_run_never_runs_the_deferred_pass(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, plan="orion-hub,build", migrations=_MIG, checker=_CHECKER_REPORT,
                            applier=_DEFERRING_APPLIER, extra_args=["--dry-run"])
    assert calls == [f"APPLY --dry-run {_MIG}"]


def test_no_migrations_never_calls_the_applier(tmp_path: Path) -> None:
    """The classifier lists no migration on a host without orion-sql-db (circe), so the applier
    is never reached there; see test_migrations_are_not_reported_on_a_host_without_the_database."""
    completed, calls = _run(tmp_path, plan="orion-hub,build", applier=_APPLIER)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-hub up -d --build"]


def _log_lines(tmp_path: Path) -> list[str]:
    log = tmp_path / "post_pull_rebuild.log"
    return log.read_text().splitlines() if log.exists() else []


def test_pr2728_retina_app_change_rebuilds_walkway_and_logs_each_step(tmp_path: Path) -> None:
    """2026-10-16, PR #2728: retina app/*.py changed, walkway profile running.

    The pull's rebuild did run both passes; the walkway one was still inside
    the old container's 30 s stop grace when someone looked, and nothing on
    disk said so. Every step must now leave a start and an end line.
    """
    completed, calls = _run(
        tmp_path,
        plan="orion-vision-retina,build",
        profiles="orion-vision-retina,walkway,orion-vision-retina-walkway",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [
        "orion-vision-retina up -d --build",
        "orion-vision-retina --profile walkway up -d --build orion-vision-retina-walkway",
    ]
    lines = _log_lines(tmp_path)
    events = [line.split(" ", 4)[3] for line in lines]
    assert events == ["RUN", "START", "END", "START", "END", "DONE"], lines
    assert "services=orion-vision-retina " in lines[0]
    assert lines[4].split(" ", 5)[4] == "rc=0"
    pids = {line.split(" ", 2)[1] for line in lines}
    assert len(pids) == 1 and next(iter(pids)).startswith("pid="), lines
    assert lines[4].endswith(
        "orion-vision-retina --profile walkway up -d --build orion-vision-retina-walkway"
    )
    assert lines[-1].endswith("DONE rebuilt=1 failed=")
    assert str(tmp_path / "post_pull_rebuild.log") in completed.stdout


def test_failed_step_exit_code_is_logged(tmp_path: Path) -> None:
    completed, _ = _run(
        tmp_path,
        plan="orion-vision-retina,build",
        profiles="orion-vision-retina,walkway,orion-vision-retina-walkway",
        safe_build_rc=3,
    )
    assert completed.returncode == 1
    lines = _log_lines(tmp_path)
    ends = [line for line in lines if " END " in line]
    assert len(ends) == 2 and all(" rc=3 " in line for line in ends), lines
    assert lines[-1].endswith("DONE rebuilt=0 failed=orion-vision-retina")


def test_dry_run_writes_no_run_log(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-vision-retina,build",
        profiles="orion-vision-retina,walkway,orion-vision-retina-walkway",
        extra_args=["--dry-run"],
    )
    assert completed.returncode == 0
    assert calls == []
    assert _log_lines(tmp_path) == []


def test_unwritable_run_log_never_fails_the_rebuild(tmp_path: Path) -> None:
    blocker = tmp_path / "post_pull_rebuild.log"
    blocker.mkdir()  # a directory where the log file should be
    completed, calls = _run(tmp_path, plan="orion-vision-retina,build")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-vision-retina up -d --build"]


def test_nothing_to_rebuild_still_leaves_a_log_line(tmp_path: Path) -> None:
    completed, calls = _run(tmp_path, plan="")
    assert completed.returncode == 0
    assert calls == []
    lines = _log_lines(tmp_path)
    assert len(lines) == 1 and " NOTHING " in lines[0], lines
