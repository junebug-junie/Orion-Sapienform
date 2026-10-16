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
    for line in os.environ.get("FAKE_POOL", "").split(";"):
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
    extra_args: list[str] | None = None,
    safe_build_reads_stdin: bool = False,
    migrations: str = "",
    checker: str | None = None,
    applier: str | None = None,
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
    )
    safe_build.chmod(0o755)

    env = os.environ.copy()
    env["REBUILD_TEST_CAPTURE"] = str(capture)
    env["FAKE_PLAN"] = plan
    env["FAKE_PROFILES"] = profiles
    env["FAKE_POOL"] = pool
    env["FAKE_MIGRATIONS"] = migrations
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
    assert calls == ["orion-diffusion-host build diffusion-host"]
    assert not any(" up " in f" {c} " for c in calls)


def test_pool_hold_still_brings_up_the_stacks_other_services(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-diffusion-host,recreate",
        pool="orion-diffusion-host,hold,diffusion-host;orion-diffusion-host,start,sidecar",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == [
        "orion-diffusion-host build diffusion-host",
        "orion-diffusion-host up -d --build --no-deps --force-recreate sidecar",
    ]


def test_running_pool_target_keeps_the_plain_up(tmp_path: Path) -> None:
    # Helper prints nothing when the launch target is running: unchanged behavior.
    completed, calls = _run(tmp_path, plan="orion-diffusion-host,build", pool="")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-diffusion-host up -d --build"]


def test_dry_run_reports_pool_hold(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-diffusion-host,build",
        pool="orion-diffusion-host,hold,diffusion-host",
        extra_args=["--dry-run"],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == []
    assert "would build without starting (GPU pool holds it stopped): orion-diffusion-host diffusion-host" in completed.stdout


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
    assert calls == [f"APPLY {_MIG}", "orion-hub up -d --build", "orion-sql-writer up -d --build"]
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


def test_no_migrations_never_calls_the_applier(tmp_path: Path) -> None:
    """The classifier lists no migration on a host without orion-sql-db (circe), so the applier
    is never reached there; see test_migrations_are_not_reported_on_a_host_without_the_database."""
    completed, calls = _run(tmp_path, plan="orion-hub,build", applier=_APPLIER)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-hub up -d --build"]
