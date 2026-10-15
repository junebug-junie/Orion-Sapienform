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
    extra_args: list[str] | None = None,
    safe_build_reads_stdin: bool = False,
    migrations: str = "",
    checker: str | None = None,
):
    fake_root = tmp_path / "repo"
    fake_scripts = fake_root / "scripts"
    fake_scripts.mkdir(parents=True)

    script = fake_scripts / SCRIPT.name
    shutil.copy2(SCRIPT, script)
    (fake_scripts / "rebuild_affected_services.py").write_text(_FAKE_CLASSIFIER)
    if checker is not None:
        (fake_scripts / "check_sql_migrations_applied.py").write_text(checker)

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


# 2026-10-15: a migration-only pull (PR #2698) recreated orion-athena-sql-db.
# The classifier now rebuilds nothing for it; the script must instead name the
# migration, its live state and the apply command, and still build nothing.
_MIG = "services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql"

_CHECKER_MISSING = """\
import sys
name = sys.argv[sys.argv.index("--file") + 1]
print("RED [        missing] " + name + "  (last changed 2026-10-15)")
print("            MISSING table vision_confirm_asks")
print("            apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney "
      "-v ON_ERROR_STOP=1 < services/orion-sql-db/" + name)
print("")
print("1 migration file(s) replayed: 1 missing; alarm window: all time")
raise SystemExit(1)
"""

_CHECKER_UNREACHABLE = """\
import sys
print("could not connect to localhost:55432/conjourney: refused", file=sys.stderr)
raise SystemExit(2)
"""


def test_migration_only_pull_builds_nothing_and_names_the_apply_command(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="",
        migrations=f"{_MIG},orion-sql-writer, orion-hub",
        checker=_CHECKER_MISSING,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == []
    out = completed.stdout
    assert f"  - {_MIG} (required by: orion-sql-writer, orion-hub)" in out
    assert "live state: NOT APPLIED" in out
    assert "apply: docker exec -i orion-athena-sql-db psql" in out
    assert "replayed" not in out  # corpus-wide summary trimmed
    assert "no affected services to rebuild" in out


def test_unreachable_database_reports_unknown_with_a_raw_apply_command(tmp_path: Path) -> None:
    completed, calls = _run(
        tmp_path,
        plan="orion-hub,build",
        migrations=f"{_MIG},orion-hub",
        checker=_CHECKER_UNREACHABLE,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert calls == ["orion-hub up -d --build"]
    out = completed.stdout
    assert "live state: UNKNOWN (checker exit 2" in out
    assert "could not connect" in out
    assert f"apply by hand (check the file header first): docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < {_MIG}" in out


def test_no_migrations_prints_no_migration_block(tmp_path: Path) -> None:
    completed, _calls = _run(tmp_path, plan="orion-hub,build")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "hand-applied SQL migration" not in completed.stdout
