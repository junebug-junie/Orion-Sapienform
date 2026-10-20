"""The post-pull rebuild must not recreate orion-retro/orion-hub while a retro game is live.

2026-10-20 22:13Z: a post-pull rebuild recreated orion-athena-retro and the hub
retro gateway mid-game; the game lived only in the retro worker's memory.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rebuild_services_from_git_diff.sh"
GUARD = ROOT / "scripts" / "check_retro_session_live.py"

_spec = importlib.util.spec_from_file_location("check_retro_session_live", GUARD)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

# Fake `docker inspect`: prints $FAKE_DOCKER_OUT, or fails with $FAKE_DOCKER_ERR.
_FAKE_DOCKER = """#!/bin/sh
if [ -n "$FAKE_DOCKER_ERR" ]; then echo "$FAKE_DOCKER_ERR" >&2; exit 1; fi
cat "$FAKE_DOCKER_OUT"
"""

# Fake planner: --plan answers $FAKE_PLAN ("svc,kind;..."), everything else empty.
_FAKE_CLASSIFIER = """\
import os, sys
args = sys.argv[1:]
if any(a.startswith("--") and a not in ("--plan", "--base") for a in args):
    raise SystemExit(0)
for line in os.environ.get("FAKE_PLAN", "").split(";"):
    if line:
        print(line.replace(",", "\\t"))
"""


def _docker_env(tmp_path: Path, *, running: bool | None, state: str | None,
                raw_status: str | None = None, err: str | None = None) -> dict:
    """Fake docker + evidence dir. running=None means no such container."""
    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok=True)
    if raw_status is not None:
        (evidence / "status.json").write_text(raw_status)
    elif state is not None:
        (evidence / "status.json").write_text(json.dumps(
            {"session_id": "s1", "state": state, "game_id": "crystalis", "frame": 0}) + "\n")
    docker = tmp_path / "fake-docker"
    docker.write_text(_FAKE_DOCKER)
    docker.chmod(0o755)
    out = tmp_path / "inspect.json"
    out.write_text(json.dumps([{
        "State": {"Running": bool(running), "Status": "running" if running else "exited"},
        "Mounts": [{"Source": str(evidence), "Destination": "/evidence"}],
    }]))
    env = {"ORION_DOCKER_BIN": str(docker), "FAKE_DOCKER_OUT": str(out), "FAKE_DOCKER_ERR": ""}
    if running is None:
        env["FAKE_DOCKER_ERR"] = "Error: No such object: orion-athena-retro"
    if err is not None:
        env["FAKE_DOCKER_ERR"] = err
    return env


def _guard(tmp_path: Path, monkeypatch, **kw) -> int:
    for k, v in _docker_env(tmp_path, **kw).items():
        monkeypatch.setenv(k, v)
    return guard.main(["--repo-root", str(tmp_path)])


@pytest.mark.parametrize("state", ["starting", "running", "paused"])
def test_live_states_in_running_container_are_live(tmp_path, monkeypatch, state):
    assert _guard(tmp_path, monkeypatch, running=True, state=state) == guard.LIVE


@pytest.mark.parametrize("state", ["stopped", "failed"])
def test_ended_states_are_not_live(tmp_path, monkeypatch, state):
    assert _guard(tmp_path, monkeypatch, running=True, state=state) == guard.NOT_LIVE


def test_stale_running_file_from_dead_container_is_not_live(tmp_path, monkeypatch):
    assert _guard(tmp_path, monkeypatch, running=False, state="running") == guard.NOT_LIVE


def test_no_container_is_not_live(tmp_path, monkeypatch):
    assert _guard(tmp_path, monkeypatch, running=None, state="running") == guard.NOT_LIVE


def test_no_status_file_is_not_live(tmp_path, monkeypatch):
    assert _guard(tmp_path, monkeypatch, running=True, state=None) == guard.NOT_LIVE


def test_corrupt_status_file_is_unknown(tmp_path, monkeypatch):
    assert _guard(tmp_path, monkeypatch, running=True, state=None, raw_status="{not json") == guard.UNKNOWN


def test_docker_daemon_error_is_unknown(tmp_path, monkeypatch):
    code = _guard(tmp_path, monkeypatch, running=True, state="running",
                  err="Cannot connect to the Docker daemon")
    assert code == guard.UNKNOWN


def test_env_file_fallback_when_mount_absent(tmp_path, monkeypatch):
    env = _docker_env(tmp_path, running=True, state="paused")
    Path(env["FAKE_DOCKER_OUT"]).write_text(json.dumps([{"State": {"Running": True}, "Mounts": []}]))
    svc = tmp_path / "services" / "orion-retro"
    svc.mkdir(parents=True)
    (svc / ".env").write_text(f"RETRO_EVIDENCE_DIR={tmp_path / 'evidence'}\n")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert guard.main(["--repo-root", str(tmp_path)]) == guard.LIVE


def _run(tmp_path: Path, *, plan: str, docker: dict, extra_env: dict | None = None,
         extra_args: list[str] | None = None, safe_body: str = "", guard_src: str | None = None):
    fake_scripts = tmp_path / "repo" / "scripts"
    fake_scripts.mkdir(parents=True)
    script = fake_scripts / SCRIPT.name
    shutil.copy2(SCRIPT, script)
    if guard_src is None:
        shutil.copy2(GUARD, fake_scripts / GUARD.name)
    else:
        (fake_scripts / GUARD.name).write_text(guard_src)
    (fake_scripts / "rebuild_affected_services.py").write_text(_FAKE_CLASSIFIER)
    capture = tmp_path / "calls.txt"
    safe = fake_scripts / "safe_docker_build.sh"
    safe.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CAPTURE"\n' + safe_body + 'exit 0\n')
    safe.chmod(0o755)
    env = os.environ.copy()
    env.pop("ORION_REBUILD_IGNORE_LIVE_RETRO", None)
    env.pop("ORION_POST_MERGE_REBUILD", None)
    env.update(docker)
    env.update({"FAKE_PLAN": plan, "CAPTURE": str(capture),
                "ORION_REBUILD_LOG": str(tmp_path / "rebuild.log")})
    env.update(extra_env or {})
    done = subprocess.run([str(script), *(extra_args or [])], cwd=tmp_path / "repo", env=env,
                          text=True, capture_output=True, check=False)
    calls = capture.read_text().splitlines() if capture.exists() else []
    log = (tmp_path / "rebuild.log").read_text() if (tmp_path / "rebuild.log").exists() else ""
    return done, calls, log


def test_live_game_skips_retro_and_hub_but_rebuilds_the_rest(tmp_path):
    done, calls, log = _run(
        tmp_path, plan="orion-retro,build;orion-hub,recreate;orion-thought,build",
        docker=_docker_env(tmp_path, running=True, state="running"))
    assert done.returncode == 0, done.stderr
    assert calls == ["orion-thought up -d --build"]
    assert "SKIPPED orion-retro reason=retro_session_live" in log
    assert "SKIPPED orion-hub reason=retro_session_live" in log
    assert log.rstrip().endswith("DONE rebuilt=1 failed= deferred=orion-retro orion-hub")
    assert "scripts/safe_docker_build.sh orion-hub up -d --build --force-recreate" in done.stderr
    assert "scripts/safe_docker_build.sh orion-retro up -d --build" in done.stderr


def test_unknown_state_skips_too(tmp_path):
    done, calls, log = _run(
        tmp_path, plan="orion-retro,build",
        docker=_docker_env(tmp_path, running=True, state=None, raw_status="garbage"))
    assert done.returncode == 0
    assert calls == []
    assert "SKIPPED orion-retro reason=retro_session_unknown" in log


def test_stale_file_with_dead_container_rebuilds(tmp_path):
    done, calls, log = _run(
        tmp_path, plan="orion-retro,build;orion-hub,build",
        docker=_docker_env(tmp_path, running=False, state="running"))
    assert done.returncode == 0, done.stderr
    assert calls == ["orion-retro build", "orion-retro up -d --build",
                     "orion-hub build", "orion-hub up -d --build"]
    assert "SKIPPED" not in log


def test_override_rebuilds_despite_live_game(tmp_path):
    done, calls, log = _run(
        tmp_path, plan="orion-hub,build",
        docker=_docker_env(tmp_path, running=True, state="paused"),
        extra_env={"ORION_REBUILD_IGNORE_LIVE_RETRO": "1"})
    assert done.returncode == 0
    assert calls == ["orion-hub up -d --build"]
    assert "RETRO-GUARD-OVERRIDE orion-hub" in log


def test_dry_run_reports_the_skip(tmp_path):
    done, calls, _log = _run(
        tmp_path, plan="orion-retro,build",
        docker=_docker_env(tmp_path, running=True, state="running"), extra_args=["--dry-run"])
    assert done.returncode == 0
    assert "[dry-run] would skip: orion-retro reason=retro_session_live" in done.stdout
    assert "[dry-run] would rebuild: orion-retro" not in done.stdout


def test_game_started_during_build_is_not_recreated(tmp_path):
    docker = _docker_env(tmp_path, running=True, state="stopped")
    status = tmp_path / "evidence" / "status.json"
    # The build "takes minutes"; Juniper starts a game meanwhile.
    body = f'case "$*" in *" build") echo \'{{"state": "running", "game_id": "g"}}\' > "{status}";; esac\n'
    done, calls, log = _run(tmp_path, plan="orion-hub,build", docker=docker, safe_body=body)
    assert done.returncode == 0, done.stderr
    assert calls == ["orion-hub build"]
    assert "SKIPPED orion-hub reason=retro_session_live" in log


def test_guard_crash_is_unknown_and_skips_with_one_line_log(tmp_path):
    crash = "import sys\nprint('line one')\nprint('line two')\nraise SystemExit(1)\n"
    done, calls, log = _run(tmp_path, plan="orion-retro,build",
                            docker=_docker_env(tmp_path, running=True, state="running"), guard_src=crash)
    assert done.returncode == 0
    assert calls == []
    skipped = [line for line in log.splitlines() if "SKIPPED" in line]
    assert len(skipped) == 1 and "reason=retro_session_unknown" in skipped[0]
    assert "line one line two" in skipped[0]


def test_done_line_lists_failed_and_deferred(tmp_path):
    docker = _docker_env(tmp_path, running=True, state="paused")
    body = 'case "$*" in orion-thought*) exit 1;; esac\n'
    done, _calls, log = _run(tmp_path, plan="orion-thought,build;orion-hub,build", docker=docker, safe_body=body)
    assert done.returncode == 1
    assert log.rstrip().endswith("DONE rebuilt=0 failed=orion-thought deferred=orion-hub")


def test_missing_docker_binary_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DOCKER_BIN", str(tmp_path / "no-such-docker"))
    assert guard.main(["--repo-root", str(tmp_path)]) == guard.UNKNOWN


def test_live_reason_reports_status_age(tmp_path, monkeypatch, capsys):
    assert _guard(tmp_path, monkeypatch, running=True, state="running") == guard.LIVE
    assert "last written 0.0h ago" in capsys.readouterr().out
