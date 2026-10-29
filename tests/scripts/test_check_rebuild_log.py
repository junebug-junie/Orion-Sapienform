"""scripts/check_rebuild_log.py: runs that died without a DONE/ABORT line."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_rebuild_log", ROOT / "scripts" / "check_rebuild_log.py")
crl = importlib.util.module_from_spec(_spec)
sys.modules["check_rebuild_log"] = crl  # dataclasses resolve their module by name
_spec.loader.exec_module(crl)

# The real 2026-10-29 lines from ~/.orion/post_pull_rebuild.log (the run stops after the last START).
INCIDENT = """\
2026-10-29T07:21:06Z pid=3250904 head=b79bbb519 RUN base=default services=orion-biometrics orion-cortex-exec orion-harness-governor orion-hub orion-thought orion-cortex-orch recreate= 
2026-10-29T07:21:06Z pid=3250904 head=b79bbb519 START orion-biometrics up -d --build
2026-10-29T07:21:16Z pid=3250904 head=b79bbb519 END rc=0 seconds=10 orion-biometrics up -d --build
2026-10-29T07:21:16Z pid=3250904 head=b79bbb519 START orion-cortex-exec up -d --build cortex-exec cortex-exec-chat cortex-exec-spark cortex-exec-background
2026-10-29T07:21:39Z pid=3250904 head=b79bbb519 END rc=0 seconds=23 orion-cortex-exec up -d --build cortex-exec cortex-exec-chat cortex-exec-spark cortex-exec-background
2026-10-29T07:21:40Z pid=3250904 head=b79bbb519 START orion-harness-governor up -d --build
""".splitlines(keepends=True)
NOW = datetime(2026, 10, 29, 17, 0, tzinfo=timezone.utc).timestamp()


def _dead(_pid):
    return False


def test_the_incident_run_is_reported_with_the_step_in_flight_and_the_queue() -> None:
    [run] = crl.died_runs(INCIDENT, now=NOW, max_age_hours=72, alive=_dead)
    assert run.in_flight == "orion-harness-governor up -d --build"
    assert run.unfinished() == ["orion-harness-governor", "orion-hub", "orion-thought", "orion-cortex-orch"]
    assert run.finish_command() == (
        "scripts/rebuild_services_from_git_diff.sh --services "
        "'orion-harness-governor orion-hub orion-thought orion-cortex-orch'"
    )


def test_a_finished_or_aborted_run_is_not_reported() -> None:
    done = INCIDENT + ["2026-10-29T07:30:00Z pid=3250904 head=b79bbb519 END rc=0 seconds=9 orion-harness-governor up -d --build\n",
                       "2026-10-29T07:40:00Z pid=3250904 head=b79bbb519 DONE rebuilt=6 failed=\n"]
    assert crl.died_runs(done, now=NOW, max_age_hours=72, alive=_dead) == []
    aborted = INCIDENT + ["2026-10-29T07:30:00Z pid=3250904 head=b79bbb519 ABORT reason=SIGTERM in_flight=x\n"]
    assert crl.died_runs(aborted, now=NOW, max_age_hours=72, alive=_dead) == []


def test_a_running_run_is_not_reported() -> None:
    assert crl.died_runs(INCIDENT, now=NOW, max_age_hours=72, alive=lambda pid: True) == []


def test_an_old_run_is_not_reported() -> None:
    later = NOW + 73 * 3600
    assert crl.died_runs(INCIDENT, now=later, max_age_hours=72, alive=_dead) == []


def test_mark_reports_once(tmp_path, capsys, monkeypatch) -> None:
    log = tmp_path / "log"
    log.write_text("".join(INCIDENT))
    monkeypatch.setattr(crl, "_alive", _dead)
    monkeypatch.setattr(crl.time, "time", lambda: NOW)
    assert crl.main(["--log", str(log), "--mark"]) == 0
    assert "died with no END line" in capsys.readouterr().err
    assert crl.main(["--log", str(log), "--mark"]) == 0
    assert capsys.readouterr().err == ""


def test_session_start_payload_and_silence(tmp_path, capsys, monkeypatch) -> None:
    log = tmp_path / "log"
    log.write_text("".join(INCIDENT))
    monkeypatch.setattr(crl, "_alive", _dead)
    monkeypatch.setattr(crl.time, "time", lambda: NOW)
    crl.main(["--log", str(log), "--session-start"])
    payload = json.loads(capsys.readouterr().out)
    assert "orion-harness-governor" in payload["hookSpecificOutput"]["additionalContext"]
    crl.main(["--log", str(tmp_path / "missing"), "--session-start"])
    assert capsys.readouterr().out == ""
