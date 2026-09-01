"""Tests for resume_compact_convex_data.sh.

compact_convex_data.sh is `set -euo pipefail` with no resume path. Confirmed
live 2026-08-31: it died at step 5b, which left the backend running on a fresh
EMPTY database with the real data only in the job dir's export.zip. The town
served an empty world for 33 hours.

The dangerous move in that state is re-running compact_convex_data.sh: its
step 1 exports the live (now empty) database into a NEW job dir and its step 6
reimports that, converting a recoverable outage into permanent data loss. This
script exists so there is a correct alternative, and its single most important
guard is refusing to import an export that looks like it came from an
already-emptied database.

The guard tests below actually run the script rather than grepping it.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_SCRIPT = _SERVICE / "scripts" / "resume_compact_convex_data.sh"
_COMPACT = _SERVICE / "scripts" / "compact_convex_data.sh"

# The script derives ROOT from its own BASH_SOURCE, so running the REPO copy
# points UPSTREAM at the real services/orion-ai-town/upstream. Where that exists
# -- the main checkout and every deploy host; it is absent here only because
# upstream/ is gitignored -- a test that clears the doc-count and env.backup
# guards would proceed into `convex env set --from-file` with a fake env.backup
# and `convex import --replace-all` with a synthetic export, against whatever
# Convex deployment is listening on :3210. That is the production town, replaced,
# from pytest. Every test therefore runs a COPY staged in tmp_path with a fake
# upstream/ and a stub PATH, so no test can reach a real deployment.
_FAKE_NODE = """#!/usr/bin/env bash
if [ "$1" = "--version" ]; then echo "v20.0.0-fake"; exit 0; fi
# stand in for `node <convex-cli> <subcommand>`
shift
case "$*" in
  *--version*)                  echo "0.0.0-fake" ;;
  *"run world:defaultWorldStatus"*)
      if [ "${FAKE_WORLD_HEALTHY:-0}" = "1" ]; then
        echo '{"worldId": "fakeworld", "status": "running"}'
      else
        exit 1
      fi
      ;;
  *) echo "fake convex: $*" ;;
esac
exit 0
"""

_FAKE_CURL = """#!/usr/bin/env bash
[ "${FAKE_BACKEND_UP:-1}" = "1" ] && exit 0
exit 1
"""

_FAKE_DF = """#!/usr/bin/env bash
# df -PB1 <path>
echo "Filesystem 1B-blocks Used Available Capacity Mounted"
echo "fake 1000000000000 0 ${FAKE_FREE_BYTES:-999000000000} 1% /"
"""


def _make_export(path: Path, doc_count: int) -> None:
    """Write an export.zip shaped like a real `convex export` archive."""
    with zipfile.ZipFile(path, "w") as z:
        rows = "\n".join(json.dumps({"_id": f"d{i}"}) for i in range(doc_count))
        if rows:
            rows += "\n"
        z.writestr("snapshot/messages/documents.jsonl", rows)


def _job_dir(tmp_path: Path, doc_count: int, with_env: bool = True) -> Path:
    d = tmp_path / "aitown-compact-20260101-000000"
    d.mkdir()
    _make_export(d / "export.zip", doc_count)
    if with_env:
        (d / "env.backup").write_text("LLM_MODEL=quick_background\n", encoding="utf-8")
    return d


def _stage(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Copy the script into a fake service tree behind a stub PATH."""
    root = tmp_path / "service"
    (root / "scripts").mkdir(parents=True)
    (root / "upstream" / "node_modules" / "convex" / "bin").mkdir(parents=True)
    (root / "upstream" / "node_modules" / "convex" / "bin" / "main.js").write_text(
        "// fake convex cli\n", encoding="utf-8"
    )

    script = root / "scripts" / "resume_compact_convex_data.sh"
    shutil.copy2(_SCRIPT, script)
    # the LLM-route check the script calls between 5c and 6
    (root / "scripts" / "check_llm_route_not_circe.py").write_text(
        "import sys; sys.exit(0)\n", encoding="utf-8"
    )

    binp = tmp_path / "bin"
    binp.mkdir()
    for name, body in (("node", _FAKE_NODE), ("curl", _FAKE_CURL), ("df", _FAKE_DF)):
        p = binp / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{binp}:{env['PATH']}"
    return script, env


def _run(job_dir: Path, tmp_path: Path,
         **overrides: str) -> subprocess.CompletedProcess:
    script, env = _stage(tmp_path)
    env.update(overrides)
    return subprocess.run(
        ["bash", str(script), str(job_dir)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_tests_never_run_the_repo_copy_against_a_real_upstream():
    """Guard on the guard: the sandbox above is what keeps pytest away from prod.

    If a future test calls the repo script directly, ROOT resolves to the real
    service dir and `import --replace-all` becomes reachable from the suite.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    body = src[src.index("def _make_export"):]
    assert 'subprocess.run(\n        ["bash", str(_SCRIPT)' not in body
    assert "str(_SCRIPT)" not in body.split("def test_")[0].replace(
        "shutil.copy2(_SCRIPT, script)", ""
    ), "only _stage() may reference the repo script, and only to copy it"


def test_refuses_an_export_of_an_already_emptied_database(tmp_path):
    """The guard that prevents turning an outage into permanent data loss."""
    r = _run(_job_dir(tmp_path, doc_count=3), tmp_path)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "refusing to import" in out, out
    # must not have gone on to touch the deployment
    assert "step 6/7" not in out, out


def test_refuses_a_corrupt_export(tmp_path):
    d = tmp_path / "aitown-compact-20260101-000000"
    d.mkdir()
    (d / "export.zip").write_bytes(b"not a zip file at all")
    (d / "env.backup").write_text("LLM_MODEL=quick_background\n", encoding="utf-8")

    r = _run(d, tmp_path)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "refusing to import" in out, out


def test_accepts_a_populated_export(tmp_path):
    """Positive control: an always-refusing guard would pass the tests above."""
    r = _run(_job_dir(tmp_path, doc_count=5000), tmp_path)

    out = r.stdout + r.stderr
    assert "export.zip holds 5000 documents" in out, out
    assert "refusing to import" not in out, out


def test_refuses_a_missing_env_backup(tmp_path):
    r = _run(_job_dir(tmp_path, doc_count=5000, with_env=False), tmp_path)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "env.backup" in out, out


def test_refuses_a_nonexistent_job_dir(tmp_path):
    r = _run(tmp_path / "does-not-exist", tmp_path)
    assert r.returncode != 0
    assert "job dir not found" in (r.stdout + r.stderr)


def test_refuses_to_run_against_a_healthy_deployment(tmp_path):
    """Wrong job dir + healthy town = --replace-all over live data.

    The no-argument mode guesses the newest /tmp/aitown-compact-*, so an
    operator can land here by accident. Everything else in the preflight
    validates the job dir; only this looks at the deployment.
    """
    r = _run(_job_dir(tmp_path, doc_count=5000), tmp_path, FAKE_WORLD_HEALTHY="1")

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "already answers world:defaultWorldStatus" in out, out
    assert "AITOWN_RESUME_CONFIRM=1" in out, out
    assert "step 6/7" not in out, out


def test_confirm_overrides_the_healthy_deployment_refusal(tmp_path):
    """Positive control: the refusal must have a documented door."""
    r = _run(_job_dir(tmp_path, doc_count=5000), tmp_path,
             FAKE_WORLD_HEALTHY="1", AITOWN_RESUME_CONFIRM="1")

    out = r.stdout + r.stderr
    assert "proceeding on explicit AITOWN_RESUME_CONFIRM=1" in out, out
    assert "step 6/7" in out, out


def test_proceeds_when_the_deployment_is_actually_broken(tmp_path):
    """The state this script exists for: functions/data gone."""
    r = _run(_job_dir(tmp_path, doc_count=5000), tmp_path, FAKE_WORLD_HEALTHY="0")

    out = r.stdout + r.stderr
    assert "consistent with the broken state" in out, out
    assert "step 6/7" in out, out


def test_refuses_a_stale_export(tmp_path):
    """A stale export is as destructive as an empty one -- it rolls time back."""
    job = _job_dir(tmp_path, doc_count=5000)
    old = int(__import__("time").time()) - 48 * 3600
    os.utime(job / "export.zip", (old, old))

    r = _run(job, tmp_path)

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "roll the world back" in out, out
    assert "step 6/7" not in out, out


def test_confirm_overrides_the_stale_export_refusal(tmp_path):
    job = _job_dir(tmp_path, doc_count=5000)
    old = int(__import__("time").time()) - 48 * 3600
    os.utime(job / "export.zip", (old, old))

    r = _run(job, tmp_path, AITOWN_RESUME_CONFIRM="1")

    assert "step 6/7" in (r.stdout + r.stderr)


def test_refuses_when_free_space_is_short(tmp_path):
    r = _run(_job_dir(tmp_path, doc_count=5000), tmp_path,
             FAKE_FREE_BYTES="1000000")

    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "AITOWN_RESUME_MIN_FREE_BYTES" in out, out


def test_never_invokes_the_compaction_script(tmp_path):
    """Re-running the compactor in the broken state is the data-loss path.

    Comments are expected to name it -- explaining the trap is the point. Only
    executable lines are checked.
    """
    text = _SCRIPT.read_text(encoding="utf-8")
    body = text[text.index("set -uo pipefail"):]
    # strip trailing comments too, not just whole-line ones
    code = [
        line.split("#")[0] for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    offenders = [line for line in code if "compact_convex_data.sh" in line]
    assert not offenders, (
        "the resume script must never shell out to the compactor: its step 1 "
        f"would export the empty database over the good export. Found: {offenders}"
    )


def test_replays_the_steps_the_compactor_never_reached(tmp_path):
    text = _SCRIPT.read_text(encoding="utf-8")
    for marker in ("dev --once", "env set --from-file", "import --replace-all",
                   "world:heartbeatWorld"):
        assert marker in text, marker


def test_verifies_functions_and_data_rather_than_trusting_health(tmp_path):
    # The container reported "healthy" for the entire 33-hour outage, because
    # the healthcheck proves the process answers, not that it has functions or
    # data. defaultWorldStatus only answers when both are back.
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "world:defaultWorldStatus" in text
    assert "== verify ==" in text


def test_script_exists_and_is_executable():
    assert _SCRIPT.exists()
    assert _SCRIPT.stat().st_mode & stat.S_IXUSR


def test_script_has_bash_syntax_ok():
    r = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True,
                       text=True, timeout=10)
    assert r.returncode == 0, r.stderr


def test_compactor_documents_the_resume_path():
    """A recovery script nobody can find is not a recovery path."""
    assert "resume_compact_convex_data.sh" in _COMPACT.read_text(encoding="utf-8")
