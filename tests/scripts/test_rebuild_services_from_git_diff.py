"""Behavioral coverage for service-specific post-merge rebuild commands."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rebuild_services_from_git_diff.sh"


def test_analytics_rebuild_selects_analytics_compose_profile(tmp_path: Path) -> None:
    fake_root = tmp_path / "repo"
    fake_scripts = fake_root / "scripts"
    fake_scripts.mkdir(parents=True)

    script = fake_scripts / SCRIPT.name
    shutil.copy2(SCRIPT, script)

    classifier = fake_scripts / "rebuild_affected_services.py"
    classifier.write_text("print('orion-analytics')\n")

    capture = tmp_path / "safe-build-args.txt"
    safe_build = fake_scripts / "safe_docker_build.sh"
    safe_build.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" > \"$REBUILD_TEST_CAPTURE\"\n"
    )
    safe_build.chmod(0o755)

    env = os.environ.copy()
    env["REBUILD_TEST_CAPTURE"] = str(capture)
    completed = subprocess.run(
        [str(script)],
        cwd=fake_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert capture.read_text().strip() == (
        "orion-analytics --profile analytics up -d --build"
    )
