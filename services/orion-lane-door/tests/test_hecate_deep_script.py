"""services/orion-vllm-host/scripts/hecate-deep.sh starts and stops the agent-deep lane door with the engine.

Runs the real script in a scratch repo layout with stub commands (no docker, nothing real is started)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "services/orion-vllm-host/scripts/hecate-deep.sh"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(0o755)


def _run(tmp_path: Path, *args: str) -> tuple[int, list[str], Path]:
    root = tmp_path / "repo"
    (root / "services/orion-vllm-host/scripts").mkdir(parents=True)
    (root / "services/orion-lane-door").mkdir(parents=True)
    (root / "scripts").mkdir()
    shutil.copy(SCRIPT, root / "services/orion-vllm-host/scripts/hecate-deep.sh")
    (root / "services/orion-vllm-host/scripts/check_image_storage.py").write_text("")
    (root / "services/orion-lane-door/.env_example").write_text("LANE_DOOR_MODE=shadow\n")
    log = tmp_path / "calls.log"
    _stub(root / "scripts/safe_docker_build.sh", f'echo "$*" >> {log}')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "hostname", "echo hecate")
    _stub(bin_dir / "findmnt", "echo /dev/nvme0n1p1")
    _stub(bin_dir / "docker", "echo false")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    proc = subprocess.run(["bash", str(root / "services/orion-vllm-host/scripts/hecate-deep.sh"), *args],
                          env=env, capture_output=True, text=True, check=False)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc.returncode, calls, root


def test_up_starts_the_door_then_the_engine_and_makes_the_door_env(tmp_path):
    rc, calls, root = _run(tmp_path, "up", "-d")
    assert rc == 0
    assert calls == ["orion-lane-door --profile hecate up -d --build lane-door-agent-deep",
                     "orion-vllm-host -f services/orion-vllm-host/docker-compose.deep.yml --profile vllm-worker up -d"]
    assert (root / "services/orion-lane-door/.env").read_text() == "LANE_DOOR_MODE=shadow\n"


def test_stop_and_down_stop_the_door_first(tmp_path):
    for i, verb in enumerate(("stop", "down")):
        rc, calls, _ = _run(tmp_path / str(i), verb)
        assert rc == 0
        assert calls[0] == "orion-lane-door --profile hecate stop lane-door-agent-deep"
        assert calls[1].endswith(f"--profile vllm-worker {verb}")


def test_logs_touch_no_door(tmp_path):
    rc, calls, _ = _run(tmp_path, "logs")
    assert rc == 0 and len(calls) == 1 and calls[0].startswith("orion-vllm-host")
