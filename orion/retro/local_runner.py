"""Run the installed offline kit in a disposable, restricted Docker container."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import uuid

from .probe import digest
from .setup_local import DEFAULT_ROOT, RUNNER_IMAGE, SOURCE_COMMIT


def checked_input(root, manifest, key):
    path = (root / manifest[key]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{key} must be a file inside the kit root")
    if digest(path) != manifest[f"{key}_sha256"]:
        raise ValueError(f"{key} fingerprint mismatch; rerun setup or inspect local changes")
    return path


def container_command(root, manifest, output, name, mode):
    if os.getuid() == 0:
        raise ValueError("run the kit as an ordinary user, not root")
    if mode not in {"pads", "smoke", "baseline"}:
        raise ValueError("unknown evaluation mode")
    if manifest["runner_image"] != RUNNER_IMAGE or manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("manifest does not match this kit's pinned dependencies")
    core, rom = (checked_input(root, manifest, key) for key in ("core", "rom"))
    source = Path(__file__).parent.resolve()
    mounts = [(source, "/app/orion/retro", True), (core, "/inputs/core.so", True),
              (rom, "/inputs/game.nes", True), (output, "/evidence", False)]
    command = ["docker", "create", "--name", name, "--network", "none", "--read-only",
               "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "64",
               "--ulimit", "core=0", "--ulimit", "fsize=67108864",
               "--memory", "256m", "--memory-swap", "256m", "--cpus", "2",
               "--log-driver", "journald", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
               "--workdir", "/app", "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for path, target, readonly in mounts:
        if "," in str(path):
            raise ValueError("Docker bind paths must not contain commas")
        mount = f"type=bind,src={path},dst={target}" + (",readonly" if readonly else "")
        command.extend(["--mount", mount])
    module = "orion.retro.pad_eval" if mode == "pads" else "orion.retro.preflight"
    command.extend([RUNNER_IMAGE, "python", "-m", module, "--core", "/inputs/core.so",
                    "--core-sha256", manifest["core_sha256"], "--rom", "/inputs/game.nes",
                    "--output", "/evidence/result.json"])
    if mode != "pads":
        command.extend(["--rom-sha256", manifest["rom_sha256"], "--frames",
                        "110000" if mode == "baseline" else "120", "--replay-frames", "1000"])
    return command


def run(root, mode):
    root = root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = root / "evidence" / run_id
    output.mkdir(mode=0o700)
    name = f"orion-retro-{run_id.lower()}"
    command = container_command(root, manifest, output, name, mode)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        inspected = subprocess.run(["docker", "inspect", name], check=True,
                                   capture_output=True, text=True).stdout
        (output / "container.json").write_text(inspected)
        print(f"Evidence: {output}", flush=True)
        print(f"Progress: tail -f {output / 'progress.log'}", flush=True)
        with (output / "progress.log").open("w") as log:
            completed = subprocess.run(["docker", "start", "--attach", name],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=2400 if mode == "baseline" else 120)
        metadata = {"mode": mode, "exit_code": completed.returncode, "setup": manifest}
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        result_path = output / "result.json"
        if completed.returncode not in (0, 1) or not result_path.exists() or not result_path.stat().st_size:
            raise RuntimeError(f"native run failed; inspect {output}")
        result = json.loads(result_path.read_text())
        print(result["status"], flush=True)
        # A smoke is useful only when EVERY applicable gate passes. Short duration
        # is intentionally never relabeled as full baseline acceptance.
        if mode == "smoke":
            checks = result.get("gates", {})
            required = {"nominal_cadence", "deadline_misses_below_one_percent",
                        "video_observed", "audio_observed", "same_core_replay",
                        "replay_1000_frames"}
            return 0 if all(checks.get(key) is True for key in required) else 1
        return 0 if result["status"] == "PASS" else 1
    finally:
        # Only this invocation's generated name is eligible for cleanup.
        subprocess.run(["docker", "stop", "--time", "2", name], capture_output=True)
        subprocess.run(["docker", "rm", name], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pads", "smoke", "baseline"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt("SIGTERM: stopping the owned diagnostic container")
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return run(args.root, args.mode)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
