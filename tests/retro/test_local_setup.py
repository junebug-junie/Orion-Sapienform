import hashlib
import json
from pathlib import Path
import struct

import pytest

from orion.retro.local_runner import checked_input, container_command
from orion.retro.pad_lab import Program, build_rom
from orion.retro.setup_local import install_bytes, RUNNER_IMAGE, SOURCE_COMMIT


def test_authored_rom_is_stable_nrom_with_valid_vectors():
    rom = build_rom()
    assert len(rom) == 16 + 16384 + 8192
    assert rom[:16] == b"NES\x1a" + bytes([1, 1]) + bytes(10)
    assert hashlib.sha256(rom).hexdigest() == "e853994621733bb30053961cc5c88d40b5e44ae3f63ec954361a777d2ae14082"
    nmi, reset, irq = struct.unpack("<HHH", rom[16 + 16384 - 6:16 + 16384])
    assert reset == 0x8000 and nmi == irq and 0x8000 < irq < 0xC000
    assert rom[16 + irq - 0x8000] == 0x40  # RTI


def test_branch_resolver_rejects_out_of_range():
    program = Program()
    program.branch(0xD0, "far")
    program.emit(*([0] * 129))
    program.label("far")
    with pytest.raises(ValueError, match="out of range"):
        program.finish()


def test_install_never_overwrites_different_local_artifact(tmp_path):
    path = tmp_path / "input"
    install_bytes(path, b"original")
    install_bytes(path, b"original")
    with pytest.raises(ValueError, match="differs"):
        install_bytes(path, b"replacement")
    assert path.read_bytes() == b"original"


@pytest.fixture
def kit(tmp_path):
    for name in ("core", "rom"):
        (tmp_path / name).write_bytes(name.encode())
    manifest = {"core": "core", "rom": "rom", "runner_image": RUNNER_IMAGE,
                "source_commit": SOURCE_COMMIT,
                **{f"{name}_sha256": hashlib.sha256(name.encode()).hexdigest()
                   for name in ("core", "rom")}}
    return tmp_path, manifest


def test_changed_input_and_path_escape_are_rejected(kit):
    root, manifest = kit
    (root / "rom").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="fingerprint"):
        checked_input(root, manifest, "rom")
    manifest["rom"] = "../outside"
    outside = root.parent / "outside"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="inside"):
        checked_input(root, manifest, "rom")


def test_container_has_only_required_mounts_and_limits(kit, monkeypatch):
    monkeypatch.setattr("os.getuid", lambda: 1000)
    monkeypatch.setattr("os.getgid", lambda: 1000)
    root, manifest = kit
    command = container_command(root, manifest, root / "output", "test", "pads")
    for option, value in (("--network", "none"), ("--user", "1000:1000"),
                          ("--cap-drop", "ALL"), ("--memory", "256m"),
                          ("--memory-swap", "256m"), ("--cpus", "2"),
                          ("--pids-limit", "64"), ("--security-opt", "no-new-privileges")):
        assert command[command.index(option) + 1] == value
    assert "--read-only" in command
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert len(mounts) == 4
    assert sum(mount.endswith(",readonly") for mount in mounts) == 3
    assert not any("docker.sock" in value for value in command)
    assert RUNNER_IMAGE in command


def test_root_and_unpinned_manifest_rejected(kit, monkeypatch):
    root, manifest = kit
    monkeypatch.setattr("os.getuid", lambda: 0)
    with pytest.raises(ValueError, match="not root"):
        container_command(root, manifest, root / "out", "test", "pads")
    monkeypatch.setattr("os.getuid", lambda: 1000)
    manifest["runner_image"] = "python:latest"
    with pytest.raises(ValueError, match="pinned"):
        container_command(root, manifest, root / "out", "test", "pads")


def test_installed_runtime_is_independent_of_checkout(tmp_path):
    import subprocess
    import sys
    from orion.retro.setup_local import install_runtime
    revision = install_runtime(tmp_path)
    assert revision == install_runtime(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"runtime": revision}))
    result = subprocess.run([sys.executable, str(tmp_path / "run.py"), "--help"],
                            cwd="/tmp", capture_output=True, text=True)
    assert result.returncode == 0
    assert "baseline" in result.stdout
    assert (tmp_path / revision / "orion/retro/local_runner.py").is_file()
    # Runtime selection follows the atomic manifest, not an embedded launcher path.
    relocated = tmp_path / "runtime/relocated"
    (tmp_path / revision).rename(relocated)
    (tmp_path / "manifest.json").write_text(json.dumps({"runtime": "runtime/relocated"}))
    result = subprocess.run([sys.executable, str(tmp_path / "run.py"), "--help"],
                            cwd="/tmp", capture_output=True, text=True)
    assert result.returncode == 0


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), TimeoutError("deadline")])
def test_interrupted_runner_cleans_its_container(kit, monkeypatch, failure):
    from types import SimpleNamespace
    from orion.retro import local_runner
    root, manifest = kit
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "evidence").mkdir()
    monkeypatch.setattr("os.getuid", lambda: 1000)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[1] == "start":
            raise failure
        return SimpleNamespace(stdout="[]", returncode=0)
    monkeypatch.setattr(local_runner.subprocess, "run", execute)
    with pytest.raises(type(failure)):
        local_runner.run(root, "pads")
    assert [command[1] for command in calls][-2:] == ["stop", "rm"]
    assert calls[-1][-1] == calls[0][3]


def test_sigterm_is_converted_to_cleanup_exception(monkeypatch):
    import os
    import signal
    from orion.retro import local_runner
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr("sys.argv", ["local_runner", "pads"])
    monkeypatch.setattr(local_runner, "run", lambda *args: os.kill(os.getpid(), signal.SIGTERM))
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        local_runner.main()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_smoke_rejects_missing_gate_report(kit, monkeypatch):
    from types import SimpleNamespace
    from orion.retro import local_runner
    root, manifest = kit
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "evidence").mkdir()
    monkeypatch.setattr("os.getuid", lambda: 1000)
    def execute(command, **kwargs):
        if command[1] == "start":
            output = next((root / "evidence").iterdir())
            (output / "result.json").write_text(json.dumps({
                "status": "INCOMPLETE_OR_FAILED", "gates": {"thirty_minute_sample": False}}))
        return SimpleNamespace(stdout="[]", returncode=1 if command[1] == "start" else 0)
    monkeypatch.setattr(local_runner.subprocess, "run", execute)
    assert local_runner.run(root, "smoke") == 1


def test_setup_discards_stale_native_build_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orion.retro import setup_local
    source = tmp_path / "sources" / f"fceumm-{SOURCE_COMMIT[:8]}"
    source.mkdir(parents=True)
    binary = source / "fceumm_libretro.so"
    binary.write_bytes(b"stale-native-code")
    (source / "Copying").write_text("fixture license")
    monkeypatch.setattr(setup_local.shutil, "which", lambda name: f"/usr/bin/{name}")
    def execute(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=SOURCE_COMMIT + "\n")
        if command[0] == "make" and command[-1] == "clean":
            binary.unlink()
        elif command[0] == "make" and not binary.exists():
            binary.write_bytes(b"rebuilt-pinned-source")
        return SimpleNamespace(stdout="fixture-compiler\n" if command[0] == "cc" else "")
    monkeypatch.setattr(setup_local, "command", execute)
    manifest = setup_local.setup(tmp_path)
    assert (tmp_path / manifest["core"]).read_bytes() == b"rebuilt-pinned-source"
