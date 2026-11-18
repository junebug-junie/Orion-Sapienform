"""Compile an explicitly synthetic core to verify the actual ctypes ABI boundary."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from orion.retro.probe import CoreProbe, digest


@pytest.fixture
def native(tmp_path):
    if not shutil.which("cc"):
        pytest.fail("native ABI gate requires a C compiler")
    library = tmp_path / "fixture.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(library),
                    str(Path(__file__).with_name("fixture_core.c"))], check=True)
    rom = tmp_path / "fixture.nes"
    rom.write_bytes(b"fixture")
    return library, rom


def test_native_callbacks_save_restore_and_distinct_ports(native):
    core = CoreProbe(*native)
    try:
        core.step()
        start = core.snapshot()
        p1 = core.step(1 << 8, 0)
        core.restore(start)
        p2 = core.step(0, 1)
        assert p1 != p2
        core.restore(start)
        assert core.step(1 << 8, 0) == p1
        assert core.audio_samples == 12
        assert core.video_frames == 4
        assert core.identity["core_name"] == "fixture"
        assert core.identity["core_sha256"] == digest(native[0])
    finally:
        core.close()
        core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.step()


def test_rejected_content(native):
    native[1].write_bytes(b"invalid")
    with pytest.raises(ValueError, match="rejected"):
        CoreProbe(*native)


def test_cli_fingerprint_mismatch_does_not_load_core(native, tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "orion.retro.preflight", "--core", str(native[0]),
        "--rom", str(native[1]), "--core-sha256", "wrong", "--rom-sha256", digest(native[1]),
        "--output", str(tmp_path / "result.json"),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "SHA-256 mismatch" in result.stderr
    assert not (tmp_path / "result.json").exists()
