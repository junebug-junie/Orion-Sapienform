from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_scripts_dir_no_stdlib_shadow as gate  # noqa: E402


def test_find_stdlib_shadows_clean_directory_returns_empty(tmp_path):
    (tmp_path / "not_a_stdlib_name.py").write_text("# fine\n")
    (tmp_path / "also_fine").mkdir()

    assert gate.find_stdlib_shadows(tmp_path) == []


def test_find_stdlib_shadows_flags_a_colliding_py_file(tmp_path):
    (tmp_path / "json.py").write_text("# shadows stdlib json\n")

    assert gate.find_stdlib_shadows(tmp_path) == ["json"]


def test_find_stdlib_shadows_flags_a_colliding_directory(tmp_path):
    # This is the EXACT shape of the real bug: scripts/platform/ was a
    # directory, not a .py file.
    (tmp_path / "platform").mkdir()
    (tmp_path / "platform" / "__init__.py").write_text("")

    assert gate.find_stdlib_shadows(tmp_path) == ["platform"]


def test_find_stdlib_shadows_ignores_dotfiles_and_dotdirs(tmp_path):
    (tmp_path / ".state").mkdir()
    (tmp_path / ".pytest_cache").mkdir()

    assert gate.find_stdlib_shadows(tmp_path) == []


def test_find_stdlib_shadows_dedupes_file_and_dir_same_name(tmp_path):
    # Pathological but cheap to handle correctly: a stdlib name appearing
    # as both a .py file conceptually and counted once, not twice.
    (tmp_path / "re.py").write_text("")

    result = gate.find_stdlib_shadows(tmp_path)

    assert result == ["re"]
    assert len(result) == len(set(result))


def test_real_scripts_directory_is_clean():
    # The actual regression test: after the scripts/platform/ ->
    # scripts/platform_audits/ rename, the real repo's scripts/ directory
    # must have zero stdlib-name collisions. This is what the rename
    # commit is actually verified by -- not a synthetic tmp_path fixture.
    real_scripts_dir = Path(__file__).resolve().parent

    assert gate.find_stdlib_shadows(real_scripts_dir) == []


def test_main_exits_zero_and_prints_clean_for_real_scripts_dir():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "check_scripts_dir_no_stdlib_shadow.py")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "clean" in result.stdout


def test_main_exits_nonzero_when_a_collision_exists(tmp_path, monkeypatch):
    (tmp_path / "types.py").write_text("# shadows stdlib types\n")
    monkeypatch.setattr(gate, "SCRIPTS_DIR", tmp_path)

    exit_code = gate.main()

    assert exit_code == 1
