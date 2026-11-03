"""Repo-root helpers moved from the retired app/self_study.py (Self Atlas retirement, 2026-11)."""
import importlib.util
import sys
import types
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
for name, path in ((PACKAGE_NAME, SERVICE_DIR), (APP_PACKAGE_NAME, APP_DIR)):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg

spec = importlib.util.spec_from_file_location(f"{APP_PACKAGE_NAME}.orch_paths", APP_DIR / "orch_paths.py")
orch_paths = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = orch_paths
spec.loader.exec_module(orch_paths)


def test_resolve_repo_root_prefers_explicit_env_override(monkeypatch, tmp_path):
    forced_root = tmp_path / "forced-root"
    forced_root.mkdir()
    monkeypatch.setenv("ORION_REPO_ROOT", str(forced_root))
    assert orch_paths._resolve_repo_root(tmp_path / "nested" / "app" / "orch_paths.py") == forced_root.resolve()


def test_resolve_repo_root_detects_normal_repo_layout_from_markers(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    repo_root = tmp_path / "Orion-Sapienform"
    service_dir = repo_root / "services" / "orion-cortex-exec" / "app"
    service_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[tool.poetry]\nname = "orion"\n', encoding="utf-8")
    (repo_root / "orion").mkdir()
    assert orch_paths._resolve_repo_root(service_dir / "orch_paths.py") == repo_root.resolve()


def test_resolve_repo_root_handles_flattened_container_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    assert orch_paths._resolve_repo_root(tmp_path / "app" / "app" / "orch_paths.py") == (tmp_path / "app").resolve()


def test_resolve_repo_root_fallback_is_deterministic_when_markers_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    module_path = tmp_path / "sandbox" / "nested" / "orch_paths.py"
    assert orch_paths._resolve_repo_root(module_path) == orch_paths._resolve_repo_root(module_path) == (tmp_path / "sandbox").resolve()


def test_correlation_id_is_a_stable_uuid():
    raw = "not-a-uuid"
    assert orch_paths._as_envelope_correlation_id(raw) == orch_paths._as_envelope_correlation_id(raw)
    u = "12345678-1234-5678-1234-567812345678"
    assert orch_paths._as_envelope_correlation_id(u) == u


def test_self_atlas_scan_is_gone_kill_means_kill():
    """Self Atlas retirement (2026-11): the daily Layer 1 scan and its timer are deleted, not flagged off."""
    assert not (APP_DIR / "self_study.py").exists()
    assert not (APP_DIR / "self_study_refresh.py").exists()
    main_src = (APP_DIR / "main.py").read_text(encoding="utf-8")
    settings_src = (APP_DIR / "settings.py").read_text(encoding="utf-8")
    env_src = (SERVICE_DIR / ".env_example").read_text(encoding="utf-8")
    for src in (main_src, settings_src, env_src):
        assert "SELF_STUDY_INSPECT_INTERVAL_SEC" not in src
        assert "self_study_refresh" not in src
    # The analysis sensor is a different thing and must survive.
    assert (APP_DIR / "self_study_analysis.py").exists()
