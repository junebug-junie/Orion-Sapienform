"""Regression guard for the 2026-08-30 crash loop.

PageIndex's upstream requirements (cloned from an unpinned ref) were pip-installed into the same
interpreter as the service after our own pins, upgrading starlette to 1.x under fastapi 0.115.6.
starlette 1.x removed ``Router(on_startup=...)`` so the app died at import on every restart.
The fix installs the service into its own venv; these checks keep it that way.
"""
from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def _text() -> str:
    return DOCKERFILE.read_text()


def test_service_requirements_install_into_isolated_venv():
    text = _text()
    assert re.search(r"python -m venv \$\{ORION_VENV\}", text)
    assert re.search(r"\$\{ORION_VENV\}/bin/pip install [^\n]*-r /tmp/requirements\.txt", text)


def test_pageindex_requirements_never_installed_into_service_venv():
    for line in _text().splitlines():
        if "/opt/PageIndex/requirements.txt" in line and "pip install" in line:
            assert "ORION_VENV" not in line and "orion-venv" not in line, line


def test_service_reqs_only_installed_via_venv_pip():
    # The system `pip` must not install the service requirements (that is what got clobbered).
    for line in _text().splitlines():
        if "-r /tmp/requirements.txt" in line:
            assert "ORION_VENV" in line, line


def test_cmd_runs_uvicorn_from_venv_without_changing_path():
    text = _text()
    assert 'CMD ["/opt/orion-venv/bin/uvicorn"' in text
    # PATH must stay untouched so PAGEINDEX_PYTHON_BIN=python3 hits the system interpreter
    # (where PageIndex and its deps live), not the service venv.
    for line in text.splitlines():
        if line.lstrip().upper().startswith("ENV"):
            assert not re.search(r"\bPATH[=\s]", line), line


def test_build_time_guard_constructs_fastapi_router_after_pageindex_install():
    text = _text()
    pageindex_idx = text.index("/opt/PageIndex/requirements.txt")
    guard = re.search(r"\$\{ORION_VENV\}/bin/python -c \"[^\"]*fastapi\.APIRouter\(\); import app\.main", text)
    assert guard and guard.start() > pageindex_idx


def test_service_requirements_declare_pyyaml():
    # orion.core.bus imports yaml; it used to leak in from PageIndex's requirements.
    reqs = (DOCKERFILE.parent / "requirements.txt").read_text().lower()
    assert re.search(r"^pyyaml==", reqs, re.MULTILINE)


def test_build_guard_runs_pageindex_cli_under_system_python3():
    assert re.search(r"cd /opt/PageIndex && python3 run_pageindex\.py --help", _text())


def test_pageindex_python_bin_stays_system_python3():
    # Pointing this at the service venv would run PageIndex without its deps.
    svc = DOCKERFILE.parent
    assert re.search(r"^PAGEINDEX_PYTHON_BIN=python3$", (svc / ".env_example").read_text(), re.MULTILINE)
    assert "${PAGEINDEX_PYTHON_BIN:-python3}" in (svc / "docker-compose.yml").read_text()
    assert 'PAGEINDEX_PYTHON_BIN: str = Field(default="python3")' in (svc / "app" / "settings.py").read_text()
