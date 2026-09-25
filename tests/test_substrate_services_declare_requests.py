"""Every service that imports ``orion.substrate`` must declare ``requests``.

Importing any ``orion.substrate.<module>`` runs the package ``__init__``, which
imports ``graphdb_store``, which does ``import requests`` at module top. A
service image without ``requests`` therefore dies at import time.

Incident 2026-09-25: PR #2343 made orion-feedback-runtime and
orion-policy-runtime import ``orion.substrate.pending_marker_reconcile``. Their
requirements never declared ``requests``; both crash-looped
(``ModuleNotFoundError: No module named 'requests'``) from the 08:19 UTC deploy.
A transitive ``requests`` from another package does not count: declare it.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_USES_SUBSTRATE = re.compile(r"^\s*(from\s+orion\.substrate[\s.]|import\s+orion\.substrate\b)", re.M)
_DECLARES_REQUESTS = re.compile(r"^\s*requests\b", re.M | re.I)


def _services_using_orion_substrate() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for req in sorted((ROOT / "services").glob("*/requirements.txt")):
        svc_dir = req.parent
        for py in svc_dir.rglob("*.py"):
            if "tests" in py.relative_to(svc_dir).parts:
                continue
            try:
                text = py.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if _USES_SUBSTRATE.search(text):
                found[svc_dir.name] = req
                break
    return found


def test_orion_substrate_still_imports_requests_at_package_import() -> None:
    # If this stops being true the gate below is stale, not wrong: revisit it.
    init = (ROOT / "orion/substrate/__init__.py").read_text()
    assert "graphdb_store" in init
    assert re.search(r"^import requests", (ROOT / "orion/substrate/graphdb_store.py").read_text(), re.M)


def test_the_scan_finds_the_incident_services() -> None:
    found = _services_using_orion_substrate()
    assert {"orion-feedback-runtime", "orion-policy-runtime"} <= set(found)


def test_every_orion_substrate_service_declares_requests() -> None:
    missing = sorted(
        svc for svc, req in _services_using_orion_substrate().items()
        if not _DECLARES_REQUESTS.search(req.read_text())
    )
    assert not missing, (
        "these services import orion.substrate (whose __init__ imports requests) "
        f"but do not declare requests in requirements.txt: {missing}"
    )
