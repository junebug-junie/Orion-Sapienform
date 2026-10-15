"""CI service containers must come from GHCR, pulled with credentials when private.

Docker Hub (2026-10-09) and ECR Public (2026-10-15, "toomanyrequests: Data limit
exceeded") both rate-limit anonymous pulls per source IP. GitHub-hosted runners
share IPs with everyone else, so those limits fail our jobs before any test runs.
GHCR pulls from Actions with GITHUB_TOKEN are not subject to either limit.
"""
from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
OWN_PREFIX = "ghcr.io/junebug-junie/"


def _service_images():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text()) or {}
        top_perms = doc.get("permissions")
        for job_name, job in (doc.get("jobs") or {}).items():
            matrix = ((job.get("strategy") or {}).get("matrix") or {})
            for svc_name, svc in (job.get("services") or {}).items():
                image = svc.get("image", "")
                images = [image]
                if image.startswith("${{ matrix."):
                    key = image.removeprefix("${{ matrix.").removesuffix("}}").strip()
                    images = list(matrix.get(key) or [])
                for img in images:
                    yield path.name, job_name, svc_name, img, svc, job.get("permissions", top_perms)


def test_there_are_service_images_to_check():
    assert list(_service_images()), "parser found no service containers; the gate would be vacuous"


def test_service_images_are_pulled_from_ghcr():
    bad = [f"{f}:{j}.{s} -> {img}" for f, j, s, img, *_ in _service_images() if not img.startswith("ghcr.io/")]
    assert not bad, "service images must come from GHCR (Docker Hub / ECR Public rate-limit runner IPs):\n" + "\n".join(bad)


def test_private_ghcr_copies_have_credentials_and_packages_read():
    bad = []
    for f, j, s, img, svc, perms in _service_images():
        if not img.startswith(OWN_PREFIX):
            continue
        creds = svc.get("credentials") or {}
        if "GITHUB_TOKEN" not in str(creds.get("password", "")):
            bad.append(f"{f}:{j}.{s} missing credentials")
        if not isinstance(perms, dict) or perms.get("packages") not in ("read", "write"):
            bad.append(f"{f}:{j} missing permissions.packages: read")
    assert not bad, "\n".join(bad)
