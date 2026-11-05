"""The sql-writer image must build from the primary checkout (2026-11-05).

A wildcard COPY source (``config/person_model/camera_priors.*.yaml``) made BuildKit
walk the whole repo as build context, and the post-pull rebuild failed on a
root-owned, unreadable ``scripts/backup/__pycache__``. COPY sources stay plain
paths, and the Dockerfile-specific ignore file keeps personal local files out.
"""
from pathlib import Path

SVC = Path(__file__).resolve().parents[1]


def _copy_sources():
    for line in (SVC / "Dockerfile").read_text().splitlines():
        parts = line.split()
        if parts and parts[0].upper() == "COPY":
            yield from (p for p in parts[1:-1] if not p.startswith("--"))


def test_no_wildcard_copy_sources():
    assert [s for s in _copy_sources() if any(c in s for c in "*?[")] == []


def test_ignore_file_keeps_personal_files_out_but_the_camera_priors_in():
    lines = (SVC / "Dockerfile.dockerignore").read_text().splitlines()
    assert "config/person_model/*.local.yaml" in lines
    assert "!config/person_model/camera_priors.local.yaml" in lines
    assert "**/__pycache__" in lines
