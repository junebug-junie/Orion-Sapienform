from __future__ import annotations

import os
import sys

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.logic import scheduler_docker_findings  # noqa: E402


def test_empty_not_dict() -> None:
    assert scheduler_docker_findings([]) == []  # type: ignore[arg-type]
    assert scheduler_docker_findings({}) == []


def test_unavailable_no_findings() -> None:
    assert scheduler_docker_findings({"available": False, "containers": []}) == []
    assert scheduler_docker_findings({"available": True, "containers": None}) == []  # type: ignore[dict-item]


def test_healthy_running_no_findings() -> None:
    skill = {
        "available": True,
        "containers": [
            {"id": "abc123def456", "name": "svc", "state": "running", "status": "Up 2 hours (healthy)"},
        ],
    }
    assert scheduler_docker_findings(skill) == []


def test_running_unhealthy_finding() -> None:
    skill = {
        "available": True,
        "containers": [
            {
                "id": "deadbeefc0ffee00000000000000000000000000000000000000000000000001",
                "name": "bad-svc",
                "state": "running",
                "status": "Up 1 minute (unhealthy)",
            },
        ],
    }
    assert scheduler_docker_findings(skill) == ["docker_unhealthy:bad-svc:deadbeefc0ff"]


def test_exited_with_unhealthy_text_skipped() -> None:
    skill = {
        "available": True,
        "containers": [
            {"id": "x", "name": "gone", "state": "exited", "status": "Exited (1) ... (unhealthy)"},
        ],
    }
    assert scheduler_docker_findings(skill) == []


def test_missing_state_requires_up_prefix() -> None:
    skill = {
        "available": True,
        "containers": [
            {
                "id": "aaaabbbbccccddddeeeeffff0000111122223333444455556666777788889999",
                "name": "n",
                "status": "Up 5s (unhealthy)",
            },
        ],
    }
    assert scheduler_docker_findings(skill) == ["docker_unhealthy:n:aaaabbbbcccc"]


def test_unnamed_container() -> None:
    skill = {
        "available": True,
        "containers": [
            {
                "id": "1111111111111111111111111111111111111111111111111111111111111111",
                "state": "running",
                "status": "Up (unhealthy)",
            },
        ],
    }
    assert scheduler_docker_findings(skill) == ["docker_unhealthy:unnamed:111111111111"]
