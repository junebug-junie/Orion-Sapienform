"""Regression for a live gap (2026-07-21): FIELD_CHANNEL_ANOMALY_* was added
to .env_example and app/settings.py (PR #1224) but never to
docker-compose.yml's explicit `environment:` list. This service's compose
file does not use a blanket `env_file:` include -- each var must be listed
individually as `- KEY=${KEY:-default}` or it silently never reaches the
container, even when present in the live .env. Confirmed live: the
container ran fine (no crash, no error) with the feature just silently
never turning on.
"""
from __future__ import annotations

from pathlib import Path

import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[1]

_ANOMALY_ENV_KEYS = (
    "FIELD_CHANNEL_ANOMALY_ENABLED",
    "FIELD_CHANNEL_ANOMALY_ENCODER_DIR",
    "FIELD_CHANNEL_ANOMALY_CHECK_INTERVAL_SEC",
    "FIELD_CHANNEL_ANOMALY_THRESHOLD_MULTIPLIER",
    "CHANNEL_FIELD_CHANNEL_ANOMALY_SCORE",
)


def _compose_environment_list() -> list[str]:
    compose = yaml.safe_load((SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["field-digester"]["environment"]


def test_anomaly_env_keys_are_all_passed_through_docker_compose() -> None:
    env_list = _compose_environment_list()
    for key in _ANOMALY_ENV_KEYS:
        assert any(entry.startswith(f"{key}=") for entry in env_list), (
            f"{key} is in .env_example/settings.py but missing from "
            f"docker-compose.yml's environment: list -- it will never reach "
            f"the container even if present in the live .env"
        )


_COMPOSE_ONLY_KEYS = frozenset(
    {
        # Used only for docker-compose's own port mapping (ports:), never
        # passed as a container env var -- see docker-compose.yml.
        "FIELD_DIGESTER_PORT",
    }
)


def test_env_example_and_compose_do_not_drift_for_any_key() -> None:
    """Broader check, not just the anomaly keys: every KEY= line in
    .env_example should have a same-named entry in docker-compose.yml's
    environment: list, since this service's compose file lists vars
    individually rather than using env_file:."""
    env_example_keys = set()
    for line in (SERVICE_ROOT / ".env_example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        env_example_keys.add(line.split("=", 1)[0])

    compose_keys = set()
    for entry in _compose_environment_list():
        key = entry.split("=", 1)[0]
        compose_keys.add(key)

    missing = env_example_keys - compose_keys - _COMPOSE_ONLY_KEYS
    assert not missing, (
        f"Keys in .env_example with no docker-compose.yml environment: "
        f"passthrough (will never reach the container): {sorted(missing)}"
    )
