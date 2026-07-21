"""Regression for a live gap (2026-07-21): telemetry-anomaly trigger keys
were added to .env_example and app/settings.py (PR #1224) but never to
docker-compose.yml's explicit `environment:` list. This service's compose
file does not use a blanket `env_file:` include -- each var must be listed
individually as `- KEY=${KEY:-default}` or it silently never reaches the
container, even when present in the live .env.
"""
from __future__ import annotations

from pathlib import Path

import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[1]

_ANOMALY_ENV_KEYS = (
    "CHANNEL_FIELD_CHANNEL_ANOMALY_SCORE",
    "EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE",
    "EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_THRESHOLD_MULTIPLIER",
)

# Keys present in .env_example that are intentionally NOT container env vars
# (compose-only interpolation, host-side config, etc). Empty for now -- kept
# so a future genuine exception has an obvious place to go rather than
# weakening the assertion below inline.
_COMPOSE_ONLY_KEYS: frozenset[str] = frozenset()


def _compose_environment_list() -> list[str]:
    compose = yaml.safe_load((SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["equilibrium-service"]["environment"]


def test_anomaly_env_keys_are_all_passed_through_docker_compose() -> None:
    env_list = _compose_environment_list()
    for key in _ANOMALY_ENV_KEYS:
        assert any(entry.startswith(f"{key}=") for entry in env_list), (
            f"{key} is in .env_example/settings.py but missing from "
            f"docker-compose.yml's environment: list -- it will never reach "
            f"the container even if present in the live .env"
        )


def test_env_example_and_compose_do_not_drift_for_any_key() -> None:
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
