"""Settings <-> .env_example parity (CLAUDE.md §7: "the checked-in
.env_example is the operator contract"). Every Settings field must appear as
a key in .env_example, and vice versa, so an operator copying .env_example
never gets a silently-defaulted value they didn't mean to accept.
"""

from __future__ import annotations

from pathlib import Path

from app.settings import Settings

SERVICE_DIR = Path(__file__).resolve().parent.parent
ENV_EXAMPLE_PATH = SERVICE_DIR / ".env_example"


def _parse_env_example_keys() -> set[str]:
    keys = set()
    for line in ENV_EXAMPLE_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        keys.add(key)
    return keys


def test_env_example_file_exists():
    assert ENV_EXAMPLE_PATH.exists(), f"missing {ENV_EXAMPLE_PATH}"


# SERVICE_NAME/SERVICE_VERSION are fixed identity constants, not meant to be
# operator-overridden (same convention as orion-vision-host's .env_example,
# which also omits them) -- excluded from the "must appear in .env_example"
# direction only.
_NOT_OPERATOR_CONFIGURABLE = {"SERVICE_NAME", "SERVICE_VERSION"}


def test_every_settings_field_is_documented_in_env_example():
    env_keys = _parse_env_example_keys()
    settings_fields = set(Settings.model_fields.keys()) - _NOT_OPERATOR_CONFIGURABLE
    missing = sorted(settings_fields - env_keys)
    assert not missing, f"Settings fields missing from .env_example: {missing}"


def test_every_env_example_key_maps_to_a_settings_field_or_known_extra():
    """extra='ignore' in Settings.model_config means an unmapped .env_example
    key would silently do nothing -- catch that here rather than at runtime.
    A small allow-list covers keys that are real infra config but not typed
    Settings fields (none expected for this service; keep empty unless a
    real exception is documented)."""
    env_keys = _parse_env_example_keys()
    settings_fields = set(Settings.model_fields.keys())
    # HOST_PORT is docker-compose's host-side port mapping (same convention
    # as orion-vision-host/.env_example) -- read by docker-compose.yml, not
    # by app/settings.py (the container's internal port is fixed in code).
    allowed_extra: set[str] = {"HOST_PORT"}
    unmapped = sorted(env_keys - settings_fields - allowed_extra)
    assert not unmapped, f".env_example keys with no matching Settings field: {unmapped}"


def test_settings_load_with_defaults():
    s = Settings()
    assert s.SERVICE_NAME == "world-model"
    assert s.CHANNEL_WORLDMODEL_INTAKE == "orion:exec:request:WorldModelService"
    assert s.CHANNEL_WORLDMODEL_PUB == "orion:worldmodel:predictions"
    assert s.state_dim == s.WM_FUSION_DIM
    assert s.devices == ["cuda:0"]
