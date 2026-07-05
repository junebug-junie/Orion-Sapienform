"""FCC env catalog: model labels from ~/.fcc/.env fixture."""
from __future__ import annotations

from pathlib import Path

from scripts.fcc_env_catalog import (
    FCC_MODEL_ENV_KEYS,
    load_fcc_env,
    model_labels_from_env,
    resolve_auth_token,
)


def test_model_labels_only_for_set_keys(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL=llamacpp/chat\n"
        "MODEL_OPUS=\n"
        "MODEL_HAIKU=llamacpp/quick\n"
        "ANTHROPIC_AUTH_TOKEN=fixture-token\n",
        encoding="utf-8",
    )
    env = load_fcc_env(env_file)
    labels = model_labels_from_env(env)
    assert labels == ["MODEL", "MODEL_HAIKU"]
    assert resolve_auth_token(env, override="") == "fixture-token"
    assert resolve_auth_token(env, override="override") == "override"


def test_model_labels_empty_when_no_keys(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n", encoding="utf-8")
    assert model_labels_from_env(load_fcc_env(env_file)) == []


def test_catalog_keys_are_stable() -> None:
    assert "MODEL" in FCC_MODEL_ENV_KEYS
    assert "MODEL_HAIKU" in FCC_MODEL_ENV_KEYS
