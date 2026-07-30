from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_settings_defaults as gate  # noqa: E402


_ALL_DEFAULTS_SETTINGS = textwrap.dedent(
    """
    from pydantic import Field
    from pydantic_settings import BaseSettings


    class Settings(BaseSettings):
        service_name: str = Field("orion-fake", alias="SERVICE_NAME")
        port: int = Field(9999, alias="FAKE_PORT")
        enabled: bool = Field(True, alias="FAKE_ENABLED")

        class Config:
            env_file = ".env"
            extra = "ignore"
            populate_by_name = True
    """
)

_MISSING_DEFAULT_SETTINGS = textwrap.dedent(
    """
    from pydantic import Field
    from pydantic_settings import BaseSettings


    class Settings(BaseSettings):
        service_name: str = Field("orion-fake", alias="SERVICE_NAME")
        # No default -- this is the synthetic bug the gate must catch.
        api_token: str = Field(..., alias="FAKE_API_TOKEN")

        class Config:
            env_file = ".env"
            extra = "ignore"
            populate_by_name = True
    """
)

_MISSING_DEFAULT_WITH_MODULE_LEVEL_SINGLETON_SETTINGS = textwrap.dedent(
    """
    from functools import lru_cache

    from pydantic import Field
    from pydantic_settings import BaseSettings


    class Settings(BaseSettings):
        service_name: str = Field("orion-fake", alias="SERVICE_NAME")
        # No default -- same synthetic bug as _MISSING_DEFAULT_SETTINGS, but this
        # fixture also mirrors services/orion-actions/app/settings.py's real
        # pattern: a trailing module-level singleton instantiation. Importing this
        # module raises a pydantic ValidationError when FAKE_API_TOKEN isn't set in
        # the real environment -- exactly like the live orion-actions file would if
        # a required field were ever introduced there.
        api_token: str = Field(..., alias="FAKE_API_TOKEN")

        class Config:
            env_file = ".env"
            extra = "ignore"
            populate_by_name = True

    @lru_cache(maxsize=1)
    def get_settings() -> Settings:
        return Settings()


    settings = get_settings()
    """
)


def _write_settings(tmp_path: Path, service: str, content: str) -> None:
    settings_path = tmp_path / "services" / service / "app" / "settings.py"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(content, encoding="utf-8")


def test_all_fields_have_defaults_passes(tmp_path, monkeypatch):
    _write_settings(tmp_path, "orion-fake", _ALL_DEFAULTS_SETTINGS)
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._SETTINGS_MODULE_PATHS, "orion-fake", "app/settings.py")
    monkeypatch.setitem(gate.REQUIRED_NO_DEFAULT, "orion-fake", frozenset())

    exit_code = gate.main(["orion-fake"])
    assert exit_code == 0


def test_missing_default_fails(tmp_path, monkeypatch, capsys):
    _write_settings(tmp_path, "orion-fake", _MISSING_DEFAULT_SETTINGS)
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._SETTINGS_MODULE_PATHS, "orion-fake", "app/settings.py")
    monkeypatch.setitem(gate.REQUIRED_NO_DEFAULT, "orion-fake", frozenset())

    exit_code = gate.main(["orion-fake"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "api_token" in out


def test_missing_default_with_module_level_singleton_still_reports_field_name(tmp_path, monkeypatch, capsys):
    """Regression test for the module-level-singleton-instantiation case (the real
    pattern services/orion-actions/app/settings.py uses at its file's end,
    `settings = get_settings()`). Importing a module like this actually
    instantiates Settings() at import time, which raises a ValidationError before
    reaching `_find_missing_defaults`'s model_fields walk if a field genuinely has
    no default and the real env doesn't supply one -- verified live this used to
    surface as a raw pydantic traceback and exit 2 ("could not run the check")
    instead of the clean exit-1 field-name listing this test asserts on."""
    _write_settings(tmp_path, "orion-fake", _MISSING_DEFAULT_WITH_MODULE_LEVEL_SINGLETON_SETTINGS)
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._SETTINGS_MODULE_PATHS, "orion-fake", "app/settings.py")
    monkeypatch.setitem(gate.REQUIRED_NO_DEFAULT, "orion-fake", frozenset())
    monkeypatch.delenv("FAKE_API_TOKEN", raising=False)

    exit_code = gate.main(["orion-fake"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "api_token" in out


def test_missing_default_allowlisted_passes(tmp_path, monkeypatch):
    """A field with no default is fine if it's on the service's explicit
    REQUIRED_NO_DEFAULT allowlist (e.g. a genuine secret that should hard-fail
    on boot rather than silently run with a placeholder)."""
    _write_settings(tmp_path, "orion-fake", _MISSING_DEFAULT_SETTINGS)
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._SETTINGS_MODULE_PATHS, "orion-fake", "app/settings.py")
    monkeypatch.setitem(gate.REQUIRED_NO_DEFAULT, "orion-fake", frozenset({"api_token"}))

    exit_code = gate.main(["orion-fake"])
    assert exit_code == 0


def test_report_only_never_fails(tmp_path, monkeypatch):
    _write_settings(tmp_path, "orion-fake", _MISSING_DEFAULT_SETTINGS)
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._SETTINGS_MODULE_PATHS, "orion-fake", "app/settings.py")
    monkeypatch.setitem(gate.REQUIRED_NO_DEFAULT, "orion-fake", frozenset())

    exit_code = gate.main(["orion-fake", "--report-only"])
    assert exit_code == 0


def test_json_output_shape(tmp_path, monkeypatch, capsys):
    _write_settings(tmp_path, "orion-fake", _MISSING_DEFAULT_SETTINGS)
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._SETTINGS_MODULE_PATHS, "orion-fake", "app/settings.py")
    monkeypatch.setitem(gate.REQUIRED_NO_DEFAULT, "orion-fake", frozenset())

    exit_code = gate.main(["orion-fake", "--json"])
    assert exit_code == 1
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["service"] == "orion-fake"
    assert payload["missing_defaults"] == ["api_token"]


def test_unknown_service_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    exit_code = gate.main(["does-not-exist"])
    assert exit_code == 2


def test_missing_settings_file_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._SETTINGS_MODULE_PATHS, "orion-fake", "app/settings.py")
    exit_code = gate.main(["orion-fake"])
    assert exit_code == 2


def test_real_orion_actions_settings_has_no_missing_defaults():
    """Regression test: the real services/orion-actions/app/settings.py, which
    this whole pilot depends on already having a default for every field --
    that's precisely what let docker-compose.yml's redundant `environment:`
    duplication (with its own set of ${KEY:-default} fallbacks) be deleted in
    favor of `env_file: [.env]` alone."""
    exit_code = gate.main(["orion-actions", "--report-only"])
    assert exit_code == 0
