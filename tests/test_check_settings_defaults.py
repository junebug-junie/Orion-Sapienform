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


# --- --example-drift mode ---------------------------------------------------

_DRIFT_SETTINGS = textwrap.dedent(
    """
    from __future__ import annotations

    from typing import Literal

    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict


    class Settings(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

        daily_cap: int = Field(default=3, alias="FAKE_DAILY_CAP")
        cooldown: float = Field(14400.0, alias="FAKE_COOLDOWN_SEC")
        enabled: bool = Field(default=False, alias="FAKE_ENABLED")
        mode: Literal["enforce", "observe"] = Field("enforce", alias="FAKE_MODE")
        peer_url: str | None = Field(default=None, alias="FAKE_PEER_URL")
        base_url: str = Field(default="http://orion-fake:1", alias="FAKE_BASE_URL")
    """
)

_ALIGNED_EXAMPLE = textwrap.dedent(
    """
    # comment lines are ignored
    FAKE_DAILY_CAP=3
    FAKE_COOLDOWN_SEC=14400
    FAKE_ENABLED=false
    FAKE_MODE="enforce"
    FAKE_PEER_URL=
    FAKE_BASE_URL=http://orion-fake:1
    """
)


def _setup_drift(tmp_path, monkeypatch, settings: str, example: str, compose: str | None = None,
                 host_specific: frozenset[str] = frozenset(), reasoned: dict | None = None) -> None:
    _write_settings(tmp_path, "orion-fake", settings)
    service_dir = tmp_path / "services" / "orion-fake"
    (service_dir / ".env_example").write_text(example, encoding="utf-8")
    if compose is not None:
        (service_dir / "docker-compose.yml").write_text(compose, encoding="utf-8")
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    monkeypatch.setitem(gate._EXAMPLE_DRIFT_SETTINGS_PATHS, "orion-fake", "app/settings.py")
    monkeypatch.setitem(gate.HOST_SPECIFIC_EXAMPLE_KEYS, "orion-fake", host_specific)
    monkeypatch.setitem(gate.REASONED_DRIFT, "orion-fake", reasoned or {})


def test_example_drift_aligned_passes(tmp_path, monkeypatch):
    """Numeric spelling (14400 vs 14400.0), bool case, quoting, Literal fields and
    an empty value against a None default are all equal, not drift."""
    _setup_drift(tmp_path, monkeypatch, _DRIFT_SETTINGS, _ALIGNED_EXAMPLE,
                 compose="environment:\n  - FAKE_ENABLED=${FAKE_ENABLED:-False}\n"
                         "  - FAKE_COOLDOWN_SEC=${FAKE_COOLDOWN_SEC:-14400.0}\n")
    assert gate.main(["orion-fake", "--example-drift"]) == 0


def test_example_drift_settings_default_fails(tmp_path, monkeypatch, capsys):
    """The 2026-10-09 shape: code cap 3, .env_example (and production) 7."""
    example = _ALIGNED_EXAMPLE.replace("FAKE_DAILY_CAP=3", "FAKE_DAILY_CAP=7")
    _setup_drift(tmp_path, monkeypatch, _DRIFT_SETTINGS, example)
    assert gate.main(["orion-fake", "--example-drift"]) == 1
    out = capsys.readouterr().out
    assert "FAKE_DAILY_CAP [settings]" in out and "default=3" in out and "=7" in out


def test_example_drift_compose_fallback_fails(tmp_path, monkeypatch, capsys):
    """A compose `${KEY:-x}` fallback is the effective default when the key is
    missing from .env, so it must match too (GPU_POOL_SHED_ENABLED's shape)."""
    example = _ALIGNED_EXAMPLE.replace("FAKE_ENABLED=false", "FAKE_ENABLED=true")
    settings = _DRIFT_SETTINGS.replace('Field(default=False, alias="FAKE_ENABLED")',
                                       'Field(default=True, alias="FAKE_ENABLED")')
    _setup_drift(tmp_path, monkeypatch, settings, example,
                 compose="environment:\n  - FAKE_ENABLED=${FAKE_ENABLED:-false}\n"
                         "  - OTHER=${OTHER:-${FAKE_ENABLED}}\n")
    assert gate.main(["orion-fake", "--example-drift"]) == 1
    out = capsys.readouterr().out
    assert "FAKE_ENABLED [docker-compose]" in out
    assert "[settings]" not in out


def test_example_drift_host_specific_exemption_passes(tmp_path, monkeypatch):
    example = _ALIGNED_EXAMPLE.replace("http://orion-fake:1", "http://127.0.0.1:1")
    _setup_drift(tmp_path, monkeypatch, _DRIFT_SETTINGS, example,
                 host_specific=frozenset({"FAKE_BASE_URL"}))
    assert gate.main(["orion-fake", "--example-drift"]) == 0


def test_example_drift_stale_exemption_fails(tmp_path, monkeypatch, capsys):
    """An exemption whose key is no longer drifted must be removed, so the
    exemption lists cannot quietly grow into a blanket pass."""
    _setup_drift(tmp_path, monkeypatch, _DRIFT_SETTINGS, _ALIGNED_EXAMPLE,
                 reasoned={"FAKE_DAILY_CAP": "pending a decision"})
    assert gate.main(["orion-fake", "--example-drift"]) == 1
    assert "FAKE_DAILY_CAP" in capsys.readouterr().out


def test_example_drift_unparseable_example_fails(tmp_path, monkeypatch, capsys):
    example = _ALIGNED_EXAMPLE.replace("FAKE_MODE=\"enforce\"", "FAKE_MODE=bogus")
    _setup_drift(tmp_path, monkeypatch, _DRIFT_SETTINGS, example)
    assert gate.main(["orion-fake", "--example-drift"]) == 1
    assert "FAKE_MODE" in capsys.readouterr().out


def test_example_drift_json_and_report_only(tmp_path, monkeypatch, capsys):
    import json

    example = _ALIGNED_EXAMPLE.replace("FAKE_DAILY_CAP=3", "FAKE_DAILY_CAP=7")
    _setup_drift(tmp_path, monkeypatch, _DRIFT_SETTINGS, example)
    assert gate.main(["orion-fake", "--example-drift", "--json", "--report-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["key"] for item in payload["drift"]] == ["FAKE_DAILY_CAP"]
    assert payload["stale_exemptions"] == []


def test_example_drift_unknown_service_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    assert gate.main(["does-not-exist", "--example-drift"]) == 2


@pytest.mark.parametrize("service", ["orion-hub", "orion-gpu-pool"])
def test_real_services_have_no_example_drift(service):
    """Regression: the curiosity caps/cooldown and the gpu-pool shed lever
    drifted from .env_example (found 2026-10-09). Real files must stay aligned."""
    assert gate.main([service, "--example-drift"]) == 0


def test_example_drift_lowercase_unaliased_field_is_compared(tmp_path, monkeypatch, capsys):
    """case_sensitive=False (pydantic-settings' default): `foo_cap: int` reads
    FOO_CAP. It must be compared, not silently skipped."""
    settings = textwrap.dedent(
        """
        from pydantic_settings import BaseSettings


        class Settings(BaseSettings):
            foo_cap: int = 3
        """
    )
    _setup_drift(tmp_path, monkeypatch, settings, "FOO_CAP=7\n")
    assert gate.main(["orion-fake", "--example-drift"]) == 1
    assert "FOO_CAP" in capsys.readouterr().out.upper()


def test_example_drift_env_prefix_and_alias_choices(tmp_path, monkeypatch, capsys):
    settings = textwrap.dedent(
        """
        from pydantic import AliasChoices, Field
        from pydantic_settings import BaseSettings, SettingsConfigDict


        class Settings(BaseSettings):
            model_config = SettingsConfigDict(env_prefix="PFX_")

            cap: int = 3
            other: int = Field(5, validation_alias=AliasChoices("NEW_OTHER", "OLD_OTHER"))
        """
    )
    _setup_drift(tmp_path, monkeypatch, settings, "PFX_CAP=3\nOLD_OTHER=9\n")
    assert gate.main(["orion-fake", "--example-drift"]) == 1
    out = capsys.readouterr().out
    assert "OLD_OTHER" in out and "CAP" not in out.replace("OLD_OTHER", "")


def test_example_drift_default_factory_and_none_vs_empty(tmp_path, monkeypatch):
    settings = textwrap.dedent(
        """
        from pydantic import Field
        from pydantic_settings import BaseSettings


        class Settings(BaseSettings):
            items: list[str] = Field(default_factory=lambda: ["a", "b"], alias="FAKE_ITEMS")
            maybe: str | None = Field(default=None, alias="FAKE_MAYBE")
            flag: bool = Field(default=False, alias="FAKE_FLAG")
        """
    )
    _setup_drift(tmp_path, monkeypatch, settings,
                 "FAKE_ITEMS='[\"a\", \"b\"]'\nFAKE_MAYBE=\nFAKE_FLAG=false # inline note\n")
    assert gate.main(["orion-fake", "--example-drift"]) == 0


def test_example_drift_false_is_not_unset(tmp_path, monkeypatch):
    """None/"" equivalence must not swallow a real False/0 vs empty mismatch."""
    assert gate._same(None, "") and not gate._same(False, None) and not gate._same(0, "")


def test_example_drift_compose_comment_and_duplicate_occurrences(tmp_path, monkeypatch, capsys):
    """A commented-out fallback is ignored; a drifted FIRST occurrence is not
    hidden by a matching later one."""
    compose = (
        "environment:\n"
        "  # - FAKE_DAILY_CAP=${FAKE_DAILY_CAP:-99}\n"
        "  - FAKE_DAILY_CAP=${FAKE_DAILY_CAP:-3}\n"
        "  - FAKE_ENABLED=${FAKE_ENABLED:-true}\n"
        "  - FAKE_ENABLED_AGAIN=${FAKE_ENABLED:-false}\n"
    )
    _setup_drift(tmp_path, monkeypatch, _DRIFT_SETTINGS, _ALIGNED_EXAMPLE, compose=compose)
    assert gate.main(["orion-fake", "--example-drift"]) == 1
    out = capsys.readouterr().out
    assert "FAKE_ENABLED [docker-compose]" in out and "'true'" in out
    assert "FAKE_DAILY_CAP" not in out
