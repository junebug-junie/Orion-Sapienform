from __future__ import annotations

import importlib.util
from pathlib import Path

# Anchored to THIS FILE, not to the process cwd.
#
# Every path below used to be a bare relative Path("services/..."), so the
# whole module only worked when pytest happened to be invoked from the repo
# root. Running `pytest tests` from inside services/orion-sql-writer -- an
# equally normal invocation, and the one CLAUDE.md section 5 implies -- made
# them resolve to services/orion-sql-writer/services/... and raise
# FileNotFoundError. That is why this suite reported a DIFFERENT failure
# count depending on which directory you ran it from (18 vs 19), which makes
# any "known failures" baseline meaningless.
SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_settings_module():
    module_path = SERVICE_ROOT / "app" / "settings.py"
    spec = importlib.util.spec_from_file_location("sql_writer_settings_phase21", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_feedback_wiring_exists_in_settings_env_and_compose() -> None:
    settings_module = _load_settings_module()
    assert "chat.response.feedback.v1" in settings_module.DEFAULT_ROUTE_MAP

    cfg = settings_module.Settings()
    assert "orion:chat:response:feedback" in cfg.sql_writer_subscribe_channels

    env_example = (SERVICE_ROOT / ".env_example").read_text(encoding="utf-8")
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "orion:chat:response:feedback" in env_example
    assert "chat.response.feedback.v1\":\"ChatResponseFeedbackSQL" in env_example
    # NOT `"orion:chat:response:feedback" in compose`: the channel list was deliberately
    # moved out of docker-compose.yml into the env_file (see that file's
    # own comment -- compose ${} substitution strips the JSON quotes).
    # Asserting on the old location tested where a string lives, not
    # whether the service is wired. What actually has to hold is that
    # compose still sources the env carrying it.
    assert _compose_sources_env_file(compose)


def test_feedback_wiring_exists_in_bus_and_schema_registry() -> None:
    channels = (REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8")
    registry = (REPO_ROOT / "orion" / "schemas" / "registry.py").read_text(encoding="utf-8")

    assert "orion:chat:response:feedback" in channels
    assert "chat.response.feedback.v1" in channels
    assert "ChatResponseFeedbackV1" in registry


def test_journal_index_wiring_exists_in_settings_env_compose_bus_and_registry() -> None:
    settings_module = _load_settings_module()
    assert "journal.entry.index.v1" in settings_module.DEFAULT_ROUTE_MAP

    cfg = settings_module.Settings()
    assert "orion:journal:index" in cfg.sql_writer_subscribe_channels

    env_example = (SERVICE_ROOT / ".env_example").read_text(encoding="utf-8")
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    channels = (REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8")
    registry = (REPO_ROOT / "orion" / "schemas" / "registry.py").read_text(encoding="utf-8")

    assert "orion:journal:index" in env_example
    assert "journal.entry.index.v1\":\"JournalEntryIndexSQL" in env_example
    # NOT `"orion:journal:index" in compose`: the channel list was deliberately
    # moved out of docker-compose.yml into the env_file (see that file's
    # own comment -- compose ${} substitution strips the JSON quotes).
    # Asserting on the old location tested where a string lives, not
    # whether the service is wired. What actually has to hold is that
    # compose still sources the env carrying it.
    assert _compose_sources_env_file(compose)
    assert "orion:journal:index" in channels
    assert "journal.entry.index.v1" in channels
    assert "JournalEntryIndexV1" in registry


def test_evidence_index_wiring_exists_in_settings_env_compose_bus_and_registry() -> None:
    settings_module = _load_settings_module()
    assert "evidence.unit.v1" in settings_module.DEFAULT_ROUTE_MAP

    cfg = settings_module.Settings()
    assert "orion:evidence:index:upsert" in cfg.sql_writer_subscribe_channels

    env_example = (SERVICE_ROOT / ".env_example").read_text(encoding="utf-8")
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    channels = (REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8")
    registry = (REPO_ROOT / "orion" / "schemas" / "registry.py").read_text(encoding="utf-8")

    assert "orion:evidence:index:upsert" in env_example
    assert "evidence.unit.v1\":\"EvidenceUnitSQL" in env_example
    # NOT `"orion:evidence:index:upsert" in compose`: the channel list was deliberately
    # moved out of docker-compose.yml into the env_file (see that file's
    # own comment -- compose ${} substitution strips the JSON quotes).
    # Asserting on the old location tested where a string lives, not
    # whether the service is wired. What actually has to hold is that
    # compose still sources the env carrying it.
    assert _compose_sources_env_file(compose)
    assert "orion:evidence:index:upsert" in channels
    assert "evidence.unit.v1" in channels
    assert "EvidenceUnitV1" in registry


def test_markdown_adapter_channel_wiring_exists_in_settings_env_compose_bus_and_registry() -> None:
    settings_module = _load_settings_module()
    cfg = settings_module.Settings()
    assert "orion:evidence:markdown:ingest" in cfg.sql_writer_subscribe_channels

    env_example = (SERVICE_ROOT / ".env_example").read_text(encoding="utf-8")
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    channels = (REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8")
    registry = (REPO_ROOT / "orion" / "schemas" / "registry.py").read_text(encoding="utf-8")

    assert "orion:evidence:markdown:ingest" in env_example
    # NOT `"orion:evidence:markdown:ingest" in compose`: the channel list was deliberately
    # moved out of docker-compose.yml into the env_file (see that file's
    # own comment -- compose ${} substitution strips the JSON quotes).
    # Asserting on the old location tested where a string lives, not
    # whether the service is wired. What actually has to hold is that
    # compose still sources the env carrying it.
    assert _compose_sources_env_file(compose)
    assert "orion:evidence:markdown:ingest" in channels
    assert "document.markdown.spec.v1" in channels
    assert "MarkdownSpecIngestV1" in registry


def test_parsed_document_channel_wiring_exists_in_settings_env_compose_bus_and_registry() -> None:
    settings_module = _load_settings_module()
    cfg = settings_module.Settings()
    assert "orion:evidence:parsed:ingest" in cfg.sql_writer_subscribe_channels

    env_example = (SERVICE_ROOT / ".env_example").read_text(encoding="utf-8")
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    channels = (REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8")
    registry = (REPO_ROOT / "orion" / "schemas" / "registry.py").read_text(encoding="utf-8")

    assert "orion:evidence:parsed:ingest" in env_example
    # NOT `"orion:evidence:parsed:ingest" in compose`: the channel list was deliberately
    # moved out of docker-compose.yml into the env_file (see that file's
    # own comment -- compose ${} substitution strips the JSON quotes).
    # Asserting on the old location tested where a string lives, not
    # whether the service is wired. What actually has to hold is that
    # compose still sources the env carrying it.
    assert _compose_sources_env_file(compose)
    assert "orion:evidence:parsed:ingest" in channels
    assert "document.parsed.v1" in channels
    assert "ParsedDocumentIngestV1" in registry


def _compose_sources_env_file(compose: str) -> bool:
    """True when the **sql-writer service block** declares `env_file: - .env`.

    This is the real wiring guarantee behind every channel assertion in this
    module: the subscribe list reaches the container through that file, so if
    the declaration is dropped the service silently subscribes to nothing --
    which is exactly the failure the deleted `"<channel>" in compose`
    assertions were reaching for, pinned to a location that has since moved.

    Scoped to the sql-writer block on purpose. A first draft scanned the whole
    document for any `env_file:` line, which review showed would return True
    when a SIDECAR service declared one and sql-writer's had been deleted --
    passing while the service subscribed to nothing, the precise breakage it
    guards. It also accepts the inline `env_file: [.env]` form and an entry at
    any position in the list, both of which are valid wiring the draft
    rejected.
    """
    import yaml

    try:
        doc = yaml.safe_load(compose) or {}
    except yaml.YAMLError:
        return False
    service = (doc.get("services") or {}).get("sql-writer")
    if not isinstance(service, dict):
        return False
    declared = service.get("env_file")
    if declared is None:
        return False
    if isinstance(declared, str):
        declared = [declared]
    return any(str(entry).strip() == ".env" for entry in declared)
