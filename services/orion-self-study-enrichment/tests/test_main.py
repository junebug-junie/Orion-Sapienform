from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.claude_runner import ClaudeRunResult  # noqa: E402
from app.evidence import EvidenceBundle  # noqa: E402
from app.main import handle_request_payload  # noqa: E402
from app.settings import Settings  # noqa: E402


def _settings(**overrides) -> Settings:
    return Settings(
        ORION_BUS_ENABLED=False,
        SELF_STUDY_ENRICHMENT_REPO_PATH="/repo",
        SELF_STUDY_ENRICHMENT_GRAPH_JSON_PATH="/repo/graphify-out/graph.json",
        SELF_STUDY_ENRICHMENT_CACHE_DIR="/tmp/self_study_enrichment_test_cache",
        SELF_STUDY_ENRICHMENT_RATE_LIMIT_STATE_PATH="/tmp/self_study_enrichment_test_rl.json",
        **overrides,
    )


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        touched_paths=("services/foo/app/x.py",),
        delta_summary={"prev_sha": "a", "head_sha": "b"},
        graph_nodes=({"id": "n1"},),
        nearby_docs=(),
    )


def test_claude_subprocess_env_has_no_api_key_and_sets_claude_config_dir():
    """Regression guard: this service authenticates `claude -p` as the
    host's already-logged-in Claude Code CLI session via CLAUDE_CONFIG_DIR
    pointing at a read-only bind-mounted `.credentials.json` -- never via a
    service-local ANTHROPIC_API_KEY. An earlier version of this patch got
    this wrong; this test exists so that mistake cannot silently come back.
    """
    settings = _settings(SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR="/root/.claude")
    captured: dict[str, dict] = {}

    def _fake_run_claude_once(prompt, **kwargs):
        captured["env"] = kwargs.get("env")
        return ClaudeRunResult(ok=True, text="a real summary", raw_stdout="{}", exit_code=0)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-leaked-from-host-env"}, clear=False), \
        patch("app.main.build_evidence_bundle", return_value=_bundle()), \
        patch("app.main.read_cached", return_value=None), \
        patch("app.main.allow_and_record", return_value=True), \
        patch("app.main.run_claude_once", side_effect=_fake_run_claude_once), \
        patch("app.main.write_cached") as mock_write:
        handle_request_payload(settings, {"touched_paths": ["services/foo/app/x.py"]})

    assert mock_write.called, "expected a successful run to persist a cache entry"
    env = captured["env"]
    assert env is not None
    assert "ANTHROPIC_API_KEY" not in env, (
        "ANTHROPIC_API_KEY must never be passed to the claude -p subprocess "
        "-- this service authenticates via the host's Claude Code session, "
        "not a separate API key billing path"
    )
    assert env["CLAUDE_CONFIG_DIR"] == "/root/.claude"


def test_settings_has_no_anthropic_api_key_field():
    """Regression guard at the settings-contract level: no field named
    ANTHROPIC_API_KEY may exist on Settings. Note this only proves the field
    itself is gone from the schema -- it does not prove an env value named
    ANTHROPIC_API_KEY is inert at runtime (Settings uses `extra="ignore"`,
    so a stray env var would be silently dropped, not surfaced as an
    attribute either way). The real runtime guard against a leaked
    ANTHROPIC_API_KEY reaching the subprocess is
    `test_claude_subprocess_env_has_no_api_key_and_sets_claude_config_dir`
    above, which asserts on the actual subprocess env dict."""
    assert not hasattr(Settings(), "ANTHROPIC_API_KEY")


def test_docker_compose_has_no_anthropic_api_key_and_requires_credentials_host_path():
    """Deterministic check (CLAUDE.md sec 4) on the actual runtime wiring
    surface, not just Python-level settings: docker-compose.yml must not
    reference ANTHROPIC_API_KEY anywhere, and the real credentials mount
    must fail fast (`:?`-guarded) rather than silently defaulting if the
    operator forgets to set the host path."""
    compose_path = SERVICE_ROOT / "docker-compose.yml"
    text = compose_path.read_text()
    assert "ANTHROPIC_API_KEY" not in text
    assert "CLAUDE_CONFIG_DIR" in text
    assert "SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH:?" in text, (
        "the credentials host path bind mount must use compose's ${VAR:?err} "
        "fail-fast form -- an unset path must not silently resolve to an "
        "empty/default bind-mount source for a credential this sensitive"
    )
