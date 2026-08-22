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
from app.main import build_subprocess_env, handle_request_payload  # noqa: E402
from app.settings import Settings  # noqa: E402


def _settings(**overrides) -> Settings:
    # _env_file=None: this service's real .env may hold a real OAuth token
    # once deployed -- tests must never read it (mirrors
    # orion-room-companion's identical test-isolation fix).
    return Settings(
        _env_file=None,
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
    """Regression guard: this service authenticates `claude -p` via a
    claude-setup-token OAuth token (Settings-sourced), never via a
    service-local ANTHROPIC_API_KEY. An earlier version of this patch got
    the ANTHROPIC_API_KEY mistake wrong; this test exists so it cannot
    silently come back.
    """
    settings = _settings(
        SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR="/root/.claude",
        SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN="a-real-oauth-token",
    )
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
        "-- this service authenticates via a claude-setup-token OAuth token, "
        "not a separate API key billing path"
    )
    assert env["CLAUDE_CONFIG_DIR"] == "/root/.claude"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "a-real-oauth-token"


def test_subprocess_env_is_an_allowlist_not_a_denylist():
    """The bug a live-code-review pass caught: an earlier version of this
    fix built the subprocess env as `dict(os.environ)` plus a single-key
    `ANTHROPIC_API_KEY` pop -- a denylist-of-one. That misses
    ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN, which is exactly what
    orion-hub's FCC lane sets (services/orion-hub/scripts/fcc_claude_bridge.py)
    to redirect `claude` at a local gateway -- inheriting either would make
    this service produce fluent-looking summaries that are not from Claude
    at all, the hardest failure here to notice by eye. orion-room-companion
    hit and fixed the identical gap on 2026-08-18; this mirrors that fix
    exactly rather than repeating the mistake.

    Deliberately does NOT include SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN
    in `hostile` here: pydantic-settings reads real env vars as legitimate
    configuration even with `_env_file=None` (that only disables the
    dotenv *file* source), so setting it in os.environ makes `_settings()`
    treat it as the real configured token, not a leak -- correct behavior,
    since that is docker-compose's actual delivery mechanism for this
    setting in production. That "exactly one copy, under the right name"
    property is covered separately by
    test_subprocess_env_has_exactly_one_copy_of_the_configured_token.
    """
    hostile = {
        "ANTHROPIC_API_KEY": "sk-leaked",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082",
        "ANTHROPIC_AUTH_TOKEN": "fcc-token",
        "ANTHROPIC_MODEL": "some-local-model",
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-someone-elses",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_SECRET_ACCESS_KEY": "leaked",
        "GOOGLE_APPLICATION_CREDENTIALS": "/x/creds.json",
        "SOME_FUTURE_REDIRECT_VAR": "whatever",
    }
    with patch.dict(os.environ, hostile, clear=False):
        # No real token configured (SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN
        # deliberately absent from both `hostile` and any override), so the
        # ambient hostile CLAUDE_CODE_OAUTH_TOKEN above has nothing to hide
        # behind -- if it leaked through, this assertion would catch it.
        env = build_subprocess_env(_settings())

    for key in hostile:
        assert key not in env, f"{key} must not reach the claude subprocess"
    # PATH still has to survive or the binary is unfindable.
    assert "PATH" in env
    assert env["CLAUDE_CONFIG_DIR"] == "/root/.claude"


def test_subprocess_env_has_exactly_one_copy_of_the_configured_token():
    """The other half of the allowlist fix: docker-compose.yml has to put
    the real token into this container's own os.environ under
    SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN for Settings to read it at
    all -- the allowlist must stop that ambient copy from ALSO reaching the
    subprocess under its own name, leaving exactly one copy, under the
    literal CLAUDE_CODE_OAUTH_TOKEN name `claude` actually reads. An
    earlier version of this fix (plain `dict(os.environ)`) put the real
    secret in the subprocess env under both names simultaneously."""
    with patch.dict(os.environ, {"SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN": "a-real-token"}, clear=False):
        env = build_subprocess_env(_settings())

    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "a-real-token"
    assert "SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN" not in env


def test_claude_subprocess_env_omits_oauth_token_key_when_unconfigured():
    """No CLAUDE_CODE_OAUTH_TOKEN key at all (not even empty) when the
    setting is unset -- mirrors orion-room-companion's identical guard. An
    empty value could still short-circuit Claude Code's auth precedence
    ahead of a legitimate /login credential."""
    env = build_subprocess_env(_settings(SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR="/root/.claude"))
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_settings_helper_does_not_leak_a_real_dotenv_token(tmp_path, monkeypatch):
    """If this service is ever deployed (a real .env with a real token
    sitting next to the code), `_settings()` in this test module must never
    accidentally read it. Plants a fake-but-real-shaped token in a `.env`
    in a scratch cwd and asserts it never reaches Settings."""
    fake_env = tmp_path / ".env"
    fake_env.write_text("SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN=a-real-looking-deployed-token\n")
    monkeypatch.chdir(tmp_path)
    settings = _settings()
    assert settings.SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN is None


def test_oauth_token_strips_paste_artifacts():
    """A manually pasted `claude setup-token` output commonly picks up
    leading/trailing whitespace; the settings validator must strip it."""
    settings = _settings(SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN="  a-token-with-whitespace  \n")
    assert settings.SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN.get_secret_value() == "a-token-with-whitespace"


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


def test_docker_compose_has_no_anthropic_api_key_and_requires_oauth_token():
    """Deterministic check (CLAUDE.md sec 4) on the actual runtime wiring
    surface, not just Python-level settings: docker-compose.yml must not
    reference ANTHROPIC_API_KEY anywhere, and the OAuth token must fail
    fast (`:?`-guarded) rather than silently starting with no auth at all
    if the operator forgets to set it."""
    compose_path = SERVICE_ROOT / "docker-compose.yml"
    text = compose_path.read_text()
    assert "ANTHROPIC_API_KEY" not in text
    assert "CLAUDE_CONFIG_DIR" in text
    assert "SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN:?" in text, (
        "the OAuth token env var must use compose's ${VAR:?err} fail-fast "
        "form -- an unset token must not silently start the container with "
        "no credential at all"
    )
    assert "SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH" not in text, (
        "the file-mount credential pattern was retired 2026-08-21 in favor "
        "of the OAuth token -- this key should not reappear in compose"
    )
    assert ".credentials.json" not in text
