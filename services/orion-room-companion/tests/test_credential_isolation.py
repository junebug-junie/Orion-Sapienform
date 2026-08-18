"""Regression guards for the reason this service exists as its own container.

These guard DEPTH, not a wall -- an earlier version of this docstring
overclaimed and is corrected here.

The credential is not stored in orion-hub, which runs as root with readable
SSH keys, a gh token and the docker socket, and where Orion's own FCC turns
run with full Bash. That removes the default read path and the accident
surface (repo tree, backups, graphify index, `docker inspect` on Hub).

It does NOT make the credential unreachable from Hub: Hub mounts
/var/run/docker.sock read-write and ships the docker CLI, so
`docker inspect orion-athena-room-companion --format '{{json .Config.Env}}'`
reads the configured token straight out of this container's config -- no
`exec` even required. True of the file-mounted credential this service used
before 2026-08-18 too (confirmed live 2026-08-14, via `docker exec ... cat`).
Docker-socket access is root-equivalent, and nothing in THIS service's config
can close it. The v2 budget cap must therefore not be built on the premise
that owning the credential enforces it.

What these tests do enforce: this container stays worth nothing to break into
on its own, and no redirection vector reaches the subprocess env.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from app.main import build_subprocess_env
from app.settings import Settings

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"
ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env_example"


def _settings(**overrides) -> Settings:
    base = {"ROOM_COMPANION_CLAUDE_CONFIG_DIR": "/root/.claude"}
    base.update(overrides)
    return Settings(**base)


def test_subprocess_env_strips_api_key_and_fcc_gateway_vars():
    """The three pops in build_subprocess_env, each for a distinct failure.

    ANTHROPIC_API_KEY would open a second pay-per-token billing relationship
    instead of reusing the subscription. ANTHROPIC_BASE_URL and
    ANTHROPIC_AUTH_TOKEN are what orion-hub's FCC lane sets
    (fcc_claude_bridge.py) to point `claude` at a LOCAL gateway -- inheriting
    either would make the room produce fluent text that is not Claude at all,
    the hardest failure here to notice by eye.
    """
    leaked = {
        "ANTHROPIC_API_KEY": "sk-leaked-from-host-env",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082",
        "ANTHROPIC_AUTH_TOKEN": "fcc-gateway-token",
    }
    with patch.dict(os.environ, leaked, clear=False):
        env = build_subprocess_env(_settings())

    for key in leaked:
        assert key not in env, (
            f"{key} must never reach the claude subprocess -- see this module's docstring"
        )
    assert env["CLAUDE_CONFIG_DIR"] == "/root/.claude"


def test_settings_has_no_api_key_field():
    """Settings must not even offer an API-key knob to turn on."""
    assert not hasattr(Settings(), "ANTHROPIC_API_KEY")


def test_compose_requires_oauth_token_and_has_no_api_key():
    """Deterministic check on the real runtime wiring, not just Python settings.

    Asserts on the PARSED environment list rather than the raw file text:
    both this compose file and .env_example discuss ANTHROPIC_API_KEY at
    length in comments explaining why it is forbidden, and a substring check
    over raw text would fire on the explanation itself -- a gate that fails
    on its own documentation teaches people to delete the documentation.
    """
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    environment = compose["services"]["room-companion"]["environment"]
    keys = {entry.split("=", 1)[0] for entry in environment}
    assert "ANTHROPIC_API_KEY" not in keys
    assert not any(k.startswith("ANTHROPIC_") for k in keys), (
        f"no ANTHROPIC_* var belongs in this service's env: {sorted(keys)}"
    )
    # The real Claude-Code-recognized name must never appear bare in compose --
    # only the ROOM_COMPANION_-prefixed one. build_subprocess_env() does the
    # translation explicitly in code; if the bare name ever lands in compose
    # too, main.py's own _ENV_DENY_PREFIXES would strip it right back out of
    # this service's os.environ (silently, since it's a deny-prefix not a
    # loud failure), which is a confusing way to discover a compose typo.
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys, (
        "set ROOM_COMPANION_CLAUDE_OAUTH_TOKEN, not the bare Claude Code name"
    )

    raw = COMPOSE.read_text(encoding="utf-8")
    # `:?` makes a missing token fail loudly instead of silently starting
    # with no credential and leaving the room mysteriously unauthenticated.
    assert "ROOM_COMPANION_CLAUDE_OAUTH_TOKEN:?" in raw


def test_compose_has_no_credentials_file_mount():
    """2026-08-18: the `.credentials.json` single-file bind mount is retired
    outright (see README's 'FIXED 2026-08-18' section) -- its ~7.5h staleness
    bug must not come back via a re-added mount, partial or otherwise."""
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    volumes = compose["services"]["room-companion"]["volumes"]
    cred_mounts = [v for v in volumes if ".credentials.json" in v]
    assert not cred_mounts, f"credential file mount must not exist any more, got {cred_mounts}"


@pytest.mark.parametrize(
    "forbidden,why",
    [
        ("/var/run/docker.sock", "docker socket is container escape"),
        ("/.ssh", "SSH keys must not be reachable from a room turn"),
        ("/.config/gh", "gh token must not be reachable from a room turn"),
        ("/repo", "no repo mount -- a room conversation has no business reading this codebase"),
    ],
)
def test_compose_does_not_mount_high_value_targets(forbidden: str, why: str):
    """This container is the one place the Claude credential lives, so it must
    not also be a place worth breaking into. orion-hub mounts all of these,
    which is precisely why the credential is not there."""
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    volumes = compose["services"]["room-companion"]["volumes"]
    for volume in volumes:
        assert forbidden not in volume, f"{volume!r} must not be mounted: {why}"


def test_compose_does_not_use_host_networking():
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = compose["services"]["room-companion"]
    assert service.get("network_mode") != "host"


def _env_example_assignments() -> dict[str, str]:
    """Active KEY=VALUE lines only -- comments are prose, not configuration."""
    out: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def test_env_example_carries_no_real_secret():
    """.env_example is the committed operator contract: host PATHS only, never
    a credential value."""
    assignments = _env_example_assignments()
    assert "ANTHROPIC_API_KEY" not in assignments
    assert not any(k.startswith("ANTHROPIC_") for k in assignments)
    # A real credential in the committed template, in any form.
    raw = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "sk-ant-" not in raw, "a real token must never land in .env_example"
    assert '"claudeAiOauth"' not in raw


def test_env_example_oauth_token_key_is_present_and_empty():
    """The operator contract documents the key and how to fill it in
    themselves (`claude setup-token`, pasted straight into their own local
    .env); the secret itself never enters the repo."""
    assignments = _env_example_assignments()
    assert "ROOM_COMPANION_CLAUDE_OAUTH_TOKEN" in assignments
    assert assignments["ROOM_COMPANION_CLAUDE_OAUTH_TOKEN"] == "", (
        "committed .env_example must ship this key empty, never a real token"
    )


def test_setting_sources_excludes_project_scope():
    """`user` only, not `user,local`: a room conversation must not inherit this
    repo's AGENTS.md development contract or its project hooks as room
    context. Claude is a participant here, not a coding agent on this repo."""
    assert Settings().ROOM_COMPANION_SETTING_SOURCES == "user"


def test_subprocess_env_is_an_allowlist_not_a_denylist():
    """A denylist has to enumerate every redirection vector, and the first
    draft missed CLAUDE_CODE_OAUTH_TOKEN -- the one variable proven to bypass
    the mounted credential entirely."""
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
        env = build_subprocess_env(_settings())

    for key in hostile:
        assert key not in env, f"{key} must not reach the claude subprocess"
    # PATH still has to survive or the binary is unfindable.
    assert "PATH" in env
    assert env["CLAUDE_CONFIG_DIR"] == "/root/.claude"


def test_compose_persists_the_claude_session_store():
    """The room->uuid map and the sessions it points at must survive a rebuild
    together, or the first turn in every room after `up -d --build` resumes a
    session that no longer exists."""
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = compose["services"]["room-companion"]
    volumes = service["volumes"]
    config_dir_mounts = [v for v in volumes if v.startswith("room_companion_claude:")]
    assert config_dir_mounts, f"CLAUDE_CONFIG_DIR must be on a named volume, got {volumes}"


def test_subprocess_env_carries_the_deliberately_configured_oauth_token():
    """The one deliberate exception to the allowlist: a token that arrives
    through Settings (i.e. from this service's own .env, translated by
    docker-compose's ROOM_COMPANION_-prefixed key) reaches the subprocess
    under Claude Code's real expected name."""
    env = build_subprocess_env(_settings(ROOM_COMPANION_CLAUDE_OAUTH_TOKEN="sk-ant-oat01-real-token"))
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-real-token"


def test_subprocess_env_omits_oauth_token_key_when_unconfigured():
    """No token configured must mean no key at all, not an empty-string one --
    an empty CLAUDE_CODE_OAUTH_TOKEN could still short-circuit Claude Code's
    auth precedence ahead of a legitimate /login credential."""
    env = build_subprocess_env(_settings())
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
