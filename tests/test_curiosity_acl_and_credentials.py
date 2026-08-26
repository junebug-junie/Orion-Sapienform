"""The two grants that make a curiosity turn possible, and their failure modes.

Both exist because a capability that is configured but not present is the
silent-failure shape this repo has hit repeatedly. The ACL does not survive a
FalkorDB restart, and the credentials do not reach the sandbox on their own --
so both are asserted rather than assumed, and both report their own absence.
"""

from __future__ import annotations

import pytest

from orion.curiosity.acl import acl_setuser_argv, assert_orion_acl
from orion.curiosity.sandbox_env import (
    CURIOSITY_ENV_KEYS,
    inject_curiosity_credentials,
)


def _argv(**over):
    kwargs = dict(
        username="orion_curiosity",
        password="s3cret",
        atlas_graph="orion_substrate",
        own_graph="orion_worldview",
    )
    kwargs.update(over)
    return acl_setuser_argv(**kwargs)


# --- the grant --------------------------------------------------------------


def test_the_atlas_grant_is_read_only_and_the_own_graph_grant_is_not() -> None:
    argv = _argv()
    assert "~orion_substrate" in argv
    assert "+graph.ro_query" in argv
    assert "(~orion_worldview +graph.query +graph.ro_query)" in argv
    # No write command anywhere OUTSIDE the selector's parentheses.
    base = [a for a in argv if not a.startswith("(")]
    assert "+graph.query" not in base


def test_nothing_else_is_granted() -> None:
    argv = _argv()
    assert "nocommands" in argv
    assert not any(a in {"+@all", "~*", "allcommands", "allkeys"} for a in argv)


def test_clearselectors_is_present_because_the_other_resets_do_not_cover_selectors() -> None:
    """MEASURED LIVE 2026-08-26 against this FalkorDB, not assumed: replaying
    the argv WITHOUT `clearselectors` appended a second identical
    `(~orion_worldview ...)` selector, and a third replay a third -- one more
    per Hub start, growing forever. `resetkeys`/`resetchannels`/`nocommands` do
    not touch selectors."""
    argv = _argv()
    assert "clearselectors" in argv
    for reset in ("resetkeys", "resetchannels", "nocommands"):
        assert reset in argv
    # And it must come before the selector it is meant to clear.
    assert argv.index("clearselectors") < argv.index(
        "(~orion_worldview +graph.query +graph.ro_query)"
    )


def test_the_password_is_one_argv_element_never_a_shell_string() -> None:
    argv = _argv(password="a b'c\"d")
    assert ">a b'c\"d" in argv


@pytest.mark.parametrize(
    "field", ["username", "password", "atlas_graph", "own_graph"]
)
def test_a_missing_piece_refuses_rather_than_granting_something_partial(field) -> None:
    with pytest.raises(ValueError):
        _argv(**{field: ""})


def test_a_misconfigured_grant_is_reported_not_raised() -> None:
    """Hub must still start. The loop blocks on the returned reason instead."""
    reason = assert_orion_acl(
        client=object(), username="", password="p", atlas_graph="a", own_graph="o"
    )
    assert reason and "misconfigured" in reason


def test_a_falkordb_that_is_down_is_reported_not_raised() -> None:
    class _Client:
        def execute_command(self, *argv):
            raise OSError("connection refused")

    reason = assert_orion_acl(
        client=_Client(),
        username="u",
        password="p",
        atlas_graph="a",
        own_graph="o",
    )
    assert reason and "OSError" in reason


def test_a_successful_assert_returns_none() -> None:
    class _Client:
        def __init__(self):
            self.calls = []

        def execute_command(self, *argv):
            self.calls.append(argv)

    client = _Client()
    assert (
        assert_orion_acl(
            client=client,
            username="u",
            password="p",
            atlas_graph="a",
            own_graph="o",
        )
        is None
    )
    assert client.calls[0][:2] == ("ACL", "SETUSER")


# --- the credentials --------------------------------------------------------


def test_only_the_allowlisted_keys_are_copied() -> None:
    """~/.fcc/.env also holds provider API keys, a Cloudflare token and a
    GitHub PAT. Handing those to every turn to solve a problem about four
    Postgres tables is not a trade worth making."""
    env: dict[str, str] = {}
    missing = inject_curiosity_credentials(
        env,
        {
            "ORION_CURIOSITY_PG_DSN": "postgresql://ro@db/x",
            "GITHUB_PAT": "ghp_secret",
            "CLOUDFLARE_API_TOKEN": "cf_secret",
            "NVIDIA_NIM_API_KEY": "nv_secret",
        },
    )
    assert env == {"ORION_CURIOSITY_PG_DSN": "postgresql://ro@db/x"}
    assert "GITHUB_PAT" not in env
    assert set(missing) == set(CURIOSITY_ENV_KEYS) - {"ORION_CURIOSITY_PG_DSN"}


def test_absent_keys_are_reported_so_the_gap_is_not_discovered_inside_a_turn() -> None:
    missing = inject_curiosity_credentials({}, {})
    assert set(missing) == set(CURIOSITY_ENV_KEYS)


def test_a_blank_value_counts_as_absent() -> None:
    """An empty DSN would otherwise be exported and fail at connect time,
    inside the turn, as prose."""
    missing = inject_curiosity_credentials({}, {"ORION_CURIOSITY_PG_DSN": "   "})
    assert "ORION_CURIOSITY_PG_DSN" in missing


def test_an_already_set_value_wins_over_the_file() -> None:
    env = {"ORION_CURIOSITY_PG_DSN": "from-container"}
    inject_curiosity_credentials(env, {"ORION_CURIOSITY_PG_DSN": "from-file"})
    assert env["ORION_CURIOSITY_PG_DSN"] == "from-container"


def test_calling_it_twice_is_safe() -> None:
    env: dict[str, str] = {}
    fcc = {k: f"v-{k}" for k in CURIOSITY_ENV_KEYS}
    inject_curiosity_credentials(env, fcc)
    before = dict(env)
    assert inject_curiosity_credentials(env, fcc) == []
    assert env == before


def test_removing_the_keys_from_the_file_is_the_kill_switch() -> None:
    """Stated as a test because it is the ONLY kill switch: a flag would have
    to be added to the harness service's explicit compose `environment:`
    allowlist to reach the container at all, which is exactly how a kill switch
    ends up configured everywhere and present nowhere."""
    env: dict[str, str] = {}
    inject_curiosity_credentials(env, {"MODEL": "llamacpp/harness"})
    assert env == {}


def test_the_allowlist_is_exactly_the_seven_keys_placed_in_the_fcc_env() -> None:
    assert set(CURIOSITY_ENV_KEYS) == {
        "ORION_CURIOSITY_PG_DSN",
        "ORION_CURIOSITY_GRAPH_HOST",
        "ORION_CURIOSITY_GRAPH_PORT",
        "ORION_CURIOSITY_GRAPH_USER",
        "ORION_CURIOSITY_GRAPH_PASSWORD",
        "ORION_CURIOSITY_GRAPH_OWN",
        "ORION_CURIOSITY_GRAPH_ATLAS",
    }


# --- the motor seam ---------------------------------------------------------


def test_the_fcc_subprocess_env_carries_the_credentials(monkeypatch) -> None:
    """`_build_subprocess_env` is `os.environ.copy()` -- the harness CONTAINER's
    environment, which has never carried these (measured live 2026-08-26 inside
    orion-athena-harness-governor: 0 matches for ORION_CURIOSITY). Without this
    wiring the credentials sit in a mounted file the prompt would have to teach
    Orion to parse by hand.
    """
    from orion.harness import fcc_motor

    monkeypatch.setattr(fcc_motor.os, "environ", {"PATH": "/usr/bin"})
    env = fcc_motor._build_subprocess_env(
        fcc_server_url="http://x:8082",
        auth_token="t",
        fcc_env={
            "ORION_CURIOSITY_PG_DSN": "postgresql://ro@db/x",
            "GITHUB_PAT": "ghp_secret",
        },
    )
    assert env["ORION_CURIOSITY_PG_DSN"] == "postgresql://ro@db/x"
    assert "GITHUB_PAT" not in env


def test_a_turn_with_no_fcc_env_is_unaffected(monkeypatch) -> None:
    """Every other FCC turn goes through this same function."""
    from orion.harness import fcc_motor

    monkeypatch.setattr(fcc_motor.os, "environ", {"PATH": "/usr/bin"})
    env = fcc_motor._build_subprocess_env(fcc_server_url="http://x:8082", auth_token="t")
    assert not any(k.startswith("ORION_CURIOSITY") for k in env)


# --- review findings --------------------------------------------------------


def test_a_blank_existing_env_value_does_not_shadow_the_real_credential() -> None:
    """`env.setdefault` treats "" as set. A compose `environment:` entry naming
    the key with no value -- exactly how these get added, and exactly the shape
    of this repo's own absent-kill-switch incident -- would then hand the
    subprocess an empty DSN AND suppress the missing-credential warning."""
    env = {"ORION_CURIOSITY_PG_DSN": ""}
    inject_curiosity_credentials(env, {"ORION_CURIOSITY_PG_DSN": "postgresql://ro@db/x"})
    assert env["ORION_CURIOSITY_PG_DSN"] == "postgresql://ro@db/x"

    env = {"ORION_CURIOSITY_PG_DSN": "   "}
    inject_curiosity_credentials(env, {"ORION_CURIOSITY_PG_DSN": "postgresql://ro@db/x"})
    assert env["ORION_CURIOSITY_PG_DSN"] == "postgresql://ro@db/x"


def test_the_graph_is_materialised_before_the_grant_is_applied() -> None:
    """Verified live 2026-08-26: `GRAPH.RO_QUERY <unknown-graph>` answers
    `ERR Invalid graph operation on empty key`, NOT an empty result. Without an
    idempotent create, a fresh deployment deadlocks: the graph reads as
    unavailable, so the prompt drops the schema section, so Orion never writes
    a node, so the graph is never created."""
    from orion.curiosity.acl import ensure_graph_exists

    class _Client:
        def __init__(self):
            self.calls = []

        def execute_command(self, *argv):
            self.calls.append(argv)

    client = _Client()
    assert ensure_graph_exists(client=client, graph_name="orion_worldview") is None
    assert client.calls == [("GRAPH.QUERY", "orion_worldview", "RETURN 1")]


def test_a_graph_create_failure_is_reported_not_raised() -> None:
    from orion.curiosity.acl import ensure_graph_exists

    class _Client:
        def execute_command(self, *argv):
            raise OSError("connection refused")

    assert "OSError" in (ensure_graph_exists(client=_Client(), graph_name="g") or "")
    assert "misconfigured" in (ensure_graph_exists(client=_Client(), graph_name="") or "")
