from __future__ import annotations

from app.remediator import build_compose_build_command, build_compose_command, build_compose_up_command
from app.roster import ProbeConfig, ProbeMode, RosterEntry


def _entry() -> RosterEntry:
    return RosterEntry(
        id="landing-pad",
        heartbeat_name="landing-pad",
        compose_dir="orion-landing-pad",
        compose_service="orion-landing-pad",
        include_bus_env=False,
        auto_remediate=True,
        probe=ProbeConfig(mode=ProbeMode.redis_and_http, intake_channels=["orion:pad:rpc:request"]),
    )


def test_tier1_force_recreate_command() -> None:
    cmd = build_compose_command(_entry(), repo_root="/repo", tier=1)
    assert cmd[:4] == ["docker", "compose", "--env-file", "/repo/.env"]
    assert "/repo/services/orion-landing-pad/.env" in cmd
    assert cmd[-4:] == ["up", "-d", "--force-recreate", "orion-landing-pad"]


def test_tier2_build_and_up_commands() -> None:
    build_cmd = build_compose_build_command(_entry(), repo_root="/repo")
    up_cmd = build_compose_up_command(_entry(), repo_root="/repo")
    assert "build" in build_cmd
    assert up_cmd[-3:] == ["up", "-d", "orion-landing-pad"]
