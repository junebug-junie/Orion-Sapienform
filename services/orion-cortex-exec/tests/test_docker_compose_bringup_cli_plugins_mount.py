"""Deterministic gate for a real live incident (2026-08-12): docker-compose.yml bind-mounted the
docker CLI binary for skills.docker.compose_service_bringup.v1 but not its CLI-plugins directory.
`docker compose` (v2) is implemented as a plugin, not a built-in subcommand -- without the plugins
directory mounted, `docker compose ...` inside the container fails ("unknown flag: --env-file" /
top-level `docker [OPTIONS] COMMAND` usage text, because the CLI treats "compose" as an unrecognized
command and chokes on the next flag it doesn't understand). Confirmed live by comparing
`docker compose version` (works on host) against `docker exec orion-athena-cortex-exec docker compose
version` (failed: "unknown command: docker compose") before this mount was added.

This asserts the mount is present in the compose file so a future edit can't silently drop it again
without a docker deploy to notice.
"""

from pathlib import Path

COMPOSE_PATH = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def _cortex_exec_volumes() -> list[str]:
    # Plain text scan, not yaml.safe_load: this compose file uses compose-spec extension tags
    # (e.g. `!override`) that plain PyYAML has no constructor for.
    lines = COMPOSE_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "volumes:")
    volumes: list[str] = []
    for ln in lines[start + 1 :]:
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        if indent < 6:  # back out to a sibling key of "volumes:" (4-space indent) or less
            break
        stripped = ln.strip()
        if stripped.startswith("-"):
            volumes.append(stripped)
    assert volumes, "cortex-exec service has no volumes list"
    return volumes


def test_docker_cli_binary_and_plugins_dir_are_both_mounted():
    volumes = _cortex_exec_volumes()
    joined = "\n".join(volumes)
    assert "ORION_HOST_DOCKER_CLI}" in joined or "ORION_HOST_DOCKER_CLI:" in joined, (
        "docker CLI binary bind-mount is missing from cortex-exec volumes"
    )
    assert "ORION_HOST_DOCKER_CLI_PLUGINS_DIR" in joined, (
        "docker CLI plugins-dir bind-mount is missing -- `docker compose` will fail inside the "
        "container as an unrecognized command even though the docker binary itself is mounted."
    )


def test_docker_cli_plugins_mount_is_read_only():
    volumes = _cortex_exec_volumes()
    plugin_line = next((v for v in volumes if "ORION_HOST_DOCKER_CLI_PLUGINS_DIR" in v), None)
    assert plugin_line is not None
    assert plugin_line.strip().endswith(":ro"), (
        f"docker CLI plugins mount must stay read-only, got: {plugin_line!r}"
    )


def test_docker_cli_plugins_mount_container_side_is_fixed_not_overridable():
    # docker's plugin search list (~/.docker/cli-plugins, /usr/local/lib(exec)/docker/cli-plugins,
    # /usr/lib/docker/cli-plugins, /usr/libexec/docker/cli-plugins) is hardcoded in the CLI itself, not
    # env-configurable. If the container-side target followed ORION_HOST_DOCKER_CLI_PLUGINS_DIR too (like
    # the host side does), an operator override to a nonstandard host path would relocate the container
    # mount target right along with it -- and the containerized CLI would never look there, silently
    # reproducing the "unknown command: docker compose" incident this mount exists to fix. Mirrors
    # ORION_HOST_DOCKER_CLI just above it, whose container side is hardcoded to /usr/bin/docker.
    volumes = _cortex_exec_volumes()
    plugin_line = next((v for v in volumes if "ORION_HOST_DOCKER_CLI_PLUGINS_DIR" in v), None)
    assert plugin_line is not None
    target = plugin_line.split(":")[-2]
    assert target == "/usr/libexec/docker/cli-plugins", (
        f"container-side plugin mount target must stay fixed to a path docker's CLI actually scans, "
        f"got: {target!r} (source line: {plugin_line!r})"
    )
    assert "${ORION_HOST_DOCKER_CLI_PLUGINS_DIR" not in target, (
        "container-side target must not be driven by the host-side override var"
    )
