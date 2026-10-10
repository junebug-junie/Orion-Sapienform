"""Every service in every tracked compose file must log to journald.

Docker's default `json-file` driver stores a container's log inside that
container's own directory, so `docker compose up` recreating a container (new
image, changed config, `down`/`up`) deletes its history. That is how the
2026-10-08 reading failure lost its evidence: a fleet restart recreated
orion-hub and the logs went with it. journald keeps the lines in the host
journal, outside the container, so they survive recreation.

How to read them: docs/operations/container-logs.md.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"driver": "journald", "options": {"tag": "{{.Name}}"}}


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates compose-only tags such as `!override`/`!reset`."""


def _construct_any(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _construct_any)


def _compose_files() -> list[Path]:
    return sorted(
        p
        for pattern in ("docker-compose*.yml", "docker-compose*.yaml", "compose*.yml", "compose*.yaml")
        for p in REPO_ROOT.glob(f"services/*/{pattern}")
    )


def test_compose_files_are_found() -> None:
    # Guard against the glob silently matching nothing (which would pass the gate).
    assert len(_compose_files()) >= 50


def test_every_compose_service_logs_to_journald() -> None:
    missing: list[str] = []
    for path in _compose_files():
        doc = yaml.load(path.read_text(), Loader=_ComposeLoader) or {}
        for name, svc in (doc.get("services") or {}).items():
            logging_cfg = (svc or {}).get("logging")
            if logging_cfg != EXPECTED:
                missing.append(f"{path.relative_to(REPO_ROOT)}::{name} -> {logging_cfg!r}")
    assert not missing, (
        "These compose services do not use the journald logging block, so their "
        "logs are deleted when the container is recreated. Add under the service:\n"
        "    logging:\n      driver: journald\n      options:\n        tag: \"{{.Name}}\"\n\n"
        + "\n".join(missing)
    )

