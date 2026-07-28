"""Regression: NOTIFY_SERVICE_URL must use a hostname that actually resolves.

Found live 2026-07-28: NOTIFY_SERVICE_URL defaulted to http://orion-notify:7140 since
this service's inception. That hostname is neither orion-notify's Docker Compose
service key (`notify`) nor its container_name (`${PROJECT}-notify`, e.g.
orion-athena-notify on this host) -- it has never resolved on any host where PROJECT
has a suffix. The 07:30 local digest run failed with NameResolutionError even after
the #1420 (Postgres) and #1423 (scheduler crash) fixes landed, because this was a
third, independent bug in the same delivery path.

The compose *service* name is the only alias Docker's default bridge network attaches
regardless of PROJECT, so that's what this asserts against -- parsed live from
orion-notify's own compose file rather than hardcoded, so this fails loudly if that
service is ever renamed instead of silently going stale.
"""

from __future__ import annotations

from pathlib import Path

import app.settings as digest_settings

NOTIFY_COMPOSE_PATH = (
    Path(__file__).resolve().parents[3] / "services" / "orion-notify" / "docker-compose.yml"
)


def _notify_compose_service_name() -> str:
    """Real YAML parsing rather than a hand-rolled regex, so this doesn't silently
    mis-parse if orion-notify's compose file ever gains a comment/blank line before
    its first service key, or a Compose-specific tag (e.g. `!override`) -- same
    tag-tolerant loader as scripts/check_service_hostname_refs.py."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _construct_undefined(loader: "yaml.SafeLoader", node: "yaml.Node") -> object:
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_scalar(node)

    _Loader.add_constructor(None, _construct_undefined)

    compose = yaml.load(NOTIFY_COMPOSE_PATH.read_text(encoding="utf-8"), Loader=_Loader) or {}
    services = compose.get("services") or {}
    assert len(services) == 1, f"expected exactly one service in {NOTIFY_COMPOSE_PATH}, found {list(services)}"
    return next(iter(services))


def test_notify_service_url_default_uses_resolvable_compose_service_name() -> None:
    service_name = _notify_compose_service_name()
    settings = digest_settings.Settings()
    assert settings.NOTIFY_SERVICE_URL == f"http://{service_name}:7140"
