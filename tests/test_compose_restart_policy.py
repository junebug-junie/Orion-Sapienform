"""Every long-running service in a base compose file must say how it restarts.

A service with no `restart:` key gets Docker's default ("no"), so it stays
down after a host reboot. That is easy to miss among ~90 services: the
otel collector was found down after the 2026-11-24 planned power outage.

Override files (docker-compose.<variant>.yml) are skipped: they merge into a
base file that carries the policy, or (like hecate's vllm variants) are
started on demand. A service that `extends` another service in the same file inherits its
policy. Services behind `profiles:` are skipped:
they are started on purpose, not at boot. A one-shot job opts out by saying
`restart: "no"` explicitly.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class _Loader(yaml.SafeLoader):
    pass


# Compose-specific tags (!override, !reset): keep the value, drop the tag.
def _untagged(loader, _suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


_Loader.add_multi_constructor("!", _untagged)


def _declares_restart(svc: dict, services: dict) -> bool:
    """True if the service, or a same-file service it `extends`, sets restart."""
    seen = set()
    while isinstance(svc, dict):
        if svc.get("restart") not in (None, ""):
            return True
        ext = svc.get("extends")
        base = ext.get("service") if isinstance(ext, dict) and "file" not in ext else None
        if base is None or base in seen:
            return False
        seen.add(base)
        svc = services.get(base)
    return False


def _missing_restart() -> list[str]:
    missing = []
    for path in sorted(ROOT.glob("services/*/docker-compose.yml")):
        doc = yaml.load(path.read_text(), Loader=_Loader) or {}
        services = doc.get("services") or {}
        for name, svc in services.items():
            if not isinstance(svc, dict) or svc.get("profiles"):
                continue
            if not _declares_restart(svc, services):
                missing.append(f"{path.relative_to(ROOT)}: {name}")
    return missing


def test_every_base_compose_service_declares_restart():
    assert _missing_restart() == [], (
        "These services have no `restart:` key, so they stay down after a reboot. "
        "Add `restart: unless-stopped` (or `restart: \"no\"` for a one-shot job):\n"
        + "\n".join(_missing_restart())
    )
