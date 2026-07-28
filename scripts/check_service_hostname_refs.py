#!/usr/bin/env python3
"""Flag .env_example URLs that reference another service by its directory name
instead of its real Docker Compose service key.

Context: found live 2026-07-28 -- services/orion-notify-digest/.env_example hardcoded
NOTIFY_SERVICE_URL=http://orion-notify:7140 since inception. That hostname has never
resolved on any host where PROJECT (used for container_name: ${PROJECT}-notify) has a
suffix -- e.g. this host's PROJECT=orion-athena gives container_name=orion-athena-notify,
never orion-notify. The daily digest scheduler crashed silently on every send for the
service's entire existence because of this, on top of two other independent bugs in the
same path (#1420 Postgres URI, #1423 scheduler datetime crash).

Docker Compose's default bridge network always attaches a DNS alias equal to the
declared `services:` key (e.g. "notify" for orion-notify), regardless of PROJECT or
container_name -- the same convention every mainstream container orchestrator uses
(Kubernetes Service DNS, ECS Service Discovery, Nomad Consul integration all resolve
by the declared logical service name, never by an instance/container name). That
compose service key is the only portable hostname; this script catches the specific,
now-recurring mistake of using the services/<dirname> directory name instead (found in
5 places across this repo as of 2026-07-28: orion-notify-digest, orion-execution-
dispatch-runtime, orion-security-watcher, orion-world-pulse x2).

Usage:
    python scripts/check_service_hostname_refs.py
    python scripts/check_service_hostname_refs.py --json
    python scripts/check_service_hostname_refs.py --report-only

Exit codes: 0 = no directory-name-as-hostname references found (or --report-only).
            1 = at least one .env_example hardcodes a service directory name as a
                hostname where the real compose service key differs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SERVICES_DIR = _REPO_ROOT / "services"

_URL_HOST_RE = re.compile(r"https?://([a-z0-9][a-z0-9\-]*[a-z0-9]):(\d+)")


class _ComposeTagTolerantLoader:
    """yaml.SafeLoader that ignores Compose-specific tags (e.g. `!override`, `!reset`)
    instead of raising -- this script only needs the `services:` keys, not merge
    semantics, so the underlying scalar/sequence/mapping value is enough."""

    _loader_cls: type | None = None

    @classmethod
    def get(cls) -> type:
        if cls._loader_cls is None:
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
            cls._loader_cls = _Loader
        return cls._loader_cls


def _compose_service_key(service_dir: Path) -> str | None:
    """Resolve service_dir's real compose service key. Single-service files (the norm
    in this repo) are unambiguous. Multi-service files are resolved the same way
    scripts/check_service_env_compose_parity.py's _select_service_block() does: by
    exact dirname match or by the "orion-"-stripped suffix (services/orion-topic-
    foundry's compose key is "topic-foundry", alongside a second unrelated service
    "topic-foundry-chat-corpus-builder" in the same file) -- without this, a
    multi-service compose file was silently treated as unresolvable and completely
    skipped, letting a real violation (orion-notify-digest's TOPIC_FOUNDRY_URL) pass
    the gate uncaught. Confirmed live 2026-07-28 via code review."""
    compose_path = service_dir / "docker-compose.yml"
    if not compose_path.is_file():
        return None
    import yaml

    compose = yaml.load(compose_path.read_text(encoding="utf-8"), Loader=_ComposeTagTolerantLoader.get()) or {}
    services = compose.get("services") or {}
    if not services:
        return None
    if len(services) == 1:
        return next(iter(services))

    dirname = service_dir.name
    suffix = re.sub(r"^orion-", "", dirname)
    candidates = [name for name in services if name in (dirname, suffix)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _service_keys_by_dirname() -> dict[str, str]:
    keys: dict[str, str] = {}
    if not _SERVICES_DIR.is_dir():
        return keys
    for entry in sorted(_SERVICES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        key = _compose_service_key(entry)
        if key:
            keys[entry.name] = key
    return keys


def _find_violations(dirname_to_key: dict[str, str]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for service_dir in sorted(_SERVICES_DIR.iterdir()):
        env_example = service_dir / ".env_example"
        if not env_example.is_file():
            continue
        for lineno, line in enumerate(env_example.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for match in _URL_HOST_RE.finditer(stripped):
                host = match.group(1)
                real_key = dirname_to_key.get(host)
                if real_key is None or real_key == host:
                    continue
                violations.append(
                    {
                        "file": str(env_example.relative_to(_REPO_ROOT)),
                        "line": str(lineno),
                        "hostname_used": host,
                        "real_compose_service_key": real_key,
                    }
                )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="always exit 0, even if violations are found (report but don't gate).",
    )
    args = parser.parse_args(argv)

    dirname_to_key = _service_keys_by_dirname()
    violations = _find_violations(dirname_to_key)

    if args.json:
        print(json.dumps({"violations": violations}))
    elif violations:
        print(
            f"check_service_hostname_refs: {len(violations)} reference(s) use a service "
            f"directory name as a hostname instead of its real compose service key:"
        )
        for v in violations:
            print(
                f"    {v['file']}:{v['line']}: uses \"{v['hostname_used']}\" -- "
                f"real compose service key is \"{v['real_compose_service_key']}\""
            )
    else:
        print("check_service_hostname_refs: OK -- no directory-name-as-hostname references found.")

    if violations and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
