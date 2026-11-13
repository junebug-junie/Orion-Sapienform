"""What exists in one checkout: services, modules, channels, verbs, schemas.

The code half of self-study's inventory, with the repo root as a parameter, so the same scan
reads today's main (``services/orion-cortex-exec/app/self_study.py`` wraps these records into
``self_knowledge_items``) and any past snapshot (``python -m orion.dig inventory <date>``). A
folder a snapshot predates yields ``[]``, not an error.

Each record: ``{"name", "source_path" (relative to root, posix), "metadata"}``.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

VERB_DECORATOR_RE = re.compile(r'@verb\([\"\']([^\"\']+)[\"\']\)')
REGISTRY_KEY_RE = re.compile(r'^\s+\"([^\"]+)\":', re.MULTILINE)
CATEGORIES = ("service", "module", "channel", "verb", "schema")


def _absent(path: Path, strict: bool) -> bool:
    if path.exists():
        return False
    if strict:
        raise FileNotFoundError(f"self inventory: {path} does not exist")
    return True


def _rec(root: Path, name: str, path: Path, metadata: dict | None = None) -> dict:
    return {"name": name, "source_path": path.relative_to(root).as_posix(), "metadata": metadata or {}}


def services(root: Path, *, strict: bool = False) -> list[dict]:
    services_dir = root / "services"
    if _absent(services_dir, strict):
        return []
    out = []
    # "_"/"." folders are build or tool junk (live 2026-10-26: a services/__pycache__ "service"
    # had been written to self_knowledge_items 40 times), never a service.
    for service_dir in sorted(p for p in services_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        main_py = service_dir / "app" / "main.py"
        settings_py = service_dir / "app" / "settings.py"
        compose = service_dir / "docker-compose.yml"
        source = (settings_py if settings_py.exists() else main_py if main_py.exists()
                  else compose if compose.exists() else service_dir)
        out.append(_rec(root, service_dir.name, source, {
            "has_app_main": main_py.exists(),
            "has_app_settings": settings_py.exists(),
            "has_docker_compose": compose.exists(),
        }))
    return out


def modules(root: Path, *, strict: bool = False) -> list[dict]:
    package_root = root / "orion"
    if _absent(package_root, strict):
        return []
    return [_rec(root, f"orion.{path.name}", path / "__init__.py")
            for path in sorted(package_root.iterdir())
            if path.is_dir() and (path / "__init__.py").exists()]


def channels(root: Path, *, strict: bool = False) -> list[dict]:
    path = root / "orion" / "bus" / "channels.yaml"
    if _absent(path, strict):
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for channel in data.get("channels") or []:
        if not isinstance(channel, dict):
            continue
        out.append(_rec(root, str(channel.get("name") or ""), path, {
            "kind": channel.get("kind"),
            "schema_id": channel.get("schema_id"),
            "message_kind": channel.get("message_kind"),
            "producer_services": list(channel.get("producer_services") or []),
            "consumer_services": list(channel.get("consumer_services") or []),
        }))
    return sorted(out, key=lambda r: r["name"])


def yaml_verbs(root: Path, *, strict: bool = False) -> list[dict]:
    verbs_dir = root / "orion" / "cognition" / "verbs"
    out = []
    for path in [] if _absent(verbs_dir, strict) else sorted(verbs_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out.append(_rec(root, str(data.get("name") or path.stem), path,
                        {"source_type": "yaml", "recall_profile": data.get("recall_profile")}))
    return out


def runtime_verbs(root: Path, *, strict: bool = False) -> list[dict]:
    app_dir = root / "services" / "orion-cortex-exec" / "app"
    out = []
    for path in [] if _absent(app_dir, strict) else sorted(app_dir.glob("*.py")):
        for match in VERB_DECORATOR_RE.finditer(path.read_text(encoding="utf-8")):
            out.append(_rec(root, match.group(1), path, {"source_type": "runtime"}))
    return sorted(out, key=lambda r: (r["name"], r["source_path"]))


def verbs(root: Path) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    for rec in [*yaml_verbs(root), *runtime_verbs(root)]:
        merged[(rec["name"], rec["source_path"])] = rec
    return sorted(merged.values(), key=lambda r: (r["name"], r["source_path"]))


def schemas(root: Path, *, strict: bool = False) -> list[dict]:
    path = root / "orion" / "schemas" / "registry.py"
    if _absent(path, strict):
        return []
    names = sorted(set(REGISTRY_KEY_RE.findall(path.read_text(encoding="utf-8"))))
    return [_rec(root, name, path) for name in names]


def layout(root: Path, *, limit: int = 40) -> list[str]:
    """Top-level entries ("name/" for folders), for a checkout older than today's layout."""
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    return [p.name + ("/" if p.is_dir() else "") for p in entries if not p.name.startswith(".")][:limit]


def inventory(root: Path) -> dict[str, list[str]]:
    """Category -> sorted unique names, for comparing two checkouts."""
    scans = {"service": services, "module": modules, "channel": channels, "verb": verbs, "schema": schemas}
    return {cat: sorted({r["name"] for r in scans[cat](root) if r["name"]}) for cat in CATEGORIES}
