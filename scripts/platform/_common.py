from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking upward until we see expected dirs."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(25):
        if (cur / "services").is_dir() and (cur / "orion").is_dir():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate repo root (expected dirs: services/, orion/)")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")


def iter_files(
    root: Path,
    exts: Tuple[str, ...],
    include_dirs: Tuple[str, ...] = ("services", "orion", "scripts"),
) -> Iterator[Path]:
    for d in include_dirs:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def service_guess_from_path(p: Path) -> str:
    parts = p.parts
    if "services" in parts:
        i = parts.index("services")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "core"


def load_yaml_if_available(text: str) -> Any:
    """Parse YAML if PyYAML is installed; otherwise return None."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return None


def load_channels_catalog(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    """Load orion/bus/channels.yaml into a mapping of name -> entry."""
    catalog_path = repo_root / "orion" / "bus" / "channels.yaml"
    if not catalog_path.exists():
        return {}

    raw = read_text(catalog_path)
    y = load_yaml_if_available(raw)
    if isinstance(y, dict) and isinstance(y.get("channels"), list):
        out: Dict[str, Dict[str, Any]] = {}
        for entry in y["channels"]:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                out[entry["name"]] = entry
        return out

    # Fallback regex parse for `- name: "..."`
    out2: Dict[str, Dict[str, Any]] = {}
    for m in re.finditer(r"^\s*-\s*name:\s*['\"]?([^'\"\n]+)['\"]?\s*$", raw, flags=re.MULTILINE):
        out2[m.group(1).strip()] = {"name": m.group(1).strip()}
    return out2


@dataclass
class Finding:
    kind: str
    file: str
    line: int
    detail: str
    severity: str = "major"  # fatal|major|minor


def relpath(repo_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(repo_root))
    except Exception:
        return str(p)
