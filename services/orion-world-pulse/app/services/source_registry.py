from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from orion.schemas.world_pulse import SourceRegistryV1


def load_source_registry(path: str) -> SourceRegistryV1:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"world pulse source registry not found: {cfg_path}")
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    registry = SourceRegistryV1.model_validate(raw)
    _validate_registry(registry)
    return registry


def _validate_registry(registry: SourceRegistryV1) -> None:
    for source in registry.sources:
        strategy = source.strategy or "rss"
        if strategy in {"rss", "atom", "sitemap", "html_section"} and not source.url:
            raise ValueError(f"source {source.source_id} requires `url` for strategy={strategy}")
        if strategy == "manual_urls" and not source.urls:
            raise ValueError(f"source {source.source_id} requires non-empty `urls` for strategy=manual_urls")
        if strategy in {"sitemap", "html_section", "manual_urls"} and not source.domains:
            raise ValueError(f"source {source.source_id} requires `domains` allowlist for strategy={strategy}")
