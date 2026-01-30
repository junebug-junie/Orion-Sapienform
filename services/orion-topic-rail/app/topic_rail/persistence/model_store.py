from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timezone

from bertopic import BERTopic


logger = logging.getLogger("topic-rail.model-store")


class ModelStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)

    def path_for(self, model_version: str) -> Path:
        safe_version = model_version.replace("/", "_")
        return self.base_dir / safe_version

    def exists(self, model_version: str) -> bool:
        path = self.path_for(model_version)
        return (path / "topic_model").exists()

    def save(self, model_version: str, topic_model: BERTopic, settings_snapshot: dict) -> None:
        path = self.path_for(model_version)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "topic_model"
        topic_model.save(str(model_path))
        config_path = path / "settings.json"
        config_path.write_text(json.dumps(settings_snapshot, indent=2))
        logger.info("Saved topic model to %s", model_path)

    def load(self, model_version: str) -> Tuple[BERTopic, Optional[dict], Optional[dict]]:
        path = self.path_for(model_version)
        model_path = path / "topic_model"
        topic_model = BERTopic.load(str(model_path))
        settings_path = path / "settings.json"
        settings_snapshot = None
        if settings_path.exists():
            try:
                settings_snapshot = json.loads(settings_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load settings snapshot: %s", exc)
        manifest = None
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load manifest: %s", exc)
        return topic_model, settings_snapshot, manifest

    def write_manifest(self, model_version: str, manifest: Dict[str, Any]) -> None:
        path = self.path_for(model_version)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def load_manifest(self, model_version: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(model_version)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load manifest: %s", exc)
            return None
