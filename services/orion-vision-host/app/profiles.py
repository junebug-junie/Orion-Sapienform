from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class Step:
    use: str
    when: Optional[str] = None


@dataclass
class PipelineDef:
    name: str
    enabled: bool
    description: str
    steps: List[Step]


@dataclass
class ProfileDef:
    name: str
    enabled: bool
    warm_on_start: bool
    kind: str
    backend: str
    model_id: str
    device: str
    dtype: str
    params: Dict[str, Any]
    cost: Dict[str, Any]
    description: str


class VisionProfiles:
    def __init__(self, path: str):
        self.path = path

        self.version: str = "unknown"
        self.runtime: Dict[str, Any] = {}

        self.task_routing: Dict[str, str] = {}
        self.pipelines: Dict[str, PipelineDef] = {}
        self.profiles: Dict[str, ProfileDef] = {}

    def load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"vision_profiles.yaml not found: {self.path}")

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        self.version = str(raw.get("version", "unknown"))
        self.runtime = raw.get("runtime", {}) or {}
        self.task_routing = raw.get("task_routing", {}) or {}

        # Pipelines
        self.pipelines.clear()
        for item in raw.get("pipelines", []) or []:
            name = item["name"]
            steps = []
            for s in item.get("steps", []) or []:
                steps.append(Step(use=s["use"], when=s.get("when")))
            self.pipelines[name] = PipelineDef(
                name=name,
                enabled=bool(item.get("enabled", True)),
                description=str(item.get("description", "")),
                steps=steps,
            )

        # Profiles
        self.profiles.clear()
        for item in raw.get("profiles", []) or []:
            name = item["name"]
            self.profiles[name] = ProfileDef(
                name=name,
                enabled=bool(item.get("enabled", True)),
                warm_on_start=bool(item.get("warm_on_start", False)),
                kind=str(item.get("kind", "")),
                backend=str(item.get("backend", "")),
                model_id=str(item.get("model_id", "")),
                device=str(item.get("device", "auto")),
                dtype=str(item.get("dtype", "auto")),
                params=item.get("params", {}) or {},
                cost=item.get("cost", {}) or {},
                description=str(item.get("description", "")),
            )

        logger.info(
            f"[PROFILES] loaded version={self.version} "
            f"pipelines={len(self.pipelines)} profiles={len(self.profiles)}"
        )

    def resolve_target(self, task_type: str) -> str:
        """
        Map task_type -> pipeline/profile name.
        If no mapping exists, treat task_type itself as a name.
        """
        return self.task_routing.get(task_type, task_type)

    def is_pipeline(self, name: str) -> bool:
        return name in self.pipelines

    def get_pipeline(self, name: str) -> PipelineDef:
        return self.pipelines[name]

    def get_profile(self, name: str) -> ProfileDef:
        return self.profiles[name]
