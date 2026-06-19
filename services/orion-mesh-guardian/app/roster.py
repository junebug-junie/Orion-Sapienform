from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProbeMode(str, Enum):
    redis = "redis"
    http = "http"
    redis_and_http = "redis_and_http"


class ProbeConfig(BaseModel):
    mode: ProbeMode
    intake_channels: list[str] = Field(default_factory=list)
    ready_url: str | None = None
    service_name: str | None = None


class RosterEntry(BaseModel):
    id: str
    heartbeat_name: str
    compose_dir: str
    compose_service: str
    include_bus_env: bool = False
    auto_remediate: bool = True
    probe: ProbeConfig


class RosterDocument(BaseModel):
    services: list[RosterEntry]


NEVER_REMEDIATE_IDS = frozenset({"mesh-guardian", "hub", "orion-hub", "notify", "orion-notify"})


def _substitute_project(value: str | None, *, project: str) -> str | None:
    if value is None:
        return None
    return value.replace("${PROJECT}", project)


def load_roster(path: str, *, project: str, node_name: str) -> RosterDocument:
    import yaml
    from pathlib import Path

    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    services: list[RosterEntry] = []
    for item in data.get("services", []):
        probe_data = dict(item.get("probe") or {})
        if probe_data.get("ready_url") is not None:
            probe_data["ready_url"] = _substitute_project(str(probe_data["ready_url"]), project=project)
        services.append(
            RosterEntry(
                id=item["id"],
                heartbeat_name=item["heartbeat_name"],
                compose_dir=item["compose_dir"],
                compose_service=item["compose_service"],
                include_bus_env=bool(item.get("include_bus_env", False)),
                auto_remediate=bool(item.get("auto_remediate", True)),
                probe=ProbeConfig(**probe_data),
            )
        )
    return RosterDocument(services=services)


def validate_roster(doc: RosterDocument) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for entry in doc.services:
        if entry.id in seen:
            errors.append(f"duplicate roster id: {entry.id}")
        seen.add(entry.id)
        if entry.probe.mode in (ProbeMode.http, ProbeMode.redis_and_http) and not entry.probe.ready_url:
            errors.append(f"{entry.id}: http probe requires ready_url")
    return errors
