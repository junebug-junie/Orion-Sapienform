from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MeshNodeStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_name: str
    tailscale_ip: Optional[str] = None
    dns_name: Optional[str] = None
    os: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    owner: Optional[str] = None
    local_backend_state: Optional[str] = None
    peer_status_classification: Literal["active", "idle", "offline", "unknown"] = "unknown"
    connection_info: Dict[str, Any] = Field(default_factory=dict)
    latency_probe: Optional[Dict[str, Any]] = None
    observed_at_utc: str = Field(default_factory=_utc_now_iso)


class MeshStatusSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = True
    backend_state: Optional[str] = None
    observed_at_utc: str = Field(default_factory=_utc_now_iso)
    node_count: int = 0
    active_nodes: List[str] = Field(default_factory=list)
    nodes: List[MeshNodeStatusV1] = Field(default_factory=list)


class DiskHealthDeviceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_name: str
    device: str
    protocol: str = "unknown"
    model: Optional[str] = None
    serial: Optional[str] = None
    health_passed: Optional[bool] = None
    overall_health: str = "unknown"
    temperature_c: Optional[float] = None
    power_on_hours: Optional[int] = None
    critical_warning: Optional[Any] = None
    percentage_used: Optional[int] = None
    media_errors: Optional[int] = None
    available_spare: Optional[int] = None
    reallocated_sectors: Optional[int] = None
    pending_sectors: Optional[int] = None
    raw_exit_status: Optional[int] = None
    parse_warnings: List[str] = Field(default_factory=list)
    observed_at_utc: str = Field(default_factory=_utc_now_iso)


class DiskHealthSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_name: str
    observed_at_utc: str = Field(default_factory=_utc_now_iso)
    devices: List[DiskHealthDeviceV1] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)


class RepoPullRequestDigestItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    author: Optional[str] = None
    state: str = "closed"
    merged_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    base_ref: Optional[str] = None
    head_ref: Optional[str] = None
    url: Optional[str] = None
    changed_files_count: int = 0
    touched_paths: List[str] = Field(default_factory=list)
    inferred_services: List[str] = Field(default_factory=list)
    short_summary: Optional[str] = None


class RepoRecentChangesDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = True
    repo: Optional[str] = None
    lookback_days: int = 7
    merged_pr_count: int = 0
    grouped_summary: List[Dict[str, Any]] = Field(default_factory=list)
    items: List[RepoPullRequestDigestItemV1] = Field(default_factory=list)
    observed_at_utc: str = Field(default_factory=_utc_now_iso)


class DockerPruneResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_name: str
    dry_run: bool = True
    requested_filters: Dict[str, Any] = Field(default_factory=dict)
    matched_container_count: int = 0
    pruned_container_count: int = 0
    reclaimed_bytes: Optional[int] = None
    protected_skips: List[str] = Field(default_factory=list)
    command: str = ""
    stdout_stderr_summary: str = ""
    status: str = "unknown"
    observed_at_utc: str = Field(default_factory=_utc_now_iso)


class DockerPruneSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    result: Optional[DockerPruneResultV1] = None


class MeshOpsRoundResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_name: str = "mesh_ops_round"
    observed_at_utc: str = Field(default_factory=_utc_now_iso)
    mesh_presence: Dict[str, Any] = Field(default_factory=dict)
    active_nodes: List[str] = Field(default_factory=list)
    storage_health: Dict[str, Any] = Field(default_factory=dict)
    recent_changes: Dict[str, Any] = Field(default_factory=dict)
    runtime_housekeeping: Dict[str, Any] = Field(default_factory=dict)
    overall_health: str = "unknown"
    partial_failures: List[str] = Field(default_factory=list)
    journal_write: Optional[Dict[str, Any]] = None


class OpsMeshRoundJournalEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_type: Literal["ops.mesh_round.v1"] = "ops.mesh_round.v1"
    body: MeshOpsRoundResultV1
    created_at_utc: str = Field(default_factory=_utc_now_iso)
