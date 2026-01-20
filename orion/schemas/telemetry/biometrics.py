from __future__ import annotations

"""Shared schema: biometrics.telemetry

This payload is intentionally shaped to match what `services/orion-biometrics`
emits and what `services/orion-sql-writer` persists.

We keep envelope validation *strict* at the Titanium/BaseEnvelope layer.
This payload schema focuses on consistent structure and forward/backward
compatibility for the *payload* only.

Canonical shape (v1):

  {
    "timestamp": "2026-01-04T06:24:00.009Z",
    "gpu": { ... },
    "cpu": { ... },
    "node": "athena",
    "service_name": "orion-biometrics",
    "service_version": "0.1.0"
  }

Compatibility notes:
  - We also accept legacy/experimental flat fields:
        cpu_util, gpu_util, gpu_mem_frac, node_name
    and normalize them into cpu/gpu dicts when present.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BiometricsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Required
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO8601 timestamp",
    )

    # Canonical nested payload
    gpu: Optional[Dict[str, Any]] = Field(default=None, description="Raw GPU telemetry blob")
    cpu: Optional[Dict[str, Any]] = Field(default=None, description="Raw CPU / sensors telemetry blob")

    # Canonical identity fields
    node: Optional[str] = None
    service_name: Optional[str] = None
    service_version: Optional[str] = None

    # Legacy/experimental fields (accepted for compatibility)
    node_name: Optional[str] = None
    cpu_util: Optional[float] = None
    gpu_util: Optional[float] = None
    gpu_mem_frac: Optional[float] = None

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "BiometricsPayload":
        # Normalize node_name -> node
        if not self.node and self.node_name:
            self.node = self.node_name

        # Normalize flat util fields into cpu/gpu dicts if nested blobs absent
        if self.cpu is None and self.cpu_util is not None:
            self.cpu = {"util": float(self.cpu_util)}

        if self.gpu is None and any(v is not None for v in [self.gpu_util, self.gpu_mem_frac]):
            gpu: Dict[str, Any] = {}
            if self.gpu_util is not None:
                gpu["util"] = float(self.gpu_util)
            if self.gpu_mem_frac is not None:
                gpu["mem_frac"] = float(self.gpu_mem_frac)
            self.gpu = gpu

        # Require at least *some* telemetry content. Timestamp-only events are not useful.
        if self.cpu is None and self.gpu is None:
            raise ValueError("BiometricsPayload requires at least one of cpu or gpu telemetry")

        return self


class BiometricsSampleV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node: Optional[str] = None
    service_name: Optional[str] = None
    service_version: Optional[str] = None

    gpu: Optional[Dict[str, Any]] = None
    cpu: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    disk: Optional[Dict[str, Any]] = None
    network: Optional[Dict[str, Any]] = None
    temps: Optional[Dict[str, Any]] = None
    power: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


class BiometricsSummaryV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node: Optional[str] = None
    service_name: Optional[str] = None
    service_version: Optional[str] = None

    pressures: Dict[str, float] = Field(default_factory=dict)
    headroom: Dict[str, float] = Field(default_factory=dict)
    composites: Dict[str, float] = Field(default_factory=dict)
    constraint: Optional[str] = None
    telemetry_error_rate: float = 0.0


class BiometricsInductionMetricV1(BaseModel):
    level: float = Field(0.0, ge=0.0, le=1.0)
    trend: float = Field(0.5, ge=0.0, le=1.0)
    volatility: float = Field(0.0, ge=0.0, le=1.0)
    spike_rate: float = Field(0.0, ge=0.0, le=1.0)


class BiometricsInductionV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node: Optional[str] = None
    service_name: Optional[str] = None
    service_version: Optional[str] = None
    window_sec: float = 30.0

    metrics: Dict[str, BiometricsInductionMetricV1] = Field(default_factory=dict)


class BiometricsClusterV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: List[str] = Field(default_factory=list)
    role_weights: Dict[str, float] = Field(default_factory=dict)

    pressures: Dict[str, float] = Field(default_factory=dict)
    headroom: Dict[str, float] = Field(default_factory=dict)
    composites: Dict[str, float] = Field(default_factory=dict)
    constraint: Optional[str] = None
