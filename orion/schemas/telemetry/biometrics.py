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

    # The binding constraint: the highest of ALL `pressures`, and which channel it is.
    #
    # This exists alongside `composites["strain"]`, which is UNCHANGED and stays that way.
    # `strain` is the mean of 7 of the 11 pressures, so it cannot see a saturated channel --
    # live on athena 2026-08-14, `power` at 1.000 with `strain` reading 0.443 -- and the four
    # it omits (`gpu_mem`, `swap`, `disk_capacity`, `fan`) are the top channel on two of three
    # nodes. `strain` is not fixed in place because it is written directly into the field
    # lattice; see `_peak_pressure` in orion/telemetry/biometrics_pipeline.py.
    #
    # Distinct from `constraint`, which is derived from the same max but keeps neither number:
    # it reports a name only above 0.7 and only for the 8 channels in `CONSTRAINTS`. Live on
    # athena 2026-08-15: peak `disk_capacity` 0.772 -- over threshold, unmapped, reported as
    # `constraint=NONE`. These two fields are what that discards.
    #
    # ABSENT, not 0.0, when no pressures were measured: nothing measured is not "nothing under
    # pressure". Same rule as `measurements`.
    peak_pressure: Optional[float] = None
    peak_pressure_channel: Optional[str] = None

    # Raw physical quantities in their own units, keyed with the unit in the name
    # (chassis_watts, gpu_watts_total, fan_pct_max, temp_c_max, cpu_cores, load_1m, load_15m).
    # Every other field on this model is normalised to 0-1, which is useful for anomaly
    # detection and useless for cost: a band fraction cannot be summed across hosts, compared
    # to last week, or turned into a kWh. `power_pressure = 0.798` says atlas is high against
    # its own recent history and cannot say how many watts -- while iLO was reporting the real
    # number all along.
    #
    # A key is ABSENT when the quantity is not measured on that node -- never 0.0. circe has
    # no reachable BMC, so a zero would silently understate any fleet total that sums this
    # dict. Absent-is-not-zero is the same rule the lane-occupancy instrument enforces for
    # unreachable hosts, and for the same reason.
    #
    # None vs {} is a real distinction and consumers must respect it:
    #   None -> the producer predates this field. Says NOTHING about the node.
    #   {}   -> a current producer measured nothing it could vouch for.
    # Defaulting this to {} would collapse the two and let a fleet total count a
    # not-yet-upgraded node as "0 W measured" -- which is exactly what atlas looks like
    # today, mid-rollout, while its iLO is demonstrably live.
    #
    # CONTAINMENT: `chassis_watts` is measured at the PSU and already includes
    # `gpu_watts_total`. Never sum the two. For a fleet total use `chassis_watts` where
    # present; `gpu_watts_total` is for nodes with no BMC, and for attributing what share of a
    # node's draw is inference.
    measurements: Optional[Dict[str, float]] = None


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

    # Fleet-level raw physical units, in the units themselves -- NOT the weighted, clamped
    # 0-1 mean the three fields above use. Watts do not normalise: `min(1.0, sum/weight)`
    # applied to 663 W is not a smaller number, it is a different kind of thing.
    #
    # Extensive quantities (chassis_watts, gpu_watts_total, gpu_count, cpu_cores) are summed;
    # intensive ones (temp_c_max, fan_pct_max) are maxed; load averages are deliberately
    # absent because they are relative to each machine's own core count. See
    # FLEET_SUM_KEYS / FLEET_MAX_KEYS in orion/telemetry/biometrics_pipeline.py.
    measurements: Optional[Dict[str, float]] = None

    # Which source nodes did NOT report each key in `measurements`. This is not diagnostics --
    # it is what makes the total honest. circe has no reachable BMC, so a fleet chassis_watts
    # is a real number that is missing an entire machine, and nothing in the number itself
    # says so. Reading `measurements["chassis_watts"]` without checking this is reading a
    # partial sum as a complete one.
    measurements_missing: Optional[Dict[str, List[str]]] = None

    # Nodes the catalog knows about that did NOT contribute to this cluster at all.
    #
    # `measurements_missing` closes the gap for a source that reported but lacked a key. It
    # cannot close this one: a node that stops publishing entirely never enters `sources`, so
    # it leaves no trace anywhere and the totals look complete. Observed live 2026-08-14 while
    # circe was down -- `chassis_watts: 646.0` with `measurements_missing: null`, a two-machine
    # sum presenting as the whole fleet.
    #
    # This is deliberately a STATEMENT OF FACT, not an alarm. Whether an absence is a problem
    # is policy that depends on the node: circe is run intermittently to save cost, so its
    # silence is expected, while the same silence from athena would be an outage. The catalog's
    # `expected_online` flag cannot settle it either -- it is already doing a different job
    # (gating whether `pressure_organ.py` suppresses a node's strain as expected staleness).
    # So the aggregate reports who is absent and leaves the judgement to the consumer.
    nodes_absent: Optional[List[str]] = None

    # Fleet binding constraint: the highest single pressure on ANY node, which channel it is,
    # and which node carries it. INTENSIVE -- taken as a max, never summed or averaged. The
    # weighted mean in `pressures` above answers "how loaded is the fleet on average", which
    # smears one saturated machine across three; this answers "what is about to stop us, and
    # where", which is the question a ceiling has.
    #
    # `peak_pressure_node` is the load-bearing half. Without it a fleet peak of 0.772 says
    # something is nearly full and gives no way to act on it.
    peak_pressure: Optional[float] = None
    peak_pressure_channel: Optional[str] = None
    peak_pressure_node: Optional[str] = None
