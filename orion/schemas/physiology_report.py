"""The body-economy report: Orion's body over a window, aggregate down to single decisions.

``GET /physiology/report`` on orion-durable-runs returns one ``PhysiologyReportV1``. It is a
read-only HTTP contract (no bus channel). Two readers:

* the Hub ``#body-economy`` page (Phase 3 of
  docs/superpowers/specs/2026-10-23-organism-body-economy-design.md), built in parallel against
  these exact top-level keys;
* the ``skills.physiology.state.v1`` skill, which turns it into plain text for Orion (chat turns
  through cortex-exec, curiosity and chat harness turns through ``python -P -m orion.dig body-economy``).

A section is never silently empty. When a section has no rows, ``missing`` carries one entry
for it with the reason (for example "prices began 2026-10-27"). A reader shows that reason
instead of an empty chart.

Units: seconds, USD, kWh, W·h, degrees C. ``h`` keys are UTC hour starts (ISO 8601). ``rhythm``
uses local America/Denver days and hours, because the claim it tests is "heavy work drifts
toward cool local hours".
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AwareDatetime, BaseModel, Field

PHYSIOLOGY_REPORT_SCHEMA_VERSION = "physiology.report.v1"
PHYSIOLOGY_REPORT_KIND = "physiology.report.v1"

REPORT_WINDOWS = ("24h", "7d", "30d")
#: Every section key a caller may ask for with ``sections=``. ``missing`` is always returned.
REPORT_SECTIONS = (
    "limits", "wants", "rhythm", "money", "unit_costs", "waste", "queue_wait", "decisions",
    "claude", "vram", "value_per_cost", "services", "processes", "hosts",
)


class ReportWindowV1(BaseModel):
    start: AwareDatetime
    end: AwareDatetime


class LimitNowV1(BaseModel):
    level: Optional[float] = None
    bind_state: Optional[str] = None
    why: str = ""


class LimitHourV1(BaseModel):
    h: AwareDatetime
    level: Optional[float] = None
    price: Optional[float] = None
    bind_state: Optional[str] = None


class LimitV1(BaseModel):
    constraint: str
    label: str
    unit: Optional[str] = None
    line: Optional[float] = Field(default=None, description="the budget line (soft line, envelope, capacity)")
    now: LimitNowV1 = Field(default_factory=LimitNowV1)
    hourly: List[LimitHourV1] = Field(default_factory=list)


class WantFamilyV1(BaseModel):
    family: str
    tier: Optional[int] = None
    funded: int = 0
    deferred: int = 0


class WantHourV1(BaseModel):
    h: AwareDatetime
    family: str
    funded: int = 0
    deferred: int = 0


class WantsV1(BaseModel):
    families: List[WantFamilyV1] = Field(default_factory=list)
    hourly: List[WantHourV1] = Field(default_factory=list)


class RhythmCellV1(BaseModel):
    """Each closed lease is spread over the local hours it was held. A lease still open at the
    window's end is not counted yet."""

    day: str = Field(description="local America/Denver date, YYYY-MM-DD")
    hour: int = Field(ge=0, le=23, description="local America/Denver hour")
    heavy: float = Field(description="lease-seconds of deferrable (background) GPU work held in this hour")
    total: float = Field(description="lease-seconds of all GPU work held in this hour")


class DailyKwhV1(BaseModel):
    d: str = Field(description="local America/Denver date, YYYY-MM-DD")
    kwh: float
    partial: bool = Field(default=False, description="the window covers only part of this local day")


class MoneyV1(BaseModel):
    mtd_usd: Optional[float] = None
    forecast_usd: Optional[float] = None
    envelope_usd: Optional[float] = None
    floor_usd: Optional[float] = None
    rate_usd_per_kwh: Optional[float] = None
    daily_kwh: List[DailyKwhV1] = Field(default_factory=list)


class UnitCostV1(BaseModel):
    unit: str
    usd: Optional[float] = None
    wh: Optional[float] = None
    heat_c: Optional[float] = None
    status: Literal["measured", "derived", "not_measured"]
    source: str


class WasteTopV1(BaseModel):
    holder: Optional[str] = None
    role: Optional[str] = None
    held_s: float
    reason: Optional[str] = None
    at: AwareDatetime


class WasteV1(BaseModel):
    failed_lease_s_by_lane: Dict[str, float] = Field(default_factory=dict)
    top: List[WasteTopV1] = Field(default_factory=list)


class QueueWaitV1(BaseModel):
    lane: str
    family: str = Field(description="want family when known (durable-run admissions), else the holder's service")
    family_source: Optional[Literal["want_ledger", "holder"]] = None
    p50_s: Optional[float] = None
    p90_s: Optional[float] = None
    n: int


class DecisionV1(BaseModel):
    decided_at: AwareDatetime
    family: str
    verb: Optional[str] = None
    tier: Optional[int] = None
    outcome: str
    reason: Optional[str] = None
    net: Optional[float] = None
    binding_constraint: Optional[str] = None
    why: str = ""
    prices_seen: Dict[str, float] = Field(default_factory=dict)


class ClaudeTokensV1(BaseModel):
    """None (not 0) when no token row exists in the window: unknown is not zero."""

    input: Optional[int] = None
    output: Optional[int] = None
    cache_read: Optional[int] = None
    cache_write: Optional[int] = None


class ClaudeLimitEventV1(BaseModel):
    at: AwareDatetime
    kind: str


class ClaudeV1(BaseModel):
    window_tokens: ClaudeTokensV1 = Field(default_factory=ClaudeTokensV1)
    usd_est: Optional[float] = None
    limit_events: List[ClaudeLimitEventV1] = Field(default_factory=list)
    status: str = ""


class VramCardV1(BaseModel):
    host: str
    card: str
    used_mb: Optional[float] = None
    total_mb: Optional[float] = None
    headroom_mb: Optional[float] = None
    roles: List[str] = Field(default_factory=list)
    watts: Optional[float] = Field(default=None, description="the card's draw now (gpuN_watts)")


class ProcessV1(BaseModel):
    """One container's share of its host's CPU package watts ABOVE IDLE (linear utilisation model,
    orion-biometrics ``container_cpu_watts.<name>``). Two pseudo-names: ``(small containers)`` and
    ``(host, not in a container)``. ``cores`` is None: busy cores are not published per container."""

    host: str
    name: str
    watts: float
    cores: Optional[float] = None


class HostHolderV1(BaseModel):
    holder: str = Field(description="GPU-pool lease holder; a durable run is named by its workflow "
                                    "(durable-runs:curiosity.investigate), other run ids are folded")
    label: str = Field(default="", description="plain words for the holder (http:anthropic -> harness turns)")
    lease_s: float = Field(description="seconds this holder HELD the card in the window (closed leases); held, "
                                       "not necessarily computing")


class HostCardV1(BaseModel):
    card: str
    watts: Optional[float] = Field(default=None, description="draw now (gpuN_watts, nvidia-smi: measured)")
    idle_w: Optional[float] = Field(default=None, description="the card's rest: its 5th percentile over the pinned rest window")
    temp_c: Optional[float] = None
    mem_used_mb: Optional[float] = None
    roles: List[str] = Field(default_factory=list)
    top_holders: List[HostHolderV1] = Field(default_factory=list)


class HostContainerV1(BaseModel):
    name: str
    cpu_w: float = Field(description="CPU package watts above idle, split by busy cores (estimated: linear model)")


class HostSponsoredV1(BaseModel):
    """Work Juniper started and only Juniper stops (config/physiology/rest_body.yaml ``sponsored``),
    shown apart from Orion's own activity (today: the Zelda dojo, pending her decision)."""

    name: str
    note: str = ""
    containers: int = 0
    cpu_w: Optional[float] = Field(default=None, description="its containers' CPU watts above idle (estimated); None: none running")


class HostDayV1(BaseModel):
    d: str = Field(description="local America/Denver date, YYYY-MM-DD")
    chassis_w_avg: float


class HostPowerV1(BaseModel):
    """What one machine draws now, against its own rest, and what the draw is made of.

    ``chassis_w_now`` is the whole machine (BMC or wall PDU: measured). ``gpu_w`` (nvidia-smi) and
    ``cpu_w`` (RAPL) are INSIDE it, so ``other_w`` = chassis - gpu - cpu is fans, PSU loss, memory,
    disks. ``chassis_w_rest`` is a low percentile of the chassis draw over a trailing window
    (``rest_basis`` names it), so "above rest" is what is spinning the machine now."""

    host: str
    at: Optional[AwareDatetime] = Field(default=None, description="the biometrics sample the 'now' figures are from")
    chassis_w_now: Optional[float] = None
    chassis_w_rest: Optional[float] = None
    rest_basis: str = ""
    recent_floor_w: Optional[float] = Field(default=None, description="5th percentile of chassis watts, trailing 7 days")
    above_rest_w: Optional[float] = None
    usd_per_month_at_this_rate: Optional[float] = Field(
        default=None, description="above_rest_w held for a 730-hour month at the newest electricity rate")
    rest_usd_per_month: Optional[float] = Field(
        default=None, description="chassis_w_rest held for a 730-hour month at the newest electricity rate")
    gpu_w: Optional[float] = None
    cpu_w: Optional[float] = None
    other_w: Optional[float] = Field(default=None, description="chassis - gpu - cpu: fans, PSU loss, memory, disks")
    gpu_w_rest: Optional[float] = Field(default=None, description="each part's own 5th percentile over the pinned "
                                                                  "rest window; the part rests do not sum to the chassis rest")
    cpu_w_rest: Optional[float] = None
    other_w_rest: Optional[float] = None
    parts_rest_window: str = Field(default="", description="the dated window the part and card rests come from")
    fan_pct: Optional[float] = None
    unattributed_above_rest_w: Optional[float] = Field(
        default=None, description="above-rest draw no act is charged for (from the per-act meter; None until it publishes)")
    cards: List[HostCardV1] = Field(default_factory=list)
    top_containers: List[HostContainerV1] = Field(default_factory=list,
                                                  description="largest CPU users, sponsored work left out")
    sponsored: List[HostSponsoredV1] = Field(default_factory=list)
    daily: List[HostDayV1] = Field(default_factory=list)


class ValuePerCostV1(BaseModel):
    family: str
    evidence_n: int
    value_metric: str
    value: Optional[float] = None
    usd_per_act: Optional[float] = None
    status: Literal["measured", "not_measured"]


class ServiceDownV1(BaseModel):
    name: str
    host: Optional[str] = None
    state: Optional[str] = None
    since: Optional[AwareDatetime] = None


class ServicesV1(BaseModel):
    checked_at: Optional[AwareDatetime] = None
    down: List[ServiceDownV1] = Field(default_factory=list)
    basis: str = ""


class MissingV1(BaseModel):
    section: str
    reason: str


class PhysiologyReportV1(BaseModel):
    """One read of Orion's body economy over a window. Every section is present; an empty one
    has a ``missing`` entry that says why."""

    schema_version: Literal["physiology.report.v1"] = PHYSIOLOGY_REPORT_SCHEMA_VERSION
    generated_at: AwareDatetime
    window: ReportWindowV1
    limits: List[LimitV1] = Field(default_factory=list)
    wants: WantsV1 = Field(default_factory=WantsV1)
    rhythm: List[RhythmCellV1] = Field(default_factory=list)
    money: MoneyV1 = Field(default_factory=MoneyV1)
    unit_costs: List[UnitCostV1] = Field(default_factory=list)
    waste: WasteV1 = Field(default_factory=WasteV1)
    queue_wait: List[QueueWaitV1] = Field(default_factory=list)
    decisions: List[DecisionV1] = Field(default_factory=list)
    claude: ClaudeV1 = Field(default_factory=ClaudeV1)
    vram: List[VramCardV1] = Field(default_factory=list)
    value_per_cost: List[ValuePerCostV1] = Field(default_factory=list)
    services: ServicesV1 = Field(default_factory=ServicesV1)
    processes: List[ProcessV1] = Field(default_factory=list)
    hosts: List[HostPowerV1] = Field(default_factory=list)
    missing: List[MissingV1] = Field(default_factory=list)
    query: Dict[str, Any] = Field(default_factory=dict, description="the window, sections, family, outcome, limit asked")
    timing_ms: Dict[str, float] = Field(default_factory=dict, description="per-section read time, for H5")


__all__ = [
    "PHYSIOLOGY_REPORT_KIND",
    "PHYSIOLOGY_REPORT_SCHEMA_VERSION",
    "REPORT_SECTIONS",
    "REPORT_WINDOWS",
    "ClaudeLimitEventV1",
    "ClaudeTokensV1",
    "ClaudeV1",
    "DailyKwhV1",
    "DecisionV1",
    "HostCardV1",
    "HostContainerV1",
    "HostDayV1",
    "HostHolderV1",
    "HostPowerV1",
    "HostSponsoredV1",
    "LimitHourV1",
    "LimitNowV1",
    "LimitV1",
    "MissingV1",
    "MoneyV1",
    "PhysiologyReportV1",
    "ProcessV1",
    "QueueWaitV1",
    "ReportWindowV1",
    "RhythmCellV1",
    "ServiceDownV1",
    "ServicesV1",
    "UnitCostV1",
    "ValuePerCostV1",
    "VramCardV1",
    "WantFamilyV1",
    "WantHourV1",
    "WantsV1",
    "WasteTopV1",
    "WasteV1",
]
