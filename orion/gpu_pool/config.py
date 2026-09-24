"""Load and validate ``config/gpu_pool.yaml``.

The YAML names roles and cards, never models. Everything model-dependent (VRAM footprint,
context per slot, vision) is resolved later against the *discovered* ``llm_profiles.yaml``
profile, so replacing a model file never needs an edit here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "gpu_pool.yaml"
PRIORITIES = ("interactive", "system", "background")


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_attempts: int = Field(3, ge=1, le=20)
    base_sec: float = Field(5, gt=0)
    max_sec: float = Field(300, gt=0)

    def delay(self, attempt: int) -> float:
        return min(self.max_sec, self.base_sec * (2 ** max(0, attempt - 1)))


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clawback_grace_sec: float = Field(60, ge=0)
    request_lease_ttl_sec: float = Field(30, gt=0)
    hold_lease_ttl_sec: float = Field(90, gt=0)
    swap_cooldown_sec: float = Field(600, ge=0)
    swap_after_wait_sec: float = Field(30, ge=0)
    swap_idle_unload_sec: float = Field(300, ge=0)
    backlog_max_age_sec: float = Field(86400, gt=0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class HostSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    address: str


class CardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vram_gb: float = Field(gt=0)
    lendable: bool = False


class SwapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evicts: list[str] | Literal["all"]
    load: str
    unload: str


class RoleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["llm", "service"]
    cards: list[str] = Field(min_length=1)
    owner: list[str] = Field(default_factory=list)
    port: int = Field(ge=1, le=65535)
    slots: int | None = Field(None, ge=1)  # services only; llm slots are discovered
    vram_gb: float | None = Field(None, gt=0)  # services only; llm VRAM comes from the profile
    health: str = "/health"
    swap: SwapSpec | None = None
    operator_only: bool = False
    max_hold_sec: float | None = Field(None, gt=0)

    @field_validator("owner", mode="before")
    @classmethod
    def _listify(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _service_shape(self):
        if self.kind == "service" and (self.slots is None or self.vram_gb is None):
            raise ValueError("service roles must declare slots and vram_gb (no profile to discover)")
        return self


class ClassSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roles: list[str] = Field(min_length=1)
    on_unavailable: Literal["wait", "backlog", "fail"] = "wait"


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    work_class: str = Field(alias="class")
    priority: Literal["interactive", "system", "background"] = "system"


class PoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    host: HostSpec
    defaults: Defaults = Field(default_factory=Defaults)
    priorities: list[str] = Field(default_factory=lambda: list(PRIORITIES))
    cards: dict[str, CardSpec]
    roles: dict[str, RoleSpec]
    classes: dict[str, ClassSpec]
    routes: dict[str, RouteSpec] = Field(default_factory=dict)
    digest: str = ""

    @field_validator("routes", mode="before")
    @classmethod
    def _route_shorthand(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {k: ({"class": v} if isinstance(v, str) else v) for k, v in value.items()}

    @model_validator(mode="after")
    def _cross_refs(self):
        errors: list[str] = []
        if sorted(self.priorities) != sorted(PRIORITIES):
            errors.append(f"priorities must be a permutation of {PRIORITIES}")
        for name, role in self.roles.items():
            for card in role.cards:
                if card not in self.cards:
                    errors.append(f"role {name}: unknown card {card}")
            for owner in role.owner:
                if owner not in self.classes:
                    errors.append(f"role {name}: owner {owner} is not a class")
                elif name not in self.classes[owner].roles:
                    errors.append(f"role {name}: owner class {owner} does not list it")
            if role.swap and role.swap.evicts != "all":
                for evicted in role.swap.evicts:
                    other = self.roles.get(evicted)
                    if other is None:
                        errors.append(f"role {name}: evicts unknown role {evicted}")
                    elif not set(other.cards) & set(role.cards):
                        errors.append(f"role {name}: evicts {evicted}, which shares no card with it")
        for name, cls in self.classes.items():
            for role in cls.roles:
                if role not in self.roles:
                    errors.append(f"class {name}: unknown role {role}")
            if len(set(cls.roles)) != len(cls.roles):
                errors.append(f"class {name}: duplicate roles")
        for name, route in self.routes.items():
            if route.work_class not in self.classes:
                errors.append(f"route {name}: unknown class {route.work_class}")
        ports: dict[int, str] = {}
        for name, role in self.roles.items():
            if role.port in ports:
                errors.append(f"roles {ports[role.port]} and {name} share port {role.port}")
            ports[role.port] = name
        if errors:
            raise ValueError("; ".join(errors))
        return self

    # --- derived views -------------------------------------------------------------
    def priority_rank(self, priority: str) -> int:
        return self.priorities.index(priority)

    def url(self, role: str) -> str:
        return f"http://{self.host.address}:{self.roles[role].port}"

    def owns(self, work_class: str, role: str) -> bool:
        return work_class in self.roles[role].owner

    def lendable_cards(self, role: str) -> list[str]:
        return [c for c in self.roles[role].cards if self.cards[c].lendable]

    def evicted_by(self, role: str) -> list[str]:
        swap = self.roles[role].swap
        if swap is None:
            return []
        if swap.evicts == "all":
            cards = set(self.roles[role].cards)
            return sorted(r for r, spec in self.roles.items() if r != role and set(spec.cards) & cards)
        return list(swap.evicts)

    def resident_roles(self) -> list[str]:
        """Roles loaded when no swap is active: everything without its own swap seat."""
        return [r for r, spec in self.roles.items() if spec.swap is None]


def load_pool_config(path: str | Path | None = None) -> PoolConfig:
    raw_bytes = Path(path or DEFAULT_PATH).read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    cfg = PoolConfig.model_validate(data)
    cfg.digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    return cfg


def check_vram(cfg: PoolConfig, footprint_gb: dict[str, float]) -> list[str]:
    """Resident roles' VRAM (from discovered profiles or service declarations) must fit
    each card, and every swap seat must fit once its evictions are gone."""
    problems: list[str] = []
    residents = cfg.resident_roles()

    def share(role: str) -> float:
        return footprint_gb.get(role, 0.0) / len(cfg.roles[role].cards)

    for card, spec in cfg.cards.items():
        used = sum(share(r) for r in residents if card in cfg.roles[r].cards)
        if used > spec.vram_gb:
            problems.append(f"{card}: residents need {used:.1f}GB of {spec.vram_gb:.0f}GB")
        for role, rspec in cfg.roles.items():
            if rspec.swap is None or card not in rspec.cards:
                continue
            evicted = set(cfg.evicted_by(role))
            after = sum(share(r) for r in residents if card in cfg.roles[r].cards and r not in evicted)
            need = after + share(role)
            if footprint_gb.get(role) and need > spec.vram_gb:
                problems.append(f"{card}: swap {role} needs {need:.1f}GB of {spec.vram_gb:.0f}GB")
    return problems
