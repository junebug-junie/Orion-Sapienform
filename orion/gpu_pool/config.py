"""Load and validate ``config/gpu_pool.yaml``.

The YAML names roles and cards, never models. Everything model-dependent (VRAM footprint,
context per slot, vision) is resolved later against the *discovered* ``llm_profiles.yaml``
profile, so replacing a model file never needs an edit here.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "gpu_pool.yaml"
PRIORITIES = ("interactive", "system", "background")
# Pool-side swap preconditions (stage 4 spec, "Guards"). A guard named here must be one the
# scheduler evaluates; the scheduler side lands with the actuation engine (stage 4.3).
SWAP_GUARDS = ("thermal", "visual_baseline")


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
    # Stage 4 (docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md).
    # Parsed and validated from 4.1; read by the hold/actuation engine from 4.3.
    hold_clawback_grace_sec: float = Field(600, ge=0)   # a recalled hold finishes its current node within this
    swap_min_residency_sec: float = Field(600, ge=0)    # after a seat unloads, evicted residents stay this long
    actuate_ack_sec: float = Field(10, gt=0)            # no "accepted" within this -> actuator_unreachable


class HostSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    address: str


class ActuatorSpec(BaseModel):
    """A host-local actuator the pool routes GpuActuateV1 to by name (stage 4: circe's
    orion-gpu-lane-controller)."""
    model_config = ConfigDict(extra="forbid")
    host: str


class CardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vram_gb: float = Field(gt=0)
    lendable: bool = False
    index: int | None = Field(None, ge=0)   # the actuator host's CUDA device number for this card


def _http_path(value: str) -> str:
    if not value.startswith("/"):
        raise ValueError(f"{value!r} must be an HTTP path starting with '/'")
    return value


class DrainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    set_path: str = Field(alias="set")     # POST {"draining": true} here
    status: str                            # poll until in_flight is false

    @field_validator("set_path", "status")
    @classmethod
    def _paths(cls, value: str) -> str:
        return _http_path(value)


class LaunchSpec(BaseModel):
    """How the role's actuator starts and stops it (stage 4 spec, "YAML shape"). The pool never
    sends a container name: the actuator acts only on what its own copy of this file says here."""
    model_config = ConfigDict(extra="forbid")
    actuator: str
    compose: str                           # repo-relative compose file
    env_file: str | None = None            # repo-relative; its committed template is <env_file>_example
    service: str                           # compose service name
    compose_profile: str | None = None
    cuda_env: str = Field(pattern=r"^[A-Z_][A-Z0-9_]*$")   # env var the actuator sets from the cards' index
    drain: DrainSpec | None = None
    ready: str = "/health"
    timeout_sec: float = Field(600, gt=0)

    @field_validator("compose", "env_file")
    @classmethod
    def _repo_relative(cls, value: str | None) -> str | None:
        if value is not None and (value.startswith("/") or ".." in Path(value).parts):
            raise ValueError(f"{value!r} must be a repo-relative path")
        return value

    @field_validator("ready")
    @classmethod
    def _ready_path(cls, value: str) -> str:
        return _http_path(value)


class SwapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evicts: list[str] | Literal["all"]
    # Stage-4 bridge verbs (the controller maps them to its fixed gpu2 transitions); deleted in
    # stage 5. A seat without them must carry a `launch` on itself and every role it evicts.
    load: str | None = None
    unload: str | None = None
    after_wait_sec: float | None = Field(None, ge=0)   # per-seat override of defaults.swap_after_wait_sec
    guards: list[Literal["thermal", "visual_baseline"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bridge_pair(self):
        if (self.load is None) != (self.unload is None):
            raise ValueError("swap.load and swap.unload come as a pair")
        if len(set(self.guards)) != len(self.guards):
            raise ValueError("swap.guards has duplicates")
        return self

    @property
    def bridged(self) -> bool:
        return self.load is not None


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
    launch: LaunchSpec | None = None
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
    actuators: dict[str, ActuatorSpec] = Field(default_factory=dict)
    priorities: list[str] = Field(default_factory=lambda: list(PRIORITIES))
    cards: dict[str, CardSpec]
    roles: dict[str, RoleSpec]
    classes: dict[str, ClassSpec]
    routes: dict[str, RouteSpec] = Field(default_factory=dict)
    digest: str = ""
    source_text: str = Field("", exclude=True, repr=False)

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
            if role.launch is not None:
                if role.launch.actuator not in self.actuators:
                    errors.append(f"role {name}: launch.actuator {role.launch.actuator} is not in actuators")
                for card in role.cards:
                    if card in self.cards and self.cards[card].index is None:
                        errors.append(f"role {name}: has a launch but card {card} has no index")
        for name, role in self.roles.items():
            if role.swap is None or role.swap.bridged:
                continue
            unlaunched = [r for r in [name, *self.evicted_by(name)] if r in self.roles and self.roles[r].launch is None]
            if unlaunched:
                errors.append(f"role {name}: swap seat has no load/unload bridge, so it and every role it "
                              f"evicts need a launch; missing on {unlaunched}")
        for name, act in self.actuators.items():
            if act.host != self.host.name:
                errors.append(f"actuator {name}: host {act.host} is not the pool host {self.host.name}")
        indices: dict[int, str] = {}
        for card, spec in self.cards.items():
            if spec.index is None:
                continue
            if spec.index in indices:
                errors.append(f"cards {indices[spec.index]} and {card} share index {spec.index}")
            indices[spec.index] = card
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

    def swap_after_wait_sec(self, role: str) -> float:
        swap = self.roles[role].swap
        if swap is not None and swap.after_wait_sec is not None:
            return swap.after_wait_sec
        return self.defaults.swap_after_wait_sec

    def resident_roles(self) -> list[str]:
        """Roles loaded when no swap is active: everything without its own swap seat."""
        return [r for r, spec in self.roles.items() if spec.swap is None]


def launch_digest(cfg: PoolConfig, role: str) -> str:
    """sha256 over what an actuation of ``role`` would touch: its launch, its swap bridge, and the
    launch of every role it evicts. Pool and actuator each compute it from their own copy of
    config/gpu_pool.yaml; GpuActuateV1.launch_digest carries the pool's, and the actuator refuses on
    a mismatch (a stale checkout on one side must not start the wrong thing). Key order, comments
    and unrelated roles do not move it."""
    spec = cfg.roles[role]
    evicted = cfg.evicted_by(role)
    body = {
        "role": role,
        "cards": {c: cfg.cards[c].index for c in spec.cards},
        "launch": spec.launch.model_dump(mode="json", by_alias=True) if spec.launch else None,
        "swap": ({"load": spec.swap.load, "unload": spec.swap.unload, "evicts": sorted(evicted)}
                 if spec.swap else None),
        "evicted": {r: (cfg.roles[r].launch.model_dump(mode="json", by_alias=True) if cfg.roles[r].launch else None)
                    for r in sorted(evicted)},
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_pool_config(path: str | Path | None = None) -> PoolConfig:
    raw_bytes = Path(path or DEFAULT_PATH).read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    cfg = PoolConfig.model_validate(data)
    cfg.digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    cfg.source_text = raw_bytes.decode("utf-8", "replace")
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


_SUBST_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}$")


def _env_pairs(environment: Any) -> dict[str, str]:
    if isinstance(environment, dict):
        return {str(k): "" if v is None else str(v) for k, v in environment.items()}
    out: dict[str, str] = {}
    for item in environment or []:
        key, _, value = str(item).partition("=")
        out[key.strip()] = value.strip()
    return out


def _read_env_template(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.split(" #", 1)[0].strip().strip('"').strip("'")
    return out


def _resolve(value: str, template: dict[str, str]) -> str | None:
    """A compose value as the operator contract resolves it: a literal, a ``${VAR:-default}``
    default, or the committed ``.env_example`` value of ``${VAR}``. None when unresolvable."""
    value = value.strip().strip('"').strip("'")
    match = _SUBST_RE.match(value)
    if match is None:
        return None if "$" in value else value
    var, default = match.groups()
    if default:
        return default
    return template.get(var) or None


def check_launch(cfg: PoolConfig, root: str | Path) -> list[str]:
    """Every ``launch`` block must name something real in its compose file (stage 4 spec,
    "Validation additions"): the service exists (under its profile, if one is named), an llm role's
    service announces that role on that port, a service role publishes that host port, the service
    sets ``cuda_env``, and -- when the compose value resolves from the committed templates -- it
    points at the cards' ``index``. Static: reads compose + ``.env_example``, never a live ``.env``."""
    root = Path(root)
    problems: list[str] = []
    for name, role in cfg.roles.items():
        launch = role.launch
        if launch is None:
            continue
        where = f"role {name}: launch"
        compose_path = root / launch.compose
        if not compose_path.is_file():
            problems.append(f"{where}.compose {launch.compose} does not exist")
            continue
        template: dict[str, str] = {}
        if launch.env_file:
            example = root / (launch.env_file + "_example")
            if not example.is_file():
                problems.append(f"{where}.env_file {launch.env_file} has no committed {example.name}")
            template = _read_env_template(example)
        compose = yaml.safe_load(compose_path.read_text()) or {}
        service = (compose.get("services") or {}).get(launch.service)
        if service is None:
            problems.append(f"{where}.service {launch.service} is not a service in {launch.compose}")
            continue
        profiles = list(service.get("profiles") or [])
        if launch.compose_profile is not None and launch.compose_profile not in profiles:
            problems.append(f"{where}.compose_profile {launch.compose_profile} is not a profile of "
                            f"{launch.service} (has {profiles})")
        env = _env_pairs(service.get("environment"))
        if role.kind == "llm":
            if env.get("LLM_ROLE") != name:
                problems.append(f"{where}: {launch.service} sets LLM_ROLE={env.get('LLM_ROLE')}, not {name}")
            announced = _resolve(env.get("LLM_ANNOUNCE_PORT", ""), template)
            if announced != str(role.port):
                problems.append(f"{where}: {launch.service} announces port {announced}, pool expects {role.port}")
        else:
            host_ports = []
            for mapping in service.get("ports") or []:
                if isinstance(mapping, dict):
                    host_ports.append(str(mapping.get("published")))
                    continue
                parts = str(mapping).rsplit(":", 1)
                if len(parts) == 2:
                    host_ports.append(_resolve(parts[0], template))
            if str(role.port) not in host_ports:
                problems.append(f"{where}: {launch.service} publishes host ports {host_ports}, pool expects {role.port}")
        if launch.cuda_env not in env:
            problems.append(f"{where}: {launch.service} does not set {launch.cuda_env}")
        else:
            device = _resolve(env[launch.cuda_env], template)
            want = ",".join(str(cfg.cards[c].index) for c in role.cards if cfg.cards[c].index is not None)
            if device is not None and want and device != want:
                problems.append(f"{where}: {launch.service} {launch.cuda_env}={device}, cards {role.cards} "
                                f"have index {want}")
    return problems
