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
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "gpu_pool.yaml"
PRIORITIES = ("urgent", "interactive", "system", "background")
# Pool-side swap preconditions (stage 4 spec, "Guards"). A guard named here must be one the
# scheduler evaluates; the scheduler side lands with the actuation engine (stage 4.3).
SwapGuard = Literal["thermal"]   # visual_baseline deleted in stage 5.4
SWAP_GUARDS = get_args(SwapGuard)


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
    # Urgent (docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md).
    urgent_preempt_grace_sec: float = Field(5, ge=0)    # a hold paused for urgent work gets this long, then is aborted + re-queued in place
    urgent_max_concurrent: int = Field(3, ge=0)         # active urgent leases at once; 0 = urgent behaves like background (rollback)


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
    # Stage 5 meaning: the COMPOSE INTERPOLATION variable the actuator sets from the cards' index.
    # check_launch requires every device entry of the service to be exactly ${cuda_env} or
    # ${cuda_env:-<index>}, never a literal, so the actuator can always move the role.
    cuda_env: str = Field(pattern=r"^[A-Z_][A-Z0-9_]*$")
    # Stage 5: a model choice per role. profile_var is the compose interpolation variable the
    # actuator sets to the chosen config/llm_profiles.yaml profile (the service's LLM_PROFILE_NAME
    # must be ${profile_var} or ${profile_var:-...}); profiles is the allow-list, first = default.
    # No profiles -> the pool sends profile=None and compose's default fills the seat.
    profile_var: str | None = Field(None, pattern=r"^[A-Z_][A-Z0-9_]*$")
    profiles: list[str] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def _profiles_shape(self):
        if self.profiles and self.profile_var is None:
            raise ValueError("launch.profiles needs launch.profile_var (the variable the actuator sets)")
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("launch.profiles has duplicates")
        if self.profile_var is not None and self.profile_var == self.cuda_env:
            raise ValueError("launch.profile_var and launch.cuda_env must be different variables")
        return self


class SwapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evicts: list[str] | Literal["all"]
    # Stage 5.6 deleted the stage-4 bridge verbs (`load`/`unload`); extra="forbid" refuses a YAML
    # that still carries them. A seat is actuated only through the `launch` on itself and every
    # role it evicts.
    after_wait_sec: float | None = Field(None, ge=0)   # per-seat override of defaults.swap_after_wait_sec
    guards: list[SwapGuard] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_guards(self):
        if len(set(self.guards)) != len(self.guards):
            raise ValueError("swap.guards has duplicates")
        return self


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
    # Stage 5: roles that must never compute at the same time as this one (same card). Symmetric:
    # the scheduler places nothing on R while a lease is active on a role R lists or that lists R.
    serialize_with: list[str] = Field(default_factory=list)

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
    priority: Literal["urgent", "interactive", "system", "background"] = "system"


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
    # Durable-run holds on non-LLM roles (diffusion). Read by durable-runs' hold placement
    # only; the gateway serves ``routes`` and never lists these as models.
    hold_routes: dict[str, RouteSpec] = Field(default_factory=dict)
    digest: str = ""
    source_text: str = Field("", exclude=True, repr=False)
    _serial: dict[str, list[str]] = PrivateAttr(default_factory=dict)   # serialize_with, symmetric

    @field_validator("routes", "hold_routes", mode="before")
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
            if len(set(role.serialize_with)) != len(role.serialize_with):
                errors.append(f"role {name}: serialize_with has duplicates")
            for other_name in role.serialize_with:
                other = self.roles.get(other_name)
                if other_name == name:
                    errors.append(f"role {name}: serialize_with names itself")
                elif other is None:
                    errors.append(f"role {name}: serialize_with unknown role {other_name}")
                elif not set(other.cards) & set(role.cards):
                    errors.append(f"role {name}: serialize_with {other_name}, which shares no card with it")
            if role.launch is not None:
                if role.launch.actuator not in self.actuators:
                    errors.append(f"role {name}: launch.actuator {role.launch.actuator} is not in actuators")
                for card in role.cards:
                    if card in self.cards and self.cards[card].index is None:
                        errors.append(f"role {name}: has a launch but card {card} has no index")
        for name, role in self.roles.items():
            if role.swap is None:
                continue
            if role.operator_only and role.launch is None:
                # Operator-only seat nothing can load yet (experiment, stage 5 Decision 3). Stage 5.7:
                # the pool refuses any operator lease on it (``not_actuatable_reason``), and the
                # scheduler never drains its residents for one, so nothing can empty its cards.
                continue
            unlaunched = [r for r in [name, *self.evicted_by(name)] if r in self.roles and self.roles[r].launch is None]
            if unlaunched:
                errors.append(f"role {name}: swap seat and every role it evicts need a launch; "
                              f"missing on {unlaunched}")
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
        for name, route in self.hold_routes.items():
            if route.work_class not in self.classes:
                errors.append(f"hold route {name}: unknown class {route.work_class}")
            if name in self.routes:
                errors.append(f"hold route {name}: also a gateway route")
        ports: dict[int, str] = {}
        for name, role in self.roles.items():
            if role.port in ports:
                errors.append(f"roles {ports[role.port]} and {name} share port {role.port}")
            ports[role.port] = name
        if errors:
            raise ValueError("; ".join(errors))
        # serialize_with read symmetrically, once: the scheduler asks per placement check.
        self._serial = {r: sorted(set(spec.serialize_with)
                                  | {o for o, other in self.roles.items() if r in other.serialize_with})
                        for r, spec in self.roles.items()}
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

    def load_profile(self, role: str) -> str | None:
        """The profile the pool sends with a ``load`` of ``role``: the first ``launch.profiles``
        entry (the allow-list's default). None when the role lists no profiles -> the actuator leaves
        compose's own default. Choosing another entry by vision / per-slot ctx / VRAM is not built
        yet (stage 5 spec, Decision 2 and "Corrections from building 5.2" 6)."""
        launch = self.roles[role].launch
        return launch.profiles[0] if launch is not None and launch.profiles else None

    def actuated_seats(self) -> frozenset[str]:
        """Stage 5.7: the swap seats the pool loads and unloads itself. A seat is actuated if and only
        if it has a ``launch`` block (which names its actuator); there is no second list to drift from
        this YAML (GPU_POOL_ACTUATE_ROLES was deleted). The validator already requires a launch on every
        non-operator swap seat and on every role a launched seat evicts."""
        return frozenset(r for r, spec in self.roles.items() if spec.swap is not None and spec.launch is not None)

    def not_actuatable_reason(self, work_class: str) -> str | None:
        """Why an operator lease on ``work_class`` must be refused, or None. A swap seat without a
        launch block (``experiment``, deferred) can never be loaded, so granting its lease would only
        drain every resident it evicts and leave the cards empty (stage 5 "Corrections from building
        5.1" item 5)."""
        cls = self.classes.get(work_class)
        for role in (cls.roles if cls else []):
            spec = self.roles[role]
            if spec.swap is not None and spec.launch is None:
                return f"not_actuatable:{role}"
        return None

    def serialized_with(self, role: str) -> list[str]:
        """``serialize_with`` read symmetrically: what ``role`` lists plus every role listing it."""
        return self._serial.get(role, [])

    def resident_roles(self) -> list[str]:
        """Roles loaded when no swap is active: everything without its own swap seat."""
        return [r for r, spec in self.roles.items() if spec.swap is None]


def launch_digest(cfg: PoolConfig, role: str) -> str:
    """sha256 over what an actuation of ``role`` would touch: its launch, its swap evictions, and the
    launch of every role it evicts. Pool and actuator each compute it from their own copy of
    config/gpu_pool.yaml; GpuActuateV1.launch_digest carries the pool's, and the actuator refuses on
    a mismatch (a stale checkout on one side must not start the wrong thing). Key order, comments
    and unrelated roles do not move it.

    Taken over the *parsed* model, not the YAML text: a new defaulted LaunchSpec field changes every
    digest even for identical YAML, so pool and actuator must run the same parser version before
    actuation is armed. That is intended -- they must agree on meaning, not just bytes."""
    evicted = cfg.evicted_by(role)

    def one(name: str) -> dict[str, Any]:
        spec = cfg.roles[name]
        launch = spec.launch
        return {
            "kind": spec.kind, "port": spec.port,
            "cards": {c: cfg.cards[c].index for c in spec.cards},
            "launch": launch.model_dump(mode="json", by_alias=True) if launch else None,
            "actuator_host": (cfg.actuators[launch.actuator].host
                              if launch and launch.actuator in cfg.actuators else None),
        }

    spec = cfg.roles[role]
    body = {
        "role": role, **one(role),
        # "load"/"unload" are the stage-4 bridge verbs, deleted in 5.6 and always null since the 5.3
        # cutover. The nulls stay in the hashed body so every digest keeps its deployed value: a pool
        # and an actuator on either side of 5.6 still agree, and 5.6 needs no lockstep deploy
        # (test_stage5_6_digest_is_unchanged_by_the_bridge_removal pins it).
        "swap": ({"load": None, "unload": None, "evicts": sorted(evicted)} if spec.swap else None),
        "evicted": {r: one(r) for r in sorted(evicted)},
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


_SUBST_HEAD_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(:?-)?")
# Container variables that pick a GPU. A launch role's service must set each one it uses only
# through its launch.cuda_env interpolation (stage 5, "Meaning change: cuda_env, plus a gate").
DEVICE_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES_OVERRIDE", "NVIDIA_VISIBLE_DEVICES")


def _parse_subst(value: str) -> tuple[str, str | None, str | None] | None:
    """``${VAR}``, ``${VAR:-d}`` or ``${VAR-d}`` spanning the WHOLE value -> (VAR, op, d); the
    default may itself be an interpolation (``${A:-${B}}``, which compose accepts). None otherwise
    (a literal, or anything mixing text and interpolations)."""
    match = _SUBST_HEAD_RE.match(value)
    if match is None:
        return None
    depth, i = 1, 2
    while i < len(value) and depth:
        if value.startswith("${", i):
            depth, i = depth + 1, i + 2
            continue
        if value[i] == "}":
            depth -= 1
        i += 1
    if depth or i != len(value):
        return None
    var, op = match.group(1), match.group(2)
    if op is None:
        return (var, None, None) if match.end() == len(value) - 1 else None
    return var, op, value[match.end():-1]


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
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.split(" #", 1)[0].strip().strip('"').strip("'")
    return out


def _resolve(value: str, template: dict[str, str]) -> str | None:
    """A compose value as Docker Compose would resolve it against the committed ``.env_example``
    (the operator contract): the env value wins; ``${VAR:-d}`` falls back to d when VAR is unset
    OR empty, ``${VAR-d}`` only when unset. A literal is itself. None when unresolvable."""
    value = value.strip().strip('"').strip("'")
    parsed = _parse_subst(value)
    if parsed is None:
        return None if "$" in value else value
    var, op, default = parsed
    current = template.get(var)
    fallback = _resolve(default, template) if default else None
    if op == ":-":
        return current or fallback
    if op == "-":
        return current if var in template else fallback
    return current or None


def _interpolates(value: str, var: str) -> tuple[bool, str | None]:
    """Is ``value`` exactly ``${var}`` or ``${var:-<default>}``? -> (ok, default)."""
    parsed = _parse_subst(value.strip().strip('"').strip("'"))
    if parsed is None or parsed[0] != var or parsed[1] not in (None, ":-"):
        return False, None
    return True, parsed[2]


def _known_profiles(root: Path) -> set[str] | None:
    """The profiles in ``<root>/config/llm_profiles.yaml`` -- the tree being checked, never another."""
    path = root / "config" / "llm_profiles.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    return set((data.get("profiles") or {}).keys())


def check_launch(cfg: PoolConfig, root: str | Path) -> list[str]:
    """Every ``launch`` block must name something real in its compose file (stage 4 spec,
    "Validation additions"): the service exists (under its profile, if one is named), an llm role's
    service announces that role on that port, a service role publishes that host port, the service
    picks its GPU only through ``${cuda_env}`` / ``${cuda_env:-<index>}`` (never a literal), and --
    when that resolves from the committed templates -- it points at the cards' ``index``. With a
    ``profile_var``, ``LLM_PROFILE_NAME`` must interpolate it, and every ``profiles`` entry must be a
    config/llm_profiles.yaml profile. Static: reads compose + ``.env_example``, never a live ``.env``."""
    root = Path(root)
    problems: list[str] = []
    known_profiles = _known_profiles(root)
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
            llm_role = _resolve(env.get("LLM_ROLE", ""), template)
            if llm_role != name:
                problems.append(f"{where}: {launch.service} sets LLM_ROLE={llm_role}, not {name}")
            announced = _resolve(env.get("LLM_ANNOUNCE_PORT", ""), template)
            if announced != str(role.port):
                problems.append(f"{where}: {launch.service} announces port {announced}, pool expects {role.port}")
        else:
            host_ports = []
            for mapping in service.get("ports") or []:
                if isinstance(mapping, dict):
                    host_ports.append(_resolve(str(mapping.get("published", "")), template))
                    continue
                # [ip:]host:container, where each part may be ${VAR...} (no ':' inside ours)
                parts = re.findall(r"\$\{[^}]*\}|[^:]+", str(mapping).split("/", 1)[0])
                if len(parts) >= 2:
                    host_ports.append(_resolve(parts[-2], template))
            if str(role.port) not in host_ports:
                problems.append(f"{where}: {launch.service} publishes host ports {host_ports}, pool expects {role.port}")
        want = ",".join(str(cfg.cards[c].index) for c in role.cards if cfg.cards[c].index is not None)
        device_keys = [k for k in DEVICE_KEYS if k in env]
        if not device_keys:
            problems.append(f"{where}: {launch.service} sets none of {list(DEVICE_KEYS)}, so "
                            f"{launch.cuda_env} cannot place it")
        for key in device_keys:
            ok, default = _interpolates(env[key], launch.cuda_env)
            if not ok:
                problems.append(f"{where}: {launch.service} {key}={env[key]} must be ${{{launch.cuda_env}}} or "
                                f"${{{launch.cuda_env}:-{want}}} (the actuator sets it from the card index; "
                                f"a literal device cannot move)")
                continue
            if default is not None and "$" not in default and want and default != want:
                problems.append(f"{where}: {launch.service} {key} defaults to {default}, cards {role.cards} "
                                f"have index {want}")
            device = _resolve(env[key], template)
            if device is not None and want and device != want:
                problems.append(f"{where}: {launch.service} {key}={device} (from templates), cards {role.cards} "
                                f"have index {want}")
        pinned = _pinned_device_ids(service, launch.cuda_env)
        if pinned:
            problems.append(f"{where}: {launch.service} pins device_ids {pinned}; the GPU must come from "
                            f"{launch.cuda_env}")
        if launch.profile_var is not None:
            ok, _ = _interpolates(env.get("LLM_PROFILE_NAME", ""), launch.profile_var)
            if not ok:
                problems.append(f"{where}: {launch.service} LLM_PROFILE_NAME={env.get('LLM_PROFILE_NAME')} must be "
                                f"${{{launch.profile_var}}} or ${{{launch.profile_var}:-...}}")
        if launch.profiles and role.kind != "llm":
            problems.append(f"{where}.profiles: only llm roles load llm_profiles.yaml profiles")
        if launch.profiles and known_profiles is None:
            problems.append(f"{where}.profiles: config/llm_profiles.yaml not found to check them against")
        for profile in launch.profiles if known_profiles is not None else []:
            if profile not in known_profiles:
                problems.append(f"{where}.profiles: {profile} is not a profile in config/llm_profiles.yaml")
                continue
            # Stage 7.2: a profile on the PrismML fork (llamacpp.server_build: prism) only boots in an
            # image that carries it; the wrapper refuses otherwise, so a load would fail on circe.
            build = _profile_server_build(root, profile)
            dockerfile = str((service.get("build") or {}).get("dockerfile") or "") \
                if isinstance(service.get("build"), dict) else ""
            if build == "prism" and not dockerfile.endswith(PRISM_DOCKERFILE):
                problems.append(f"{where}.profiles: {profile} needs llamacpp.server_build=prism but "
                                f"{launch.service} builds from {dockerfile or '(no build section)'}, "
                                f"not {PRISM_DOCKERFILE}")
    return problems


PRISM_DOCKERFILE = "Dockerfile.prism"


def _profile_server_build(root: Path, profile: str) -> str | None:
    data = yaml.safe_load((root / "config" / "llm_profiles.yaml").read_text()) or {}
    llamacpp = ((data.get("profiles") or {}).get(profile) or {}).get("llamacpp") or {}
    return llamacpp.get("server_build")


def _pinned_device_ids(service: dict[str, Any], cuda_env: str) -> list[str]:
    """``device_ids`` under deploy.resources.reservations.devices (or a ``gpus`` list) that are not
    ``${cuda_env}``/``${cuda_env:-...}``: a second, compose-level device pin the actuator cannot move."""
    out: list[str] = []
    devices = (((service.get("deploy") or {}).get("resources") or {}).get("reservations") or {}).get("devices") or []
    gpus = service.get("gpus")
    if isinstance(gpus, list):
        devices = [*devices, *gpus]
    for dev in devices:
        if isinstance(dev, dict):
            for d in dev.get("device_ids") or []:
                if not _interpolates(str(d), cuda_env)[0]:
                    out.append(str(d))
    return out
