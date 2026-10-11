"""The discovery bridge: which model actually fills each role right now.

Three facts that existed separately and were never joined:
  1. every llama.cpp worker container carries ``LLM_PROFILE_NAME`` (a ``config/llm_profiles.yaml``
     key) and now announces it, with its role and port, on ``orion:llm:worker:announce``;
  2. ``config/llm_profiles.yaml`` names the model file for that profile (``llamacpp.hf_filename``);
  3. llama.cpp's own ``GET /props`` reports the file actually loaded, slots, ctx per slot, vision.

A role is ``confirmed`` only when all three agree. Anything else gets no grants and is shown in
red: ``mismatch`` (announced profile's file is not what is loaded), ``silent`` (the port answers
but nobody announced, or the announcement went stale), ``down`` (no answer). This module is pure;
the runtime does the HTTP and bus I/O and hands the results in.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from orion.gpu_pool.config import PoolConfig
from orion.gpu_pool.scheduler import CardLive, RoleLive
from orion.schemas.gpu_pool import DiscoveredRoleV1, LlmWorkerAnnounceV1

PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "llm_profiles.yaml"


def load_profiles(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(Path(path or PROFILES_PATH).read_text()) or {}
    return dict(data.get("profiles") or {})


def profile_model_file(profile: dict[str, Any]) -> str | None:
    llamacpp = profile.get("llamacpp") or {}
    name = llamacpp.get("hf_filename") or llamacpp.get("model_file") or llamacpp.get("model_path")
    return PurePosixPath(str(name)).name if name else None


@dataclass(frozen=True)
class Probe:
    """One HTTP look at a role's port. ``props`` is llama.cpp /props JSON, None for services."""

    ok: bool
    props: dict[str, Any] | None = None
    error: str | None = None
    checked_at: datetime | None = None
    backend: str = "llamacpp"


def resolve_roles(
    cfg: PoolConfig,
    profiles: dict[str, dict[str, Any]],
    announcements: dict[str, LlmWorkerAnnounceV1],
    probes: dict[str, Probe],
    cards: dict[str, CardLive],
    now: datetime,
    announce_stale_sec: float = 120.0,
) -> tuple[list[DiscoveredRoleV1], dict[str, RoleLive], list[str]]:
    from orion.gpu_pool.scheduler import _Ctx  # evicted/loaded logic lives with the scheduler

    view = _Ctx(cfg, {}, cards, now, {}, {}, set(), set(), set())
    discovered: list[DiscoveredRoleV1] = []
    live: dict[str, RoleLive] = {}

    for role, spec in cfg.roles.items():
        url = cfg.url(role)
        probe = probes.get(role)
        base = dict(role=role, kind=spec.kind, cards=list(spec.cards), url=url,
                    checked_at=probe.checked_at if probe else None)

        if not view.loaded(role):
            status = "evicted" if view.evicted(role) else "unloaded"
            discovered.append(DiscoveredRoleV1(**base, status=status))
            live[role] = RoleLive(role, False)
            continue

        if spec.kind == "service":
            healthy = bool(probe and probe.ok)
            discovered.append(DiscoveredRoleV1(
                **base, status="static" if healthy else "down", slots=spec.slots or 0,
                vram_gb=spec.vram_gb, detail=None if healthy else (probe.error if probe else "not probed")))
            live[role] = RoleLive(role, healthy, spec.slots or 0)
            continue

        ann = announcements.get(role)
        if ann is not None and (now - ann.announced_at).total_seconds() > announce_stale_sec:
            ann = None
        if probe is None or not probe.ok or not probe.props:
            discovered.append(DiscoveredRoleV1(
                **base, status="down", profile_name=ann.profile_name if ann else None,
                detail=probe.error if probe else "not probed"))
            live[role] = RoleLive(role, False)
            continue

        if spec.backend == "vllm":
            row, role_live = _resolve_vllm(cfg, role, base, profiles, ann, probe)
            discovered.append(row)
            live[role] = role_live
            continue
        if probe.backend != spec.backend:
            discovered.append(DiscoveredRoleV1(**base, status="mismatch", detail="probe backend mismatch"))
            live[role] = RoleLive(role, False)
            continue

        props = probe.props
        loaded_file = PurePosixPath(str(props.get("model_path") or "")).name or None
        slots = int(props.get("total_slots") or 0)
        ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        vision = bool((props.get("modalities") or {}).get("vision"))
        facts = dict(model_file=loaded_file, model_path=str(props.get("model_path") or "") or None,
                     slots=slots, ctx_per_slot=int(ctx) if ctx else None, vision=vision)

        if ann is None:
            discovered.append(DiscoveredRoleV1(**base, status="silent", **facts,
                                               detail="port answers but no fresh announcement"))
            live[role] = RoleLive(role, False, slots, facts["ctx_per_slot"], vision)
            continue
        if ann.port != spec.port:
            discovered.append(DiscoveredRoleV1(**base, status="mismatch", profile_name=ann.profile_name, **facts,
                                               detail=f"announced port {ann.port} != role port {spec.port}"))
            live[role] = RoleLive(role, False, slots, facts["ctx_per_slot"], vision)
            continue
        profile = profiles.get(ann.profile_name)
        expected = profile_model_file(profile) if profile else None
        if profile is None or expected != loaded_file:
            why = ("profile not in llm_profiles.yaml" if profile is None
                   else f"profile expects {expected}, server loaded {loaded_file}")
            discovered.append(DiscoveredRoleV1(**base, status="mismatch", profile_name=ann.profile_name,
                                               **facts, detail=why))
            live[role] = RoleLive(role, False, slots, facts["ctx_per_slot"], vision)
            continue
        discovered.append(DiscoveredRoleV1(**base, status="confirmed", profile_name=ann.profile_name, **facts))
        live[role] = RoleLive(role, slots > 0, slots, facts["ctx_per_slot"], vision)

    unclaimed = sorted(
        f"{a.host}:{a.port} role={a.role} profile={a.profile_name}"
        for a in announcements.values()
        if a.role not in cfg.roles and (now - a.announced_at).total_seconds() <= announce_stale_sec
    )
    return discovered, live, unclaimed


def _resolve_vllm(cfg, role, base, profiles, ann, probe):
    """Confirm the alias AND loaded path, live scheduler capacity, and all TP cards.

    /orion/server-info is the running engine's config, not Orion's desired launch config.
    No /props fabrication, guessed slots, or config-only readiness.
    """
    facts = {}
    status, detail = "mismatch", "invalid vLLM discovery"
    try:
        if probe.backend != "vllm":
            raise ValueError("probe backend mismatch")
        info = probe.props["server_info"]["vllm_config"]
        model = info["model_config"]
        path = model["model"]
        ctx = model["max_model_len"]
        slots = info["scheduler_config"]["max_num_seqs"]
        tp = info["parallel_config"]["tensor_parallel_size"]
        if any(info["parallel_config"].get(key) != 1 for key in
               ("pipeline_parallel_size", "data_parallel_size")):
            raise ValueError("pool vLLM roles require a single TP replica")
        ids = [m["id"] for m in probe.props["models"]["data"]]
        if any(type(v) is not int or v <= 0 for v in (ctx, slots, tp)):
            raise ValueError("invalid live capacity")
        if not isinstance(path, str) or not path or not ids:
            raise ValueError("missing loaded model identity")
        if ann is None:
            status = "silent"
            raise ValueError("port answers but no fresh announcement")
        spec = cfg.roles[role]
        if ann.host != cfg.role_host(role) or ann.port != spec.port:
            raise ValueError("announcement host/port differs from role")
        profile = profiles.get(ann.profile_name) or {}
        serving = profile.get("vllm") or {}
        alias = serving.get("served_model_name")
        if profile.get("backend") != "vllm" or not alias or alias not in ids:
            raise ValueError("profile backend or served model differs from engine")
        if path != profile.get("model_id"):
            raise ValueError("profile model path differs from engine")
        if tp != len(spec.cards) or tp != (profile.get("gpu") or {}).get("tensor_parallel_size"):
            raise ValueError("tensor parallel size differs from allocated cards")
        devices = [str(cfg.cards[c].index) for c in spec.cards]
        if ann.cuda_visible_devices != ",".join(devices):
            raise ValueError("announced CUDA devices differ from allocated cards")
        facts = dict(model_file=alias, model_path=path, slots=slots, ctx_per_slot=ctx, vision=None)
        status, detail = "confirmed", None
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        detail = str(exc)
    row = DiscoveredRoleV1(**base, status=status, profile_name=ann.profile_name if ann else None,
                           detail=detail, **facts)
    return row, RoleLive(role, status == "confirmed", row.slots, row.ctx_per_slot, False)
