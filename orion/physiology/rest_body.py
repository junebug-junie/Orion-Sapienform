"""Orion's resting body as data: ``config/physiology/rest_body.yaml``.

Per host, the whole-machine (chassis) watts at rest, PINNED and dated. Not a rolling percentile:
live 2026-10-28, the Zelda dojo raised athena's daily p05 by ~350 W, so a rolling rest would have
called that work "rest" (docs/superpowers/specs/2026-10-30-orion-energy-allowance-proposal.md).

THE GUARD: a host whose ``rest_w`` exceeds its ``approved_rest_w`` is refused (absent from
``hosts``, its reason in ``refused``). Raising rest hides activity, so only a reviewed change to
``approved_rest_w`` can raise it. A missing or malformed file is a reason, never a rest of 0.

``sponsored``: workloads Juniper runs (container globs per host, optional end date). Their energy
above rest is reported apart from Orion's activity. Pure parsing plus one file read; the caller
passes ``now``.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
REST_BODY_PATH = _REPO_ROOT / "config" / "physiology" / "rest_body.yaml"
VERSION = 1


@dataclass(frozen=True)
class HostRest:
    host: str
    rest_w: float
    approved_rest_w: float
    measured: str
    method: str
    approved_at: Optional[str]


@dataclass(frozen=True)
class Sponsored:
    name: str
    containers: str
    host: str
    until: Optional[date]
    note: str = ""

    def active(self, at: datetime) -> bool:
        return self.until is None or at.date() < self.until


@dataclass(frozen=True)
class RestBodyConfig:
    hosts: Mapping[str, HostRest]
    sponsored: tuple[Sponsored, ...] = ()
    refused: Mapping[str, str] = field(default_factory=dict)

    def rest_w(self, host: str) -> tuple[Optional[float], Optional[str]]:
        """(rest_w, None) or (None, reason)."""
        if host in self.hosts:
            return self.hosts[host].rest_w, None
        return None, self.refused.get(host) or f"no rest for {host} in rest_body.yaml"

    def sponsored_containers(self, host: str, at: datetime) -> list[str]:
        """The active sponsored container globs on ``host`` at ``at``."""
        return [s.containers for s in self.sponsored if s.host == host and s.active(at)]


def is_sponsored(container: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(container, g) for g in globs)


def _watts(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0:
        raise ValueError(f"{what}: not a non-negative number: {v!r}")
    return float(v)


def parse_rest_body(doc: Any) -> RestBodyConfig:
    """YAML document -> config. Raises ValueError on a malformed document; a host over its
    approved rest is refused (listed in ``refused``), not raised, so the other hosts still read."""
    if not isinstance(doc, dict) or doc.get("version") != VERSION:
        raise ValueError(f"not a rest_body version {VERSION} document")
    hosts: dict[str, HostRest] = {}
    refused: dict[str, str] = {}
    for name, e in (doc.get("hosts") or {}).items():
        if not isinstance(e, dict):
            raise ValueError(f"hosts.{name}: not a mapping")
        rest = _watts(e.get("rest_w"), f"hosts.{name}.rest_w")
        approved = _watts(e.get("approved_rest_w"), f"hosts.{name}.approved_rest_w")
        if rest > approved:
            refused[str(name)] = (f"rest_body refused for {name}: rest_w {rest:g} W > approved_rest_w {approved:g} W "
                                  "(raising rest needs Juniper's approval)")
            continue
        hosts[str(name)] = HostRest(host=str(name), rest_w=rest, approved_rest_w=approved,
                                    measured=str(e.get("measured") or ""), method=str(e.get("method") or ""),
                                    approved_at=str(e["approved_at"]) if e.get("approved_at") else None)
    sponsored = []
    for i, s in enumerate(doc.get("sponsored") or []):
        if not isinstance(s, dict) or not s.get("containers") or not s.get("host"):
            raise ValueError(f"sponsored[{i}]: need name, containers, host")
        until = s.get("until")
        if until is not None and not isinstance(until, date):
            until = date.fromisoformat(str(until))
        sponsored.append(Sponsored(name=str(s.get("name") or s["containers"]), containers=str(s["containers"]),
                                   host=str(s["host"]), until=until, note=str(s.get("note") or "")))
    return RestBodyConfig(hosts=hosts, sponsored=tuple(sponsored), refused=refused)


def load_rest_body(path: str | Path = REST_BODY_PATH) -> tuple[Optional[RestBodyConfig], Optional[str]]:
    """(config, None) or (None, reason). Never raises."""
    import yaml

    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError:
        return None, f"missing:{p.name}"
    except OSError as exc:
        return None, f"read_failed:{p.name}:{type(exc).__name__}"
    try:
        return parse_rest_body(yaml.safe_load(text)), None
    except Exception as exc:  # noqa: BLE001 -- malformed config is a reason
        return None, f"malformed:{p.name}:{str(exc)[:120]}"


__all__ = ["HostRest", "REST_BODY_PATH", "RestBodyConfig", "Sponsored", "is_sponsored", "load_rest_body",
           "parse_rest_body"]
