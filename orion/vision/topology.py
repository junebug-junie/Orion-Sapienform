"""Camera topology: the portals between Orion's street cameras.

Config: ``config/vision_topology.yaml`` (override with ``VISION_TOPOLOGY_PATH``).
Design: docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md,
section 2. One consumer: orion-sql-writer's handoff reducer
(``app/vision_handoff.py``).

Three portal kinds:

- ``overlap``: both streams see the same place at the same time
  (``max_skew_sec``).
- ``approach``: a story on ``from.stream`` moving with ``from.dir_depth`` is
  followed by one of ``to.event`` (vision_events types) on ``to.stream``
  inside ``prior_sec``.
- ``gap``: a story leaving ``from.stream`` through ``from.edge`` reappears on
  ``to.stream`` through ``to.edge`` after crossing ``prior_gap_frame_widths``
  of unseen ground.

Only priors live in the file; learned values live in the database. A bad file
raises ``TopologyError`` with the reason, so the reducer refuses to start on a
half-valid topology instead of silently dropping a portal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

DEFAULT_TOPOLOGY_PATH = Path(__file__).resolve().parents[2] / "config" / "vision_topology.yaml"
PORTAL_KINDS = ("overlap", "approach", "gap")
EDGES = ("left", "right", "top", "bottom")


class TopologyError(ValueError):
    """config/vision_topology.yaml is malformed."""


@dataclass(frozen=True)
class Portal:
    id: str
    kind: str
    from_stream: Optional[str] = None
    to_stream: Optional[str] = None
    streams: Tuple[str, ...] = ()
    from_edge: Optional[str] = None
    to_edge: Optional[str] = None
    from_dir_depth: Optional[int] = None
    to_events: Tuple[str, ...] = ()
    max_skew_sec: float = 0.0
    prior_sec: Tuple[float, float] = (0.0, 0.0)
    prior_gap_frame_widths: float = 0.0

    @property
    def lateral_sign(self) -> int:
        """Gap portals: the dir_lateral a departure through ``from_edge`` has
        (+1 left-to-right exits right; -1 exits left). Arrival through
        ``to_edge`` has the same sign: entering on the left means moving
        right."""
        return {"right": 1, "left": -1}.get(self.from_edge or "", 0)

    @property
    def arrival_sign(self) -> int:
        return {"left": 1, "right": -1}.get(self.to_edge or "", 0)


@dataclass(frozen=True)
class Topology:
    portals: Tuple[Portal, ...]
    place_zones: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def by_id(self, portal_id: str) -> Optional[Portal]:
        return next((p for p in self.portals if p.id == portal_id), None)

    def of_kind(self, kind: str) -> Tuple[Portal, ...]:
        return tuple(p for p in self.portals if p.kind == kind)


def _req(d: dict, key: str, pid: str):
    if key not in d or d[key] in (None, ""):
        raise TopologyError(f"portal {pid!r}: missing {key!r}")
    return d[key]


def _stream(d: dict, pid: str, side: str) -> dict:
    v = _req(d, side, pid)
    if not isinstance(v, dict) or not str(v.get("stream") or "").strip():
        raise TopologyError(f"portal {pid!r}: {side!r} needs a stream")
    return v


def parse_topology(raw: dict) -> Topology:
    if not isinstance(raw, dict) or int(raw.get("version") or 0) != 1:
        raise TopologyError("topology needs version: 1")
    portals = []
    seen = set()
    for p in raw.get("portals") or []:
        pid = str((p or {}).get("id") or "").strip()
        if not pid:
            raise TopologyError("every portal needs an id")
        if pid in seen:
            raise TopologyError(f"duplicate portal id {pid!r}")
        seen.add(pid)
        kind = str(p.get("kind") or "")
        if kind not in PORTAL_KINDS:
            raise TopologyError(f"portal {pid!r}: kind must be one of {PORTAL_KINDS}, got {kind!r}")
        if kind == "overlap":
            streams = tuple(str(s) for s in _req(p, "streams", pid))
            if len(streams) != 2 or streams[0] == streams[1]:
                raise TopologyError(f"portal {pid!r}: overlap needs two different streams")
            skew = float(_req(p, "max_skew_sec", pid))
            if skew < 0:
                raise TopologyError(f"portal {pid!r}: max_skew_sec must be >= 0")
            portals.append(Portal(id=pid, kind=kind, streams=streams, max_skew_sec=skew))
        elif kind == "approach":
            f, t = _stream(p, pid, "from"), _stream(p, pid, "to")
            depth = int(_req(f, "dir_depth", pid))
            if depth not in (-1, 1):
                raise TopologyError(f"portal {pid!r}: from.dir_depth must be -1 or 1")
            events = tuple(str(e) for e in (t.get("event") or []))
            if not events:
                raise TopologyError(f"portal {pid!r}: to.event needs at least one event type")
            lo, hi = (float(v) for v in _req(p, "prior_sec", pid))
            if not 0 <= lo < hi:
                raise TopologyError(f"portal {pid!r}: prior_sec must be [lo, hi] with 0 <= lo < hi")
            portals.append(Portal(id=pid, kind=kind, from_stream=str(f["stream"]), to_stream=str(t["stream"]),
                                  from_dir_depth=depth, to_events=events, prior_sec=(lo, hi)))
        else:
            f, t = _stream(p, pid, "from"), _stream(p, pid, "to")
            fe, te = str(f.get("edge") or ""), str(t.get("edge") or "")
            if fe not in ("left", "right") or te not in ("left", "right"):
                raise TopologyError(f"portal {pid!r}: gap edges must be left or right (got {fe!r}, {te!r})")
            if f["stream"] == t["stream"]:
                raise TopologyError(f"portal {pid!r}: a gap joins two different streams")
            gap = float(_req(p, "prior_gap_frame_widths", pid))
            if gap <= 0:
                raise TopologyError(f"portal {pid!r}: prior_gap_frame_widths must be > 0")
            portals.append(Portal(id=pid, kind=kind, from_stream=str(f["stream"]), to_stream=str(t["stream"]),
                                  from_edge=fe, to_edge=te, prior_gap_frame_widths=gap))
    if not portals:
        raise TopologyError("topology has no portals")
    places = {str(k): tuple(str(z) for z in (v or [])) for k, v in (raw.get("place_zones") or {}).items()}
    return Topology(portals=tuple(portals), place_zones=places)


def load_topology(path: Optional[str | Path] = None) -> Topology:
    p = Path(path or os.getenv("VISION_TOPOLOGY_PATH") or DEFAULT_TOPOLOGY_PATH)
    if not p.exists():
        raise TopologyError(f"topology file not found: {p}")
    return parse_topology(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
