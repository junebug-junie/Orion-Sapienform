"""Camera zones: which part of the picture a box is in, and whether it may be embedded.

Shared by orion-vision-host (skip crop embeddings in no-embed zones), the
sql-writer individuals reducer (zone + dwell on sightings, patio presence),
and the attention score. Config: ``config/vision_zones.yaml``.

The patio rule lives here and in a DB CHECK constraint, not in prose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

DEFAULT_ZONES_PATH = Path(__file__).resolve().parents[2] / "config" / "vision_zones.yaml"


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: Tuple[Tuple[float, float], ...]
    embed: bool = True
    dwell_rare_sec: Optional[float] = None


def _point_in_polygon(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def load_zones(path: Optional[str | Path] = None) -> Dict[str, List[Zone]]:
    """stream_id -> ordered zones. Missing file means no zones anywhere."""
    p = Path(path or os.getenv("VISION_ZONES_PATH") or DEFAULT_ZONES_PATH)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: Dict[str, List[Zone]] = {}
    for stream_id, cfg in (raw.get("streams") or {}).items():
        zones = []
        for z in (cfg or {}).get("zones") or []:
            poly = tuple((float(a), float(b)) for a, b in z["polygon"])
            if len(poly) < 3:
                raise ValueError(f"zone {z.get('name')} on {stream_id} needs >= 3 points")
            dwell = z.get("dwell_rare_sec")
            zones.append(Zone(
                name=str(z["name"]),
                polygon=poly,
                embed=bool(z.get("embed", True)),
                dwell_rare_sec=float(dwell) if dwell is not None else None,
            ))
        out[str(stream_id)] = zones
    return out


def zone_for_box(
    zones: Sequence[Zone], box_xyxy: Sequence[float], width: float, height: float
) -> Optional[Zone]:
    """First zone containing the box's bottom-center point, else None."""
    if not zones or width <= 0 or height <= 0 or len(box_xyxy) != 4:
        return None
    x1, _y1, x2, y2 = (float(v) for v in box_xyxy)
    # Clamp into the frame: a box touching (or past) the bottom edge must land
    # inside a polygon whose edge is y=1.0, not fall outside every zone.
    eps = 1e-6
    px = min(max(((x1 + x2) / 2.0) / float(width), 0.0), 1.0 - eps)
    py = min(max(y2 / float(height), 0.0), 1.0 - eps)
    for z in zones:
        if _point_in_polygon(px, py, z.polygon):
            return z
    return None


def may_embed(zone: Optional[Zone]) -> bool:
    return zone is None or zone.embed
