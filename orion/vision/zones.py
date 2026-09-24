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


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def on_seg(p, q, r):
        return min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and \
            min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (o1 == 0 and on_seg(a, c, b)) or (o2 == 0 and on_seg(a, d, b)) or \
        (o3 == 0 and on_seg(c, a, d)) or (o4 == 0 and on_seg(c, b, d))


def _rect_intersects_polygon(rect: Tuple[float, float, float, float], poly: Sequence[Tuple[float, float]]) -> bool:
    x1, y1, x2, y2 = rect
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in poly):
        return True
    if any(_point_in_polygon(cx, cy, poly) for cx, cy in corners):
        return True
    rect_edges = list(zip(corners, corners[1:] + corners[:1]))
    poly_edges = list(zip(poly, list(poly[1:]) + [poly[0]]))
    return any(_segments_cross(a, b, c, d) for a, b in rect_edges for c, d in poly_edges)


def intersects_no_embed(
    zones: Sequence[Zone], box_xyxy: Sequence[float], width: float, height: float
) -> bool:
    """Does any part of the box overlap a no-embed zone (the patio)?

    Stricter than ``zone_for_box``, which places a box by its bottom-center
    only: a person standing just outside the patio can still have patio
    pixels in their crop. Anything that would store those pixels (a crop
    embedding, a thumbnail) must check this. Unplaceable boxes (bad frame
    size or shape) answer True -- fail closed.
    """
    no_embed = [z for z in zones if not z.embed]
    if not no_embed:
        return False
    if width <= 0 or height <= 0 or len(box_xyxy) != 4:
        return True
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    rect = (
        min(max(min(x1, x2) / width, 0.0), 1.0), min(max(min(y1, y2) / height, 0.0), 1.0),
        min(max(max(x1, x2) / width, 0.0), 1.0), min(max(max(y1, y2) / height, 0.0), 1.0),
    )
    return any(_rect_intersects_polygon(rect, z.polygon) for z in no_embed)
