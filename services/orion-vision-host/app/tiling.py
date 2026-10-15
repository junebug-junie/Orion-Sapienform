"""
Tiled full-resolution detection: tile layout, box mapping, cross-tile merge.

Why: GroundingDINO's processor resizes every image to 800 px on the short
side (1333 max long side). The walkway frame is 2560x1920, so a person on the
far sidewalk (~50 px tall at full resolution) reaches the model at ~21 px.
Running the detector on overlapping crops of the frame keeps those people
near native size. See docs/superpowers/pr-reports/
2026-10-15-walkway-tiled-detection-pr.md for the measurements.

Pure Python and dependency-free (like detections.py) so it is testable
without torch.

Merge rule, in order:
  1. Every tile box is shifted into full-frame pixels. A tile box is "cut"
     when it touches a tile edge that is not also a frame edge -- the object
     probably continues into the next tile.
  2. A cut box mostly inside a same-label whole box (from the full-frame
     pass or another tile) is a fragment of that object: drop it.
  3. Cut boxes of the same label from different tiles that overlap each
     other are two halves of one object split by a seam: replace them with
     their union (score = the higher of the two).
  4. Per-label NMS over everything, then the caller's max_detections cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .detections import nms


@dataclass(frozen=True)
class Tile:
    """A crop of the frame in full-frame pixel coordinates (x1/y1 exclusive)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_box(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def _axis_spans(start: int, end: int, n: int, overlap: float) -> List[Tuple[int, int]]:
    length = end - start
    if n <= 1 or length <= 0:
        return [(start, end)]
    overlap = min(max(float(overlap), 0.0), 0.9)
    size = length / (n - (n - 1) * overlap)
    stride = size * (1.0 - overlap)
    spans = []
    for i in range(n):
        a = start + int(round(i * stride))
        b = start + int(round(i * stride + size))
        if i == n - 1:
            b = end
        spans.append((a, min(b, end)))
    return spans


def tile_layout(
    width: int,
    height: int,
    grid: Sequence[int] = (2, 2),
    overlap: float = 0.2,
    region: Optional[Sequence[float]] = None,
) -> List[Tile]:
    """Tiles covering ``region`` (normalized x0,y0,x1,y1; default the whole
    frame) in a cols x rows grid. ``overlap`` is the fraction of a tile's
    width/height shared with its neighbour."""
    cols, rows = int(grid[0]), int(grid[1])
    if cols < 1 or rows < 1:
        raise ValueError(f"tile grid must be >= 1x1, got {cols}x{rows}")
    rx0, ry0, rx1, ry1 = region if region else (0.0, 0.0, 1.0, 1.0)
    px0 = max(0, int(round(float(rx0) * width)))
    py0 = max(0, int(round(float(ry0) * height)))
    px1 = min(width, int(round(float(rx1) * width)))
    py1 = min(height, int(round(float(ry1) * height)))
    if px1 <= px0 or py1 <= py0:
        raise ValueError(f"empty tile region {region!r} for {width}x{height}")
    tiles = []
    for ya, yb in _axis_spans(py0, py1, rows, overlap):
        for xa, xb in _axis_spans(px0, px1, cols, overlap):
            tiles.append(Tile(xa, ya, xb, yb))
    return tiles


def to_frame_coords(obj: Dict[str, Any], tile: Tile, frame_w: int, frame_h: int,
                    edge_margin_frac: float = 0.01) -> Dict[str, Any]:
    """Copy of a tile-space detection shifted into frame space, with
    ``_cut`` set when the box touches an interior tile edge."""
    bx0, by0, bx1, by1 = [float(v) for v in obj["box_xyxy"]]
    fx0, fy0, fx1, fy1 = bx0 + tile.x0, by0 + tile.y0, bx1 + tile.x0, by1 + tile.y0
    fx0, fy0 = max(0.0, fx0), max(0.0, fy0)
    fx1, fy1 = min(float(frame_w), fx1), min(float(frame_h), fy1)
    mx = max(2.0, edge_margin_frac * tile.width)
    my = max(2.0, edge_margin_frac * tile.height)
    cut = (
        (tile.x0 > 0 and fx0 - tile.x0 <= mx)
        or (tile.y0 > 0 and fy0 - tile.y0 <= my)
        or (tile.x1 < frame_w and tile.x1 - fx1 <= mx)
        or (tile.y1 < frame_h and tile.y1 - fy1 <= my)
    )
    out = dict(obj)
    out["box_xyxy"] = [fx0, fy0, fx1, fy1]
    out["_cut"] = bool(cut)
    out["_tile"] = tile.as_box()
    return out


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw * ih if iw > 0 and ih > 0 else 0.0


def _contained(inner: Sequence[float], outer: Sequence[float]) -> float:
    a = _area(inner)
    return _inter(inner, outer) / a if a > 0 else 0.0


def merge_tiled_detections(
    full_frame: List[Dict[str, Any]],
    tile_objects: List[Dict[str, Any]],
    nms_iou: float,
    contain_frac: float = 0.7,
    seam_merge_frac: float = 0.25,
) -> List[Dict[str, Any]]:
    """Merge the full-frame pass with tile detections already mapped by
    :func:`to_frame_coords`. Returns plain objects (internal keys stripped),
    NMS-ed, ordered by score. The caller applies max_detections."""
    whole = [dict(o, _cut=False) for o in full_frame]
    whole += [o for o in tile_objects if not o.get("_cut")]
    cut = [o for o in tile_objects if o.get("_cut")]

    # 2. Fragments inside a whole box of the same label.
    cut = [
        c for c in cut
        if not any(
            w.get("label") == c.get("label") and _contained(c["box_xyxy"], w["box_xyxy"]) >= contain_frac
            for w in whole
        )
    ]

    # 3. Union seam halves (different tiles, same label, overlapping).
    merged: List[Dict[str, Any]] = []
    used = [False] * len(cut)
    for i, a in enumerate(cut):
        if used[i]:
            continue
        group = dict(a)
        tiles = {a.get("_tile")}
        for j in range(i + 1, len(cut)):
            b = cut[j]
            if used[j] or b.get("label") != group.get("label") or b.get("_tile") in tiles:
                continue
            gb, bb = group["box_xyxy"], b["box_xyxy"]
            smaller = min(_area(gb), _area(bb))
            if smaller > 0 and _inter(gb, bb) / smaller >= seam_merge_frac:
                group["box_xyxy"] = [min(gb[0], bb[0]), min(gb[1], bb[1]), max(gb[2], bb[2]), max(gb[3], bb[3])]
                group["score"] = max(float(group.get("score", 0.0)), float(b.get("score", 0.0)))
                tiles.add(b.get("_tile"))
                used[j] = True
        merged.append(group)

    out = nms(whole + merged, nms_iou)
    return [{k: v for k, v in o.items() if not k.startswith("_")} for o in out]


def tiling_params(params: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve tile knobs: request ``detect_tiles`` (bool or dict of
    overrides) over the profile's ``tile_*`` params."""
    req = request.get("detect_tiles")
    over = req if isinstance(req, dict) else {}

    def pick(key: str, default: Any) -> Any:
        if key in over:
            return over[key]
        return params.get(f"tile_{key}", default)

    grid = pick("grid", [2, 2])
    region = pick("region", None)
    return {
        "grid": [int(grid[0]), int(grid[1])],
        "overlap": float(pick("overlap", 0.2)),
        "region": [float(v) for v in region] if region else None,
        "include_full_frame": bool(pick("include_full_frame", True)),
        "batch": bool(pick("batch", False)),
        "contain_frac": float(pick("contain_frac", 0.7)),
        "seam_merge_frac": float(pick("seam_merge_frac", 0.25)),
    }


def tiling_requested(request: Dict[str, Any]) -> bool:
    req = request.get("detect_tiles")
    return req is True or (isinstance(req, dict) and req.get("enabled", True) is not False)
