"""Walkway tiled-detection eval: run GroundingDINO on a directory of real
walkway frames once as the live host does today (whole frame, one pass) and
once per tiling layout, and report per-label counts, detections only the
tiled pass found, detections it lost, and latency.

Uses the host's own pieces: app/gdino.py (the forward pass), app/tiling.py
(layout + merge), app/detections.py (NMS + cap), the retina_detect_open_vocab
params from config/vision_profiles.yaml, and the walkway prompts from
config/vision_frame_router.yaml -- so it tests the shipped config.

Frames are NOT committed (they show neighbours); pass a directory of copies.
CPU by default so it never competes with the live P4/T10 lanes.

Usage (read-only, inside the vision-host image):
  docker run --rm -v <repo>:/w:ro -v <frames>:/frames:ro -v <out>:/out \\
    -v /mnt/telemetry/models:/mnt/telemetry/models:ro \\
    -e HF_HOME=/mnt/telemetry/models/vision/hf -e HF_HUB_OFFLINE=1 \\
    -e PYTHONPATH=/w:/w/services/orion-vision-host -w /w/services/orion-vision-host \\
    --entrypoint python3 orion-vision-host-vision-host \\
    evals/walkway_tiled_detection_eval.py /frames --out /out \\
    --layout shipped --layout 2x2:2,2,0.2 --layout band:2,1,0.2,0,0.15,1,0.6

Layout spec: name:cols,rows,overlap[,x0,y0,x1,y1] (region normalized), or
`shipped` for the profile's tile_* params.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[3]
TRACKED = ("person", "dog", "bicycle", "stroller")


def _detect_profile() -> Dict[str, Any]:
    cfg = yaml.safe_load((REPO / "config/vision_profiles.yaml").read_text())
    for p in cfg["profiles"]:
        if p["name"] == "retina_detect_open_vocab":
            return p["params"]
    raise SystemExit("retina_detect_open_vocab profile missing")


def _walkway_request() -> Dict[str, Any]:
    cfg = yaml.safe_load((REPO / "config/vision_frame_router.yaml").read_text())
    return cfg["streams"]["walkway"]["triggered"]["request"]


def _parse_layout(spec: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from app.tiling import tiling_params

    if spec == "shipped":
        lay = tiling_params(params, {"detect_tiles": True})
        lay["name"] = "shipped"
        return lay
    name, _, nums = spec.partition(":")
    v = [float(x) for x in nums.split(",")]
    lay = tiling_params(params, {"detect_tiles": {
        "grid": [int(v[0]), int(v[1])],
        "overlap": v[2],
        "region": v[3:7] if len(v) >= 7 else None,
    }})
    lay["name"] = name
    return lay


def _match(a: Dict[str, Any], pool: List[Dict[str, Any]]) -> bool:
    from app.tiling import _contained

    for b in pool:
        if b["label"] != a["label"]:
            continue
        if max(_contained(a["box_xyxy"], b["box_xyxy"]), _contained(b["box_xyxy"], a["box_xyxy"])) >= 0.5:
            return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", action="append", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", action="store_true", help="also time tiles as one batched forward pass")
    ap.add_argument("--draw", action="store_true", help="write annotated JPEGs of frames with differences")
    args = ap.parse_args(argv)

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from app.detections import cap_by_score, nms
    from app.gdino import detect_images
    from app.tiling import merge_tiled_detections, tile_layout, tiles_bounds, to_frame_coords

    params = _detect_profile()
    req = _walkway_request()
    prompts = req["prompts"]
    text = " . ".join(prompts) + " ."
    box_th = float(params.get("box_threshold", params.get("score_threshold", 0.25)))
    text_th = float(params.get("text_threshold", 0.25))
    nms_iou = float(params.get("nms_iou", 0.6))
    max_det = int(params.get("max_detections", 30))
    layouts = [_parse_layout(s, params) for s in (args.layout or ["shipped"])]

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(args.device).eval()

    def cast(inputs):
        return {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def run(images):
        t = time.perf_counter()
        out = detect_images(model, processor, images, text, box_th, text_th, args.device, cast)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return out, time.perf_counter() - t

    frames = sorted(args.frames.glob("*.jpg"))
    if args.limit:
        frames = frames[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    # Warm-up (first forward pays allocator/kernel setup).
    run([Image.open(frames[0]).convert("RGB")])

    per_frame = []
    totals = {"single": Counter(), **{l["name"]: Counter() for l in layouts}}
    extra = {l["name"]: Counter() for l in layouts}
    lost = {l["name"]: Counter() for l in layouts}
    lat = {"single": [], **{l["name"]: [] for l in layouts}}
    lat_batch = {l["name"]: [] for l in layouts}
    for f in frames:
        img = Image.open(f).convert("RGB")
        raw, dt = run([img])
        single = cap_by_score(nms(raw[0], nms_iou), max_det)
        lat["single"].append(dt)
        totals["single"].update(o["label"] for o in single)
        rec: Dict[str, Any] = {"frame": f.name, "single": single, "single_s": round(dt, 3)}
        for lay in layouts:
            tiles = tile_layout(img.width, img.height, lay["grid"], lay["overlap"], lay["region"])
            crops = [img.crop(t.as_box()) for t in tiles]
            t_total = dt if lay["include_full_frame"] else 0.0
            tile_objs = []
            for tile, crop in zip(tiles, crops):
                r, tdt = run([crop])
                t_total += tdt
                tile_objs += [to_frame_coords(o, tile, img.width, img.height, bounds=tiles_bounds(tiles)) for o in r[0]]
            if args.batch:
                _, bdt = run(crops)
                lat_batch[lay["name"]].append(bdt + (dt if lay["include_full_frame"] else 0.0))
            merged = merge_tiled_detections(
                raw[0] if lay["include_full_frame"] else [], tile_objs, nms_iou,
                lay["contain_frac"], lay["seam_merge_frac"],
            )
            merged = cap_by_score(merged, max_det)
            lat[lay["name"]].append(t_total)
            totals[lay["name"]].update(o["label"] for o in merged)
            new = [o for o in merged if not _match(o, single)]
            gone = [o for o in single if not _match(o, merged)]
            extra[lay["name"]].update(o["label"] for o in new)
            lost[lay["name"]].update(o["label"] for o in gone)
            rec[lay["name"]] = {"objects": merged, "extra": new, "lost": gone, "seconds": round(t_total, 3)}
            if args.draw and any(o["label"] in TRACKED for o in new + gone):
                canvas = img.copy()
                d = ImageDraw.Draw(canvas)
                for o in single:
                    d.rectangle(o["box_xyxy"], outline=(0, 120, 255), width=3)
                for o in new:
                    d.rectangle(o["box_xyxy"], outline=(255, 40, 40), width=4)
                    d.text((o["box_xyxy"][0], o["box_xyxy"][1] - 12), f"{o['label']} {o['score']:.2f}", fill=(255, 40, 40))
                for o in gone:
                    d.rectangle(o["box_xyxy"], outline=(255, 220, 0), width=4)
                for t in tiles:
                    d.rectangle(t.as_box(), outline=(0, 255, 0), width=1)
                canvas.save(args.out / f"{lay['name']}__{f.stem}.jpg", quality=85)
        per_frame.append(rec)
        print(f"{f.name} single={len(single)} " + " ".join(
            f"{l['name']}={len(rec[l['name']]['objects'])}(+{len(rec[l['name']]['extra'])}/-{len(rec[l['name']]['lost'])})"
            for l in layouts), flush=True)

    def stat(xs):
        return {"mean_s": round(statistics.mean(xs), 3), "p50_s": round(statistics.median(xs), 3),
                "max_s": round(max(xs), 3)} if xs else None

    summary = {
        "frames": len(frames),
        "device": args.device,
        "prompts": prompts,
        "box_threshold": box_th,
        "layouts": layouts,
        "counts": {k: dict(v) for k, v in totals.items()},
        "extra_only_in_tiled": {k: dict(v) for k, v in extra.items()},
        "lost_vs_single": {k: dict(v) for k, v in lost.items()},
        "latency": {k: stat(v) for k, v in lat.items()},
        "latency_batched_tiles": {k: stat(v) for k, v in lat_batch.items() if v},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "per_frame.json").write_text(json.dumps(per_frame, indent=1))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
