"""Walkway box cut-off calibration eval (2026-11-03).

Question it answers: at which per-label cut-off does the walkway detector see
the real cars, people and dogs in every weather, without letting phantom
boxes into the person pipeline?

Three steps, so the slow model pass runs once and the cut-off sweep is free:

1. ``score``  -- run GroundingDINO on the hourly walkway frame samples
   (/mnt/telemetry/vision/hourly_frames/<day>/walkway_<HH>Z.jpg) at a LOW
   floor (default 0.20) with the host's own code (app/gdino.detect_images,
   the shipped tile layout from app/tiling). Raw boxes of the whole-frame
   pass and of each tile pass are stored, before any cut-off is applied.
   Needs torch + the model: run inside the vision-host image (below).
2. ``pool``   -- replay the live pipeline at the floor (whole frame only, and
   whole frame + tiles), keep tracked labels, de-duplicate the two, number
   every box, and write annotated JPEGs plus a labels CSV template. A human
   (or agent) looks at each JPEG and fills ``verdict``: real | false | dup.
3. ``report`` -- for every cut-off, replay the live pipeline exactly as
   services/orion-vision-host/app/runner.py does it (per-label cut-off on
   each pass -> tile merge -> size floor -> NMS -> cap), match every kept box
   to its labelled pool box, and report per label and per condition:
   objects recalled, real boxes kept, false boxes kept, and the share of
   frames where a real vehicle was seen.

Pure stdlib + the host's pure-python modules for steps 2-3 (numpy/PIL for
drawing). Labelled data: evals/fixtures/walkway_cutoff_labels.csv (boxes) and
evals/fixtures/walkway_cutoff_frames.csv (condition + objects missed even at
the floor). Frames are NOT committed (they show neighbours); the scored
JSONL is, so ``report`` re-runs without a model.

Usage::

  # 1. score (read-only, CPU, inside the vision-host image; ~20 s/frame on 32 cores)
  docker run --rm --cpus 32 -v <worktree>:/w:ro \\
    -v /mnt/telemetry/vision/hourly_frames:/frames:ro -v <out>:/out \\
    -v /mnt/telemetry/models/hf:/mnt/telemetry/models/hf:ro \\
    -e TRANSFORMERS_CACHE=/mnt/telemetry/models/hf/transformers -e HF_HUB_OFFLINE=1 \\
    -e PYTHONPATH=/w/services/orion-vision-host -w /w/services/orion-vision-host \\
    --entrypoint python3 orion-vision-host-vision-host \\
    evals/walkway_cutoff_calibration_eval.py score /frames --out /out/scored.jsonl --threads 32

  # 2. pool (needs numpy, PyYAML, loguru, Pillow -- e.g. the repo .venv; numbered
  #    JPEGs + contact sheets for labelling, and a labels CSV template)
  PYTHONPATH=services/orion-vision-host .venv/bin/python \\
    services/orion-vision-host/evals/walkway_cutoff_calibration_eval.py pool \\
    <out>/scored.jsonl --frames /mnt/telemetry/vision/hourly_frames \\
    --draw <out>/pool --sheet <out>/sheets --out <out>/labels_template.csv

  # 3. report on the committed labelled set (no model, no frames; from repo root)
  PYTHONPATH=services/orion-vision-host .venv/bin/python \\
    services/orion-vision-host/evals/walkway_cutoff_calibration_eval.py report \\
    --families person vehicle

  # 4. blind-detector proxy: per day, daylight hourly frames with any box
  #    at the shipped cut-offs vs a flat 0.35 (whole frame; --tiled adds tiles)
  PYTHONPATH=services/orion-vision-host .venv/bin/python \\
    services/orion-vision-host/evals/walkway_cutoff_calibration_eval.py hours --cutoff 0.35

``configured`` rows use config/vision_profiles.yaml as checked out, so after a
config change the same command shows the new values against the old ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "fixtures"
SCORED = DATA / "walkway_cutoff_scored.jsonl"
LABELS = DATA / "walkway_cutoff_labels.csv"
FRAMES = DATA / "walkway_cutoff_frames.csv"
STREAM = "walkway"
# Lowest row is 0.25, not the 0.20 scoring floor: GroundingDINO names a box
# only from prompt tokens above text_threshold (0.25 live), so every stored box
# under 0.25 has an empty label and no cut-off below it can change anything.
CUTOFFS = (0.25, 0.28, 0.30, 0.32, 0.35)
# Detector phrase -> family. GroundingDINO can return merged phrases
# ("vehicle mail truck"); a mail truck is a vehicle on this street.
FAMILY = {
    "person": "person",
    "dog": "dog",
    "bicycle": "bicycle",
    "stroller": "stroller",
    "vehicle": "vehicle",
    "mail truck": "vehicle",
    "vehicle mail truck": "vehicle",
    "package": "package",
}
TRACKED = ("person", "vehicle", "dog", "bicycle", "stroller")


def family(label: Any) -> Optional[str]:
    return FAMILY.get(str(label or "").strip().lower())


def detect_params(profiles_path: Path = REPO / "config/vision_profiles.yaml") -> Dict[str, Any]:
    cfg = yaml.safe_load(profiles_path.read_text())
    for p in cfg["profiles"]:
        if p["name"] == "retina_detect_open_vocab":
            return p["params"]
    raise SystemExit("retina_detect_open_vocab profile missing")


def walkway_prompts() -> List[str]:
    cfg = yaml.safe_load((REPO / "config/vision_frame_router.yaml").read_text())
    return cfg["streams"][STREAM]["triggered"]["request"]["prompts"]


def walkway_frames(root: Path) -> List[Path]:
    return sorted(root.glob(f"*/{STREAM}_*Z.jpg"))


def frame_key(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


# ---------------------------------------------------------------- 1. score
def cmd_score(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image, ImageFile
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from app.gdino import detect_images
    from app.tiling import tile_layout, tiling_params, to_frame_coords

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    params = detect_params()
    prompts = walkway_prompts()
    text = " . ".join(prompts) + " ."
    text_th = float(params.get("text_threshold", 0.25))
    lay = tiling_params(params, {"detect_tiles": True})
    min_short = int(params.get("tile_min_short_side_px", 1200))
    if args.threads:
        torch.set_num_threads(args.threads)
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(args.device).eval()

    def run(images):
        return detect_images(model, processor, images, text, args.floor, text_th, args.device,
                             lambda i: {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in i.items()})

    done = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        done = {json.loads(l)["frame"] for l in args.out.read_text().splitlines() if l.strip()}
    frames = walkway_frames(args.frames)
    if args.limit:
        frames = frames[: args.limit]
    with args.out.open("a") as fh:
        for f in frames:
            key = frame_key(f)
            if key in done:
                continue
            try:
                img = Image.open(f).convert("RGB")
            except OSError as exc:
                print(f"{key} SKIPPED unreadable: {exc}", flush=True)
                continue
            t0 = time.perf_counter()
            full = run([img])[0]
            tiles_raw: List[Dict[str, Any]] = []
            if min(img.width, img.height) >= min_short:
                for t in tile_layout(img.width, img.height, lay["grid"], lay["overlap"], lay["region"]):
                    tiles_raw += [to_frame_coords(o, t, img.width, img.height) for o in run([img.crop(t.as_box())])[0]]
            rec = {
                "frame": key, "width": img.width, "height": img.height, "floor": args.floor,
                "text_threshold": text_th, "prompts": prompts, "full_raw": full, "tiles_raw": tiles_raw,
                "seconds": round(time.perf_counter() - t0, 2),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{key} full={len(full)} tiles={len(tiles_raw)} {rec['seconds']}s", flush=True)
    return 0


# ------------------------------------------------- live pipeline replay
def thresholds_for(params: Mapping[str, Any], cutoff: Any):
    """BoxThresholds for the walkway. ``cutoff`` is a float (every label at
    that value), a dict label->value (other labels keep the stream default),
    or None (the profile as configured)."""
    from app.box_thresholds import BoxThresholds, resolve_box_thresholds

    if cutoff is None:
        return resolve_box_thresholds(params, STREAM)
    if isinstance(cutoff, Mapping):
        base = resolve_box_thresholds(params, STREAM)
        merged = dict(base.by_label)
        merged.update({k.lower(): float(v) for k, v in cutoff.items() if k != "default"})
        return BoxThresholds(float(cutoff.get("default", base.default)), merged)
    return BoxThresholds(float(cutoff), {})


def measurable_floor(rec: Mapping[str, Any]) -> float:
    """Lowest cut-off the stored boxes can answer for: the scoring floor, or
    text_threshold when that is higher (boxes under it carry no label)."""
    return max(float(rec["floor"]), float(rec.get("text_threshold") or 0.0))


def replay(rec: Mapping[str, Any], params: Mapping[str, Any], th, tiled: bool,
           allow_below_text: bool = False) -> List[Dict[str, Any]]:
    """Same order as runner._run_detect_grounding_dino: per-label cut-off on
    each pass, tile merge, size floor, NMS, cap. Pinned against the real
    runner by tests/test_walkway_cutoff_calibration_eval.py
    (test_replay_matches_real_runner)."""
    from app.box_thresholds import apply_box_thresholds
    from app.detections import cap_by_score, drop_undersized, nms
    from app.tiling import merge_tiled_detections, tiling_params

    low = float(rec["floor"]) if allow_below_text else measurable_floor(rec)
    if th.floor < low - 1e-9:
        raise SystemExit(f"cut-off {th.floor} is below what the stored boxes can measure ({low}: "
                         f"scoring floor {rec['floor']}, text_threshold {rec.get('text_threshold')})")
    nms_iou = float(params.get("nms_iou", 0.6))
    max_det = int(params.get("max_detections", 30))
    objs = apply_box_thresholds([dict(o) for o in rec["full_raw"]], th)
    if tiled and rec.get("tiles_raw") is not None:
        lay = tiling_params(params, {"detect_tiles": True})
        tile_objs = apply_box_thresholds([dict(o) for o in rec["tiles_raw"]], th)
        objs = merge_tiled_detections(objs if lay["include_full_frame"] else [], tile_objs, nms_iou,
                                      lay["contain_frac"], lay["seam_merge_frac"])
    floors = params.get("min_box_long_side_frac_by_stream") or {}
    objs, _ = drop_undersized(objs, (floors.get(STREAM) or {}), rec["width"], rec["height"])
    return cap_by_score(nms(objs, nms_iou), max_det)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _contain(inner: Sequence[float], outer: Sequence[float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / a if a > 0 else 0.0


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return max(_iou(a, b), min(_contain(a, b), _contain(b, a)))


def match(box: Mapping[str, Any], pool: Iterable[Mapping[str, Any]], min_overlap: float = 0.5) -> Optional[Mapping[str, Any]]:
    """Best same-family pool box by IoU (or mutual containment)."""
    fam = family(box["label"])
    best, best_v = None, min_overlap
    for p in pool:
        if p["family"] != fam:
            continue
        v = _overlap(box["box_xyxy"], p["box_xyxy"])
        if v >= best_v:
            best, best_v = p, v
    return best


# ----------------------------------------------------------------- 2. pool
def build_pool(rec: Mapping[str, Any], params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    th = thresholds_for(params, float(rec["floor"]))
    pool: List[Dict[str, Any]] = []
    for tiled in (False, True):
        for o in replay(rec, params, th, tiled, allow_below_text=True):
            fam = family(o["label"])
            if fam not in TRACKED:
                continue
            o = {**o, "family": fam}
            hit = match(o, pool)
            if hit is None:
                pool.append(o)
            elif o["score"] > hit["score"]:
                # Known limit: the pool box takes the higher-scoring geometry,
                # so a whole-frame box that only partly overlaps it could later
                # read "unlabelled" in report (0 on the committed set).
                hit.update(o)
    pool.sort(key=lambda o: (o["box_xyxy"][1], o["box_xyxy"][0]))
    for i, o in enumerate(pool, 1):
        o["box_id"] = i
    return pool


def _detuple(o: Dict[str, Any]) -> Dict[str, Any]:
    """JSON turns app.tiling's tuples into lists; the merge hashes them."""
    if "_tile" in o:
        o["_tile"] = tuple(o["_tile"])
    if "_cut_edges" in o:
        o["_cut_edges"] = [tuple(e) for e in o["_cut_edges"]]
    return o


def load_scored(path: Path) -> List[Dict[str, Any]]:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in recs:
        r["tiles_raw"] = [_detuple(o) for o in r.get("tiles_raw") or []]
    return recs


def cmd_pool(args: argparse.Namespace) -> int:
    params = detect_params()
    rows = []
    crops: List[Tuple[str, Dict[str, Any]]] = []
    for rec in load_scored(args.scored):
        pool = build_pool(rec, params)
        for o in pool:
            x1, y1, x2, y2 = (round(v) for v in o["box_xyxy"])
            rows.append({"frame": rec["frame"], "box_id": o["box_id"], "label": o["label"], "family": o["family"],
                         "score": round(o["score"], 3), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "verdict": "", "object": "", "note": ""})
        if args.draw and args.frames:
            _draw(args.frames / rec["frame"], pool, args.draw / rec["frame"].replace("/", "__"))
        if args.sheet and args.frames:
            crops += [(rec["frame"], o) for o in pool]
    if args.sheet and crops:
        _sheets(args.frames, crops, args.sheet)
    out = args.out or (args.draw / "labels_template.csv" if args.draw else LABELS)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} pool boxes -> {out}")
    return 0


def _sheets(root: Path, crops: List[Tuple[str, Dict[str, Any]]], out: Path, per_sheet: int = 24) -> None:
    """Contact sheets of every pool box (padded crop, captioned frame/id/score)
    so verdicts can be given many boxes at a time."""
    from PIL import Image, ImageDraw, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    out.mkdir(parents=True, exist_ok=True)
    cell, cols = 240, 6
    cache: Dict[str, Any] = {}
    for n in range(0, len(crops), per_sheet):
        chunk = crops[n:n + per_sheet]
        rows_n = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows_n * (cell + 28)), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        for i, (frame, o) in enumerate(chunk):
            if frame not in cache:
                cache.clear()
                cache[frame] = Image.open(root / frame).convert("RGB")
            img = cache[frame]
            x1, y1, x2, y2 = o["box_xyxy"]
            side = max(x2 - x1, y2 - y1) * 1.6 + 40
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box = (max(0, cx - side / 2), max(0, cy - side / 2), min(img.width, cx + side / 2), min(img.height, cy + side / 2))
            c = img.crop(box)
            sc = cell / max(c.width, c.height)
            c = c.resize((max(1, int(c.width * sc)), max(1, int(c.height * sc))))
            cd = ImageDraw.Draw(c)
            cd.rectangle([(x1 - box[0]) * sc, (y1 - box[1]) * sc, (x2 - box[0]) * sc, (y2 - box[1]) * sc],
                         outline=(255, 40, 40), width=2)
            gx, gy = (i % cols) * cell, (i // cols) * (cell + 28)
            sheet.paste(c, (gx, gy))
            tag = f"{frame[5:10]} {frame.split('_')[-1][:3]} #{o['box_id']} {o['family'][:3]} {o['score']:.2f}"
            d.text((gx + 2, gy + cell + 4), tag, fill=(255, 255, 255))
        sheet.save(out / f"sheet_{n // per_sheet:03d}.jpg", quality=85)


def _draw(src: Path, pool: List[Dict[str, Any]], dst: Path) -> None:
    from PIL import Image, ImageDraw, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(img)
    colours = {"person": (255, 40, 40), "vehicle": (40, 160, 255), "dog": (255, 220, 0)}
    for o in pool:
        c = colours.get(o["family"], (40, 255, 40))
        d.rectangle(o["box_xyxy"], outline=c, width=5)
        x, y = o["box_xyxy"][0], max(0, o["box_xyxy"][1] - 44)
        d.rectangle([x, y, x + 150, y + 42], fill=(0, 0, 0))
        d.text((x + 4, y + 2), f"{o['box_id']} {o['family'][:3]} {o['score']:.2f}", fill=c, font_size=34)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.resize((img.width // 2, img.height // 2)).save(dst, quality=80)


# --------------------------------------------------------------- 3. report
def load_labels(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_frame: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if not r["verdict"]:
                continue
            by_frame[r["frame"]].append({
                **r, "box_id": int(r["box_id"]), "score": float(r["score"]),
                "box_xyxy": [float(r[k]) for k in ("x1", "y1", "x2", "y2")],
            })
    return by_frame


def load_frames(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            missed = defaultdict(int)
            for part in filter(None, (r.get("missed") or "").split(";")):
                fam, _, n = part.partition(":")
                missed[fam.strip()] += int(n or 1)
            out[r["frame"]] = {"condition": r["condition"], "missed": dict(missed)}
    return out


def evaluate(scored: List[Mapping[str, Any]], labels: Mapping[str, List[Mapping[str, Any]]],
             frames: Mapping[str, Mapping[str, Any]], params: Mapping[str, Any], cutoff: Any,
             tiled: bool) -> Dict[Tuple[str, str], Dict[str, int]]:
    """(condition|'all', family) -> counts. Objects = distinct (frame, object)
    among real pool boxes, plus objects missed even at the floor."""
    th = thresholds_for(params, cutoff)
    acc: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in scored:
        key = rec["frame"]
        if key not in frames:  # frames.csv lists every labelled frame, boxes or not
            continue
        cond = frames[key]["condition"]
        pool = labels.get(key, [])
        objects = defaultdict(set)
        for p in pool:
            if p["verdict"] == "real":
                objects[p["family"]].add(p["object"] or f"box{p['box_id']}")
        seen = defaultdict(set)
        kept = replay(rec, params, th, tiled)
        for o in kept:
            fam = family(o["label"])
            if fam not in TRACKED:
                continue
            hit = match(o, pool)
            if hit is None:
                kind = "unlabelled"
            elif hit["verdict"] == "false":
                kind = "false"
            else:
                obj = hit["object"] or (f"box{hit['box_id']}" if hit["verdict"] == "real" else "")
                # A second kept box on an object already seen this frame (e.g.
                # "vehicle" + "mail truck" on one truck: NMS is per label) is a dup.
                kind = "real_boxes" if hit["verdict"] == "real" and obj not in seen[fam] else "dup"
                if obj:
                    seen[fam].add(obj)
            for c in (cond, "all"):
                acc[(c, fam)][kind] += 1
        for fam in TRACKED:
            n_obj = len(objects[fam]) + frames[key]["missed"].get(fam, 0)
            for c in (cond, "all"):
                b = acc[(c, fam)]
                b["objects"] += n_obj
                b["recalled"] += len(seen[fam] & objects[fam])
                if n_obj:
                    b["frames_with"] += 1
                    b["frames_seen"] += 1 if seen[fam] & objects[fam] else 0
    return acc


def _fmt_cut(c: Any) -> str:
    if c is None:
        return "configured"
    if isinstance(c, Mapping):
        return ",".join(f"{k}={v}" for k, v in c.items())
    return f"{c:.2f}"


def cmd_report(args: argparse.Namespace) -> int:
    params = detect_params(args.profiles)
    scored = load_scored(args.scored)
    labels = load_labels(args.labels)
    frames = load_frames(args.frames_csv)
    labelled = [r for r in scored if r["frame"] in frames]
    conds = sorted({frames[r["frame"]]["condition"] for r in labelled})
    per_cond = ", ".join("%s=%d" % (c, sum(frames[r["frame"]]["condition"] == c for r in labelled)) for c in conds)
    print(f"labelled frames: {len(labelled)} ({per_cond})")
    n_boxes = defaultdict(lambda: defaultdict(int))
    for k in labels:
        if k in frames:
            for p in labels[k]:
                n_boxes[p["family"]][p["verdict"]] += 1
    print("labelled pool boxes: " + "; ".join(f"{f} {dict(v)}" for f, v in sorted(n_boxes.items())))
    unnamed = sum(1 for r in labelled for o in r["full_raw"] + r["tiles_raw"] if not str(o.get("label") or "").strip())
    print(f"raw boxes with an empty label (score under text_threshold, never countable): {unnamed}")
    cuts: List[Any] = list(args.cutoff or CUTOFFS) + [None]
    for extra in args.per_label or []:
        cuts.append(json.loads(extra))
    out: Dict[str, Any] = {}
    for tiled in (False, True):
        mode = "whole+tiles" if tiled else "whole"
        print(f"\n== {mode} pass (tiles run on ~1 frame per 20 s live) ==")
        print(f"{'cut-off':<28} {'cond':<9} {'label':<8} {'objects':>7} {'recall':>7} {'frames_seen':>11} {'real':>5} {'false':>5} {'dup':>4} {'unlab':>5}")
        for cut in cuts:
            acc = evaluate(labelled, labels, frames, params, cut, tiled)
            for cond in ["all", *conds]:
                for fam in args.families:
                    b = acc.get((cond, fam))
                    if not b or (not b["objects"] and not b["false"]):
                        continue
                    rec = b["recalled"] / b["objects"] if b["objects"] else float("nan")
                    fs = f"{b['frames_seen']}/{b['frames_with']}"
                    print(f"{_fmt_cut(cut):<28} {cond:<9} {fam:<8} {b['objects']:>7} {rec:>7.2f} {fs:>11} {b['real_boxes']:>5} {b['false']:>5} {b['dup']:>4} {b['unlabelled']:>5}")
                    out[f"{mode}|{_fmt_cut(cut)}|{cond}|{fam}"] = dict(b)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


def cmd_hours(args: argparse.Namespace) -> int:
    """Per hourly frame: does the whole-frame pass keep ANY box (the
    blind-detector alarm's "saw something", any label) at each cut-off?
    One frame per hour is a proxy for the alarm's 10-minute windows, not a
    replay of it: unlabelled, so a kept phantom counts as "saw something".
    "Daylight" is a fixed UTC hour range (default 14Z-00Z), not sunrise."""
    params = detect_params(args.profiles)
    cuts: List[Any] = [None, *(args.cutoff or [])] + [json.loads(x) for x in (args.per_label or [])]
    by_day: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    print("frame".ljust(28) + " ".join(_fmt_cut(c)[:22].rjust(22) for c in cuts))
    for rec in load_scored(args.scored):
        day, hour = rec["frame"].split("/")[0], int(rec["frame"].split("_")[-1][:2])
        row = []
        for c in cuts:
            kept = replay(rec, params, thresholds_for(params, c), tiled=args.tiled)
            row.append(len(kept))
            if args.day_start <= hour or hour < args.day_end:
                by_day[day][_fmt_cut(c)].append(1 if kept else 0)
        if args.verbose:
            print(rec["frame"].ljust(28) + " ".join(str(n).rjust(22) for n in row))
    mode = "whole+tiles" if args.tiled else "whole"
    print(f"\nshare of daylight hourly frames (UTC hour >= {args.day_start:02d} or < {args.day_end:02d}) with any box, {mode} pass:")
    for day in sorted(by_day):
        print(day.ljust(12) + " ".join(
            f"{_fmt_cut(c)}={sum(v)}/{len(v)}" for c in cuts for v in [by_day[day][_fmt_cut(c)]]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("frames", type=Path, help="hourly_frames root (<day>/walkway_<HH>Z.jpg)")
    s.add_argument("--out", type=Path, default=SCORED)
    s.add_argument("--floor", type=float, default=0.20)
    s.add_argument("--device", default="cpu")
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_score)
    p = sub.add_parser("pool")
    p.add_argument("scored", type=Path, nargs="?", default=SCORED)
    p.add_argument("--frames", type=Path)
    p.add_argument("--draw", type=Path)
    p.add_argument("--sheet", type=Path, help="write contact sheets of every pool box here")
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_pool)
    r = sub.add_parser("report")
    r.add_argument("--scored", type=Path, default=SCORED)
    r.add_argument("--labels", type=Path, default=LABELS)
    r.add_argument("--frames-csv", type=Path, default=FRAMES)
    r.add_argument("--profiles", type=Path, default=REPO / "config/vision_profiles.yaml")
    r.add_argument("--cutoff", type=float, action="append")
    r.add_argument("--per-label", action="append", help='JSON, e.g. \'{"vehicle": 0.28}\'')
    r.add_argument("--families", nargs="+", default=list(TRACKED))
    r.add_argument("--json", type=Path)
    r.set_defaults(fn=cmd_report)
    h = sub.add_parser("hours")
    h.add_argument("--scored", type=Path, default=SCORED)
    h.add_argument("--profiles", type=Path, default=REPO / "config/vision_profiles.yaml")
    h.add_argument("--cutoff", type=float, action="append")
    h.add_argument("--per-label", action="append")
    h.add_argument("--day-start", type=int, default=14, help="first daylight UTC hour")
    h.add_argument("--day-end", type=int, default=1, help="daylight ends before this UTC hour (wraps midnight)")
    h.add_argument("--tiled", action="store_true", help="whole frame + shipped tiles (live: ~1 frame per 20 s)")
    h.add_argument("-v", "--verbose", action="store_true")
    h.set_defaults(fn=cmd_hours)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
