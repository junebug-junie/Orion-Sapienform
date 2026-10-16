"""Walkway caption eval: replay real walkway frames through the host's own
caption path (VisionRunner._run_caption_frame) and print what each captioner
says, and whether it would become a caption at all.

Two modes, one model per process (settings are read at import):

  blip  -- today's walkway path: athena's BLIP-base, no prompt.
  qwen  -- the new path: Qwen2-VL-2B-Instruct with the walkway
           caption_dispatch prompt + null answers read from
           config/vision_frame_router.yaml (so the eval tests the shipped
           prompt, not a copy of it).

Frames are NOT committed (they can show neighbours); pass a directory of
copies. Runs on CPU by default so it never competes with the live T10/P4
lanes; --device cuda:0 works inside a GPU container.

Usage (read-only, inside the vision-host image):
  docker run --rm -v <repo>:/w:ro -v <frames>:/frames:ro \\
    -v /mnt/storage-warm/models/vision-circe-qwen:/mnt/storage-warm/models/vision-circe-qwen:ro \\
    -e HF_HOME=/mnt/storage-warm/models/vision-circe-qwen/hf -e HF_HUB_OFFLINE=1 \\
    -e VISION_VLM_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct -w /w/services/orion-vision-host \\
    -e PYTHONPATH=/w:/w/services/orion-vision-host <image> \\
    python3 evals/walkway_caption_eval.py --mode qwen --frames /frames
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[3]


def walkway_caption_cfg() -> dict:
    policy = yaml.safe_load((REPO / "config" / "vision_frame_router.yaml").read_text())
    return dict(policy["streams"]["walkway"]["caption_dispatch"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blip", "qwen"], required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", default=None, help="qwen only: try this prompt instead of the configured one")
    ap.add_argument("--crop-box", default=None,
                    help="qwen only: x1,y1,x2,y2 detector box to crop to (as caption_dispatch does), "
                         "applied to every frame. 2026-10-16: frames WITH a person/car in them are the "
                         "case that matters -- the 10-15 prompt was chosen on empty frames only and "
                         "then said 'nothing notable' to every real one.")
    args = ap.parse_args(argv)

    from app import runner as runner_mod
    from app.runner import VisionRunner

    settings = runner_mod.settings
    if args.device == "cpu":
        settings.VISION_DTYPE = "fp32"
    request_extra: dict = {}
    if args.mode == "qwen":
        cfg = walkway_caption_cfg()
        request_extra = {"caption_prompt": args.prompt or cfg["prompt"]}
        if cfg.get("null_answers"):
            request_extra["caption_null_answers"] = cfg["null_answers"]
        if args.crop_box:
            request_extra["caption_crop_boxes"] = [[float(v) for v in args.crop_box.split(",")]]
            request_extra["caption_crop_pad"] = float(cfg.get("crop_pad", 0.5))
    model_id = settings.VISION_VLM_MODEL_ID
    if (args.mode == "qwen") != ("qwen" in model_id.lower()):
        print(f"VISION_VLM_MODEL_ID={model_id!r} does not match --mode {args.mode}", file=sys.stderr)
        return 2

    r = VisionRunner.__new__(VisionRunner)  # skip artifacts dir creation; read-only run
    r.models = runner_mod.ModelManager()
    profile = SimpleNamespace(name="vlm_caption", model_id="", dtype="", params={})

    frames = sorted(Path(args.frames).glob("*.jpg"))
    if len(frames) > args.limit:
        step = len(frames) / args.limit
        frames = [frames[int(i * step)] for i in range(args.limit)]
    for f in frames:
        warnings: list = []
        t0 = time.time()
        out = r._run_caption_frame(profile, {"image_path": str(f), **request_extra}, args.device, warnings)
        cap = out["caption"]
        print(json.dumps({
            "mode": args.mode,
            "frame": f.name,
            "caption": cap.get("text"),
            "confidence": cap.get("confidence"),
            "rejected_reason": cap.get("rejected_reason"),
            "crop_box": out.get("crop_box"),
            "becomes_caption": bool(cap.get("text")),
            "warnings": warnings,
            "seconds": round(time.time() - t0, 2),
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
