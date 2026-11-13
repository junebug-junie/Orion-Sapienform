#!/usr/bin/env python3
"""Regenerate orion/vision/age_band_text_vectors.json (the child/adult text side
of the age-band veto). One-off: needs torch + transformers + the SigLIP weights,
so run it inside the orion-vision-host image, e.g.

    docker run --rm -v "$PWD:/repo" -v /mnt/telemetry/models/hf:/hf:ro \
      -e HF_HOME=/tmp/hf --entrypoint python3 orion-vision-host-vision-host \
      /repo/scripts/generate_age_band_text_vectors.py --cache-dir /hf/transformers

The sql-writer never loads a model: it reads the committed JSON.
Design: docs/superpowers/specs/2026-11-13-person-age-band-soft-veto-design.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orion.vision.age_band import AGE_BAND_MODEL_ID, AGE_BAND_PROMPTS, TEXT_VECTORS_PATH, prompt_hash  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=str(TEXT_VECTORS_PATH))
    args = ap.parse_args()
    import torch
    from transformers import AutoProcessor, SiglipModel

    model = SiglipModel.from_pretrained(AGE_BAND_MODEL_ID, cache_dir=args.cache_dir).eval()
    proc = AutoProcessor.from_pretrained(AGE_BAND_MODEL_ID, cache_dir=args.cache_dir)
    out = {"model_id": AGE_BAND_MODEL_ID, "prompt_hash": prompt_hash()}
    with torch.no_grad():
        for band, prompts in AGE_BAND_PROMPTS.items():
            f = model.get_text_features(**proc(text=list(prompts), padding="max_length", return_tensors="pt"))
            f = f / f.norm(dim=-1, keepdim=True)
            m = f.mean(dim=0)
            m = m / m.norm()
            out[band] = [round(float(x), 7) for x in m.tolist()]
        out["logit_scale"] = round(float(model.logit_scale.exp()), 6)
    Path(args.out).write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out} dim={len(out['child'])} logit_scale={out['logit_scale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
