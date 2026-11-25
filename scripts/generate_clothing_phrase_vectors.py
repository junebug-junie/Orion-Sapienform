#!/usr/bin/env python3
"""Regenerate services/orion-sql-writer/evals/clothing_phrase_vectors.json (text side of
run_clothing_clue_eval.py). One-off: needs torch + transformers + the SigLIP weights, so run
it inside the orion-vision-host image, as scripts/generate_age_band_text_vectors.py does:

    docker run --rm -v "$PWD:/repo" -v /mnt/telemetry/models/hf:/hf:ro \\
      -e HF_HOME=/tmp/hf --entrypoint python3 orion-vision-host-vision-host \\
      /repo/scripts/generate_clothing_phrase_vectors.py --cache-dir /hf/transformers

Each phrase key (and "neutral") becomes the normalised mean of its prompts' text features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EVALS = Path(__file__).resolve().parents[1] / "services" / "orion-sql-writer" / "evals"
PHRASES = EVALS / "clothing_phrases.json"
OUT = EVALS / "clothing_phrase_vectors.json"


def phrase_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps({"neutral": spec["neutral"], "phrases": spec["phrases"]},
                                     sort_keys=True).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    import torch
    from transformers import AutoProcessor, SiglipModel

    spec = json.loads(PHRASES.read_text())
    model = SiglipModel.from_pretrained(spec["model_id"], cache_dir=args.cache_dir).eval()
    proc = AutoProcessor.from_pretrained(spec["model_id"], cache_dir=args.cache_dir)

    def mean_vec(prompts):
        with torch.no_grad():
            f = model.get_text_features(**proc(text=list(prompts), padding="max_length", return_tensors="pt"))
        f = f / f.norm(dim=-1, keepdim=True)
        m = f.mean(dim=0)
        return [round(float(x), 7) for x in (m / m.norm()).tolist()]

    out = {"model_id": spec["model_id"], "phrase_hash": phrase_hash(spec), "neutral": mean_vec(spec["neutral"]),
           "phrases": {k: mean_vec(v) for k, v in spec["phrases"].items()}}
    OUT.write_text(json.dumps(out) + "\n")
    print(f"wrote {OUT} phrases={len(out['phrases'])} dim={len(out['neutral'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
