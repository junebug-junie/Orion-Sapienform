#!/usr/bin/env python3
"""Human-run CLI: enroll the one identity_face gallery subject from real photos.

docs/superpowers/specs/2026-08-21-seeing-juniper-identity-and-situated-
observation-design.md section 4: "One enrolled subject. Gallery does not
grow." This script is the ONLY code path that writes a gallery entry --
runner.py's request-handling path (app/identity_gallery.py's
match_embedding) only ever reads one. Run this by hand, locally, with real
photos; nothing in the running service calls it.

Usage (from services/orion-vision-host/):
    python3 scripts/enroll_identity_face.py --subject juniper photo1.jpg photo2.jpg photo3.jpg

Multiple photos are averaged into one embedding (better generalization
across angle/lighting than a single shot) -- pass 3-5 clear, varied photos
for a real enrollment, not a debug run.

Re-running with the same --subject overwrites the existing gallery entry
(re-enrollment, not accumulation) -- there is deliberately no "add more
photos to the existing gallery" mode, keeping "gallery does not grow"
true of this script too, not just of the runtime request path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def enroll(subject: str, image_paths: list[str], gallery_dir: str) -> Path:
    import numpy as np
    from PIL import Image

    from app.identity_gallery import save_gallery_embedding
    from app.model_manager import ModelManager

    manager = ModelManager()
    model, mtcnn = manager.load_face_identity_models(profile_name="identity_face_enroll", device="cpu")

    embeddings = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        faces, probs = mtcnn(img, return_prob=True)
        if faces is None:
            print(f"  SKIP {image_path}: no face detected", file=sys.stderr)
            continue
        if faces.dim() == 3:
            faces = faces.unsqueeze(0)
        # Keep only the highest-confidence face per photo -- an enrollment
        # photo should have exactly one clear subject in frame; a second
        # detected face (someone in the background) must not silently pull
        # the averaged embedding toward a stranger.
        best_idx = 0
        if isinstance(probs, list) and len(probs) > 1:
            best_idx = max(range(len(probs)), key=lambda i: probs[i] or 0.0)
        import torch

        with torch.inference_mode():
            embedding = model(faces[best_idx : best_idx + 1])
        embeddings.append(embedding[0].detach().float().cpu().numpy())
        print(f"  OK {image_path}: face detected, prob={probs[best_idx]:.4f}")

    if not embeddings:
        raise SystemExit(f"No usable face found in any of {len(image_paths)} image(s) -- enrollment aborted.")

    mean_embedding = np.mean(np.stack(embeddings), axis=0)
    path = save_gallery_embedding(gallery_dir, subject, mean_embedding, sample_count=len(embeddings))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Paths to real photos of the subject's face")
    parser.add_argument("--subject", default="juniper", help="Gallery subject name (default: juniper)")
    parser.add_argument(
        "--gallery-dir",
        default=None,
        help="Override the gallery directory (default: this service's IDENTITY_GALLERY_DIR setting)",
    )
    args = parser.parse_args()

    gallery_dir = args.gallery_dir
    if gallery_dir is None:
        from app.settings import Settings

        gallery_dir = Settings().IDENTITY_GALLERY_DIR

    print(f"Enrolling subject={args.subject!r} from {len(args.images)} image(s) -> {gallery_dir}")
    path = enroll(args.subject, args.images, gallery_dir)
    print(f"Wrote gallery entry: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
