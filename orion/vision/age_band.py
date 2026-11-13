"""Child / adult band of a person crop, read from its stored SigLIP vector.

Design: docs/superpowers/specs/2026-11-13-person-age-band-soft-veto-design.md.

The camera already stores an L2-normalised SigLIP image vector for every person
crop (``vision_crop_observation.embedding``). Comparing it with the text
"a photo of a child" vs "a photo of an adult" (three prompts each, averaged)
gave the right band on 83% of the crops Juniper has named (offline eval,
2026-11-13). The text side is committed in ``age_band_text_vectors.json``
(``scripts/generate_age_band_text_vectors.py``), so this module is pure numpy
and loads no model.

It is used for one thing: a VETO in the person model. A guess "this cluster is
X" is dropped when the cluster confidently reads as one band and Juniper said X
is in the other. Nothing here names anyone, and nothing here is stored. There
is deliberately no gender read: on body crops it misread women and girls 42% of
the time (same eval), which is misgendering, not a signal.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

# The stored vectors this read is valid for. The vision host tries SigLIP2 first
# and falls back to this model; a vector from any other model is in a different
# space, so it gets no read (None), and NULL-id rows (2026-10-15/16) are skipped.
AGE_BAND_MODEL_ID = "google/siglip-so400m-patch14-384"
AGE_BAND_RULE = "age_band_v1"
AGE_BAND_PROMPTS: Dict[str, Sequence[str]] = {
    "child": ("a photo of a child", "a photo of a kid", "a photo of a young child walking"),
    "adult": ("a photo of an adult", "a photo of a grown-up person", "a photo of an adult walking"),
}
TEXT_VECTORS_PATH = Path(__file__).with_name("age_band_text_vectors.json")
# Child = under this many years, from Juniper's stated age.
CHILD_UNDER_YEARS = 13
# A cluster read needs at least this many crop reads, and a mean this far to one side.
DEFAULT_CONF = 0.75
DEFAULT_MIN_N = 2


def prompt_hash() -> str:
    return hashlib.sha256(json.dumps(AGE_BAND_PROMPTS, sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def text_vectors() -> Dict[str, object]:
    """The committed text side: unit child/adult vectors and the logit scale."""
    raw = json.loads(TEXT_VECTORS_PATH.read_text())
    if raw.get("model_id") != AGE_BAND_MODEL_ID or raw.get("prompt_hash") != prompt_hash():
        raise ValueError(f"{TEXT_VECTORS_PATH.name} is stale: regenerate with scripts/generate_age_band_text_vectors.py")
    return {"child": np.asarray(raw["child"], dtype=np.float64), "adult": np.asarray(raw["adult"], dtype=np.float64),
            "logit_scale": float(raw["logit_scale"])}


def p_child(vec, model_id: Optional[str]) -> Optional[float]:
    """P(child) for one stored crop vector, or None when the vector is not from
    ``AGE_BAND_MODEL_ID`` (or has the wrong size). Two-way sigmoid of the
    scaled cosine difference; SigLIP's logit bias cancels in a two-way read."""
    if model_id != AGE_BAND_MODEL_ID or vec is None:
        return None
    tv = text_vectors()
    v = np.asarray(vec, dtype=np.float64)
    if v.shape != tv["child"].shape:
        return None
    n = float(np.linalg.norm(v))
    if not n:
        return None
    v = v / n
    z = tv["logit_scale"] * (float(v @ tv["child"]) - float(v @ tv["adult"]))
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def cluster_age_band(reads: Iterable[Optional[float]], conf: float = DEFAULT_CONF,
                     min_n: int = DEFAULT_MIN_N) -> Optional[str]:
    """"child" | "adult" from the mean of a cluster's crop reads, or None when
    there are fewer than ``min_n`` reads or the mean sits between 1-conf and
    conf. "Unknown" is the normal answer for a thin or mixed cluster."""
    xs = [float(r) for r in reads if r is not None]
    if len(xs) < max(1, min_n):
        return None
    m = sum(xs) / len(xs)
    if m >= conf:
        return "child"
    if m <= 1.0 - conf:
        return "adult"
    return None


_YEARS = re.compile(r"(\d{1,3})")


def band_from_age_answer(text: Optional[str]) -> Optional[str]:
    """Juniper's free-text age answer -> band, or None when it does not say.
    "8 years old" / "Esme is 7" -> child; "Adult, early 20s" / "late 40's" -> adult.
    The first number decides; the word "adult" alone means adult."""
    if not text:
        return None
    t = text.strip().lower()
    m = _YEARS.search(t)
    if m:
        return "child" if int(m.group(1)) < CHILD_UNDER_YEARS else "adult"
    if "adult" in t or "grown" in t:
        return "adult"
    return None
