"""Age-band read (orion/vision/age_band.py). Synthetic vectors only: no real crop
vector is committed (they belong to passersby and children)."""
from __future__ import annotations

import json

import numpy as np

from orion.vision import age_band as ab


def test_committed_text_vectors_match_model_and_prompts() -> None:
    raw = json.loads(ab.TEXT_VECTORS_PATH.read_text())
    assert raw["model_id"] == ab.AGE_BAND_MODEL_ID and raw["prompt_hash"] == ab.prompt_hash()
    assert set(raw) == {"model_id", "prompt_hash", "child", "adult", "logit_scale"}   # no other attribute
    tv = ab.text_vectors()
    for band in ("child", "adult"):
        assert tv[band].shape == (1152,) and abs(np.linalg.norm(tv[band]) - 1.0) < 1e-4
    assert set(ab.AGE_BAND_PROMPTS) == {"child", "adult"}


def test_p_child_reads_the_text_side_and_only_the_pinned_model() -> None:
    tv = ab.text_vectors()
    assert ab.p_child(tv["child"], ab.AGE_BAND_MODEL_ID) > 0.9
    assert ab.p_child(tv["adult"], ab.AGE_BAND_MODEL_ID) < 0.1
    mid = tv["child"] + tv["adult"]
    assert abs(ab.p_child(mid, ab.AGE_BAND_MODEL_ID) - 0.5) < 1e-6
    assert ab.p_child(tv["child"], None) is None
    assert ab.p_child(tv["child"], "google/siglip2-so400m-patch14-384") is None
    assert ab.p_child(np.ones(8), ab.AGE_BAND_MODEL_ID) is None
    assert ab.p_child(np.zeros(1152), ab.AGE_BAND_MODEL_ID) is None
    assert ab.p_child(None, ab.AGE_BAND_MODEL_ID) is None


def test_cluster_age_band_needs_enough_confident_reads() -> None:
    assert ab.cluster_age_band([0.99]) is None                    # one read is not a cluster read
    assert ab.cluster_age_band([0.99, 0.01]) is None               # mean 0.5: unknown
    assert ab.cluster_age_band([0.9, 0.7]) == "child"              # mean 0.8
    assert ab.cluster_age_band([0.1, 0.2]) == "adult"
    assert ab.cluster_age_band([0.74, 0.74]) is None               # just under the bar
    assert ab.cluster_age_band([None, 0.9, None]) is None          # Nones are not reads
    assert ab.cluster_age_band([0.9, 0.9, 0.9], min_n=3) == "child"


def test_band_from_age_answer() -> None:
    cases = {
        "8 years old": "child", "Someone is 9 years old; 10 in January.": "child", "about 7 or 8?": "child",
        "Adult": "adult", "Adult, early 20s": "adult", "early 30s": "adult", "late 40's early 50's?": "adult",
        "13": "adult", "12": "child", "": None, None: None, "I don't know": None,
        "18 months": "child", "3 weeks old": "child", "born 2015": None, "2015": None,
        "not an adult": None, "isn't grown up yet": None, "a toddler": "child", "just a kid": "child",
    }
    for text, want in cases.items():
        assert ab.band_from_age_answer(text) == want, text
    # an old answer ages: "12" given two years ago is 14 now
    assert ab.band_from_age_answer("12", years_since=2.0) == "adult"
    assert ab.band_from_age_answer("8 years old", years_since=1.0) == "child"
