"""The smear eval separates a packet-loss-smeared frame from a clean one."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from frame_smear import smear_fraction  # noqa: E402


def _scene(h=320, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8)


def test_clean_frame_reads_zero():
    assert smear_fraction(_scene()) == 0.0


def test_frame_smeared_from_30_percent_down():
    img = _scene()
    img[96:] = img[95]  # decoder copies the last good row down
    assert 0.65 <= smear_fraction(img) <= 0.72


def test_flat_sky_or_dark_band_is_not_smear():
    img = _scene()
    img[:128] = 200   # overexposed sky
    img[224:] = 3     # dark night band at the bottom
    assert smear_fraction(img) == 0.0
