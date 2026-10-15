"""The corruption eval flags smear and decoder gray, and passes clean scenes.

All frames are synthesized here; no camera frames live in the repo.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from frame_smear import frame_report, gray_fraction, main, smear_fraction  # noqa: E402

# What FFmpeg's error concealment produces once OpenCV converts it to BGR:
# measured YCrCb (130, 128, 128) on a keyframe-less H.264 decode.
DECODER_GRAY_BGR = (130, 130, 130)


def _scene(h=320, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8)


def _color_scene(h=480, w=640, seed=0):
    """Textured colour scene with structure at more than one scale."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8)
    img = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    img += rng.integers(-25, 25, (h, w, 3), dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _jpeg(bgr, q=88):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# --- smear (existing behaviour, kept) ---------------------------------------


def test_clean_frame_reads_zero():
    assert smear_fraction(_scene()) == 0.0


def test_frame_smeared_from_30_percent_down():
    img = _scene()
    img[96:] = img[95]  # decoder copies the last good row down
    assert 0.65 <= smear_fraction(img) <= 0.72


def test_flat_sky_or_dark_band_is_not_smear():
    img = _scene()
    img[:128] = 200  # overexposed sky
    img[224:] = 3  # dark night band at the bottom
    assert smear_fraction(img) == 0.0


def test_high_res_streaks_with_wide_flat_columns_are_smear():
    """Regression, walkway 2026-10-15: 2560-wide streaks read 0.00.

    Wide flat columns keep the per-pixel change across each strip under the
    2.0 texture floor at full resolution, so the old code skipped every strip.
    """
    rng = np.random.default_rng(3)
    edges = np.cumsum(rng.integers(40, 160, 64))
    edges = edges[edges < 2560]
    row = np.zeros(2560, np.uint8)
    for a, b in zip(np.r_[0, edges], np.r_[edges, 2560]):
        row[a:b] = rng.integers(20, 235)
    img = np.empty((1920, 2560), np.uint8)
    img[:192] = rng.integers(0, 255, (192, 2560), dtype=np.uint8)  # real top 10%
    img[192:] = row  # everything below is the copied row
    strip = img[200:216].astype(np.float32)
    assert np.abs(np.diff(strip, axis=1)).mean() < 2.0  # the old code skipped this strip
    assert smear_fraction(img) > 0.5


# --- the five scenes the gate must judge ------------------------------------


def test_streak_smear_is_corrupt():
    img = _color_scene()
    img[144:] = img[143]
    r = frame_report(_jpeg(img))
    assert r["corrupt"] and r["smear_fraction"] > 0.5


def test_decoder_gray_blocks_are_corrupt_even_without_smear():
    img = _color_scene()
    gray = np.zeros(img.shape[:2], bool)
    gray[64:, :] = True  # ~85% of the frame lost to concealment
    gray[200:260, 320:400] = False  # a few speckled patches still decode
    img[gray] = DECODER_GRAY_BGR
    r = frame_report(_jpeg(img))
    assert r["gray_fraction"] > 0.7
    assert r["smear_fraction"] == 0.0  # smear alone would have passed this frame
    assert r["corrupt"]


def _clean(name):
    if name == "clean_textured_scene":
        return _color_scene(seed=7)
    if name == "clear_blue_sky":
        img = _color_scene(seed=7)
        # flat, saturated sky over the top 60%, brighter toward the horizon
        ramp = np.linspace(0, 40, 288, dtype=np.float32)[:, None]
        img[:288, :, 0] = np.clip(230 + ramp * 0.5, 0, 255).astype(np.uint8)  # B
        img[:288, :, 1] = np.clip(150 + ramp, 0, 255).astype(np.uint8)  # G
        img[:288, :, 2] = np.clip(70 + ramp, 0, 255).astype(np.uint8)  # R
        return img
    # night: dark, noisy, one lit point
    img = np.random.default_rng(7).integers(0, 12, (480, 640, 3), dtype=np.uint8)
    img[100:110, 300:310] = 220
    return img


@pytest.mark.parametrize("name", ["clean_textured_scene", "clear_blue_sky", "night"])
def test_clean_scenes_pass(name):
    r = frame_report(_jpeg(_clean(name)))
    assert r["smear_fraction"] == 0.0, r
    assert r["gray_fraction"] < 0.05, r
    assert not r["corrupt"]


def test_neutral_but_not_mid_gray_walls_are_not_decoder_gray():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:240] = 200  # flat white wall
    img[240:] = 60  # flat dark-gray wall
    assert gray_fraction(_jpeg(img)) == 0.0


# --- CLI gate -----------------------------------------------------------------


def _write(dirpath, name, bgr):
    cv2.imwrite(str(dirpath / name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def _three_clean_one_gray(dirpath):
    for i in range(3):
        _write(dirpath, f"ok_{i}.jpg", _color_scene(seed=i))
    bad = _color_scene(seed=9)
    bad[64:] = DECODER_GRAY_BGR
    _write(dirpath, "z_bad.jpg", bad)


def test_cli_exits_nonzero_on_any_corrupt_frame(tmp_path, capsys):
    _three_clean_one_gray(tmp_path)
    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "gray_fraction=" in out and "corrupt=1" in out and "grayed=1" in out


def test_cli_max_corrupt_fraction_tolerates_a_share(tmp_path):
    _three_clean_one_gray(tmp_path)
    assert main([str(tmp_path), "--max-corrupt-fraction", "0.25"]) == 0
    assert main([str(tmp_path), "--max-corrupt-fraction", "0.2"]) == 1


def test_cli_clean_frames_exit_zero_and_empty_dir_exits_two(tmp_path):
    assert main([str(tmp_path)]) == 2
    for i in range(3):
        _write(tmp_path, f"ok_{i}.jpg", _color_scene(seed=i))
    assert main([str(tmp_path), "--quiet"]) == 0
