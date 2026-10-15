"""Gate: whisper-tts is pinned to athena's P4 by compose device_ids.

2026-10-15 swap: speech (a few bursts a month) moved from the T10 to the
P4; vision-host took the T10. The P4 is Pascal (sm_61): checked in the built
image that torch 2.8.0+cu126 ships sm_60 kernels, XTTS loads fp32, and STT
passes fp16=False -- the last is pinned below.
"""

from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]


def _env_example() -> dict:
    out = {}
    for line in (SERVICE_DIR / ".env_example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_compose_pin_is_required_device_id():
    text = (SERVICE_DIR / "docker-compose.yml").read_text()
    assert 'device_ids: ["${WHISPER_TTS_GPU_DEVICE_ID:?' in text


def test_env_example_pins_the_p4():
    assert _env_example()["WHISPER_TTS_GPU_DEVICE_ID"] == "00000000:86:00.0"


def test_stt_stays_fp32_on_pascal():
    text = (SERVICE_DIR / "app" / "stt.py").read_text()
    assert "fp16=False" in text
