from pathlib import Path
import sys
import types
import logging

ROOT = Path(__file__).resolve().parents[3]
APP_PATH = ROOT / "services" / "orion-recall"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(APP_PATH) not in sys.path:
    sys.path.append(str(APP_PATH))
if str(APP_PATH / "app") not in sys.path:
    sys.path.append(str(APP_PATH / "app"))

sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logging.getLogger("test")))

from profiles import get_profile, load_profiles  # type: ignore  # noqa: E402


def test_profiles_load_default():
    profiles = load_profiles()
    assert "reflect.v1" in profiles
    prof = get_profile("reflect.v1")
    assert prof["profile"] == "reflect.v1"
    assert prof["max_total_items"] > 0
