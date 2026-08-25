from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.profiles import VisionProfiles
from app.runner import VisionRunner

# Review finding, 2026-08-25: the same load-config-yaml-and-parse helper had
# already been independently copy-pasted into test_caption_profile_routing.py
# and test_run_vlm_vqa.py before a third copy landed in
# test_dead_profiles_removed.py -- three call sites that would each need an
# independent update if VisionProfiles.load()'s signature or the config path
# ever changed. One fixture instead of three copies of the loading logic.
#
# Function-scoped, not session-scoped, on purpose: ProfileDef is a plain
# (non-frozen) dataclass and test_run_vlm_vqa.py's
# test_warm_profiles_kind_allowlist_includes_vlm already relies on mutating
# its OWN loaded instance's `profile.warm_on_start` in place without
# affecting any other test -- confirmed live that this codebase already
# depends on that isolation. A session-scoped singleton would silently
# reintroduce test-order-dependent pollution the moment two tests touch the
# same profile object. The YAML itself is tiny (parses in low single-digit
# milliseconds), so re-parsing it per test costs nothing worth trading
# correctness for.
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"


@pytest.fixture
def vision_profiles() -> VisionProfiles:
    p = VisionProfiles(str(_CONFIG_PATH))
    p.load()
    return p


@pytest.fixture
def vlm_runner(vision_profiles: VisionProfiles) -> VisionRunner:
    """A VisionRunner with the two caption/VQA profiles enabled -- what
    test_run_vlm_vqa.py's own `_runner()` built ad hoc before this fixture
    replaced it. Not shared more broadly (yet): every other file's own
    `enabled_names` set differs by what it's testing, so a single
    one-size-fits-all runner fixture would just trade one duplication for a
    misleadingly-named one-size-doesn't-fit-all shim."""
    tmp = tempfile.mkdtemp()
    return VisionRunner(profiles=vision_profiles, enabled_names=["vlm_vqa", "vlm_caption"], cache_dir=tmp)
