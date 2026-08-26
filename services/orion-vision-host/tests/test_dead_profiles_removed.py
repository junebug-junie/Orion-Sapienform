from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.when_guard import KNOWN_GUARD_FLAGS

# Deleted chore, 2026-08-25: confirmed zero references anywhere in real code
# (see docs/superpowers/specs/2026-08-21-seeing-juniper-identity-and-
# situated-observation-design.md section 3.1's blast-radius-checked table).
# identity_face is the deliberate survivor -- see its own comment in
# config/vision_profiles.yaml -- not covered here.
#
# Paired as (profile_name, task_type) tuples, not two independently-indexed
# lists -- review finding, 2026-08-25: two parallel same-length tuples with
# no assertion tying them together lets a future edit update one and forget
# the other, and the suite would keep passing while silently no longer
# covering the missed pairing.
DELETED_PROFILE_TASK_PAIRS = (
    ("pose_estimation", "pose"),
    ("depth_estimation", "depth_map"),
    ("action_recognition", "action_classify"),
    ("ocr_read", "ocr"),
    ("scene_graph", "scene_graph"),
    ("person_reid", "person_reid"),
    ("affect_signals", "affect_signals"),
)
DELETED_PROFILE_NAMES = tuple(name for name, _task_type in DELETED_PROFILE_TASK_PAIRS)
DELETED_TASK_TYPES = tuple(task_type for _name, task_type in DELETED_PROFILE_TASK_PAIRS)


def test_pair_list_names_and_task_types_are_actually_paired():
    """Not a real regression test on its own -- guards the fixture data
    above against exactly the drift the pairing was introduced to prevent:
    same length, no duplicates hiding a missed row."""
    assert len(DELETED_PROFILE_NAMES) == len(DELETED_TASK_TYPES) == len(DELETED_PROFILE_TASK_PAIRS)
    assert len(set(DELETED_PROFILE_NAMES)) == len(DELETED_PROFILE_NAMES)


def test_dead_profiles_no_longer_defined(vision_profiles):
    for name in DELETED_PROFILE_NAMES:
        assert name not in vision_profiles.profiles, f"{name} should have been deleted, still present"


def test_dead_task_types_no_longer_route_anywhere(vision_profiles):
    """resolve_target() itself never raises -- an unmapped task_type falls
    back to treating the task_type string as a name (its own documented
    behavior). The real failure point is one layer up: runner.execute()
    wraps is_pipeline()/get_profile() in a try/except KeyError and returns
    error_code=unknown_task. Confirms the fallback name is neither a real
    pipeline nor a real profile, i.e. it WOULD hit that KeyError branch."""
    for profile_name, task_type in DELETED_PROFILE_TASK_PAIRS:
        target = vision_profiles.resolve_target(task_type)
        assert not vision_profiles.is_pipeline(target), (
            f"task_type={task_type} (was {profile_name}) still resolves to a live pipeline {target!r}"
        )
        assert target not in vision_profiles.profiles, (
            f"task_type={task_type} (was {profile_name}) still resolves to a live profile {target!r}"
        )


def test_identity_face_deliberately_survives(vision_profiles):
    """The one profile from the same original batch that is NOT dead weight
    -- design doc section 4. Implemented 2026-08-26 (Juniper's direct
    go-ahead) -- see test_run_identity_face.py/test_identity_gallery.py for
    the real, working behavior. Still `enabled: false` by deliberate
    choice, not because it's unbuilt: two open findings (unvalidated live
    thresholds; a reply-channel wildcard fan-out, both documented on the
    profile's own comment in config/vision_profiles.yaml) gate live
    dispatch, independent of whether the code itself works."""
    assert "identity_face" in vision_profiles.profiles
    identity = vision_profiles.get_profile("identity_face")
    assert identity.kind == "identity"
    assert identity.enabled is False
    assert vision_profiles.resolve_target("identity_face") == "identity_face"


def test_pipeline_retina_dense_no_longer_references_pose_estimation(vision_profiles):
    dense = vision_profiles.get_pipeline("pipeline_retina_dense")
    used_profiles = {step.use for step in dense.steps}
    assert "pose_estimation" not in used_profiles


def test_want_pose_removed_from_known_guard_flags():
    """The only guard that ever named want_pose (pipeline_retina_dense's
    pose_estimation step) is gone -- leaving it in the allowlist would be a
    phantom flag nothing guards on."""
    assert "want_pose" not in KNOWN_GUARD_FLAGS


def test_adaptive_degrade_no_longer_names_deleted_profiles(vision_profiles):
    degrade_steps = vision_profiles.runtime["adaptive_degrade"]["steps"]
    drop_optional = next(s for s in degrade_steps if s["name"] == "drop_optional_profiles")
    refuse_ultra = next(s for s in degrade_steps if s["name"] == "refuse_ultra_tasks")

    for name in DELETED_PROFILE_NAMES:
        assert name not in drop_optional["disable_profiles"]
    for task_type in DELETED_TASK_TYPES:
        assert task_type not in refuse_ultra["refuse_task_types"]

    # Survivors still there -- this isn't a blanket wipe of the block.
    assert "identity_face" in drop_optional["disable_profiles"]
    assert "identity_face" in refuse_ultra["refuse_task_types"]
