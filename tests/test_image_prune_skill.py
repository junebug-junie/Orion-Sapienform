"""skills.runtime.image_prune.v1 -- Orion's second mutating action.

Two invariants here are load-bearing safety properties, not style checks:

  1. `docker image prune` must NEVER be invoked with `-a`. Without `-a` it
     removes only dangling (untagged, unreferenced) images. With `-a` it also
     removes every tagged image not backing a running container, which on this
     host means deleting the images of every stopped service.

  2. The gate must count DANGLING images, not Docker's `system df`
     "Reclaimable" figure. Live at authoring time those differ enormously
     (3,833 dangling vs 96.56 GB reclaimable across all unused images), and
     gating on a quantity the action does not move is precisely the defect
     builder_prune shipped and had to fix.

The classifier tests exist because builder_prune was registered in NONE of the
three skill classifiers on arrival and spent its early life advertising itself
as read_only + idempotent + observational -- the one skill in the repo that
deletes host data. That was patched three separate times in one day.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = REPO_ROOT / "services" / "orion-cortex-exec" / "app" / "verb_adapters.py"
VERB_YAML = REPO_ROOT / "orion" / "cognition" / "verbs" / "skills.runtime.image_prune.v1.yaml"


@pytest.fixture(scope="module")
def adapter_source() -> str:
    return ADAPTERS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def image_prune_block(adapter_source: str) -> str:
    """The image-prune section: its helper AND the verb class.

    Starts at the helper, not at the `@verb` decorator -- the dangling-image
    query lives in `_image_prune_dangling_stats`, which is defined above the
    decorator, so a decorator-anchored slice silently excluded the very command
    the safety assertions are about.

    Scoped to this section rather than the whole file so an assertion cannot
    pass because an unrelated sibling verb elsewhere in this 3,000-line module
    happens to contain the same string.
    """
    start = adapter_source.index("async def _image_prune_dangling_stats")
    return adapter_source[start:]


# ---------------------------------------------------------------------------
# Safety invariant 1: never `-a`
# ---------------------------------------------------------------------------


def test_prune_command_is_dangling_only_never_all(image_prune_block: str) -> None:
    """`-a`/`--all` must not appear in this verb's docker invocation.

    Asserted against the source rather than by executing docker, because the
    failure this guards is catastrophic and irreversible on a live host: `-a`
    deletes the image of every stopped service.
    """
    assert '["docker", "image", "prune", "-f"]' in image_prune_block
    for forbidden in ('"-a"', "'-a'", '"--all"', "'--all'"):
        assert forbidden not in image_prune_block, f"{forbidden} must never appear in image_prune"


def test_runner_allows_only_docker(image_prune_block: str) -> None:
    """No skill argument may talk the runner into another binary."""
    assert 'allowed_commands={"docker"}' in image_prune_block


# ---------------------------------------------------------------------------
# Safety invariant 2: the gate measures what the action changes
# ---------------------------------------------------------------------------


def test_gate_counts_dangling_images_not_system_df_reclaimable(image_prune_block: str) -> None:
    """The gate must read the dangling COUNT, which is the set prune removes.

    `_builder_prune_cache_stats` (the `docker system df` reader) must not be
    used for the gate here: its Reclaimable figure covers all unused images,
    including tagged ones this action never touches.
    """
    assert "_image_prune_dangling_stats" in image_prune_block
    assert "min_dangling_count" in image_prune_block
    assert "_builder_prune_cache_stats" not in image_prune_block


def test_dangling_filter_is_exactly_the_prune_target(image_prune_block: str) -> None:
    assert '"--filter", "dangling=true"' in image_prune_block


def test_unreadable_count_declines_rather_than_assuming_zero(image_prune_block: str) -> None:
    """An unreadable daemon must not read as "nothing to do" OR as "act".

    Either default would be a fabricated gate evaluation.
    """
    assert "declined_unmeasurable" in image_prune_block


def test_zero_reclaimed_is_not_reported_as_success(image_prune_block: str) -> None:
    """The builder_prune lesson: a prune that reclaims nothing is not a win."""
    assert "pruned_nothing" in image_prune_block
    assert "if reclaimed <= 0:" in image_prune_block


def test_mutation_is_gated_on_execute_not_on_absence_of_a_preview_flag(
    image_prune_block: str,
) -> None:
    """Dry-run-by-default is only real if the side effect checks for `execute`.

    This repo has shipped a "dry-run by default" that still performed a real
    write because it tested for the absence of a flag instead.
    """
    assert 'if run_mode != "execute":' in image_prune_block
    assert '"would_act"' in image_prune_block


def test_outcome_is_measured_from_real_disk_readings(image_prune_block: str) -> None:
    """`bytes_reclaimed` must come from before/after, not from Docker's estimate."""
    assert 'reclaimed = max(0, int(before["used_bytes"]) - int(after["used_bytes"]))' in (
        image_prune_block
    )
    assert "bytes_reclaimed=reclaimed" in image_prune_block


# ---------------------------------------------------------------------------
# Registration -- the builder_prune misclassification, prevented structurally
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "services/orion-cortex-exec/app/actions_skill_registry.py",
        "orion/cognition/skills_manifest.py",
    ],
)
def test_image_prune_is_registered_as_host_mutating(module_path: str) -> None:
    """Present in the single shared marker list, in BOTH duplicated modules.

    Both files carry the same classifier logic. builder_prune was once fixed in
    one and not the other; naming both here makes that asymmetry a test
    failure.
    """
    source = (REPO_ROOT / module_path).read_text(encoding="utf-8")
    assert "HOST_MUTATING_SKILL_MARKERS" in source
    assert '"image_prune",' in source


@pytest.mark.parametrize(
    "module_path",
    [
        "services/orion-cortex-exec/app/actions_skill_registry.py",
        "orion/cognition/skills_manifest.py",
    ],
)
def test_all_three_classifiers_use_the_shared_list(module_path: str) -> None:
    """Family, risk, and execute-opt-in must all consult one source.

    Three independent substring checks is how builder_prune ended up correct in
    one classifier and wrong in the other two.
    """
    source = (REPO_ROOT / module_path).read_text(encoding="utf-8")
    assert source.count("_is_host_mutating_skill(") >= 4  # 1 def + 3 call sites


def test_no_classifier_still_hardcodes_builder_prune_alone() -> None:
    """Regression guard on the unification itself.

    If someone re-adds a bare `"builder_prune" in sid` branch, the next skill
    can silently half-register again.
    """
    for module_path in (
        "services/orion-cortex-exec/app/actions_skill_registry.py",
        "orion/cognition/skills_manifest.py",
    ):
        source = (REPO_ROOT / module_path).read_text(encoding="utf-8")
        assert '"builder_prune" in sid' not in source
        assert '"builder_prune" in skill_id.lower()' not in source


# ---------------------------------------------------------------------------
# Verb definition
# ---------------------------------------------------------------------------


def test_verb_yaml_exists_and_names_itself_consistently() -> None:
    spec = yaml.safe_load(VERB_YAML.read_text(encoding="utf-8"))
    assert spec["name"] == "skills.runtime.image_prune.v1"
    assert spec["plan"][0]["name"] == "skills.runtime.image_prune.v1"


def test_timeout_budgets_are_ordered_innermost_first() -> None:
    """subprocess (600s) < verb step (660s) < maintenance route RPC (720s).

    Hand-checked, not copied: if the verb step budget were below the subprocess
    budget, a real prune would be killed by the outer layer and the failure
    would be attributed to the wrong thing.
    """
    spec = yaml.safe_load(VERB_YAML.read_text(encoding="utf-8"))
    verb_step_sec = float(spec["timeout_ms"]) / 1000.0

    adapters = ADAPTERS.read_text(encoding="utf-8")
    subprocess_sec = None
    for line in adapters.splitlines():
        if line.startswith("IMAGE_PRUNE_TIMEOUT_SEC"):
            subprocess_sec = float(line.split("=", 1)[1].strip())
    assert subprocess_sec is not None

    policy = yaml.safe_load(
        (REPO_ROOT / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    route_sec = float(policy["proposal_kind_to_cortex"]["maintain"]["rpc_timeout_sec"])

    assert subprocess_sec < verb_step_sec < route_sec, (
        subprocess_sec,
        verb_step_sec,
        route_sec,
    )


def test_thresholds_are_declared_and_disk_bar_is_below_builder_prunes() -> None:
    """The lower disk bar is a reasoned choice and must stay explicit.

    Dangling images have no reuse value, so reclaiming them trades nothing
    away; build cache does, so it keeps the higher bar. If these ever invert,
    that reasoning has silently been discarded.
    """
    adapters = ADAPTERS.read_text(encoding="utf-8")
    values: dict[str, float] = {}
    for line in adapters.splitlines():
        for key in ("IMAGE_PRUNE_MIN_DISK_PCT", "BUILDER_PRUNE_MIN_DISK_PCT"):
            if line.startswith(key + " "):
                values[key] = float(line.split("=", 1)[1].strip())
    assert values["IMAGE_PRUNE_MIN_DISK_PCT"] < values["BUILDER_PRUNE_MIN_DISK_PCT"]


# ---------------------------------------------------------------------------
# Classifier invariants, executed rather than grepped.
#
# Verified once at authoring time by importing the pre-change module alongside
# the new one and diffing every classification: the ONLY skill whose family or
# risk changed was image_prune itself (system_inspection/read_only ->
# runtime_housekeeping/high_impact, i.e. off the dangerous default). These
# tests keep that true without needing the old module around.
# ---------------------------------------------------------------------------


def _manifest_module():
    import importlib.util
    import sys

    path = REPO_ROOT / "orion" / "cognition" / "skills_manifest.py"
    spec = importlib.util.spec_from_file_location("_sm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sm_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_every_host_mutating_marker_classifies_dangerously() -> None:
    """Being in the list must actually produce high_impact + non-idempotent.

    Otherwise the list is decoration and a skill can be "registered" while
    still advertising itself as safe.
    """
    module = _manifest_module()
    for marker in module.HOST_MUTATING_SKILL_MARKERS:
        skill_id = f"skills.runtime.{marker}.v1"
        risk_class, idempotent, observational = module._risk_for_skill(skill_id)
        assert risk_class == "high_impact", skill_id
        assert idempotent is False, skill_id
        assert observational is False, skill_id
        assert module._family_for_skill(skill_id) == "runtime_housekeeping", skill_id


def test_a_read_only_skill_is_not_swept_up_by_the_marker_list() -> None:
    """The inverse guard.

    Without it, `_is_host_mutating_skill` returning True unconditionally would
    pass every test above while making every read-only skill require
    confirmation.
    """
    module = _manifest_module()
    risk_class, idempotent, observational = module._risk_for_skill("skills.some.random.probe.v1")
    assert risk_class == "read_only"
    assert idempotent is True
    assert observational is True
    assert module._family_for_skill("skills.some.random.probe.v1") == "system_inspection"


def test_unrelated_skill_families_are_untouched() -> None:
    """Spot-check the branches that sit ABOVE the shared check in the chain.

    The marker check was inserted mid-function; an earlier branch must still
    win for skills it already matched.
    """
    module = _manifest_module()
    assert module._family_for_skill("skills.gpu.nvidia_smi.v1") == "gpu_presence"
    assert module._family_for_skill("skills.biometrics.snapshot.v1") == "biometrics_snapshot"
    assert module._risk_for_skill("skills.notify.operator.v1")[0] == "benign_actuation"
