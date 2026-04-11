from __future__ import annotations

from pathlib import Path

from app.actions_skill_registry import ActionsSkillRegistry
from app.capability_bridge import _SEMANTIC_VERB_TO_SKILL, resolve_capability_decision


def test_semantic_verb_pins_cover_prompt_skills() -> None:
    assert _SEMANTIC_VERB_TO_SKILL["answer_current_datetime"] == "skills.system.time_now.v1"
    assert _SEMANTIC_VERB_TO_SKILL["inspect_gpu_status"] == "skills.gpu.nvidia_smi_snapshot.v1"
    assert _SEMANTIC_VERB_TO_SKILL["show_biometrics_snapshot"] == "skills.biometrics.snapshot.v1"
    assert _SEMANTIC_VERB_TO_SKILL["list_biometrics_recent_readings"] == "skills.biometrics.raw_recent.v1"
    assert _SEMANTIC_VERB_TO_SKILL["inspect_docker_container_status"] == "skills.docker.ps_status.v1"
    assert _SEMANTIC_VERB_TO_SKILL["send_operator_notification"] == "skills.system.notify_chat_message.v1"
    assert _SEMANTIC_VERB_TO_SKILL["show_landing_pad_metrics"] == "skills.landing_pad.metrics_snapshot.v1"


def test_resolve_capability_decision_uses_pins() -> None:
    repo = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "verbs"
    reg = ActionsSkillRegistry(verbs_dir=repo)
    for verb, skill in _SEMANTIC_VERB_TO_SKILL.items():
        d = resolve_capability_decision(verb=verb, preferred_skill_families=["mesh_presence"], registry=reg)
        assert d.selected_skill == skill
        assert d.confidence == 1.0


def test_docker_ps_skill_family_isolated_from_mesh_bucket() -> None:
    repo = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "verbs"
    reg = ActionsSkillRegistry(verbs_dir=repo)
    ps = [x for x in reg.list() if x.skill_id == "skills.docker.ps_status.v1"][0]
    assert ps.family == "docker_inventory"


def test_biometrics_skills_split_families() -> None:
    repo = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "verbs"
    reg = ActionsSkillRegistry(verbs_dir=repo)
    snap = [x for x in reg.list() if x.skill_id == "skills.biometrics.snapshot.v1"][0]
    raw = [x for x in reg.list() if x.skill_id == "skills.biometrics.raw_recent.v1"][0]
    assert snap.family == "biometrics_snapshot"
    assert raw.family == "biometrics_recent"
