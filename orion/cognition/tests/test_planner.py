from pathlib import Path
from orion_cognition.planner import SemanticPlanner, SystemState

def test_introspect_plan():
    base = Path(__file__).resolve().parents[1]
    planner = SemanticPlanner(
        verbs_dir=base/"verbs",
        prompts_dir=base/"prompts"
    )

    state = SystemState(name="Idle")
    plan = planner.build_plan("introspect", system_state=state)

    assert plan.verb_name == "introspect"
    assert len(plan.steps) >= 1
