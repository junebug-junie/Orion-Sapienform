from app.executor import _should_prepare_brain_reply_context
from orion.schemas.cortex.schemas import ExecutionStep


def test_runtime_skill_skips_chat_stance_autonomy_context_prep():
    step = ExecutionStep(
        verb_name="skills.runtime.docker_prune_stopped_containers.v1",
        step_name="docker_prune",
        order=0,
        services=["SkillService"],
    )
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is False


def test_non_runtime_brain_step_keeps_context_prep_enabled():
    step = ExecutionStep(
        verb_name="chat_general",
        step_name="answer",
        order=0,
        services=["LLMGatewayService"],
    )
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is True
