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


def test_chat_quick_skips_heavy_brain_reply_context_prep():
    step = ExecutionStep(
        verb_name="chat_quick",
        step_name="llm_chat_quick",
        order=1,
        services=["LLMGatewayService"],
    )
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is False


def test_chat_kids_story_skips_heavy_brain_reply_context_prep():
    step = ExecutionStep(
        verb_name="chat_kids_story",
        step_name="llm_chat_kids_story",
        order=1,
        services=["LLMGatewayService"],
    )
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is False


def test_introspect_spark_skips_heavy_brain_reply_context_prep():
    step = ExecutionStep(
        verb_name="introspect_spark",
        step_name="generate_introspection",
        order=0,
        services=["LLMGatewayService"],
    )
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is False


def test_memory_graph_suggest_skips_heavy_brain_reply_context_prep():
    step = ExecutionStep(
        verb_name="memory_graph_suggest",
        step_name="llm_memory_graph_suggest",
        order=0,
        services=["LLMGatewayService"],
    )
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is False
