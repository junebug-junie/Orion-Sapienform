from orion.cognition.workflows import get_workflow_definition, resolve_user_workflow_invocation, workflow_registry_payload


def test_alias_resolution_matches_registered_workflow() -> None:
    match = resolve_user_workflow_invocation('Hey Orion, run your dream cycle')

    assert match is not None
    assert match.workflow_id == 'dream_cycle'
    assert match.matched_alias == 'run your dream cycle'


def test_non_matching_prompt_falls_back_to_none() -> None:
    assert resolve_user_workflow_invocation('Can you help me debug this service?') is None


def test_actions_skill_name_is_not_treated_as_workflow_identity() -> None:
    assert resolve_user_workflow_invocation('run skills.system.notify_chat_message.v1') is None


def test_registry_payload_exposes_machine_readable_workflows() -> None:
    payload = workflow_registry_payload(user_invocable_only=True)
    workflow_ids = {item['workflow_id'] for item in payload}

    assert {'dream_cycle', 'journal_pass', 'journal_discussion_window_pass', 'self_review', 'concept_induction_pass'} <= workflow_ids
    assert get_workflow_definition('journal_pass').may_call_actions is False
    assert get_workflow_definition('journal_discussion_window_pass').may_call_actions is False


def test_journal_last_n_minutes_resolves_discussion_window_workflow() -> None:
    m = resolve_user_workflow_invocation('Journal the last 47 minutes')
    assert m is not None
    assert m.workflow_id == 'journal_discussion_window_pass'
