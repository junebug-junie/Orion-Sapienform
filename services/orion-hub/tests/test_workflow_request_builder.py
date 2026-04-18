from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'cortex_request_builder.py'
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPEC = importlib.util.spec_from_file_location('hub_cortex_request_builder_workflows', MODULE_PATH)
assert SPEC and SPEC.loader
hub_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_builder)


def test_named_workflow_prompt_routes_into_explicit_workflow_request() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={'mode': 'auto'},
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='Run through your concept induction graphs',
    )

    assert req.mode == 'brain'
    assert req.route_intent == 'none'
    assert req.verb is None
    assert req.metadata['workflow_request']['workflow_id'] == 'concept_induction_pass'
    assert req.metadata['workflow_request']['execution_policy']['invocation_mode'] == 'immediate'
    assert debug['workflow_id'] == 'concept_induction_pass'


def test_explicit_workflow_prompts_resolve_for_all_supported_named_workflows() -> None:
    cases = [
        ("Would you please run a journal pass?", "journal_pass"),
        ("Please journal the last 47 minutes", "journal_discussion_window_pass"),
        ("Journal our chat discussion for the last day", "journal_discussion_window_pass"),
        ("Run a self review", "self_review"),
        ("Run your dream cycle", "dream_cycle"),
        ("Run through your concept induction graphs", "concept_induction_pass"),
    ]
    for prompt, expected_workflow_id in cases:
        req, debug, _ = hub_builder.build_chat_request(
            payload={"mode": "auto"},
            session_id="sid-workflow",
            user_id="user-1",
            trace_id="trace-workflow",
            default_mode="brain",
            auto_default_enabled=False,
            source_label="hub_http",
            prompt=prompt,
        )
        assert req.metadata["workflow_request"]["workflow_id"] == expected_workflow_id
        assert req.mode == "brain"
        assert req.route_intent == "none"
        assert debug["fallback_route"] == "workflow_lane"


def test_explicit_named_workflow_takes_precedence_over_chat_general_verb_selection() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={"mode": "auto", "verbs": ["chat_general"]},
        session_id="sid-workflow",
        user_id="user-1",
        trace_id="trace-workflow",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Would you please run a journal pass?",
    )

    assert req.mode == "brain"
    assert req.route_intent == "none"
    assert req.verb is None
    assert req.metadata["workflow_request"]["workflow_id"] == "journal_pass"
    assert debug["workflow_resolution_reason"] == "explicit_named_workflow_match"


def test_http_and_ws_workflow_resolution_parity_for_explicit_named_workflow() -> None:
    prompt = "Run through your concept induction graphs"
    http_req, _, _ = hub_builder.build_chat_request(
        payload={"mode": "auto"},
        session_id="sid-workflow",
        user_id="user-1",
        trace_id="trace-http",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt=prompt,
    )
    ws_req, _, _ = hub_builder.build_chat_request(
        payload={"mode": "auto"},
        session_id="sid-workflow",
        user_id="user-1",
        trace_id="trace-ws",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_ws",
        prompt=prompt,
    )
    assert http_req.metadata["workflow_request"]["workflow_id"] == ws_req.metadata["workflow_request"]["workflow_id"]
    assert http_req.mode == ws_req.mode == "brain"


def test_non_matching_prompt_preserves_existing_routing_behavior() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={'mode': 'auto'},
        session_id='sid-plain',
        user_id='user-1',
        trace_id='trace-plain',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='Please compare these deployment options',
    )

    assert req.mode == 'auto'
    assert req.route_intent == 'auto'
    assert req.metadata.get('workflow_request') is None
    assert debug['workflow_id'] is None


def test_named_workflow_prompt_with_schedule_and_notify_parses_execution_policy() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={'mode': 'auto'},
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='Run concept induction every Sunday and notify me when done',
    )

    policy = req.metadata['workflow_request']['execution_policy']
    assert policy['invocation_mode'] == 'scheduled'
    assert policy['notify_on'] == 'completion'
    assert policy['schedule']['kind'] == 'recurring'
    assert policy['schedule']['cadence'] == 'weekly'
    assert debug['workflow_execution_policy']['notify_on'] == 'completion'


def test_schedule_phrase_with_clock_time_routes_to_scheduled_policy() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={'mode': 'auto'},
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='Orion, would you schedule a self review for 2:46 PM?',
    )

    policy = req.metadata['workflow_request']['execution_policy']
    assert req.metadata['workflow_request']['workflow_id'] == 'self_review'
    assert policy['invocation_mode'] == 'scheduled'
    assert policy['schedule']['kind'] == 'one_shot'
    assert policy['schedule']['label'] == 'for 14:46'
    assert debug['workflow_execution_policy']['invocation_mode'] == 'scheduled'


def test_relative_schedule_phrase_in_one_minute_is_scheduled() -> None:
    req, _, _ = hub_builder.build_chat_request(
        payload={'mode': 'auto'},
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='Run a dream cycle in one minute',
    )

    policy = req.metadata['workflow_request']['execution_policy']
    assert req.metadata['workflow_request']['workflow_id'] == 'dream_cycle'
    assert policy['invocation_mode'] == 'scheduled'
    assert policy['schedule']['kind'] == 'one_shot'
    assert policy['schedule']['label'] == 'in one minute'


def test_management_prompt_routes_to_schedule_management_request() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={'mode': 'auto'},
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='What workflow runs do I have scheduled?',
    )

    assert req.metadata.get('workflow_request') is None
    assert req.metadata['workflow_schedule_management']['operation'] == 'list'
    assert debug['workflow_management_operation'] == 'list'


def test_management_update_prompt_builds_bounded_patch() -> None:
    req, _, _ = hub_builder.build_chat_request(
        payload={'mode': 'auto'},
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='Move my nightly journal pass to 10pm',
    )

    mgmt = req.metadata['workflow_schedule_management']
    assert mgmt['operation'] == 'update'
    assert mgmt['workflow_id'] == 'journal_pass'
    assert mgmt['patch']['hour_local'] == 22


def test_workflow_request_override_allows_bounded_modal_actions_without_prompt_matching() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            'mode': 'brain',
            'workflow_request_override': {
                'workflow_id': 'concept_induction_pass',
                'action': 'synthesize_to_journal',
            },
        },
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='irrelevant prompt text',
    )
    workflow_request = req.metadata['workflow_request']
    assert workflow_request['workflow_id'] == 'concept_induction_pass'
    assert workflow_request['action'] == 'synthesize_to_journal'
    assert workflow_request['execution_policy']['invocation_mode'] == 'immediate'
    assert debug['workflow_resolution_reason'] == 'workflow_request_override'


def test_workflow_request_override_preserves_concept_induction_details_for_synth_action() -> None:
    req, _, _ = hub_builder.build_chat_request(
        payload={
            'mode': 'brain',
            'workflow_request_override': {
                'workflow_id': 'concept_induction_pass',
                'action': 'synthesize_to_journal',
                'concept_induction_details': {
                    'profiles': [{'subject': 'orion', 'profile_id': 'profile-1', 'revision': 4}],
                    'trace': {'artifacts': {'lookup_rows': [{'subject': 'orion'}]}},
                },
            },
        },
        session_id='sid-workflow',
        user_id='user-1',
        trace_id='trace-workflow',
        default_mode='brain',
        auto_default_enabled=False,
        source_label='hub_http',
        prompt='ignored',
    )

    workflow_request = req.metadata['workflow_request']
    assert workflow_request['workflow_id'] == 'concept_induction_pass'
    assert workflow_request['action'] == 'synthesize_to_journal'
    assert workflow_request['concept_induction_details']['profiles'][0]['profile_id'] == 'profile-1'
