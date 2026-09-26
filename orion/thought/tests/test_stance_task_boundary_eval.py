import copy

import pytest
from pydantic import ValidationError

from orion.thought.evals.stance_task_boundary import assess_response


def response():
    return {'imperative': 'Read the paper using the handoff.', 'tone': 'Careful',
            'strain_refs': [], 'evidence_refs': ['hub:turn:test'],
            'stance_harness_slice': {'task_mode': 'technical_collaboration',
                                     'conversation_frame': 'technical',
                                     'answer_strategy': 'source_review'}}


def test_accepts_actual_stance_without_transport_metadata():
    assert assess_response(response(), 'test')['valid_stance']


@pytest.mark.parametrize('key', ['grounding_capsule', 'autonomy_slice', 'summary', 'priors_tested'])
def test_rejects_invented_or_downstream_fields_without_mutating_input(key):
    raw = response()
    raw[key] = {}
    before = copy.deepcopy(raw)
    result = assess_response(raw, 'test')
    assert not result['valid_stance']
    assert key in result['unexpected_keys']
    assert raw == before


@pytest.mark.parametrize('key', ['imperative', 'tone', 'strain_refs', 'stance_harness_slice'])
def test_missing_stance_fields_are_not_synthesized(key):
    raw = response()
    del raw[key]
    with pytest.raises(ValidationError):
        assess_response(raw, 'test')


def test_rejects_ungrounded_evidence():
    raw = response()
    raw['evidence_refs'] = ['invented']
    assert not assess_response(raw, 'test')['valid_stance']
