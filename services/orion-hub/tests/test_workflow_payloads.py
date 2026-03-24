from __future__ import annotations

import importlib.util
from pathlib import Path

from orion.schemas.cortex.contracts import CortexClientResult


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'workflow_payloads.py'
SPEC = importlib.util.spec_from_file_location('hub_workflow_payloads', MODULE_PATH)
assert SPEC and SPEC.loader
workflow_payloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow_payloads)


def test_extract_workflow_payload_returns_normalized_shape() -> None:
    result = CortexClientResult(
        ok=True,
        mode='brain',
        verb='dream_cycle',
        status='success',
        final_text='Workflow: Dream Cycle',
        memory_used=False,
        recall_debug={},
        steps=[],
        metadata={
            'workflow_status': 'completed',
            'workflow_request': {'matched_alias': 'run your dream cycle', 'invoked_from_chat': True},
            'available_workflows': [
                {'workflow_id': 'dream_cycle', 'display_name': 'Dream Cycle', 'user_invocable': True},
            ],
            'workflow': {
                'workflow_id': 'dream_cycle',
                'display_name': 'Dream Cycle',
                'status': 'completed',
                'main_result': 'Dream synthesis complete.',
                'persisted': ['dream.result.v1'],
                'scheduled': [],
            },
        },
    )

    payload = workflow_payloads.extract_workflow_payload(result)

    assert payload is not None
    assert payload['id'] == 'dream_cycle'
    assert payload['display_name'] == 'Dream Cycle'
    assert payload['status'] == 'completed'
    assert payload['user_invocable'] is True
    assert payload['persisted'] == ['dream.result.v1']
    assert payload['matched_alias'] == 'run your dream cycle'
