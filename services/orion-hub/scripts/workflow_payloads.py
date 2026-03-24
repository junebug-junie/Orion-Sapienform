from __future__ import annotations

from typing import Any, Dict


def _normalize_workflow(workflow: Any) -> Dict[str, Any] | None:
    if hasattr(workflow, 'model_dump'):
        workflow = workflow.model_dump(mode='json')
    if not isinstance(workflow, dict):
        return None
    workflow_id = str(workflow.get('workflow_id') or workflow.get('id') or '').strip()
    if not workflow_id:
        return None
    return workflow


def extract_workflow_payload(cortex_result: Any) -> Dict[str, Any] | None:
    if cortex_result is None:
        return None
    metadata = getattr(cortex_result, 'metadata', None)
    if hasattr(metadata, 'model_dump'):
        metadata = metadata.model_dump(mode='json')
    if not isinstance(metadata, dict):
        return None
    workflow = _normalize_workflow(metadata.get('workflow'))
    if workflow is None:
        return None

    request = metadata.get('workflow_request') if isinstance(metadata.get('workflow_request'), dict) else {}
    registry = metadata.get('available_workflows') if isinstance(metadata.get('available_workflows'), list) else []
    registry_entry = None
    for item in registry:
        if isinstance(item, dict) and str(item.get('workflow_id') or '') == workflow.get('workflow_id'):
            registry_entry = item
            break

    persisted = workflow.get('persisted') if isinstance(workflow.get('persisted'), list) else []
    scheduled = workflow.get('scheduled') if isinstance(workflow.get('scheduled'), list) else []
    status = str(workflow.get('status') or metadata.get('workflow_status') or getattr(cortex_result, 'status', '') or '').strip().lower() or None
    display_name = str(
        workflow.get('display_name')
        or (registry_entry or {}).get('display_name')
        or workflow.get('workflow_id')
        or ''
    ).strip() or None
    summary = str(workflow.get('main_result') or getattr(cortex_result, 'final_text', '') or '').strip() or None
    user_invocable = bool((registry_entry or {}).get('user_invocable', True))

    return {
        'id': workflow.get('workflow_id'),
        'workflow_id': workflow.get('workflow_id'),
        'display_name': display_name,
        'status': status,
        'summary': summary,
        'persisted': persisted,
        'scheduled': scheduled,
        'user_invocable': user_invocable,
        'invoked_from_chat': bool(request.get('invoked_from_chat')),
        'matched_alias': request.get('matched_alias'),
        'request': request,
        'raw_metadata': workflow,
    }
