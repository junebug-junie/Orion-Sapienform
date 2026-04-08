from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JS_PATH = REPO_ROOT / 'services' / 'orion-hub' / 'static' / 'js' / 'workflow-ui.js'


def _run_node(script: str) -> str:
    node = shutil.which('node')
    assert node is not None, 'node is required to exercise Hub workflow helpers'
    result = subprocess.run([node, '-e', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def test_workflow_ui_normalizes_metadata_and_chip_label() -> None:
    script = f"""
const helpers = require({json.dumps(str(JS_PATH))});
const workflow = helpers.normalizeWorkflow({{
  workflow_id: 'journal_pass',
  display_name: 'Journal Pass',
  status: 'completed',
  main_result: 'Drafted a reflective entry.',
  persisted: ['journal.entry.write.v1:abc'],
  request: {{ matched_alias: 'do a journal pass', invoked_from_chat: true }},
  user_invocable: true,
}});
console.log(JSON.stringify({{
  id: workflow.id,
  badge: helpers.getWorkflowBadgeLabel(workflow),
  status: helpers.getWorkflowStatusLabel(workflow.status),
  details: helpers.buildWorkflowDetailRows(workflow),
}}));
"""
    payload = json.loads(_run_node(script))

    assert payload['id'] == 'journal_pass'
    assert payload['badge'] == 'Workflow · Journal Pass'
    assert payload['status'] == 'Completed'
    assert any(row[0] == 'Persisted' and 'journal.entry.write.v1:abc' in row[1] for row in payload['details'])


def test_workflow_ui_run_again_visibility_rules_and_non_workflow_passthrough() -> None:
    script = f"""
const helpers = require({json.dumps(str(JS_PATH))});
const completed = helpers.normalizeWorkflow({{
  workflow_id: 'self_review',
  display_name: 'Self Review',
  status: 'completed',
  summary: 'Reflective summary',
  user_invocable: true,
  request: {{ matched_alias: 'run a self review', invoked_from_chat: true }},
}});
const failed = helpers.normalizeWorkflow({{
  workflow_id: 'self_review',
  display_name: 'Self Review',
  status: 'failed',
  user_invocable: true,
}});
console.log(JSON.stringify({{
  showCompleted: helpers.shouldShowWorkflowForMessage({{ workflow: completed }}),
  canRunAgainCompleted: helpers.canRunAgain(completed),
  canRunAgainFailed: helpers.canRunAgain(failed),
  canRunAgainMissing: helpers.canRunAgain(null),
  missingWorkflow: helpers.shouldShowWorkflowForMessage({{ message_id: 'plain-1', title: 'hello' }}),
}}));
"""
    payload = json.loads(_run_node(script))

    assert payload['showCompleted'] is True
    assert payload['canRunAgainCompleted'] is True
    assert payload['canRunAgainFailed'] is False
    assert payload['canRunAgainMissing'] is False
    assert payload['missingWorkflow'] is False
