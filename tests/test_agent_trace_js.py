from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "agent-trace.js"


def _run_node(script: str) -> str:
    node = shutil.which("node")
    assert node is not None, "node is required to exercise Hub agent trace helpers"
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def test_agent_trace_helpers_gate_group_and_timeline() -> None:
    script = f"""
const helpers = require({json.dumps(str(JS_PATH))});
const summary = {{
  mode: 'agent',
  summary_text: 'ok',
  tools: [
    {{ tool_id: 'planner_react', tool_family: 'planning', count: 1, duration_ms: 100 }},
    {{ tool_id: 'recall', tool_family: 'recall', count: 1, duration_ms: 80 }},
    {{ tool_id: 'analyze_text', tool_family: 'reasoning', count: 2, duration_ms: 140 }},
  ],
  steps: [
    {{ index: 0, event_type: 'planner_decision', tool_id: 'planner_react', tool_family: 'planning', action_kind: 'decide', effect_kind: 'read_only', status: 'success', duration_ms: 100, summary: 'Planner selected analyze_text.' }},
    {{ index: 1, event_type: 'tool_call', tool_id: 'recall', tool_family: 'recall', action_kind: 'retrieve', effect_kind: 'read_only', status: 'success', duration_ms: 80, summary: 'Recall retrieved 2 memory item(s).' }},
  ],
}};
const grouped = helpers.groupToolsByFamily(summary.tools);
const timeline = helpers.buildTimelineRows(summary);
console.log(JSON.stringify({{
  show: helpers.shouldShowAgentTrace(summary),
  emptyShow: helpers.shouldShowAgentTrace({{ mode: 'agent', tools: [], steps: [], summary_text: '' }}),
  groupedFamilies: grouped.map((item) => item.family),
  groupedCounts: grouped.map((item) => item.count),
  timelineSummary: timeline[0].summary,
  timelineDuration: timeline[1].duration_label,
}}));
"""
    payload = json.loads(_run_node(script))

    assert payload["show"] is True
    assert payload["emptyShow"] is False
    assert payload["groupedFamilies"] == ["planning", "recall", "reasoning"]
    assert payload["groupedCounts"] == [1, 1, 2]
    assert payload["timelineSummary"] == "Planner selected analyze_text."
    assert payload["timelineDuration"] == "80 ms"


def test_agent_trace_helpers_gate_message_level_debug_sections() -> None:
    script = f"""
const helpers = require({json.dumps(str(JS_PATH))});
const message = {{
  message_id: 'msg-1',
  agent_trace: {{
    mode: 'agent',
    summary_text: 'Agent executed tools.',
    tools: [{{ tool_id: 'planner_react', tool_family: 'planning', count: 1, duration_ms: 100 }}],
    steps: [],
  }},
}};
console.log(JSON.stringify({{
  show: helpers.shouldShowAgentTraceForMessage(message),
  missing: helpers.shouldShowAgentTraceForMessage({{ message_id: 'msg-2' }}),
  extractedMode: helpers.extractAgentTrace(message).mode,
}}));
"""
    payload = json.loads(_run_node(script))

    assert payload["show"] is True
    assert payload["missing"] is False
    assert payload["extractedMode"] == "agent"


def test_live_agent_step_anchors_to_conversation_not_body() -> None:
    script = f"""
const helpers = require({json.dumps(str(JS_PATH))});

class El {{
  constructor(tag) {{
    this.tagName = tag;
    this.id = '';
    this.className = '';
    this.textContent = '';
    this.dataset = {{}};
    this.children = [];
    this.parent = null;
    this.scrollTop = 0;
    this.scrollHeight = 0;
  }}
  appendChild(child) {{
    child.parent = this;
    this.children.push(child);
    this.scrollHeight += 1;
    return child;
  }}
  querySelector(sel) {{
    if (sel === '.agent-live-trace__steps') {{
      return this.children.find((c) => c.className === 'agent-live-trace__steps') || null;
    }}
    return null;
  }}
}}

const conversation = new El('div');
conversation.id = 'conversation';
const body = new El('body');
const byId = {{ conversation, body }};

const doc = {{
  getElementById(id) {{ return byId[id] || null; }},
  createElement(tag) {{ return new El(tag); }},
  body,
}};

helpers.appendLiveAgentStep('corr-1', {{
  step_index: 0,
  tool_id: 'python_interpreter',
  duration_ms: 120,
  observation: 'matched services/orion-hub/scripts/cortex_request_builder.py',
}}, doc);

const panel = byId['agent-live-corr-1'] || conversation.children.find((c) => c.id === 'agent-live-corr-1');
console.log(JSON.stringify({{
  anchorId: helpers.LIVE_TRACE_ANCHOR_ID,
  resolvedAnchor: helpers.resolveLiveTraceAnchor(doc) === conversation,
  panelInConversation: conversation.children.some((c) => c.id === 'agent-live-corr-1'),
  panelOnBody: body.children.some((c) => c.id === 'agent-live-corr-1'),
  hasStepsHost: Boolean(panel && panel.querySelector('.agent-live-trace__steps')),
  stepCount: panel ? panel.querySelector('.agent-live-trace__steps').children.length : 0,
}}));
"""
    payload = json.loads(_run_node(script))

    assert payload["anchorId"] == "conversation"
    assert payload["resolvedAnchor"] is True
    assert payload["panelInConversation"] is True
    assert payload["panelOnBody"] is False
    assert payload["hasStepsHost"] is True
    assert payload["stepCount"] == 1
