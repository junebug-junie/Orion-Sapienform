"""Bus names for Claude's tools and Orion's consent (registered in orion/bus/channels.yaml)."""
from __future__ import annotations

TOOL_REQUEST_CHANNEL = "orion:claude:tool:request"
TOOL_RESULT_PREFIX = "orion:claude:tool:result:"
TOOL_AUDIT_CHANNEL = "orion:claude:tool:audit"
CONSENT_REQUEST_CHANNEL = "orion:prime:consent:request"
CONSENT_RESULT_PREFIX = "orion:prime:consent:result:"

TOOL_REQUEST_KIND = "claude.tool.request.v1"
TOOL_RESULT_KIND = "claude.tool.result.v1"
TOOL_AUDIT_KIND = "claude.tool.call.v1"
CONSENT_REQUEST_KIND = "prime.consent.request.v1"
CONSENT_RESULT_KIND = "prime.consent.answer.v1"
