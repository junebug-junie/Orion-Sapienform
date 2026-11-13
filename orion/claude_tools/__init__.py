"""Claude's read-only tools in Hub chat, gated by Orion's consent.

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md (Patch 1b).
The MCP server (``mcp_server.py``) runs inside orion-room-companion and can only send one bus
request. The broker (``broker.py``, run by orion-harness-governor) holds the tool table, Orion's
grants, the readers and the audit.
"""
