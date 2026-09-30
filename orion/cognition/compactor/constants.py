"""Shared digest-call budgets for the compactor workflows (GitHub + chat).

Both compactors now summarize a FULL window (all merged PRs / all chat turns of
the previous Denver calendar day) and store the journal body untruncated, because
the digest text is embedded verbatim in the daily letter. Volume is handled by
map-reduce (chunk digests, then one merge call), not by dropping input.
"""
from __future__ import annotations

# Gateway routes tried in order for every digest call (chunk, single, or merge).
# `agent` is durable-admission work (config/gpu_pool.yaml: class agent -> roles
# [agent, agent-gpu2, chat], on_unavailable: backlog): the pool itself already
# spills onto chat's card only while Juniper has it lent, so naming `chat` here
# would poach Juniper's reserved Hub lane (scripts/check_chat_route_poachers.py).
# The second entry is a plain retry on the same route: live failures were
# transient (empty completion / truncated JSON), not route-specific.
DIGEST_LLM_ROUTES: tuple[str, ...] = ("agent", "agent")

# Per-call input budget, measured as len(json.dumps(item)) summed over the items
# in a chunk. Sized to the SMALLEST context a digest call can land on: the agent
# class may fall through to chat's card (65k tokens/slot). 100k chars of JSON is
# ~30-33k tokens (+~10% for the template's indent=2 rendering), leaving room for
# the ~1k-token template and DIGEST_MAX_TOKENS of completion.
DIGEST_INPUT_CHAR_BUDGET = 100_000

# Completion budget per digest call, passed as options.max_tokens (exec's
# `_resolve_llm_chat_max_tokens` honors ctx.max_tokens first). The old path used
# LLM_CHAT_GENERAL_MAX_TOKENS (8000 live), and a full-day narrative with no
# journal_body cap needs headroom so the JSON object is not cut at
# finish_reason=length (-> structured_output_rejected).
DIGEST_MAX_TOKENS = 16_000

# Wall-clock budgets. Verb YAML timeout_ms (exec step), orch's per-call RPC wait
# (verb + bus slack), and a whole-pass ceiling across every digest call of one
# workflow run (chunks + merge + retries). The actions scheduler's RPC wait
# (ACTIONS_WORKFLOW_DISPATCH_TIMEOUT_SECONDS) must exceed
# fetch wait + COMPACTOR_DIGEST_TOTAL_BUDGET_SEC.
DIGEST_VERB_TIMEOUT_MS = 600_000
DIGEST_ORCH_RPC_TIMEOUT_SEC = 660.0
COMPACTOR_DIGEST_TOTAL_BUDGET_SEC = 3000.0
# A digest call with less than this left on the pass budget is not started.
DIGEST_MIN_CALL_SEC = 60.0
