"""Shared digest-call budgets for the compactor workflows (GitHub + chat).

Both compactors now summarize a FULL window (all merged PRs / all chat turns of
the previous Denver calendar day) and store the journal body untruncated, because
the digest text is embedded verbatim in the daily letter. Volume is handled by
map-reduce (chunk digests, then one merge call), not by dropping input.
"""
from __future__ import annotations

# Gateway route for every digest call (chunk, single, or merge). `agent` is
# durable-admission work (config/gpu_pool.yaml: class agent -> roles
# [agent, agent-gpu2, chat], on_unavailable: backlog): the pool itself already
# spills onto chat's card only while Juniper has it lent, so naming `chat` here
# would poach Juniper's reserved Hub lane (scripts/check_chat_route_poachers.py).
# Each call runs inside an admitted durable run (workflow `compactor.digest`)
# holding a GPU pool hold: a busy pool is a checkpointed wait, and a failed call
# is one of DURABLE_RUNS_RETRY_MAX_ATTEMPTS bounded attempts -- there is no
# in-process retry loop any more.
DIGEST_LLM_ROUTE = "agent"

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

# Wall-clock budgets. Verb YAML timeout_ms (exec step) and the per-call RPC wait
# the durable run gives each digest call (verb + bus slack; also the admitted
# node's execute() timeout, brief.timeout_sec).
DIGEST_VERB_TIMEOUT_MS = 600_000
DIGEST_ORCH_RPC_TIMEOUT_SEC = 660.0

# How long a compactor.digest run may wait for the GPU pool and work, counted from
# the END of the window it digests (admission.deadline_at). For the scheduled day
# window (ends at Denver midnight, dispatched ~06:00) that is ~18h: yesterday's
# digest still lands before today's pass. Derived from the window, not from "now",
# so a re-submission of the same window is byte-identical (idempotent run_id).
COMPACTOR_RUN_DEADLINE_AFTER_WINDOW_SEC = 24 * 3600

# The durable run's finalize call back into cortex-orch (memory card + journal
# write, no LLM).
COMPACTOR_FINALIZE_RPC_TIMEOUT_SEC = 180.0

# A window whose durable run ended failed/cancelled is re-submitted under the next
# generation (`<run_id>:g2`, ...), at most this many per identical input.
COMPACTOR_MAX_RUN_GENERATIONS = 5
