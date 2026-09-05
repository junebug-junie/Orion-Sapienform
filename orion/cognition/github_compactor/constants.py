from __future__ import annotations

REPO_DEV_SNAPSHOT_SLOT = "repo_dev_snapshot"
REPO_DEV_SNAPSHOT_TAG = "repo_dev_snapshot"

PR_BODY_MAX_CHARS = 2000
CARD_SUMMARY_MAX_CHARS = 800
JOURNAL_TITLE_MAX_CHARS = 120
# Sized for a high-merge day (~30 PRs): the digest LLM still owns an 8k-token
# completion budget, so the journal body can carry more than a short blurb.
JOURNAL_BODY_MAX_CHARS = 8000

DEFAULT_LOOKBACK_DAYS = 1
# Cap PR rows passed to the digest LLM (bodies are already truncated per-PR at fetch).
# Raised 8 -> 32 so a ~30-merge day reaches the summarizer instead of being
# silently truncated before the LLM call (live miss 2026-09-05).
MAX_DIGEST_INPUT_PRS = 32
# Digest prompt uses a tighter body cap than fetch to keep LLM context bounded.
# Raised alongside the chat compactor's equivalent caps (see
# chat_history_compactor/constants.py): real PR bodies in this repo commonly
# run 1500-4000+ chars (markdown headers, bullet lists, code fences), and the
# old 600-char cap combined with a blind char slice (fixed to word-boundary
# truncation in the same patch) discarded most of a typical PR body before
# the digest LLM ever saw it. 32 items * 1500 chars is large but still within
# the chat-worker context window this verb uses.
DIGEST_INPUT_BODY_MAX_CHARS = 1500

# Wall-clock budgets must stay aligned: verb YAML timeout_ms, exec step timeout,
# and orch's call_verb_runtime wait. Orch adds a short bus slack over the verb.
DIGEST_VERB_TIMEOUT_MS = 600_000
DIGEST_ORCH_RPC_TIMEOUT_SEC = 660.0
# Fetch can hit one GitHub /files call per merged PR (sequential). With
# per_page=100 that easily exceeds the old 120s orch wait on a busy day.
GITHUB_FETCH_ORCH_RPC_TIMEOUT_SEC = 300.0
