from __future__ import annotations

REPO_DEV_SNAPSHOT_SLOT = "repo_dev_snapshot"
REPO_DEV_SNAPSHOT_TAG = "repo_dev_snapshot"

PR_BODY_MAX_CHARS = 2000
CARD_SUMMARY_MAX_CHARS = 800
JOURNAL_TITLE_MAX_CHARS = 120
JOURNAL_BODY_MAX_CHARS = 4000

DEFAULT_LOOKBACK_DAYS = 1
# Cap PR rows passed to the digest LLM (bodies are already truncated per-PR at fetch).
MAX_DIGEST_INPUT_PRS = 8
# Digest prompt uses a tighter body cap than fetch to keep LLM context bounded.
DIGEST_INPUT_BODY_MAX_CHARS = 600
