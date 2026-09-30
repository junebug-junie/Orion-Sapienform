from __future__ import annotations

from orion.cognition.compactor.constants import (  # noqa: F401  (re-exported for callers)
    DIGEST_ORCH_RPC_TIMEOUT_SEC,
    DIGEST_VERB_TIMEOUT_MS,
)

REPO_DEV_SNAPSHOT_SLOT = "repo_dev_snapshot"
REPO_DEV_SNAPSHOT_TAG = "repo_dev_snapshot"

# Safety cap on ONE PR body, applied at fetch. Not a summarization budget: the
# digest now sees every merged PR in full and handles volume by map-reduce.
# Real PR bodies here are long markdown PR reports (live 2026-09-26..29: median
# ~9-10k chars, max 38.6k), so the cap sits above nearly all of them and only
# guards against a pathological body (a pasted log) eating a whole chunk. It
# must stay well under compactor.constants.DIGEST_INPUT_CHAR_BUDGET.
PR_BODY_MAX_CHARS = 30_000
CARD_SUMMARY_MAX_CHARS = 800
JOURNAL_TITLE_MAX_CHARS = 120
# No journal_body cap: the body is stored untruncated (embedded verbatim in the
# daily letter). Only the memory-card fields above stay capped.

DEFAULT_LOOKBACK_DAYS = 1

# GitHub `pulls?state=closed&sort=updated` pagination: stop once a page's oldest
# updated_at is before the window start (a PR merged in the window has
# updated_at >= merged_at >= window start). This caps the walk on a
# pathological history, and a hit is reported as `page_cap_hit` in the fetch.
GITHUB_PULLS_PER_PAGE = 100
GITHUB_PULLS_MAX_PAGES = 20

# Fetch can hit one GitHub /files call per merged PR (sequential). With
# per_page=100 that easily exceeds the old 120s orch wait on a busy day.
GITHUB_FETCH_ORCH_RPC_TIMEOUT_SEC = 300.0
