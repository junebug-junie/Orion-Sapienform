from __future__ import annotations

from orion.cognition.compactor.calendar_day import DEFAULT_COMPACTOR_TIMEZONE

CHAT_DEV_DIGEST_TAG = "chat_dev_digest"
COMPACTOR_KIND = "chat_history_log"
DEFAULT_TIMEZONE = DEFAULT_COMPACTOR_TIMEZONE
DEFAULT_LOOKBACK_HOURS = 24
# The compactor asks the discussion-window skill for every turn in its window,
# not a trailing slice: the digest is embedded verbatim in the daily letter, so
# a busy day must be covered in full (volume is handled by map-reduce). This is
# a ceiling on one fetch (DiscussionWindowRequestV1.max_turns le=5000), far above
# live volume (5-14 turns/day, 2026-09-20..29). Hitting it is reported as
# `input_truncated` in workflow metadata, never silently.
COMPACTOR_MAX_TURNS = 5000

CARD_SUMMARY_MAX_CHARS = 1600
JOURNAL_TITLE_MAX_CHARS = 120
# No journal_body cap: stored untruncated (embedded verbatim in the daily
# letter). Only the memory-card fields above stay capped.

# Per-turn safety caps before the digest LLM. Not a summarization budget: live
# turns top out ~1.5k chars (2026-09-20..29), so these only stop one runaway
# paste from filling a whole chunk. Must stay well under
# compactor.constants.DIGEST_INPUT_CHAR_BUDGET.
DIGEST_TURN_PROMPT_MAX_CHARS = 8000
DIGEST_TURN_RESPONSE_MAX_CHARS = 16000
