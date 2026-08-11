from __future__ import annotations

CHAT_DEV_DIGEST_TAG = "chat_dev_digest"
COMPACTOR_KIND = "chat_history_log"
DEFAULT_TIMEZONE = "America/Denver"
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_MAX_TURNS = 30

CARD_SUMMARY_MAX_CHARS = 1600
JOURNAL_TITLE_MAX_CHARS = 120
JOURNAL_BODY_MAX_CHARS = 4000

# Per-turn trim before digest LLM.
# Live-verified 2026-08-11: a real pasted-transcript turn was 710/1559 chars
# (prompt/response) -- well past the old 400/600 caps -- so the digest LLM
# was fed a hard mid-sentence fragment (< half the turn) for any turn that
# ran long, producing thin and inaccurate digests. Raised to give normal
# multi-paragraph turns headroom; still bounded so a runaway paste can't
# blow the LLM context (max_turns=30 * (1200+2000) ~= 96k chars worst case).
DIGEST_TURN_PROMPT_MAX_CHARS = 1200
DIGEST_TURN_RESPONSE_MAX_CHARS = 2000
