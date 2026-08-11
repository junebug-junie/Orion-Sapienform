# Dev-economics session ledger: real token/cost/duration parser + replay

Status: **DONE**. Implements `docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md`'s
own "Recommended next patch": `claude_code_ingest.py` extended to produce normalized per-session
usage records (real token counts, model, effort tier, wall-clock duration, assistant + human
word counts), run against this machine's real transcript history, per that doc's own acceptance
check. Offline only -- no pricing table, no bus wiring, no service -- per that doc's explicit
phasing ("before pricing, before Cursor, before any coupling decision").

This is the largest single cluster (~10 of the ~25 originally-brainstormed ideas from PR #1491)
of the "other 20 ideas" backlog. `doc-semantic-drift` and the remaining noted-only ideas
(concurrent-session-count, time-engaged-vs-24h) are still open, tracked separately.

## What shipped

- `orion/dev_economics/claude_code_ingest.py`: new `SessionUsageRecord` dataclass +
  `parse_session_usage_record()` + `iter_all_session_usage_records()`, sharing the existing
  transcript-walking primitives (`iter_transcript_files`, the vanished-file/subdirectory
  resilience already built for the affective-state signal). One record per transcript *file*,
  including subagent transcripts as their own records (`is_subagent` flag) -- excluding them
  would systematically undercount real $ cost, since subagent dispatch is real, separately
  billed API usage.
- `scripts/replay_dev_economics_ledger.py`: the doc's own acceptance check -- a normalized table
  Juniper can eyeball directly (token totals, model mix, session count, duration).
- 26 tests, all passing.

## A critical bug, self-caught before it shipped

First pass summed token usage and turn count **per JSONL line**. Code review (`orion-repo-agent`)
found this overcounts real tokens by **1.8x-2.7x**: a single logical assistant turn
(thinking -> text -> N tool_use blocks) is logged as multiple separate lines sharing one
`message.id`, each repeating the identical, final, cumulative `usage` for that turn. Verified
against the full real corpus (1,280 files) before and after a dedup-by-`message.id` fix:

| field | before (buggy) | after (fixed) |
|---|---|---|
| input_tokens | ~4.18M | ~2.27M |
| output_tokens | ~106M | ~39.3M |
| cache_creation_input_tokens | ~1.52B | ~654M |
| cache_read_input_tokens | ~43.1B | ~22.5B |

Fixing this surfaced a **second**, self-inflicted bug before it ever shipped: the first dedup
attempt also gated word-count accumulation behind "first occurrence of this `message.id`" --
but the real `text` content block for a multi-line turn doesn't reliably land on the first line
of the group (a thinking-only or tool_use-only line commonly comes first). That silently dropped
~66% of real assistant word count (2,033,660 -> 691,702 in the live replay). Caught by comparing
the replay's own before/after numbers, not by review -- fixed by decoupling word-count
accumulation (unconditional, every line -- safe because at most one line per real turn ever
carries a `text` block, verified against the full corpus) from the token/turn-count dedup (still
gated by `message.id`).

Two further real findings from the same review, both fixed:

- **`duration_sec` skew from non-conversational line types**: `_observe()` originally ran on
  every line's timestamp, not just `user`/`assistant` turns. Confirmed live: a real session's ID
  got reused 11.7 days after its actual last chat turn just to record an unrelated `pr-link`
  line, skewing that one session's reported duration to ~11.9 days. Fixed by restricting
  `_observe()` to `obj_type in ("user", "assistant")`.
- **Subagent dispatch prompts miscounted as Juniper's words**: a subagent transcript's first
  `user` line is the orchestrator's own Task-tool dispatch prompt -- a plain string with
  `promptSource`/`origin` genuinely absent (not because it's old data), which otherwise passes
  the shared human-turn filter's "conservative toward inclusion" rule. Confirmed live: this
  affected essentially every real subagent transcript. Fixed by gating human-turn counting on
  `not is_subagent`.

A third, smaller finding (`model == "<synthetic>"` harness-injected placeholder/error turns,
e.g. `"API Error: Server error mid-response"`) was caught and fixed earlier in this same session,
before the formal review pass, via direct inspection of the real corpus.

## Real replay results (1,280 real transcripts: 114 top-level sessions, 1,166 subagent)

- input_tokens: 2,273,052 · output_tokens: 39,362,007 · cache_creation_input_tokens: 653,878,312
  · cache_read_input_tokens: 22,470,808,230
- total assistant visible word count: 2,044,856 (text-block content only -- excludes internal
  `thinking` and `tool_use` blocks, matching Juniper's own "how much do I have to read" framing)
- total human word count (top-level sessions only): 372,964
- Model mix: `claude-sonnet-5` dominant (1,146 transcripts), with real minority usage of
  `claude-fable-5`, `claude-opus-4-8`, `claude-opus-5`, `claude-haiku-4-5-20251001`, and several
  non-Anthropic models (`z-ai/glm-*`, `xai/grok-4.5`, `openai/o3`, `moonshotai/kimi-*`) -- real,
  non-degenerate diversity, not a single-model monoculture.
- Effort mix: high (522) dominates over medium/xhigh/low (34/14/14 combined) -- plausible given
  this repo's own default-to-high convention, not obviously wrong.
- Top-level session duration: median 235.9 min (~4 hours), mean pulled up to 1834.9 min by a
  small number of very long-lived/resumed sessions (max 306.2 hr / ~12.75 days) -- the design
  doc's own flagged concern ("any time you step away mid-session inflates it") confirmed real,
  not a bug this patch introduces or claims to solve.

Per-session records intentionally not committed to the repo (aggregate distribution stats only,
in this report) -- same privacy discipline as the affective-state replay.

## Cursor investigation (design doc's acceptance check)

Doc requires "a plain yes/no answer (does comparable local data exist) before any Cursor
ingestion code is written." Answer: **yes**, comparable local data exists --
`~/.cursor/projects/` (2,134 files) and `~/.cursor/chats/<workspace>/<session>/` (14 session
directories) on this machine. Format not investigated further -- per the doc's own phasing,
that's explicitly the next step after this patch, not part of it. Not writing `cursor_ingest.py`
against an assumed schema.

## Tests run

```text
venv/bin/python3 -m pytest orion/dev_economics/tests -q
26 passed
```

## Evals run

`scripts/replay_dev_economics_ledger.py` against the real local transcript corpus *is* the eval
this patch's governing spec required -- see "Real replay results" above.

## Docker/build/smoke checks

None -- no service/container/bus changes in this patch (offline library + script only).

## Review findings fixed

Two `orion-repo-agent` review passes: one initial pass (found the critical token-overcount bug
plus the duration-skew and subagent-dispatch-prompt bugs), one verification pass after fixing
(confirmed the fix was correct and complete, found two low-severity nits — an empty-string
`message.id` edge case with zero real occurrences, and a test-coverage gap in the dedup test's
own fixture). All fixed; see "A critical bug, self-caught before it shipped" above for the
material ones. The two low-severity nits: `message_id` dedup now explicitly excludes an empty
string (`isinstance(message_id, str) and message_id`), and the dedup test's fixture was rebuilt
to match the real corpus's actual line shape (thinking-only/tool_use-only lines around the one
line that carries real text) with an explicit `assistant_word_count` assertion, rather than an
unrealistic all-lines-have-text fixture that couldn't have caught the word-count regression.

## Restart required

No restart required -- no running service changed.

## Risks / concerns

- Severity: note, not blocking.
  - Concern: `duration_sec`'s "first timestamp to last timestamp" definition is still an
    imprecise proxy for real engaged time on any multi-day resumed session -- the design doc's
    own open question, not resolved by this patch (deliberately; that's a "coarser
    self-reported/estimated field" design decision, not a parsing bug).
  - Mitigation: none needed for this patch's scope; worth flagging to Juniper before any future
    consumer treats `duration_sec` as precise.
- Severity: note, not blocking.
  - Concern: pricing (`pricing.py`) is explicitly out of scope for this patch -- current output
    is token counts only, not $ cost.
  - Mitigation: next natural slice per the design doc's own phasing.

## PR link

(added after push)
