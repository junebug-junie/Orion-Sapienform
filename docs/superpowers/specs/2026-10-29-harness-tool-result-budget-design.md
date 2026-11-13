# Harness tool-result budget: keep every list item, not the first one

Status: design (no runtime code changed). Date: 2026-10-29.
Incident turn: `e0fdd52a-2cf9-4531-8383-bd379d70a0a1` (2026-10-27 05:43 UTC).
Juniper asked: "Had a few cool PRs come out. What do you think?"

## Arsonist summary

Orion asked GitHub for the 5 newest PRs. GitHub sent back about 62,000 characters,
because every PR comes with its full description. Our MCP proxy cuts each tool
result to the first 12,000 characters. One PR description alone is longer than
that, so Orion saw **one PR out of five**, cut off mid-sentence.

Orion did not have what it asked for, so it asked again. It made the same call 3
times, then tried pages 2, 3 and 4, then 10 per page. Each call added about 3,900
tokens. The lane auto-compacts at about 36,000 tokens and the turn starts at
about 28,000, so there is room for only about two of these results. The CLI
compacted twice, the second summary lost the task, and the turn took about 8
minutes. The final answer then gave a wrong title for PR #2918.

A smaller cap would **not** have saved this turn. A smaller cap shows Orion even
less of the first PR, and Orion would loop more. The fix is to shape the result
so every item in a list survives, with long text fields clipped. Five PRs with
bodies clipped to 400 characters fit in about 7,000 characters (about 1,800
tokens). That fits in one call, under the cap, with room to spare.

## Current architecture

- Motor: `orion/harness/fcc_motor.py` runs `claude -p` 2.1.311 with
  `--input-format stream-json` inside `orion-athena-harness-governor`. Lane window
  65,536 tokens, local Qwen3.6-35B-A3B.
- Live env (container): `HARNESS_FCC_MAX_OUTPUT_TOKENS=16384`,
  `HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE=90`, `HARNESS_FCC_CONTEXT_PRESSURE_PCT=70`,
  `ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS=12000`, `MAX_MCP_OUTPUT_TOKENS` unset.
- Compaction line (`orion/fcc/context_budget.py::autocompact_trigger_tokens`):
  usable = 65,536 - 16,384 = 49,152; trigger = min(49,152 x 0.9, 49,152 - 13,000)
  = **36,152 tokens**. Matches the turn: the CLI did not compact at 35,894 input
  tokens (seq 23) and did compact one result later (`pre_tokens` 39,033).
- Tool-result cap: `orion/fcc/mcp_stdio_proxy.py` wraps the `github` MCP server
  (`orion/fcc/fcc_claude_mcp.template.json`). It keeps the **first** 11,880
  characters of any text block over 12,000 and appends
  `[orion-fcc-mcp-proxy: truncated N chars to 12000. Summarize from this excerpt; do not re-fetch the same bulk list.]`.
  It does not look at the shape of the payload. `firecrawl` is not proxied.
- Repeat guard: `orion/fcc/repeat_failure_breaker.py` (PreToolUse hook, every
  turn). It blocks a call only after it has **failed** 3 times. A call that
  succeeded is never blocked. The refetch rule (block a repeat of a successful
  fetch) is `WebFetch`-only and is armed only on reading-only turns.
- Context-pressure nudge: `build_context_pressure_step` emits
  "Context nearly full — answer from what you have; no more tools." It is a
  trace step for the operator UI only. Nothing writes it to the CLI's stdin, so
  **the model never sees it**. In this turn it fired at seq 22 and seq 41, and
  the model called a tool right after both times (seq 24, seq 43).
- Context telemetry (`context_obs` on each step): an estimate in characters,
  re-anchored to the provider's `usage` x 4 chars/token whenever an assistant
  event arrives (`fcc_motor.py` ~line 1731, `observed_context_chars`).

## Evidence from the turn

### Per tool call

| seq | call | raw size from GitHub | sent to model | PRs visible | input tokens next hop |
|---|---|---|---|---|---|
| 14 | ToolSearch `select:list_pull_requests` | - | `tool_reference` | - | 27,707 -> 28,090 |
| 17 | list_pull_requests perPage 5 | 62,280 chars | 12,001 chars | 1 (#2917) | 31,992 |
| 20 | identical | 62,280 | 12,001 | 1 (#2917) | 35,894 |
| 24 | identical | 62,280 | 12,001 | 1 (#2917) | compaction 1 (`pre_tokens` 39,033) |
| 36 | page 2 | 28,029 | 12,001 | 1 (#2911) | 30,893 |
| 39 | page 3 | 46,202 | 12,001 | 1 (#2909) | 35,120 |
| 43 | page 4 | 31,166 | 12,001 | 4 (#2906, #2902, #2901, #2675) | compaction 2 (`pre_tokens` 38,274) |
| 56 | page 1, perPage 10 | 93,337 | 12,001 | 1 (#2919) | 30,535 |

Each 12,001-character result costs **3,902 tokens** (31,992 - 28,090 = 35,894 -
31,992). Turn start was 28,090 tokens; headroom to the compaction line was
36,152 - 28,090 = **8,062 tokens = two results**. The third result compacted.
Every compaction was driven by `list_pull_requests` results; nothing else in the
turn was large.

Measured PR sizes (seq 44 text, complete objects): full PR object 1,495 / 5,807 /
4,476 chars; with `body` clipped to 400 chars: 1,374 / 1,463 / 1,459 chars. So
five PRs with clipped bodies = about **7,000 chars, about 1,800 tokens**.

### The 41k -> 120k jump (seq 34 -> 37)

Not a new payload. It is the telemetry catching up with reality. After a
compaction the motor rebases its estimate to the post-compaction message size
(seq 33: 35,720 chars; seq 34 adds the 5,211-char summary = 40,931). That
estimate cannot see the CLI system prompt and tool schemas. At seq 35 the first
assistant event reports `usage.input_tokens` = 26,983, and the motor re-anchors
to 26,983 x 4 = 107,932 chars. Seq 37 adds the 12,001-char tool result =
119,933. The same jump happens at seq 13 (32,202 -> 110,828 = 27,707 x 4) and at
seq 55 (33,335 -> 107,424). The real context after compaction is about 27,000
tokens: `post_tokens` 8,930 of messages plus about 18,000 tokens of fixed system
prompt and tool schemas.

One real extra cost: after each compaction the `SessionStart:compact` hook of the
context-mode plugin re-injects 12,056 chars (seq 31) / 12,146 chars (seq 50) of
`additionalContext` (about 3,000 tokens). That is part of the 18k fixed overhead
the next hop starts from. Not the main driver here; noted as a missing question.

### Why the identical call 3 times

1. Not a ToolSearch artifact. ToolSearch ran once (seq 14), returned a
   `tool_reference`, and the next hop called the real tool. The results that
   came back were real text (12,001 chars each), and input tokens grew by 3,902
   per result, so the model did see them.
2. Not a hook. No PreToolUse block or hook output appears between seq 17 and 25.
3. Every assistant hop is the same Qwen pattern: a `"...\n\n"` text block, then a
   tool_use, with `output_tokens: 0`. No reasoning text is recorded.
4. Most likely cause: the result did not answer the request. The model asked for
   5 PRs and got one PR cut off mid-body, with a marker saying "do not re-fetch".
   With no other way to get the list, it re-fetched, then paginated. The marker
   tells the model to stop but does not give it the data it needs. This is
   inference from the payloads; the model's reasoning is not recorded.

### Would `MAX_MCP_OUTPUT_TOKENS` have helped?

Claude Code 2.1.311 honors it. From the bundle in the container
(`/usr/local/bin/claude`): `MAX_MCP_OUTPUT_TOKENS` if > 0, else a remote flag
(`tengu_velvet_ibis.mcp_tool`), else **25,000 tokens**. Over the limit, the CLI
keeps the first `limit x 4` characters and appends
`[OUTPUT TRUNCATED - exceeded N token limit]` plus "use pagination or filtering".
That is the same head-first cut our proxy does.

- Any value >= 3,000 does nothing here: the proxy already cut to 12,000 chars
  (about 3,000 tokens) before the CLI sees the result.
- A lower value (for example 1,000 tokens = 4,000 chars) makes each result
  about 1,000 tokens, but each result then shows **less than one PR** (#2917's
  body alone is longer). The model has even less of what it asked for. Seven
  calls x ~1,000 tokens from 28,090 = ~35,000, just under the line, but the
  model would very likely make more calls. A cap does **not** fix this turn.
- It is also unproxied for `firecrawl`, so setting it is a reasonable backstop
  for that server. That is a separate small change.

### How common (last 14 days, `cockpit_turn_sighting`)

- 1,732 motor turns. 13 had >= 1 compaction, **8 had >= 2**, 4 had >= 3;
  33 compactions in total. Rare, but each one costs minutes and can lose the task.
- Tool called last before each compaction (tool_use content blocks only): Read
  10, `mcp__orion-introspect__curiosity` 6, Bash 6,
  `mcp__github__list_pull_requests` 4, firecrawl_scrape 2, node_history 2,
  others 1 each.
- Identical repeat calls are rare: 14 extra identical calls in 14 days in total.
  `list_pull_requests` 3 (2 turns), Read 3, ToolSearch 3, `pull_request_read` 2,
  curiosity 2, ScheduleWakeup 1.
- Proxy truncations: 17 of 377 recorded tool results, in 8 turns.
- Turns with >= 2 compactions are mostly long Bash/Read work (4 of 8) or many
  distinct introspection/scrape calls. This turn is the one where a single
  oversized list result is the whole story.

### Harm

The second compaction kept no summary of the task (seq 54,
`context_compaction_degenerate`). The final answer (seq 58/59) lists PR #2918
as "docs(spec): curiosity compaction schema-constrained answers + retry". The
real #2918 is "fix(hub): Zelda stops choosing wait in an empty room". No tool
result in this turn showed #2918, #2916 or #2915. The model filled the table in.

## Missing questions

1. Does Qwen re-fetch less when the result is complete but clipped? Expected yes,
   since a complete list answers the request. Check: replay the same prompt in
   the eval below.
2. Should the context-pressure nudge reach the model? Today it is UI-only and the
   wording claims more than it does. Options: deliver it through the existing
   PreToolUse hook (block new tools past the line, like the reading-only tool
   window rule), or rename it so nobody thinks the model was told.
3. Is the context-mode `SessionStart:compact` re-injection (~3,000 tokens every
   compaction) worth its cost on the 64k lane?
4. Should `firecrawl` go through the same proxy?

## Proposed schema / API changes

None. No bus, schema, or env changes.

### Patch 1 (recommended): item-preserving shrink in `mcp_stdio_proxy`

When a text block is over `max_chars` and the text parses as JSON:

1. If it is a list of objects (or an object holding one), clip every string
   value longer than a field limit, trying limits 2,000 -> 800 -> 400 -> 160 ->
   0 chars, until the re-serialized list fits in `max_chars`.
2. If it still does not fit with all strings clipped, keep the first k whole
   items that fit.
3. Append a marker that tells the truth: `[orion-fcc-mcp-proxy: N items, all
   shown; long text fields clipped to L chars (original M chars). Fetch one item
   by id for full text.]` or `... showing k of N items ...`.
4. If the text is not JSON, keep today's head cut unchanged.

Generic, not GitHub-specific: no per-tool field lists, no tool-name tracking.
On this turn: page 1 (62,280 chars) becomes about 7,000 chars with all 5 PRs
and 400-char bodies. Start 28,090 + ~1,800 = ~29,900 tokens, 6,000 tokens under
the compaction line. The model gets the list it asked for on the first call.

### Patch 2 (follow-up): repeat-success guard for MCP list calls

Extend `repeat_failure_breaker.py` so an MCP call whose identical input already
**succeeded** this turn is blocked on the second repeat, with "you already have
this result above". Saves the two wasted calls here (7,800 tokens), but alone it
would not have saved the turn: pages 2-4 still overflow. Low frequency (14 extra
identical calls in 14 days), so it waits for Patch 1's eval result.

### Not proposed

- Lowering `ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS` or setting
  `MAX_MCP_OUTPUT_TOKENS`: both cut head-first and leave the model with less of
  the list. See the evidence above.

## Files likely to touch

- `orion/fcc/mcp_stdio_proxy.py`: JSON-aware item-preserving shrink.
- `orion/fcc/tests/test_mcp_stdio_proxy.py` (new): fixture built from the seq 44
  PR objects; 5 PRs at 62k chars -> all 5 numbers present, output <= 12,000;
  non-JSON text keeps today's behavior; a list too big even fully clipped keeps
  whole leading items and says "showing k of N".
- `orion/harness/evals/introspection_scoring.py`: it matches the proxy marker;
  update the pattern for the new wording.
- `orion/harness/evals/` (new small eval): replay of this turn's prompt against
  a stubbed `list_pull_requests` returning the recorded payload; pass = no
  compaction, <= 2 list calls, every PR number in the answer appears in a tool
  result.

## Non-goals

- No change to the compaction line, output reserve, or window.
- No per-tool field allowlists (that is a keyword cathedral in waiting).
- No change to how `context_obs` estimates fill. The 41k -> 120k jump is
  correct behavior.
- No change to the context-mode plugin in this patch.

## Acceptance checks

1. Unit: the recorded 62,280-char page-1 payload goes through the proxy and
   comes out <= 12,000 chars with PR numbers for all 5 items and a marker that
   says "all shown".
2. Unit: a plain-text 50,000-char block still gets today's head cut and marker.
3. Live (after deploy): ask Orion the same question. Pass = the
   `cockpit_turn_sighting` rows for that turn show 0 `compact_boundary` events,
   <= 2 `list_pull_requests` tool_use blocks, and the tool_result text shows all
   requested PR numbers.
4. 14-day re-count after 2 weeks: turns where a proxy-truncated result is the
   last tool before a compaction should drop to 0.

## Recommended next patch

Patch 1: JSON-aware, item-preserving shrink in `orion/fcc/mcp_stdio_proxy.py`,
with the unit tests above and the one-turn replay eval. It is the only change
that gives the model the full list in one call; caps and repeat guards only
limit the damage after the model is already missing data.
