# Correct the human-message denominator for affect/economics signals

## Summary

- Three silent counting bugs in `orion/dev_economics/claude_code_ingest.py`, the shared
  parser that decides which local Claude Code transcript turns count as "words Juniper
  typed." None crashed, logged, or violated a schema — the counts were just wrong.
- **`/compact` continuation summaries were counted as Juniper's prose.** Model-authored
  text injected as `type=user` with no `promptSource`, so the "absent means typed"
  default admitted all 109. **30.7% of the denominator, 45.6% of the numerator.**
- **`promptSource: "queued"` was dropped as synthetic** — a message typed *while a turn
  was running*, held and submitted when the agent came free. 75 real messages carrying
  **9 real swear words**.
- **Harness output was scored as prose** — `<local-command-stdout>`, `<bash-input>`,
  `<bash-stdout>`, plus `<bash-stderr>` and `<tool_use_error>`: 128 messages, 2,112 words.
- Deliberately **not** a blanket tag strip — that would delete 515 real words.

## Outcome moved

`swear_frequency` on `orion:substrate:juniper_affective_state` drops **−21.4%**
(0.000591 → 0.000464) because nearly half the swearing it had ever measured was a model
quoting Juniper back at itself.

At the signal's real 15-minute granularity:

- **34 of 219 non-calm windows (16%) were fabricated** by machine-authored text — the
  actual Juniper messages in those windows contained zero swears.
- **5 windows were 100% summary text with no typed Juniper words at all** — a published
  affect score for a window in which Juniper said nothing. That is the empty-shell
  failure root `CLAUDE.md` §0A names, live.
- **3 windows become non-calm** that were reading calm — real frustration that was
  invisible.

| | messages | words | swears | rate |
|---|---|---|---|---|
| before | 4,117 | 1,201,647 | 704 | 0.000586 |
| after | 3,956 | 835,488 | 388 | **0.000464** |
| delta | −161 | −366,159 | −316 | **−21.4%** |

The compact-summary term dominates: it alone is 176x the harness-output leak that
prompted this patch. Both smaller fixes still matter — they move which *windows* read
non-calm, which is what a consumer would act on — but the headline number is the
summaries.

## How these were found

Not by reading code. Juniper's read of the live logs — 22 of 23 ticks publishing
`message_count=0` — prompted a direct audit of whether the windows were losing real
activity.

**The windows turned out to be exactly correct.** All 38 of the day's published ticks
were reconstructed from the container log and recomputed offline against the raw
transcripts: 57 logged messages, 57 recomputed, **zero missing**, word counts identical
per window. The 8-hour gap was genuinely idle — every transcript line of every type in
that span was a single `/model sonnet` command, and `dev_economics_ledger_log`
independently reads `session_count=0, total_tokens=0` across the same period.

The denominator bugs surfaced only from cross-tabulating `promptSource` against content
type over the whole corpus while confirming that.

## Current architecture

`~/.claude/projects/**/*.jsonl` → `iter_all_human_messages()` → `score_message()` →
`aggregate_scores()` → `JuniperAffectiveStateV1` → `orion:substrate:juniper_affective_state`
(still `consumer_services: []`, so still dropped on publish — a separate, open problem).

The same parser also feeds `parse_session_usage_record()` →
`DevEconomicsLedgerV1` → `dev_economics_ledger_log` (live, 128 rows).

## Architecture touched

Only the parser and its tests. No schema, channel, service, env, or contract change.

## Files changed

- `orion/dev_economics/claude_code_ingest.py`: `isCompactSummary` rejected outright;
  `_HUMAN_PROMPT_SOURCES` constant replaces the bare `!= "typed"` check; 5 tags added to
  `_WRAPPER_TAG_PATTERN`; module docstring filtering rules updated with the measured
  evidence.
- `orion/dev_economics/tests/test_claude_code_ingest.py`: 12 new tests.

## The boundary this fix must not cross

`<name>`, `<question>`, `<service>`, `<path>` appear inside admitted messages 60/38/20/17
times — those are **Juniper's own placeholder prose** in instructions to an agent, not
harness wrappers. A generic `<(\w+)>...</\1>` strip would delete **515 real words across
6 messages**, 373 of them from a single `<section>`-wrapped instruction.

What separates the two empirically: a harness wrapper *is* the entire message and
therefore begins it; the placeholders appear mid-sentence. Only tags observed in the
leading position are on the strip list — `bash-stderr` being the one deliberate
exception, justified above.

## Schema / bus / API changes

None. `JuniperAffectiveStateV1` and `DevEconomicsLedgerV1` are untouched.

## Env/config changes

None. No `.env_example` change, so no sync required.

## Tests run

```text
.venv/bin/python -m pytest orion/dev_economics/tests/ -q
60 passed
```

All 12 new tests mutation-verified against the specific wrong implementation each exists
to catch — not assumed:

```text
unfixed parser                          -> 6 tests fail
admit every promptSource                -> test_system_and_sdk_prompt_sources_stay_excluded
strip every tag                         -> test_placeholder_angle_brackets_in_juniper_prose_survive
drop the isCompactSummary check         -> test_compact_summary_is_not_juniper_prose
                                           test_compact_summary_does_not_recount_quoted_swears
greedy .* instead of .*?                -> test_wrapper_strip_is_non_greedy_across_repeated_tags
drop re.DOTALL                          -> test_wrapper_strip_spans_newlines
```

Each mutation is caught by exactly the test written for it and by no other.

**One unfailable fixture was caught this way and rewritten.** The placeholder guard
originally used *unclosed* tags (`<service>`, `<path>`); the over-broad regex it existed
to catch requires a closing tag, so it passed against the very mutation it was written
for. Rewritten with a closed `<section>...</section>` pair — which is also the real-world
shape that actually loses words. Found by running the mutation, not by re-reading the
test.

Pre-existing and unrelated: `orion/cocreation/tests/test_affective_signals.py` has 4
failures on clean `main` (`ModuleNotFoundError: spellchecker` — `pyspellchecker` is in the
service image and `requirements-dev.txt` but not the repo venv). All 4 are `typo_rate`
tests; `typo_rate` is not on any wire schema.

## Evals run

No eval harness for this parser.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-cocreation-signals build   -> Image built
scripts/safe_docker_build.sh orion-cocreation-signals up -d   -> Started
StartedAt=2026-08-13T18:08:49Z Running=true RestartCount=0
```

First live tick under the fully fixed parser:

```text
18:09:17 cocreation_affective_state_published message_count=24 word_count=6782
         swear_count=10 swear_frequency=0.001474491300501327
```

Recomputing that exact window offline reproduces the published numbers to the digit
(`24 / 6782 / 10 / 0.001474`), confirming the deployed image is running this code.

**Stated plainly: this tick does not demonstrate the fix.** Recomputing the same window
with the *old* parser produces identical numbers, because this particular hour happens to
contain no compact summary, no queued message, and no harness output. The live tick
proves the deployment is correct; the effect of the fix is evidenced by the 31-day corpus
replay above, not by this line.

## Review findings fixed

- **Finding (critical): the dominant contaminant was missed entirely.** `/compact`
  continuation summaries — model-authored text injected as `type=user` with no
  `promptSource` — were admitted by the "absent means typed" default. 109 messages
  carrying **30.7% of the denominator and 45.6% of the numerator**, 176x larger than the
  harness-output leak this patch was originally written to fix.
  - Fix: `if obj.get("isCompactSummary") is True: return False` — one field check on the
    same JSON object the filter already reads.
  - Evidence: independently reproduced before accepting. All 109 lines carry
    `isCompactSummary: True, isVisibleInTranscriptOnly: True`; excluding them moves
    `swear_frequency` 0.000591 → 0.000464 (−21.4%). At 15-minute granularity, **34 of 219
    non-calm windows (16%) were fabricated by machine text** whose real Juniper messages
    had zero swears, and 5 windows were 100% summary with no typed words at all.
- **Finding: "6475 words of real Juniper prose" was inflated.** 76.4% of that total is a
  single message of ~5 typed words followed by a pasted Postgres log; the median queued
  message is 7 words.
  - Fix: commit message and this report now rest the claim on the **9 recovered swear
    words**, not the word volume.
  - Evidence: reproduced — top-1 message 4,949 words / 0 swears, top-5 share 89.5%.
    (The reviewer read all 75 payloads and confirmed the *classification* is correct;
    only the framing was wrong.)
- **Finding: the persisted ledger discontinuity was undisclosed in the commit.**
  - Fix: now stated in the commit message and in Risks below.
  - Evidence: 50 of 1216 session records change; `total_human_word_count` +0.77%,
    `total_human_turn_count` −1.7%.
- **Finding: two regex properties had zero test coverage.** Mutating `.*?` → `.*`
  (greedy) and dropping `re.DOTALL` each passed all 34 tests. The DOTALL case is severe:
  every real `<bash-stdout>` body is multi-line, so that mutation would silently no-op the
  whole fix on live data while passing every single-line test.
  - Fix: `test_wrapper_strip_is_non_greedy_across_repeated_tags` and
    `test_wrapper_strip_spans_newlines`.
  - Evidence: both mutations now caught, verified by re-running them.
- **Finding: `<tool_use_error>` still leaked** (1 message, 13 words).
  - Fix: added to the strip list.
- **Finding: the commit undercounted its own test strength** — 6 tests fail against the
  unfixed parser, not 5. Corrected.

**Checked and confirmed clean by the reviewer**, each by running code against the live
corpus rather than reading the diff: all 75 `queued` payloads are genuinely Juniper; all
5 `sdk` and all 705 `system` payloads are correctly excluded; no `promptSource` value was
missed; zero tags remain in leading position across all 4,065 admitted messages; no
`queued` message carries a non-human `origin.kind`; **no second unfailable fixture
exists**; and the regex has no backtracking pathology (29 ms across 12 MB / 5,345
messages, linear on a 200k-char unclosed tag).

**Known residual, accepted:** one message with 3 *unclosed* `<system-reminder>` tags
leaks entirely (3,077 words). Only 2 unbalanced messages exist corpus-wide. Fixing this
needs an unclosed-tag rule, which risks the placeholder-prose boundary above for 0.25% of
the denominator — not worth the trade today.

## Restart required

Already deployed as part of this patch.

## Risks / concerns

- Severity: medium
- Concern: `human_word_count`/`human_turn_count` in `dev_economics_ledger_log` (127 live
  rows) derive from this same parser, so rows written before and after this deploy are
  not strictly comparable on those two columns. Token and cost columns are unaffected.
- Mitigation: 50 of 1216 session records change (`total_human_word_count` +0.77%,
  `total_human_turn_count` -1.7%). The post-fix numbers are the correct ones. Recorded
  here rather than silently absorbed.

---

- Severity: low
- Concern: the strip list is enumerated from tags observed in the real corpus as of
  2026-08-13. A future harness version emitting a new wrapper tag will silently inflate
  the denominator again, exactly as these three did.
- Mitigation: none built. A periodic re-run of the leading-tag audit would catch it; that
  is a real, disclosed maintenance cost of parsing an unversioned append-only format.

---

- Severity: low
- Concern: this improves a signal whose events are still dropped on publish
  (`consumer_services: []` on `orion:substrate:juniper_affective_state`). The corrected
  numbers are not being kept anywhere.
- Mitigation: out of scope here; that is the persistence slice, tracked separately.

## PR link

<!-- filled in -->
