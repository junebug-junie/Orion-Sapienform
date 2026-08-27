# Observing the limit — and three corrections to how it was measured

> **Status:** Implementation + measurement. Supersedes the ground-truth numbers in
> `2026-08-27-quota-window-calibration-finding.md`. **The conclusion there survives;
> the evidence for it was wrong in three ways and is corrected here.**

## What was built

`orion/dev_economics/rate_limit_events.py` — reads the constraint instead of predicting it.

The limit announces itself in first-party text at the moment it binds:

```
You've hit your session limit · resets 3:30am (UTC)
You've hit your weekly limit · resets Aug 18, 3am (UTC)
```

`observe()` returns `clear` / `limited` / `unknown`, plus `resets_at`,
`seconds_until_reset`, `event_count` (graded pressure) and `staleness_sec`.

**The anticipation trade turned out not to be required.** The reactive-vs-predictive
loss this was expected to accept does not apply: the message carries the reset time,
so this reports not only that the pool is empty but exactly when it refills — an
authoritative answer rather than a forecast fitted to spend.

## Correction 1 — the detector was measuring itself

The first version matched the substring `rate_limit_error` across the corpus. Run
against the live 5h window it reported **12 events**. Every one was this session's own
tool output discussing rate limits — a detector matching the investigation that
produced it.

Detection is now structural: `isApiErrorMessage == true` **and** a known limit
phrasing. `isApiErrorMessage` alone is insufficient; it also covers 401 auth failures,
"Prompt is too long", and disabled-subscription errors, none of which are scarcity.
The live window now reports **0 events**, correctly.

## Correction 2 — the event count was wrong in both directions

The calibration finding cited **66 events** from substring matching. That number was
simultaneously inflated by self-matches and **missing the actual phrasing entirely** —
it searched for `"usage limit reached"` and `"claude ai usage limit"`, while the real
text is `"You've hit your session limit"`.

Structural detection over the same corpus finds **132 real events**: 122 session
limits, 10 weekly limits.

## Correction 3 — "genuine silence" was never verified

The design spec and the calibration finding both claim the ledger's 18 consecutive
all-zero ticks on 2026-08-26 18:14–22:31 UTC were confirmed as genuine silence by
checking that zero transcript files were modified in that window.

**That check could not have returned anything else.** Those messages live in
long-lived session files still being appended today, so their mtime is *now* and can
never fall inside a past window. The check was structurally incapable of detecting the
activity it was looking for.

Reading message timestamps instead finds **192 messages carrying 5.3M tokens** in the
window that was called silent.

Two clocks, and they are not the same one:

| | |
|---|---|
| message `timestamp` | when it **happened** |
| file mtime | when it was last **written** |

Filtering candidates by `mtime >= window_start` is sound — a file last written before
the window cannot contain a message inside it. The reverse inference is false. Only
the sound direction is used in the new module, and its docstring says why.

**Consequence for the metric gate:** item 4 (live-data sanity) in
`2026-08-27-claude-quota-contested-scarcity-design.md` does **not** pass as written.
The rest state was not verified. Whether the ledger was honest-about-disk (transcripts
flushed late) or genuinely blind is *not determined* — distinguishing them needs
write-time observation not available retroactively. Treat ledger spend figures as
**floors of unknown tightness**, not totals.

## The refutation survives, with better numbers

Recomputed against the 12 distinct hours carrying a real limit event inside ledger
coverage:

| event (UTC) | kind | 5h spend |
|---|---|---:|
| 2026-08-14 06:58 | session | $181.74 |
| 2026-08-14 18:42 | session | $188.51 |
| 2026-08-16 10:20 | **weekly** | $254.39 |
| 2026-08-19 03:54 | session | $208.05 |
| 2026-08-19 04:28 | session | $215.04 |
| 2026-08-20 01:46 | session | $144.37 |
| 2026-08-20 05:38 | session | $154.01 |
| 2026-08-20 17:44 | session | **$85.39** |
| 2026-08-23 04:46 | **weekly** | **$56.40** |
| 2026-08-25 22:51 | session | $152.60 |
| 2026-08-26 05:19 | session | **$289.76** |
| 2026-08-27 00:17 | session | $171.21 |

**$56.40 to $289.76 — a 5.1x spread**, wider than the 3.4x first reported, and the
largest 5h window ever observed ($419.05) still did not trip it. No threshold
separates limited from not-limited. **The dollar denominator is refuted, now on
correct evidence.**

A further reason it could never have worked, visible only once the kinds were
separated: the two **weekly** limits sit at $254 and $56 of trailing 5h spend. A
weekly limit is not about the last five hours at all. The axis was conflating two
different constraints with different periods, so a single 5h threshold was
unfittable by construction.

## Honest limits of the new signal

- **Flush latency is real.** Transcripts are written by live sessions, not
  synchronously. A limit that fired moments ago may not be on disk. `staleness_sec`
  exposes this; callers wanting instantaneous truth should threshold on it.
- **Cost.** ~5–9s for a 5h window over the live corpus (~14 candidate files of a
  1.2GB tree). Fine for an occasional decision, not for a hot loop.
- **`unknown` policy is deliberately not decided here.** For a spend budget the safe
  direction was to refuse. Here `unknown` usually means nobody has used Claude
  recently — which is when the shared pool is *least* contended. That is a caller's
  call, and hard-coding either answer would be wrong for the other.

## Recommended next patch

Wire `observe()` into a decision as a three-state signal with `resets_at`, and leave
the dollar allowance unset and unbuilt. `ORION_QUOTA_ALLOWANCE_USD_PER_WINDOW` should
not be configured.
