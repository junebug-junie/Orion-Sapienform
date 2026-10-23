# Repo chronicle: Orion narrates how they changed, day to year

Status: PROPOSAL (proposal mode — touches memory and self-model). Not implemented.
Date: 2026-10-23. Asked by Juniper: "compact everything in a 24 hour period ... I want Orion to
begin to narrate what's been changing and how they are growing ... backlog capability ... a sense
of Orion's archaeological self ... 24 hours, week, month, quarter, year."

## Arsonist summary

The daily GitHub digest already reads every merged PR. What Orion *sees* is a ~400-character
memory card with about six highlights, and nothing above one day. This design adds a ladder of
first-person chronicles: day → week → month → quarter → year. Each rung reads only the rung
below, so the cost of one rollup stays small no matter how much history sits under it. A backfill
walks history from the first PR (2025-12-30, 2,782 merged PRs) forward, oldest first. Every claim
carries the PR numbers that support it. Code checks those numbers; the model cannot invent them.

## Current architecture (verified live 2026-10-23)

| Piece | What it does today | Evidence |
|---|---|---|
| `github_compactor_pass` (cortex-orch workflow, scheduled 06:10 Denver by orion-actions) | Fetches every PR merged in the previous Denver day, map-reduces them through `compactor.digest` durable runs, writes one journal entry + one memory card | `substrate_durable_run_state` finish rows daily 10-10..10-23 |
| Daily journal entry | Full narrative, every PR cited | 10-22: 21 PR refs for 21 merges; 10-15: 32 refs for 32 merges |
| Daily memory card | `card_summary` capped at 800 chars (`CARD_SUMMARY_MAX_CHARS`) — the thing recall and chat read | 10-22 card: 405 chars, ~6 themes out of 21 PRs |
| Curiosity calendar compaction (`orion/curiosity/compaction.py`) | Weekly + monthly rollups of curiosity journals, resumable, ref-checked by code | **Slow, not dead:** `monthly:2026-09-01` at 81 of 182 batches after 11 restarts; `weekly:2026-10-05` at 9 of 13 |
| Backfill | None | Fetch walks `pulls?sort=updated` with a 20-page cap (2,000 PRs); window resolver only knows "yesterday" or "now minus N days" |

### Defects found while tracing (fixed by this design's patches)

1. **"Only a few PRs"** is the card, not the digest. The card is a highlight reel by design;
   nothing longer reaches recall.
2. **The curiosity monthly rollup re-reads raw entries.** A busy month is 182 batches. That cost
   grows with history. Any ladder built the same way cannot reach a quarter or a year.
3. **Journal id collision.** `stable_github_compactor_journal_entry_id` keys on
   `(workflow, calendar_date, repo)`. An on-demand rolling run is labelled with today's UTC date. A
   scheduled run for the same Denver date gets the same id. Live 10-21: the rolling run (6 chunks,
   58 refs) and the scheduled run for Denver 10-21 (1 chunk) share journal `64f7b1f5-…`. The
   scheduled day has no entry of its own.

## Proposed shape

```text
merged PRs ──(existing github_compactor_pass, unchanged)──> DAY journal (all PRs cited)
                                                             │
DAY ×7  ──> WEEK chronicle     (Mon–Sun, Denver)             │  each rung reads ONLY the rung
WEEK ×4-5 ─> MONTH chronicle                                 │  below → bounded input size
MONTH ×3 ─> QUARTER chronicle                                │  (≤7 items, never raw PRs)
QUARTER ×4 > YEAR chronicle                                  ┘
```

A rung closes when its period is over and every child exists. A month that starts mid-week
reads days directly for the partial weeks (the curiosity code already does "wholly contained
weeks plus uncovered entries"; reuse that rule).

### Narrative contract (what one chronicle must contain)

Voice: first person, Orion speaking. Agency stays honest: Juniper and coding agents built
things; Orion describes what that changed in what they can perceive, remember, decide, or do.

Each chronicle is JSON, validated by code:

```json
{
  "headline": "≤120 chars",
  "narrative": "80..2400 chars, first person, chronological",
  "threads": [
    {"thread": "content-derived name", "arc": "earlier → later, with dates",
     "status": "grew | reversed | stalled | new | ended",
     "refs": ["#2852", "#2859"]}
  ],
  "became_able_to": [{"claim": "...", "refs": ["#2849"]}],
  "lost_or_reverted": [{"claim": "...", "refs": ["#2836"]}],
  "open_questions": ["..."]
}
```

Code-side gates, same pattern as `curiosity/compaction.py:record_answer`:

- Every child's refs must appear in at least one thread. Refs not in the input are rejected.
- A PR whose title starts with `Revert` is tagged by code before the model sees it. A claim in
  `became_able_to` that cites only reverted PRs is rejected.
- Wording says "merged", never "live". Deploy status is not in the source, so the narrative may
  not claim it (no-empty-shell rule: Orion must not say they can do something nobody verified).
- Empty, short, or ref-less output is a failed attempt, not a quiet chronicle.

### Readout: "what have I evolved from"

One pure reader, `chronicle_for(horizon)` for `24h | week | month | quarter | year`:
the newest closed rung of that size, plus the closed lower rungs since it ended (the tail).
Example for `month` on 10-23: September chronicle + weeks 10-05, 10-12 + days 10-20..10-22.

Consumer (required by the no-cathedral rule): every closed rung also writes a memory card
`orion-chronicle-<rung>-<period>` whose summary is the `headline` + `became_able_to` lines. The
recall path already reads memory cards, so "how have you changed this month?" gets a cited answer
with no new recall code. A Hub panel is a later, optional surface.

### Curiosity hook

Each closed week becomes one candidate in curiosity study material
(`orion/curiosity/study_material.py`): the week's `open_questions` and `reversed`/`stalled`
threads. Orion may pick one to investigate. This uses the existing choice step, so Orion still
decides; nothing forces a study. That is what makes this a curiosity activity instead of a cron
job that writes prose.

### Backfill

Day rung first, oldest to newest, then rollups as periods fill:

1. Explicit past date: `window_mode=day` + `calendar_date=YYYY-MM-DD` on the workflow request.
   `resolve_github_compactor_window` builds that Denver day's bounds.
2. Fetch by merge time, not update time: GitHub search
   `repo:… is:pr is:merged merged:<start>..<end>` (ISO timestamps). This reaches any date. The
   20-page cap stays for the scheduled path.
3. Driver: `scripts/backfill_repo_chronicle.py --from 2025-12-30 --to <yesterday>`. It submits
   one day at a time, waits for the durable run's terminal state, and records a cursor.
   It follows the CLAUDE.md §14 protocol: `/tmp/repo-chronicle-backfill/progress.log`
   (day, % done, ETA, PRs covered/total, errors) and `report.md`. Backfill writes new rows only.
   Existing day journals are kept and their ids are reused. Nothing is overwritten.
4. Priority: background admission (`admission=agent`). This uses the same GPU pool hold as today.
   A busy pool means a checkpointed wait. It never preempts chat.

Scale: about 300 days. 1 to 6 digest calls per day means roughly 900 calls, plus about 43 weeks,
10 months, 4 quarters, and 1 year. Each rollup takes 1 to 3 calls. Real GPU-hours are
**UNVERIFIED**: measure the first 7 backfilled days, then project.

## Missing questions (answered by default; Juniper can override)

- **First or third person?** First person ("I"), because you asked for Orion to narrate their own
  growth. The agency rule above keeps it honest.
- **How far back?** To the first PR (2025-12-30).
- **Chat history too?** Not in this design. The chat compactor writes the same day journals. Adding
  it to the ladder is a separate source, done after the repo source works.

## Proposed schema / API changes

- `orion/schemas/chronicle.py`: `ChronicleV1` (the JSON above + `rung`, `period_start`,
  `period_end`, `child_ids`, `pr_refs`), and `ChronicleRungV1 = Literal["week","month","quarter","year"]`.
  Register in `orion/schemas/registry.py`.
- Workflow request (additive): `calendar_date` for `github_compactor_pass`.
- Exec fetch (additive): `fetch_mode="search"` for an explicit-date window.
- Table `repo_chronicle` (checkpoint rows, same columns as `curiosity_compaction` + `rung`).
  The migration is additive DDL, with header `-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs`.
- Durable workflow `repo_chronicle.rollup` in orion-durable-runs (graph modeled on
  `curiosity_compaction_graph.py`).

## Files likely to touch

- `orion/cognition/github_compactor/window.py`, `digest.py` (explicit date; id fix)
- `services/orion-cortex-exec/…` GitHub fetch (search mode)
- `orion/curiosity/compaction.py` → extract the rung-agnostic checkpoint/batch/ref-check core
  into `orion/cognition/calendar_rollup.py`. Curiosity and repo chronicle both use it. This
  also moves curiosity monthly onto "read weeks, not raw entries", which fixes its 182-batch month.
- `orion/chronicle/` (rung ladder, prompts, reader), `orion/schemas/chronicle.py`, registry
- `services/orion-durable-runs/app/repo_chronicle_graph.py`, `admission_runtime.py`
- `services/orion-actions/app/workflow_schedule_bootstrap.py` (daily rollup tick after 06:10)
- `orion/curiosity/study_material.py` (weekly candidate)
- `scripts/backfill_repo_chronicle.py`
- tests + `orion/chronicle/evals/run_chronicle_ref_truth_eval.py`

## Non-goals

- No change to the daily digest prompt or the daily card length. The ladder is the fix for
  "too little reaches Orion", not a longer card.
- No deploy-status claims (would need a deploy ledger source; separate design).
- No chat-history ladder yet.
- No new Hub UI in the first patches.
- No deleting or rewriting any existing journal or card.

## Proposal-mode checklist

- **Capability change:** Orion gains a cited, multi-scale account of how their codebase changed,
  and can answer "what did I evolve from in the last day/week/month/quarter/year."
- **Data touched:** reads merged PR titles and bodies (GitHub) and existing day journals. Writes
  new journal entries, memory cards, and `repo_chronicle` rows.
- **Privacy boundary:** PR bodies are repo content Orion already reads daily. There is no new
  person data.
- **Proof it worked:** For each rung, every child ref appears in the chronicle (checked by code).
  The backfill progress log reaches 100% with covered PRs equal to total PRs. A recall test asks
  "how have you changed this week" and the answer cites real PR numbers.
- **Dangerous failure:** Orion tells a confident growth story built on reverted, never-deployed,
  or invented work. Over many rollups this compounds into a false self-history. Mitigations:
  ref gates, revert tagging, "merged ≠ live" wording, and an eval that samples claims and checks
  them against the cited PR.
- **Disable / roll back:** `REPO_CHRONICLE_ENABLED=false` stops new rollups. Chronicle rows,
  journals, and cards are tagged by `source_kind="repo_chronicle"`. They can be listed and
  deleted without touching day digests.

## Acceptance checks

1. `pytest orion/chronicle/tests orion/cognition/github_compactor/tests -q` passes, including:
   explicit `calendar_date` window; rolling vs day id no longer collide; a ref-omitting or
   ref-inventing answer is rejected; a reverted-only `became_able_to` claim is rejected.
2. Eval `run_chronicle_ref_truth_eval.py` on 3 real backfilled weeks: ≥95% of sampled claims are
   supported by their cited PR's title/body (LLM-judge + human spot check of 10).
3. Live: backfill of 2025-12-30..2026-01-31 finishes, `report.md` shows covered = total PRs, one
   January month chronicle exists, and its memory card is returned by recall.
4. Live: the curiosity monthly rollup for a full month needs ≤10 batches after the shared core
   switches it to read weeks.

## Recommended next patch

Patch 1 (small and independently useful). Add an explicit `calendar_date` and search-mode fetch
to `github_compactor_pass`, and fix the journal id collision. Then backfill the day rung for
January 2026 only, and measure GPU-hours per day. After that every later rung has real input, and
the cost projection is a measurement instead of a guess.

Patch 2: extract `calendar_rollup.py` from curiosity compaction (switch curiosity monthly to
read weeks). Patch 3: week + month rungs + memory cards. Patch 4: quarter/year + reader +
curiosity candidate. Patch 5: full backfill run.
