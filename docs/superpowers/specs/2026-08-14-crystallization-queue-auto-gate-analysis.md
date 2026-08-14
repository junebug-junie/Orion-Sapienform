# Crystallization review queue: what can and cannot be auto-gated

**Date:** 2026-08-14
**Question asked:** "is there any correlation on what gets approve/rejected by me
automatically, and are there auto gates we can apply so I don't have to
approve/reject everything by hand?"

**Short answer:** one auto-gate is available and was shipped (source platform,
which removes 96% of the queue). A second, content-based gate is *visible in the
data but not yet buildable* — the sample is 16 decisions. Every numeric score the
pipeline already attaches is useless for this, and one of them is provably dead.

---

## 1. What the queue actually contained

Live `memory_crystallizations` on 2026-08-14, before any change:

| status | count |
|---|---|
| proposed (the review queue) | 621 |
| active | 588 |
| rejected | 10 |

Every one of the 621 was `kind=stance`, `tags=[consolidation_window]`,
`dominant_shift=STANCE`, `formation_policy=governor_queue`.

Origin, resolved by joining `provenance.memory_window_id` →
`memory_consolidation_windows.turn_correlation_ids` → `chat_history_log.correlation_id`
→ `client_meta.external_room.platform`:

| window composition | count |
|---|---|
| every turn from ai-town | 599 |
| mixes ai-town turns with real turns | 11 |
| every turn a real conversation | 11 |

**96.5% of the queue was NPC dialogue.** Repo-wide the ratio is similar: 1577 of
1697 `chat_history_log` rows (93%) are ai-town.

### The mixed windows are a separate bug

`services/orion-memory-consolidation/app/window_state.py:17` — `_get_open_window()`
selects *one global open window*, with no partitioning by session or platform.
ai-town turns and Juniper's turns append to the same window. **26 windows across
the full history are mixed.** This is not merely queue noise; NPC dialogue is
landing inside the memory windows built from real conversations.

Not fixed in this patch (it changes window lifecycle semantics and wants its own
proposal). The gate is written to *tolerate* it: a window only counts as external
when every turn agrees, so mixed windows still reach the human queue rather than
being auto-activated with Juniper's words inside them.

**Known coverage cost of tolerating it**, surfaced in review: `close_current_window()`
(`window_state.py:113`) seeds the *next* window with the turn that closed the
previous one — `next_turns = [closing_turn]`. If that closing turn is a direct
conversation with Juniper, the following window is permanently "mixed" no matter
how many pure ai-town turns land in it, so a burst of NPC dialogue immediately
after Juniper speaks still reaches the queue. This fails in the safe direction,
but it is indistinguishable from "gate working as designed" in the numbers, so
it is recorded here rather than left to be rediscovered. It disappears when the
window cursor is partitioned properly.

---

## 2. Why no existing score can gate anything

All 16 human decisions ever recorded (10 reject, 6 approve — the 588 `active` rows
were never reviewed by a human; they auto-activated on the `TOPIC`→`semantic`
path) carry **identical metadata**:

```
kind=stance  confidence=certain  salience=1.0  dominant_shift=STANCE
gate_reasons=["repair_signal"]  window_novelty_max≈1.0
```

Turn count and subject length overlap completely between the two classes
(approve: 2,2,2,3,3,3 turns / 17–500 chars; reject: 2,2,2,2,2,3,3,3 turns /
12–500 chars).

### salience is not weak — it is mathematically pinned

From `orion/memory/crystallization/salience.py:85`, for any stance crystallization
from the consolidation-window path:

| term | value | why it is constant |
|---|---|---|
| `KIND_BASE["stance"]` | 0.85 | kind is always stance on this path |
| evidence_boost | 0.075 | `min(0.15, 0.75 × 0.1)`; strength is the hardcoded 0.75 at `intake_consolidation_window.py:112` |
| planning_boost | 0.05 | stance always gets planning_effects |
| confidence_boost | 0.05 or 0.10 | "likely" or "certain" |

Total: **1.025 – 1.075, clamped to 1.0.** It cannot return any other value here.
All 621 proposed rows read exactly 1.0. Confirmed against live data, not inferred.

`window_significance_max` is saturated rather than pinned: 490 of 621 sit at
exactly 1.00, the rest span 0.74–0.99. It carries a little information but not
enough to separate the classes (an approved item sits at 0.74 and another at 1.00).

### Related finding: no summarization happens on this path

`_window_summary()` (`intake_consolidation_window.py:77`) returns **the last
non-empty turn's raw prompt, truncated to 500 chars**, and it is assigned to both
`subject` and `summary`. `subject == summary` for all 621 rows. `planning_effects`
is the template string `"Carry forward this stance when planning: <that same raw
prompt>"`.

So a "crystallization" on this path is a verbatim copy of one user utterance with
a label attached. Per CLAUDE.md §0A this is empty-shell cognition — schema-valid,
cognition-shaped, no cognitive substance. Flagged, not fixed here.

---

## 3. The gate that was shipped

Source platform is the only honest discriminator in the data, and it was already
recorded — just never propagated past `chat_history_log.client_meta`.

```
chat.history (client_meta.external_room.platform)
  → sql-writer: MemoryTurnPersistedV1.source_platform
    → window_state: per-turn source_platform in the window row
      → intake: provenance.source_platform, only if ALL turns agree
        → formation_policy: unanimous external platform ⇒ AUTO_ACTIVATE
```

Configured by `MEMORY_FORMATION_AUTO_ACTIVATE_PLATFORMS` (default `aitown`; empty
disables). Orion still forms, projects and recalls ai-town memory — Juniper is
just never asked to approve it a turn at a time.

Ordering inside `resolve_formation_policy` is deliberate: the platform check sits
**below** the intimate-sensitivity and identity-scope checks (privacy and
self-model boundaries outrank a convenience gate) and **above** the
`GATED_KINDS` check (bypassing the stance gate is the point).

**Measured on live data** via `scripts/smoke_aitown_crystallization_gate.py`,
which replays the real queue through the real policy functions:

```
live proposed crystallizations: 621
  would AUTO-ACTIVATE: 599
  would STAY QUEUED:    22   (all confirmed to contain ≥1 real turn)
```

22, not 11, because unanimity is required: the 11 mixed windows stay queued.
Zero of the 22 were artifacts of a pruned chat row.

---

## 4. The gate that was NOT shipped, and why

Content separates the 16 decisions cleanly, and the split is legible:

| | pattern | examples |
|---|---|---|
| **Rejected** | imperatives, test artifacts, dev-infra narration | "Run github compactor" · "Compact the last 24 hours of chat into a memory digest" · "you want to take a crack? Please add a worktree before you modify" · "burst test 1" · "just sending through a test turn" · "Added a trust entry to the host's ~/.claude.json" · "blerg. Applied another fix to your permissions" |
| **Approved** | relational, biographical, affective | "juneipurs daghter" · "congrats, Orion! We did it!!!" · "Just curious about your take on me, yourself, the world?" · "Gosh such wonderful questions. Love your curiosity" · "Good news... Found a disk repair place" |

One counterexample: *"Thanks for your kinds words, Orion. I'll try to save up a
few hundred"* was rejected despite being relational — plausibly hand-deduplicated
against the approved "Good news... disk repair place" item covering the same
episode.

**This was not built.** Per the CLAUDE.md metric-quality gate:

1. *Provenance* — would be a new classifier over `subject`, not an existing signal. OK.
2. *Independence* — genuinely independent of every existing score (all of which are dead here). OK.
3. **Theory anchor — fails.** "Commands aren't stances, feelings are" is a plausible story, not a named theory. There is no principled reason the boundary sits where these 16 points suggest rather than somewhere else.
4. **Live-data sanity — fails on sample size.** n=16, one known counterexample, all from two sittings ~16 days apart. Any threshold fitted here is fitted to noise.

Building a detector on this now would be a keyword cathedral with a confusion
matrix attached.

**Revisit when:** the queue has accumulated 40–60 hand-reviewed *real*
conversations. That is now realistic — the queue is 22 instead of 621, so every
future decision is signal rather than NPC-clearing reflex. The
`memory_crystallization_history` table already records each one with actor and
timestamp; the bulk purge deliberately used actor `bulk_aitown_purge` so machine
clearing can never be mistaken for a human judgment in that future analysis.

### Two cheaper things worth considering before a classifier

- **Test-turn suppression at the source.** "burst test 1" / "just sending through
  a test turn" are load-test artifacts, not conversation. Cleaner to tag them at
  the chat layer than to teach a classifier to recognise them.
- **Fix `_window_summary` first.** A content gate reading a verbatim raw prompt is
  classifying the user's last sentence, not what Orion concluded. Real
  summarization would change what any such classifier is even looking at, so
  building the classifier first means building it twice.

---

## 5. Open items

| item | status |
|---|---|
| Global open-window cursor mixes platforms (26 windows) | **open** — needs a proposal; gate tolerates it for now |
| `_window_summary` returns a raw prompt as both subject and summary | **open** — empty-shell cognition on the stance path |
| `salience` pinned at 1.0 for the whole stance path | **open** — dead instrument, still displayed in the UI |
| Content-based auto-gate | **deferred** — n=16, revisit at n≈50 |
| ai-town source gate | **shipped** |
| Backlog purge (599 rows) | **applied**, reversible |
