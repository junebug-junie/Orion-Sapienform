# Run now override, cap 6, and Orion's own corpus

## Summary

Follow-up to #1918, which merged while these four commits were still in flight —
so `main` has the Curiosity tab but none of the work below.

- **Run now** override: a turn on demand, skipping the cooldown and daily cap.
- Daily cap `3` → `6`, by Juniper.
- The crystallization sample is Orion's corpus again, not 63% AI Town.
- The concept-induction join resolves.
- Each run can see the thread it is on.
- The atlas page shows what a run actually did.

## The override

`tick(force=True)` skips the **cooldown and the daily cap and nothing else**.
Those two bound cost and stop a redeploy spending a slot; an operator asking for
a run has already made that call. `enabled` is deliberately not overridable — a
loop switched off is a decision already made, and a button that quietly undid it
would make the switch meaningless. Every health gate still applies: the Postgres
role, the graph ACL, the stores, whether there is material at all. Those answer
*can this work*, which nobody overrides by wanting it.

A forced run still counts against today. One that did not would make the daily
counter lie, and that counter is what the atlas page compares against to notice a
run that left no trace.

**The lock came with it.** There was no in-flight guard before and there did not
need to be: a turn runs ~20 minutes and the cooldown put the next one 4 hours
away, so overlap was impossible. A button removes that guarantee — it can land
mid-turn, and two turns would race on the same run-id key and the same FCC
sandbox. The turn is split into `_investigate` so one lock is held across it
rather than merely checked at the top.

`POST /curiosity/api/run-now` plus a confirmed button. The route fires the turn
as a task and returns the acceptance instead of holding an HTTP connection open
for 20 minutes; the task is kept in a module-level set, because asyncio holds
only a weak reference to a running task and the turn could otherwise be collected
mid-run — the button would return ok and nothing would happen.

`curiosity_routes.py`'s docstring said "no POST routes here and there should not
be". That was a claim about writes to Orion's graph stated as a claim about HTTP
verbs. Narrowed rather than quietly broken: nothing here writes Orion's memory,
and the test pins the one write route **by path** so a second cannot appear
without someone justifying it.

## AI Town was 63% of the sample

`formation_policy.DEFAULT_DISCARD_PLATFORMS` discards the platform outright now,
but every AI Town row already in the table predates that gate (all written
2026-07-30/31) and this sampler was the last place still serving them: **185 of
the 295 rows** it could draw from. Orion got twelve cards, about eight of them
character dialogue — *"Steam is inference. Water is fact."* — under a heading
claiming Juniper had approved them. Filtered on a source row in
`aitown_chat_history_log`, AI-Town-only by construction since the #1734 table
split. **295 → 110.**

That heading was false anyway: **21** of the 651 it called approved were
approved; 630 were auto-activated. The counts were also taken over
`status='active'` while the sample came from something narrower, which is how it
announced 651 and drew from 295. Same pool now, and the prompt states how many a
human actually reviewed.

## The induction join was a string format bug

Candidates are stored `crys_6ab0a44c28f2469db2f8dc67be6d4c3f`; crystallization
ids are `6ab0a44c-28f2-469d-…`. Same id, one prefixed and undashed —
`crystallization/repository.py:193` does exactly this conversion. Raw comparison
resolved **0 of 550**, and the comment in the file concluded induction was
"recording judgements about concepts it did not keep". It was not. Normalised:
**235 candidates resolve, 41 decisions have both ends live.** No semantic search,
no traversal, no LLM call.

The AI Town filter then applies to both ends, and that is the point rather than a
detail: **38 of those 41 touch AI Town.** Fixing the join alone would have piped
it into the surface Juniper had just said to keep it out of.

## Continuity was one run deep and pointed inward

The loop read a single `:TurnOutcome` and passed on the note that run left itself
— always some form of "go deeper on X" — so a run could not tell whether X was
new or the fourth consecutive visit. Three runs on one subject is what that
produces. The prompt now shows the last four runs as **subjects**, before the
priors menu: that menu answers "what could I pick", this answers "what have I
been picking". Stated as fact and nothing more, because code choosing Orion's
subject is what this arc deleted.

No `collect()` in that query — with `decode_responses` FalkorDB returns a
collected list as one flat string, and claims contain commas.

## The page

Hop notes, finding evidence and the run's journal prose were all stored and never
rendered. An ISO `written_at` was read as missing, which labelled a run that HAD
written an outcome as having died before writing one, and then let that undated
run mask a genuinely traceless one. The "wrote nothing" banner compared a counter
keyed in `HUB_ENDOGENOUS_OUTREACH_TZ` against a count the browser made in its own
zone — right only while those agree.

## Env/config changes

- `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP`: `3` → `6`.
- `.env_example` updated; local `.env` synced in the same session.
- No keys added, removed or renamed.

The cooldown is now the binding constraint rather than the cap: at a 4h minimum
gap a local day holds exactly six slots, the tick is 300s, and a run starts at the
first tick *after* the cooldown clears — so each run lands a few minutes later
than the last and the sixth can slip past local midnight. Expect 5–6, not a
dependable 6. Lower `HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC` if 6 must be
reliable.

## Tests run

```text
255 passed (curiosity investigation, worldview, study material, atlas,
            atlas template, acl)
node --check services/orion-hub/static/js/app.js  → clean
page <script> --check                             → clean
```

Mutation-tested: letting `force` override `enabled` goes red; removing the
in-flight guard made the lock test **hang** rather than fail, so it is bounded
with `wait_for` — a test that hangs is a test nobody can run.

Live-verified against real Postgres and FalkorDB: clean pool 110 (semantic 87,
stance 20, open_loop 3), 88 resolvable induction decisions after filtering, and
the thread section rendering the three real runs.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-hub up -d
```

Env is read at boot and `orion/` is baked into the image, so a restart alone is
not enough.

## Risks / concerns

- Severity: LOW. The page has still not been seen in a browser — no browser on
  the build host. Layout and paint are unverified by anything but reading.
- Severity: LOW. `POST /api/run-now` is unauthenticated, like the rest of Hub's
  operator routes on the tailnet. It spends a real turn, so it confirms in the
  UI first; the API itself does not.

