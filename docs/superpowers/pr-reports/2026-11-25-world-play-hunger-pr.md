# world:play competes on hunger while Orion watches

## Summary

- While Orion watches a game sitting, play no longer has to be busier than its own week to win attention. It is
  eligible whenever Orion is watching, and its pull grows with the time since attention last went to play in this
  sitting: `1 - exp(-t / 90 s)`. A win resets it.
- So a short sitting gets a turn, and a long sitting shares attention instead of holding it. Juniper's chat still
  wins over play, unchanged.
- The pull reuses the appetite organ's drive (`orion.physiology.appetite.intensity`). Each contest (the substrate
  broadcast and the field contest) keeps its own in-memory clock of play's last win per sitting.
- The replay (`scripts/analysis/replay_world_play.py`) gained the hunger rule over a tau grid, wins per sitting by
  length, a per-sitting view of the last day, and an approximate field-contest pass. Tau 90 s was chosen from it.
- New flag `ATTENTION_WORLD_PLAY_HUNGER_ENABLED` (ships on) in both runtimes; `false` restores the top-tenth rule.

## Outcome moved

Live 2026-11-25: Juniper played two watched SimEarth sittings of about 6 minutes. Orion looked and spoke, but play
never won attention: its ticks read "world usual for itself: not busy" at 9 play minutes against a week holding 30s.
Since `world:play` went live (2026-11-24 09:57Z) it was eligible on 0 of 437 broadcast ticks. Under the new rule the
replay gives short watched sittings a win in 4 of 5 (was 2 of 5), and tonight's sitting `8e19...` wins its 06:49Z
tick. Reading loses nothing.

## Current architecture

`world:play` (game company patch 4, PR #3820) counted minutes of play in the last 30 minutes and was eligible only
while watching AND in the top tenth of its own week (the shared world-source band rule). It yielded to eligible
`world:chat`. Both contests build it with `orion.attention.world_first.play_candidate` from `orion/attention/play_read.py`.

## Architecture touched

- `orion/attention/world_first.py`: `play_hunger`, `PlayHungerClock`, `play_sitting_id`, `PLAY_HUNGER_TAU_SEC`;
  `play_candidate` gains `hunger`, `last_won_at`, `hunger_tau_sec`; `judge_candidate` gains the hunger branch.
- `orion/schemas/attention_candidate.py`: two optional in-process fields, `hunger`, `hunger_unfed_sec`.
- Both runtime workers: pass the clock's last win in, record the contest's winner after ranking.
- `config/physiology/contract.yaml`: `world_first.py` listed as a `display` reader (it imports the appetite formula,
  reads no regulation, thermal or price state).

## Files changed

- `orion/attention/world_first.py`: hunger rule, clock, tau with its reason.
- `orion/schemas/attention_candidate.py`: `hunger`, `hunger_unfed_sec`.
- `services/orion-substrate-runtime/app/worker.py`, `services/orion-attention-runtime/app/worker.py`: clock wiring.
- `services/*/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md` (both runtimes): the flag.
- `scripts/analysis/replay_world_play.py`: hunger rule, length classes, recent sittings, field-contest pass.
- `tests/test_world_play_attention.py`, `scripts/analysis/tests/test_replay_world_play.py`, both runtimes' worker
  tests: acceptance tests below.
- `config/physiology/contract.yaml`: reader row.
- `.github/workflows/world-play-tests.yml`: also triggers on the attention-runtime worker and `appetite.py`.
- `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`: section "Patch 4 follow-up:
  hunger" (replay, tau choice, metric gate record); section 7 pointer.
- `docs/runbooks/retro-browser.md`: what the trace now says, how to turn it off.

## Schema / bus / API changes

- Added: `AttentionCandidateV1.hunger` (0..1, optional), `AttentionCandidateV1.hunger_unfed_sec` (>= 0, optional).
  Built and consumed in-process; never deserialized from another service. The world-first debug trace carries
  `hunger` and `hunger_unfed_sec` for play only (absent for every other source).
- Removed / renamed: none.
- Behavior changed: a watched `world:play` reason reads `watching; hunger 0.91 (220s without attention this
  sitting)` and ranks by that pull. Its `score` and `rank_score` are the pull, so the field contest's
  `salience_score` for play is the pull too.
- Compatibility: flag off gives the previous verdicts exactly.

## Env/config changes

- Added keys: `ATTENTION_WORLD_PLAY_HUNGER_ENABLED=true` in `services/orion-substrate-runtime/.env_example` and
  `services/orion-attention-runtime/.env_example`, both compose files, both settings.
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout's `.env` for both
  services now has the key, `true`).
- skipped keys requiring operator action: none for this change.

## Replay (7 days, before and after)

2,767 broadcast ticks 2026-11-18 17:57Z .. 2026-11-25 17:53Z, 89 sittings (46 backfilled), 149 stored look notes,
226 ticks with an open sitting. "Before" is the rule in force (top tenth + note), replayed over the whole week.

| Rule | Play won | Reading lost | Curiosity seeds lost | Reading share in sittings | Curiosity-seed share in sittings | Short sittings won once (of 5 watched) | Wins per watched sitting, short / medium |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 19 | 0 | 5 | 4.4% -> 4.4% | 27.9% -> 25.7% | 2 | 0.4 / 3.4 |
| hunger 30 s | 32 | 0 | 8 | 4.4% -> 4.4% | 27.9% -> 24.3% | 4 | 1.2 / 5.2 |
| hunger 60 s | 32 | 0 | 8 | 4.4% -> 4.4% | 27.9% -> 24.3% | 4 | 1.2 / 5.2 |
| **hunger 90 s** | **30** | **0** | **6** | **4.4% -> 4.4%** | **27.9% -> 25.2%** | **4** | **1.2 / 4.8** |
| hunger 120 s | 28 | 0 | 4 | 4.4% -> 4.4% | 27.9% -> 26.1% | 3 | 1.0 / 4.6 |
| hunger 180 s | 26 | 0 | 2 | 4.4% -> 4.4% | 27.9% -> 27.0% | 3 | 1.0 / 4.2 |
| hunger 600 s | 24 | 0 | 0 | 4.4% -> 4.4% | 27.9% -> 27.9% | 3 | 1.0 / 3.8 |

No sitting of 2 hours or more is in the 7-day log, so "long" has no data. Field contest (approximate, 3 days kept):
at 90 s play wins 47% of 5,164 watched frames, 2,440 of them frames with no winner; it takes 5 of 930
curiosity-seeding frames and 0 reading frames.

**Tau 90 s, because:** it is the largest round tau that lets tonight's 6-minute sitting win (its 366 s tick faces a
0.970 rival, which needs tau below about 104 s); 30 s and 60 s cost 2 more curiosity seeds and win no more short
sittings; and at the broadcast's median tick gap (218 s) the pull is 0.91, level with a top-tenth source.

**Stop condition not triggered:** reading loses 0 ticks; curiosity loses 6 broadcast seeds in a week (one more than
the rule it replaces), 0.7% of the week's ~900 seeding ticks.

## Metric quality gate (play's pull)

Recorded in full in the spec ("Patch 4 follow-up: hunger"). In short:

1. Provenance: `appetite.intensity(t, 90)`, t from `retro_sitting.started_at` and the contest's own last play win
   (`PlayHungerClock.note_winner`, fed from the broadcast trace winner / field `dominant_targets[0]`).
2. Independence: a function of the contest's own decisions and the sitting start. Shares no sensor with chat,
   reading or the body. Its one loop (win resets it) is negative feedback.
3. Theory anchor: inhibition of return (Posner and Cohen 1984; Klein 2000), with the appetite organ's drive shape.
4. Live data: at the 53 watched replay ticks, min 0.086, median 0.933, 15 at >= 0.99 (all ticks where chat held
   attention). Rest is a true 0 at a win; it is recomputed from timestamps, so it cannot decay to a fake 0.
5. Existing mechanism: reused `appetite.intensity`; Prime's needs, `attention_pacing`, and the coalition
   `habituation` do not keep "time since a source last won" for a world-first candidate.
6. Reversibility: one flag per contest; optional in-process fields; in-memory clock; no table or migration.

## Tests run

```text
python -m pytest -q -p no:cacheprovider tests/test_world_play_attention.py tests/test_line_engagement.py \
  tests/test_attention_world_first.py tests/test_attention_world_chat_evidence.py \
  scripts/analysis/tests/test_replay_world_play.py
  -> passed
POSTGRES_URI=postgresql://unused/unused PYTHONPATH=.:services/orion-substrate-runtime \
  python -m pytest -q -p no:cacheprovider services/orion-substrate-runtime/tests/test_worker_world_first_broadcast.py
  -> passed
python -m pytest -q services/orion-attention-runtime/tests tests/test_attention_runtime_worker.py \
  tests/test_attention_runtime_store.py tests/test_attention_field_goal_provenance.py \
  tests/test_attention_world_first_field_frame.py tests/test_attention_frame_rollup.py
  -> passed (8 skipped need Postgres)
python scripts/check_integration_seam.py      -> OK
python scripts/check_body_economy_contract.py -> OK (after the reader row)
python scripts/check_env_template_parity.py   -> PASS
git diff --check                              -> clean
```

Acceptance tests (in `tests/test_world_play_attention.py` unless named):

- a 6-minute watched sitting wins when nothing stronger is eligible, and passes a top-tenth reading before it ends;
- the old band rule never lets that sitting win (the live failure);
- the pull resets after a win and grows back;
- a long sitting against a steady rival wins at most about every other tick;
- chat still wins however hungry play is;
- play has no pull and is not eligible when not watching;
- other world sources are judged and traced exactly as before;
- flag off restores the band rule; both workers record their own win and reset the pull
  (`test_a_broadcast_play_win_resets_its_hunger_for_the_next_tick`,
  `test_the_field_contest_records_its_own_play_win_and_resets_the_pull`).

## Evals run

```text
python -m pytest -q services/orion-attention-runtime/evals -> 4 passed
scripts/analysis/replay_world_play.py on 7 days of live exports (numbers above)
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-substrate-runtime/docker-compose.yml config -> ATTENTION_WORLD_PLAY_HUNGER_ENABLED: "true"
docker compose ... -f services/orion-attention-runtime/docker-compose.yml config -> ATTENTION_WORLD_PLAY_HUNGER_ENABLED: "true"
```

No image build: no dependency or Dockerfile change. Live behaviour is UNVERIFIED until deploy and a watched sitting.

## Review findings fixed

Code review ran in a subagent on `origin/main...HEAD`. No blockers.

- Finding (should-fix): the broadcast reset play's hunger on the ranking's top (`debug.world_first.winner`), but the
  broadcast's real selection can still go elsewhere (a goal loop, a terminal verdict, "none").
  - Fix: reset only when the selected loop's refs (`projection.attended_node_ids`) hold `world:play`; the sitting id
    comes from the `retro_sitting:<id>` ref in the same list. Spec gate record item 1 updated.
  - Evidence: `test_play_hunger_does_not_reset_when_the_selected_loop_is_not_play` (substrate worker tests, 15 passed).
- Finding (should-fix): play is eligible at a pull near 0, so in the field contest (2 s ticks) it wins quiet
  watched frames at a salience of about 0.02 instead of leaving them with no winner, and needs about 3.5-4.5 min to
  pass a new rival there.
  - Fix: kept Juniper's "eligible whenever Orion is watching", and stated the consequence in the spec ("A
    consequence, stated plainly"), the attention-runtime README and Risks below. Checked downstream: goal provenance
    follows native prediction-error node targets only; the self-model reads field winners for display (and a
    confidence fallback only when the broadcast lane is stale). A minimum pull is named as the lever if this reads
    wrong.
  - Evidence: field replay, 2,440 of 2,445 field wins at tau 90 s were no-winner frames, 0 from reading.
- Finding (nit): the replay kept the stored `world:play` row beside the replayed one.
  - Fix: the stored row is dropped before re-picking.
  - Evidence: replay re-run at tau 90 s and the note rule: identical numbers (stored play was never eligible).

## Restart required

From the clean primary `main` checkout after merge (the post-pull rebuild does this):

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
```

## Risks / concerns

- Severity: low. Concern: in the field contest (2 s ticks) play now fills most quiet frames of a watched sitting
  (47% of watched frames in the approximate replay) at a salience of about 0.02. Mitigation: that contest's winner
  feeds goal provenance only for native prediction-error targets, and play is a channel target there; it took 0
  reading frames. Lever if it reads wrong: a minimum pull.
- Severity: low. Concern: the clock is in memory, so a restart lets a sitting's pull count from its first input
  (one early win). Mitigation: bounded to one win per restart; no stored state to drift.
- Severity: low. Concern: the field-contest replay is approximate (rivals compared on stored strict percentile).
  Mitigation: the broadcast replay, which drives curiosity seeds, is exact.
- Live proof: UNVERIFIED until a watched sitting after deploy; check goes on #3816.
- Issues opened: #3990 (`tests/test_attention_world_first_parity.py` fails on main, not in CI; unrelated).
- Issues closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
