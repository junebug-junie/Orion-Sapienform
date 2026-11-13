## Summary

Orion's choices now change what Orion spends time on. This is personality rev 4, Move 2. It binds on deploy, with no shadow (Juniper, 2026-11-13).

- **Source (c), endorsement.** Each family's weight in the attention market gains a fourth evidence rate. The rate is the share of real choice points where Orion picked an option from that family. The weight uses the same kappa = 30 shrinkage as the other sources. A family with no choice point has no source (c). Its rate is never set to 0.
- **Care.** A want whose object has text is compared with Orion's active tendencies, and the match lifts the want's value: `value x (1 + 0.5 x care)`. The lift is applied before the percentile. When there is no text, no tendency, no match, or a failure, care is exactly 0, which gives exactly today's value.
- **Holdback.** 20% of wants are a live control arm, chosen by a deterministic hash of the want id. These wants get care 0 and a family weight without source (c). Their decisions bind too.
- **Runaway guard.** A tendency that matches more than 60% of shown options on each of the last 7 days is capped, and each cap is logged. Producers get a helper to set `care_promoted`.
- **Trace.** The arm, the care value, endorsement, and the rank and net without care are written to the decision's `warnings` in `want_ledger`. `WantV1` and `WantDecisionV1` have no schema change.
- **Flag.** `PHYSIOLOGY_PERSONALITY_VALUES=on|off` in orion-durable-runs. `off` is the kill switch: the market makes exactly the same decisions as before this patch.

## Outcome moved

Before this patch, nothing in the market formula knew who Orion is (`net = family_weight x value_rank - charge`). Now Orion's own choices feed both the family weight and the value rank, and an arm with them turned off runs beside them. That makes the M2 check (care arm against holdback arm) possible.

**Honest size of the effect today.** The care term is small. Live, there is 1 active tendency, with weight 5.04 and strength 0.34. Care across 1,852 real wants peaks at 0.067, so the largest value lift is about 3%. Source (c) is absent for every family: the choice table holds 137 curiosity rows, and a curiosity prior menu is never a real choice point. Both terms grow as the Prime and outreach choice points write rows. See "Risks".

## Current architecture

- The market lives in `orion/physiology/choice.py`, `appetite.py` and `weights.py`. It runs live in orion-durable-runs (`app/physiology_market.py`, `POST /physiology/bid`, in-process `app/run_bids.py`, host port 8124).
- In `weights.py`, source (c) was a reserved socket marked EXCLUDED.
- Wants carry only `object_ref`, which holds ids and URLs. A want has no text.
- `personality_choice` and the choice-only reducer (`orion/personality/graph.py`) shipped in #3321. That PR is merged and deployed.

## Architecture touched

- `orion/physiology/personality_values.py` (new, pure): care, strength `w/(w+10)`, holdback arm, runaway guard, `care_promoted_keys`, trace build and parse.
- `orion/physiology/weights.py`: `ChoiceRecord`, the lane-to-family map, `option_family`, `endorsement_for`, source (c) in `Evidence`, and `endorsement_exclusions` (the independence check).
- `orion/physiology/appetite.py`: `rank_value(..., care=)`.
- `orion/physiology/choice.py`: `decide(..., personality=)`, `MarketContext.weights_holdback`, and the trace.
- `orion/physiology/client.py`: `bid(..., object_text=)`, sent as a query parameter.
- orion-durable-runs:
  - `app/personality_values.py` (new): the reads, the embed cache, the background refresh and guard, and the per-bid term.
  - `app/physiology_market.py`: weights with and without (c), the term passed into every decide, and the log.
  - `app/run_bids.py`: `want_object_text`, which reads `brief.retrieval_query` or else `brief.title`.
  - `app/main.py`: the wiring, plus `?object_text=` on `POST /physiology/bid`.
  - `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.

**Family mapping for source (c):**

| Choice lane | Want family |
|---|---|
| outreach | `hub.outreach` |
| reading | `hub.world_pulse_read` |
| curiosity | `durable.curiosity_investigate` |
| prime | `prime` |
| chat | none (Tier 0 never bids) |

An option key of the form `<family>::<id>` (`weights.endorsement_option_key`) overrides the lane mapping. A choice point counts only when a "none" option was on offer (prime and outreach, the same lanes as `graph.DECLINE_TEXT`), when the row declined all options, or when the options span two or more families.

## Files changed

- `orion/physiology/personality_values.py`: new pure module (care, arm, guard, trace).
- `orion/physiology/weights.py`: source (c) and the independence check.
- `orion/physiology/appetite.py`: the care multiplier before the percentile.
- `orion/physiology/choice.py`: the personality term, holdback weights, and trace warnings.
- `orion/physiology/client.py`: optional `object_text`.
- `orion/physiology/tests/test_personality_values.py`, `test_weights_endorsement.py`, `test_choice.py`: new tests.
- `services/orion-durable-runs/app/personality_values.py`: new reader.
- `services/orion-durable-runs/app/physiology_market.py`, `run_bids.py`, `main.py`, `settings.py`: the wiring.
- `services/orion-durable-runs/.env_example`, `docker-compose.yml`, `README.md`: the flag and the docs.
- `services/orion-durable-runs/tests/test_personality_values_market.py`: new. Five existing test fakes accept `object_text`.
- `services/orion-durable-runs/evals/personality_values_replay.py`: new read-only live replay.

## Schema / bus / API changes

- Added:
  - An optional query parameter `object_text` on `POST /physiology/bid`.
  - An optional kwarg `object_text` on `orion.physiology.client.bid` and `abid`.
  - New `personality:*` lines in `WantDecisionV1.warnings`.
- Removed: none.
- Renamed: none.
- Behavior changed: decisions bind on care and on source (c) when the flag is on.
- Compatibility notes:
  - There is no change to `WantV1` or `WantDecisionV1` and no new channel.
  - Old callers are unaffected, because a bid without `object_text` gets care 0.

## Env/config changes

- Added keys:
  - `PHYSIOLOGY_PERSONALITY_VALUES=on`
  - `PHYSIOLOGY_PERSONALITY_EMBED_TIMEOUT_SEC=1.0` (orion-durable-runs)
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-durable-runs`: yes. Both keys were added.
- Skipped keys requiring operator action: none.

## Metric quality gate (CLAUDE.md 0A)

**Care**

1. **Provenance.**
   - The tendencies are `personality_disposition` rows with `origin='choice'` and `status='active'`, written by the #3321 reducer. Each direction is that row's centroid, which is the mean of `unit(embed(chosen) - mean(embed(declined)))`.
   - The want text comes from `run_bids.want_object_text`, which reads the run's `brief.retrieval_query` or `brief.title`. The prompt is never used, because it is mostly instructions.
   - Embeddings come from vector-host `/embedding`, the same endpoint the reducer uses.
2. **Independence.**
   - Care is circular by construction: it is a market input derived from choices the market offered. The holdback arm, `care_promoted` at half weight, and the 60% / 7-day cap handle this.
   - Care is not a transform of any existing market input. It is a third factor on value, beside intensity and outcome.
3. **Theory anchor.**
   - Revealed preference (Samuelson 1938): a choice among available alternatives reveals a preference.
   - CAPS if-then signatures (Mischel and Shoda 1995): a tendency is a situation-dependent pattern of choosing.
   - `w/(w+10)` reaches half strength at the spec's own 10-weighted-choices line for a real tendency.
4. **Live sanity.** Read-only replay on 1,852 real wants over 72 h. The live state had 1 active tendency (weight 5.04). The replay re-folded 217 choices and got the same single tendency.
   - **Care is exactly 0 for 92.2% of wants.** This passes the spec's at-least-30% rule. Among the 272 wants that have text, 46.7% read 0.
   - The rest point is reachable: a want with no text or no positive cosine reads exactly 0.0 by construction, not by decay.
   - Not saturated: 0 of the 272 wants read care of 0.99 or more. The maximum is 0.067, the median of positive values is 0.013, and p90 is 0.034.
   - The distribution moves. It is not flat: 145 wants read above 0, across 5 families (pr_field_note, world_pulse_read, self_inquiry, curiosity_investigate, consult).
   - **Finding:** the effect is small today, at most a 3.4% value lift. Two causes: one weak tendency (strength 0.34), and small cosines between raw text and a chosen-minus-declined direction (at most about 0.2).
   - Holdback draw: 18.8% of these wants (349 of 1,852).
5. **Existing mechanism.**
   - The care term uses the reserved socket in `weights.py`.
   - The holdback copies Prime's `draw_arm` sha256 pattern, with its own salt.
   - Embeddings reuse vector-host and the reducer's `http_embedder`.
   - No new table, channel or schema.
6. **Reversibility.** One flag, `PHYSIOLOGY_PERSONALITY_VALUES=off`, followed by a restart, gives exactly the old decisions. A test checks this. Nothing is persisted except warning strings, so removal is cheap.

**Endorsement, source (c)**

1. **Provenance.** `personality_choice` rows, read by `CHOICES_SQL` in `app/personality_values.py`. The families come from the lane map or the option key.
2. **Independence from (b) Juniper engagement.**
   - The check is a Pearson correlation between the two, per family. It uses non-overlapping daily rates, because a rolling 30-day series is autocorrelated and would read falsely high.
   - If r > 0.8 on at least 7 paired days, (c) is left out for that family.
   - Each weight refresh logs `personality_values_endorsement`.
   - Live today: there are no paired days, because (c) has no real choice points yet. The check is UNVERIFIED on live data.
3. **Theory anchor.** Revealed preference: a rate per offer, not a count.
4. **Live sanity: degenerate input found and excluded.**
   - The 137 live choice rows in the 30-day window are all curiosity, with 0 declines. Each one picks from a one-family prior menu with no "none" option, so its rate would always read 1.0, which is saturated.
   - Such menus are therefore not choice points. Today no family has source (c), and the weights are unchanged.
   - (c) starts reading when Prime ("no move", #3330, merged) and outreach ("say nothing") write rows.
5. **Existing mechanism.** The reserved socket, the kappa shrinkage, and the 0.1-per-day drift cap are unchanged.
6. **Reversibility.** The same flag. `weights_now` without `choices` is identical to before this patch.

## Tests run

```text
pytest orion/physiology/tests orion/personality -q                       491 passed
cd services/orion-durable-runs && pytest tests -q                          905 passed, 136 skipped
```

## Evals run

```text
python3 services/orion-durable-runs/evals/personality_values_replay.py --dsn <live, read-only> --hours 72
tendencies: 1 (weight 5.05, strength 0.34)
runaway guard: shown_options_7d=715 capped=[]
wants 1852, with_text 272, care_zero_share 0.9217, care_zero_share_with_text 0.4669,
care_positive 145, p50_pos 0.0126, p90_pos 0.0335, max 0.067, saturated 0
source (c): 137 choice rows in 30 days, real choice points: 0
holdback arm: 349 of 1852 (18.8%)
checks: care_zero_share_at_least_30pct True, not_saturated True, care_not_inert True   (exit 0)
```

## Docker/build/smoke checks

```text
Not run. This patch is pure Python inside the existing durable-runs image. There are no new dependencies (httpx is already used by the personality driver). There is no deploy from a worktree.
```

## Review findings fixed

See the PR comment or the section below (filled in after the review subagent).

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run it from the clean primary `main` checkout after merge. orion-durable-runs is the only service that needs a restart. Hub, mind and the other producers do not.

## Risks / concerns

- Severity: medium.
  - Concern: the care term is nearly inert today, with a value lift of at most 3.4%. Live there is 1 tendency, and the cosine of raw text against a difference direction is small. With effects this small, M2 (the chi-square of care arm against holdback) will likely not reach significance until there are more and stronger tendencies.
  - Mitigation: none in this patch. Recommendation: after 7 days of Prime and outreach choice rows, re-run the replay. If p90 care is still below 0.1, normalize the cosine by the tendency's own choice-direction spread, or compare `embed(text) - tendency baseline` instead of the raw text. Either one is a one-line change in `care_for`.
- Severity: medium.
  - Concern: source (c) has no data. Prime's options use concern ids, so the Prime lane maps to the `prime` family as a whole: picking a concern counts as endorsing Prime acting at all, not the family the concern came from.
  - Mitigation: `endorsement_option_key(family, id)` is ready for Prime and outreach to adopt. This patch does not edit their files (others own them).
- Severity: low.
  - Concern: outreach wants carry no text yet. Their `object_ref` is "juniper", so they read care 0.
  - Mitigation: `client.bid(object_text=...)` is ready for outreach to pass its topic.
- Severity: low.
  - Concern: the runaway guard has never fired live. 715 shown options in 7 days, capped none.
  - Mitigation: unit-tested, and every cap is logged.

## PR link

(see below)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
