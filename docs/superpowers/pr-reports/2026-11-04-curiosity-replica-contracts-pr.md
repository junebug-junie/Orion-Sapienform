# feat(curiosity): curiosity replica patch 2 -- the contracts

Design: `docs/superpowers/specs/2026-11-04-curiosity-replica-design.md` (Juniper approved it and the build
order on 2026-11-04). Patch 0 (value arm on) and patch 1 (loop library) are merged and live. This is
patch 2: the shapes the replica and Hub will talk through. Nothing in this patch makes a model call or
sends a plan request.

## Summary
- **The arm draw can now pick one of three orders** (value, replica, uncertainty). Hub gives the replica
  arm no chance yet, so every run gets exactly the arm it got before. Each offer row now records the
  chance every arm had.
- **A validated plan can fill the top of the offer.** `select_priors` takes the replica's picks and puts
  them first; the rest of the layout (6 ordered, 2 random, the stale bucket) does not change.
- **One sentence about the order, true on all three arms.** The old sentence said "the code estimates a
  test could change your mind the most". That is false when the replica sets the order. The new one says
  only that the order was chosen before the run and is not a recommendation.
- **The replica's answer shapes exist**: the plan request, the plan result, the model's answer, and
  support / challenge / refuse stances. Registered in both schema registries, with two bus channels.
- **Curiosity can refuse a focus Prime queued**, with a reason. The focus leaves the slot and reads
  `refused_by_organ`. A focus Juniper asked for can never be refused: the code, the SQL guard and a table
  CHECK each block it.
- **A body-economy family for the replica**: `curiosity.replica`, tier 2, lane chat, 90 s, no receptors.

## Outcome moved
Nothing live moves on purpose. This patch is the contract layer the replica (patch 3) will plug into.
What a reviewer can check today:
- The two-arm 50/50 experiment is unchanged run for run (a test re-derives 3,000 run ids with the old
  formula and the new one).
- After deploy, every new `curiosity_offer_decisions` row carries
  `arm_propensities = {"value_order": 0.5, "replica_order": 0.0, "uncertainty_order": 0.5}`.
- The kickoff prompt text is byte-identical across the three arms except for the prior lines (a test
  renders one view three ways).

## How the replica arm is kept off
Hub calls `draw_offer_arm(run_id, value_enabled=..., value_propensity=...)` and passes no replica
propensity. The replica band then has width 0, so no run can land in it. There is no new env key: the
switch (`HUB_CURIOSITY_REPLICA_ARM_ENABLED`, `..._PROPENSITY`) arrives in patch 3 together with the code
that answers a plan request, and ships on then. The value band comes first and uses the same hash as
before, so turning the replica on later does not move any run that was on the value arm.

## Current architecture
- `offer_arm(run_id, enabled, propensity)` was a two-way hash draw (value / uncertainty).
- `select_priors` sorted fresh priors by uncertainty or expected nats; no way for an outside plan to set
  the top.
- The live-priors section of the kickoff said the order puts first "where the code estimates a test could
  change your mind the most".
- A focus row could only leave the slot by being taken, replaced, cleared by Juniper, or expiring.

## Architecture touched
- `orion/curiosity/` (value, worldview, kickoff_prompt, focus), `orion/schemas/`, `orion/prime/state.py`,
  `orion/bus/channels.yaml`, `config/physiology/families.yaml`, Hub's offer-row writer, two SQL migrations.

## Files changed
- `orion/schemas/curiosity_replica.py` (new): request, candidate, pending focus, pick, stances, answer,
  result; channel names, kinds, limits; `plan_row_status_allowed`.
- `orion/schemas/registry.py`: the three plan models and `PrimeConcernsViewV1`, in both registries.
- `orion/bus/channels.yaml`: `orion:curiosity:replica:plan:request` (hub -> mind, single consumer) and
  `orion:curiosity:replica:plan:result:*` (mind -> hub).
- `config/physiology/families.yaml`: `curiosity.replica`.
- `orion/curiosity/value.py`: `ARM_REPLICA_ORDER`, `OFFER_ARMS`, `ArmDraw`, `draw_offer_arm`; `offer_arm`
  is now a thin wrapper with the same result.
- `orion/curiosity/worldview.py`: `select_priors(head_ids=...)`.
- `orion/curiosity/kickoff_prompt.py`: `ORDER_SENTENCE` (the self-inquiry line renders the same section).
- `orion/curiosity/focus.py`: `REFUSED_BY_ORGAN`, `refuse_focus`, `organ_refusal_blocker`, the three new
  columns on every read, the story line; a refused question re-queued by Prime is not reported as queued.
- `orion/prime/state.py`: `PrimeConcernsViewV1` and `concerns_view` (what Prime will publish for the
  replica to read; the writer comes with the reader in patch 3).
- `services/orion-hub/scripts/curiosity_investigation.py`: uses `draw_offer_arm`, passes the propensities.
- `services/orion-hub/scripts/curiosity_offer_decisions.py`: writes `arm_propensities`.
- `services/orion-sql-db/manual_migration_curiosity_replica_plan_v1.sql` (new).
- `services/orion-sql-db/manual_migration_curiosity_focus_request_refused_v1.sql` (new).
- `.github/workflows/orion-memory-episode-tests.yml`: the real-Postgres focus test also runs when the new
  focus migration changes.
- `services/orion-hub/README.md`: arm draw, order sentence, refusal.
- Tests: `orion/curiosity/tests/test_replica_contracts.py` (new), `test_curiosity_value.py`,
  `test_focus_lifecycle.py` + `focus_fixtures.py`, `tests/test_curiosity_worldview.py`,
  `services/orion-hub/tests/test_curiosity_spend_log.py`, `test_curiosity_investigation.py`.
- `config/metrics/metric_definitions.lock.json`: re-locked for the two new channels.

## Schema / bus / API changes
- Added: `CuriosityOfferPlanRequestV1`, `CuriosityOfferPlanResultV1`, `CuriosityOfferPlanAnswerV1`
  (with `CuriosityOfferCandidateV1`, `CuriosityPendingFocusV1`, `CuriosityOfferPickV1`,
  `CuriosityFocusStanceV1`, `CuriositySignalStanceV1`), `PrimeConcernsViewV1`; two channels.
- Removed / renamed: none.
- Behavior changed: the offer row gains `arm_propensities`; the kickoff's order sentence is new text on
  both live arms (the design says to compare only runs randomized at the same time, so this is safe for
  the experiment); focus reads gain the refusal columns.
- Compatibility: no existing `extra="forbid"` model gained a field, so there is no consumer-first ordering
  to respect. New models have no producer on the bus yet.

## Migrations
- `manual_migration_curiosity_replica_plan_v1.sql` -- `REQUIRED-BY: orion-hub`. Adds
  `curiosity_offer_decisions.arm_propensities` and creates `curiosity_offer_plans` (no grant to
  `orion_readonly`). Idempotent.
- `manual_migration_curiosity_focus_request_refused_v1.sql` -- `REQUIRED-BY: orion-hub`. Adds
  `refused_by`, `refused_reason`, `refused_act_id` and two `NOT VALID` checks. Not re-runnable by hand
  (Postgres has no `ADD CONSTRAINT IF NOT EXISTS` outside a `DO` block, which the applier refuses); a
  second run fails and rolls back, changing nothing.
- Both pass `auto_apply_refusals` with the live tables present (test), and the live drift checker reads
  both as NOT APPLIED, so the post-pull applier applies them before Hub rebuilds. Both applied cleanly to
  a throwaway Postgres 16, and the focus lifecycle suite passed against it.

## Env/config changes
- Added / removed / renamed keys: none. No `.env_example` changed, so no local `.env` sync was needed.

## Tests run
```text
(filled in below)
```

## Evals run
```text
(filled in below)
```

## Docker/build/smoke checks
No Docker build: no dependency, Dockerfile or compose change. Hub and its migrations deploy through the
normal post-pull path.

## Review findings fixed
(filled in below)

## Restart required
```bash
# 1. git pull on primary main applies both migrations first (post-pull applier). If it did not:
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney --single-transaction -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_curiosity_focus_request_refused_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney --single-transaction -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_curiosity_replica_plan_v1.sql
# 2. then Hub (the deploy gate refuses it until both are applied):
scripts/safe_docker_build.sh orion-hub up -d --build
```
orion-mind needs no restart (nothing it runs changed).

## Risks / concerns
- Severity: low
  - Concern: if Hub starts before the focus migration, every focus read fails (missing column), the same
    as the `refs` column before it.
  - Mitigation: `REQUIRED-BY: orion-hub`; the deploy gate refuses Hub until it is applied, and the
    post-pull applier applies it first.
- Severity: low
  - Concern: a refusal is stored as `cleared` plus `refused_by`. Raw SQL that reads only `status` would
    count it as Juniper's Clear.
  - Mitigation: every reader in code goes through `orion/curiosity/focus.py`, which reports
    `refused_by_organ`; both migration headers say so. Nothing refuses a focus until patch 4.

## PR link
(filled in below)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
