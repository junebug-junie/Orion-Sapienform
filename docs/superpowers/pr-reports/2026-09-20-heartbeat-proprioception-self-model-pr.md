# Heartbeat proprioception on the self-model and Hub

Date: 2026-09-20
Branch: `feat/heartbeat-proprioception`

## Summary

- Heartbeat `/h1` now reports who has been silent, whether the lattice already looks the same far away as nearby, and whether the five organs are talking as one blob.
- The attention self-model copies those fields. The basis string names dark seats, not the saturated mean-ratio.
- The Attention Organ tab leads with dark seats / smear / distinctness. Mean ratio is secondary.
- Empty occupancy is `untracked`, not `none`. The smear history chart can go above 1.
- Mind, Stance, field, and chat unified are not wired.

## Outcome moved

Heartbeat stops lying with a capacity-saturated mean as the headline. The reading Orion already computes (who is dark, whether coupling is smeared, whether seats are distinct) is now on `/h1`, on the attention self-model, and on the Hub chart that already shows this service.

Live Hub still serves the old `/h1` until heartbeat, substrate-runtime, and hub restart. That check is UNVERIFIED.

## Current architecture

Before this patch, `/h1` and the self-model led with mean-ratio + verdict. Mean-ratio saturates under real traffic. Dark seats / smear / distinctness existed only as probe math, not as a live reading. The Hub Attention Organ tab treated mean-ratio as the story.

## Architecture touched

- `orion-heartbeat` — occupancy window on absorb; `/h1` attaches proprioception
- `AttentionSelfModelV1` — additive fields + basis rewrite
- `orion-hub` Attention Organ — headline, history, snapshot link
- Metric definition lock — two new numeric inner-state fields

No bus channels. No field node. No Mind/Stance/chat consumer.

## Files changed

- `services/orion-heartbeat/app/substrate/proprioception.py`: dark seats, distinctness, smear
- `services/orion-heartbeat/app/substrate/reconstruction.py`: attach proprioception to `/h1`
- `services/orion-heartbeat/app/service.py`: record last-64 organ fires, pass counts into H1
- `services/orion-heartbeat/app/substrate/routing.py`: `SITE_ORGAN_MAP`
- `services/orion-heartbeat/app/substrate/ensemble.py`: result fields
- `services/orion-heartbeat/README.md`: `/h1` headline
- `orion/schemas/attention_self_model.py`: additive proprioception fields
- `orion/substrate/attention_self_model.py`: copy fields; basis names dark seats
- `services/orion-hub/scripts/attention_organ_routes.py`: history + snapshot link
- `services/orion-hub/static/js/attention-organ.js`: headline, smear chart, untracked vs none
- `config/metrics/metric_definitions.lock.json`: lock the two new scalars
- Tests and this PR report

## Schema / bus / API changes

- Added: `/h1` `dark_seats`, `organ_fire_counts`, `organ_distinctness`, `smear`, `smeared`
- Added: `AttentionSelfModelV1.heartbeat_dark_seats`, `heartbeat_organ_fire_counts`, `heartbeat_organ_distinctness`, `heartbeat_smear`, `heartbeat_smeared`
- Added: lock entries for `heartbeat_organ_distinctness` and `heartbeat_smear`
- Removed: none
- Renamed: none
- Behavior changed: self-model `heartbeat_basis` names dark seats (`untracked` / `none` / organ list), not mean_ratio
- Compatibility notes: additive fields default empty/None. Old `/h1` without occupancy extras still populates verdict. Equilibrium still reads `prediction_error_confidence` from the JSON dict, not the full model.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_attention_organ_page.py -q
→ 41 passed

cd services/orion-heartbeat && PYTHONPATH=repo+service pytest \
  tests/test_reconstruction_h1.py tests/test_proprioception.py \
  tests/test_service_filtering.py tests/test_ensemble.py -q
→ 43 passed

PYTHONPATH=. pytest orion/substrate/tests/test_attention_self_model.py -q \
  -k "Heartbeat or proprioception or all_lit or missing_proprioception"
→ 17 passed, 46 deselected

PYTHONPATH=. pytest tests/test_attention_schema_surface.py \
  orion/substrate/tests/test_attention_self_model.py -q
→ 85 passed

PYTHONPATH=.:services/orion-substrate-runtime pytest \
  services/orion-substrate-runtime/tests/test_worker_attention_self_model_tick.py -q
→ 20 passed

python scripts/check_definition_drift.py --update
→ 662 defs; added heartbeat_organ_distinctness, heartbeat_smear
```

## Evals run

```text
No proprioception quality eval harness on orion-heartbeat.
Live smear rest/calm after restart is UNVERIFIED.
```

## Docker/build/smoke checks

```text
No compose or env change. Live /h1 on 127.0.0.1 still lacks dark_seats
until heartbeat + substrate-runtime + hub restart. UNVERIFIED.
```

## Review findings fixed

- Finding: empty dark seats rendered as "none" when occupancy was untracked.
  - Fix: `formatSeats` and the snapshot link take fire counts; empty+empty is "untracked".
  - Evidence: `test_attention_organ_js_leads_with_proprioception_not_mean_ratio`.
- Finding: smear history chart clamped at 1 while smear can exceed 1.
  - Fix: chart max is the window's actual smear high-water mark.
  - Evidence: `smearMax` in attention-organ.js.
- Finding: self-model basis said `dark_seats=none` when occupancy extras were missing.
  - Fix: basis is `untracked` with no fire counts, `none` only when counts exist and every seat is lit.
  - Evidence: `test_missing_proprioception_still_populates_verdict`, `test_all_lit_seats_basis_says_none_not_untracked`.

Second review on the uncommitted tree after those Hub fixes: no remaining findings.

## Restart required

```bash
# from a worktree, not the shared checkout. Do not sudo.
scripts/safe_docker_build.sh orion-heartbeat up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Then read `/h1` and the Attention Organ tab. Old Hub will keep showing mean-ratio until all three restart.

## Risks / concerns

- Severity: low
- Concern: Hub tests prove the JS source contains the helpers; they do not execute `formatSeats` on fixtures.
- Mitigation: Python proprioception + self-model tests cover the numbers. Live UI after restart is the real check.
- Severity: low
- Concern: smear rest/calm after deploy is UNVERIFIED.
- Mitigation: do not treat unit tests as a live calm reading.
- Severity: low
- Concern: `check_metric_dead_wiring` blocked the first commit because live Postgres has never stored these keys (n=0 over 1h). That is the chicken-and-egg of a first producer, not a dead metric we hid.
- Mitigation: committed with `ORION_ALLOW_DEAD_METRIC_WIRE=1`. After heartbeat + substrate-runtime restart, the next ticks should produce the keys.

## Metric quality gate

New numeric inner-state fields: `heartbeat_organ_distinctness`, `heartbeat_smear`.

1. **Provenance.** Distinctness is occupancy Shannon entropy over the five allowlisted organs in the last 64 absorbs (`occupancy_distinctness` in `services/orion-heartbeat/app/substrate/proprioception.py`). Smear is far/near of the current 9-cut mean entropy profile (`profile_smear`, same cut indices as the lattice kick probe).
2. **Independence.** Distinctness is a count distribution. Smear is a geometry ratio. Neither is mean_ratio, std_ratio, bulk depth, or field prediction-error.
3. **Theory.** Proprioception: which organs are silent, whether coupling already looks the same far away as nearby, whether seats are speaking as one blob.
4. **Live-data sanity.** Unit-tested. Live rest/calm of smear after deploy is UNVERIFIED until heartbeat is restarted and `/h1` is read.
5. **Existing mechanism.** These were probe-only. Not previously on `/h1` or the self-model.
6. **Reversibility.** Schema fields plus a lock update. Cheap to drop. No field node, no stance switch.

## Non-goals

No IIT claim. No χ raise. No field node. No Mind/Stance injection. No chat recitation.

## PR link

(filled after `gh pr create`)
