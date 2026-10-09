# GPU pool stage 5.5: labels derived from the pool

## Summary

Three places still had a hand-typed answer to "which card is this?". This PR replaces each one with
the pool's own answer.

- **The Hub biometrics GPU badges now show what the pool says is on each card.** Before this, the
  labels came from two hand-maintained env keys that were `{}` in production, so every card read
  "unassigned". Now circe's card 2 reads `world, diffusion`, or `agent-gpu2, world` while the 27B is
  loaded. Card 0 reads `chat (lent)`, card 1 `agent`, and card 3 `metacog, fast`.
  - A card whose pool state is missing or stale reads `no pool state` (grey). That means unknown,
    which is a different fact from "nothing assigned".
  - athena has no pool, so its cards still read `unassigned`.
  - The env keys `GPU_LANE_MAP_{ATHENA,CIRCE}_JSON` are deleted, with no fallback.
- **The pool now says which host it manages and each card's device number.** These are two new
  optional fields: `GpuPoolStateV1.host` and `GpuCardStateV1.index`. `config/gpu_pool.yaml` gains
  `index: 0/1/3` for gpu0/1/3. Those numbers were verified live on circe (evidence below).
  - Launch digests are unchanged, so the circe controller's fence does not move.
- **Image generation's power intent names its card from the service's own `CUDA_VISIBLE_DEVICES`.**
  That is the variable the actuator sets from the card index, and the same one torch uses to pick
  the card.
  - It only counts as a physical index when `CUDA_DEVICE_ORDER=PCI_BUS_ID`, which is baked into the
    image. If the index can't be resolved, the service declares nothing and logs
    `power_intent_gpu_index_unresolved`. It never guesses.
  - `DIFFUSION_POWER_INTENT_GPU_INDEX` is deleted.
- **The controller's pool path no longer builds the fixed-slot `GpuSlotRequestV1`.** The gpu2 bridge
  now takes a local `gpu2.Transition(target, operation_id, generation)` that accepts only the two
  fixed targets. After this PR, `orion/schemas/gpu_slot.py` has no production importer, so 5.6 can
  delete it.
- **CI now gates the Hub label tests.** They were not run by any workflow before; they now run in
  `orion-gpu-pool-tests.yml`.

## Outcome moved

- The biometrics modal shows real GPU assignments instead of `unassigned` everywhere. The
  urgent-curiosity evidence bundle gets the same labels, so Orion and Juniper see the same thing.
- When a card moves in `gpu_pool.yaml`, the power-intent card follows it without anyone editing an
  env file. Today's value is unchanged: 2.

## Current architecture

- **Hub labels:** `biometrics_preview_routes._parse_lane_map` read
  `settings.GPU_LANE_MAP_{ATHENA,CIRCE}_JSON`. Both keys were `{}` in the live Hub container, so
  every card read "unassigned".
- **Pool state:** the per-card state carried only the card name (`gpu2`), not its device number, and
  the state named no host.
- **Diffusion-host:** `settings.DIFFUSION_POWER_INTENT_GPU_INDEX=2` was a literal in settings,
  `.env_example` and the live `.env`.
- **Controller:** `actuator_bus._run`'s bridge branch built
  `GpuSlotRequestV1(slot="circe-gpu2", ...)`, the HTTP-era fixed card→target ownership contract.

## Architecture touched

- **Contract:** `orion/schemas/gpu_pool.py`, two additive optional fields. The
  `orion:gpu_pool:state` description in `orion/bus/channels.yaml` is updated. No new channels or
  kinds.
- **Services:** orion-gpu-pool (producer), orion-hub (consumer), orion-diffusion-host,
  orion-gpu-lane-controller, and orion-world-model (comments only).
- **Config:** `config/gpu_pool.yaml` (card indices).

## Files changed

- `orion/schemas/gpu_pool.py`: `GpuPoolStateV1.host`, `GpuCardStateV1.index`.
- `services/orion-gpu-pool/app/runtime.py`: the snapshot fills both fields.
- `config/gpu_pool.yaml`: `index` on gpu0/1/3, with a live-verification comment.
- `orion/bus/channels.yaml`: the state channel description.
- `services/orion-hub/scripts/biometrics_preview_routes.py`: `lane_map_from_pool_state`,
  `pool_lane_map`, and the new `lane_assigned` card field. `_parse_lane_map` is deleted.
- `services/orion-hub/scripts/urgent_evidence.py`: uses the pool labels.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: the keys are deleted.
- `services/orion-hub/static/js/biometrics-view.js`: a `laneBadge()` helper. The badge greys on the
  server's `lane_assigned`, not on a string match.
- `services/orion-diffusion-host/app/{main,settings}.py`, `.env_example`: `resolve_power_intent_gpu_index`,
  a boot check, and publish withholding. The key is deleted.
- `services/orion-world-model/{app/gpu.py,.env_example}`: comments that cited the deleted key.
- `services/orion-gpu-lane-controller/app/{gpu2,actuator_bus}.py`: `gpu2.Transition` replaces
  `GpuSlotRequestV1`.
- Tests:
  - `services/orion-hub/tests/test_biometrics_preview_api.py`, `test_urgent_evidence.py`,
    `test_biometrics_view_ui.py`, and `static/js/biometrics-view.test.js`;
  - `services/orion-diffusion-host/tests/test_power_intent_publish.py`;
  - `services/orion-gpu-lane-controller/tests/{test_gpu2,test_actuator_bus}.py`;
  - `services/orion-gpu-pool/tests/test_runtime.py` and `orion/gpu_pool/tests/test_stage4_contracts.py`.
- `.github/workflows/orion-gpu-pool-tests.yml`: a Hub labels step, plus paths.

## Schema / bus / API changes

- **Added:**
  - `GpuPoolStateV1.host: str | None`;
  - `GpuCardStateV1.index: int | None`;
  - `/api/biometrics/preview/gpu` card field `lane_assigned: bool`.
- **Removed:** none on the bus. `orion/schemas/gpu_slot.py` stays for 5.6, but it now has no
  production importer.
- **Behavior changed:**
  - the `lane` text is now pool-derived;
  - a new label value, `no pool state`, appears when the pool state is missing, stale (over 60 s) or
    from a pre-5.5 pool.
- **Compatibility:**
  - No consumer validates the whole `GpuPoolStateV1` or `GpuCardStateV1`. Hub keeps the raw dict;
    placement validates only `DiscoveredRoleV1` rows, which are unchanged. So an old Hub facing a
    new pool is fine.
  - A new Hub facing an old pool reads `no pool state` on circe cards until the pool is redeployed.

## Env/config changes

- **Removed keys:** `GPU_LANE_MAP_ATHENA_JSON` and `GPU_LANE_MAP_CIRCE_JSON` (orion-hub), and
  `DIFFUSION_POWER_INTENT_GPU_INDEX` (orion-diffusion-host).
- **Added / renamed keys:** none.
- **`.env_example` updated:** yes, for hub, diffusion-host and world-model (comment only).
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py`: run; it reports no
  new keys. The sync script does not prune deleted keys.
- **Keys still in live `.env` files.** Both services use `extra="ignore"`, so these are harmless at
  runtime. They are safe to delete by hand:
  - athena `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env:640-641`: `GPU_LANE_MAP_ATHENA_JSON={}` and `GPU_LANE_MAP_CIRCE_JSON={}` (also in the running `orion-athena-hub` container env);
  - athena `/mnt/scripts/Orion-Sapienform/services/orion-diffusion-host/.env:66`: `DIFFUSION_POWER_INTENT_GPU_INDEX=2`;
  - circe `/mnt/scripts/Orion-Sapienform/services/orion-diffusion-host/.env:53`: `DIFFUSION_POWER_INTENT_GPU_INDEX=2` (and the comment above it).

## Metric quality gate: `power_intent_settled.gpu_index` (derived source)

1. **Provenance.** The value is `resolve_power_intent_gpu_index(os.environ["CUDA_VISIBLE_DEVICES"],
   os.environ["CUDA_DEVICE_ORDER"])` in `services/orion-diffusion-host/app/main.py`. It is resolved
   once at import, and `_publish_power_intent` puts it in `PowerIntentV1.gpu_index`.
   - The path is `orion:power:intent` → `orion-biometrics/app/power_intent.py` settler →
     `power_intent_settled.gpu_index`.
   - `CUDA_VISIBLE_DEVICES` is compose `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}`, which the
     controller sets. The 5.1 CI gate forces it to equal `cards.gpu2.index`.
2. **Independence.** This is not a new signal. It is the same number from a different source. The
   power prior keys on the same index, so nothing downstream changes meaning.
3. **Theory anchor.** The settler samples nvidia-smi on that index. It must be the card the model
   actually runs on, and `CUDA_VISIBLE_DEVICES` is literally what torch uses to pick that card.
   Under `PCI_BUS_ID` order, CUDA and nvidia-smi numbering agree.
4. **Live check.**
   - **Before:** the last 20 `reverie_diffusion` settlements (2026-09-28 14:40 → 2026-09-29 21:30
     UTC) are all `gpu_index=2`, `node=circe`, all `settled`. Peak watts range from 241.5 to 268.1,
     which is a real load on that card, not an idle reading.
   - Over all time: 950 rows, every one `gpu_index=2`.
   - The running circe container (`docker inspect orion-circe-diffusion-host`) has
     `CUDA_VISIBLE_DEVICES=2` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The resolver maps that pair to
     `(2, None)`, pinned by `test_live_circe_env_resolves_to_todays_index`.
   - **After deploy: UNVERIFIED.** Re-run the query below after diffusion-host restarts. The new
     rows must all read 2.
5. **Existing mechanism.** None. The pool's card `index` is the only other source, and it is what
   sets this variable.
6. **Reversibility.** Revert the commit and restore the env key. Nothing new is persisted.

```sql
SELECT gpu_index, count(*), min(settled_at), max(settled_at)
FROM (SELECT * FROM power_intent_settled WHERE workload_kind='reverie_diffusion'
      ORDER BY settled_at DESC LIMIT 20) t GROUP BY 1;
```

## Card index verification (gpu0/1/3)

On circe, 2026-09-29, `docker inspect`:

| container | device setting | role |
| --- | --- | --- |
| atlas-llamacpp-chat | `CUDA_VISIBLE_DEVICES_OVERRIDE=0` | `LLM_ROLE=chat` |
| atlas-llamacpp-agent | `=1` | agent |
| atlas-llamacpp-metacog | `=3` | metacog |
| atlas-llamacpp-fast | `=3` | fast |

`nvidia-smi` on circe shows 4 cards, indices 0 to 3. These roles have no launch block, so CI
cannot check the indices against compose (their `ATLAS_*_CUDA_VISIBLE_DEVICES` keys are empty in
`.env_example`). That risk is listed below.

Live dry run of the Hub labels against the running pool's `/v1/pool`, with the new fields added:
`{'0': 'chat (lent)', '1': 'agent', '2': 'world, diffusion', '3': 'metacog, fast'}`. athena returns
`({}, 'unassigned')`. The same state without the fields (a pre-5.5 pool) returns `no pool state`.

## Tests run

```text
services/orion-hub  test_biometrics_preview_api + test_urgent_evidence (-k "not router_registered")  45 passed
services/orion-hub  test_biometrics_view_ui -k lane_badge                                             1 passed
node --test services/orion-hub/static/js/biometrics-view.test.js                                      7 passed
services/orion-diffusion-host/tests + services/orion-gpu-lane-controller/tests                        165 passed
orion/gpu_pool/tests                                                                                  290 passed
services/orion-gpu-pool/tests (no local Postgres)                                                     93 passed, 7 skipped
scripts/check_gpu_pool_config.py                                                                      ok
orion-static-gates.yml (every step, incl. metric definition drift + node:test)                        all OK
scripts/check_env_template_parity.py                                                                  PASS
```

Pre-existing and unrelated (also fails on main):
`test_biometrics_view_ui.py::test_app_js_deactivates_biometrics_view_when_leaving_the_hub_tab`. Its
700-character window in `app.js` no longer reaches the `deactivate()` call, which does exist
(`app.js:1132`). CI runs only this PR's `lane_badge` test from that file.

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py   VERDICT: PASS
```

There is no eval harness for Hub biometrics labels or diffusion-host. The label mapping is
deterministic and covered by tests. The metric gate's after-deploy query above is the live eval.

## Docker/build/smoke checks

Nothing was deployed. Per the task, there were no builds or restarts. Launch digests for
`agent-gpu2` and `diffusion` are unchanged against main (computed), so the circe controller does
not need the YAML to deploy in lockstep.

## Review findings fixed

- **Finding:** the branch conflicted with main. Stage 5.2's `launch_exec` branch in
  `actuator_bus._run` still built `GpuSlotRequestV1`.
  - **Fix:** rebased onto main. The bridge branch now uses `gpu2.Transition`, and the docstring was
    updated for 5.2 having landed.
  - **Evidence:** controller tests 79 → pass. `test_pool_path_does_not_use_the_fixed_slot_contract`
    passes.
- **Finding:** the new CI step would fail. `test_router_registered_on_api_routes` imports
  `api_routes`, which needs spacy, and the job doesn't install it.
  - **Fix:** added `-k "not router_registered"`, plus a comment on the step's dependency on the
    previous step's Hub pins.
- **Finding:** the `orion:gpu_pool:state` channel doc didn't mention `host` or `cards[].index`.
  - **Fix:** updated `orion/bus/channels.yaml`.
- **Finding:** the PR needed a deploy-order note.
  - **Fix:** see Restart required and Compatibility.
- **Finding:** deleted keys are still in local `.env` files.
  - **Fix:** listed above for operator removal. They are harmless under `extra="ignore"`.
- **Nit:** a role row without a status rendered as `name [None]`.
  - **Fix:** it now renders `[unknown]`, with a test.
- **Nit:** the urgent-evidence test patched `__globals__`.
  - **Fix:** `pool_lane_map(node, now=)`; `urgent_evidence` passes its own `_now_utc()`.
- **Nit:** the staleness clock assumption was undocumented.
  - **Fix:** added a comment. The review assumed the pool's clock is circe's, but the pool runs on
    athena, the same host as Hub.

## Restart required

Order: the pool first, then Hub. If Hub goes first, circe cards read `no pool state` until the pool
catches up. Diffusion-host and the circe controller are independent of that order.

```bash
# athena (after merge + pull)
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# circe (after pull; image rebuild needed: code changed)
scripts/safe_docker_build.sh orion-diffusion-host up -d --build
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
```

Restarting diffusion-host drops the loaded model while it reloads. Do it when gpu2 is idle
(`GET :8127/v1/pool`: gpu2 `swap_state=idle`, no diffusion hold granted).

## Risks / concerns

- **Severity: low.** gpu0/1/3 indices are checked live, not in CI. If a resident llama worker moves
  card, the label will be wrong until the YAML is edited. The mitigation is the YAML comment. The
  real fix is launch blocks for the residents, which the experiment seat also needs (stage 5
  Decision 3).
- **Severity: low.** A browser holding a cached pre-deploy `biometrics-view.js` would show
  `no pool state` in the "assigned" colour. Hub's asset version changes on rebuild, so this clears
  itself.
- **Severity: low.** If a future diffusion launch sets several devices or drops `PCI_BUS_ID`, power
  intents stop, loudly: a boot and first-publish error, and `power_intent_settled` stops growing.
  That is deliberate. An intent on a guessed card would settle the wrong card's watts.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2411

🤖 Generated with [Claude Code](https://claude.com/claude-code)
