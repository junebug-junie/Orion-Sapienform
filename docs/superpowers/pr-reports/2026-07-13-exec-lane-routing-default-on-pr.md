# PR: default EXEC_LANE_ROUTING_ENABLED on -- stops duplicate exec of non-chat verbs

Branch: `feat/exec-lane-routing-default-on` → `main`

## Summary

Found via a "wait, is this real?" skepticism check on `EXEC_LANE_ROUTING_ENABLED` being off. It was more than dead infrastructure — it was actively costing real, invisible duplicate LLM inference on every `spark`/`background`-lane verb orch dispatches. Flipped the flag on, applied it live, made it durable in code.

## Outcome moved

`introspect_spark`, `dream_cycle`, `journal.compose`, `log_orion_metacognition`, and agent/council-mode verbs — whenever dispatched via `orion-cortex-orch` (not via `orion-actions`/`orion-harness-governor` publishing directly, which was already fine) — now route to their own dedicated cortex-exec channel/container instead of being independently double-executed by both the `legacy` and `chat`-lane containers. Real compute cost eliminated on the affected verbs. `orion-cortex-exec-spark`, previously idle, now receives real traffic.

## Current architecture

`orion-cortex-orch`'s `resolve_execution_lane` (`execution_lanes.py`) computes a lane (`chat`/`spark`/`background`) for every dispatched verb. `orchestrator.py::call_verb_runtime`'s `use_direct_exec = exec_lane_routing_enabled and lane != "chat"` decides whether to publish directly to that lane's dedicated channel (`CHANNEL_EXEC_REQUEST_SPARK`/`_BACKGROUND`) or fall back to the shared `orion:verb:request` broadcast channel, consumed by `Hunter` — documented in `orion/core/bus/bus_service_chassis.py` as "Fire-and-forget consumer... Subscribes to patterns" (pub/sub broadcast, not a competing-consumer queue).

Four `orion-cortex-exec` containers run in production: `legacy` (base), `chat`, `spark`, `background` — each tagged via `EXEC_LANE` and each with its own `CHANNEL_EXEC_REQUEST[_LANE]`. Both `legacy` (`EXEC_LANE=legacy`) and `chat` (`EXEC_LANE=chat`) satisfy `main.py`'s `_lane in {"chat", "legacy", ""}` condition and both register a `Hunter` listener on the shared `orion:verb:request` channel.

## Investigation (this is the real content of the PR)

`EXEC_LANE_ROUTING_ENABLED` has been `false` since it was introduced (`bfac773b`, "Phase 2 exec lane isolation with opt-in routing," 2026-05-14) — roughly two months of the dedicated `spark`/`background` containers running while receiving little-to-no orch-originated traffic.

Confirmed the actual damage, not just the theoretical gap:
- **Empirical**: sampled `verb_runtime_intake` log lines across a rolling 1-hour window on both `legacy` and `chat` containers. **16/16 `request_id`s appeared in both.** Every one.
- **Traced one real example end to end**: `request_id=dd71496e-a2fe-45f2-ad50-9eb9d58353c1` (`introspect_spark`, spark-introspector's post-turn cognition RPC) — both `legacy` and `chat` independently ran the full plan and both called the LLM gateway. `legacy`'s log shows a real, separate completion: `completion_tokens=56`. Not a cache hit, not a log duplicate — a second, real, paid inference call.
- **Why it's invisible to users**: `orchestrator.py::_wait_for_result()` returns on the first reply matching `request_id` and exits its `bus.subscribe()` context (unsubscribing). The second (slower) container's result publish lands on a channel with no active subscriber and is silently dropped. No error, no duplicate answer — just wasted compute on every affected turn.
- **The design doc for this exact program names this failure mode explicitly** (`docs/superpowers/specs/2026-05-13-spark-introspection-lane-isolation-design.md`, Phase 2 section): *"**Constraint until Phase 4:** at most one consumer per PubSub lane to avoid duplicate exec."* Violated by `legacy` and `chat` both matching the shared-listener condition.
- Separately confirmed `orion-cortex-exec-spark` had processed **zero requests** in a 30+ minute sampling window prior to this fix — real, deployed, completely idle infrastructure.

**What this fix does and does not cover**: `use_direct_exec` explicitly requires `lane != "chat"`, so chat-lane traffic is structurally excluded from this flag's effect in either direction — it always used, and still uses, the shared broadcast channel. This fix stops duplication for the *other* lanes only (`spark`, `background`, and anything resolving to those). The classic (non-unified-turn) `chat_general` path would be duplicated by the same broadcast mechanism, but that is not something this flag can fix by design, and current live traffic volume on that specific path (vs. unified-turn's `orion_voice_finalize`, which is not duplicated — confirmed, it uses direct single-consumer RPC via `orion-harness-governor`) was not measured in this pass.

## Architecture touched

`orion-cortex-orch` only — settings default, `.env_example`, `docker-compose.yml`, README. No changes to `orion-cortex-exec`, the `Hunter`/`Rabbit` bus chassis, or any reducer/schema.

## Files changed

- `services/orion-cortex-orch/app/settings.py` — `exec_lane_routing_enabled` default `False` → `True`.
- `services/orion-cortex-orch/.env_example` — flag flipped, comment rewritten to explain why (the duplicate-exec finding, not just "Phase 2 opt-in").
- `services/orion-cortex-orch/docker-compose.yml` — inline `${EXEC_LANE_ROUTING_ENABLED:-false}` → `:-true`.
- `services/orion-cortex-orch/README.md` — new row in the Environment Variables table.
- **`services/orion-cortex-orch/.env` (live, gitignored) — flipped directly in the same session this was found**, ahead of this PR, so the running system stopped duplicating exec immediately rather than waiting for a deploy cycle. This PR makes that change durable in code/`.env_example` rather than leaving it as an unrecorded manual edit that a future `.env` resync could silently revert.

## Schema / bus / API changes

None. Pure routing-decision default flip; no new channel, no new schema.

## Env/config changes

- Changed default: `EXEC_LANE_ROUTING_ENABLED` `false` → `true` (orion-cortex-orch).
- `.env_example` updated: yes.
- Local `.env` synced: **yes, live**, done directly and immediately as part of the investigation (see above) — this is the one case this session where the live `.env` was edited ahead of the code PR rather than left as a documented operator follow-up, because the finding (real ongoing duplicate compute cost) warranted stopping it immediately rather than waiting.
- Skipped keys: none.

## Tests run

```
cd /mnt/scripts/Orion-Sapienform-exec-lane-routing-on
/tmp/orion-test-venv/bin/python -m pytest services/orion-cortex-orch/tests/test_lane_routing.py services/orion-cortex-orch/tests/test_execution_lanes.py services/orion-cortex-orch/tests/test_mind_skip_logging.py services/orion-cortex-orch/tests/test_route_grammar_emit.py -q
→ 26 passed, 1 failed
```
The one failure (`test_call_verb_runtime_sets_mind_requested_when_mind_enabled_true`, `AttributeError: 'SimpleNamespace' object has no attribute 'trigger'`) confirmed pre-existing and identical via `git stash` against unmodified `main` — unrelated to this change. All tests that reference `exec_lane_routing_enabled` construct it explicitly (`True`/`False`) rather than relying on the bare default, so the default flip doesn't change any test's behavior.

## Evals run

Not applicable — routing-config default only.

## Docker/build/smoke checks

Live verification already in progress on the deployed system (flag flipped in `.env` ahead of this PR). Restart required to fully pick up the code-level default (already live via the direct `.env` edit, so this is really about keeping future redeploys consistent — see Restart required).

## Review findings fixed

None from an external pass; the investigation itself is the review — traced the mechanism precisely (broadcast pub/sub vs. competing-consumer queue), confirmed real duplicate LLM cost with a concrete request_id trace rather than asserting it from code reading alone, and checked the design doc's own stated constraint before proposing the fix.

## Restart required

Already applied live via direct `.env` edit (`EXEC_LANE_ROUTING_ENABLED=true`) in this session, ahead of this PR. Once this PR merges, a normal redeploy will pick up the same value from the code-level default:
```bash
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d --build
```
Verification:
```bash
docker exec orion-athena-cortex-orch env | grep EXEC_LANE_ROUTING_ENABLED
# Should show: EXEC_LANE_ROUTING_ENABLED=true

# After a spark-introspection cycle or dream_cycle/journal turn, confirm no more overlap:
docker logs orion-athena-cortex-exec-spark --since 10m | grep -c verb_runtime_intake   # should be >0 now
docker logs orion-athena-cortex-exec --since 10m | grep -oP "request_id=\S+" | sort > /tmp/legacy_ids.txt
docker logs orion-athena-cortex-exec-chat --since 10m | grep -oP "request_id=\S+" | sort > /tmp/chat_ids.txt
comm -12 /tmp/legacy_ids.txt /tmp/chat_ids.txt | wc -l   # should trend toward 0 for non-chat verbs
```

## Risks / concerns

- Severity: low. Not yet verified whether `introspect_spark`'s previously-duplicated run also wrote a duplicate persisted row somewhere downstream (spark candidate, tissue update) versus being purely wasted compute with no storage-side duplication. This fix stops the *duplication*, so the question is now moot going forward, but it means the exact scope of pre-existing side effects from the ~2-month period this was live is not fully characterized. Not investigated further in this pass.
- Severity: low. Classic (non-unified-turn) `chat_general` traffic, if it still runs at meaningful volume, remains duplicated by the same broadcast mechanism -- structurally unfixable by this flag (`use_direct_exec` explicitly excludes `lane == "chat"`). Current volume on that path vs. unified-turn was not measured. Worth a follow-up if `client_mode != "orion"` traffic turns out to be non-negligible.
- Severity: none. Chat-lane behavior (the primary, highest-stakes path) is provably unaffected by this flag in either direction -- confirmed via the `lane != "chat"` gate before making the live change, not asserted after the fact.

## PR link

Push and open via: `git push -u origin feat/exec-lane-routing-default-on`, then open the compare URL GitHub prints (no `gh` auth in this environment).
