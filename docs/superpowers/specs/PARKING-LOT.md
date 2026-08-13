# Parking lot

Real findings that are **not** the current phase. One line, a date, a pointer.

Rule (`2026-08-13-scarcity-and-repertoire-execution-plan.md` §7.2): anything discovered mid-phase
that is not the phase lands here, **not in the branch**. This file exists because every entry
below is genuinely interesting and every one of them would have been a squirrel.

Nothing here is scheduled. Nothing here is abandoned. Adding a line costs one line.

---

## 2026-08-13

- **Arena urgency degeneracy.** `proposal_urgency()` (`orion/proposals/scoring.py:268`) falls back
  to `max()` over all four `PRESSURE_DIMENSIONS` for any template declaring `dimensions: {}`. Five
  of thirteen do. Result: 7.03 of 10 candidates tied at the frame max, `corr(urgency) = 1.000000`
  for 10 of 28 template pairs. Rank order is decided entirely by hand-authored `base_priority`.
  Inverted, too — declaring the dimension you care about is a strict handicap. → Plan Phase 5.

- **The feedback loop is open.** 302,974 `substrate_feedback_frames` rows. Nothing in
  `orion/proposals/` or `orion-proposal-runtime` reads any of them. `orion/reverie/efficacy.py`
  (the "was that action useful" module) has **zero live callers**. → Plan Phase 4.

- **Feedback `score` carries no information.** Exactly one distinct value per `outcome_kind`
  (0.25 / 0.4 / 0.55 / 0.85 / 0.1). It is a lookup table on dispatch status, not a measurement.

- **Outcome deltas are tick-attributed, not action-attributed.** `orion/feedback/builder.py:272`
  keys `improved`/`worsened`/`unchanged` on `field_after.tick_id`. With 5 concurrent dispatches
  per tick this is unattributable to any action. `builder_prune`'s own comment already says so.

- **`proposed_effect` is a category label, not a prediction.** Three coarse values across 13
  templates (`increase_observability` ×8, `preserve_stability` ×4, `prepare_for_policy_gate` ×1).
  Used correctly for policy gating; not usable as a falsifiable claim.

- **46.6% of policy frames carry zero decisions.** Investigated: this is the `action_warrant` tick
  gate working correctly (clean split at 0.5), i.e. program outcome O1. **Not a defect.** Recorded
  so nobody "fixes" it again — I nearly did.

- **`search_web` is synthetic.** `Simulate a web search via LLM using known memory + synthetic
  results`. Not world contact. Anything built on it launders invention as perception.

- **Orion cannot reach the repo from the exec lane.** `orion-athena-cortex-exec-background` mounts
  the repo read-only, git fails with `detected dubious ownership`, and the 75 sibling worktrees at
  `/mnt/scripts/Orion-Sapienform-*` are outside the mount entirely. Any repo-touching skill needs
  a broader mount + rw + a git config change — three privilege expansions. Not taken.

- **`image_prune` is built, tested, and unrouted.** The only existing path to Orion's first
  `acted=true` in 88,409 dispatches. One-line config change whenever wanted. → Plan §8.

- **Pre-existing, unrelated:** `check_service_env_compose_parity orion-execution-dispatch-runtime`
  reports 3 of 24 `.env_example` keys missing from `docker-compose.yml`
  (`EXECUTION_DISPATCH_STALENESS_MIN_SEC`, `_MAX_SEC`, `EXECUTION_DISPATCH_RUNTIME_PORT`).
  Identical on `main`.

- **Pre-existing, unrelated:** `services/orion-execution-dispatch-runtime/tests/test_heartbeat_chassis.py`
  fails collection with `FileNotFoundError` on `main` too.

- **Carried from the prior spec's §2.3, still open:** `action_warrant.py:280` `pinned` guard
  requires `zscore == 0.0` exactly (channel median |z| is 0.0007); `node:prometheus` reads exactly
  0.0 on all 74,264 ticks (dead producer, still consumed); `node:circe` bottoms at 3e-323 (decay
  artifact); `template_match_score` provably dead for all 13 templates; `prune_build_cache` has no
  cooldown; "7 dimensions" prose stale in `orion/field/pressure.py:14,249,259-267`.

## 2026-08-13 (Phase E0)

- **`counterfactual` and `context_exec_memory_contradiction_review` are dead.** Both return an
  empty string in ~0.5 s while reporting `status=success`, on all 10 runs. Never executed before
  E0, so nobody knew. Empty-shell-cognition reporting success. NOT fixed — E0 hit a kill gate and
  stopped.

- **Recall dominates supplied context.** Ten `goal_formulate` runs across ten different field
  ticks, each given Orion's real live pressure readings as the explicit `intention`, all returned
  paraphrases of the same recalled Juniper coding session. Possibly the largest blocker in the
  system for any cognitive-verb work: routed into the arena, these verbs would narrate session
  history rather than Orion's condition. Deserves its own measurement.

- **`goal_formulate` is a translator, not a generator.** Its prompt reads
  `{{ intention or text or request }}` — it structures a *supplied* intention. It cannot produce
  one from state. Any verb sharing that prompt shape has the same limit.

- **Verb names cannot be trusted as capability descriptions.** Assumed `goal_formulate` formulated
  goals. It does not. Check the prompt template before routing any verb.
