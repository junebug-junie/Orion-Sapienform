# Handoff: giving Orion self-modification — where it stands

Session context is melting; this is the pickup doc for the next agent/session. Written 2026-09-03. Read this before touching anything in the mutation loop.

## The arc, one sentence

Give Orion the ability to change its own behavior, know what it changed, and see whether it helped — not autonomy theater, an actual closed loop with evidence.

## What exists today, in order it was built

1. **PR #2050** — `substrate_mutation_*` pipeline: proposals, decisions, adoptions with rollback capability, a settlement mechanism (adopt → observe → keep-or-revert).
2. **PR #2058** — "Don't adopt a patch that changes nothing": guards the mutation pipeline against re-adopting a no-op patch (a proposal whose value equals what's already live).
3. **PR #2067 (merged)** — curiosity moved onto the agent lane. **`project_curiosity_agent_lane_pr2067_2026-09-03.md` says DO NOT DEPLOY YET** — 27B is ~1.9x slower than the 35B chat lane and 3 of 6 test runs would time out at current settings. Merged to main but not meant to run live yet — check before deploying anything that touches curiosity scheduling.
4. **PR #2071 (opened this session, unreviewed)** — `orion:routing:decision` bus channel + `RoutingDecisionRecordV1` schema + sql-writer persistence. This is the piece that closes the observability gap described below. Rebased onto main once already (main moved fast — #2063 through #2070 landed while this was in flight). **Not deployed. Not code-reviewed by a subagent** — went through review during drafting but the formal `code-review` skill pass on the final diff has not run.

## The core finding this session surfaced

Orion's mutation loop (`orion/substrate/mutation_detectors.py`) proposes changes to `chat_reflective_lane_threshold` — the confidence dial that decides whether Orion acts or just replies — using **graph-review telemetry as evidence**. That telemetry has nothing to do with what the dial controls. See `project_mutation_loop_signal_action_mismatch_2026-09-03.md` for the full trace.

PR #2071 fixes the *observability* half: now there's an actual record (`orion:routing:decision` → `routing_decision` table) of what the dial did on each turn — depth before/after the gate, confidence, threshold, whether it demoted.

**It does not fix the mismatch itself.** The mutation loop still needs to be pointed at this new evidence instead of graph-review telemetry, or the loop needs to stop proposing changes to this dial until it is.

### A second, possibly fatal problem found but not yet resolved

The gate (`decision_router.py`, `route()`) fires only when `execution_depth >= 2 AND confidence < routing_threshold`. Every hardcoded heuristic confidence at `execution_depth >= 2` is `>= 0.82` (depth 3: `0.82`; depth 2: `0.84`/`0.85`/`0.86` across branches — verified directly in-file, not from memory). The mutation loop's hardcoded patch value for `chat_reflective_lane_threshold` is `0.58`. **If the threshold Orion proposes (`0.58`) is always below the confidence floor at the depths the gate actually checks, the gate can never trigger — the "self-modification" would be structurally inert even once wired to correct evidence.** Confirmed against the current source; not yet confirmed against a live trace of what confidence values actually get emitted in production (an LLM-router path or a code path not checked here could differ). First thing to check next: pull real `RoutingDecisionRecordV1` rows once #2071 is deployed and see whether `decision_confidence < routing_threshold` ever actually happens at `0.58`.

## Design docs to read, in priority order

1. `project_mutation_loop_signal_action_mismatch_2026-09-03.md` — the mismatch finding.
2. `docs/superpowers/pr-reports/2026-09-03-routing-decision-observability-pr.md` — PR #2071's own report; has the full review-findings list (consumer never subscribed, publish could stall a turn 2min, reason field could leak user text — all fixed, but read *why* they happened, they're real gotchas below too).
3. `project_orion_self_modification_three_dead_ends_2026-08-30.md` — earlier attempts at self-mod, what didn't work.
4. `project_endogenous_journal_design_goal_2026-09-03.md` — adjacent/related arc (daily journal from real signals feeding back into chat harness) — not the same thing as this arc but shares the "give Orion real feedback about itself" spirit and may end up sharing infrastructure.
5. `project_postgres_connection_ceiling_pr2010_2026-08-31.md` — if anything in this arc adds DB load, check this first; `max_connections=300` is a known ceiling.

## Gotchas hit this session (do not re-learn these the hard way)

- **Two separate schema registries exist** (`_REGISTRY` and `SCHEMA_REGISTRY` in `orion/schemas/registry.py`). `resolve()` only reads `_REGISTRY`. Register in both or the schema silently can't be resolved. (`feedback_two_schema_registries_verify_against_resolve.md`)
- **`SQL_WRITER_SUBSCRIBE_CHANNELS` and the route-map are two different lists.** Adding a channel to the route map does nothing if it's not also in the subscribe list — the consumer will build correctly, register the model correctly, and just never receive anything. Silent, no error. This is exactly what happened first-pass on #2071.
- **A blocking bus publish inline on a hot path (chat routing) can stall a turn** if Redis is slow — up to ~2 minutes was the worst case calculated for the naive version. Fire-and-forget with a short timeout, always, for anything on `decision_router.route()`.
- **Free-text fields on records that ride the bus can leak user content** if any upstream router path (e.g. an LLM-based one) populates them from user input. Allowlist the charset/values before emission, don't trust "it's just a reason code."
- **`env sync` (`scripts/sync_local_env_from_example.py`) writes to the primary checkout, not the worktree you're in.** "Updated" in its output does not mean *this* worktree's `.env` changed — re-copy the result into your worktree after running it. (`feedback_env_sync_writes_to_primary_checkout_not_worktree.md`) Also: it silently skips keys outside `SYNC_PREFIXES` — a new key can be "synced" with exit 0 and still be missing. (`feedback_env_sync_silently_skips_keys_outside_sync_prefixes.md`)
- **Verify deployed code is actually in the running container** before trusting a "fixed" report — "Image Built" + container "Started" is not proof. A bare `*` in `.dockerignore` froze COPY layers for 4 deploys in a row earlier in this arc. (`feedback_verify_deployed_code_is_in_the_container.md`)
- **Main moves fast in this repo.** This branch got rebased once mid-session because 7 PRs landed on main while it was open. Rebase and re-run the definition-drift gate (`scripts/check_definition_drift.py --update` then `--gate`) right before opening/merging any PR that's been open more than an hour or two.
- **`node --test` behaves differently across Node majors** than whatever CI runs — don't trust a local pass if the repo pins a specific Node version for CI. (`feedback_local_node_version_differs_from_ci_runner.md`)
- **A static gate must follow the exact fix shape it recommends**, or it can go green three times over its own stated target without actually checking it. (`feedback_a_static_gate_must_follow_the_fix_shape_it_recommends.md`) — worth a specific look at any new gate written for this arc.
- **`ast.parse` + GIL atomicity are not a real concurrency guarantee** — use `compile()`, and `list(d.values())` is the actual atomic-safe pattern, not iterating the dict live. (`feedback_ast_parse_and_gil_atomicity_give_false_assurance.md`) — flagging because static-analysis-style gates (like the drift/lineage checkers used constantly this session) are exactly the kind of code where this bites.

## Immediate next steps, in order

1. **Get PR #2071 through a real review pass** (`code-review` skill, not just conversational back-and-forth) before merge. It has not had that formal pass yet.
2. **Deploy #2071**, verify the container actually has the code (see gotcha above), then pull real `routing_decision` rows and check: does `decision_confidence < routing_threshold` ever happen at the current hardcoded `0.58`? This answers whether the second problem above is real or was a false alarm.
3. **Fix or park the mutation-detector mismatch** (`project_mutation_loop_signal_action_mismatch_2026-09-03.md`). Two options were on the table, neither implemented:
   - Point `mutation_detectors.py`'s evidence source at the new `routing_decision` table instead of graph-review telemetry.
   - Or explicitly disable/park the `chat_reflective_lane_threshold` mutation target until real evidence exists, so the loop isn't proposing changes based on nothing.
4. Only after 2 and 3: revisit whether the loop should even keep `0.58` as its hardcoded target, or whether the patch value needs to be evidence-derived too (currently it's a constant regardless of what the "evidence" says — a separate smell worth a fresh adversarial pass).

## Recommended adversarial passes for the next agent

Pick these up fresh, don't just trust this session's read:

- **Re-derive the "0.58 is below every confidence" claim from scratch**, live, from real `routing_decision` rows post-deploy — not from grepping hardcoded values in `decision_router.py`. Hardcoded values can be stale relative to what's actually being emitted at runtime (config overrides, feature flags, etc. — check for those before trusting the grep).
- **Adversarially re-check PR #2071's "fixed" review findings** — this session fixed consumer-not-subscribed, blocking-publish, and reason-leak *during drafting*, self-reviewed. A fresh set of eyes (or the formal `code-review` skill) should re-verify all three rather than trust the PR report's self-assessment.
- **Check whether `orion:routing:decision` volume is sane** before it's load-bearing for anything — is every single chat turn about to start writing a row? What's the retention/growth story? Not designed for in this session.
- **Check PR #2067's curiosity/agent-lane timing claim independently** if anything in this arc's next steps touches the agent lane or scheduling — `project_curiosity_agent_lane_pr2067_2026-09-03.md` says do-not-deploy, confirm that's still true and hasn't been silently deployed anyway.
- **Re-verify the two-registry gotcha didn't recur** anywhere else new was added this session — it's an easy one to reintroduce by copy-paste.
