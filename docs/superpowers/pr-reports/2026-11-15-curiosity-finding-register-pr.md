## Summary

- After every finished curiosity turn, Hub now labels the finding and scores how new it is to Orion, one row in `curiosity_finding_register`. This is step 2 of curiosity event wants.
- The judge is the curiosity lane itself (Juniper, 2026-11-15). A label counts only when Flash-Next served it, because that model passed the agreement eval.
- Novelty = 1 - nearest cosine against everything Orion wrote before the run: graph findings and answers, prior claims, Day letters. Embeddings are cached per embedding model in `curiosity_novelty_corpus`.
- Brings in the judge-agreement eval from draft #3222 (rebased; #3222 is closed as superseded) and records the Flash-Next result.
- Flag `HUB_CURIOSITY_FINDING_REGISTER_ENABLED`, default ON.

## Outcome moved

Before: nothing said which curiosity findings are philosophical, or new, so curiosity could not compete for attention. After: every finished run gets a label and a novelty score. Step 3 (event-want producer) can read it.

## Current architecture

Curiosity turns end in Hub (`curiosity_investigation.py`), which writes the outcome row and runs the fit hop. No judge ran on live findings: no local non-writer lane passed the eval (best kappa 0.60).

## Architecture touched

- orion-hub: new background writer after the turn ends, on the three turn-end paths.
- Postgres: two new tables (additive DDL, post-pull auto-apply allowlisted).
- orion-llm-gateway: called over the existing bus intake channel (route `agent`); no change to it.
- vector-host `/embedding`: called; no change.

## Files changed

- `orion/curiosity/register.py`: passed-model check, corpus Cypher, corpus item/selection helpers.
- `services/orion-hub/scripts/curiosity_finding_register.py` (new): judge call, corpus sync, novelty, retries, insert.
- `services/orion-hub/scripts/curiosity_investigation.py`: schedule the register task at turn end; per-run in-flight guard.
- `services/orion-hub/app/settings.py`, `scripts/main.py`, `.env_example`, `README.md`: the flag and wiring.
- `services/orion-sql-db/manual_migration_curiosity_finding_register_v1.sql` (new).
- `scripts/analysis/eval_curiosity_register_judge.py`: reply cap 160 -> 400, Flash-Next result.
- `docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md`: judge decision recorded.
- Tests: `services/orion-hub/tests/test_curiosity_finding_register.py` (new), `orion/curiosity/tests/test_register.py`.

## Schema / bus / API changes

- Added: tables `curiosity_finding_register`, `curiosity_novelty_corpus`.
- Removed / Renamed: none.
- Behavior changed: one extra LLM call (~12/day) on the `agent` route after each curiosity turn, plus embeddings (first run embeds ~1,600 items once).
- Compatibility notes: no bus channel or schema registry change; uses `llm.chat.request` on the existing intake channel.

## Env/config changes

- Added keys: `HUB_CURIOSITY_FINDING_REGISTER_ENABLED=true` (orion-hub). Reuses `HUB_CURIOSITY_SEARCH_EMBED_URL`, `CHANNEL_LLM_INTAKE`, `LLM_GATEWAY_SERVICE_NAME`, `HUB_CURIOSITY_INVESTIGATION_LLM_ROUTE`.
- Removed / Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py orion-hub`: yes.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-hub/tests/test_curiosity_finding_register.py orion/curiosity/tests/test_register.py -> 50 passed
pytest services/orion-hub/tests/test_curiosity_investigation.py test_curiosity_spend_log.py test_curiosity_self_inquiry_outcomes_pg.py -> 179 passed, 1 skipped
scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
eval_curiosity_register_judge.py, Flash-Next (agent seat), 263 reference runs, cap 400:
three-way kappa 0.734 (gate 0.8 FAIL); philosophical-or-not kappa 0.856 (n=251, PASS);
phil-agree 0.87 (gate 0.90 FAIL); winners 5/6 (91241b6d: every lane says operational); 12 not_json.
Juniper accepted Flash-Next on 2026-11-15 knowing about 1 in 8 philosophical findings read operational.
Not run: the dressing eval (rewrite operational findings reflectively; <= 10% may flip). Ticketed.
```

## Docker/build/smoke checks

```text
Live smoke (no writes): judge_finding over the real bus, route agent ->
  philosophical answer: register=philosophical, model=qwen3.8-flash-next-iq3_s
  plumbing answer:     register=operational,   model=qwen3.8-flash-next-iq3_s
vector-host /embedding -> embedding_model=BAAI/bge-large-en-v1.5, 1024 dims
Hub not rebuilt from this branch (deploys only from main).
```

## Review findings fixed

- Finding: one transient judge/embed failure left a run unlabeled forever; gateway refusals read as "wrong model".
  - Fix: refusals and empty replies are `call:` errors; transient failures retried twice (2 min, 10 min) before the row is written.
  - Evidence: `test_a_transient_judge_failure_is_retried_before_the_row_is_written`, `test_a_gateway_refusal_is_a_call_error_not_a_wrong_model`.
- Finding: cached vectors not tied to the embedding model; a size mismatch was skipped, giving novelty on a partial corpus.
  - Fix: cache keyed by vector-host's reported `embedding_model`; any size mismatch is `dimension_mismatch`.
  - Evidence: two tests; live reply carries `BAAI/bge-large-en-v1.5`.
- Finding: Findings without run_id silently dropped from the corpus.
  - Fix: Findings keyed by `finding_id` (live: 1 of 658 lacks run_id; all have finding_id).
- Finding: failed embed left sibling tasks running; Postgres connections unbounded on first sync.
  - Fix: batches of 16 with `return_exceptions`, one connection + `executemany` per batch.
- Finding: cosine scan blocked the event loop.
  - Fix: `asyncio.to_thread`.
- Finding: two attempts of one run could both be judged.
  - Fix: per-run in-flight guard; early exists check.
- Not fixed (low): anatomy runs with no SelfDefinition get "(question not recorded)" -- the turn request does not carry the question family, and such a run has no answer to judge. A cap-truncated reply reads `not_json`; the live smoke shows thinking is off on this route.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

The migration auto-applies on the post-pull rebuild (additive DDL, REQUIRED-BY orion-hub); the deploy wrapper refuses Hub until it is applied.

## Risks / concerns

- Severity: medium. Concern: the judge shares the `agent` route with curiosity and other agent work; under load a label waits or fails after retries. Mitigation: retries, failures recorded as `call:` errors, never as labels.
- Severity: medium. Concern: Orion could learn to dress plumbing findings as philosophical. Mitigation: insert-only labels, fixed hashed prompt; dressing eval still owed (ticket).
- Severity: low. Concern: the first run after deploy embeds ~1,600 items (budget 600 s); if it times out, that run's novelty is NULL and the next run finishes the cache.
- Issues: closes #3372 (judge decision). Opens: dressing eval ticket, live-verify ticket.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
