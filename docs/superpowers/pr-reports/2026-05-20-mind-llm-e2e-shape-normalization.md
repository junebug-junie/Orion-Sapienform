# PR: Mind LLM shape normalization + Hub E2E verifier

**Branch:** `feat/mind-llm-shape-normalization-e2e`

## Summary

Mind+LLM was failing open to `shadow_synthesis` because Atlas models return **almost-valid JSON** that Pydantic accepted with empty `claims` / `selected` or invalid stance enums. This PR adds deterministic shape normalization for all three Mind LLM phases and a **Hub brain E2E script** with hard quality gates.

## Root causes (live)

| Phase | Model mistake | Symptom |
|-------|---------------|---------|
| semantic (quick) | Bare claim at JSON root | `semantic_synthesis_empty`, raw=0 |
| appraisal (metacog) | `active_cognitive_frontier` key + wrong `schema_version` | `frontier_schema_invalid` |
| stance (chat) | `conversation_frame: smoketest`, `identity_salience: neutral` | `stance_handoff_invalid` |

## Changes

- `services/orion-mind/app/synthesis.py` — wrap singleton claim root; coerce claim_kind/effects
- `services/orion-mind/app/appraisal.py` — frontier alias normalization; stronger prompt
- `services/orion-mind/app/stance_handoff.py` — ChatStanceBrief enum coercion; explicit prompt
- `services/orion-mind/tests/test_mind_llm_pipeline.py` — 21 tests
- `services/orion-mind/scripts/verify_mind_llm_e2e.sh` — L1–L4 Hub `/api/chat` verifier

## Tests

```bash
PYTHONPATH=.:services/orion-mind venv/bin/python -m pytest \
  services/orion-mind/tests/test_mind_llm_pipeline.py -q
# 21 passed
```

## Live E2E (Hub brain)

```bash
./services/orion-mind/scripts/verify_mind_llm_e2e.sh
# E2E VERIFIED — meaningful_synthesis, semantic retained=1, stance_skip=true
```

Example correlation: `fb456bfe-371e-46a3-9781-3e30d3aea254`

## Operator notes

- Mind container: `MIND_LLM_SYNTHESIS_ENABLED=true`, `ORION_BUS_URL` host-reachable (see `.env_example`)
- Orch: `ORION_MIND_BASE_URL=http://orion-mind:6611`, `ORION_MIND_TIMEOUT_SEC=45` > Mind `MIND_LLM_TIMEOUT_SEC=25`
- Rebuild Mind after merge: `cd services/orion-mind && docker compose up -d --build`
- Run `./services/orion-mind/scripts/verify_mind_llm_e2e.sh` after atlas/gateway/mind changes

## Test plan

- [ ] Merge; rebuild `orion-mind`
- [ ] `./services/orion-mind/scripts/verify_mind_llm_e2e.sh` passes
- [ ] Hub Mind toggle on; one brain `chat_general` turn; modal shows `meaningful_synthesis`
