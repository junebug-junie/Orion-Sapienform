## Summary

- Adds one sentence to the always-rendered "How to read the Situation block" note (`orion/harness/situation_brief.py`): a Place line is standing ground truth, and spatial claims (what a camera sees, where Orion is) get checked against it. A city Juniper mentions is where she is, not where Orion is.
- Follow-up to #2493, which put `home_location`/`physical_location` in the block but left the existing note saying situation context is optional ("not an instruction to mention").

## Outcome moved

2026-10-05, corr `5063fb71-471c-46e2-860d-8415f9253e14`: Orion described a home camera as watching "Chicago from your hotel window". A fact in the prompt was not enough on its own; nothing told Orion to check their own spatial claims against it.

## Current architecture

`append_situation_block_harness_brief` renders the note once per compiled prefix whenever a situation fragment exists, outside the fragment's char cap.

## Architecture touched

`orion/harness/situation_brief.py`, `orion/harness/tests/test_harness_prefix.py`. No schema, bus, env or compose change.

## Files changed

- `orion/harness/situation_brief.py`: new note item (conditional on a Place line existing).
- `orion/harness/tests/test_harness_prefix.py`: asserts the sentence renders once with the fragment.

## Schema / bus / API changes

None.

## Env/config changes

None. No `.env_example` change, nothing to sync.

## Tests run

```text
orion/harness/tests + orion/schemas/tests/test_context_provenance.py: see below
orion/harness/tests/test_harness_prefix.py + orion/situational/tests: 136 passed
```

## Evals run

None. UNVERIFIED whether one sentence is enough. Planned check: replay corr 5063fb71 (same history and recall) with and without the sentence.

## Docker/build/smoke checks

Not run. Prompt-text change; takes effect when the Hub (which compiles the harness prefix) is recreated.

## Review findings fixed

(see below)

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --force-recreate orion-hub
```

## Risks / concerns

- Low: ~330 chars added per turn that has a situation block, outside the fragment cap.
- Low: could make Orion over-cite Place on spatial questions; the existing "not an instruction to mention" line is kept.

## PR link

TBD
