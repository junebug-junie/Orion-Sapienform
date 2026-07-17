# Concept Induction: read chat-history text from Postgres, not the bus envelope

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1162
Branch: `fix/concept-induction-pg-chat-source`

## Trigger

Juniper reported chat-stance artifacts bleeding into "the new orion concept induction" —
concrete example seen: a graph node reading something like "speak as an ongoing presence
rather than customer support software."

## Investigation

Traced the full chain in code before touching anything:

1. `orion:chat:history:log` intake (`orion/spark/concept_induction/bus_worker.py`,
   `_extract_text`) merged the bus envelope's own `prompt`+`response` fields verbatim.
2. Orion's live replies can echo stance-scaffold vocabulary — phrases like "ongoing
   cognitive presence" and hazard labels such as `avoid_customer_support_tone` live in
   `services/orion-cortex-exec/app/chat_stance.py`'s identity-kernel fallbacks and hazard
   vocabulary. Existing mitigations (`strip_identity_recital_leadin`,
   `suppress_chat_general_speech_identity_priming`) only strip a *leading* recital
   sentence, not vocabulary appearing mid-reply.
3. `SpacyConceptExtractor` (`orion/spark/concept_induction/extractor.py`) has zero
   stopword/denylist filtering — confirmed by grep, and independently confirmed by an
   earlier 2026-07-11 audit recorded in memory (`project_concept_induction_deactivation_decisions`).
4. When `USE_CORTEX_ORCH=true`, the refinement LLM (`concept_induction_prompt.j2`) is told
   to freely "merge/rename concepts into clear, human-readable labels" with no guardrail
   against synthesizing an instructional-sounding sentence out of leaked vocabulary.

Presented this diagnosis plus three options (ingestion-side filter, filter + chase the
upstream reply leak, or diagnose-only). Juniper's actual instruction was a fourth option
not on that list: replace the bus-envelope read with a direct Postgres
`chat_history_log` lookup, on the belief that the persisted row is cleaner.

## What shipped

- `orion/spark/concept_induction/chat_history_pg.py` (new): pooled `asyncpg` lookup of
  `chat_history_log` by `correlation_id`, bounded connect/query timeouts, retried on both
  a miss (SQL-writer commit race) and a transient connection error, fail-open to the old
  envelope-derived path.
- `bus_worker.py`: `_extract_text` (now async) tries the Postgres row first for the hub
  chat-history channels (`orion:chat:history:log`, `orion:chat:history:turn`); old body
  preserved as `_extract_text_from_envelope`. Pool closed on worker shutdown.
  `chat_pg_lookup_stats` (hit/miss/skipped) surfaced via the existing
  `/debug/concept-induction` endpoint.
- `settings.py` / `.env_example` / `.env` / `docker-compose.yml`: `CONCEPT_CHAT_PG_LOOKUP_ENABLED`,
  `CONCEPT_CHAT_PG_DSN`, `CONCEPT_CHAT_PG_LOOKUP_RETRIES`, `CONCEPT_CHAT_PG_LOOKUP_RETRY_DELAY_SEC`.
- Tests: 184 passing (existing + new coverage for the pooled lookup and the
  extraction-path branching).

## Important: this does not close the reported symptom by itself

Verified directly against `services/orion-sql-writer/app/worker.py`: `prompt`/`response`
are written to Postgres straight from the same bus-published payload, with no cleaning
step. **The Postgres row and the bus envelope carry identical text.** If a reply contains
leaked stance vocabulary, it's in both sources equally — swapping the read source doesn't
strip it.

What this patch *does* deliver on its own merits: one canonical, already-committed text
per turn (no partial/mid-write reads, no duplicate extraction across message-level vs
turn-level bus events for the same turn) instead of trusting whatever shape a given
intake channel's envelope happens to carry.

The layer that would actually need to change to stop leaked vocabulary from becoming a
concept-graph node is `extractor.py`'s total lack of stopword/denylist filtering (option
A from the original diagnosis, not taken). This is documented explicitly in
`docs/concept_induction.md` and in a `settings.py` comment so the patch isn't mistaken
for more than it is.

## Review findings fixed

A 3-angle code-review pass (correctness, cleanup, altitude/conventions) via subagent
found real issues in the first draft, all fixed before shipping:

- Broad `except Exception: return None` in the retry loop meant a transient connection
  error aborted immediately instead of retrying — only the "row not found yet" case
  actually got retried. Fixed: retries now cover both paths uniformly.
- No timeout on `asyncpg.connect()`/`fetchrow()` — worst case ~60s × up to 3 attempts
  blocking the worker's single sequential bus-consume loop (which also handles drives,
  tensions, homeostatic signals — not just chat). Fixed: `create_pool(timeout=3.0,
  command_timeout=3.0)`.
- Fresh TCP+auth connection opened and closed per chat turn, no pooling, despite 7 other
  services in this repo already using `asyncpg.create_pool`. Fixed: lazy DSN-keyed shared
  pool, closed on worker shutdown.
- No runtime evidence the Postgres path is actually exercised in production (only a
  debug-level log on exception) — violates this repo's "runtime truth beats config
  truth" convention. Fixed: `chat_pg_lookup_stats` counters surfaced via the debug
  endpoint.
- Altitude finding (the big one): risk of this PR being read as "fixed the stance-leak
  bug." Not fixed — documented instead (see section above).
- Fake configurability: a `CONCEPT_CHAT_PG_TABLE` setting existed alongside hardcoded SQL
  column names, so pointing it anywhere but `chat_history_log` would silently fail open
  with zero signal. Fixed: removed the setting, hardcoded consistently.

## Env parity

`.env_example` updated. The live `.env` lives only in the primary checkout
(`/mnt/scripts/Orion-Sapienform/services/orion-spark-concept-induction/.env`) since `.env`
is gitignored and worktrees don't carry it — `scripts/sync_local_env_from_example.py` run
from this worktree correctly found nothing to sync there. Edited the primary checkout's
`.env` directly with the matching new keys. `check_service_env_compose_parity.py` reports
N/A (service declares `env_file:`, so all keys reach the container regardless of the
compose `environment:` list).

## Live effect

None yet: `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` as of the 2026-07-11 decision
(unrelated to this PR, not touched here) — the concept-profile/clustering trigger this
data feeds is currently disabled. This PR improves the data source for whenever that
trigger is re-enabled.

## Status

DONE_WITH_CONCERNS — implementation complete, tests pass, review findings fixed, branch
pushed, PR open. Concern: this does not close the originally-reported symptom (stance
artifacts in the concept graph); that requires a follow-up in `extractor.py` or upstream
in the chat-reply path, not attempted here per Juniper's specific instruction.
