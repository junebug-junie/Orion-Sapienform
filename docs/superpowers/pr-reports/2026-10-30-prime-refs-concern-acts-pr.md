# feat(prime): references contract + concerns as graded acts (Orion Prime step 1)

## Summary

- Prime can now point a curiosity run at an arc or an act in a way the run can actually open. A question carries a short list of references (`refs`). Before Prime sends it, Prime opens every reference with the same read the run will use. If one does not open, the question is not sent and Prime is told why.
- Curiosity runs got two new dig commands: `python -P -m orion.dig arc <id>` and `act <id>`. The run's focus section now lists each cited id with the exact command that opens it.
- A question that writes an id in its text but leaves it out of `refs` is refused. That is the live failure from 2026-10-30: Prime asked about "arc 8d87adeca2ce06a5", and the run had no way to open an arc.
- Every concern Prime holds is now an act on the ledger, graded on follow-through. It scores 1 when a question Prime sent about it gets a run that writes something. It scores 0 when Prime drops it, lets it fade, or holds it 6 days without that happening.
- Prime sees its own track record in every thought: of its last 20 concerns, how many settled, how many it dropped (and how many of those within an hour), how many faded or lapsed. Nothing forbids a drop; the record is the cost.

## Outcome moved

- **Failure mode closed:** a Prime question could cite an id the consumer could not open. Now it cannot leave Prime: a code check refuses it, and the frame eval replays the live 10-30 question to prove the refusal.
- **Behaviour now measured:** most concerns lived under an hour on 10-30, replaced each time attention moved. Each concern now leaves a graded act. Prime's frame shows the drop rate. The body economy's value for Prime's next thought falls with every abandoned concern (see "How the economy feels it").

## Current architecture

- Prime (orion-mind `app/prime_loop.py`, pure logic in `orion/prime/`) held up to 3 concerns in Redis `orion:prime:state`. Concerns had no act. A dropped or faded concern left only a line in the thought's close note.
- Prime's one move was a free-text question for Hub's curiosity focus slot. Any id in the text reached the run as plain words. The dig kit had no command for arcs or acts. orion-durable-runs served arcs only by day (`/temporal-self/arcs?day_id=`), and no service served one act by id.

## Architecture touched

- **New module `orion/refs.py`** (no service, no registry): the closed set of reference kinds (`arc`, `act`), one resolver each, the id shapes, and the id-looking-word check. Prime and the dig kit both call `orion.refs.resolve`.
- **orion-durable-runs:** two read-only by-id routes, `GET /temporal-self/arcs/{arc_id}` and `GET /acts/{act_id}`. Each checks the id shape before any query (422), and answers 404 when absent.
- **Dig kit:** `arc <id>` and `act <id>` commands.
- **Prime (orion-mind):** the move carries `refs`; the runtime resolves them before sending. Concerns get `predict` acts; a persisted concern record is both the track record and the outbox for act closes.
- **Hub + focus table:** `curiosity_focus_request.refs` (new nullable jsonb column). Hub stores the refs and the run's focus section prints them with their dig commands.
- **Contract:** `QueueCuriosityFocusArguments.refs` (optional list of `{kind, id}`).

## How it works, plainly

**References.** A reference is a kind plus an id. There are two kinds today. An `arc` is one stretch of Orion's attention on one subject (Temporal Self). An `act` is one row of the act ledger: what Orion did, what it expected, and how it closed. Each kind has exactly one way to open it, on orion-durable-runs. Prime opens every reference before it sends a question; a curiosity run opens it with the dig kit, which calls the same function. So a reference that Prime could send is one the run can open.

The id-word check: any word in the question or its "why" that looks like an id must be in `refs`. "Looks like an id" means a hex run of 12 or more characters that mixes digits and letters a-f (arc, concern, thought and run ids), an `act_` id, or a UUID (focus ids). Long plain numbers, money amounts and ordinary words do not match.

A refused move is shown to Prime in its next frame as an outcome line, not as news. Prime does not wake up again because of it, so a refusal cannot loop.

**Concerns as acts.** Holding a concern is a prediction: "acting on this will produce something". So each concern is a `predict` act. `predict` is the honest verb: the concern makes no call (`ask` is the question), and holding it is not just waiting (`wait` would grade patience; this grades follow-through).

- **Settled = 1:** a question Prime *sent* about this concern (the move's `concern_id`) got a curiosity run that wrote something, within 144 hours of the act opening. A held-back control question was never sent, so it cannot settle a concern.
- **0:** Prime dropped the concern, let it fade (not kept for a day), or held it past 144 hours unsettled ("lapsed").
- Settling or lapsing closes the act but leaves the concern in place. Prime decides whether to keep holding it.
- **Why 144 hours:** only one Prime question can be open at a time. A fresh concern may first wait out another concern's open question, then its own. One question can take up to 72 hours to settle: the 48-hour focus wait (measured: the investigate line's longest gap was 21.1 h over 30 days) plus the run's own 24-hour act horizon. Two of those is 144 hours. An unkept concern fades after 24 hours, long before this, so the horizon only binds concerns Prime keeps. Live, Prime had asked 2 questions in total, too few to measure its own settle time; re-derive once there are more.

**Track record.** It lives in `orion:prime:state` next to the concerns, so it survives a restart. It resets only if the concerns themselves are lost, which keeps the two consistent. The ledger was the other choice; orion-mind has no database access, and reading it would mean a new read path for one line of text.

**Exactly one close per concern.** The record is also an outbox. A close is marked sent only after the bus accepts it. If orion-mind dies between sending and saving, it sends the same close again on restart, and the act ledger keeps the first close of an act. Act ids are stable on the concern id. Concerns held from before this change get their act on the first tick.

**How the economy feels it.** Concern acts are `actor = 'prime'`, and the body economy reads `actor` as the family (`orion/physiology/appetite.py`, `ActRecord.family = grammar_acts.actor`). Once the family has 30 graded closes in 30 days, the value of Prime's next bid is its hunger times the family's mean outcome, ranked against its own history. Every abandoned concern is a 0 in that mean, so it lowers what Prime's next thought is worth. No new price and no veto were added. One side effect to know: any close, a 0 included, also resets the family's hunger clock, the same as a thought's close does today.

## Files changed

- `orion/refs.py` (new): reference kinds, resolvers, id shapes, id-word check.
- `services/orion-durable-runs/app/main.py`, `app/temporal_self_store.py`: by-id arc and act reads.
- `orion/dig/__main__.py`, `orion/dig/README.md`: `arc` / `act` commands.
- `orion/prime/state.py`: concern act fields, `PrimeConcernRecordV1`, `close_concern`, `track_record`; fades record a close.
- `orion/prime/acts.py`: `concern_act` (verb `predict`), `CONCERN_HORIZON_SEC`, `lapse_concerns`, `settle_concern`.
- `orion/prime/thought.py`: `refs` on the move, ref and id-word checks, drops record a close, track record and concern grade in the frame, prompt text.
- `services/orion-mind/app/prime_loop.py`: resolves refs before sending, sends refs to Hub, publishes concern opens/closes from the outbox, settles concerns from their questions.
- `orion/schemas/introspect.py`: `QueueCuriosityFocusArguments.refs`.
- `orion/curiosity/focus.py`: `refs` column read/write, focus section prints each ref's dig command.
- `services/orion-hub/scripts/curiosity_focus_listener.py`: stores and echoes refs.
- `services/orion-sql-db/manual_migration_curiosity_focus_request_refs_v1.sql` (new).
- `orion/schemas/grammar.py`: notes the `predict` producer.
- `orion/prime/evals/run_prime_frame_eval.py`: `refs` and `record` checks. `orion/prime/evals/run_prime_report.py`: concerns tally from the ledger.
- Tests: `orion/dig/tests/test_refs_contract.py` (new), `orion/prime/tests/*`, `services/orion-mind/tests/test_prime_loop.py`, `orion/curiosity/tests/*`, `services/orion-hub/tests/test_curiosity_focus.py`, `services/orion-durable-runs/tests/test_ref_reads.py` (new), `services/orion-durable-runs/tests/test_temporal_self_chronicle_postgres.py`.
- CI: `.github/workflows/orion-prime-tests.yml` (refs contract step, paths), `.github/workflows/dig-kit-tests.yml` (paths).
- Docs: `services/orion-mind/README.md`, `services/orion-durable-runs/README.md`.

## Schema / bus / API changes

- Added: `QueueCuriosityFocusArguments.refs: list[RefV1]` (optional, default empty, at most 4). `PrimeStateV1.concern_record`, `PrimeConcernV1.act_opened_at` and `.graded` (all defaulted; the live persisted state loads unchanged, tested with its exact shape). Routes `GET /temporal-self/arcs/{arc_id}`, `GET /acts/{act_id}` on orion-durable-runs. Column `curiosity_focus_request.refs jsonb NULL`. New act kind on the existing grammar channel: `actor prime, verb predict`.
- Removed / renamed: none.
- Behavior changed: a Prime question with an unopenable or unlisted id is not sent.
- Compatibility: consumer-first. Hub accepts `refs` in this PR. orion-mind omits the `refs` key when a question cites nothing, so an older Hub still accepts those. A question WITH refs sent to an older Hub is refused by Hub (`extra=forbid`) and its act closes 0. Deploy Hub (after the migration) before orion-mind. The post-pull rebuild applies the migration first and rebuilds both.
- No channel or schema-registry entry changes (same channels, same envelope kinds).

## Env/config changes

- Added / removed / renamed keys: none. The dig kit reuses `ORION_PHYSIOLOGY_REPORT_URL` (already the orion-durable-runs base in orion-harness-governor); Prime uses its existing `PRIME_DURABLE_RUNS_URL`.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Migration

- `services/orion-sql-db/manual_migration_curiosity_focus_request_refs_v1.sql`: `ALTER TABLE curiosity_focus_request ADD COLUMN IF NOT EXISTS refs JSONB;`, header `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`.
- Fits the post-pull auto-apply allowlist: `scripts/apply_post_pull_migrations.py --dry-run` says "would apply (single transaction, lock_timeout=10s)". `scripts/check_migration_required_by.py`: OK.
- Until it is applied, Hub's focus reads fail (the column is selected), so the deploy gate refuses `up` for orion-hub, as designed.

## Tests run

```text
TESTS_PLACEHOLDER
```

## Evals run

```text
EVALS_PLACEHOLDER
```

## Docker/build/smoke checks

```text
SMOKE_PLACEHOLDER
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```bash
RESTART_PLACEHOLDER
```

## Risks / concerns

RISKS_PLACEHOLDER

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
