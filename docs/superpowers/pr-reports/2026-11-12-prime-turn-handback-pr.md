# Living Prime Part 2: every turn ends back in Prime (turn hand-back)

Design: PR #3224, `docs/superpowers/specs/2026-11-12-living-prime-one-orion-design.md`, "Part 2". Juniper signed off 2026-11-12.

## Summary

- Prime now hears every finished chat, curiosity, consult and letter turn. It listens to the cockpit hops Hub already publishes on `orion:cockpit:hop`. No new bus event.
- The tick folds each turn into Prime's state by code, not by a model: who, a one-line gist, links to the turn's promises, and the correlation id.
- A turn that made a promise opens a concern tied to that turn (`event_ref = turn:<correlation_id>`). A turn whose prompt cites an open `prime_concern:<id>` and replied settles that concern.
- Only those two wake a thought. The existing funded-thought rule still decides. Every handed-back turn shows in Prime's next frame.
- Turns Prime started (its curiosity question) still close through the existing nomination grading. The run's turn correlation id is now linked onto the act's close note and the turn record.
- Loop guard: at most 1 turn-opened concern held at once, at most 4 opened in 24 h, on top of the existing 3-concern cap and 24 h fade. Kill switch `PRIME_TURN_HANDBACK_ENABLED`, shipped ON.

## Outcome moved

Before: Prime knew Juniper talked (conversation arcs) but not what was said, decided or promised. After: Prime's frame lists the last 5 finished turns with gists, and a promise made in a turn becomes a concern Prime holds and can act on.

## Current architecture

- Prime (`services/orion-mind/app/prime_loop.py`, `orion/prime/`) read the body, the funded event wants and curiosity runs. It had no reader of any turn record.
- Hub publishes `CockpitHopV1` on `orion:cockpit:hop` for every unified turn. Consumers were orion-sql-writer (into `cockpit_turn_sighting`) and Hub.

### Which turn record, and the live evidence (2026-11-12)

Candidates checked:

| Record | On the bus at turn end? | Has who + reply + promises? |
|---|---|---|
| `cockpit_turn_sighting` hops (`orion:cockpit:hop`) | yes, `ingress` + `finalize` | yes: ingress has `session_id`, prompt, `mode`; finalize has the whole `HarnessRunV1` incl. `reflection` |
| `harness_turn_trace` / `orion:harness:run:artifact` | yes | no session, so no "who" |
| `orion:chat:history:turn` | yes | only Juniper's chat session and outreach (44 rows in 3 days vs 1,099 hops); no curiosity or consult |
| `cortex_turn` attention row | written at stance time | not at turn end |

Chosen: `orion:cockpit:hop`. Evidence:

- `PUBSUB NUMSUB orion:cockpit:hop` = 1 (orion-sql-writer); every `cockpit_turn_sighting` row is written by sql-writer from that channel (`services/orion-sql-writer/app/settings.py:201`).
- Last 3 days, ingress -> finalize pairs by session: `orion_curiosity` 69 -> 59, Juniper's chat with `mode` 12 -> 11, `consult-*` 7 -> 2, plus 300+ world-pulse reads, self-sense evals and PR field notes that are not handed back.
- The finalize hop's `raw` is the full `HarnessRunV1` (keys checked live: `final_text`, `compliance_verdict`, `reflection`, ...).
- `reflection.promises_made` is not in any finalize hop yet (0 of 55 in the last 6 h): the intention ledger D0 judge fields (PR #3217) are merged but not deployed. Until they deploy, no turn opens a thread.

## Architecture touched

- orion-mind: new second consumer of `orion:cockpit:hop`; new Redis key `orion:prime:turns` (Prime's hand-back record, kept apart from `orion:prime:state` so an older image still loads Prime's state after a rollback).
- `orion/prime/`: new pure module `handback.py`; `thought.py` frame gets one optional section and a "from turn" concern label.
- Prime state model unchanged. A turn concern uses the existing `PrimeConcernV1.event_ref`, so the Part 1 renderer (parallel PR) reads it with no change.

## Files changed

- `orion/prime/handback.py`: the hop -> signal cut, who classification, gist, fold, loop guard, frame lines.
- `orion/prime/thought.py`: `FrameInputs.turns`; TURNS THAT HANDED BACK TO YOU section; concern line says "from turn <corr>".
- `services/orion-mind/app/prime_loop.py`: bus listener (resubscribe with backoff), in-memory inbox, fold on the tick, save/load of `orion:prime:turns`, nomination close links the run's turn.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `PRIME_TURN_HANDBACK_ENABLED=true`.
- `orion/bus/channels.yaml`: orion-mind added as a consumer of `orion:cockpit:hop`.
- `orion/prime/evals/run_turn_handback_replay.py`: read-only live replay eval.
- `orion/prime/tests/test_prime_handback.py`, `orion/prime/tests/fixtures/turn_hops_recorded.json`, `services/orion-mind/tests/test_prime_turn_handback.py`: tests.
- `.github/workflows/orion-prime-tests.yml`: Prime tests also run when the cockpit hop schema or publisher changes.

## Schema / bus / API changes

- Added: consumer `orion-mind` on `orion:cockpit:hop`. Pure models `TurnSignalV1`, `PrimeTurnV1`, `PrimeTurnsStateV1` (Redis `orion:prime:turns`, not on the bus).
- Removed: none.
- Renamed: none.
- Behavior changed: Prime opens and settles concerns from turns (rules above). A nomination close note may end with `turn <correlation_id>`.
- Compatibility notes: no payload change on any channel. A turn concern is an ordinary `PrimeConcernV1` with `event_ref = "turn:<corr>"`.

New concern fields: none on the model. A concern a turn opened has:

- `event_ref`: `turn:<correlation_id>` (the turn that opened it);
- `text`: `a thread I opened in a <who> (<n> promise(s), harness_turn_trace:<corr>#promises_made[0] ...): <gist>`. The promise ref is the same reference the intention ledger's promise acts use (design 2026-11-11, D2 `test_before`): linked, not copied.

The hand-back record per turn (`PrimeTurnV1`): `correlation_id`, `who`, `gist`, `started_at`, `finished_at`, `verdict`, `promise_refs`, `promises_kept`, `opened_concern_id`, `thread_refused`, `settled_concern_ids`, `started_by_act`.

## Env/config changes

- Added keys: `PRIME_TURN_HANDBACK_ENABLED` (orion-mind, default `true`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, `services/orion-mind/.env_example`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-mind: +PRIME_TURN_HANDBACK_ENABLED='true'`).
- skipped keys requiring operator action: none.

## Tests run

```text
python -m pytest -q -p no:cacheprovider orion/prime orion/mindloop orion/cockpit      -> 140 passed
(cd services/orion-mind && python -m pytest -q -p no:cacheprovider tests)             -> 246 passed
orion-static-gates.yml steps run locally                                               -> see Docker/build section
```

Covered: a recorded turn record folds deterministically; a thread opens a concern citing the turn's correlation id within one tick (runtime test, real Hub publisher envelope through the listener's decode path); an answered concern settles; a reply that only mentions a concern does not; offers are not threads; the three loop bounds; TTL fade; dedupe of a replayed hop, a duplicate finalize in one batch, a restart between ingress and finalize; flag off = no fold, no record, no frame section; nomination link.

## Evals run

```text
PYTHONPATH=. python -m orion.prime.evals.run_turn_handback_replay --days 3   (live, read-only)
checks: deterministic, coverage, not_hollow, bounded, replay_safe -> all true
1,099 hops -> 72 turns handed back: curiosity 59, chat_juniper 11, consult 2; missing 0
threads opened 0 (no promises_made on live hops until D0 deploys), concerns answered 0

PYTHONPATH=. python orion/prime/evals/run_prime_frame_eval.py   -> all PASS
PYTHONPATH=. python orion/prime/evals/run_prime_wake_replay.py  -> all PASS (unchanged: hand-back adds no wake without a thread or answer)
```

## Docker/build/smoke checks

```text
Not deployed (per instructions). No image dependency changed.
Live path: UNVERIFIED until orion-mind is rebuilt. After deploy, check:
  docker logs orion-athena-mind 2>&1 | grep -E 'prime_handback_subscribed|prime_turn_handback'
  redis-cli -u $ORION_BUS_URL PUBSUB NUMSUB orion:cockpit:hop     # expect 2
  redis-cli -u $ORION_BUS_URL GET orion:prime:turns | head -c 600
```

## Review findings fixed

(filled in after review)

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the "thread opened" signal is the intention ledger D0 judge field, not deployed yet; until it is, Part 2's thread-to-concern path is idle on live data. Mitigation: deploy #3217's governor change; the replay eval reports `turns_with_promises` so the first live thread is visible.
- Severity: low. Concern: pub/sub is not durable; a turn that finishes while orion-mind is down is never handed back. Mitigation: documented; the pending ingress list survives a restart, so only turns finishing during the outage are lost.
- Severity: low. Concern: "answered" means the turn's PROMPT cited the concern. Juniper rarely types concern ids, so today this fires mainly for Prime-built prompts (consult, Part 3 letter). Mitigation: deliberate; a reply mention would let a "what are you holding?" answer settle every concern once Part 1 puts ids in every prefix.
- Severity: low. Concern: a promise concern does not settle when the promise is later kept, because the ledger's promise act ids (D2) do not exist yet. It fades after 24 h unless a thought keeps it. Mitigation: `promises_kept` act ids are already recorded on the turn record for when D2 lands.
- Severity: low. Concern: feedback loop (turns open concerns, concerns steer turns). Mitigation: 1 held / 4 a day bound on top of the 3-cap and 24 h fade; refusals recorded on the turn. Watch `prime_turn_handback ... opened=` lines per day in the first week.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
