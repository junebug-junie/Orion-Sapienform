# Attention patch 3d: the market's consent gate (dream compaction is its first consumer)

## Summary

- The want market now answers one question for acts that are hard to put back: "is this act allowed,
  and who agreed?". One pure function gives `allow`, `refuse` or `needs_consent` and names the rule it
  used (`orion/physiology/consent.py`, rules in `config/physiology/consent.yaml`).
- A (family, verb) that has a consent rule must ask: a want for it without a request is refused.
- Choice asks the gate first, before prices, the body state or the act history. A `refuse` or
  `needs_consent` becomes a deferral with reason `consent`. The binding rule never admits it, not even
  at the window's end (`orion/physiology/binding.py` `consent_block`).
- A want without a consent request never reaches the gate. Its decision is byte-identical to before.
- Dream compaction apply asks the gate first, even while apply is off. After each staged REM delta the
  dream service asks through the applier with no memory store, so every real delta leaves a verdict and
  nothing can be written.
- No standing consent ships. Every dream ask reads `needs_consent` until Juniper merges one. Prune is
  `irreversible` and always refused.

## Outcome moved

The retired policy rung (L8) used to answer "may this act change something that cannot be put back?".
Since its retirement (PR #3161) nothing answered it, so dream compaction apply had to stay off with a
consent callable that always said no. Now a real, inspectable gate answers it, with a rule name on every
verdict, a `want_ledger` row, and a log line. Turning apply on stays Juniper's separate decision.

## Current architecture

- `POST /physiology/bid` on orion-durable-runs decides wants by price, hunger and capacity
  (`orion/physiology/choice.py`). No consent concept.
- `orion/physiology/binding.py` `bind_verdict`: a "no answer" (`physiology_unavailable`,
  `unknown_family`) is admitted, and any deferral is admitted at the window's end.
- `services/orion-dream/app/compaction_applier.py`: hot gate first, then an injected consent callable
  (never wired: always no), then an injected store (never wired). Nothing called it live.
- REM stages about 2 deltas a day (115 rows in `dream_compaction_delta`; in the last 14 days only
  consolidate gist cards, 0 downscale, 0 prune).

## Architecture touched

- `orion/physiology/consent.py` (new): `parse_consent`, `load_consent`, `consent_verdict`.
- `orion/physiology/choice.py`: `consent_gate` and `with_consent`; `decide(..., consent_rules=...)`.
- `orion/physiology/binding.py`: `consent_block`; `bind_verdict` never admits a consent block;
  `decision_detail` carries `consent_verdict` / `consent_rule`.
- orion-durable-runs market: loads `consent.yaml` at boot, runs the gate before any read, logs
  `physiology_consent verdict=... rule=... want=...`, `/health` counts.
- orion-dream: `app/compaction_consent.py` (new), applier gate order, REM pass consent ask.
- Old policy rung mapping (from `config/policy/substrate_policy.v1.yaml` and `orion/policy/evaluator.py`
  at `b37a1820a^`): `hard_blocks: destructive_action` -> `irreversible` must be `refused`; review below
  reversibility 0.5 -> a non-reversible act needs a standing consent; unknown kind deferred -> unknown
  non-reversible act refused; `allow_execution_without_operator: false` -> `consent: none` only for
  `reversible`.

## Files changed

- `orion/physiology/consent.py`: the gate (pure, deterministic, strict parser).
- `config/physiology/consent.yaml`: rules `dream.compaction.downscale_only` and `dream.compaction.prune`;
  `standing: {}`.
- `config/physiology/families.yaml`: new family `dream.compaction` (Tier 2, no lanes, no receptors,
  86,400 s window, `weighted: false`); new optional field `weighted`.
- `orion/physiology/families.py`, `orion/physiology/weights.py`: the `weighted` field (a family out of the
  tier's weight normalization).
- `config/physiology/contract.yaml`: registers the dream consent bid site.
- `orion/schemas/physiology_want.py`: `ConsentRequestV1`, `ConsentVerdictV1`, `WantV1.consent_request`,
  `WantDecisionV1.consent`, reason `consent`.
- `orion/physiology/choice.py`, `orion/physiology/binding.py`, `orion/physiology/report_text.py`.
- `services/orion-durable-runs/app/{physiology_market,main,settings}.py`, `.env_example`,
  `docker-compose.yml`, `Dockerfile` (copies `consent.yaml`), `README.md`.
- `services/orion-dream/app/{compaction_applier,compaction_consent,rem_compaction,main,settings}.py`,
  `.env_example`, `docker-compose.yml`, `README.md`.
- Tests: `orion/physiology/tests/test_consent.py`, `orion/physiology/tests/consent_fixture.py`,
  `services/orion-durable-runs/tests/test_physiology_consent_market.py`,
  `services/orion-dream/tests/test_compaction_consent.py`, updates to `test_compaction_applier.py` and
  `test_event_wants.py`.
- Eval: `services/orion-durable-runs/evals/consent_gate_fixture.py` (fixture and `--live`).
- CI: `.github/workflows/orion-durable-runs-tests.yml` runs the eval and the dream compaction tests
  (the applier tests were not in any CI job before).

## Schema / bus / API changes

- Added: `ConsentRequestV1` (`want_id`, `family`, `verb`, `object_ref`, `payload_digest`,
  `reversibility`, `attests`), `ConsentVerdictV1` (`verdict`, `rule_id`, `reason`, `consent_ref`),
  `WantV1.consent_request`, `WantDecisionV1.consent`, decision reason `consent`.
- Removed: none.
- Renamed: none.
- Behavior changed: a want with a consent request is gated; the applier asks consent before the hot gate;
  new applier status `no_store`.
- Compatibility notes: both new fields are omitted when None (same pattern as patch 1's event fields), so
  every existing want and decision serializes exactly as before. `WantV1`/`WantDecisionV1` are
  `extra=forbid`, so a decision that carries `consent` needs rebuilt consumers. Only dream compaction
  sends a consent request, so rebuild order is: the shared-schema consumers (sql-writer first), then
  durable-runs, then dream last. A dream that reaches an old durable-runs gets a 422, which the client
  turns into `None`, which the applier reads as `not_approved` (fails closed). No new channel, no SQL
  migration (`want_ledger.decision` is jsonb).
- Spec ambiguity resolved: the spec's `ConsentRequestV1` field list has no `reversibility`, but "an
  unknown pair with reversibility other than reversible" needs one. Added `reversibility` to the request;
  the rule's class is authoritative and a request that claims a different class is refused.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_CONSENT_PATH` (orion-durable-runs, default
  `/app/config/physiology/consent.yaml`), `DREAM_COMPACTION_CONSENT_ASK_ENABLED` (orion-dream, default
  `true`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, both services (and compose `environment:`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes
  (`--all-keys orion-durable-runs orion-dream`: both keys added to the primary checkout's `.env`).
- skipped keys requiring operator action: none. (Pre-existing diverged keys reported, not changed:
  `POSTGRES_URI`, `DURABLE_RUNS_GRAPH_HOST` in orion-durable-runs.)

## Tests run

```text
orion/physiology/tests                         302 passed
services/orion-durable-runs/tests              836 passed, 130 skipped
services/orion-dream/tests + evals             374 passed, 9 skipped
cortex-orch / actions / hub / thought shadow-bid tests, sql-writer want_ledger shape,
tests/test_act_ledger_contract.py              all passed
topic-foundry shadow-bid test                  collection error (no joblib in the local venv; same on main)
scripts/check_body_economy_contract.py         OK (103 modules registered)
scripts/check_env_template_parity.py           PASS
scripts/check_service_env_compose_parity.py    orion-durable-runs OK; orion-dream N/A (env_file)
scripts/check_env_key_single_source.py         OK
scripts/check_personal_terms.py                OK
Mutation checks: disabling the gate in decide -> 5 tests fail; disabling consent_block in bind_verdict -> 2 fail;
putting dream.compaction back in the weight pool -> 2 tests and the eval fail
```

## Evals run

```text
$ PYTHONPATH=. python services/orion-durable-runs/evals/consent_gate_fixture.py
ok  prune is irreversible: refused
ok  unknown non-reversible act: refused
ok  unknown reversible act: allowed
ok  missing attest: refused
ok  downscale_only, both attests, standing consent: allowed
ok  downscale_only, shipped (no standing consent): needs consent
ok  no rules read: refused
ok  dead market (no body state): reason=consent reads=0
ok  dead market (history unreadable): reason=consent reads=0
ok  no consent request: 9/9 live want shapes decided byte-identically
    (one side: no consent rules and no dream.compaction family; other side: both; uneven weight evidence)
PASS

$ PYTHONPATH=. python services/orion-durable-runs/evals/consent_gate_fixture.py --live --days 3
live 3 d: staged deltas=3 consent asks=0
UNVERIFIED: no consent asks yet (orion-dream and orion-durable-runs not rebuilt, or no REM delta staged)
```

The live lane is the acceptance-check-7 dream line. It is UNVERIFIED until both services are rebuilt and
a sleep stages a delta.

## Docker/build/smoke checks

```text
Not run: no deploy from a worktree. The durable-runs Dockerfile copy of consent.yaml is covered by
services/orion-durable-runs/tests/test_dockerfile_copies_physiology_config.py (passes).
```

## Review findings fixed

Review ran in a subagent against `origin/main...HEAD` of this branch. It found no fail-open path for a
consent-requested act. Findings and fixes:

- Finding (must-fix): the new `dream.compaction` family changed decisions for wants WITHOUT a consent
  request. Weights are normalized per tier; a family with no acts of its own sits at its prior forever and
  lowers every other Tier 2 weight. The "byte-identical" test missed it because both sides used the same
  families.yaml and empty weights.
  - Fix: a `weighted: false` field in `config/physiology/families.yaml` (parsed in
    `orion/physiology/families.py`, honored in `orion/physiology/weights.py`). `dream.compaction` sets it,
    so it bids at its prior and does not move the tier's mean.
  - Evidence: `test_adding_the_consent_only_family_moves_no_other_familys_weight` and
    `test_live_want_shapes_decide_identically_with_and_without_the_new_family` compare families.yaml with
    and without the family, with uneven real-shaped use evidence. The eval does the same through the market.
    Removing the fix fails both tests and the eval (`vision.ask: decision changed`, 5 fails).
- Finding (should-fix): deploy order for `extra=forbid` readers was not documented.
  - Fix: sql-writer, then durable-runs, then dream, in this report's restart section and the durable-runs
    README.
- Finding (should-fix): the gate ran only when a producer asked.
  - Fix: a want for a (family, verb) that has a rule but carries no request is refused (`request_missing`).
  - Evidence: `test_a_ruled_act_without_a_request_is_refused`.
- Finding (should-fix): a `compact` verdict could stand in for a `prune` ask of the same delta (same id,
  same digest).
  - Fix: the verb is in the want id, and `judge` also checks verb and reversibility.
  - Evidence: `test_a_compact_verdict_never_stands_in_for_a_prune_ask`.
- Finding (nit): the sleep cycle asked before it persisted the delta.
  - Fix: `_rem` asks after its own persist, only for a stored delta; the REM pass asks only when its
    persister returned true.
  - Evidence: `test_a_delta_that_failed_to_persist_is_not_asked_about`, `test_the_sleep_cycle_asks_after_its_own_persist`.
- Finding (nit): the market evaluated the gate two or three times per bid.
  - Fix: the market runs `consent_gate` once and prices with `decide_priced`.
- Finding (nit): a standing consent that no rule reads failed silently.
  - Fix: `parse_consent` refuses the file, so a typo in Juniper's consent PR fails its own CI.
  - Evidence: `test_a_standing_consent_no_rule_reads_is_a_config_error`.
- Finding (nit): enum tuples duplicated the schema Literals.
  - Fix: `test_the_enum_tuples_match_the_schema_literals`.
- Finding (nit): `ask_consent` copied a `ConsentAnswer` by position.
  - Fix: `dataclasses.replace(ans, approved=False)`.

## Restart required

Rebuild in this order (consumers of the new decision field first, dream last):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
```

Hub, actions, cortex-orch, mind and thought also parse `WantDecisionV1`, but none of them sees a decision
with `consent` (only dream sends a consent request). They pick up the schema on their next normal rebuild.

## Risks / concerns

- Severity: low. Concern: one extra market bid per staged REM delta (about 2 a day) writes a deferred
  `dream.compaction` row to `want_ledger`. Mitigation: deferred rows do not count toward the exploration
  floor; `DREAM_COMPACTION_CONSENT_ASK_ENABLED=false` stops the ask.
- Severity: low. Concern: the dream applier obeys consent only; it ignores a price or hunger deferral
  (it is one-shot, with no hold loop). Mitigation: `bind_verdict` with the hold ending now still never
  admits a consent block; apply stays off.
- Severity: info. Concern: the gate reads `consent.yaml` once at boot. A merged standing consent needs a
  durable-runs rebuild (the file is baked into the image). Mitigation: same as families.yaml today.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3185

🤖 Generated with [Claude Code](https://claude.com/claude-code)
