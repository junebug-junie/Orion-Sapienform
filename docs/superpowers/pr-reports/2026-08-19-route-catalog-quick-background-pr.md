# Make `quick_background` visible to every operator surface

Branch: `fix/route-catalog-quick-background`
Closes: the catalog half of the follow-up named in `orion/llm/routes.py`, from PR #1708
Status: **DONE**

## Summary

- Orion's own journalling has run on `quick_background` since PR #1708. That lane was absent from
  `GET /routes`, from the Hub, and from the routes smoke.
- The follow-up note predicted four hardcoded `("chat","quick","agent","metacog")` lists. **There
  were six.**
- The two extras are the point: widening the Hub client's `VALID_ROUTE_IDS` changed **nothing
  visible**, because two further hardcoded tuples backfilled and reordered the payload
  afterwards, reassembling a four-route response out of a five-route one.
- All six now derive from `LLM_ROUTE_DISPLAY_ORDER`, which **raises at import** if a route is
  accepted but not placed in it.
- The Hub picker is deliberately *not* the same set. It filters on each route's own `priority`,
  so a background lane is visible to operators while staying unpickable by a human.
- Two more stale assumptions found in the smoke while extending it (below).

## Outcome moved

`GET /routes` and the Hub both report five routes. `quick_background` carries
`priority=background reserved_free_slots=2`. The routes smoke now exercises it in both halves —
the catalog check *and* the RPC dispatch that proves it actually serves traffic.

## Current architecture

`orion/llm/routes.py` (PR #1708) unified how three services *normalize* an `llm_route` override,
and its docstring explicitly listed four further copies as out of scope. Those four governed
what the catalog *displays*, so a route could be routable and simultaneously invisible.

## Architecture touched

- **orion/llm** — `LLM_ROUTE_DISPLAY_ORDER` (ordered, completeness-checked at import) and
  `BACKGROUND_LLM_ROUTES` (what a route *is*, independent of config).
- **orion-llm-gateway** — catalog derives from the shared order; `priority` /
  `reserved_free_slots` surfaced; probes deduped per distinct upstream URL.
- **orion-hub** — client derives from the shared order in all three places; picker filters on
  `priority`; normalization split into a pure function so it is testable without HTTP.
- **scripts** — smoke widened in both halves.

## Files changed

- `orion/llm/routes.py`: `LLM_ROUTE_DISPLAY_ORDER`, `BACKGROUND_LLM_ROUTES`, import-time gates.
- `services/orion-llm-gateway/app/route_catalog.py`: derived catalog; `priority` /
  `reserved_free_slots` in the entry; one probe per distinct URL; `_probe_one` deleted.
- `services/orion-hub/scripts/llm_gateway_client.py`: derived set; `_normalize_routes_payload`
  extracted; `_priority_for` fail-safe.
- `services/orion-hub/static/js/app.js`: picker derived from the catalog by property.
- `scripts/smoke_llm_gateway_routes.py`: both halves widened; two stale assumptions fixed.
- Tests: `test_route_catalog_background.py` (11), `test_llm_gateway_client_routes.py` (13),
  `test_route_catalog.py` (fixture root-caused and fixed).

## Schema / bus / API changes

- Added: `priority` and `reserved_free_slots` on each entry of `GET /routes` and the Hub's
  `/api/llm-routes`. Purely additive.
- Behavior changed: `GET /routes` now lists five routes instead of four. A route in the
  vocabulary but absent from `LLM_GATEWAY_ROUTE_TABLE_JSON` appears as `not_configured` rather
  than being omitted — a lane missing from the table and a lane that was never a route are
  different problems.
- Compatibility: consumers keying on the old four still find them, in the same relative order.

## Env/config changes

None. No new keys.

## Tests run

```text
$ pytest services/orion-llm-gateway/tests -q
269 passed, 18 warnings in 4.61s

$ pytest services/orion-hub/tests/test_llm_gateway_client_routes.py -q
13 passed

$ node --check services/orion-hub/static/js/app.js
JS SYNTAX OK
```

## Evals run

```text
No eval harness exists for orion-llm-gateway or orion-hub. The quality question here is
"is the catalog complete", which is a contract question, not a model-output one -- covered
by the smoke below rather than by an eval.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-llm-gateway up -d --build
$ scripts/safe_docker_build.sh orion-hub          up -d --build

$ curl -fsS http://localhost:8210/routes
  chat               status=down            priority=None       reserved=None
  quick              status=up              priority=None       reserved=None
  quick_background   status=up              priority=background reserved=2
  metacog            status=up              priority=None       reserved=None
  agent              status=down            priority=None       reserved=None

$ curl -fsS http://localhost:8080/api/llm-routes
  routes    : ['chat', 'quick', 'quick_background', 'metacog', 'agent']
  background: ['quick_background']
  picker    : ['chat', 'quick', 'metacog', 'agent']

$ _verify_routes_http('http://localhost:8210', 10.0)
[ok] GET /routes default_route=quick routes=['chat','quick','quick_background','metacog','agent']

$ _expected_served_by('quick_background')  -> atlas-worker-fast-1
$ _expected_served_by('made_up_route')     -> AssertionError naming the drift, not a KeyError
```

`chat` and `agent` reporting `down` is correct: **circe is powered off.** The catalog is telling
the truth about a machine that is not running.

## Found while working, fixed here

- **The smoke asserted `default_route == "chat"`** while the live gateway *and*
  `services/orion-llm-gateway/.env_example` both say `LLM_ROUTE_DEFAULT=quick` — so it could only
  ever pass against a configuration nobody runs. Now asserts the default is a *real* route;
  which lane is default is an operator choice.
- **Its success line printed `default_route=chat` literally**, regardless of the response —
  reporting the fact it was asserting rather than the fact it observed. That is how the stale
  assertion above stayed invisible.

## Review findings fixed

Code review at `high` returned **eight findings. All eight are fixed.**

- Finding (**MEDIUM**): the picker filtered on `priority === 'background'`, which fails *open*.
  Two real paths deliver `quick_background` with `priority: null` — a rolling deploy where the
  Hub ships before the gateway, and the route being absent from the route table. Both would have
  offered the exact lane this PR exists to keep unpickable.
  - Fix: `BACKGROUND_LLM_ROUTES` in the shared module, applied by both the gateway and the Hub
    when the payload does not say. Plus a mirrored floor in JS for a page held open across a
    deploy — drift there degrades *gracefully*, since a new background route missing from the
    floor is still excluded by its `priority`.
  - Evidence: `TestPriorityIsFailSafe` (4 cases, incl. an old gateway omitting the route
    entirely); `test_an_unconfigured_background_route_still_declares_itself`.
- Finding (**MEDIUM**): the Hub backfill hardcoded `"priority": None` — the concrete producer of
  the above.
  - Fix: `_priority_for(route_id, reported)`.
  - Evidence: `test_old_gateway_omitting_the_route_entirely`.
- Finding (**MEDIUM**): the smoke hard-failed on a valid config. `reserved_free_slots` is
  genuinely optional (`priority_admission` falls back to a default), so a route table carrying
  only `"priority": "background"` is a working background lane the smoke called broken.
  - Fix: type-checked only when present, with `bool` excluded from the `int` check.
- Finding (**LOW**): the smoke's RPC dispatch still hardcoded four routes while the module
  docstring claimed otherwise, and `DEFAULT_ROUTE_SERVERS` had no `quick_background` entry, so
  widening it raised a bare `KeyError`.
  - Fix: widened, entry added, and the `KeyError` turned into a message that names the drift.
  - Evidence: `_expected_served_by('made_up_route')` output above.
- Finding (**LOW**): a configured-but-empty URL reported `not_configured` instead of `down`,
  collapsing a misconfigured route into one that was never in the table.
  - Fix: configured-but-unprobeable is `down` with `served_by` intact.
  - Evidence: `test_a_configured_route_with_an_empty_url_is_down_not_absent`.
- Finding (**LOW**): the test asserted the bug it documented.
  - Fix: root-caused instead. `monkeypatch.setenv` never reached the loader because `settings` is
    a pydantic instance built at import, so *every* row read `not_configured` and the
    `quick_background` assertion proved nothing. Patching the settings attribute makes the
    fixture real; `chat` now resolves to its configured `served_by`.
- Finding (**LOW**): `_probe_one` was dead after the dedup.
  - Fix: deleted. Evidence: `test_probe_one_is_gone`.
- Finding (**LOW**): the dedup key did not rstrip, so `http://atlas:8013` and
  `http://atlas:8013/` counted as two workers and cost six calls per refresh instead of three.
  - Fix: `_probe_key` matches the probe helpers' normalisation.
  - Evidence: `test_a_trailing_slash_does_not_defeat_the_url_dedup`.

## Restart required

Already applied from the worktree via `scripts/safe_docker_build.sh`. To redeploy:

```bash
cd /mnt/scripts/Orion-Sapienform-route-catalog-bg
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-hub          up -d --build
```

## Risks / concerns

- Severity: low
  Concern: `DEFAULT_ROUTE_SERVERS` in the smoke still claims `chat` and `agent` run on
  `atlas-worker-1`; they actually run on **circe**. Left unfixed — it is the "latent trap" the
  scarcity roadmap's A2 section already documents, and it only fires if both
  `LLM_ROUTE_*_SERVED_BY` and the route table's `served_by` are absent, which is not the current
  deployment.
  Mitigation: documented in place, directly above the dict.

- Severity: low
  Concern: `BACKGROUND_LLM_ROUTES` is mirrored in JS because the browser cannot import Python.
  Mitigation: it is a floor, not the mechanism — a route missing from the mirror is still
  excluded by its `priority`, so drift loses the second line of defence and never produces a
  wrong answer. Stated in the comment.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1733
