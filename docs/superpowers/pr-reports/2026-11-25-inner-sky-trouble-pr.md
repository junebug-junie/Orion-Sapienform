## Summary

- The Inner Sky shows **trouble in red and amber**, taken only from what the rows themselves record:
  - An experience whose own outcome says something broke lands **red**: metacog `critical`, a faculty `broken`.
  - One whose outcome says something went wrong lands **amber**: `degraded`, a claim that didn't hold up, `refused`, `unavailable`, `miscalibrated`, `unreliable`, `abandoned`, a reading with no evidence, a reading that never closed or finished late.
  - That experience's part flares the same colour, the feed line gets a coloured edge, and a "trouble" count joins the header.
- A **link whose latest hop, within the last 10 minutes, is far slower than its own usual** turns red, with red pulses. This uses the `/api/bus-synaptic-graph/anomalies` rule (`latency_zscore > 3`, `count > 5`), slower only.
- A **service that was live and goes silent** fades out through red.
- **Normal stays normal:** a guard that didn't fire (`not_crossed`), a gate doing its job (`granted`), a prediction checked (`confirmed`, `disconfirmed`, `unscored`), or a choice not to act.
- The verdict table lives in `scripts/inner_sky_verdicts.py`, which is pure. Unknown verdicts stay neutral, and the eval lists them as `unsorted_verdicts`.

## Outcome moved

Orion, shown the sky, pointed out that it already shows guards that *didn't* fire and parts that chose to suppress. Juniper asked for failures to show too. Now the sky separates "something broke" from "a gate held", using each row's own words.

Live (12 h, 2026-11-25): 957 experiences, of which 52 are red and 121 amber, nearly all metacognition flags. 2 of 248 links are slow: vision-council → vision-scribe at 78 spreads above usual, and vision-council → gpu-pool. Unsorted verdicts over 7 days: none after this patch.

## Current architecture

Inner Sky after #3893 and #3894 drew every experience in its part's colour, whatever its outcome, and every link in the same blue.

## Architecture touched

- Hub route `scripts/inner_sky_routes.py`:
  - Events gain `verdict` and `severity`.
  - Links gain `z` and `slow`.
- New `scripts/inner_sky_verdicts.py`.
- Page `static/inner-sky.html` and `static/js/inner-sky.js`.
- Eval `evals/run_inner_sky_feed_eval.py`.
- README.

## Files changed

- `services/orion-hub/scripts/inner_sky_verdicts.py`: the outcome table: red, amber and normal words; prefix rules for reading verdicts that carry a reason.
- `services/orion-hub/scripts/inner_sky_routes.py`: severity on events; link slowness (z > 3, count > 5, last hop under 10 minutes ago).
- `services/orion-hub/static/js/inner-sky.js`, `static/inner-sky.html`: red/amber comets, part flares, feed edges, trouble count, red slow links and pulses, the went-silent fade, legend.
- `services/orion-hub/evals/run_inner_sky_feed_eval.py`: reports `trouble`, `unsorted_verdicts` and `slow_links`; loads the table by file path.
- `services/orion-hub/tests/test_inner_sky_routes.py`: verdict cases, including all 7-day live words; slow-link rules; the eval.
- `services/orion-hub/README.md`: what red and amber mean.

## Schema / bus / API changes

- **Added:** `/api/inner-sky/events` rows gain `verdict` and `severity`. `/api/inner-sky/state` links gain `z` and `slow`.
- **Behavior changed:** these are additive JSON fields, and the only reader is this page.

## Env/config changes

None.

## Tests run

```text
pytest test_inner_sky_routes.py test_inner_sky_page.py     36 passed
node --test services/orion-hub/static/js/*.test.js         456 pass, 0 fail
```

## Evals run

```text
run_inner_sky_feed_eval.py (worktree code against live data): experiences 957, trouble {amber: 121, red: 52},
  unsorted_verdicts {}, slow_links 2, verdict PASS
run_inner_sky_feed_eval.py --hub http://127.0.0.1:8080 (deployed Hub, pre-merge): imports and runs, PASS
7-day verdict vocabulary (Postgres SELECT): every word now sorted
```

## Docker/build/smoke checks

```text
Hub-only Python + static change; the normal post-pull Hub rebuild applies it.
Browser render: UNVERIFIED here (headless Chromium blocked); tracked by #3891.
```

## Review findings fixed

- **Finding:** the went-silent red fade could never fire, because it compared against the already-updated clock.
  - **Fix:** it compares against the previous poll's time, and the red clears when the service speaks again.
  - **Evidence:** code at the services loop in `pollState`.
- **Finding:** the eval crashed on import. The repo `scripts` package shadowed the Hub's, and `app.settings` failed validation.
  - **Fix:** the table is a pure module that the eval loads by path.
  - **Evidence:** the eval runs from the repo root (above).
- **Finding:** six live verdicts were unsorted.
  - **Fix:** `confirmed`, `disconfirmed`, `unscored` and `no_lasting_trace` are normal; bare `no_read_evidence` and `late:<N>s` are amber.
  - **Evidence:** parametrized tests; eval `unsorted_verdicts: {}`.
- **Finding:** the hover text was always amber.
  - **Fix:** it shows red, with a count of broken outcomes, when the part had a red one.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

The post-pull rebuild does this on merge.

## Risks / concerns

- **Severity: low.**
  - **Concern:** the sky reads red-heavy, because metacognition raises about 250 degraded/critical flags a day.
  - **Mitigation:** #3898 (design: a toggle between all live trouble and trouble outside its usual rate).
- **Issues opened:** #3898. **Closed:** none.

## PR link

This PR.
