# Brain-frame live stimulus→response smoke (acceptance §6)

Prereq: substrate-runtime + hub running; migration applied; flags on (Task 4).

1. Baseline: `curl -s http://localhost:<hub_port>/api/self-brain/frames/tail?limit=1 | jq '.frames[0].phase, (.frames[0].regions[] | select(.dimension=="lane"))'`
   Expect phase `live` (or `warming` at cold start).
2. Drive load: send a deep, tool-heavy chat turn through the normal chat path.
3. Within a few frames (~15s), re-poll tail and assert:
   - `lane:execution_trajectory` intensity rises / state → firing.
   - `self_state:execution_pressure` and `self_state:reasoning_pressure` intensity rise.
   - `node_kind:concept` stays dim (starving/steady).
4. Evidence to capture: two tail JSON snapshots (before/after) with `frame_id`s, and a screenshot of the Self tab brain with execution lit.
5. Verify a badly-lagged/absent lane renders `stale`/held, not jittering (acceptance §7).

Record PASS/FAIL + frame_ids in the PR report. Until run against the mesh: **UNVERIFIED**.
