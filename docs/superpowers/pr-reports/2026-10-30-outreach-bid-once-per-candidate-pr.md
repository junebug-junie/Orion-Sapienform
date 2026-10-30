# Outreach bids once per held want, not on every tick

## Summary

- Hub's outreach loop checks every 10 seconds. When strain alone held it back, it placed a new shadow bid on every check.
- Now a strain hold places one bid per window. The window is the outreach cooldown (45 minutes live), and never less than 10 minutes.
- A real compose (Orion writes a message) still places one bid each time. That path did not change.
- The real outcome of every check is unchanged. Shadow bids are logged and never obeyed.
- The rows already in the ledger stay. The floor stays skewed until they age out, about 2026-11-06.

## Outcome moved

The want ledger records one row per held want instead of one per check.

Live, from 2026-10-27 16:46 to 2026-10-30 (56.6 hours):

| | hub.outreach bids | per strained hour |
|---|---|---|
| Before (live `want_ledger`) | 1,693 | ~360 |
| After (replay of the same checks) | 38 | 1-2 |

- 1,662 of the 1,693 rows came from strain-held checks. They match `endogenous_outreach_decisions` one for one (for example 10-29 21:00 UTC: 359 `arousal_strained` checks, 359 bids).
- The other 31 rows came from real composes. They stay.
- Replay: the 1,662 strain-held check times from `endogenous_outreach_decisions`, put through the new rule (one bid per 2,700 s), give 7 bids. 7 + 31 = 38.
- The regression test runs one hour of 10-second strain-held checks: 360 bids before the fix, 2 after.

**Why it matters for the floor.** The exploration floor funds a Tier 2 family that has under 5% of Tier 2's funded wants over 7 days. Live this week, hub.outreach held 880 of 1,370 funded Tier 2 wants (64%). That huge count made every other family look small, so the floor treated all of them as starved. With the fix, new rows reflect real wants. The 1,662 old rows stay in the 7-day window until about 2026-11-06, so the floor stays skewed until then. No cleanup was run: deleting ledger rows needs Juniper's approval and a snapshot first.

**Hunger does not change.** A family's hunger is the time since its last closed act (`orion/physiology/appetite.py`, `since_last_close`). It reads acts, not bids. Fewer bids do not move it.

## Current architecture

- `EndogenousOutreach.maybe_outreach` runs every `HUB_ENDOGENOUS_OUTREACH_TICK_SEC` (10 s). It makes one `ShadowBid` per check.
- `_outreach_once` placed that bid in two places: at the compose point, and when strain (`arousal_strained`) was the only gate that held. Strain is checked last, so a strain hold means every other gate passed.
- While strain stayed latched, every check placed a new bid with a new want id.

## Architecture touched

Hub only: `services/orion-hub/scripts/endogenous_outreach.py`. No bus, schema, env or API change.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: `_strain_shadow_bid_due()` and `STRAIN_SHADOW_BID_MIN_WINDOW_SEC`. The strain-held bid is placed only when it is due. A forced debug call still bids, and does not use up the window.
- `services/orion-hub/tests/test_physiology_shadow_bid.py`: three tests. One hour of strain holds gives 2 bids. A zero cooldown still uses the 10-minute floor. Each compose still bids.
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`: one note under Patch C3a results.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: fewer `hub.outreach` rows in `want_ledger` while strain holds outreach.
- Compatibility notes: none. The shadow-vs-real table reads the same rows.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-hub && pytest tests/test_physiology_shadow_bid.py tests/test_endogenous_outreach*.py -q
284 passed
With the fix reverted: the two new strain tests fail (360 == 2, 120 == 2).
```

## Evals run

```text
No eval harness for shadow bids. The live replay above is the evidence:
1,693 -> 38 hub.outreach bids over 56.6 h.
```

## Docker/build/smoke checks

```text
Not run. Pure Python logic change inside one loop; no dependency, port or config change.
```

## Review findings fixed

Independent review in a subagent. No must-fix findings. It confirmed: the real outcome does not change, the check runs inside the send lock, and hunger reads acts, not bids. It also removed the fix in a scratch copy: the two new tests failed.

- Finding (should-fix): the old rows keep the exploration floor skewed for 7 more days. The report must not say the floor is correct at deploy.
  - Fix: the PR report and the spec note now say the floor stays skewed until about 2026-11-06. A cleanup is proposed, not run.
  - Evidence: this report, "Why it matters for the floor".
- Finding (nit): a forced debug call used up the window and hid the next real strain bid.
  - Fix: a forced call bids without using the window.
  - Evidence: `endogenous_outreach.py`, `(force or self._strain_shadow_bid_due())`.
- Finding (nit): the docstring said "marks the bid as placed", but it marks an attempt. Restart and episode behaviour were not written down.
  - Fix: the docstring now states all three.
  - Evidence: `_strain_shadow_bid_due` docstring.
- Finding (nit): the floor test computed its expected count from the constant.
  - Fix: it asserts the constant is 600 and the count is 2.
  - Evidence: 284 passed.
- Not changed: the window is not reset when strain clears. This is a choice: one held want per cooldown, the same rate as a real send. It is written in the docstring.
- Not changed: my extra CI workflow was removed. The existing "Hub — full tests folder" job already runs this test file.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run it from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low
- Concern: a strain-held check does not test whether Orion has something to say. It still bids for the held want once per window.
- Mitigation: the bid now stands for one held want per window, the same rate a real send could happen. Checking for content would add database reads to every check, which a shadow must not do.

- Severity: medium
- Concern: the 1,662 old rows stay in the floor's 7-day window until about 2026-11-06. Until then the floor counts other Tier 2 families as too small and over-funds them in shadow.
- Mitigation: none in this PR. Option for Juniper: snapshot, then delete the duplicate strain-hold rows (section 14 backfill protocol). Or wait 7 days.

- Severity: low
- Concern: the window lives in memory. A Hub restart allows one new bid right away.
- Mitigation: one extra row per restart.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2988

🤖 Generated with [Claude Code](https://claude.com/claude-code)
