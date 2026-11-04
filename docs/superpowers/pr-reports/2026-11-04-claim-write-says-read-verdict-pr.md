## Summary

- `dig capacity claim` now tells Orion, in the same reply, whether the claim holds on the next `capacities` read.
- The verdict comes from the read's own code (`_capacity_reading` → `attach_ledgers`), so the write message and the read cannot disagree.
- A failing claim prints the failing references and how to fix it: "Every id the because names counts, even one you mention to drop. Claim it again naming only what holds, or `capacity withdraw`."
- Claims are still recorded the same way. #3109 chose a loose write with a strict read, and this PR keeps that.

## Outcome moved

Turn `14cc4d06` (2026-11-04 19:39Z): Orion claimed metabolism again with "... I drop pg:energy_bill_forecast, which is dormant". The write said "claimed". The next read said "evidence does not hold", because the read counts every id the text names. Orion did not know why. They withdrew the claim, claimed again, and then reported the cause wrongly ("the checker still evaluated the old claim"). Now the write itself names the failing reference.

## Current architecture

- Write: `claims.check_claim` → `cites_in`, which needs at least one valid cite.
- Read: `capacities.attach_ledgers` → `_claim_status`, which applies these gates in order: Juniper's rejection, a dead part, Juniper's yes, a near miss ("rule says no"), a dispute, then `claims.check_evidence`. `check_evidence` checks every id the because names, and the job's shape.
- Before this PR, the write message never showed the read's verdict.

## Architecture touched

- `orion/dig/__main__.py` `cmd_capacity`: after the ledger append, it re-reads the capacities with the atlas already in hand (no second live atlas read), finds this claim, and reports its status.

## Files changed

- `orion/dig/__main__.py`: the verdict on write.
- `orion/dig/tests/test_capacities_cmd.py`: 3 tests.
  - The write verdict matches the next read, for both failing and holding claims.
  - A claim Juniper rejected never prints "it holds now".
  - An S2 claim on a near-miss part never prints "it holds now".

## Schema / bus / API changes

- None. The ledger format is unchanged.

## Env/config changes

- None.

## Tests run

```text
pytest orion/dig/tests orion/body_atlas/tests tests/test_organ_names_partition.py tests/test_report_dig_kit_usage.py -> 356 passed
pytest orion/physiology/tests/test_report_text.py -> 28 passed
```

## Evals run

```text
orion/body_atlas/evals/run_*.py -> all exit 0
```

## Docker/build/smoke checks

This was a live read-only smoke: real atlas (FalkorDB 127.0.0.1:6380, bus Redis), with `ORION_DIG_HOME` pointed at a scratch copy of the ledgers. I used Orion's exact 19:42Z because:

```text
claimed orion-energy metabolism (cites: ...); written, but it does NOT hold on today's atlas: pg:energy_bill_forecast: dormant, not live (0 rows). Every id the because names counts, even one you mention to drop. Claim it again naming only what holds, or `capacity withdraw`
```

The real `/mnt/orion-fcc/self/capacity_claims.jsonl` is unchanged (8 lines before and after).

## Review findings fixed

- Finding (material): my first version called only `check_evidence`. The write could then print "it holds now" for a claim Juniper rejected, or for an S2 claim on a near-miss part, while the next read showed "rejected" or "rule says no". The reviewer reproduced both.
  - Fix: the verdict now comes from the read's own status code.
  - Evidence: `test_claim_write_never_says_holds_when_juniper_rejected_it`, `test_claim_write_never_says_holds_when_a_rule_says_no`.
- Finding (material): the test checked only the write message, not that the read agrees.
  - Fix: the test now runs `capacities` after each write and asserts the matching line.
  - Evidence: `test_claim_write_says_at_once_what_the_next_read_shows`.
- Finding (minor): the "even one you mention to drop" hint printed even when only a `shape:` line failed.
  - Fix: the hint prints only when a reference failed.
- Finding (minor): the list of failures was cut at 3 without saying so.
  - Fix: it uses `MAX_FAILED_SHOWN` and prints "(+N more)".

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

The post-pull rebuild covers the harness. The new sandbox sync (#3117) moves Orion's copy of the repo, which is the code Orion's turns import.

## Risks / concerns

- Severity: low. Each claim write now does one more ledger read and one more `read_capacities` on the atlas already loaded. This costs well under a second next to the ~27 s live atlas read.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3119

🤖 Generated with [Claude Code](https://claude.com/claude-code)
