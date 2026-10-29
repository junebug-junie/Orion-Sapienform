## Summary

- The first bench (2k-token prompt, 512 tokens out, one slot) did not look like real gpu2 work. It wrote at 49 tok/s; real work writes at 30 tok/s.
- New `--prompt-sentences` flag sets the prompt size (480 sentences = 16,221 tokens on Bonsai 2, measured live).
- New `--slots 0,1` mode sends one request per slot at once (a "round"), like the agent lane does.
- A round records all-in joules per written token over the whole round, from the first prompt read to the last reply. Per-phase energy is not split, because one slot reads while the other writes.
- Each summary row now also has the card's output rate (written tokens per second of the whole round) and all-in J/token. Both are compared against the highest cap.
- New `scripts/gpu_pool_clear_gpu2.py` keeps Orion's own work off gpu2 while the bench runs. It takes GPU pool operator holds on gpu1 and on both gpu2 slots, and releases them on every exit. The first two-slot run lost every round to Orion's agent calls.

## Outcome moved

The 150 W cap decision can now be checked against work shaped like gpu2's real load, not a short single-slot request.

Real gpu2 load (llama-server timing lines, 72 h, 1,317 requests):

| | median | top 10% |
|---|---|---|
| new prompt tokens | 6,546 | 16,053 (max 82,686) |
| tokens written | 2,542 | 13,585 |
| write speed | 30 tok/s | — |

Writing takes 91% of busy time.

First bench result (2k in, 512 out, one slot): 150 W wrote at 0.89x speed for 0.83x J/token.

**Long-context result** (2026-10-29 05:26 UTC, circe gpu2 cleared by the hold script, 3 rounds per cap, 0 dropped; each round has two slots, 16,221 tokens in and 4,096 out per slot):

| | 250 W | 150 W | 150 W vs 250 W |
|---|---|---|---|
| write speed one turn sees | 29.4 tok/s | 25.2 tok/s | 0.86x |
| card output (both slots, reading included) | 45.0 tok/s | 38.1 tok/s | 0.85x |
| prompt read speed | 553 tok/s | 449 tok/s | 0.81x |
| all-in energy per written token | 4.89 J | 3.93 J | **0.80x** |
| card average power | 220 W | 150 W | |
| max card temperature | 78 °C | 67 °C | −11 °C |

At 250 W the bench wrote at 29.4 tok/s, which matches real gpu2 work (30 tok/s). Juniper chose 150 W.

## Current architecture

`scripts/gpu_power_bench.py` sends one request at a time to one slot, with a fixed 60-sentence prompt.

## Architecture touched

The bench script, a new operator hold script, and their tests. The hold script uses the existing GPU pool operator verbs (`hold`/`release` over the bus) through `orion/evals/model_replay/pool_hold.py`, which is unchanged. No service, config, schema or bus change.

## Files changed

- `scripts/gpu_power_bench.py`: `--prompt-sentences`, `--slots`, `one_round`, `to_round_sample`, `busy_slots`/`SlotWatch` accept several slots, two new `Sample` fields, wider summary table.
- `scripts/gpu_pool_clear_gpu2.py`: new. It holds gpu1 first, so class `agent` falls through to gpu2. Then it holds both gpu2 slots, sending the second hold only while the pool shows gpu2's other slot free. A miss is given back and retried. It refuses half a card, and it reports leftovers from the ledger.
- `tests/test_gpu_pool_clear_gpu2.py`: 9 tests (free-slot rule including a 1-slot or unloaded seat, all three held then released, waits for a free slot, retry on a miss, give up and release, SIGTERM after READY, hold request lost mid-flight is swept, failed release reported with exit 4, no bus URL).
- `tests/test_gpu_power_bench.py`: 18 new tests (prompt size, multi-slot ignore, concurrent send, failure re-raise, round energy math, single-slot round, two-slot run end-to-end, bad slots refused, own-slot queueing kept).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: CLI `--slot N` → `--slots N[,M]` (default `0`; the old flag was never passed in any documented command)
- Behavior changed: the default run is unchanged except for two extra CSV/summary columns.
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest tests/test_gpu_power_bench.py -q                      -> 48 passed
PYTHONPATH=. .venv/bin/python -m pytest tests/test_gpu_pool_clear_gpu2.py -q     -> 9 passed
PYTHONPATH=. .venv/bin/python -m pytest orion/evals/model_replay/tests/test_pool_hold.py -q -> 16 passed
git diff --check                                              -> clean
```

## Evals run

```text
None: this is an operator measurement script. The bench run is the measurement.
```

## Docker/build/smoke checks

```text
Live, on circe gpu2 at its normal 250 W limit (no root, no cap change):
one two-slot round, --prompt-sentences 480, 128 tokens out
  prompt_n per request: 16,221
  waited: slot0 1.58 s, slot1 24.88 s   <- server reads our two prompts one after another
  round: 225 W average, 68 C, all-in J/token computed

First two-slot bench, no holds (04:08-04:48 UTC): 4 rounds, all dropped other_slot_busy.
The server log showed Orion's ~5k-token agent calls inside every round. That is correct behaviour.

Hold script, live:
  try 1 (05:08): second class-agent hold fell through to role chat (gpu0 is lent);
                 given back in 0.15 s, everything released (ledger + pool lease table)
  try 2 (05:12): gpu1 + agent-gpu2 x2 granted within 1 s; gpu2 slots idle, 43 W;
                 released after 37.8 min on SIGTERM, ledger clean, pool shows no live lease
Long-context bench under the holds: 6 rounds, 0 dropped (table above); card restored to 250 W
```

## Review findings fixed

- Finding (self, from the live smoke): the server reads the two prompts one after another, so the second request always "waited" ~25 s. The old max-wait check would have dropped every two-slot round.
  - Fix: a round is clean when its *first* request did not wait (`min`). Queueing between our own requests is real and part of the measurement.
  - Evidence: `test_queueing_between_our_own_slots_is_not_a_dirty_round` fails on the old rule and passes now.
- Finding (review, medium-high): with `--slots 0,1` every slot is ours, so the slot watcher ignored all of them. A foreign agent request landing on slot 0 after our slot-0 reply (while slot 1 still ran) would have gone unseen. The token counter misses it too, because llama.cpp only counts a prompt after it is read.
  - Fix: `one_round` records when each reply arrived. The watcher marks the round dirty if one of our slots is busy again more than 1 s after its reply.
  - Evidence: `test_foreign_work_on_a_slot_we_already_freed_is_dirty`, `test_a_slot_busy_just_after_its_reply_is_not_dirty`. Live gpu2: a slot reads idle 0.018 s *before* its reply arrives, so 1 s grace cannot cause false drops.
- Finding (review, low): a fast-failing request waited for the other slot's request (up to 600 s) before the error reached the restore.
  - Fix: `one_round` raises the first failure at once.
  - Evidence: `test_one_round_fails_fast_without_waiting_for_the_other_slot`.
- Finding (review, low): `to_round_sample` divided by `prompt_ms`/`predicted_ms` with no zero guard.
  - Fix: guarded like `to_sample`.
  - Evidence: `test_zero_server_time_does_not_crash_the_round`.
- Finding (review, low): `--slots ""`, `--slots 0,`, `--prompt-sentences 0` gave a traceback or a near-empty prompt.
  - Fix: refused with exit 2 before anything is touched.
  - Evidence: `test_bad_flags_are_refused_cleanly`.
- Finding (review of the hold script, medium): the second gpu2 hold went through `_acquire_extra`, which does not mark a hold request that failed mid-flight as unknown. A Ctrl+C or a bus timeout there could leave an untracked operator hold. On the chat card it would have no expiry.
  - Fix: the script takes that hold through `PoolHolds._hold` (ledger row before the request; a failed request triggers a sweep of the pool's lease table on release) and checks the grant's role itself.
  - Evidence: `test_second_hold_rpc_dying_mid_flight_is_swept`.
- Finding (review of the hold script, medium): a release that failed was dropped from the held list, but the script still printed "released".
  - Fix: success is read from the ledger (a failed release keeps its id). The script exits 4 and prints the `--release-leftovers` command.
  - Evidence: `test_failed_release_is_reported_not_called_released`.
- Finding (review of the hold script, low): a signal that landed as the last hold was granted still printed READY; tests patched shared `asyncio` functions.
  - Fix: `stop.is_set()` is checked before READY. The tests now patch only the script's own `POLL_SEC`/`RETRY_SEC`, and send a real SIGTERM.
  - Evidence: `test_signal_while_holding_releases`.
- Finding (review of the hold script, low): the free-slot check read "free" for a 1-slot profile or an unloaded seat, and pool-read errors waited silently.
  - Fix: it requires role status `confirmed` and `slots >= 2` (both fields checked against live `/v1/pool`). The first read error is printed.
  - Evidence: `test_free_slot_reads_the_pool_lease_table`.
- Not changed (review, informational): `gen_avg_w` in a multi-slot row is whole-round watts, not write-phase watts (documented in `to_round_sample`); only slot 0 is warmed up (clocks are per card, not per slot).

## Restart required

```text
No restart required.
```

To rerun the bench: on athena, hold the cards (the agent lane waits while they are held):

```bash
ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/gpu_pool_clear_gpu2.py --minutes 50
```

After READY, run on circe (as root, ~25 min):

```bash
sudo python3 /mnt/scripts/Orion-Sapienform/scripts/gpu_power_bench.py --gpu 2 --port 8016 --caps 250,150 --prompt-sentences 480 --n-predict 4096 --slots 0,1
```

## Risks / concerns

- Severity: low
  - Concern: one prompt shape (16k in, 4k out per slot). The real distribution is wide (p10 64 tokens in, p90 16k).
  - Mitigation: this is the heavy end. If 150 W holds up here and in the first short bench, it holds across the range.
- Severity: low
  - Concern: the hold script pauses the agent lane on circe (gpu1 and gpu2) for as long as it runs.
  - Mitigation: `--minutes` caps it (default 60), and every exit path releases. `--release-leftovers` covers a SIGKILL.
- Severity: low
  - Concern: diffusion or world-model on the same card still shows only as power noise.
  - Mitigation: unchanged from #2945. Rounds where another server slot works are still dropped.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2963

🤖 Generated with [Claude Code](https://claude.com/claude-code)
