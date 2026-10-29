## Summary

- The first bench (2k-token prompt, 512 tokens out, one slot) did not look like real gpu2 work. It wrote at 49 tok/s; real work writes at 30 tok/s.
- New `--prompt-sentences` flag sets the prompt size (480 sentences = 16,221 tokens on Bonsai 2, measured live).
- New `--slots 0,1` mode sends one request per slot at once (a "round"), like the agent lane does.
- A round records all-in joules per written token over the whole round, from the first prompt read to the last reply. Per-phase energy is not split, because one slot reads while the other writes.
- Each summary row now also has the card's output rate (written tokens per second of the whole round) and all-in J/token. Both are compared against the highest cap.

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

## Current architecture

`scripts/gpu_power_bench.py` sends one request at a time to one slot, with a fixed 60-sentence prompt.

## Architecture touched

Only the bench script and its tests. No service, config, schema or bus change.

## Files changed

- `scripts/gpu_power_bench.py`: `--prompt-sentences`, `--slots`, `one_round`, `to_round_sample`, `busy_slots`/`SlotWatch` accept several slots, two new `Sample` fields, wider summary table.
- `tests/test_gpu_power_bench.py`: 10 new tests (prompt size, multi-slot ignore, concurrent send, failure re-raise, round energy math, single-slot round, two-slot run end-to-end, bad slots refused, own-slot queueing kept).

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
.venv/bin/python -m pytest tests/test_gpu_power_bench.py -q   -> 40 passed
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
```

## Review findings fixed

- Finding (self, from the live smoke): the server reads the two prompts one after another, so the second request always "waited" ~25 s. The old max-wait check would have dropped every two-slot round.
  - Fix: a round is clean when its *first* request did not wait (`min`). Queueing between our own requests is real and part of the measurement.
  - Evidence: `test_queueing_between_our_own_slots_is_not_a_dirty_round` fails on the old rule and passes now.

## Restart required

```text
No restart required.
```

Run on circe (as root, ~25-30 min):

```bash
sudo python3 /mnt/scripts/Orion-Sapienform/scripts/gpu_power_bench.py --gpu 2 --port 8016 --caps 250,150 --prompt-sentences 480 --n-predict 4096 --slots 0,1
```

## Risks / concerns

- Severity: low
  - Concern: one prompt shape (16k in, 4k out per slot). The real distribution is wide (p10 64 tokens in, p90 16k).
  - Mitigation: this is the heavy end. If 150 W holds up here and in the first short bench, it holds across the range.
- Severity: low
  - Concern: diffusion or world-model on the same card still shows only as power noise.
  - Mitigation: unchanged from #2945. Rounds where another server slot works are still dropped.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
