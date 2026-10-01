# GPU pool stage 7.1: two Bonsai runs on one card, and the prompt-cache bleed canary

**Status: NOT RUN YET.** This is the skeleton. Every number below is a placeholder until Juniper's
circe run fills it in from that run's `fieldnote_draft.md` and `summary.json`.

Spec: `docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md`, acceptance checks 1 and 2
and PR row 7.1. Tool: `scripts/bench/stage7_1_bakeoff.sh`.

## What this decides

Stage 7.2 gives the gpu2 agent seat a two-slot Bonsai worker, so the card can serve two Orion runs
at once. Two things have to be true first:

1. **Two at once must be worth it.** In #2434, two concurrent Bonsai runs at 4x65K got 23.4 tok/s
   each (~47 total) against ~52 for one run alone: slower in total, not faster. The layout stage 7
   wants, 2 slots x 131K, has never been measured with two runs active at depth.
2. **One conversation must never leak into another.** llama.cpp #27148 (open) can paste a finished
   conversation into a different slot through the host-RAM prompt cache. On the agent lanes that
   would mix Orion runs with Claude Code turns that can carry Juniper's conversation. The
   2026-10-01 probe on metacog/fast (stock b10398, dense Qwen3-8B) did not reproduce it, but the bug
   was reported on a hybrid model, and Bonsai (Prism build) is the model 7.2 makes multi-slot.

## Setup (filled from the run)

| | Bonsai pass | Q4 27B pass (`--with-q4`, optional) |
|---|---|---|
| Card | gpu2 (V100 32 GB) | gpu2 |
| Image / build | `llamacpp-bonsai-prism:server-local-volta`, build _TBD_ | `orion-llamacpp-host:0.1.0`, build _TBD_ |
| Layout | `--parallel 2`, ctx 262,144 (2 x 131,072) | `--parallel 2`, ctx 131,072 (2 x 65,536; 2x131K does not fit) |
| Flash attention | on (profile; launch line in `bonsai/argv.txt`) | as the stock image's profile has it (`q4/argv.txt`) |
| Prompt cache | llama.cpp defaults (`--cache-ram 8192`, idle-slot publishing on) | same |
| Peak VRAM / temp / power | _TBD_ (`bonsai/gpu2.csv`) | _TBD_ |

## Check 1: throughput at depth

Method: per depth, one cold request times prefill; then the same prompt again (it is now in the
slot) decodes 512 tokens with `ignore_eos`, thinking off. For 2 runs, two different prompts are
prefilled together, then both decode together at that depth. Two repeats; medians. "Total" is the
sum of the two runs' own decode tok/s; the wall-clock aggregate is shown beside it as a cross-check.

**Rule:** 2 runs' total tok/s >= 1.3 x 1 run's tok/s at **every** depth.

| Depth (tokens) | 1 run tok/s | 2 runs total tok/s (wall) | Ratio | Pass (>= 1.3) | Prefill 1 run (s) | Prefill 2 runs (s) |
|---:|---:|---:|---:|:--:|---:|---:|
| 14,000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 32,000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 61,000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 100,000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Bench verdict: _TBD_ (PASS / FAIL / INCOMPLETE)

## Check 2: bleed canary

Each conversation has a random codeword (`NX` + 10 characters) in a ~4,600-token document, and each
tool-call turn returns a fresh ledger value the model must report. Every request offers a tool (the
agent-lane shape). Turn mix: recall, chat, tool call + tool result, thinking (reasoning on), chat.

- **Pairs:** 2 conversations x 100 turns, both requests sent at the same moment.
- **Interleaved:** 4 conversations x 50 turns over 2 slots, consecutive requests switching
  conversation, so every turn lands on a slot another conversation just used (LRU reuse and RAM-cache
  restore all the time).
- **Fresh:** a new one-turn conversation every 10 rounds (the upstream symptom).

Detectors:
1. another conversation's codeword or ledger value appears in content, reasoning, or tool-call arguments;
2. `timings.cache_n` exceeds the tokens this prompt truly shares with any earlier prompt (token
   prefix via `/apply-template` + `/tokenize`, plus the earlier reply's length when that whole
   prompt is a prefix, floored at the shared chat-template header).

**Rule:** zero hits on either detector. A run where the model does not repeat its own codeword on
recall turns (< 80%) is **WEAK**, not PASS: the detector was not shown able to see anything.

| Pass | Requests | Conversation turns | Bleed (detector 1) | cache_n over true prefix (2) | Own-codeword recall | Tool calls emitted | Thinking turns | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| Bonsai (Prism) | _TBD_ | _TBD_ / 400 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Q4 27B (stock) | _TBD_ | _TBD_ / 400 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

The canary is capped by wall time (40 min Bonsai, 35 min Q4); if the cap stops it early, the run
says so and the verdict is WEAK, not PASS.

## Go / no-go for 7.2

| Bench | Canary | Decision |
|---|---|---|
| PASS | PASS | **Go:** 7.2 ships the 2-slot Bonsai profile on `agent-gpu2`. |
| FAIL | any | **No-go on 2 slots.** Per spec: stage 7.2 runs Bonsai at 1 slot, or re-test at N=2 with flash attention off (#2434 measured 41 tok/s each with FA off at N=2). |
| any | FAIL | **Juniper's answer 2 applies:** turn the idle-slot RAM prompt cache off (`--cache-ram 0 --no-cache-idle-slots`) on every multi-slot lane, metacog and fast included, and re-run the canary with it off. Needs two profile fields + `append_flag` lines in `services/orion-llamacpp-host/app/main.py`. |
| any | WEAK | Not a pass. Re-run the canary (longer cap, or fix what made the model miss its own codeword). |

## Tool validation (2026-10-01, before the real run)

The client was run end to end against circe metacog (:8012, b10398, Qwen3-8B, 4 x 4,096) from
athena, kept tiny (depths <= 2.5K, 64-token decodes, 31 canary requests, paced):

- Bench: prompts landed within 1-3% of the target depth; every warm decode hit the slot's cache
  (`cache_n` = prompt - 1). Ratios were 1.47 then 0.96 at 1K, 0.98 at 2.5K: the same depth
  disagreed between two runs minutes apart, because metacog is a live lane with other traffic on
  the card. **These are not evidence about concurrency**, only that the harness works.
- Canary: 0 bleed, 0 cache over the true prefix, own-codeword recall 1.0, 6 tool calls emitted, the
  thinking turn ran. The first attempt flagged a false positive: the very first request reused
  the 3-4 token chat header from a slot other traffic had used, and the bound had no header floor.
  Fixed (floor = header shared by any two prompts), and the
  turn cycle was fixed so tool follow-ups no longer skip the thinking turn.
- `preflight` was run for real on circe (read-only): it reached the pool from circe and read gpu2
  (1,248 MiB, world-model only), and `run` without `--yes` changed nothing.
- **UNVERIFIED until the real run:** the pool controls inside the Bonsai image on circe
  (`stage7_1_pool_ctl.py`; its first step, `check`, aborts before anything is paused if the
  imports or the bus fail), the 2 x 131K boot, and every number above.

## Raw evidence

On circe: `~/stage7_1_bakeoff/<UTC timestamp>/` with `run.log`, `summary.json`,
`fieldnote_draft.md`, and per pass `bench.json`, `bench.log`, `canary.json`, `canary.log`,
`props.json`, `argv.txt`, `gpu2.csv`, `override-<pass>.yml`.
