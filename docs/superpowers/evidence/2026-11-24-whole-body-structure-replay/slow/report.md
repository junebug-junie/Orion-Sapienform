# The slow test: are Orion's modes stable over longer blocks? (#3296)

Run 2026-11-24 by `scripts/whole_body_structure_replay.py --stability-only` (seed 3290). Numbers
are in `stability.json`.

## Question

The replay found no mode stable from one day to the next (bar 0.7). Two questions:
1. Is a day-to-day match of 0.6 real, or what random data would give anyway?
2. Do the modes settle when the blocks are longer?

**Method:**
1. Split each span into back-to-back blocks of L days.
2. Match each block's top 8 modes to the next block's.
3. Build **chance** the same way on shifted copies of the data. In a shifted copy, every channel
   comes from a different time. Each chance draw is the median over the same number of pairs as
   the real value, and there are 20 draws.

A rank is **stable** when its median match clears 0.7 **and** beats every chance draw.

**Two limits of the chance floor:**
- **It is conservative at long blocks.** The copies then overlap in time and can keep slow shared
  structure. In the whole body, the closest two channels in a copy are 12-20 buckets (1-1.7 h)
  apart.
- **It is only a screen.** There are 64 rank-and-length checks, and each has about a 1 in 21
  chance of passing by luck. One pass alone is a screen, not proof.

## Whole body (55 channels, 8 days, first block 11-16 04:00 UTC)

| Block | Pairs | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Chance max (rank 1-4) |
|---|---|---|---|---|---|---|
| 1 day | 7 | 0.56 | 0.63 | 0.47 | 0.51 | 0.33, 0.33, 0.31, 0.31 |
| 2 days | 3 | 0.69 | 0.15 | **0.71** | 0.65 | 0.36, 0.39, 0.37, 0.30 |
| 4 days | 1 | **0.87** | 0.55 | 0.54 | 0.54 | 0.65, 0.60, 0.50, 0.41 |

## Body only (16 channels, 27 days, first block 10-28 08:00 UTC)

| Block | Pairs | Rank 1 | Rank 2 | Chance max (rank 1-2) |
|---|---|---|---|---|
| 1 day | 26 | 0.56 | 0.59 | 0.46, 0.46 |
| 2 days | 12 | 0.65 | 0.70 (0.696) | 0.52, 0.51 |
| 3 days | 8 | 0.65 | **0.72** | 0.66, 0.54 |
| 5 days | 4 | **0.74** | 0.60 | 0.68, 0.59 |
| 7 days | 2 | 0.69 | 0.52 | 0.63, 0.64 |

## What it shows

- **The modes are real.** Day to day, every top mode beats every chance draw. This is true in the
  whole body (0.56-0.63 against at most 0.33) and in the body alone (0.56-0.59 against at most
  0.46). They are not noise. They drift.
- **The 0.7 bar is the only thing they fail.** At 0.7, about half of a mode's pattern carries
  over from one block to the next (0.7² ≈ 0.49). At 0.6 it is about a third.
- **Longer blocks help.**
  - Whole-body rank 1 rises from 0.56 (1 day) to 0.69 (2 days) to 0.87 (4 days). The 4-day value
    is one pair.
  - Body ranks 1-2 rise from about 0.57 to about 0.67 at 2-3 days.
- **Ranks swap.** Whole-body rank 2 drops to 0.15 at 2 days, most likely because two modes with
  near-equal size trade places. Matching by rank is weak when two modes are about the same size.
- **Where the day is cut matters.** The first replay cut body days at 04:00 UTC (first block
  10-28 04:00) and read body ranks 1-2 at 0.66 and 0.67. This run cuts at 08:00 UTC and reads 0.56
  and 0.59. The spans also differ by 4 hours at the ends. A 0.1 swing from the cut alone means a
  single day-to-day number has about ±0.05 of noise in it.
- **No length gives two stable ranks under the 0.7 bar.** So there are still no organs for Φ.

## Verdict

- The modes are real but slow. Day-long blocks are too short to see them clearly.
- The fixed 0.7 bar is now the deciding choice, not chance. Juniper's 0.65 would pass:
  - whole-body ranks 1, 3 and 4 at 2 days (on only 3 pairs);
  - body rank 2 at 2 days, and body rank 1 at 2 days only just misses (0.646).
- **Recommendation:** hold the bar at 0.7. Measure at 2-4 day blocks. Re-run around 12-08, when bus
  history covers 24+ days (5 pairs of 4-day blocks).
