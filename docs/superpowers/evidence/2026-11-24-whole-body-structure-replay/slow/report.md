# The slow test: are Orion's modes stable over longer blocks? (#3296)

Run 2026-11-24 by `scripts/whole_body_structure_replay.py --stability-only` (seed 3290). Numbers
are in `stability.json`.

## Question

The replay found no mode stable from one day to the next (bar 0.7). Do the modes settle when the
blocks are longer? Each span is split into back-to-back blocks of L days. Each block's top 8 modes
are matched to the next block's. **Chance** is the same match between two shifted copies of the
data, in which every channel comes from a different time, so neither copy holds a real mode. A
rank is stable when its median match clears 0.7 **and** every one of 20 chance draws.

## Whole body (55 channels, 8 days)

| Block | Pairs | Rank 1 | Rank 2 | Rank 3 | Chance (median, rank 1-3) |
|---|---|---|---|---|---|
| 1 day | 7 | 0.56 | 0.63 | 0.47 | 0.25 |
| 2 days | 3 | 0.69 | 0.15 | **0.71** | 0.28 |
| 4 days | 1 | 0.87 | 0.55 | 0.54 | 0.40 (max 0.89) |

- The day-to-day matches (0.56, 0.63) are more than twice chance. The modes are real. They drift.
- Rank 1 gets more stable as blocks get longer: 0.56, then 0.69, then 0.87. That is what a slow
  mode looks like. But the 4-day value is one pair, and one chance draw reached 0.89, so it is not
  evidence yet.
- Rank 3 passes at 2 days, on 3 pairs. Rank 2 collapses at 2 days (0.15), most likely because two
  modes with near-equal size swap places. One stable rank is not two organs.

## Body only (16 channels, 27 days)

| Block | Pairs | Rank 1 | Rank 2 | Chance (median, rank 1-2) | Chance max (rank 1-2) |
|---|---|---|---|---|---|
| 1 day | 26 | 0.56 | 0.59 | 0.43, 0.48 | 0.64, 0.60 |
| 2 days | 12 | 0.65 | 0.70 | 0.43, 0.41 | 0.81, 0.71 |
| 3 days | 8 | 0.65 | **0.72** | 0.31, 0.33 | 0.64, 0.69 |
| 5 days | 4 | 0.74 | 0.60 | 0.47, 0.47 | 0.87, 0.68 |
| 7 days | 2 | 0.69 | 0.52 | 0.39, 0.43 | 0.78, 0.56 |

- With only 16 channels, chance is high: a random pair of mode sets matches at about 0.45, and
  some draws reach 0.8.
- Day to day, body modes sit barely above chance median and below chance max.
- The earlier replay's 0.66 and 0.67 for body ranks 1-2 came from day blocks cut at 04:00 UTC.
  Cut at 08:00 UTC, the same data reads 0.56 and 0.59. A mode whose stability depends on where
  midnight falls is not stable. A 0.65 bar would have passed noise.
- No block length gives two stable body ranks.

## Verdict

- **No organs yet**, at any block length. Φ on the integration rung (#3296) still has nothing
  stable to cut between.
- **The whole body is the place to look, not the body alone.** Its modes are far above chance and
  rank 1 settles as blocks get longer. The body's modes are close to chance.
- **What would settle it:** the whole-body block needs about 3 weeks of bus history to get 5 or
  more pairs of 4-day blocks. Bus history started 11-13, so re-run about 12-08.
