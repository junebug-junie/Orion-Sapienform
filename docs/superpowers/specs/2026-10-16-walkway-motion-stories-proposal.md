# Walkway motion stories: proposal

Status: proposal (design only, no service code). Date: 2026-10-16. Requested by Juniper.

Orion can say *who* was on the walkway today ("an unfamiliar person", a named individual). Orion cannot say *what happened*: that someone walked from the mailbox to the patio, or that two people walked past together. This proposal turns each passing object into one plain sentence, using only arithmetic on the detector's boxes. No AI model is involved.

Everything below was measured read-only on the live database (`conjourney`, READ ONLY transactions) on 2026-10-16 between 19:40 and 20:10 UTC. A replay prototype produced the sample stories. The prototype lives outside the repo, in the session scratchpad. It is not committed.

## Arsonist summary

- **The camera does not look often enough to see fast things.** The detector looks at a walkway frame every 5.0 s (median, clean frames). A walker covers about 3 body-heights in that time, so we can follow a walker. A runner covers about 10, a cyclist about 25, a car more than the whole frame. Juniper's example, "a child ran across the yard across 5 frames", needs about **one detection per second** while something moves. Today, runners, cyclists and cars get one box or none, so they get no story. This is the biggest finding, and the first slice cannot fix it.
- **Individual ids cannot be the passages.** Juniper's brief said to reuse the continuation lane (PR #2719) as the tracker. Under the old matching rule, every one of the 9 moving tracks in the replay was spread over 2 to 12 individual ids. The continuation lane only went live at about 18:56 UTC today. In the clean window since then, nobody walked past, so its effect on moving people is **UNVERIFIED**. Its geometry also has a limit: it joins a box only within 1.5 box heights. In the replay, 86% of bicycle steps and 50% of dog steps were longer than that, and 6% of person steps. So stories should link boxes **by geometry, using the same idea as the continuation lane but with a velocity prediction**. The individual id then supplies the name only. This step uses no embeddings and keeps no identity, so it is not a second identity tracker.
- **What works today, at today's cadence:** walkers, people with a dog, a stroller or a bicycle at walking pace, groups walking together, zone paths (mailbox, patio, walkway), duration and direction. The replay produced 158 moving stories and 534 "standing" stories in 32 hours. On 10-15, between 21:00 and 00:00 UTC, there were 94 moving stories.
- **"Standing" is mostly junk until the prop filter covers it.** At night, the patio skeleton and parked bicycles produce dozens of "a person stood on the patio" stories every hour. The existing still-box rule (`vision_individual.stationary`) must gate them.
- **Nothing here has been checked against pixels.** Frames are kept for only 300 s. Only 6% of walkway crops (1,217 of 20,836) carry a thumbnail. Every sample story below is **UNVERIFIED** until a person looks at it. The first patch must save a small frame strip with each story, so Juniper can spot-check it.

## Current architecture

- **Camera and cadence.** `orion-vision-retina-walkway` captures 2560x1920 frames at 2 fps (`WALKWAY_RETINA_FPS=2.0`). `orion-vision-frame-router` sends a subset of them to the athena `orion-vision-host` for GroundingDINO detection (`config/vision_frame_router.yaml` `streams.walkway`). The baseline tier uses `every_n_frames: 5` and a 3 s minimum. The triggered tier uses a 2 s minimum. A tiled pass runs every 20 s (PR #2691). The real limit is the GPU: the P4 is about 100% busy, at about 5.4 s per walkway task plus about 2.3 s per cam0 task.
- **Boxes.** `vision_crop_observation` holds one row per detected box. Each row has `observed_at`, `label` (person/dog/bicycle/stroller/vehicle; `package` and `mail truck` are detected but never become crops), `score`, `box_xyxy` in 2560x1920 px, and `zone`. Since PR #2709 a row also has `individual_id`, and it can have `thumb_ref`. Retention is 7 days (`VISION_CROP_RETENTION_DAYS=7`).
- **Zones.** `config/vision_zones.yaml` defines three polygons in normalised coordinates, and the first match wins. A box belongs to the zone that holds its bottom-centre point. **patio** is lower right, `[[0.58,0.53],[1,0.47],[1,1],[0.56,1]]`. **mailbox** is left-middle, `[[0.15,0.39],[0.38,0.39],[0.38,0.66],[0.15,0.66]]`. **walkway** is everything below the sky line, `y >= 0.14`. Note that mailbox lies inside walkway, so the ordering matters. Loader: `orion/vision/zones.py`.
- **Individuals.** `services/orion-sql-writer/app/vision_individuals.py` `apply_batch` assigns `individual_id` through the face lane, the continuation lane (same individual within 10 s and 1.5 box heights, body cosine >= 0.88, live since ~18:56 UTC today per the container uptime), and re-identification (cosine >= 0.90). It writes `vision_individual` and `vision_individual_sighting` (dwell, `zone_counts`, `last_box_xyxy`, `stationary`). Sightings are kept 90 days.
- **Street line.** `orion/situational/perception_reader.py` `fetch_street_summary` makes `summarize_street`. That gives a captions line, a "who is around" line, expectations, unresolved percepts and patio presence. It feeds chat stance (`services/orion-cortex-exec/app/chat_stance.py`, 420-char street clip) and the situation brief (`orion/situational/context.py`). `orion/situational/street_material.py` feeds reveries, dreams and curiosity.
- **Surprise.** `node:substrate.perception_walkway` (`services/orion-substrate-runtime/app/worker.py`) is a whole-frame embedding prediction error. It is fenced from attention. `vision_attention_score.py` scores sightings. Neither one knows anything about motion.
- **Gap.** Nothing reads two boxes of the same object in time order and asks how it moved.

### Measured: detected-frame interval

Measurement: the time between consecutive distinct `observed_at` values on the walkway. A detected frame shows up here only if it holds at least one crop box. Parked cars make that almost every frame in daylight. At night the measurement undercounts frames.

| Window | Intervals | p10 | p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|---|---|---|
| Clean, 2026-10-16 18:59-19:47 UTC | 589 | 4.87 s | 4.95 s | **5.00 s** | 5.04 s | 5.09 s | 5.28 s |

27 of those 589 intervals (4.6%) were about 2.5 s. One was 7.3 s, and three were about 10 s.

Hourly medians over the last 32 hours range from **2.5 s** (2026-10-15 22:00 and 2026-10-16 01:00, triggered tier, busy) to **7.5 s** (2026-10-15 13:00-20:00). The p90 reaches 10-35 s in quiet night hours. The gap between consecutive boxes of the same individual (32 h, gaps of 30 s or less) is: person p50 2.55 s, p75 5.3 s, p90 9.9 s; dog p50 2.9 s; bicycle p50 10.0 s.

**What this means.** Linking two boxes needs the object to move less than about 4 body-heights between looks. That allows walkers (about 0.8 body-heights per second) at 5 s. Runners (about 2 body-heights per second) need about 2 s. Cyclists (4-6 bicycle-heights per second) need under 1 s. Cars need about 0.4 s.

### Measured: speed in body-heights per second

A fixed oblique camera does not show ground speed in pixels: a far person moves fewer pixels than a near one. The proposal divides the foot-point travel by the object's own box height. Lateral motion scales with distance the same way the box height does, so "box heights per second" is roughly a ground speed in body lengths. Motion toward or away from the camera is foreshortened, so it reads slow. This is a known underestimate. It does not cause false "running" claims.

Data: geometric tracks with 3 or more detections that travelled 2 or more box heights, all 32 h of data. Pre-fix data is used here for calibration only (see Acceptance checks).

| Kind | Tracks | p10 | p25 | p50 | p75 | p90 | max seen |
|---|---|---|---|---|---|---|---|
| person | 62 | 0.03 | 0.08 | 0.25 | 0.45 | 0.57 | 1.44 (2 detections, jitter) / 0.82 (3+ detections) |
| bicycle (bike-heights/s) | 3 | 0.50 | | 0.82 | | 0.82 | 0.82 |
| dog | 1 | 0.38 | | | | | |
| vehicle | 3 | 0.03 | | 0.11 | | 0.11 | 0.28 (clean, sample S9) |

Theory anchor: an adult walks at 1.2-1.5 m/s with a body height of about 1.7 m, which is 0.7-0.9 body-heights per second laterally. A child walks at about 1.1 m/s with a height of about 1.2 m, about 0.9. Jogging at 2.5 m/s is 1.5 (adult) to 2 (child). The live maximum for walkers with 3+ detections, 0.82, sits right at the top of the walking band. That is what the theory predicts. **No runner, riding cyclist or driving car was ever tracked over 3 or more detections.** That matches the cadence limit above. It is not evidence that nothing fast passed.

Thresholds proposed for slice 1 (body-heights per second of net travel):

| Kind | standing | middle class | fast class |
|---|---|---|---|
| person | net travel < 0.5 BH | walking 0.08-1.2 | running >= 1.2 (needs >= 3 detections) |
| dog | < 0.5 BH | walking 0.08-1.5 | running >= 1.5 (>= 3 detections) |
| bicycle | parked (no story) | at walking pace < 1.5 | riding >= 1.5 (>= 3 detections) |
| stroller | parked (no story) | being pushed | |
| vehicle | parked (no story) | creeping < 0.6 | driving >= 0.6 (>= 3 detections) |

The fast classes need 3 or more detections. Two boxes 2.5 s apart can jitter by a full box height, and that alone reads as "running". In the replay, every 2-detection "running" case was jitter: speed 1.44 over 2 s.

## How a story is built (the prototype)

Inputs are rows of `vision_crop_observation` for one stream, in `observed_at` order. Everything below is deterministic.

1. **Link boxes into passages.** The first link joins a box to the same label within 11 s and within **4 box heights**. Later links use a constant-velocity prediction from the last two boxes, with a **2 box-height** gate around the predicted point. The box height may change by at most 2x per link. One box joins one passage per frame (cannot-link). Assignment is greedy by distance. A passage ends after 11 s with no link. The 4-height first gate is wider than the continuation lane's 1.5, because at 5 s cadence a walker moves about 3 heights.
2. **Drop props.** Drop passages whose individual is `stationary`. Drop bicycles, strollers and vehicles that travelled less than 0.5 box heights: those are parked, not a story. *Gap found:* before ids existed, nothing caught the patio skeleton. Slice 1 must also drop passages whose box overlaps a known `vision_individual.still_box`.
3. **Kinematics.** Track the foot point (bottom-centre). Compute net travel and path length in box heights, duration, and speed = net travel / duration.
4. **Direction.** Left-to-right or right-to-left when the foot point moves at least 8% of the frame width. "Toward/away from the house" when it moves at least 6% of the frame height, or when the box height changes by 1.4x or more (log ratio 0.35). The camera looks out from under the porch, so moving down the frame means moving toward the house.
5. **Zone path.** Each detection gets the zone of its foot point. A 3-wide majority filter smooths the zones, because a person standing on the patio edge otherwise "crosses" a dozen times. The result is collapsed to the zones entered, and standing passages get only their majority zone.
6. **Together.** Two moving passages are together when they share 2 or more instants (within 1 s), stay within 2 box heights at every shared instant, and head the same way (cosine >= 0.7). A person passage with a dog, bicycle or stroller partner becomes "a person with a dog". Person with person becomes "together with 1 other person".
7. **Name.** Slice 1 rule: name an individual only when one *named* `individual_id` covers 2/3 or more of the passage's crops and no other named id appears. Otherwise say "a person". The prototype used a looser rule, and sample S10 shows why the strict one is needed.
8. **Sentence.** Pattern: `<kind or name> [with <companion>] <verb> <direction> <zone phrase>, over <N> s (<k> detections)[, together with ...].`

Replay totals (32 h of walkway crops, 20,836 boxes): 2,161 passages. Of those, 692 have 2 or more detections and travel or dwell: 158 moving, 534 standing, and 22 together-pairs. Of the 158 moving stories, 75 rest on only 2 detections, which is weak evidence. Slice 1 publishes a moving story only with **3 or more detections**. Shorter ones are counted as glimpses only.

## Ten sample stories from the replay

Read these with care:

- Only S9 and S10 come from clean frames (after 2026-10-16 18:59 UTC). Between 18:59 and 19:47 UTC (13:00-13:47 in Juniper's MDT time zone), nobody walked along the walkway.
- S1-S8 come from frames the RTSP corruption may have touched. The smear covered roughly the bottom 40% of the frame, and some frames went gray.
- **None of these stories has been checked against pixels.** The frames are gone (300 s retention) and the crops carry no thumbnails.
- Times are UTC.
- Individual ids are the first 8 characters, and names are never printed.
- Each evidence table shows the first 3 and the last 2 rows when a story has more than 6.

#### S1. 2026-10-15 16:33:29 to 16:33:49 UTC (before the stream fix)
> A person with a dog walked right to left and toward the house on the walkway, over 20 s (3 detections).

Maths: 3 detections, median box height 283 px, net travel 2.1 box heights (path 2.7), speed 0.10 box heights/s, foot point moved -0.18 of the frame width and +0.20 of its height, zones walkway, individual ids none.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 16:33:29.51 | person | 0.52 | 1404,491,1463,659 | walkway | - |
| 16:33:39.53 | person | 0.67 | 909,717,1029,999 | mailbox | - |
| 16:33:49.55 | person | 0.68 | 893,740,1055,1035 | walkway | - |

#### S2. 2026-10-15 21:39:14 to 21:39:24 UTC (before the stream fix)
> A person walked right to left and toward the house on the walkway, over 10 s (4 detections), together with 1 other person moving the same way.

Maths: 4 detections, median box height 179 px, net travel 3.8 box heights (path 3.8), speed 0.38 box heights/s, foot point moved -0.26 of the frame width and +0.07 of its height, zones walkway, individual ids none.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 21:39:14.34 | person | 0.58 | 1840,646,1946,829 | walkway | - |
| 21:39:19.30 | person | 0.42 | 1722,668,1809,844 | walkway | - |
| 21:39:21.86 | person | 0.42 | 1436,748,1531,924 | walkway | - |
| 21:39:24.33 | person | 0.46 | 1179,775,1260,968 | walkway | - |

#### S3. 2026-10-15 22:16:40 to 22:16:52 UTC (before the stream fix)
> A person with a bicycle walked right to left on the walkway, over 13 s (4 detections).

Maths: 4 detections, median box height 185 px, net travel 4.3 box heights (path 4.6), speed 0.34 box heights/s, foot point moved -0.31 of the frame width and -0.05 of its height, zones walkway, individual ids none.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 22:16:40.10 | person | 0.43 | 2490,681,2558,881 | walkway | - |
| 22:16:47.67 | person | 0.49 | 2250,611,2328,789 | walkway | - |
| 22:16:50.13 | person | 0.60 | 2087,627,2212,818 | walkway | - |
| 22:16:52.68 | person | 0.55 | 1692,642,1779,791 | walkway | - |

#### S4. 2026-10-15 22:37:15 to 22:38:00 UTC (before the stream fix)
> A person walked left to right and away from the house from the mailbox, through the walkway, to the patio, over 45 s (16 detections).

Maths: 16 detections, median box height 207 px, net travel 6.6 box heights (path 7.9), speed 0.15 box heights/s, foot point moved +0.53 of the frame width and -0.07 of its height, zones mailbox > walkway > patio, individual ids none.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 22:37:15.58 | person | 0.55 | 684,1130,817,1252 | mailbox | - |
| 22:37:18.05 | person | 0.55 | 669,1145,798,1265 | mailbox | - |
| 22:37:23.11 | person | 0.50 | 706,1135,929,1270 | walkway | - |
| ... 11 more | | | | | |
| 22:37:53.07 | person | 0.60 | 1343,857,1489,1085 | walkway | - |
| 22:38:00.63 | person | 0.55 | 2058,823,2175,1124 | patio | - |

#### S5. 2026-10-15 22:53:55 to 22:54:35 UTC (before the stream fix)
> A person walked right to left and toward the house from the walkway, through the mailbox, to the walkway, over 40 s (15 detections).

Maths: 15 detections, median box height 214 px, net travel 4.4 box heights (path 10.5), speed 0.11 box heights/s, foot point moved -0.19 of the frame width and +0.41 of its height, zones walkway > mailbox > walkway, individual ids none.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 22:53:55.97 | person | 0.44 | 864,562,904,653 | walkway | - |
| 22:53:58.45 | person | 0.48 | 918,559,963,659 | walkway | - |
| 22:54:00.92 | person | 0.51 | 960,556,1006,668 | walkway | - |
| ... 10 more | | | | | |
| 22:54:33.43 | person | 0.41 | 614,1028,755,1261 | mailbox | - |
| 22:54:35.89 | person | 0.47 | 306,1183,475,1447 | walkway | - |

#### S6. 2026-10-16 00:03:19 to 00:03:32 UTC (before the stream fix)
> A person with a stroller walked right to left and toward the house on the walkway, over 12 s (6 detections).

Maths: 6 detections, median box height 203 px, net travel 6.1 box heights (path 5.5), speed 0.48 box heights/s, foot point moved -0.39 of the frame width and +0.38 of its height, zones walkway, individual ids none.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 00:03:19.88 | person | 0.48 | 1370,524,1426,671 | walkway | - |
| 00:03:22.41 | person | 0.44 | 1259,571,1308,699 | walkway | - |
| 00:03:24.86 | person | 0.42 | 1128,628,1215,802 | walkway | - |
| 00:03:27.41 | person | 0.45 | 940,703,1047,935 | walkway | - |
| 00:03:29.89 | person | 0.47 | 660,833,792,1122 | mailbox | - |
| 00:03:32.37 | person | 0.60 | 293,1043,528,1402 | walkway | - |

#### S7. 2026-10-16 01:22:38 to 01:22:46 UTC (before the stream fix)
> A dog walked right to left and toward the house on the walkway, over 7 s (3 detections).

Maths: 3 detections, median box height 194 px, net travel 2.8 box heights (path 3.4), speed 0.38 box heights/s, foot point moved -0.20 of the frame width and +0.12 of its height, zones walkway, individual ids none.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 01:22:38.68 | dog | 0.42 | 1261,948,1473,1131 | walkway | - |
| 01:22:43.64 | dog | 0.43 | 641,1123,929,1356 | walkway | - |
| 01:22:46.12 | dog | 0.36 | 758,1164,971,1359 | walkway | - |

#### S8. 2026-10-16 18:16:59 to 18:17:14 UTC (before the stream fix)
> A person walked right to left and toward the house from the walkway, through the mailbox, to the walkway, over 15 s (7 detections).

Maths: 7 detections, median box height 274 px, net travel 3.8 box heights (path 4.1), speed 0.25 box heights/s, foot point moved -0.36 of the frame width and +0.24 of its height, zones walkway > mailbox > walkway, individual ids d5ecc760, e9c3fe6a, fca7797a.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 18:16:59.05 | person | 0.42 | 1153,659,1269,898 | walkway | e9c3fe6a |
| 18:17:01.64 | person | 0.41 | 808,758,937,1033 | mailbox | e9c3fe6a |
| 18:17:04.10 | person | 0.41 | 533,866,670,1173 | mailbox | e9c3fe6a |
| ... 2 more | | | | | |
| 18:17:11.56 | person | 0.38 | 549,982,675,1228 | mailbox | d5ecc760 |
| 18:17:14.09 | person | 0.41 | 221,1108,356,1354 | walkway | fca7797a |

#### S9. 2026-10-16 19:33:21 to 19:33:36 UTC (clean frames)
> A vehicle crept right to left and toward the house on the walkway, over 15 s (4 detections).

Maths: 4 detections, median box height 58 px, net travel 4.2 box heights (path 4.5), speed 0.28 box heights/s, foot point moved -0.09 of the frame width and +0.03 of its height, zones walkway, individual ids 1960d2f4, 65925658, 87a843ab, f7e337bd.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 19:33:21.10 | vehicle | 0.36 | 1922,329,1993,374 | walkway | f7e337bd |
| 19:33:31.10 | vehicle | 0.39 | 1806,351,1858,403 | walkway | 1960d2f4 |
| 19:33:33.55 | vehicle | 0.46 | 1722,353,1835,421 | walkway | 87a843ab |
| 19:33:36.09 | vehicle | 0.47 | 1694,363,1747,428 | walkway | 65925658 |

#### S10. 2026-10-16 19:33:26 to 19:33:53 UTC (clean frames)
> Named person <4354c877> stood on the patio, over 27 s (12 detections).

Maths: 12 detections, median box height 1176 px, net travel 0.8 box heights (path 1.7), speed 0.03 box heights/s, foot point moved +0.24 of the frame width and -0.38 of its height, zones patio, individual ids 05220624, 4354c877, a44ecc9f, f5242d40.

| t (UTC) | label | score | box x1,y1,x2,y2 (px) | zone | individual |
|---|---|---|---|---|---|
| 19:33:26.07 | person | 0.47 | 1436,705,2053,1914 | patio | 4354c877 |
| 19:33:28.59 | person | 0.49 | 1296,797,2021,1915 | patio | 4354c877 |
| 19:33:31.10 | person | 0.51 | 1182,808,1906,1916 | patio | 4354c877 |
| ... 7 more | | | | | |
| 19:33:51.04 | person | 0.49 | 1667,610,2086,1619 | patio | a44ecc9f |
| 19:33:53.55 | person | 0.43 | 2216,468,2493,1180 | patio | 05220624 |

### What the samples teach

- **S1 may be a wrong link.** Between the first box and the second, the box grew 1.7x and its foot point moved about 575 px in 10 s. That fits inside the 4-height first gate. A 10 s gap is long, so slice 1 should shrink the first gate in proportion to the gap.
- **S2 and S8 show the old id rule splitting one walker.** S8 spans 3 individual ids in 15 s. Under the old rule (before ~18:56 UTC), individual ids cannot define a passage.
- **S9 is the continuation lane, live, splitting a far car.** It gave 4 vehicle ids in 15 s. The box is 58 px tall, and there is a 10 s gap between the first two boxes. The geometric link held. The id lane did not.
- **S10 is the dangerous case.** The box reaches the bottom edge of the frame (y2 = 1914-1916 of 1920), so the "foot point" is the frame edge, not a foot. The track carries **two different named ids**, and the prototype printed a name. Slice 1 needs two rules. First, a box that touches the frame edge gives no zone and no travel. Second, the strict naming rule from step 7 applies.
- **S4 "away from the house" rests on dy = -0.07,** just over the 0.06 threshold. Slice 1 should raise the depth threshold to 0.10.

## Proposal mode (CLAUDE.md "Proposal mode before invasive cognition changes")

- **Capability that changes.** Orion's street line gains "what moved", not only "who": up to 2 recent motion sentences per 15 minutes. Street material for reveries, dreams and curiosity gains the same sentences. Orion can then say "two people walked from the mailbox toward the patio", grounded in box rows.
- **Data touched.** Reads: `vision_crop_observation` and `vision_individual` (label, stationary, still_box). Writes: one new derived table, `vision_motion_story`, plus small frame-strip JPEGs.
  - The table holds numbers, a sentence and crop ids. It holds no embeddings and no new identity.
  - Retention: stories 90 days, the same as sightings. Strips 7 days, the same as crops.
  - Retention for crops (7 d) and sightings (90 d) is unchanged.
- **Privacy boundary.** Juniper chose unconstrained recording of the walkway, patio included (2026-10-15, `manual_migration_walkway_privacy_v2.sql`). Stories add no new kind of data. They are arithmetic on boxes that are already stored. The naming boundary is unchanged: a name appears only when Juniper gave it through the ask flow, and only under the strict rule in step 7. No story is ever sent outside Orion.
- **Trace that proves it worked.** One log line per reducer pass: `[MOTION_STORY] stream=walkway closed=N published=M glimpses=G dropped_props=P cursor=<ts>`. Each published row carries its `crop_ids`, so any sentence can be traced back to its boxes. The Hub card shows the sentence next to its frame strip. `fetch_street_summary` with a fixed `now` returns the motion line in replay.
- **Dangerous failure modes.**
  1. A **wrong name**: a passer-by gets a family member's name because the id lane mixed them (S10). This is the worst one. Mitigation: the strict naming rule, plus "no name if any crop in the passage touches the frame edge".
  2. **Invented speed**: "ran" from box jitter. Mitigation: fast classes need 3 or more detections.
  3. **Prop spam**: every night says "a person stood on the patio". Mitigation: the still-box gate and the stationary flag.
  4. **Quiet street read as calm**: at 5 s cadence, a cyclist who never gets a story reads as "nothing happened". The street line must never say "nobody passed". Absence claims stay with the expectation reducer.
- **Disable / roll back.**
  - `VISION_MOTION_STORIES_ENABLED=false` (orion-sql-writer) stops the reducer.
  - `STREET_MOTION_LINE_ENABLED=false` (consumers) removes the line from the street summary.
  - The table is additive. Dropping it removes the feature completely, and nothing else reads it.
  - Per Juniper's standing rule, both flags ship **ON**.

## Missing questions

1. **Cadence.** Is one detection per second worth a GPU while something moves? Juniper's own example needs it. There are two options with no model in them:
   - (a) a cheap pixel-difference motion box on the retina's 2 fps frames, used only to carry a passage between detector frames;
   - (b) a second detector slot on another GPU.
   Without one of these, runners, cyclists and cars stay out of reach.
2. **Shadow first?** Should stories go to Orion's street line from day one (flags on, per the standing rule)? Or should they stay Hub-only until Juniper has spot-checked 50 of them?
3. **Child.** Box height compared with the usual adult height at the same foot row is pure arithmetic. Measured person box-height medians by foot row: 195 px at y 800-959, 192 px at y 960-1119, 344 px at y 1120-1279. A person well under the usual height could be told as "a small person". Does Juniper want that word in stories? The tables are polluted by sitting people and props today, so this would be slice 3 at the earliest.

## Proposed schema / API changes

New table, created by manual migration `services/orion-sql-db/manual_migration_vision_motion_story_v1.sql`:
- Header: `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`.
- Additive `CREATE TABLE` and `CREATE INDEX` only, so it fits the post-pull auto-apply allowlist.

```sql
CREATE TABLE IF NOT EXISTS vision_motion_story (
  story_id          text PRIMARY KEY,          -- sha1(stream_id | first crop_id)
  stream_id         text NOT NULL,
  kind              text NOT NULL,             -- person|dog|bicycle|stroller|vehicle
  companion_kinds   text[] NOT NULL DEFAULT '{}',
  together_story_ids text[] NOT NULL DEFAULT '{}',
  individual_id     text,                      -- only when the strict naming rule holds
  started_at        timestamptz NOT NULL,
  ended_at          timestamptz NOT NULL,
  detection_count   integer NOT NULL,
  speed_class       text NOT NULL,             -- standing|walking|running|walked|riding|pushed|creeping|driving
  speed_bh_per_sec  real NOT NULL,
  net_travel_bh     real NOT NULL,
  path_bh           real NOT NULL,
  dir_lateral       smallint NOT NULL,         -- -1 right-to-left, 0, +1 left-to-right
  dir_depth         smallint NOT NULL,         -- -1 away from the house, 0, +1 toward
  zone_path         text[] NOT NULL,
  edge_clipped      boolean NOT NULL DEFAULT false,
  published         boolean NOT NULL,          -- false = glimpse (< 3 detections) or prop
  sentence          text NOT NULL,
  crop_ids          text[] NOT NULL,
  strip_ref         text,                      -- frame strip, 7 d
  rules_version     integer NOT NULL,
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS vision_motion_story_stream_time_idx ON vision_motion_story (stream_id, ended_at DESC);
```

- No bus channel and no schema-registry entry. This follows the existing walkway reducers (individuals, rhythm, object permanence): a DB loop in orion-sql-writer that the reader reads.
- `WALKWAY_REDUCER_EVENT_TYPES` is unchanged, because stories do not go into `vision_events`.
- New env keys for orion-sql-writer: `VISION_MOTION_STORIES_ENABLED=true`, `VISION_MOTION_STORY_INTERVAL_SEC=15`, `VISION_MOTION_STORY_CLOSE_GAP_SEC=11`, `VISION_MOTION_STORY_MIN_DETECTIONS=3`.
- New env key for consumers: `STREET_MOTION_LINE_ENABLED=true`.
- New reader output: `summarize_street(..., motion=rows)` adds the line `"Moving on the walkway in the last 15 minutes: <sentence>; <sentence>."` after the "who" line.

## Files likely to touch

- `services/orion-sql-writer/app/vision_motion.py` (new). Pure functions: link, kinematics, zone path, together, naming, sentence. Port of the prototype.
- `services/orion-sql-writer/app/vision_motion_loop.py` (new). Clocked loop with a cursor on `vision_crop_observation (stream_id, observed_at)`. It closes passages 11 s after their last box.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the new keys.
- `services/orion-sql-db/manual_migration_vision_motion_story_v1.sql`: the new table.
- `services/orion-sql-writer/tests/test_vision_motion.py`: synthetic tracks covering walker, runner with 2 vs 3 detections, edge-clipped box, two named ids, zone-edge jitter, a still prop, and a together pair.
- `services/orion-sql-writer/evals/run_vision_motion_story_replay_eval.py`: the read-only replay (see Acceptance checks).
- `orion/situational/perception_reader.py`: `_STREET_MOTION_SQL` and the motion line in `summarize_street`.
- `orion/situational/street_material.py`: the motion line for reveries, dreams and curiosity.
- Tests next to each of those.
- `services/orion-hub/`: "Motion stories" card with the frame strip and right/wrong buttons (slice 2).
- `services/orion-vision-host/` or the retina: frame-strip capture (slice 2). Frames sit at `/mnt/telemetry/vision/frames` for only 300 s, and orion-sql-writer does not mount that directory today.

## Non-goals

- No AI model, captions or embeddings in slice 1.
- No new identity matching, and no change to `apply_batch`.
- No change to retention for crops (7 d) or sightings (90 d).
- No feed into `node:substrate.perception_walkway`. That node is a whole-frame embedding prediction error, and mixing motion counts into it would break its meaning (metric gate, independence check). A later "unusual motion" input belongs in `vision_attention_score.py`, as a rare (kind, speed class, zone path, hour) against 14 days of stories. It must pass the metric gate with real story history first. Story history does not exist yet.
- No speed claims for runners, cyclists or cars until cadence question 1 is answered.
- No "nobody passed" statements.
- No "child" until question 3 is answered.

## Acceptance checks

1. **Replay eval (gate for slice 1).**
   - `run_vision_motion_story_replay_eval.py --since 2026-10-16T18:59:00Z` runs read-only over live crops.
   - The default window starts at the stream-fix cut. Pre-cut rows are rejected unless `--include-corrupt` is set. With that flag, they count toward speed and cadence tables only, never toward accuracy.
   - It reports: passages, published, glimpses, props dropped, together pairs, the speed-class histogram, and the detected-frame interval distribution.
   - It fails if any published story contains a name under the loose rule but not under the strict one.
   - It fails if any published "running", "riding" or "driving" story has fewer than 3 detections.
   - It fails if any published story has an edge-clipped zone claim.
2. **Unit gate.** `pytest services/orion-sql-writer/tests/test_vision_motion.py` runs in under 2 s with synthetic tracks: a walker gives "walked left to right", jitter gives no "ran", and a prop gives no story.
3. **Live trace after deploy.**
   - A `[MOTION_STORY]` log line with a moving cursor.
   - `SELECT count(*) FROM vision_motion_story WHERE published` grows during daylight.
   - `fetch_street_summary('walkway')` returns the motion line within 15 minutes of a walker.
4. **Accuracy (Juniper spot-check).**
   - The Hub card shows each published story with its frame strip: first, middle and last frame, boxes drawn, 640 px wide.
   - Juniper marks five fields right or wrong: kind, direction, speed class, zone path, together.
   - Target: at least 80% of 50 clean-frame stories fully right, and **zero wrong names**.
   - Until this passes, the PR report states the accuracy as UNVERIFIED.
5. **Prop check.** Over one clean night (22:00-12:00 UTC), at most 2 published standing stories per hour, while the patio skeleton is present.

## Recommended next patch

Slice 1, in one PR:
- The pure `vision_motion.py`, ported from the prototype with the fixes the samples showed: the strict naming rule, edge-clipped boxes, the 0.10 depth threshold, a first gate scaled by the gap, and the still-box gate.
- The loop and the table.
- The street-line consumer.
- The replay eval and the unit tests.

The frame strip and the Hub card are slice 2. Juniper's spot-check is the gate for calling the stories accurate. The cadence change (question 1) is its own design.

Prototype (not committed): `/tmp/claude-1000/-mnt-scripts-Orion-Sapienform/03df7418-1971-45d7-9d94-aa24101c7e9e/scratchpad/motion/motion_stories.py`, `geotrack.py`, export `crops.psv`, output `stories_geo.json`.
