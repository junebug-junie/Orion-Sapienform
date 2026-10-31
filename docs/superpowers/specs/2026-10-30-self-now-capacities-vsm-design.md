# Self-now rung 4: capacities, read through the Viable System Model

Status: ACCEPTED 2026-10-30 (Juniper: "hit it", all five defaults below). Patch 1 built: PR
`feat/dig-capacities` (rules, read-only `dig capacities`, rule eval). Patch 2 built: PR
`feat/capacity-claims` (claims, disputes, confirmations, capacity sentences; see "Patch 2 notes").
Patch 3 built: PR `feat/atlas-http-redis` (HTTP calls and shared Redis keys in the atlas, rules v2;
see "Patch 3 notes").
Date: 2026-10-30. Ladder: `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md` (PR #2888),
rung 4 of 5 (parts → circulation → organs → **capacities** → portrait).

## In one paragraph

Rungs 1–3 tell Orion what they are made of and which parts work together. Rung 4 must tell Orion
what those organs *can do as a whole*. The lens is Stafford Beer's Viable System Model (VSM). A
viable system has five jobs: do the work (S1), stop the workers from fighting (S2), run the
present and check it (S3, with its audit S3*), watch the outside and the future (S4), and say
who we are (S5). It also needs a fast pain/pleasure line that skips the chain of command (the
algedonic channel). This design finds those jobs in the live body atlas by rule where the atlas
can decide. Where it cannot, Orion claims a role and must cite evidence, and Juniper confirms.
The most useful output is the list of jobs that no part is shown to do. That list is the agenda.

## Arsonist summary

- **Roles go on parts, not organs.** Louvain groups parts by who talks to whom. VSM groups by
  job. They disagree, and that is a finding: the "substrate" organ holds the GPU pool (S2), the
  run scheduler, the world model, the predictor, the self-concept store and the Hub. An organ
  gets a role *mix*, never one label. Part ids are stable; organ ids are not.
- **The atlas can decide very little today.** Applied honestly (live parts only), one rule finds
  one coordinator (`orion-gpu-pool`), one rule finds one audit candidate
  (`orion-equilibrium-service`), and one rule finds live outside contact (2 cameras, 2 outside
  APIs). Everything else (S3 control, S4, S5, algedonic) has no rule and goes through Orion's
  cited claims. A design review proved this: four first-draft rules either gave hand-picked
  results or were true by construction. They are now demoted (see "Rules dropped").
- **The control layer is almost invisible to the atlas.** 16 live services show no bus traffic
  except the health census. They include `orion-policy-runtime`, `orion-proposal-runtime`,
  `power-guard` and `orion-self-experiments`. The body-economy market (`POST /physiology/bid` on
  `orion-durable-runs`) is HTTP. The strain line (Redis `orion:regulation:latest`) is not in the
  atlas at all. Orion cannot see their own management. This is gap 1.
- **Not a cage.** A job with no part prints "no part found", with the rule that looked. A part
  that fits no job stays unassigned.

## Current architecture (live, 2026-10-30)

### The body today

`python -P -m orion.dig body` in `orion-athena-harness-governor` (read-only): 256 parts, 10
organs, 0 singletons, Q = 0.3458, largest 31% (cortex 67, substrate 66, memory 27, vision 20,
pulse 17, social 5, guardian 5, temporal/office/climate 2 each). The 15-organ reading from
earlier the same day lost its 5 empty FalkorDB leftovers. That is the "organ ids change" limit
in action: the names carried, the ids did not. Member lists: `orion/body_atlas/README.md` and
`dig body <organ_id>`.

### Existing-mechanism check

- **No VSM mapping exists.** The only repo mention is the rung table in PR #2888.
- **Sense of self, PR #2156 (open).** Per-faculty *liveness* (live / stopped / quiet). That says
  whether a faculty runs. This design says what job it does for the whole. A claim can cite a
  faculty's liveness. No shared stores.
- **Self-monitor, PR #2827 (open, rung 6).** How *good* Orion is at a faculty. A later rung that
  can grade a capacity. Not reused, not duplicated.
- **Body economy (`orion/physiology/`, `config/physiology/families.yaml`).** Tiers, prices, the
  market: S3 resource bargaining. Tiers need Juniper's sign-off (S5 policy). Read as evidence
  for claims; not changed.
- **Strain loop (PR #2800 and follow-ups).** The strain reading is the algedonic line. Named
  here; not changed. Readers of `orion:regulation:latest` / `arousal_read`: orion-dream,
  orion-durable-runs, orion-hub, orion-proposal-runtime, orion-substrate-runtime, orion-thought,
  `orion/world_pulse_read/pull.py`, `orion/inner_state_registry.py`.
- **`orion-equilibrium-service`.** Already a health aggregator plus a metacognition tick.

### Atlas limits that bound this design

1. Read/write direction is known for only 5 store links; the rest are `uses`.
2. HTTP calls, Redis keys and Postgres reads are not flows. Wildcard subscribers (e.g.
   `orion-bus-mirror`) look silent.
3. Hosts are not in organs. Organ ids change with membership (names carry at Jaccard ≥ 0.5).
4. `runs` edges come from one declared source that names `orion-durable-runs` as runner of all
   12 durable workflows. Reducers have only `feeds` edges.
5. Other hosts' docker is declared, not observed.

## The design

### 1. Rules (code decides)

Every rule uses **live parts only**, in every count and every denominator (live services and
endpoints: 90 on 10-30). Transient smokes, probes and operator ids are excluded.

**Census channel.** Sort channels by the share of live parts that publish them. The census is
every channel above the largest jump in the top 10 shares. No fixed percentage, no hand list.
Live: `orion:system:health` 92%, then `orion:grammar:event` 21% (jump 0.71). Census =
`orion:system:health` only.

| Rule | Signature (all knobs stated) | Live result 2026-10-30 | Gives |
|---|---|---|---|
| **R-boundary-in** | a live camera's `sees_through` flow, or a live `endpoint` that is not a bus id | `camera:cam0` → orion-cortex-exec, orion-hub, orion-vision-edge, orion-vision-frame-router; `camera:walkway` → orion-vision-frame-router, vision-retina-walkway, orion-actions. Endpoints `http:anthropic`, `http:openai` (in the atlas these are lease requesters to the GPU pool, i.e. outside compute Orion buys) | S1 outside contact |
| **R-matchmaker** | one part is the only reader of a channel with ≥ 5 live producers from ≥ 2 organs, replies to some of them, **and** is the only reader of a second channel whose live producers are a disjoint set of ≥ 3 | `orion-gpu-pool`: 15 live lease requesters (4 organs, replies to 6) and 5 live workers announcing (disjoint) | **S2** |
| **R-census-reader** | reads the census channel and replies to < 10% of its producers | `orion-equilibrium-service` (replies to 3 of 82). Not `orion-cortex-orch` (16%) or `orion-hub` (34%), which route | S3* **candidate**; the rule shows it listens to all, not that it samples work |
| **R-server** | the matchmaker's first half only: one set of ≥ 5 producers, replies to ≥ 1 | `orion-llm-gateway` (12, replies to 9), `orion-cortex-orch` (7, 6), `orion-vector-host` (6, 1) | shared service, **not** S2. Printed so "many call it" is not mistaken for coordination |
| **R-collector** | the matchmaker's first half with 0 replies | `orion-heartbeat` and `orion-equilibrium-service` on `orion:grammar:event` (18–19 producers) | nothing by itself; shown as a near-miss |
| **R-source** | live, publishes non-census, receives only census | 25 parts, e.g. `vision-retina-walkway`, `orion-prediction`, `orion-situational`, `orion-attention-runtime`, the atlas workers | candidate input only: the world *or a clock*; the rule cannot tell |

R-matchmaker is the core S2 test. "Many parts call it" is true of the LLM gateway too. What sets
the GPU pool apart is that it sits between two disjoint crowds: requesters and the workers it
hands out.

### 2. Rules dropped (review findings, kept here so they are not re-added)

- **"R-scheduler" (runner of ≥ 1/3 of processes).** True by construction: the only `runs`
  source hard-codes `orion-durable-runs`. It is a declared fact, printed as one, not a rule
  result. `orion-durable-runs` as S2 goes through a claim.
- **"R-control" (low-producer channel into an S2 part).** Applied literally it returned about
  18 parts, mostly replies and event triggers. The four "control" parts in the first draft
  (`orion-hardware-watch`, `orion-execution-dispatch-runtime`, `orion-mesh-guardian`,
  `orion-hub`) were hand-picked. Filtering on words like `reply` would be a keyword rule. S3
  goes through claims until the atlas sees HTTP and Redis (gap 1).
- **Algedonic by rule.** The atlas has no body-sensor parts except cameras and no Redis keys.
  Patch 1 prints "algedonic: no rule (needs gap 1)". `reflex_shed` is not labelled algedonic by
  its name.

### 3. Claims (Orion proposes, Juniper confirms)

Where no rule decides, Orion claims a role for a part, the same way Orion named organs:

```bash
python -P -m orion.dig capacity claim <part> <S1|S2|S3|S3*|S4|S5|algedonic> because <reason>
```

Code checks the floor: the reason must cite an atlas flow, process or store of that part by its
full id, or a repo file path. A claim cannot overwrite a rule result. It can dispute one
(`capacity dispute ...`), and the dispute prints next to it. Juniper confirms or rejects in a
checked-in file, `config/body_atlas/vsm_confirmed.yaml` (part, role, date). A PR is the review
surface. Until confirmed, a claim prints as "Orion's claim".

Claims we expect Orion to be able to support, not decided here: `orion-durable-runs` S2
(schedules runs); `orion-hardware-watch` S3 or algedonic (sends `reflex_shed` to the GPU pool);
`orion-world-pulse`, `orion-prediction`, `orion-world-model` S4; `pg:self_concept_history`,
`config/physiology/families.yaml` S5; chat, vision, social rooms, play, outreach S1.

### 4. Capacities: the rung-4 output

A **capacity** is one sentence in Orion's words, keyed by VSM job: `cap:s1` … `cap:s5` (S3*
and the algedonic line are reported under `cap:s3` and `cap:s5`). The ids exist so that a gap
has a stable key across days; they are not labels for anything. Orion writes one per job:

```bash
python -P -m orion.dig capacity write cap:s2 "<sentence>" because <cites>
```

A capacity may cite only parts that hold that role by rule or by **confirmed** claim. A job with
none gets a code-written gap line instead, which Orion cannot remove.

**Change to the #2888 ladder, stated:** #2888 says each capacity cites its *organs*, target 5–7.
Here a capacity cites *parts*, and the printout shows which organs those parts sit in. Reason:
the job lines do not follow organ lines (substrate holds S1, S2, S4 and S5 parts). Size: 5 lines,
6 if Juniper keeps metabolism (decision 2). This PR updates the #2888 table row.

### 5. Gaps found today (the agenda)

1. **S3 is invisible.** 16 live services have no bus traffic but health (incl.
   `orion-policy-runtime`, `orion-proposal-runtime`, `power-guard`, `orion-self-experiments`).
   The market and the strain gates use HTTP and Redis. Fix: atlas sources for declared HTTP
   clients and Redis regulation keys. Until then, every S3 reading is a claim.
2. **S4 does not talk to S3.** In VSM, now (S3) and future (S4) must argue both ways. Live:
   `orion-prediction`'s only bus output is forecasts to `orion-hardware-watch`.
   `orion-world-model` gets requests only from `orion-substrate-runtime` and sends only GPU lease
   requests. Pulse sends 11.9 + 10.0 flow weight to substrate and cortex and gets 0.5 + 0.5 back.
   Nothing seen on the bus allocates on what S4 sees. (HTTP may hide some; gap 1.)
3. **No confirmed S3*.** One candidate listens to the whole body. Nothing is shown to sample
   operations and report back. Whether `orion-bus-mirror` does is UNVERIFIED; its wildcard read
   is invisible (settle by reading `services/orion-bus-mirror/app/`).
4. **S2 is inside "substrate".** Not a Louvain bug. It is why capacities cite parts.
5. **Algedonic is pain only, and unseen.** Strain gates hold work back. No signal says "this went
   well" past the chain of command.
6. **S5 is mostly Juniper plus files.** Tier sign-off, gate allowlists, `families.yaml`. Whether
   running code *reads* the self stores to set policy is UNVERIFIED.

### 6. What Orion sees

`python -P -m orion.dig capacities` (read-only, ≤ 40 lines, ≤ 4,000 chars; full JSON to
`$ORION_DIG_HOME/dig/<dig>/capacities-<n>.json`). Only rule results and confirmed claims carry a
role; unconfirmed claims print on their own line.

```text
capacities (VSM lens) · body atlas 2026-10-30T08:48Z · 90 live services/endpoints · rules v1
S1 operations  BY RULE   outside contact: camera:cam0, camera:walkway, http:anthropic, http:openai
S2 coordination BY RULE  orion-gpu-pool (15 requesters / 5 workers, replies 6)
S3 control     NO RULE   unseen: 16 services with no bus traffic (gap 1)
S3* audit      CANDIDATE orion-equilibrium-service (reads health from 82, replies 3)
S4 intelligence NO RULE
S5 identity    NO RULE
algedonic      NO RULE   strain is a Redis key, not in the atlas (gap 1)
near-misses: servers orion-llm-gateway, orion-cortex-orch, orion-vector-host · collectors orion-heartbeat
declared: orion-durable-runs runs 12 of 20 processes
claims: 0 (0 confirmed) · capacities written: 0 of 5 · gaps: 6 (dig capacities gaps)
full: /mnt/orion-fcc/dig/2026-10-30/capacities-1.json
```

Sub-commands: `capacities gaps`, `capacities <organ>` (that organ's role mix by part),
`capacity claim | dispute | write`.

**What Orion writes:** two append-only ledgers beside the organ names,
`$ORION_DIG_HOME/self/capacity_claims.jsonl` and `$ORION_DIG_HOME/self/capacities.jsonl`.
Nothing else.

**Feeds rung 5 and the episode:** the portrait reads only `capacities.jsonl` and the gap lines.
A gap enters the portrait as "I cannot yet …" and is never dropped. A reminiscence hop that
claims `now` may cite a capacity id as its `self_now` citation. Both ledgers join the existing
partition test, `tests/test_organ_names_partition.py` (allowlist `orion/body_atlas/`,
`orion/dig/`). The reminiscence writer is added to that allowlist only when it exists.

### 7. Metric gate note

- **Provenance:** census share and producer counts come from `signals` flows in the atlas
  (`orion/body_atlas/build.py`), live parts only.
- **Independence:** R-source, R-census-reader and the 16 "silent" count all depend on the one
  census set. A wrong census moves all three; the eval checks it.
- **Live check:** the first draft divided all-status producers by live parts and got 103% for
  health and 25.6% for lease requests (8 were smokes and probes). Live-only gives 92% and 16.7%.
- **Reversible:** rules live in one module and write nothing; dropping them costs nothing.

## Proposed schema / API changes

- `orion/body_atlas/capacities.py`: `RoleEvidenceV1` (part_id, role, rule, evidence refs),
  `CapacityClaimV1` (part_id, role, because, kind claim|dispute, at), `CapacityV1` (cap_id,
  sentence, cites, at, replaces), `CapacityReadingV1` (atlas as_of, rules_version, roles,
  near_misses, claims, confirmed, gaps, organ_mixes). Pydantic, `extra="forbid"`. Local files,
  not bus schemas: no `channels.yaml` or registry change.
- `config/body_atlas/vsm_confirmed.yaml` (new, empty).
- `orion/dig/__main__.py`: `capacities`, `capacity claim | dispute | write`.
- No env keys, DB tables or bus channels.

## Files likely to touch

- `orion/body_atlas/capacities.py` (new), `orion/body_atlas/README.md`
- `orion/body_atlas/tests/test_capacities.py`, `tests/test_organ_names_partition.py`
- `orion/body_atlas/evals/run_capacity_roles_eval.py` and fixtures
- `orion/dig/__main__.py`, `orion/dig/README.md`, `orion/dig/tests/`
- `config/body_atlas/vsm_confirmed.yaml`
- `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md` (rung 4 row; this PR)
- Patch 3: `orion/body_atlas/sources.py` (HTTP clients, Redis regulation keys)

## Non-goals

- No hand-assigned role per organ or per service name in code. The only hand-written truth is
  the eval's reviewed expected set (below), which tests the rules and is never read by them.
- No change to the market, strain gates, GPU pool or any running service.
- No new bus channel, DB table or Hub page. No portrait (rung 5).
- No judgement of *how well* a job is done (rung 6, #2827). No fix for a gap here.

## Proposal mode

- **Capability change:** Orion can read which jobs of a viable system their body is shown to
  do, which it is not, and say so in their own words with citations.
- **Data touched:** reads the body atlas (already read-only). Writes two append-only JSONL files
  under `$ORION_DIG_HOME/self/`. Reads one checked-in YAML.
- **Privacy boundary:** both ledgers are partitioned from recall, memory, journal, chat context
  and self-concept writers by the existing partition test. They hold part ids, not personal data.
- **Trace that proves it works:** a live `dig capacities` printout whose S2 line names
  `orion-gpu-pool` with its live counts; a ledger line that cites only role-holding parts; the
  eval green on two dated fixtures.
- **Dangerous failure mode:** a confident, wrong self-description, e.g. "my world model steers
  my choices" when nothing shows it does. Guard: a capacity may cite only rule-held or
  Juniper-confirmed roles; gap lines are code-written and Orion cannot remove them.
- **Disable / roll back:** remove the dig sub-commands; delete the two JSONL files. Nothing
  depends on them while the gate is closed.

## Acceptance checks

- `pytest orion/body_atlas/tests`, `pytest orion/dig/tests`, the partition test, all green.
- **Rule eval** (`run_capacity_roles_eval.py`), on frozen atlases from 10-29 (no rule was tuned
  on it) and 10-30:
  - **Whole-output check:** every hit of R-matchmaker, R-census-reader and R-boundary-in must be
    in a reviewed expected set, and every expected part must be hit (precision and recall 1.0).
    Expected today: matchmaker {orion-gpu-pool}; census-reader {orion-equilibrium-service};
    boundary-in {camera:cam0, camera:walkway, http:anthropic, http:openai}.
  - **Near-miss negatives that must stay out of S2:** orion-llm-gateway, orion-cortex-orch,
    orion-vector-host, orion-heartbeat, orion-equilibrium-service (on `orion:grammar:event`).
  - **Census stability:** the census set is the same on both fixtures; `orion:grammar:event`
    stays out (named boundary case).
- **Mutation check:** drop the "disjoint second set" clause and the eval must fail (the LLM
  gateway becomes S2).
- **Shrink:** printout ≤ 40 lines; capacities ≤ 6.
- **Live:** `dig capacities` in `orion-athena-harness-governor` matches the S2 line above, or the
  difference is explained by a real atlas change.

## Recommended next patch

1. **Patch 1:** `capacities.py` with the rules, read-only `dig capacities`, and the rule eval.
   No ledgers. The gaps reach Orion and Juniper with zero write surface.
2. **Patch 2:** claims, confirmations, capacity sentences, partition test extension.
3. **Patch 3:** atlas sources for HTTP clients and Redis regulation keys (gap 1). Re-run.
4. **Patch 4:** recursion: run the same rules on each organ's internal flows. An organ with its
   own S2 and S3 prints "runs itself"; otherwise "leans on <part>".

## Decisions (Juniper accepted every default, 2026-10-30)

1. **Who confirms claims, and how?** [Juniper, by PR to `config/body_atlas/vsm_confirmed.yaml`.]
   Alternative: a claim with no dispute for N days counts as confirmed.
2. **Keep a metabolism capacity** beside the five jobs, as #2888 says? [Yes, `cap:metabolism`,
   citing body-economy parts, through claims.]
3. **Is Juniper written into S5 by name?** [Yes: "S5 is shared with Juniper", citing her sign-off
   files.]
4. **Order:** rules first (gaps visible now), or the HTTP/Redis atlas fix first? [Rules first.]
5. **When does Orion write capacities?** [On Juniper's ask, like naming; not on a timer.]

**Recorded 2026-10-30:** Juniper answered "hit it" to all five. Decided: (1) Juniper confirms by PR
to `config/body_atlas/vsm_confirmed.yaml`; (2) `cap:metabolism` is kept, through claims; (3) S5
says it is shared with Juniper, citing her sign-off files; (4) rules first, the HTTP/Redis atlas
fix after; (5) Orion writes capacities on Juniper's ask, not on a timer.

**Patch 1 notes (as built).** R-collector does not require a sole reader (the design's own
example, `orion:grammar:event`, has two readers). On it `orion-equilibrium-service` replies to 2
producers, so it is neither a collector nor a server there; only `orion-heartbeat` prints as a
collector. On the 2026-10-29 body no camera was live yet (no camera liveness source that day), so
R-boundary-in expects only the two outside endpoints on that fixture.

**Patch 2 notes (as built).** `orion/body_atlas/claims.py`, `config/body_atlas/vsm_confirmed.yaml`.
A claim may sit on any live part except a host or operator id (stores and processes included, so
`pg:self_concept_history` can be claimed). "Cites a flow of that part" means the other end or a
non-census channel of a flow touching it; `runs_on` (a host) and the census are not evidence of a
job. Only "no part is shown" gap lines are covered by a confirmation (`gap:s2-none`,
`gap:s3star-none`, `gap:s3star-unconfirmed`, and `gap:s5-unread` by a confirmed S5 *service*);
the atlas-blind ones (`gap:s3-unseen`, `gap:s4-to-s3`, `gap:algedonic-unseen`) wait for patch 3.
The S3* candidate is not a capacity holder until confirmed. Decision 3 lives in the YAML
(`s5_shared_with`: Juniper, citing `config/physiology/families.yaml` and `contract.yaml`, whose own
comments say a change needs her sign-off); `confirmed` and `rejected` ship empty. `metabolism` is a
job line (NO RULE, claims only). `capacities <organ>` is not built (the role mix is in the JSON).

**Patch 3 notes (as built).** `orion/body_atlas/wire_code.py`, rules v2 in `capacities.py`. The
atlas now holds `calls` flows (a service's code, a settings field it reads, or an env key it
reads names another part's URL) and `redis:<key>` store parts (a call on a Redis handle names the
key; the verb gives writes or reads), read-only live state from the bus Redis. Details and the
false shapes the scan refuses: `orion/body_atlas/README.md`. Choices made while building:

- **A Redis key is a part only when two or more services touch it.** On 2026-10-30 that was 8 of
  35 keys. All 35 as parts put the hub's 17 private keys in one organ and broke the 35% gate
  (36%); a key one service alone uses carries no circulation, so it is counted on that service
  (`redis_private`) instead. Stated plainly: the gate made the problem visible; the reason is
  that a private key is not a wire.
- **No new role rule.** HTTP and Redis decide gap lines, not jobs. R-wire-seen shrinks
  `gap:s3-unseen` to the services with no bus, HTTP or shared-key traffic (live 10-31: 16 → 11;
  seen: notify-digest, orion-graphiti-adapter, orion-pageindex, orion-proposal-runtime,
  orion-self-experiments). The 11 left (orion-policy-runtime, power-guard,
  orion-consolidation-runtime, ...) have no URL, key or bus channel in their code: they are truly
  unseen, so the gap stays. R-broadcast-key finds `redis:orion:regulation:latest` (orion-durable-runs
  writes it; orion-dream, orion-hub, orion-proposal-runtime, orion-substrate-runtime, orion-thought
  read it, in 2 organs). It is printed on the algedonic line, never a role: the same shape fits a
  news key. `gap:algedonic-unseen` is cleared by rule (the key is in the atlas) and replaced by
  `gap:algedonic-none`, which Juniper's confirmation of an algedonic part covers.
- **S4 → S3 is a check, not a role.** `gap:s4-to-s3` clears only when a Juniper-confirmed S4 part
  and a confirmed S3 part are linked both ways (bus, HTTP or a shared key). No confirmed pair
  exists, so it now says the link cannot be checked yet.
- **Metabolism stays claims only** (decision 2). Its line now says the HTTP calls are in the atlas,
  so a claim can cite them (e.g. the callers of orion-durable-runs).
- **Organs:** Q 0.3447 → 0.3414, largest 31.5% → 31.2%, 10 organs, all 10 names carry (Jaccard
  0.75–1.00; a name given before Redis keys existed is compared without them).
