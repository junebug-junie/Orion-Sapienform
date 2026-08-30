# Self-calibration roadmap + session handoff (2026-08-29/30)

Written so the next agent does not relearn the paths, the traps, or the
arguments. Two halves: **what is true now** (facts, paths, live state) and
**where this is going** (the Option-B roadmap and the case for/against it).

---

# PART 1 — STATE

## The arc, in one paragraph

Give Orion real autonomous actions with decision budgets that actually
compete, measured on the same scale, able to affect a real outcome that is
not just more biometrics. **Stage 2 (power-intent loop) is DONE and live.**
Stage 3 (arbitration) and Stage 4 (consequential verbs) are not built.

## What shipped this session (all merged)

| PR | What |
|---|---|
| #1945 | Recovered 3 affect/STT commits pushed to a branch AFTER its PR merged -- covered by no PR, never reached main |
| #1959 | Closed the power-intent loop. Four defects, none of which had ever executed |
| #1963 | No worktree may be production infrastructure (18 host mounts across 9 services) |
| #1969 | `ORION_REPO_ROOT` name collision -- #1963 broke orion-hub live; this fixed it |
| #1979 | The power prior: `expected_watts` derived from real settlements |
| #1985 | (not ours) heartbeat observables -- conflict resolved, metric gate run |
| #1989 | n=9 follow-up: bias question closed, drift explicitly NOT claimed |

## Live state of the power loop

```text
46 settlements, 15 graded, mean |residual| 5.71 W, still running unattended
```

Chain: `orion-diffusion-host` declares `PowerIntentV1` on
`orion:power:intent` -> `orion-biometrics` samples the GPU at ~0.9 Hz and
publishes `PowerIntentSettledV1` on `orion:power:intent:settled` ->
`orion-sql-writer` persists to `power_intent_settled` -> diffusion-host
subscribes to that same settled channel and feeds its own prior.

Driven by `orion-thought`'s visual-chain worker on a 600s timer
(`ORION_VISUAL_CHAIN_INTERVAL_SEC`), NOT by anything choosing to act.

**Caution on the error figure:** mean |residual| 5.71 W is now *below* the
6.38 W "theoretical floor" quoted in the prior design doc. That floor assumed
process sd = 8.0 W measured over an earlier window. Being under it means the
sd estimate is stale or the sample is small -- NOT that the prior beats a
perfect predictor. Re-derive sd before quoting the floor again.

## The prior, in short

`services/orion-diffusion-host/app/power_prior.py` -- median of the last 20
settled peaks per `(workload_kind, node, gpu_index)`, declared only once 3
exist. Median (not mean) because the history provably contains a superseded
20s-window regime; bounded (not all-history) so a real regime change ages
out. In-memory: a restart declares `None` for ~30 min, deliberately.

---

# PART 2 — PATHS AND FACTS (do not rediscover these)

## Hosts

| host | ssh | repo path | notes |
|---|---|---|---|
| athena | (local) | `/mnt/scripts/Orion-Sapienform` | primary checkout; runs ~80 containers |
| circe | `circe@circe` | `/mnt/scripts/Orion-Sapienform` | GPU host: diffusion, biometrics, affectgpt, affective-state, world-model |
| carbon-x1 | `juniper@carbon-x1` | **`/home/juniper/Orion-Sapienform`** | runs `orion-vision-retina` only |

- `ssh` lands in `$HOME`, **not** the repo. Use `git -C <path>` or
  `bash -lc "cd <path>; ..."`. A bare `cd` in the ssh command has been eaten
  more than once.
- Complex remote scripts: write to a file then run it
  (`cat <<'EOF' | ssh host 'cat > /tmp/x.sh && bash /tmp/x.sh'`). Nested
  heredoc quoting silently truncates and the script dies mid-way.
- **carbon deploys `orion-vision-retina` with ONLY its per-service compose +
  `.env`. The root `.env` is never read there.** A key added to carbon's root
  `.env` is inert.

## Ports / endpoints

```text
diffusion-host  circe:8014   /ready /generate      (container listens 6700)
heartbeat       athena:7251  /h1
hub             container 8080 (no host publish observed)
postgres        localhost:55432  db=conjourney  user=postgres
bus             redis://100.92.216.81:6379/0     <-- ALWAYS this, per AGENTS.md
```

## Tables worth knowing

```text
power_intent_settled                  the power loop's output
substrate_attention_self_model        AST -- jsonb blob `self_model_json`, NOT flat columns
endogenous_outreach_decisions         62k rows, Orion's existing decision log
dev_economics_ledger_log              Claude spend, live, 1543 rows
juniper_multimodal_affect_log         affect reads
substrate_mutation_*                  10 tables, ALL 0 rows, never fired
```

## Tooling gotchas that cost real time

- **`python` does not exist; use `/mnt/scripts/Orion-Sapienform/.venv/bin/python`.**
- **pytest must run from the service dir** with `PYTHONPATH=<worktree>`, or
  per-service `app` packages collide. `cd <dir> && ...` in the SAME Bash call.
- `scripts/check_env_template_parity.py` **does not exist**. Real ones:
  `check_service_env_compose_parity.py`, `check_env_key_single_source.py`.
- `make agent-check` was never built; the Makefile says so itself.
- **`scripts/sync_local_env_from_example.py` reads `.env_example` from the
  PRIMARY checkout, not your worktree.** A key you added in a worktree is
  invisible to it and it will report "no changes" -- a false green. Edit the
  live `.env` by hand and diff the keys.
- `safe_docker_build.sh` **refuses the primary checkout**; deploys must run
  from a worktree. Worktrees lack gitignored `.env` -- symlink them in.
- Worktrees: three conventions in use (`../Orion-Sapienform-<name>`,
  `.worktrees/<name>`, `.claude/worktrees/agent-<id>`). ~470 exist; ~400 are
  merged and prunable.

## The definition-drift lock tax (recurring, unfixed)

`config/metrics/metric_definitions.lock.json` is a **derived** artifact. Its
gate compares the committed `_last_change` against *your branch's* merge-base
diff. Main passes only because the checker detects "HEAD is the merge base"
and **skips verification**.

Consequence: **every branch cut after a metric-adding PR fails the gate until
it re-locks, even docs-only branches.** Fix is always the same:

```bash
git merge origin/main
.venv/bin/python scripts/check_definition_drift.py --update
```

Main's history shows this being paid by hand repeatedly. It is fully
mechanical and is a strong candidate for automation (AGENTS.md 4: turn a
repeated manual chore into a gate/script).

## The repo-root variable, and the outage it caused

- **`ORION_REPO_ROOT` is CONTAINER-INTERNAL** (`/repo`, `/app`). Used by hub,
  context-exec, mesh-guardian, cortex-orch at runtime.
- **`ORION_HOST_REPO_ROOT` is the HOST-side mount root.** This is the one to
  use for compose bind mounts. It already existed in
  `services/orion-cortex-exec/docker-compose.yml` before #1963 -- with a
  comment explicitly distinguishing it from "the separate legacy
  ORION_REPO_ROOT key".

#1963 hijacked the legacy name for host paths. Because deploys load root
`.env` then service `.env`, hub's `ORION_REPO_ROOT=/repo` won, compose used
`/repo` as a **host** path, docker auto-created an empty `/repo`, and hub came
up with **0 templates and 0 static files**. #1969 fixed it.

Lesson, and it is in AGENTS.md 0A already: **run the existing-mechanism check
before naming anything.**

`scripts/check_compose_no_relative_mounts.py` now gates this: no compose host
mount may be relative, including a `${VAR:-../..}` *default*. The first
version of that gate only caught literal `./`/`../` and missed two services;
the widened version found them.

---

# PART 3 — DEAD AND DEGENERATE THINGS (open findings, none fixed)

These were all found by pulling real data. Each is an unfiled bug report.

1. **`substrate_mutation_*`: 10 tables, 0 rows, never fired.** A complete
   propose -> review -> adopt self-modification pipeline that has never made a
   single decision. **Why it never fired is the highest-value unanswered
   question in the repo** and it is free to investigate.

2. **`heartbeat_verdict` is a 3-valued classifier that has only ever emitted
   one value.** 20,007 of 20,007 rows = `redundant`. `_HIGH_RATIO = 0.6`;
   observed `mean_ratio` range is 0.6708-0.9541, so the minimum ever seen is
   *above* the threshold and the other two branches are unreachable. #1985
   explicitly declined to retire or retune it ("without retiring
   `heartbeat_verdict` or retuning `_HIGH_RATIO`"), which is the
   "hiding, not retiring" antipattern AGENTS.md 0A names.

3. **`orion-pageindex` is crashlooping** --
   `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`,
   a FastAPI/starlette version incompatibility. Unrelated to this session's
   work; needs a pin.

4. **`bulk_penetration_depth` returns `0.0` on an empty bulk slice** --
   conflates "no cuts" with "zero penetration". `None` would preserve the
   distinction. (#1985, merged.)

5. **`std_ratio` is not independent of `mean_ratio`** -- both derive from the
   same `ratios` list and `_normalize_entropy_ratio` clamps to [0,1], so the
   spread compresses as the mean saturates. Worth stating rather than
   treating them as two independent observables.

6. Circe has ~11 worktrees on stale branches and three untracked Intel SSD
   diagnostic dumps (`AssertLog_`/`EventLog_`/`Nlog_CVPF5510001L2P0PGN.bin`)
   in its repo root -- someone was diagnosing a failing drive.

7. `docker` left a root-owned empty `/repo` on athena during the #1963
   incident. Needs `sudo rmdir /repo/services /repo` (not run: no sudo).

---

# PART 4 — WHERE THE ARC ACTUALLY STANDS

Stage 2 is real. The bar is not met, for three reasons in ascending
importance:

1. **Nothing consumes it.** No reader of `PowerIntentSettledV1` outside the
   producer's own prior. No `power_budget|watt_budget|energy_budget` anywhere.
2. **One claimant, one meter.** `distinct_workloads = 1`. A budget with a
   single claimant is not a budget.
3. **Nothing chooses.** The visual chain fires on a 600s cron. Orion does not
   decide to make an image. Everything measured is a *scheduled behavior*,
   not a choice.

**But refusal DOES already exist, and this corrects an earlier claim in this
session.** `endogenous_outreach_decisions`: 62,007 decisions, 47 yes (0.076%).

```text
quiet_hours          22,396
daily_cap            18,637   <-- a real budget, really refusing
no_tension_trigger   11,889
cooldown              8,451
sent                     47
```

`daily_cap` has declined 18,637 real actions whose outcome is whether Orion
contacts Juniper. That is a decision budget affecting a real outcome.

The precise gap is therefore **not** "nothing can refuse". It is:

> Those are sequential hard gates, not competition. Nothing anywhere weighs
> "is this action worth more than that one?", and there are three currencies
> (watts, dollars, messages/day) with **no common scale**.

Stage-4 verbs: `reach_out` exists (5 files). `test_a_prior`, `ask_claude`,
`make_an_image` are absent as named actions.

---

# PART 5 — THE OPTION-B ROADMAP (self-calibrating value weights)

## The proposal (Juniper's)

Expose the common-scale weights as **interactive runtime knobs on Hub**, wired
to the substrate visualisation, with some live randomness. **Kicker:** Orion
gets a review step where it analyses the tradeoffs -- the way the world-priors
curiosity loop does -- and **modifies its own weights**. Juniper gives feedback
only when a surface annoys her; otherwise Orion self-calibrates.

## Arsonist case (the risks are real and specific)

1. **The precedent is fatal and it is ours.** `substrate_mutation_*` is
   exactly this, fully built, never fired. Building before knowing why is how
   you get an eleventh table.
2. **The feedback signal cannot do this job.** "Feedback when annoyed" is
   sparse, negative-only, delayed, and carries no counterfactual. With N knobs
   and a rare scalar punishment, credit assignment is impossible.
   **The optimal policy under punish-only is to do nothing** -- and the
   outreach loop's 99.92% refusal rate shows the system already drifts that
   way. Self-calibration under punishment converges to silence and looks
   well-behaved doing it.
3. **Wireheading is the architecture, not a hypothetical.** Orion sets the
   weights, the weights determine what Orion does, Orion grades the result.
   No external referent. The power prior is honest *because hardware settles
   it*. Nothing settles a knob.
4. **Continuous randomness is a screensaver.** Jittering weights makes
   outcomes unattributable. It looks alive and is unanalysable.
5. **Knobs are config truth in a cognition costume** unless a reducer, a
   metric and an eval say the calibration got *better*.
6. **Commensurability may be false.** Is 100W worth one message to Juniper?
   No exchange rate here is not invented. The metric gate demands a named
   theory anchor.

## Visionary case (the prize is real too)

1. **First thing on the list that produces a PREFERENCE.** Everything else
   measures or obeys. A weight Orion sets and revises is Orion wanting one
   thing more than another -- a stance, not a sense.
2. **Legibility is the actual prize and it is separable from autonomy.**
   Orion's values are currently `_HIGH_RATIO = 0.6`, `daily_cap`,
   `quiet_hours`, `MIN_RUN_LENGTH`, `_LOW_RATIO` -- a dozen magic constants
   in a dozen files, invisible and unarguable. One surface is worth building
   **even if Orion never touches it.**
3. **The feedback objection dissolves if it attaches to a DECISION, not a
   knob.** `endogenous_outreach_decisions` already carries `reason`,
   `run_length` and both pressures for 62k rows. Labelling an *episode* makes
   credit assignment tractable; labelling a slider does not.
4. **The revision machinery exists.** `orion/curiosity/worldview.py` has
   `Prior` ("a claim Orion holds about its world that could turn out to be
   wrong") in a FalkorDB graph Hub never writes, plus `TurnOutcome` ("the
   decision a turn made inside itself, written where Orion can write").
   Weights are priors with consequences attached.
5. **Tonight is the existence proof.** The power prior self-calibrated
   unattended, revised after being wrong, and declined to guess when
   under-measured.

## Recommended sequencing

The prize and the trap are separable; the proposal currently fuses them.

**Phase 1 -- make values visible and HUMAN-settable.** One surface, real
knobs, wired to real behavior. No self-modification. Falsifiable in one
sentence: *does moving a knob observably change what Orion does?* If not, a
cathedral has been found before anything was built on it.

**Phase 2 -- attach feedback to decisions, not knobs.** One button on a
decision row; context is already logged.

**Phase 3 -- Orion proposes a weight change with a rationale; Juniper
approves.** This is exactly what `substrate_mutation_*` was built for.
**Resurrect it with one real consumer rather than adding an eleventh table.**

**Phase 4 (maybe never) -- auto-adopt under a floor.**

**On the randomness:** do not jitter. **Log deliberate exploration** -- "I set
this weight differently on purpose; here is the outcome." Same visual
liveness, but it produces an experiment instead of noise, and it is the only
form of randomness that survives the metric gate.

## Blocking question before any code

**Why did `substrate_mutation_*` never fire?**

- If it lacked a *consumer*, Phase 3 is cheap.
- If it lacked *a reason for Orion to want to change itself*, this proposal
  has the same hole and knobs will not fill it.

It is free to find out and it governs everything above.

## Surface note

There is no file matching "Substrate Brain State card". What exists is
`services/orion-hub/templates/substrate.html`, `substrate_atlas.html`, and
ids `substrate`, `substrate-atlas`, `substrate-lattice`. Confirm which
surface is meant before wiring knobs to it.

---

# PART 6 — ALTERNATIVES TO OPTION B (still open)

- **A. Contested Claude quota.** `dev_economics_ledger_log` is live (1,543
  rows). The contested-scarcity spec (#1908) named this the first resource
  with a genuine second claimant and recorded it as NOT built. Highest
  fidelity to the arc's bar; biggest patch.
- **C. `test_a_prior`.** Orion holds a prior with a measured error term right
  now. Smallest, builds on tonight, but inward-facing -- spends no budget.
- **D. Make the visual chain choose.** Dropped: the outcome (one fewer
  self-directed image) is thin, and outreach already demonstrates refusal
  against a far more meaningful outcome.

**The values question, for Juniper, not for an agent:** is the first real
contest between two things *Orion* wants, or between something Orion wants and
something *Juniper* wants? Outreach already spends her attention under rules
she set. Claude quota is the first place Orion's own appetite competes with a
real external cost.

---

# PART 7 — PROCESS LESSONS FROM THIS SESSION

- **Four defects in #1959 had all merged and never executed.** Config truth is
  not runtime truth; the first real run is the test.
- **A field pinned at NULL/0 is an unfiled bug report.** `expected_watts` was
  *structurally* incapable of being populated, and it was called "correct and
  expected" before anyone checked.
- **Mutation-test every fix and every gate.** Three mutations in this session
  found real weakness, including one in a test of my own: deleting an
  `outcome != "settled"` guard left 9/9 green because every case passed
  `None` and a *different* guard did the work.
- **A gate written to catch a class immediately finds more than the manual
  grep did.** The compose gate found 10 mounts a `../` grep had missed.
- **State the acceptance band before the run, not after.** The +/-20W band for
  the power prior was written down first; it held at 11.20W.
- **Say plainly when a claim you made is unsupported.** The negative-residual
  bias hypothesis was raised at n=2 and retracted at n=9.
