# Parking lot

Real findings that are **not** the current phase. One line, a date, a pointer.

Rule (`2026-08-13-scarcity-and-repertoire-execution-plan.md` §7.2): anything discovered mid-phase
that is not the phase lands here, **not in the branch**. This file exists because every entry
below is genuinely interesting and every one of them would have been a squirrel.

Nothing here is scheduled. Nothing here is abandoned. Adding a line costs one line.

---

## 2026-08-13

- **Arena urgency degeneracy.** `proposal_urgency()` (`orion/proposals/scoring.py:268`) falls back
  to `max()` over all four `PRESSURE_DIMENSIONS` for any template declaring `dimensions: {}`. Five
  of thirteen do. Result: 7.03 of 10 candidates tied at the frame max, `corr(urgency) = 1.000000`
  for 10 of 28 template pairs. Rank order is decided entirely by hand-authored `base_priority`.
  Inverted, too — declaring the dimension you care about is a strict handicap. → Plan Phase 5.

- **The feedback loop is open.** 302,974 `substrate_feedback_frames` rows. Nothing in
  `orion/proposals/` or `orion-proposal-runtime` reads any of them. `orion/reverie/efficacy.py`
  (the "was that action useful" module) has **zero live callers**. → Plan Phase 4.

- **Feedback `score` carries no information.** Exactly one distinct value per `outcome_kind`
  (0.25 / 0.4 / 0.55 / 0.85 / 0.1). It is a lookup table on dispatch status, not a measurement.

- **Outcome deltas are tick-attributed, not action-attributed.** `orion/feedback/builder.py:272`
  keys `improved`/`worsened`/`unchanged` on `field_after.tick_id`. With 5 concurrent dispatches
  per tick this is unattributable to any action. `builder_prune`'s own comment already says so.

- **`proposed_effect` is a category label, not a prediction.** Three coarse values across 13
  templates (`increase_observability` ×8, `preserve_stability` ×4, `prepare_for_policy_gate` ×1).
  Used correctly for policy gating; not usable as a falsifiable claim.

- **46.6% of policy frames carry zero decisions.** Investigated: this is the `action_warrant` tick
  gate working correctly (clean split at 0.5), i.e. program outcome O1. **Not a defect.** Recorded
  so nobody "fixes" it again — I nearly did.

- **`search_web` is synthetic.** `Simulate a web search via LLM using known memory + synthetic
  results`. Not world contact. Anything built on it launders invention as perception.

- **Orion cannot reach the repo from the exec lane.** `orion-athena-cortex-exec-background` mounts
  the repo read-only, git fails with `detected dubious ownership`, and the 75 sibling worktrees at
  `/mnt/scripts/Orion-Sapienform-*` are outside the mount entirely. Any repo-touching skill needs
  a broader mount + rw + a git config change — three privilege expansions. Not taken.

- **`image_prune` is built, tested, and unrouted.** The only existing path to Orion's first
  `acted=true` in 88,409 dispatches. One-line config change whenever wanted. → Plan §8.

- **Pre-existing, unrelated:** `check_service_env_compose_parity orion-execution-dispatch-runtime`
  reports 3 of 24 `.env_example` keys missing from `docker-compose.yml`
  (`EXECUTION_DISPATCH_STALENESS_MIN_SEC`, `_MAX_SEC`, `EXECUTION_DISPATCH_RUNTIME_PORT`).
  Identical on `main`.

- **Pre-existing, unrelated:** `services/orion-execution-dispatch-runtime/tests/test_heartbeat_chassis.py`
  fails collection with `FileNotFoundError` on `main` too.

- **Carried from the prior spec's §2.3, still open:** `action_warrant.py:280` `pinned` guard
  requires `zscore == 0.0` exactly (channel median |z| is 0.0007); `node:prometheus` reads exactly
  0.0 on all 74,264 ticks (dead producer, still consumed); `node:circe` bottoms at 3e-323 (decay
  artifact); `template_match_score` provably dead for all 13 templates; `prune_build_cache` has no
  cooldown; "7 dimensions" prose stale in `orion/field/pressure.py:14,249,259-267`.

## 2026-08-13 (Phase E0)

- **`counterfactual` and `context_exec_memory_contradiction_review` are dead.** Both return an
  empty string in ~0.5 s while reporting `status=success`, on all 10 runs. Never executed before
  E0, so nobody knew. Empty-shell-cognition reporting success. NOT fixed — E0 hit a kill gate and
  stopped.

- **Recall dominates supplied context.** Ten `goal_formulate` runs across ten different field
  ticks, each given Orion's real live pressure readings as the explicit `intention`, all returned
  paraphrases of the same recalled Juniper coding session. Possibly the largest blocker in the
  system for any cognitive-verb work: routed into the arena, these verbs would narrate session
  history rather than Orion's condition. Deserves its own measurement.

- **`goal_formulate` is a translator, not a generator.** Its prompt reads
  `{{ intention or text or request }}` — it structures a *supplied* intention. It cannot produce
  one from state. Any verb sharing that prompt shape has the same limit.

- **Verb names cannot be trusted as capability descriptions.** Assumed `goal_formulate` formulated
  goals. It does not. Check the prompt template before routing any verb.

## 2026-08-13 (PR #1617 review)

- **`builder_prune`'s `pruned_nothing` branch reports `status="success"`.** Identical defect to
  the one fixed in `image_prune` this round: a prune that ran and reclaimed nothing falls through
  to `_skill_result_output`'s defaults (`ok=True, status="success"`), with the honesty living only
  in `result["decision"]`, which no generic consumer reads. NOT fixed — builder_prune is routed
  and live, so changing its reported status is a behaviour change to a shipped path and deserves
  its own patch. `verb_adapters.py` ~3052.

- **`_resolve_docker_prune_run_mode` has default-open edges.** Reproduced: `{'dry_run': None}`,
  `{'dry_run': ''}` and `{'dry_run': 0}` all resolve to **execute**, while the string
  `{'dry_run': 'false'}` resolves to preview (a truthy string is *safer* than the boolean False).
  It also infers execute from natural language in `description`/`text`. Pre-existing and shared
  with `builder_prune`. No producer currently puts free text into `skill_args` for these verbs
  (`executor.py:2371` gates that to `docker_prune_stopped_containers`), so reachability is
  unproven — but `/mnt/docker` reads 78% right now, above `image_prune`'s 70% gate, so this is
  the first time those edges sit in front of a gate that is actually open. `image_prune` is
  unrouted, which is the only reason this is parked rather than blocking.

- **`mount_path` from `skill_args` flows unguarded into `shutil.disk_usage`** in both prune verbs;
  a nonexistent path raises `FileNotFoundError` out of `execute()`.

## 2026-08-13 (scarcity revision)

- **Circe fan/thermal telemetry is unreachable** — the NIC does not read in. Power is a usable
  proxy but heat and noise are what actually make the office unusable, so the ceiling is being
  designed against a proxy rather than the felt quantity. Separate problem; noted so it is not
  lost.

- **`orion_biometrics` carries real per-GPU `power_draw_watts` for athena/atlas/circe**, 15,713
  circe rows back to 2026-07-24, and nothing in the cognition path reads it. This is the only
  live telemetry found that prices Orion's own cost in a unit Juniper actually pays.

- **Standing rule established:** on this system an idle utilisation reading is NOT evidence of
  available capacity. With a concurrency limit of one, the queue forms at arrival, not in the
  meter. Establish the concurrency limit and residency set before any utilisation number is
  used for anything.

## 2026-08-13 (plant survey)

- **There is no LLM call telemetry table.** 193 tables in `conjourney`, none logging inference.
  Consumer attribution has to be parsed out of `orion-llm-gateway` container logs, which rotate.
  Any allocation mechanism needs a ledger and there is none. → the first real blocker for
  anything budget-shaped. `2026-08-13-the-plant-three-ceilings.md` §6.1.

- **`chat` and `agent` routes point at a host that is off.** Both resolve to
  `http://100.112.254.99:8011` (circe-worker-1); circe does not ping and its biometrics stop at
  08:21 today. 1 chat request in 6 h. Whether the gateway degrades, retries, or hard-fails is
  untraced. Orion's deep-cognition lane is currently unavailable and nothing represents that.

- **`priority_admission.py` is wired to AI Town only.** A live, working, `/slots`-driven
  background admission gate (`reserved_free_slots=2`, fail-open) guards exactly one consumer:
  `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_background`. Orion's own autonomous cognition —
  3,750 of 3,821 gateway requests in 6 h, all from `cortex-exec` — uses foreground `quick`/
  `metacog` and is subject to no gate. → candidate first phase, `the-plant` §5.

- **Zero admission-wait events in 6 h of gateway logs.** Cannot distinguish "gate never had to
  fire" from "gate is inert". Needs a deliberate load test before anything is built on it.

- **circe GPU2 holds 21.3 GB of weights and has never been driven above 80% in 7 days**
  (mean util 0.1%, p95 0, zero samples at util≥80). Pure residency, zero output. Whose model
  and why loaded is unknown.

- **athena's P100 is the perception organ, not a spare card.** `orion-athena-vision-host`
  (uvicorn :6600, 5,050 MiB) + `orion-athena-whisper-tts`. p95 util 92% on the same box that
  runs postgres/redis/FalkorDB/hub — so its ceiling is interference with the substrate, not
  inference contention. Moving perception off the orchestration host is the only relief.

- **Chassis power is the dominant term and is entirely unmeasured.** `orion_biometrics.cpu`
  carries `{util, cores, loadavg}` and no power field; there is no wall measurement anywhere in
  the system. Juniper observes 700-1200 W all-in for two nodes, against measured GPU residency
  of only 107 W (atlas) / 153 W (circe). A single 4.5 s inference is ~0.02% of one machine-hour.
  **A smart plug per node would convert ~99% of the cost model from estimate to measured** and
  is the cheapest instrument available to this whole arc.

- **athena's CPU is the loaded resource, not its GPU.** 80 cores at 44.1% mean, 15-min load
  average peaking at **120** (1.5x oversubscribed), on the box that also runs postgres, redis,
  FalkorDB and the hub. atlas (96c) and circe (72c) sit at 0.52% and 0.21% -- pure GPU boxes
  with idle CPUs. The interference ceiling is a CPU story that the GPU tables completely miss.

- **`strain` dilutes the binding constraint by 7x, in production.**
  `orion/telemetry/biometrics_pipeline.py:180` averages 7 pressure channels flat, so one fully
  saturated channel can never push strain above 0.143. Live right now: atlas reads power 0.798
  and memory 0.812 and reports **strain 0.232**. `homeostasis = 1 - strain` is therefore
  anti-informative under concentrated load. Channels are not substitutes -- idle disk does not
  relieve memory pressure -- so the mean is the wrong aggregate; max or count-above-threshold is
  right. **Live behaviour change to a widely-read field signal: enumerate consumers first.**
  → `the-plant` §5.2 / §7 I1.

- **`fan_pressure` and `disk_capacity_pressure` are computed and excluded from `strain`.**
  athena's disk_capacity is **0.748** -- the highest single pressure on that node -- and feeds
  nothing. Worth checking which mount (root reads 22%; candidates are docker/postgres/graphdb/
  telemetry) given the 2026-07-23 Postgres disk death.

- **iLO is live on atlas and athena; raw chassis watts are read and then discarded.**
  `fan_pressure` comes only from iLO and is non-zero on both (0.610, 0.470), so `power_pressure`
  on those hosts is real `ilo_power_watts` -- passed through `EwmaBand` into a unitless 0-1 and
  never persisted raw. Watts are the only quantity here that sums across hosts; band fractions
  do not. **Storing the number already in memory is the highest-value, lowest-cost fix in the
  whole arc.** Circe reads fan 0.000 -> iLO not configured there (the NIC problem).

- **`disk_bw_mbps=200.0` and `net_bw_mbps=125.0` are global constants for 3 heterogeneous
  hosts.** 200 MB/s is spinning-disk-era; against NVMe it understates disk pressure ~10x.
  125 MB/s is 1 GbE. Both need to be per-node.

- **The atlas ceiling is self-inflicted by batch dispatch.** At measured offered load
  a = 0.453 erlangs with c = 4 slots, Erlang-B predicts 0.111% blocking; observed all-4-busy is
  **7.4%** -- **66x Poisson**. Cause is visible in the arrival process: over 3 h / 2,287
  requests, the burst-size mode above 3 is exactly **5** (123 bursts, more than sizes 3+4+6+7
  combined) -- the arena's ~5-proposals-per-tick batch. **A batch of 5 into a 4-slot lane blocks
  by construction at any average load.** So smoothing dispatch beats adding slots, and Orion is
  its own principal competitor (98% of gateway traffic is cortex-exec). → `the-plant` §6.7.

- **Commensurability rules established** (`the-plant` §6, the thing this branch is named for):
  for a ceiling report P(saturated) not the mean; compare lanes by blocking probability not
  utilisation (at c=1 they coincide, at c=4 they diverge 100x -- this is the "8% vs 1/1 slots"
  question); never average non-substitutable channels; keep raw physical units alongside any
  normalised band. Plus four cross-domain relations that hold on real data, incl.
  `load15/threads - cpu_util` as a free I/O-blocking estimator (athena: ~3.4 threads in D-state)
  and fan-as-leading-thermal-indicator (atlas thermal 0.000 while fan 0.610 -- temperature is
  flat *because* the BMC is spending fan speed, and fan is the closest signal in the system to
  the heat/noise Juniper actually pays).

- **APC units not yet wired in (Juniper, in progress).** Real per-node wall draw. This is the
  instrument that converts the dominant cost term from unmeasured to measured. Every chassis
  power figure this arc produced was an estimate from core count and every one was wrong;
  circe is a Gigabyte HA01 with 3x 2200 W PSUs, not the 200-280 W box two drafts assumed.
  → `the-plant` §5 TODO 1. Blocks any real cost model.

- **RAPL is present on athena and root-blocked.** `/sys/class/powercap/intel-rapl:{0,1}/energy_uj`
  exists (2x Xeon Gold 6138, 125 W package limit each = 250 W) but is mode 400. A permission
  change or a root-run collector gives real CPU package power today at zero hardware cost.
  Partial (athena only, CPU only) but immediate. → `the-plant` §5 TODO 2.

- **Mean GPU utilisation is the wrong statistic for a contention ceiling.** Juniper flagged
  atlas as under-reported at 6.8%/22.1% mean util given it runs inference ~24/7. Correct: a
  live 1 Hz `/slots` poll shows `quick` (:8013) is **bimodal** -- 101 of 121 samples completely
  idle, 9 samples completely full, nothing in between. It hits all-4-busy **7.4%** of the time
  while averaging 11.2% of capacity. `nvidia-smi utilization.gpu` additionally samples ~1 s out
  of every 31 s (3% of the timeline) and under-reports bandwidth-bound LLM decode. `/slots` is
  the right meter and the admission gate already reads it. 2-min window -- needs 24 h.

- **No disk I/O telemetry anywhere**, on an 80-thread box running 81 containers at load 42
  (7-day load15 max 120.2, i.e. 1.5x oversubscribed). Only capacity is visible (22% of 197 G).
  Plausible second interference channel on athena, currently invisible.

- **Athena thermals are readable and uncollected:** `x86_pkg_temp` 56 C / 65 C (two sockets),
  `pch_lewisburg` 45 C, via `/sys/class/thermal`. Nothing writes these to biometrics.

- **Method correction, recorded because it nearly shipped:** `avg((util>0)::int)` is not a duty
  cycle. It reported athena's P100 at "99.74% busy" when its mean utilisation is 12.8% — the
  card is near-continuously *active at low intensity*, which the >0 test cannot distinguish
  from saturation. Same session also dropped 60% of gateway requests by matching
  `route=([a-z_]+)` against lines carrying `route=None`. Both caught before the spec was
  written; both are the same family as the four sampling errors already logged today.
