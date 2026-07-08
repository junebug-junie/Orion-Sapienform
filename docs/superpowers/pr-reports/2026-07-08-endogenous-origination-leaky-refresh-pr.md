# PR: Refresh endogenous-origination spec for the leaky drive engine

**Status:** Docs-only. Re-derives the Step-1 (endogenous drive origination) spec's
activation model against the merged leaky `DriveEngine`, so the eventual Step-1
build is correct-by-construction. No runtime code. Step 1 remains hard-gated on
measurement 0(a) (see Handoff below).

## Why

The endogenous-origination spec was written against the old `soft_saturate`
pressure math (`soft_saturate(0.5)=0.593`). That path was replaced by the leaky
integrator (merged PR #879), which rests at 0, is cadence-invariant, and has no
fixed point. Every accumulation figure in the spec was therefore stale, and a
builder following it would have hard-coded wrong thresholds.

## What changed

`docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`:

- **Ground-truth math** now describes the leaky update, flags `soft_saturate` as
  legacy-only.
- New **"Origination dynamics under leaky math"** section with a table + closed
  form, all verified by replaying the real merged `DriveEngine`:
  - single firing @ cap 0.5 → `p=0.500` (not 0.593); cannot activate alone.
  - `p₂ = m + m(1−m)·e^(−Δt/τ)`, plateau `p* = m / (1 − e^(−Δt/τ)(1−m))`.
  - two-firing activation window `Δt ≤ τ·ln(m(1−m)/(activate−m)) ≈ 1321s ≈ 22 min`.
- **New load-bearing constraint surfaced:** `ORIGINATION_COOLDOWN_SEC` is now
  coupled to the activate threshold through τ — a value ≥ 1800s makes endogenous
  drives *silently un-activatable*. The 900s seed is inside the 22-min window by
  design, not arbitrary.
- Decay tail (0.65 → 0.42 = 13.2 min) sits inside the 900s cooldown → the
  self-referential-loop mitigation is now a checkable invariant.
- Saturation-masking failure mode noted as largely resolved by the leaky migration
  (the flat-0.731 pin is gone).
- Updated test #5 and two failure-mode notes.

## Verification of the numbers

Ran the merged `DriveEngine` (leaky, τ=1800, activate 0.62 / deactivate 0.42) at
magnitude cap 0.5:

```text
single firing:            p=0.500  active=False
every 900s:  p1 0.500 -> p2 0.652 (active) -> plateau 0.718
every 1321s: p1 0.500 -> p2 0.620 (active exactly) -> plateau 0.658
every 1800s: p1 0.500 -> p2 0.592 -> plateau 0.613 (NEVER activates)
closed-form plateau @900s: 0.7176 ; two-firing window 1321s ; decay tail 791s
```

Algebra and engine replay agree to 4 dp.

---

## Handoff — testing that must happen on the host (I cannot from the dev env)

These are the gates before any Step-1 (endogenous origination) code is written,
plus the still-open Task-10 verify for the merged homeostatic consumer. Postgres
(`:5432`) and the drive-audit Fuseki graph are not reachable from the dev
environment; all of the below need a DB-reachable host.

### 1. Run the measurement gate (unblocks Step 1)

The Step-1 spec says: *"Do not write code until 0(a) passes."* 0(a) = does
`SelfStateV1` drift during exogenous silence.

```bash
# on a Postgres-reachable host, from repo root:
python scripts/analysis/measure_autonomy_gate.py
# (set the DSN / Fuseki envs the script expects if not defaulted:
#  DEFAULT_POSTGRES_URI = postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
#  Fuseki query URL     = http://orion-athena-fuseki:3030/orion/query )
```

Report back:
- **Verdict (a) — drift:** GO/NO-GO. If NO-GO, the origination signal is inert and
  Step 1 must instead source dynamics from unresolved-pressure persistence (the
  spec's documented fallback) — that changes the design before any code.
- **Verdict (b) — co-activation / resource_pressure:** how often ≥2 drives
  co-activate and whether `resource_pressure` rises (this gates Step 4, internal
  economy — don't build it if scarcity never binds).

### 2. Task 10 — live-verify the homeostatic consumer (merged #882)

After deploying current main and restarting concept-induction:

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

Then tap live for ~60s and confirm:
- `drive:audit` / drive_state shows **differentiated, non-pinned** pressures that
  move with biometrics/mesh-health/failures and decay toward ~0 in quiet windows
  (NOT all six pinned at 0.7309).
- The `orion:signals:vision` / scene_state flood mints **0** homeostatic tensions
  (that channel is never subscribed).
- `dominant_drive` reflects real events (not constant "autonomy"/None).

### 3. Check the drive-audit Fuseki persistence

From the dev env, the reachable Fuseki `/orion` dataset has **1,891 triples, only
`Collapse` types — no `DriveAudit`, no `autonomy/drives` graph**. Either the
rdf-writer targets a different dataset, or drive-audit persistence has been broken
since ~June 19. On the host:

```bash
# does the drives graph have recent audits?
curl -s -X POST http://orion-athena-fuseki:3030/orion/query \
  -H "Content-Type: application/sparql-query" -H "Accept: text/csv" \
  --data-binary 'PREFIX orion: <http://conjourney.net/ns/orion#>
    SELECT (COUNT(?a) AS ?n) (MAX(?ts) AS ?latest)
    WHERE { GRAPH <http://conjourney.net/graph/autonomy/drives> { ?a a orion:DriveAudit ; orion:timestamp ?ts } }'
```
If `n=0` or `latest` is ~June 19, drive-audit persistence needs a fix before the
gate's Fuseki-side (co-activation) verdict is meaningful.

---

## Restart required

None (docs-only PR).

## Risks / concerns

- Low — documentation only. The one substantive claim (cooldown/threshold
  coupling) is engine-verified. Step 1 stays gated; no cognition behavior changes.

## PR link

<to be filled by `gh pr create` / GitHub UI>
