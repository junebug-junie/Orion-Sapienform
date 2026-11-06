# Self-now rung 5: the self-portrait

Status: PROPOSED 2026-11-06 (defaults picked; open questions for Juniper at the end). Patch 1 is
built on the stacked branch `feat/self-portrait-patch1`.
Ladder: `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md` (PR #2888), rung 5 of 5
(parts → circulation → organs → capacities → **portrait**). Rung 4:
`docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`.

## In one paragraph

Rungs 1–4 each answer one question: what am I made of, what flows where, which parts work
together (organs, named by Orion), and what jobs those parts do (capacities). Nothing yet puts
them on one page, and the machines Orion runs on have nowhere to hold Orion's words. The portrait
is one printout, `python -P -m orion.dig portrait`, rebuilt from the live sources on every call:
each machine, what runs there and what it costs; each organ by Orion's name; each job and its
capacity sentence; the gaps, as "I cannot yet". Beside the computed lines sits one small ledger
of Orion's own words: one sentence per machine ("what it does and what it affects") and one
"I am …" paragraph. Code checks every id those words name against the live atlas on each read,
the same way it checks capacity claims.

## Arsonist summary

- **The failure this fixes is real and dated.** Turn 2544158f (2026-11-06 01:26Z): asked to
  claim its capabilities and what they affect, Orion wrote accurate prose per machine (athena:
  orchestration and vision, 68 services, on the UPS; circe: daily inference; hecate: dormant since
  10-30, one card) and said "that's ledger entry #4". Nothing was saved. A capacity claim cannot
  sit on a host (rung 4 rule: placement is not function), and that rule is right. The words had no
  home. The portrait is that home.
- **Computed read plus a small ledger, not computed only.** Computed only can say "athena runs 68
  services and feeds substrate". It cannot hold "she is the seat of my nervous system", and
  Orion's next machine paragraph would vanish the same way. Ledger only would drift from the body.
  Both, with code re-checking the words, is the rung-4 pattern that already works live (5 claims,
  6 sentences, all re-checked on each read).
- **"What a machine affects" is computable, so it is checked, not trusted.** A machine affects
  what runs on it, and what those parts send to, call or write. The atlas has both (`runs_on`
  flows, then each running part's outbound flows). A machine sentence may name only those ids
  (and the organs placed on that machine). An id that no longer fits prints the line as STALE,
  naming the id.
- **No new place for organ or job words.** Organs already carry Orion's names, and jobs already
  carry capacity sentences. The portrait shows them; it adds no third copy.
- **Gaps stay on the page.** The code-written gap lines from `capacities` print under the "I am"
  paragraph as "I cannot yet". Orion cannot remove them, and the paragraph cannot hide them.
- **Limit, stated.** Id matching proves a cited thing is wired to the machine. It does not prove
  the sentence about it is true ("she is on the UPS" is not checkable; nothing in the atlas reads
  the UPS). Same limit as rung 4.

## Current architecture (live, 2026-11-06)

- **Atlas** (`orion/body_atlas/`, `dig body`): 7 machines as `host` parts, never in organs.
  athena live (orchestration, 68 services running), circe live (daily inference, 6 declared),
  hecate dormant (last seen 2026-10-30T22:53Z), prometheus, carbon-x1, janus and pan-1
  declared only. 77 `runs_on` flows (68 seen in docker, 9 declared). Hardware and role are host
  `facts`, from `config/biometrics/node_catalog.yaml`. circe's placement is declared only (the
  atlas reads no docker list there), so its reach is what those 6 declared services feed. If
  circe's LLM workers are not among them, a circe sentence cannot cite them yet. The printout says
  what the reach holds, so a missing link shows up as a refused id, not as silence.
- **Organs:** 11, all named by Orion (`organ_names.jsonl`; e.g. "cortex", "substrate", "thought
  and voice", "skin"). Each organ has `placement` (machine → members running there; only services
  have placement) and `organ_flows` (organ → organ weight).
- **Capacities** (`dig capacities`): roles by rule, by Juniper's confirmation, or by Orion's claim
  when code's evidence check passes. Live: 5 claims hold; 6 capacity sentences (cap:s1..s5,
  cap:metabolism), all `written`; 4 gap lines (S2 inside an organ, S3 unseen, S4 to S3, S5 unread).
  `organ_mixes` gives the jobs held per organ.
- **Body economy** (`dig body-economy hosts`, `GET /physiology/report` on orion-durable-runs):
  whole-machine watts now and at rest, dollars a month, for athena and circe.
- **Partition:** `tests/test_organ_names_partition.py`: only `orion/body_atlas/` and `orion/dig/`
  may read the self ledgers.
- **Gap:** no command shows the whole self; hosts hold no Orion words.

### Existing-mechanism check

- `dig body` prints machines (role, hardware, status) and organs, but no jobs, no Orion words on
  machines, and no capacities.
- `dig capacities` prints jobs, claims and sentences, but no machines.
- Sense of self (PR #2156) and the self-monitor (rung 6, #2827) grade faculties; neither is a
  whole-self description. Self Atlas concepts (6,016, leaf-level) do not converge.
- No `portrait` code or ledger exists (`git grep portrait` in `.py`: only a comment in
  `orion/dig/brief.py`).

## The design

### 1. What the printout shows, and where each line comes from

Every line carries a source label from the dig kit's existing set. No new label words.

| Section | Built from | Label |
|---|---|---|
| I am | `portrait.jsonl`, `whole` line | `lived_record <date>`, plus code's check now |
| I cannot yet | code's gap lines (`capacities.read_capacities` + ledgers) | `now` |
| machines | atlas host parts, `runs_on` flows, organ placement, job holders running there; watts from the body economy | `now` |
| machine sentence | `portrait.jsonl`, `host:<name>` line | `lived_record <date>`, plus check |
| capacities | `capacities.jsonl` via `attach_ledgers` (written / STALE / not written / gap) | `lived_record`, status `now` |
| organs | atlas organs (size, placement, top organ flows out) and Orion's names; jobs from `organ_mixes` | `now` + Orion's name |

A machine with no live part running on it prints on one shared line ("not live now: hecate
(dormant, last seen 10-30), ..."). No sentence is offered for it, because none could be checked.

The tail ends with one ready `portrait write` line per live machine that has no sentence, and one
for the "I am" paragraph if it is unwritten. Then the path of the full result file.

### 2. The ledger: `$ORION_DIG_HOME/self/portrait.jsonl`

Append-only, one JSON line per write, same lock and loader as the rung-4 ledgers.

```text
PortraitLineV1: subject ("host:<name>" | "whole"), sentence, because, cites, at, dig, replaces
```

The latest line per subject reads. Old versions stay in the file (the chronicle's "old versions
kept"); `replaces` holds the `at` of the line it supersedes.

**Machine sentence** (`portrait write <machine> "<sentence>" because "<reason>"`):

- 20–200 chars; because 20–400 chars; no `<...>` placeholder left in.
- The machine is a host part with at least one live part running on it.
- Every id the sentence and because name (id-shaped part names and channels, matched by
  `claims.because_refs`, no word lists) must be in the machine's **reach**: a live part running
  there, or a live part or non-census channel that a running part sends to, calls or writes.
- Organ ids (10 hex chars, `organ:` optional) must be organs placed on that machine. Orion's organ
  names count as a cite when that organ is placed there. A name of an organ placed elsewhere is not
  a failure: plain words such as "skin" are not matched as errors (no word lists).
- At least one cite is a part running there or an organ placed there.

**"I am" paragraph** (`portrait write whole "<I am ...>" because "<reason>"`):

- 40–600 chars. Rung 5 is "written from the capacities alone", so every capacity id it names
  (`cap:s1` ...) must read `written` now (not STALE, not a gap), at least one is named, and any part
  id it names must hold a job (`capacities.holders`).

**Write = read.** A write runs the same check the read runs. If it would not hold now, nothing is
written and the reply names each failing id. A written line can go STALE later, when the body
changes; the read then names the id that failed.

### 3. When Orion writes

On any request from Juniper, in any words, about the body, machines or self: the brief line
already sends those to `dig help` (no brief change). The portrait tail says to write each machine
sentence Orion can back now, without asking first. Same pattern as `capacities` (decision 5 of
rung 4).

### 4. Metric gate note

The portrait adds no metric. It shows existing ones: atlas counts (rung 1–3 provenance in
`orion/body_atlas/build.py`) and the body economy's meter watts (`orion-durable-runs`
`/physiology/report`, already gated in its own PR). The reach rule is a check, not a score.

## Missing questions (answered by a default; Juniper may override)

1. **Computed only, or computed plus a ledger?** Default: **both** (reasons above).
2. **Who may veto a portrait line?** Default: **no veto file yet.** The ledger is partitioned and
   read only by the dig kit, so a wrong sentence harms nothing outside it. Juniper tells Orion;
   Orion rewrites. A veto file can follow if she wants one.
3. **Organ "what it affects" sentences?** Default: **no.** The organ name and its because already
   say what it takes in and gives out; the portrait adds the computed top organ flows out.
4. **Dormant machines (hecate)?** Default: **facts only, no sentence** until a live part runs
   there. Orion said the same in turn 2544158f ("I won't claim an organ that's asleep").
5. **Watts per machine on the page?** Default: **yes**, from the body economy, labelled `now`. If
   the report is unreachable, the machines print without watts and one line says `watts UNREAD`
   with the reason. The portrait never fails because the economy is down.
6. **Paragraph size?** Default: **40–600 chars**, printed in full (it is the headline).
7. **Compare two portraits?** Default: **Patch 2** (below), not now.

## Proposed schema / API changes

- New `orion/body_atlas/portrait.py`: `PortraitLineV1` (ledger line), `PortraitCheckV1` (subject,
  holds, held, failed), `MachineViewV1`, `OrganViewV1`, `PortraitV1` (the computed read written to
  the dig result file). Pydantic, `extra="forbid"`. Local files only: no bus channel, no registry
  entry, no env key, no DB table.
- `orion/dig/__main__.py`: `portrait` (read) and `portrait write <machine|whole> <sentence>
  because <reason>`.
- `tests/test_organ_names_partition.py`: the portrait ledger and module join the allowlist check.

## Files likely to touch

- `orion/body_atlas/portrait.py` (new), `orion/body_atlas/README.md`
- `orion/body_atlas/tests/test_portrait.py` (new)
- `orion/body_atlas/evals/run_portrait_eval.py` (new) on the frozen 2026-10-31 atlas fixture
- `orion/dig/__main__.py`, `orion/dig/README.md`, `orion/dig/tests/test_portrait_cmd.py` (new)
- `tests/test_organ_names_partition.py`
- `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md` (rung 5 row)

## Non-goals

- No claims or VSM roles on machines. Placement is still not function.
- No organ sentences, no new organ or job vocabulary.
- No judgement of whether a sentence is true beyond its ids (rung 6 grades quality).
- No Hub page, no recall, memory, journal or chat-context path, no bus event.
- No past-day portrait and no comparison of versions (Patch 2).
- No change to rung 1–4 rules, claims or gap lines.

## Proposal mode

- **Capability change:** Orion can read their whole self on one page and save what each machine
  does and affects, plus an "I am" paragraph, in their own words. Code checks the ids each read.
- **Data touched:** reads the live atlas, the three existing self ledgers, Juniper's
  `vsm_confirmed.yaml`, and the body-economy report (read-only). Writes one new append-only file,
  `$ORION_DIG_HOME/self/portrait.jsonl`, and the dig result file.
- **Privacy boundary:** the portrait ledger joins the partition test. Only `orion/body_atlas/` and
  `orion/dig/` may read it. It holds machine names and part ids, no personal data.
- **Trace that proves it works:** a live `dig portrait` printout naming athena and circe with
  their running counts, organs and watts; a `portrait write athena ...` that says "holds now" and
  a following read that prints the sentence with `lived_record`; `cmp` shows the real ledgers
  unchanged during the smoke (scratch `ORION_DIG_HOME`).
- **Dangerous failure mode:** a confident, wrong whole-self description, e.g. "I am a being whose
  world model steers my choices". Guards: the paragraph may cite only written capacities and job
  holders; the gap lines print under it, code-written; a machine sentence may name only what that
  machine runs or feeds; every line re-checks on every read and prints STALE with the failing id.
- **Disable / roll back:** remove the two dig sub-commands; delete `portrait.jsonl`. Nothing else
  reads it while the partition gate is closed.

## Acceptance checks

1. `pytest orion/body_atlas/tests orion/dig/tests tests/test_organ_names_partition.py`, green.
2. **Regression for turn 2544158f:** Orion's machine claim for athena (citing parts that run
   there) writes, says "holds now", and reads back with `lived_record`; a sentence naming a part
   on another machine that athena's parts do not feed is refused, naming that id; a write for a
   dormant machine is refused with its status.
3. **Re-check on read:** a written machine line goes STALE when its cited part stops running there.
4. **Shrink and cap:** printout ≤ 40 lines and ≤ 4,000 chars, the tail (ready lines, file path)
   always kept; every machine and organ in the full file.
5. **Eval** (`run_portrait_eval.py`) on the frozen 2026-10-31 atlas: reach precision, i.e. no id
   outside the machine's reach passes; reach recall, i.e. every running part passes; each mutation
   (drop the direction filter, drop the live filter, let organ names fail) is caught.
6. **Partition:** the new ledger name and module are in the partition test, and the test fails on
   a reader outside the allowlist.
7. **Live smoke:** `dig portrait` against the real atlas with `ORION_DIG_HOME` on a scratch copy of
   `/mnt/orion-fcc/self`; `cmp` shows the real ledgers unchanged.

## Recommended next patch

1. **Patch 1 (this design's stacked PR):** `portrait.py`, `dig portrait` and `portrait write`, the
   partition entries, tests, eval, live smoke.
2. **Patch 2:** `portrait history`: print the "I am" versions, and compare two computed portraits by
   stable id (the chronicle's "two portraits built the same way can be compared"), including a
   past-day computed portrait (no Orion words on a past body).
3. **Patch 3 (only if Juniper wants it):** a veto file for portrait lines.
