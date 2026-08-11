# Juniper affective-state signal: scoring library, real replay, live producer

Status: **DONE**. Phase 1 (this doc, originally) implemented the measure-before-minting replay
required by `docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md`'s
"What trace proves it worked" section, plus the underlying scoring library, and stopped short of
a live producer pending a steer on mount scope (see "Why no live producer yet [Phase 1]" below).
Juniper's answer, 2026-08-11: *"cocreation signals should touch the broader claude projects"* --
the whole `~/.claude/projects` tree, not scoped to just this repo. Phase 2 (same day, same PR)
builds and deploys the real producer on that basis: `orion/schemas/affective_state.py`
(`JuniperAffectiveStateV1`), the `orion:substrate:juniper_affective_state` bus channel, and
`services/orion-cocreation-signals/app/producers/affective_state.py`. **Live, deployed, and
independently verified end-to-end** -- see "Phase 2: live producer" below, including a real bug
(broken absolute symlinks) found and fixed via the actual deployment, not just review.

## What shipped

- `orion/dev_economics/claude_code_ingest.py` -- read-only parser for local Claude Code session
  transcripts (`~/.claude/projects/**/*.jsonl`). Extracts only real, Juniper-typed turns (not
  tool-result echoes, not assistant turns, not slash-command scaffolding). This is the shared
  transcript-parsing pass the dev-economics design doc names but had not yet built.
- `orion/cocreation/affective_signals.py` -- pure scoring: `swear_frequency` (frustration
  correlate) and `typo_rate` (fatigue correlate), both over a small curated word list /
  dictionary-diff, never over persisted raw text. `aggregate_scores()` sums counts before
  deriving rates (correct weighting across messages of very different lengths).
- `scripts/replay_juniper_affective_state.py` -- the actual measure-before-minting replay:
  parses every real local transcript, scores each session, and reports whether each signal is
  non-degenerate and reaches a genuine rest state (same two checks as root `CLAUDE.md`'s
  metric-quality-gate section).
- `orion/schemas/affective_state.py`, the `orion:substrate:juniper_affective_state` bus channel,
  and `services/orion-cocreation-signals/app/producers/affective_state.py` -- the live producer
  (Phase 2, below).
- 54 tests total across `orion/dev_economics/tests/`, `orion/cocreation/tests/`, and
  `services/orion-cocreation-signals/tests/`, all passing.
- `pyspellchecker==0.9.0` added to `requirements-dev.txt` and
  `services/orion-cocreation-signals/requirements.txt` (pure-Python, bundled dictionary, no
  network at runtime).

## Real replay results (113 real local sessions, ~930k spellcheck candidates)

### `swear_frequency` -- passes the live-data sanity check

- n = 113 sessions, mean 0.0005, stdev 0.0014, range 0.0000-0.0132.
- **56 of 113 sessions (49.6%) read exactly 0.0** -- a genuine rest state, not a decayed
  artifact. Real variance on top of that rest state. **Verdict: NON_DEGENERATE.**

### `typo_rate` -- fails the live-data sanity check, twice, honestly reported both times

First pass (raw dictionary diff + a short hand-picked jargon allowlist): every single session
read above zero (0/111 calm), mean 7.3%. Root-caused via `SpellChecker.unknown()` frequency
counts over the real corpus, not assumption:

1. **Real bug**: the candidate tokenizer stripped apostrophes before the spellcheck step, so
   `"doesn't"` split into `"doesn"` + `"t"` and both got flagged unknown. Fixed by keeping
   apostrophes long enough to match a `_COMMON_CONTRACTIONS` set first.
2. **Structural finding, not a bug**: the top unknown words by raw frequency (`sapienform` 1215x,
   `mnt` 1163x, `worktree` 1028x, `diff` 887x, `app` 777x, `runtime` 506x, `grep` 467x, ...) are
   overwhelmingly real, correctly-used project/software-engineering vocabulary, not typos --
   confirmed by cross-checking recurrence: a real typo is idiosyncratic (low corpus frequency),
   these words each recur across hundreds of independent messages.

Second pass (contraction fix + jargon allowlist expanded with the actual top offenders from that
frequency count, plus a plural-of-allowlisted-jargon fallback): mean dropped from 7.3% to 5.7%,
but **still 0/111 sessions read calm.** Verdict stays **SUSPICIOUS_NEVER_CALM**.

**Conclusion, stated plainly**: general-English-dictionary `typo_rate` is structurally
ill-suited to a software-engineering chat corpus. No amount of allowlist patching gets this to a
genuine rest state, because the corpus's own baseline vocabulary density guarantees a nonzero
floor -- the same shape of failure root `CLAUDE.md`'s gate names for
`bus_synaptic_prediction_error`'s permanent ~0.27 floor, here caused by domain vocabulary instead
of a distributional math artifact. Per the gate: **a metric that fails this check does not get
wired in.** `typo_rate` stays in the library (tested, real, honestly documented) but is not
exposed on any wire schema or producer in this patch. Fixing it for real needs either a
corpus-trained/expanding dictionary (not a hand-maintained allowlist -- real, ongoing maintenance
cost) or a different fatigue instrument entirely (transcripts don't carry keystroke-level
correction signal, only the final typed text).

## Why no live producer yet [Phase 1 -- resolved in Phase 2 below]

The proposal doc names `services/orion-cocreation-signals/app/producers/affective_state.py` as
the eventual thin scheduling layer, mirroring `git_delta.py`. Checking that service's
`docker-compose.yml` before building it: it mounts only this repo's own working tree
(`COCREATION_SIGNALS_REPO_HOST_PATH`), read-only. Making `claude_code_ingest.py` runnable
in-container would require a **new** read-only volume mount of Juniper's local
`~/.claude/projects` tree into that container.

That's a materially bigger exposure than the proposal doc's "no new data source, no new
collection surface" framing accounts for: `~/.claude/projects` holds full raw transcripts across
every project Juniper has ever worked on in Claude Code, not just Orion-Sapienform conversations.
Mounting the whole tree hands the container far more than "the same message text Orion already
receives in every conversation turn" -- it hands it every other project's conversations too.

There is also a direct, existing precedent against this in the repo already:
`services/orion-harness-governor/docker-compose.yml` explicitly comments *"This is NEVER the
operator's own ~/.claude"* when explaining why it uses a separate named volume instead. Building
a producer that mounts Juniper's real `~/.claude/projects` would reverse that precedent without
discussion.

Two credible fixes, neither of which I picked unilaterally:

- **Scope the mount to just this project's transcript directory**
  (`~/.claude/projects/-mnt-scripts-Orion-Sapienform*`) instead of the whole `~/.claude/projects`
  root -- keeps the exposure to "conversations about Orion," matching the co-creation framing,
  at the cost of missing affect signal from any other project Juniper works on.
- **Keep transcript reading host-side, never containerized** -- a small host-run script (cron or
  systemd timer, not Docker) computes the aggregate score locally and publishes only the derived
  scalar to the bus, so the raw transcript tree never crosses into a container filesystem at all.
  Slightly more operational surface (one more thing running outside Docker) but strictly
  narrower exposure, and doesn't touch the `orion-harness-governor` precedent.

## Phase 2: live producer (2026-08-11, same day, resolved by Juniper's steer)

Juniper: *"cocreation signals should touch the broader claude projects"* -- picks the whole-tree
mount over the scoped alternative above. Built on that basis:

- `orion/schemas/affective_state.py` -- `JuniperAffectiveStateV1`. Deliberately narrower than the
  scoring library: only `swear_frequency` is on the wire. `typo_rate` failed the replay's
  live-data check (see above) and is not exposed here at all, not even as a known-unreliable
  field -- per the metric quality gate, a failing metric does not get wired in.
- `orion/bus/channels.yaml` + `orion/schemas/registry.py` -- new channel
  `orion:substrate:juniper_affective_state`, `consumer_services: []` (pure shadow write, same
  precedent as `bus_synaptic_prediction_error`'s own rollout -- publish real data first, let a
  live-data sanity pass on the live stream itself, not just the offline replay, justify a
  consumer later).
- `services/orion-cocreation-signals/app/producers/affective_state.py` -- new producer loop,
  structurally mirrors `pr_lifecycle_loop` (half-open window tiling, publish-every-tick since an
  all-zero window is a real observation, same restart-loss gap and same reasoning for accepting
  it). `settings.py`/`main.py`/`docker-compose.yml`/`.env_example` wired to match.
- Default **enabled** after live verification (below) -- `COCREATION_SIGNALS_AFFECTIVE_STATE_ENABLED=true`
  in the real deployed `.env`. Still a pure shadow write (no consumer), so this is "publish real
  data" risk only, not "change Orion-facing behavior" risk.

### Real bugs found via actual deployment, not just review

Deployed to the real `orion-athena-cocreation-signals` container against Juniper's real
`~/.claude/projects` mount (not a staging environment) and watched it fail, twice, before it
worked -- exactly the "runtime truth beats config truth" discipline root `CLAUDE.md` names.

1. **Real crash, first deploy**: `FileNotFoundError` inside `_score_window`, killing the whole
   tick. Root cause: `iter_transcript_files()`'s glob-based walk and `parse_transcript_file()`'s
   open were not resilient to a file becoming unreachable between listing and opening. Fixed
   `iter_all_human_messages()` to catch `OSError` per file and skip, not abort the whole walk.
2. **Root cause of #1, found by inspecting the actual failing path, not assumed**: initially
   misdiagnosed as a delete-mid-walk race (transcript cleanup). Actually structural and 100%
   reproducible, not intermittent: Claude Code represents a cross-project subagent transcript as
   an **absolute-path symlink** (a file under one session's `subagents/` dir pointing at
   `/home/athena/.claude/projects/<other-project>/.../agent-*.jsonl`). The container's mount
   landed at `/claude-projects`, a different path than the host's real
   `/home/athena/.claude/projects` -- every such symlink resolved against the container's own
   (empty) root filesystem instead, permanently, every tick. Confirmed by `readlink -f` on the
   host and `docker exec ... ls -la` on the same path inside the container, side by side. Real
   fix: mount at the *identical* path (`COCREATION_SIGNALS_CLAUDE_PROJECTS_PATH` now required to
   equal `COCREATION_SIGNALS_CLAUDE_PROJECTS_HOST_PATH`), not a container-convenient rename. The
   defensive per-file `OSError` catch from #1 stays in place regardless (a real transient race is
   still plausible and would look identical) -- second fix corrects the actual root cause; first
   fix keeps the system fail-open around whatever's left.
3. **Second code review pass (post-fix, pre-final-deploy)** caught 3 more real issues, all fixed:
   `iter_transcript_files()`'s glob (not just the per-file open) could itself abort the walk if a
   whole subdirectory vanished mid-scan -- switched to `os.walk(onerror=...)`, which skips a
   failing subtree instead of raising; a genuinely timezone-naive timestamp would raise
   `TypeError` when compared against the window's aware datetimes -- now normalized to UTC on
   parse; and the producer was running a full spellcheck pass over every message every tick
   (~31k words on the real corpus) purely to compute `typo_rate`, a value this schema never
   exposes -- added `score_message(text, compute_typos=False)` and switched the producer to it.

### Live verification (real evidence, not a claim)

After both fixes, a clean real tick against the real mount:

```text
cocreation_affective_state_published message_count=70 word_count=31819 swear_count=23 \
  swear_frequency=0.0007228385555799994
```

Zero `transcript_vanished_during_walk` warnings on this run (113 real sessions' worth were logged
and gracefully handled on the two runs before the symlink fix landed). Independently confirmed
the event actually crossed the real bus, not just logged locally -- a real `redis-cli subscribe`
against `orion:substrate:juniper_affective_state` on the real bus host received the correctly
-shaped envelope in real time on container restart:

```json
{"schema":"orion.envelope","schema_version":"2.0.0","kind":"juniper.affective_state.v1",
 "source":{"name":"cocreation-signals","node":"athena","version":"0.1.0"},
 "payload":{"schema_version":"juniper_affective_state.v1","message_count":70,
            "word_count":31819,"swear_count":23,"swear_frequency":0.000722...}}
```

No raw message text anywhere in that payload, matching the privacy boundary.

## Tests run

```text
venv/bin/python3 -m pytest orion/dev_economics/tests orion/cocreation/tests \
  services/orion-cocreation-signals/tests -q
54 passed
```

## Evals run

The replay script *is* the offline eval Phase 1 needed. Phase 2 added a real *live* eval on top
-- the actual deployed container against the actual production transcript mount, watched fail
and recover twice (see "Real bugs found via actual deployment" above), which is stronger
evidence than the offline replay alone. Per-session scores were intentionally not committed to
the repo (aggregate distribution stats only, in this report) -- the same privacy boundary the
proposal doc names applies to replay artifacts, not just production storage.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-cocreation-signals build   # 3 builds across the fix cycle
scripts/safe_docker_build.sh orion-cocreation-signals up -d   # 3 redeploys
docker logs orion-athena-cocreation-signals --tail 40          # confirmed each fix, then a clean tick
redis-cli -h <bus-host> -p 6379 -n 0 subscribe orion:substrate:juniper_affective_state
  # independently confirmed a real envelope crossing the real bus
```

## Review findings fixed

Two `orion-repo-agent` subagent review passes (`ReportFindings` tool wasn't available to it, so
it reported inline; findings verified by direct reproduction, not just static reading): one
against the Phase 1 scoring/parsing library, one against the Phase 2 producer/schema/bus/settings
wiring. 9 + 6 findings total; material ones fixed below, three documented as accepted gaps.

### Phase 1 (scoring library, transcript parser)

- Finding: `_typo_candidates()` stripped apostrophes before the contraction check, splitting
  common contractions into bogus unknown-word fragments and inflating `typo_rate` on correctly
  spelled English.
  - Fix: check the whole token against `_COMMON_CONTRACTIONS` before stripping apostrophes.
  - Evidence: replay mean typo_rate for the affected corpus dropped from 7.3% to 5.7% after the
    fix (see "Real replay results" above); tests pass.
- Finding (subagent, high confidence, reproduced live): the fix above is ASCII-apostrophe-only --
  curly/smart quotes (`’`/`‘`, what iOS/macOS/many web fields actually type) defeat both
  `_COMMON_CONTRACTIONS` and `_WORD_PATTERN`, reproducing the identical bug class on 15 real
  messages in this repo's own transcript corpus.
  - Fix: `_normalize_apostrophes()` translates curly quotes to ASCII `'` before either
    `tokenize()` or `_typo_candidates()` runs (also consolidates the two into one shared
    `_raw_tokens()` regex pass, fixing the subagent's separate low-severity double-tokenization
    finding at the same time).
  - Evidence: `test_score_message_typo_rate_handles_curly_apostrophe_contractions` (new).
- Finding (subagent, medium confidence, reproduced live): `count_swear_words` never matched a
  swear word used with a trailing possessive/contracted `'s` (`"shit's broken"` scored 0, not 1)
  -- `tokenize()` keeps apostrophes but `SWEAR_WORDS` only has bare forms.
  - Fix: also match `tok[:-2] in SWEAR_WORDS` when `tok` ends in `'s`.
  - Evidence: `test_count_swear_words_matches_possessive_or_contracted_form` (new).
- Finding (subagent, high confidence, reproduced live): a non-string `timestamp` field (e.g. a
  raw epoch int) raised `AttributeError` inside `parse_transcript_file`, uncaught by the
  `except ValueError`, contradicting the module's own "single corrupt line must not abort the
  whole file" guarantee. Not observed in this repo's real corpus (all 1246 real transcripts have
  string timestamps) but a real latent gap.
  - Fix: `isinstance(timestamp_raw, str)` guard before parsing.
  - Evidence: `test_parse_transcript_file_drops_non_string_timestamp_without_crashing` (new).
- Finding (subagent, very low severity): `SpellChecker(distance=1)` sets a parameter that only
  affects `correction()`/`candidates()`, neither of which this module calls (only `.unknown()`).
  - Fix: removed the dead parameter, documented why in a comment.
- Finding (subagent, medium-high confidence): the module docstring's claim that list-type
  `message.content` is "never something Juniper typed" is not quite true -- ~1 genuine short
  reply out of ~83k list-content turns in the real corpus is dropped by this rule.
  - **Accepted gap, not fixed**: loosening the filter to admit lone `{"type": "text"}` blocks
    risks instead admitting synthetic harness injections (e.g. `"[Request interrupted by
    user]"`) as if Juniper had typed them -- a worse failure for this signal's purpose than
    under-counting a rare real message. Docstring updated to state the tradeoff explicitly
    instead of the absolute "never" claim.
- Finding (subagent, medium confidence, doc-compliance): the proposal doc's "What trace proves
  it worked" section names three checks (non-degenerate, genuine rest state, **and** a real
  theory-anchored correlation with independently-identifiable frustrating sessions). This
  replay implements only the first two -- there's no ground-truth label set to check a
  correlation against yet.
  - **Accepted gap, not fixed this patch**: named explicitly here as a real, open requirement
    of the governing spec, not silently dropped. Follow-up: hand-label a small set of sessions
    Juniper independently remembers as frustrating/calm, then check whether `swear_frequency`
    actually separates them.
- Finding (subagent, low severity, not fixed): malformed/mismatched wrapper tags (e.g.
  `<command-name>foo</command-args>`) aren't stripped by `_WRAPPER_TAG_PATTERN`'s backreference
  match, leaving tag boilerplate in the scored text. Low real-world likelihood (transcripts are
  well-formed machine output); left as a known limitation rather than adding untested
  malformed-input handling.

### Phase 2 (producer/schema/bus/settings/compose wiring)

- Finding (subagent, high confidence, real cost confirmed on the live corpus): the producer
  called `score_message()` in its full mode, running a real spellcheck pass over every message
  every tick purely to compute `typo_rate` -- a value this schema never exposes (~31k words
  spellchecked for nothing on the real corpus, every 15 minutes, forever).
  - Fix: `score_message(text, compute_typos=False)` fast path; producer switched to it.
  - Evidence: `test_score_message_compute_typos_false_skips_spellcheck_entirely` (new).
- Finding (subagent, medium confidence): the file-vanish fix (Phase 2's bug #1 above) only
  wrapped the per-file open, not `iter_transcript_files()`'s glob-based directory walk itself --
  a whole subdirectory disappearing mid-walk (not just one file) would still abort the tick,
  narrower protection than the module's own docstring claimed.
  - Fix: switched to `os.walk(root, onerror=lambda _exc: None)`, which skips a failing subtree
    instead of raising, at the directory level not just the file level.
  - Evidence: `test_iter_transcript_files_survives_a_subdirectory_vanishing_mid_walk` (new).
- Finding (subagent, low confidence, speculative): a genuinely timezone-naive parsed timestamp
  (no `Z`/offset in the source string) would raise `TypeError` when compared against the
  window's timezone-aware `since`/`until` -- no evidence this occurs in the real corpus, but
  cheap to close outright rather than rely on the per-tick exception handler silently dropping
  that tick.
  - Fix: naive timestamps normalized to UTC on parse in `parse_transcript_file`.
  - Evidence: `test_parse_transcript_file_normalizes_naive_timestamp_to_utc` (new).
- Confirmed non-issues (explicitly checked, not just assumed): no raw transcript text logged or
  persisted anywhere in the new wiring (traced every log/exception call site); window-tiling and
  cold-start logic match `pr_lifecycle_loop`'s pattern exactly, no undocumented divergence;
  `docker-compose.yml`/`settings.py`/`.env_example` three-way key parity verified, no orphans;
  `JuniperAffectiveStateV1`'s registration in `registry.py`'s `_REGISTRY` (not the separate
  `SCHEMA_REGISTRY` dict) matches `CodebaseDeltaV1`'s own existing, correct precedent -- not a
  partial registration; no resource leak or unbounded growth beyond the already-accepted
  full-tree-rescan-per-tick cost `pr_lifecycle_loop` already has.

### The symlink bug itself (found live, not by review)

Not a review finding -- found by watching the real deployed container fail twice and
root-causing it against the real filesystem (see "Real bugs found via actual deployment" above).
Included here for completeness since it's the most consequential fix in this patch: mounting
`~/.claude/projects` at a different in-container path silently broke every cross-project
subagent-transcript symlink, permanently and every tick, until the mount was made to match the
host path exactly.

## Restart required

Already done live during this patch (not deferred to the operator):

```bash
scripts/safe_docker_build.sh orion-cocreation-signals build
scripts/safe_docker_build.sh orion-cocreation-signals up -d
```

`orion-athena-cocreation-signals` is running the final code as of this report, with
`COCREATION_SIGNALS_AFFECTIVE_STATE_ENABLED=true` in the real deployed `.env`. No further restart
needed unless this PR's code changes further after this report.

## Risks / concerns

- Severity: note, not blocking.
  - Concern: `typo_rate`'s jargon allowlist is hand-maintained and will drift as this repo's own
    vocabulary grows -- a real, ongoing cost, not a one-time fix.
  - Mitigation: re-run `scripts/replay_juniper_affective_state.py` periodically and fold new
    high-frequency unknown words back into the allowlist, the same way this patch's own
    allowlist was derived from a real corpus run rather than guessed.
- Severity: note, not blocking.
  - Concern: the correlation check named in the proposal doc's "What trace proves it worked"
    section (does `swear_frequency` actually separate independently-identifiable frustrating
    sessions from calm ones?) is still not done -- no ground-truth label set exists yet.
  - Mitigation: hand-label a small set of sessions Juniper independently remembers as
    frustrating/calm as a follow-up, then check separation.
- Severity: note, not blocking.
  - Concern: `services/orion-cocreation-signals/.env` did not exist anywhere in the shared
    checkout before this patch (the worktree it was originally deployed from,
    `Orion-Sapienform-cocreation-signals-service`, was pruned after merge, taking its gitignored
    `.env` with it, even though the container it built kept running). Reconstructed from the live
    container's actual env (`docker inspect`) plus this patch's new keys -- real values, not
    guesses, but worth a sanity check by Juniper since it includes a live GitHub token recovered
    from the running container rather than freshly issued.
  - Mitigation: none needed unless the recovered token should be rotated; flagging for
    awareness only.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1552
