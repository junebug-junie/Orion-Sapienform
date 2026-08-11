# Juniper affective-state signal: scoring library + replay findings

Status: **DONE_WITH_CONCERNS**. Implements the measure-before-minting replay required by
`docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md`'s "What trace
proves it worked" section, and the underlying scoring library. Does **not** wire a live producer
-- the replay surfaced a real architectural/privacy mismatch (see "Why no live producer yet"
below) that the original proposal doc didn't fully account for, and that needs an explicit steer
from Juniper before it's built.

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
- 20 tests across both new modules (`orion/dev_economics/tests/test_claude_code_ingest.py`,
  `orion/cocreation/tests/test_affective_signals.py`), all passing.
- `pyspellchecker==0.9.0` added to `requirements-dev.txt` (pure-Python, bundled dictionary, no
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

## Why no live producer yet (the real finding this replay surfaced)

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

## Tests run

```text
venv/bin/python3 -m pytest orion/dev_economics/tests/test_claude_code_ingest.py \
  orion/cocreation/tests/test_affective_signals.py -q
20 passed
```

## Evals run

The replay script itself *is* the eval this signal needed before any live wiring -- see results
above. Per-session scores were intentionally not committed to the repo (aggregate distribution
stats only, in this report) -- the same privacy boundary the proposal doc names applies to
replay artifacts, not just production storage.

## Docker/build/smoke checks

None -- no service/container/compose changes in this patch (see "Why no live producer yet").

## Review findings fixed

Ran `orion-repo-agent` in a subagent against the pre-fix diff (`ReportFindings` tool wasn't
available to it, so it reported inline; findings verified by direct reproduction, not just
static reading). 9 findings total; material ones fixed below, two documented as accepted gaps.

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

## Restart required

No restart required -- no running service changed.

## Risks / concerns

- Severity: should-fix before wiring a live producer.
  - Concern: the transcript-mount scope question above is unresolved.
  - Mitigation: needs Juniper's explicit steer (scoped mount vs. host-side-only scoring) before
    `services/orion-cocreation-signals/app/producers/affective_state.py` gets built.
- Severity: note, not blocking.
  - Concern: `typo_rate`'s jargon allowlist is hand-maintained and will drift as this repo's own
    vocabulary grows -- a real, ongoing cost, not a one-time fix.
  - Mitigation: re-run `scripts/replay_juniper_affective_state.py` periodically and fold new
    high-frequency unknown words back into the allowlist, the same way this patch's own
    allowlist was derived from a real corpus run rather than guessed.

## PR link

(added after push)
