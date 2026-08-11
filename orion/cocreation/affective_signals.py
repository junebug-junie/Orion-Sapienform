"""Pure word-level scoring for the Juniper affective/engagement-state signal
(``docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md``).

Two correlates, per that doc's "What capability changes" section:

- ``swear_frequency`` -- a frustration correlate.
- ``typo_rate`` -- a fatigue/busyness correlate.

Both operate on already-in-hand message text (see
``orion.dev_economics.claude_code_ingest``) and return only aggregate scalars.
Nothing here persists raw text or which specific words tripped a score --
callers own that boundary, but this module makes it easy to respect: no
function below returns per-word detail, only counts and rates.

Word list note (no-keyword-cathedral gate, root CLAUDE.md §0A): ``SWEAR_WORDS``
is a small, deliberately curated set, each entry directly consumed by
``swear_frequency``'s count -- not a taxonomy or an attempt at exhaustive
profanity coverage. A narrower list undercounts (real signal, just noisier);
a broader list starts flagging normal technical language ("kill the process",
"nuke the cache") as swearing, which is a worse failure for a frustration
proxy than undercounting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

SWEAR_WORDS: frozenset[str] = frozenset(
    {
        "fuck",
        "fucking",
        "fucked",
        "fucker",
        "shit",
        "shitty",
        "damn",
        "damnit",
        "goddamn",
        "ass",
        "asshole",
        "bullshit",
        "crap",
        "bastard",
        "piss",
        "pissed",
        "hell",
        "wtf",
    }
)

# Domain jargon that a general-English dictionary flags as a typo but Juniper
# uses correctly and constantly -- excluding these keeps typo_rate a fatigue
# proxy, not a "how much Orion-specific vocabulary did you use" proxy. Kept
# short and lowercase; matched case-insensitively like everything else here.
_JARGON_ALLOWLIST: frozenset[str] = frozenset(
    {
        "orion",
        "juniper",
        "falkordb",
        "postgres",
        "postgresql",
        "redis",
        "docker",
        "kubectl",
        "pytest",
        "github",
        "gitlab",
        "graphify",
        "sql",
        "json",
        "yaml",
        "toml",
        "api",
        "apis",
        "cli",
        "http",
        "https",
        "url",
        "urls",
        "env",
        "envs",
        "repo",
        "repos",
        "config",
        "configs",
        "docstring",
        "docstrings",
        "async",
        "sync",
        "backend",
        "frontend",
        "webhook",
        "webhooks",
        "healthcheck",
        "sql-writer",
        "prometheus",
        "grafana",
        "cortex",
        "recall",
        "substrate",
        "hub",
        "cocreation",
        # Everything below was pulled from the top of a real, live corpus-
        # frequency count over 113 real transcripts (~930k spellcheck
        # candidates) run 2026-08-11 -- see
        # scripts/replay_juniper_affective_state.py and
        # docs/superpowers/pr-reports/2026-08-11-juniper-affective-state-
        # signal-replay.md. Every word here recurred across dozens to
        # thousands of independent messages, the opposite signature of an
        # idiosyncratic typo. This list needs periodic refresh as this
        # project's own vocabulary evolves -- a real, disclosed maintenance
        # cost of using general-English typo_rate as a fatigue proxy on a
        # software-engineering transcript corpus, not a one-time fix.
        "mnt", "worktree", "worktrees", "diff", "diffs", "app", "runtime",
        "grep", "readme", "readmes", "dict", "pre", "tmp", "str", "codebase",
        "pys", "etc", "yml", "prs", "venv", "metacog", "multi", "orions",
        "falkor", "txt", "pydantic", "stdout", "stderr", "reportfindings",
        "orch", "jsonl", "asyncio", "rdf", "fuseki", "ctx", "hardcoded",
        "metadata", "mds", "init", "timestamp", "datetime", "stat", "bool",
        "regex", "int", "fcc", "def", "llm", "llms", "driveengine",
        "introspector", "pythonpath", "tuple", "graphdb", "selfstatev",
        "upsert", "evals", "dataclass", "html", "eval", "jsonb", "sha",
        "uuid", "kwargs", "args", "func", "curl", "middleware", "linter",
        "cwd", "requirements", "dockerfile", "namespace", "namespaces",
    }
)

_WORD_PATTERN = re.compile(r"[A-Za-z']+")
# All-caps tokens up to length 5 are treated as acronyms (PR, SQL, API, HTTP,
# ...), not spelled-out words -- a real dictionary would flag most of them.
_ACRONYM_MAX_LEN = 5

# Curly/smart quotes (U+2019 RIGHT SINGLE QUOTATION MARK, U+2018 LEFT SINGLE
# QUOTATION MARK) are what iOS/macOS/many web text fields actually type for
# an apostrophe -- confirmed live 2026-08-11 against this real transcript
# corpus: 15 real messages contain one. Normalized to ASCII ``'`` before
# either tokenize() or _typo_candidates() runs, so _COMMON_CONTRACTIONS and
# the apostrophe-aware swear/typo logic below see one consistent form instead
# of silently missing every curly-quote contraction.
_CURLY_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


def _normalize_apostrophes(text: str) -> str:
    return text.translate(_CURLY_APOSTROPHES)


def _raw_tokens(text: str) -> list[str]:
    """Apostrophe-normalized word tokens, shared by tokenize() and
    _typo_candidates() so the text is only regex-scanned once per call."""
    return _WORD_PATTERN.findall(_normalize_apostrophes(text))


# Common contractions -- excluded from spellcheck entirely rather than passed
# through as bare words. Confirmed live 2026-08-11 (replay against 113 real
# transcripts): stripping the apostrophe before spellchecking splits "doesn't"
# into "doesn" + "t", and both get flagged as unknown -- inflating typo_rate
# on ordinary, correctly-spelled English, not fatigue-driven errors.
_COMMON_CONTRACTIONS: frozenset[str] = frozenset(
    {
        "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't",
        "won't", "wouldn't", "can't", "couldn't", "shouldn't",
        "haven't", "hasn't", "hadn't",
        "i'm", "i've", "i'll", "i'd", "it's", "that's", "there's", "here's",
        "let's", "you're", "you've", "you'll", "you'd",
        "we're", "we've", "we'll", "we'd",
        "they're", "they've", "they'll", "they'd",
        "what's", "who's", "where's", "how's", "let's",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Deliberately simple (``[A-Za-z']+``) -- this
    is a frequency/rate denominator, not a linguistic tokenizer; code
    fragments, paths, and URLs naturally fragment into short, mostly-excluded
    tokens rather than being specially detected."""
    return [w.lower() for w in _raw_tokens(text)]


def count_swear_words(tokens: list[str]) -> int:
    """Matches a bare swear word, or one with a trailing possessive/contracted
    ``'s`` (``"shit's broken"``, ``"what the hell's going on"``) -- tokenize()
    keeps apostrophes, so without this a swear word used as a contraction
    would silently never match ``SWEAR_WORDS``'s bare forms. Confirmed live
    2026-08-11: undercounted on real messages before this fix."""
    count = 0
    for tok in tokens:
        if tok in SWEAR_WORDS:
            count += 1
        elif tok.endswith("'s") and tok[:-2] in SWEAR_WORDS:
            count += 1
    return count


@lru_cache(maxsize=1)
def _spellchecker():
    # Imported lazily and cached: this pulls in pyspellchecker's bundled
    # frequency dictionary on first use only, not at module import time.
    # Only .unknown() (pure dictionary membership) is ever called on this --
    # no correction()/candidates() call, so there's no edit-distance
    # parameter to set here.
    from spellchecker import SpellChecker

    return SpellChecker()


def _typo_candidates(text: str) -> list[str]:
    """Words eligible for spellcheck: tokens (apostrophes kept so a whole
    contraction can be recognized before splitting), length >= 3, not a
    common contraction, not an all-caps acronym, not in the jargon
    allowlist. Apostrophes are stripped only after the contraction check, so
    a contraction that survives to the spellchecker is never split into two
    bogus fragments (see ``_COMMON_CONTRACTIONS``'s docstring note)."""
    candidates = []
    for raw in _raw_tokens(text):
        lowered = raw.lower()
        if lowered in _COMMON_CONTRACTIONS:
            continue
        stripped = raw.replace("'", "")
        if len(stripped) < 3:
            continue
        if stripped.isupper() and len(stripped) <= _ACRONYM_MAX_LEN:
            continue
        lowered_stripped = stripped.lower()
        if lowered_stripped in _JARGON_ALLOWLIST:
            continue
        # Plain plural of an allowlisted term (e.g. a future "orion-foo"
        # service referenced as "orion-foos") -- cheaper than growing the
        # allowlist by every inflection of every entry already in it.
        if lowered_stripped.endswith("s") and lowered_stripped[:-1] in _JARGON_ALLOWLIST:
            continue
        candidates.append(lowered_stripped)
    return candidates


@dataclass(frozen=True)
class AffectiveWordScores:
    """Aggregate scores over one or more messages. Rates are ``None`` (not a
    fabricated 0.0) when their denominator is zero -- e.g. a message with no
    spellcheckable words yields ``typo_rate=None``, not "0% typos"."""

    word_count: int
    swear_count: int
    checked_word_count: int
    unknown_word_count: int

    @property
    def swear_frequency(self) -> float | None:
        if self.word_count == 0:
            return None
        return self.swear_count / self.word_count

    @property
    def typo_rate(self) -> float | None:
        if self.checked_word_count == 0:
            return None
        return self.unknown_word_count / self.checked_word_count


def score_message(text: str, *, compute_typos: bool = True) -> AffectiveWordScores:
    """``compute_typos=False`` skips ``_typo_candidates``/the spellcheck pass
    entirely -- for a caller that only wants ``swear_frequency`` (e.g.
    ``affective_state.py``'s producer, which never reads ``typo_rate`` since
    it's not on the wire schema; see that schema's own docstring for why),
    running a real dictionary lookup over every eligible word in every
    message is pure waste. Confirmed live 2026-08-11: a real production tick
    tokenized/spellchecked ~31k words purely to compute a value nothing
    downstream ever read. ``checked_word_count``/``unknown_word_count`` (and
    therefore ``typo_rate``) are ``0``/``None`` in this mode -- indistinguishable
    from "nothing was spellcheckable in this text", which is an accepted,
    documented ambiguity since no caller needs to tell the two apart today."""
    tokens = tokenize(text)
    swear_count = count_swear_words(tokens)
    if compute_typos:
        candidates = _typo_candidates(text)
        unknown = _spellchecker().unknown(candidates) if candidates else set()
    else:
        candidates, unknown = [], set()
    return AffectiveWordScores(
        word_count=len(tokens),
        swear_count=swear_count,
        checked_word_count=len(candidates),
        unknown_word_count=len(unknown),
    )


def aggregate_scores(scores: list[AffectiveWordScores]) -> AffectiveWordScores:
    """Sums counts first, then derives rates from the totals -- not an
    average of per-message rates, which would over-weight short messages
    (a 1-word message with a swear in it is 100% swear_frequency on its own,
    but should not carry equal weight to a 200-word message with the same
    swear count)."""
    return AffectiveWordScores(
        word_count=sum(s.word_count for s in scores),
        swear_count=sum(s.swear_count for s in scores),
        checked_word_count=sum(s.checked_word_count for s in scores),
        unknown_word_count=sum(s.unknown_word_count for s in scores),
    )
