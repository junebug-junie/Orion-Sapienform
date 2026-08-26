"""What has Juniper been talking about today that they were not talking about before?

The deterministic half of Orion's curiosity loop. Produces a concrete thing to
be curious ABOUT, so the non-deterministic half (a real unified turn, in
Orion's own voice) has somewhere to go instead of free-associating.

WHY THIS CORPUS. `chat_history_log` -- Orion's own chat table -- carries 3-16
prompts a day averaging ~80 characters (measured 2026-08-25). A term cannot
surface above its own baseline in that; the signal Juniper described ("they
talked about cats 10x today") is not physically present there. Juniper's real
typed words live in the local Claude Code transcripts, and
`orion.dev_economics.claude_code_ingest.iter_all_human_messages` already parses
exactly those -- Juniper's typed turns only, never tool results, hook output,
slash-command scaffolding, or the assistant's own text. That parser is already
in production feeding the Juniper affective-state signal, under the dev-
economics spec's own framing: "no new data source, no new collection surface."
This module adds no collection; it reads what is already read.

THE STATISTIC. For each term, compare its share of today's tokens against its
share of the trailing baseline, scaled to today's volume:

    expected = (baseline_count / baseline_tokens) * recent_tokens
    lift     = recent_count / expected

That is a rate comparison, not a raw count, so it is not fooled by a busy day
producing more of everything -- and it is the reason the bars below are on
`lift` rather than on `recent_count` alone.

THREE BARS, and the third is the one that matters:

  * `MIN_RECENT_COUNT` -- said enough times to be a subject, not a slip.
  * `MIN_LIFT` -- said disproportionately more than usual.
  * `MIN_RECENT_MESSAGES` -- said across SEPARATE messages. Measured live: the
    first version of this fired on `foveal`(80) but also on a pile of worktree
    directory names pasted repeatedly inside single messages. A path echoed
    forty times in one message is one topic mentioned once. Requiring the term
    to recur across distinct messages is what separates "a subject Juniper kept
    returning to" from "a string that appeared in a paste", and it did most of
    the work of cleaning the output.

NOT A SCORE, NOT A METRIC. Nothing here is wired into field pressure, proposal
scoring, or any model. Its only consumer is a prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

# --- bars ------------------------------------------------------------------
#
# DISCLOSED, UNCALIBRATED STARTING VALUES, chosen against one real day
# (2026-08-25: 198 messages / 23,358 tokens recent, 2,078 / 252,315 baseline)
# and stated next to every number they produce, so a reader can see the bar
# rather than trust it.
MIN_RECENT_COUNT = 5
MIN_RECENT_MESSAGES = 3
MIN_LIFT = 3.0
# A term absent from the entire baseline has infinite lift. It still has to
# clear the two count bars, so "new" means "new AND repeatedly said", never
# "appeared once".
NEW_TERM_LIFT = float("inf")

DEFAULT_RECENT_HOURS = 24.0
DEFAULT_BASELINE_DAYS = 14.0

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z'-]{3,}")

# Filesystem and branch shapes, not subjects. Worktree directory names
# (`orion-sapienform-foveal-probe-via-gateway`) dominated the first live run
# and are an artifact of how Juniper works, not a thing being discussed.
_PATH_SHAPED = re.compile(r"^orion-|^services?-|-v\d+$|^[a-z]+-[a-z]+-[a-z]+")

# Deliberately a small, boring closed list rather than an imported corpus: this
# runs on every tick and must not acquire a dependency or a download for it.
_STOPWORDS = frozenset(
    """
    this that with from have what when they them then there here your yours about
    would could should which been were will just like because into more than some
    very also only know need want think really going still dont cant wont thats
    okay yeah sure right good well much over back down take give call work time
    thing things other same does doesnt didnt isnt arent lets make gonna maybe
    even said says tell look
    """.split()
)


@dataclass(frozen=True)
class SurfacedTerm:
    """One term Juniper is talking about more than they used to."""

    term: str
    recent_count: int
    recent_messages: int
    baseline_count: int
    expected_count: float
    lift: float

    @property
    def is_new(self) -> bool:
        return self.baseline_count == 0

    def describe(self) -> str:
        """One honest sentence, with the bar visible in it."""
        if self.is_new:
            return (
                f'"{self.term}" appears {self.recent_count} times across '
                f"{self.recent_messages} separate messages today, and not once "
                "in the whole baseline window."
            )
        return (
            f'"{self.term}" appears {self.recent_count} times across '
            f"{self.recent_messages} separate messages today, against "
            f"{self.expected_count:.1f} expected from its own baseline rate "
            f"({self.lift:.1f}x; {self.baseline_count} times in the baseline "
            "window)."
        )


@dataclass(frozen=True)
class SurfacingReport:
    """Everything the prompt needs, including what did NOT surface."""

    generated_at: datetime
    recent_since: datetime
    baseline_since: datetime
    recent_messages: int
    baseline_messages: int
    recent_tokens: int
    baseline_tokens: int
    terms: list[SurfacedTerm] = field(default_factory=list)

    @property
    def has_signal(self) -> bool:
        return bool(self.terms)

    @property
    def underpowered(self) -> bool:
        """True when the corpus is too thin for the comparison to mean
        anything. Distinct from `has_signal`: "nothing surfaced" and "there was
        not enough to look at" are different claims and must not be collapsed
        -- a quiet day and a broken transcript reader look identical otherwise.
        """
        return self.recent_tokens < 200 or self.baseline_tokens < 2000


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in (match.lower() for match in _TOKEN.findall(text or ""))
        if token not in _STOPWORDS and not _PATH_SHAPED.match(token)
    ]


def _window_stats(
    messages: Iterable[tuple[datetime, str]], *, since: datetime, until: datetime
) -> tuple[Counter, Counter, int]:
    """(term totals, term document-frequency, message count) for one window."""
    totals: Counter = Counter()
    documents: Counter = Counter()
    count = 0
    for stamp, text in messages:
        if not (since <= stamp < until):
            continue
        count += 1
        tokens = _tokenize(text)
        totals.update(tokens)
        documents.update(set(tokens))
    return totals, documents, count


def build_surfacing_report(
    messages: Sequence[tuple[datetime, str]],
    *,
    now: datetime,
    recent_hours: float = DEFAULT_RECENT_HOURS,
    baseline_days: float = DEFAULT_BASELINE_DAYS,
    limit: int = 8,
) -> SurfacingReport:
    """Pure. Takes (timestamp, text) pairs so it is testable without a corpus,
    a filesystem, or a clock."""
    recent_since = now - timedelta(hours=recent_hours)
    baseline_since = recent_since - timedelta(days=baseline_days)

    recent_totals, recent_docs, recent_msgs = _window_stats(
        messages, since=recent_since, until=now
    )
    baseline_totals, _, baseline_msgs = _window_stats(
        messages, since=baseline_since, until=recent_since
    )
    recent_tokens = sum(recent_totals.values())
    baseline_tokens = sum(baseline_totals.values())

    terms: list[SurfacedTerm] = []
    if recent_tokens and baseline_tokens:
        for term, count in recent_totals.items():
            if count < MIN_RECENT_COUNT or recent_docs[term] < MIN_RECENT_MESSAGES:
                continue
            baseline_count = baseline_totals.get(term, 0)
            expected = (baseline_count / baseline_tokens) * recent_tokens
            lift = (count / expected) if expected > 0 else NEW_TERM_LIFT
            if lift < MIN_LIFT:
                continue
            terms.append(
                SurfacedTerm(
                    term=term,
                    recent_count=count,
                    recent_messages=recent_docs[term],
                    baseline_count=baseline_count,
                    expected_count=expected,
                    lift=lift,
                )
            )

    # Rank by how much was actually said, not by lift: lift is unbounded for a
    # brand-new term, so ranking on it would put every one-off neologism above
    # a subject Juniper genuinely returned to all day.
    terms.sort(key=lambda t: (-t.recent_count, -t.recent_messages, t.term))
    return SurfacingReport(
        generated_at=now,
        recent_since=recent_since,
        baseline_since=baseline_since,
        recent_messages=recent_msgs,
        baseline_messages=baseline_msgs,
        recent_tokens=recent_tokens,
        baseline_tokens=baseline_tokens,
        terms=terms[:limit],
    )
