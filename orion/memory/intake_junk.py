"""Is a user prompt worth remembering on its own? (memory intake, Stage 0A)

The consolidation gate used to judge a window by the user's prompt AND Orion's
reply together. Orion's reply is almost never small talk, so "sup", "hi" and
"Run github compactor." all passed as substantive and became memories. This
module judges the user's prompt alone and answers one narrow question: is it a
greeting/filler, or a Hub skill command? Anything else is left for the rest of
the gate to judge -- this filter only removes, it never admits.

Commands come from the real Hub workflow registry
(`orion.cognition.workflows.registry`), not a hand-kept list, so a new skill
alias is filtered the day it ships.

Design spec: docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md
(Stage 0, "stop command and greeting turns entering").
"""

from __future__ import annotations

import re
from typing import Literal

from orion.memory.low_info_social import is_low_info_social

JunkReason = Literal["low_info_social", "hub_command"]

# Greetings, acknowledgements and filler. A prompt made only of these (plus
# stopwords) carries nothing to remember.
_FILLER = frozenset(
    {
        "hi", "hey", "hello", "yo", "sup", "hiya", "howdy", "heya", "morning",
        "evening", "afternoon", "night", "gm", "gn",
        "thanks", "thank", "thx", "ty", "tysm", "cheers",
        "ok", "okay", "kk", "yep", "yup", "yeah", "yes", "nope", "nah",
        "sure", "cool", "nice", "great", "awesome", "lol", "haha", "hehe",
        "hmm", "hm", "oh", "ah", "oooh", "ooh", "wow", "woot", "meh",
        "orion", "juniper", "friend", "buddy", "dude", "bruh",
        "good", "fine", "well", "doing", "going", "goes", "things",
        "again", "too", "also", "just", "so", "sooo", "please", "pls", "plz",
    }
)

# Plain English function words. Deliberately small and boring: content words
# (queue, Austin, labs, mom) are never in here.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of",
        "in", "on", "at", "for", "with", "from", "by", "about", "as", "into",
        "i", "im", "i'm", "ive", "i've", "me", "my", "you", "your", "you're",
        "youre", "we", "us", "our", "it", "its", "it's", "this", "that",
        "these", "those", "there", "here", "is", "are", "was", "were", "be",
        "been", "am", "do", "does", "did", "have", "has", "had", "got", "get",
        "what", "whats", "what's", "which", "who", "how", "hows", "how's",
        "why", "when", "where", "else", "any", "some", "all", "up", "out",
        "now", "still", "can", "could", "would", "will", "should",
    }
)

# Negations are CONTENT, never filler: "I'm not ok" and "not good" are how a
# bad day is said briefly (review of PR #2457). Any word in this set, or
# ending in "n't", counts as a content word.
_NEGATIONS = frozenset({"not", "no", "never", "nothing", "nobody", "cant", "dont", "wont", "isnt"})

# Content words that are themselves social small talk, so a question made only
# of them ("you back?", "what's new?", "what else is on your mind?") asks
# nothing. A question with any OTHER content word ("where is mom?", "when is
# the surgery?") is kept.
_SOCIAL_QUESTION_WORDS = frozenset(
    {"back", "new", "mind", "happening", "crackalacking", "today", "tonight", "there", "around"}
)

# Words that open an acknowledgement ("thanks for those updates", "huh? sorry,
# not following"). An ack with one content word after it is still an ack.
# Greetings ("hey", "hi") are deliberately NOT here: "hey, I'm pregnant" opens
# with a greeting and is the most important thing said all week.
_ACK_OPENERS = frozenset(
    {
        "thanks", "thank", "thx", "ty", "tysm", "ok", "okay", "kk", "cool",
        "nice", "great", "awesome", "haha", "lol", "hehe", "huh", "hmm", "yup",
        "yep", "yeah", "sure", "gotcha",
    }
)

# Words that make a prompt a question even without a "?".
_QUESTION_OPENERS = frozenset(
    {"what", "whats", "what's", "how", "hows", "how's", "which", "who", "where", "why", "when"}
)

# Unicode-aware: any script's letters and digits make words. The word lists
# above are English, so a word in another script is never in them and always
# counts as content.
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_NON_ASCII_LETTER_RE = re.compile(r"[^\W\d_a-zA-Z]", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w.strip("'") for w in _WORD_RE.findall(str(text or "").lower()) if w.strip("'")]


def _is_negation(word: str) -> bool:
    return word in _NEGATIONS or word.endswith("n't")


def _content_words(words: list[str]) -> list[str]:
    return [
        w
        for w in words
        if _is_negation(w) or (w not in _FILLER and w not in _STOPWORDS)
    ]


def _has_unjudgeable_letters(text: str) -> bool:
    """Letters outside plain ASCII (Cyrillic, CJK, Hebrew, accented Latin).

    The filler and stopword lists are English, so text with letters these
    rules cannot read is never classified "no content" (review of PR #2457:
    'мама умерла сегодня' was dropped as small talk).
    """
    return bool(_NON_ASCII_LETTER_RE.search(str(text or "")))


def is_low_info_prompt(prompt: str) -> bool:
    """True for a user prompt with nothing in it to remember.

    Four rules, each narrow on purpose (lean toward remembering):

    1. The existing courtesy check (`is_low_info_social`): "hi", "thanks".
    2. Nothing but greetings, filler and function words: "sup yo", "ty!",
       "howdy, how goes it".
    3. A purely social question -- every content word is small-talk
       vocabulary: "you back?", "what's new?", "what else is on your mind?".
       A question with any other content word is kept ("where is mom?").
    4. A short acknowledgement with one non-negated content word: "thanks
       for those updates".

    Never junk: text with letters these English lists cannot judge
    (non-Latin scripts, accented Latin), and any negation ("I'm not ok").
    A short *statement* is kept -- "I've got the blues." and "sleepy" are real.
    """
    text = str(prompt or "").strip()
    if not text:
        return True
    if _has_unjudgeable_letters(text):
        return False
    words = _words(text)
    content = _content_words(words)
    if any(_is_negation(w) for w in content):
        return False
    if is_low_info_social(text):
        return True
    if not content:
        return True
    short = len(words) <= 8
    is_question = text.endswith("?") or words[0] in _QUESTION_OPENERS
    if short and is_question and all(w in _SOCIAL_QUESTION_WORDS for w in content):
        return True
    if short and len(content) <= 1 and words[0] in _ACK_OPENERS:
        return True
    return False


# How many words a command may carry beyond its alias ("please", "now", a
# greeting) and still be a command rather than a sentence that mentions one.
_COMMAND_SLACK_WORDS = 3


def hub_command_workflow(prompt: str) -> str | None:
    """The Hub workflow id when the prompt IS a skill command, else None.

    `resolve_user_workflow_invocation` matches an alias anywhere in the text,
    which is right for routing but too loose for memory: "I keep wondering
    what have we been building, and whether it matters" contains an alias and
    is not a command. So the prompt must be the alias plus a few words at most.
    The time-bounded journal command ("journal the last 34 minutes") has no
    fixed alias; its own resolver already requires the prompt to start with
    the command verb, so it is accepted as-is.
    """
    try:
        from orion.cognition.workflows.registry import resolve_user_workflow_invocation
    except Exception:  # pragma: no cover - registry import failure must not break intake
        return None
    match = resolve_user_workflow_invocation(prompt or "")
    if match is None:
        return None
    if match.matched_alias.endswith("_v1"):
        # Synthetic alias of the time-bounded journal resolver.
        # The window phrase ("the last 34 minutes") is the command itself;
        # anything past ~12 words is a message, not a command.
        return match.workflow_id if len(_words(prompt)) <= 12 else None
    prompt_words = match.normalized_prompt.split()
    alias_words = match.matched_alias.split()
    extra = list(prompt_words)
    # Remove the alias occurrence; what is left is what the user added.
    for i in range(len(prompt_words) - len(alias_words) + 1):
        if prompt_words[i : i + len(alias_words)] == alias_words:
            extra = prompt_words[:i] + prompt_words[i + len(alias_words) :]
            break
    # The slack may only be politeness ("please", "now", "hey orion"). Any real
    # content -- "Do a journal pass about my labs" -- makes it a memory too
    # (review of PR #2457).
    if len(extra) <= _COMMAND_SLACK_WORDS and not _content_words(extra) and not _has_unjudgeable_letters(prompt):
        return match.workflow_id
    return None


def prompt_junk_reason(prompt: str) -> JunkReason | None:
    """Why this prompt should not become a memory on its own, or None."""
    if hub_command_workflow(prompt) is not None:
        return "hub_command"
    if is_low_info_prompt(prompt):
        return "low_info_social"
    return None
