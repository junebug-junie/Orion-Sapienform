"""Which experience outcomes the Inner Sky draws as trouble. Pure: no settings, no I/O, so the page route and
the feed eval (evals/run_inner_sky_feed_eval.py) read the one table."""

from __future__ import annotations

from typing import Any, Optional

# What a row's own verdict says happened, for the page's colour: "red" = broken, "amber" = a problem while things
# still work, None = normal (a guard that did not fire, a gate doing its job, a choice not to act). Only words in
# these tables are coloured; any other verdict stays neutral and the eval lists it (run_inner_sky_feed_eval.py),
# so a new outcome word is sorted on purpose rather than painted by guess.
VERDICT_SEVERITY: dict[str, str] = {
    "critical": "red",
    "broken": "red",
    "degraded": "amber",
    "claim_upheld=false": "amber",
    "refused": "amber",
    "unavailable": "amber",
    "miscalibrated": "amber",
    "unreliable": "amber",
    "abandoned": "amber",
    # intention_close (integration patch 7a): a promise Prime took on was not carried
    "not_produced": "amber",
}
# Verdicts that carry a reason after a fixed head (reading_turn): the head decides.
VERDICT_SEVERITY_PREFIX: dict[str, str] = {
    "no_read_evidence": "amber",   # reading found nothing to read (bare, or ":<reason>")
    "no producer close": "amber",  # reading never closed within its horizon
    "late:": "amber",              # reading finished, late ("late:<N>s stage1_completed")
}
VERDICT_NORMAL = frozenset({
    "not_crossed", "crossed", "claim_upheld=true", "granted", "stage1_completed", "decayed_unattended", "resolved",
    "acted", "dismissed", "asked", "produced", "reliable", "absent", "present", "recent", "unknown",
    # expectation_verdict: a prediction checked against what happened; being wrong is learning, not breakage
    "confirmed", "disconfirmed", "unscored",
    # dream_day: a quiet day left nothing to keep
    "no_lasting_trace",
    # personality_choice (integration patch 7a): Prime chose an option, or chose to wait; both are normal
    "chose", "declined_all",
})
def severity_of(verdict: Any) -> Optional[str]:
    v = str(verdict or "").strip()
    if not v:
        return None
    if v in VERDICT_SEVERITY:
        return VERDICT_SEVERITY[v]
    for head, sev in VERDICT_SEVERITY_PREFIX.items():
        if v.startswith(head):
            return sev
    return None


def verdict_is_classified(verdict: Any) -> bool:
    v = str(verdict or "").strip()
    return not v or v in VERDICT_NORMAL or severity_of(v) is not None
