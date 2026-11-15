"""Eval: do tendency labels say what a choice was ABOUT? Replayed on live 2026-11-14 option texts.

Live 11-14 the Personality tab showed labels like "a thread from my letter over a promise I made and a thread
from my letter, 2 of 2": Prime's concern texts lead with where they came from, and the label is the first six
words of each side. Since 2026-11-15 Prime's choice record leads with the subject (orion/prime/choice.py
``option_text``) and the label maker drops a leading category phrase (orion/personality/graph.py ``gist``).

The fixture holds the real concerns from Redis ``orion:prime:state`` (texts, event refs, promise sentences),
the three real Prime decisions among them (personality_choice, lane prime), live outreach option texts, and the
category-led labels stored on personality_disposition. For each it prints the label before and after. Checks:

* prime_subject_first -- no label built from a Prime decision leads with a category phrase on any side;
* no_refs_in_text     -- no Prime option text carries markdown or a trace ref;
* outreach_titles     -- an outreach read's label starts with the title, a daydream's with what it showed;
* stored_tidy         -- stored labels whose subject survived are tidied now; category-only ones are flagged
                         so the reducer replaces them on their next choice (``reinforce``).

Run: ``PYTHONPATH=. python orion/prime/evals/run_choice_label_replay.py`` (exit 1 on any failed check).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from orion.personality.graph import category_led, label_base, pair_label
from orion.prime.choice import ShownConcern, option_text

FIXTURE = Path(__file__).parent / "fixtures" / "choice_labels_live_2026-11-14.json"


def main() -> int:
    data = json.loads(FIXTURE.read_text())
    concerns = {c["concern_id"]: ShownConcern(c["concern_id"], c["text"], c.get("event_ref"),
                                              promise_text=c.get("promise_text")) for c in data["concerns"]}
    checks: dict[str, bool] = {}

    print("Prime decisions (label from the concern text with today's label maker -> from the new record text):")
    ok_subject, ok_refs = True, True
    for chosen, declined in data["decisions"]:
        before = pair_label(concerns[chosen].text, [concerns[d].text for d in declined])
        texts = [option_text(concerns[chosen]), *(option_text(concerns[d]) for d in declined)]
        after = pair_label(texts[0], texts[1:])
        sides = after.replace(" and ", " over ").split(" over ")
        ok_subject &= not any(category_led(s) for s in sides)
        ok_refs &= not any(m in t for t in texts for m in ("**", "`", "_trace:", "[curiosity:", "[juniper_chat:"))
        print(f"  concern text: {before}\n  record text:  {after}")
        for t in texts:
            print(f"    text: {t[:160]}")
    checks["prime_subject_first"] = ok_subject
    checks["no_refs_in_text"] = ok_refs

    print("Outreach options:")
    reads = [t for t in data["outreach_options"] if t.startswith("A read")]
    dreams = [t for t in data["outreach_options"] if t.startswith("A daydream")]
    label = pair_label(reads[0], [dreams[0]])
    print(f"  {label}")
    title = reads[0].split('"')[1]
    checks["outreach_titles"] = (label.startswith(" ".join(title.split()[:4]))
                                 and " over a room with a white wall" in label)

    print("Stored labels (now -> tidied):")
    ok_tidy = True
    for stored in data["stored_labels"]:
        base = label_base(stored)
        stale = category_led(base)
        print(f"  {stored}\n    -> {base}{'   [category only: replaced on next choice]' if stale else ''}")
        if stored.startswith(("A read", "A daydream")):
            ok_tidy &= not stale
    checks["stored_tidy"] = ok_tidy

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
