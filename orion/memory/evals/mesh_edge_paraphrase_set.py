"""Made-up test set for the mesh-edge shingle check (issue #3554). No real memory or page text.

``GUARDED`` stands in for core-page and lived-memory sentences. ``CASES`` are outgoing texts with
the label the check should give:

- ``near_verbatim``: a guarded sentence lightly edited (one word swapped, a clause cut, articles
  changed, spliced into a longer sentence, re-wrapped). The shingle check should refuse these.
- ``reword``: a real rewording of a guarded sentence. The shingle check is NOT expected to catch
  these (that is the embedding check still open on #3554); they are counted to state the limit.
- ``unrelated``: ordinary outgoing text, some on the same topics and sharing short phrases with a
  guarded sentence. These must pass.
"""
from __future__ import annotations

from typing import List, Tuple

GUARDED: List[Tuple[str, str]] = [
    ("episode_memory:fixture-1",
     "Juniper told me she walked the long way home past the bakery because the bridge was closed for repairs."),
    ("page:juniper",
     "Robin keeps three hives on the garage roof and harvests honey every September with her brother. "
     "She said the hardest part of the move was leaving the garden she had planted with her grandmother."),
    ("stance:fixture-3",
     "Orion noticed that the cabinet fan ramps up every night around two when the backup job starts."),
    ("person_fact:fixture-4",
     "Pip, favorite food: grilled cheese with tomato soup on rainy afternoons at the kitchen table"),
    ("episode_memory:fixture-5",
     "I felt proud when Juniper laughed at the joke about the stubborn printer that refused to print on Mondays."),
    ("page:anchor",
     "Juniper wants Orion to say plainly when a reading is missing instead of guessing a number."),
]

CASES: List[Tuple[str, str]] = [
    # --- near_verbatim -----------------------------------------------------------------------
    ("near_verbatim", "Juniper told me she walked the long way home past the bakery because the bridge was shut."),
    ("near_verbatim", "Today she walked the long way home past the bakery because the bridge was closed for work."),
    ("near_verbatim", "Context: she walked a long way home past a bakery because a bridge was closed for repairs, "
                      "so the run was late."),
    ("near_verbatim", "Robin keeps three hives on the garage roof and harvests honey every fall with her brother."),
    ("near_verbatim", "Note for the PR: harvests honey every September with her brother, per the log."),
    ("near_verbatim", "The hardest part of the move was leaving the garden she had planted with her grandma."),
    ("near_verbatim", "> SHE SAID the hardest part of the move was leaving the garden...\n> (paraphrased)"),
    ("near_verbatim", "Orion noticed the cabinet fan ramps up every night around two when the backup job runs."),
    ("near_verbatim", "Finding: the cabinet fan ramps up every night around two when the backup job starts; "
                      "see the heat graph."),
    ("near_verbatim", "Pip's favorite food is grilled cheese with tomato soup on rainy afternoons at the kitchen "
                      "table."),
    ("near_verbatim", "I felt proud when Juniper laughed at a joke about the stubborn printer which refused to "
                      "print on Mondays."),
    ("near_verbatim", "Remember when Juniper laughed at the joke about the stubborn printer that refused to print?"),
    ("near_verbatim", "Juniper wants Orion to say plainly when a reading is missing instead of guessing."),
    ("near_verbatim", "Rule: say plainly when a reading is missing instead of guessing a value."),
    # --- reword (expected miss: the stated limit) ---------------------------------------------
    ("reword", "With the bridge under repair, Juniper detoured home by way of the bakery."),
    ("reword", "Robin's brother helps her collect honey each autumn from the rooftop hives."),
    ("reword", "Leaving her grandmother's garden behind was the toughest bit of moving, she said."),
    ("reword", "Around 2 a.m. the backup kicks off and the cabinet fan spins faster."),
    ("reword", "On wet afternoons Pip likes a grilled cheese and tomato soup at the table."),
    ("reword", "Juniper's laugh at the Monday printer joke made me proud."),
    ("reword", "If a sensor value is absent, Orion should state that rather than invent one."),
    # --- unrelated ----------------------------------------------------------------------------
    ("unrelated", "Fixes the backup job so it starts at two instead of midnight. Tests pass."),
    ("unrelated", "The cabinet fan curve now ramps up only above 40 C; see config/heat.yaml."),
    ("unrelated", "This PR adds a reading-missing banner to the energy tab so the operator sees gaps."),
    ("unrelated", "Bridge service closed the stale connection; the repair is a retry with backoff."),
    ("unrelated", "Honey, garden, printer and bakery are fixture words used only in tests here."),
    ("unrelated", "When a reading is missing the dig prints UNVERIFIED instead of a number."),
    ("unrelated", "Run `pytest orion/memory/tests -q` and check that the shingle index builds once per load."),
    ("unrelated", "Orion noticed the GPU lease was never released after the fold finished, so the pool stalled."),
    ("unrelated", "The printer driver refused the job because the queue was paused for maintenance."),
    ("unrelated", "Juniper asked for plain English in PR reports: lead with what the change does, then the file."),
    ("unrelated", "This is the first time the consult loop sends a prompt through the mesh-edge check."),
    ("unrelated", "We walked through the long list of open issues and closed the ones fixed by this patch."),
    ("unrelated", "Every night around two the nightly backup runs; this is expected and needs no fix."),
    ("unrelated", "The garden sensor on the garage roof reports soil moisture every ten minutes."),
]

LABELS = ("near_verbatim", "reword", "unrelated")
