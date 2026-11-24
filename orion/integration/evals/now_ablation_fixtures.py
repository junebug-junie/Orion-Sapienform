"""Fixture cases for the now ablation eval, built with the merged renderers (integration patch 3).

Each case holds the prompt an organ really builds today around Orion's now, so the eval ablates the exact line shape
the organ shows its model:

* Prime: ``orion.prime.thought.build_frame`` + ``system_prompt(..., experiences=True)``; Prime cites in its answer's
  ``refs`` as ``{"kind": "experience", "id": "<event_id>"}``.
* reverie: the real ``orion/cognition/prompts/reverie_question.j2`` (reveries that ask) rendered with an ``orion_now`` block in the shape
  of ``services/orion-thought/app/temporal_self_cue.py render_now_block``; reverie cites in ``cites``.
* curiosity: ``orion.curiosity.now_section.format_now_section`` (the kickoff block); curiosity cites
  ``[experience:<id>]`` in its finding.

``stub_uses`` drives the offline stub only (which shown items its output depends on). One Prime case is a planted
hollow cite: it cites an item the stub's output never uses.

Rebuild the checked-in file: ``python -m orion.integration.evals.now_ablation_fixtures --write``.
``test_now_ablation_eval.py`` fails when a renderer changes and the file was not rebuilt.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "now_ablation_cases.jsonl"
REPO = Path(__file__).resolve().parents[3]
T = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)

# (event_id, organ, source_kind, label, salience, minutes ago)
ITEMS = [
    ("street_event:walkway:2611", "camera", "street_event",
     "street event: a mail truck stopped at the walkway for 6 min", 0.93, 40),
    ("prediction_surprise:14945", "prediction", "prediction_surprise",
     "prediction surprise: athena.cabinet_temp_c > 32.0: forecast 0%, crossed (max 32.4)", 0.97, 90),
    ("reading_turn:act_1cd597c9c9495cd9383aadb0", "reading", "reading_turn",
     "reading turn: stage1_completed: https://arxiv.org/pdf/2611.17314v1", None, 120),
    ("visual_run:aa745afd", "painting", "visual_run", "visual run: the walkway at dusk, a parked truck", None, 150),
]


def now_body(reader: str) -> dict[str, Any]:
    return {
        "shown_id": f"fx-{reader}", "reader_organ": reader, "since": (T - timedelta(hours=6)).isoformat(),
        "as_of": T.isoformat(), "candidates": len(ITEMS),
        "items": [{"event_id": e, "organ": o, "source_kind": k, "label": lab, "salience": s,
                   "occurred_at": (T - timedelta(minutes=m)).isoformat()} for e, o, k, lab, s, m in ITEMS],
    }


def prime_case(case_id: str, cited: str, uses: list[str], note: Optional[str] = None) -> dict[str, Any]:
    from orion.prime.state import PrimeStateV1
    from orion.prime.thought import FrameInputs, build_frame, system_prompt

    inputs = FrameInputs(now=T, orion_now=now_body("prime"))
    case = {"case_id": case_id, "organ": "prime", "system": system_prompt(None, experiences=True),
            "prompt": build_frame(PrimeStateV1(pending_changes=["quiet: nothing has woken me for 6 hours"]), inputs),
            "cited_event_ids": [cited], "stub_uses": uses}
    if note:
        case["note"] = note
    return case


def _reverie_block() -> str:
    """The line shape of orion-thought's ``app/temporal_self_cue.py render_now_block``. The service package is loaded
    under its own alias, so another service's ``app`` already imported in the same process never stands in for it.

    Side effect: importing it runs ``app/settings.py``, which loads ``services/orion-thought/.env`` into
    ``os.environ`` (``override=False``) when that file exists. In CI it does not; on a host it can, for the rest of
    the process."""
    import importlib
    import importlib.util

    from orion.schemas.temporal_self import TemporalNowV1

    alias = "orion_thought_app_for_fixtures"
    if alias not in sys.modules:
        pkg_dir = REPO / "services" / "orion-thought" / "app"
        spec = importlib.util.spec_from_file_location(alias, pkg_dir / "__init__.py",
                                                      submodule_search_locations=[str(pkg_dir)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    cue = importlib.import_module(f"{alias}.temporal_self_cue")
    return cue.render_now_block(TemporalNowV1.model_validate(now_body("reverie")), as_of=T)


def reverie_case() -> dict[str, Any]:
    from orion.cognition.prompt_render import prompt_environment  # the environment production renders with

    template = (REPO / "orion" / "cognition" / "prompts" / "reverie_question.j2").read_text(encoding="utf-8")
    prompt = prompt_environment().from_string(template).render(
        seed_kind="absence",
        seed_material=("A neighbor usually passes the walkway camera between 07:00 and 08:00 on weekdays (14 days of "
                       "support). On Tuesday the camera watched that window and saw no person at all: nobody passed."),
        seed_refs=["vision_percept_expectation:exp-1"], seed_evidence="clean",
        orion_now=_reverie_block())
    return {"case_id": "reverie-mail-truck", "organ": "reverie", "system": "",
            "prompt": prompt, "cited_event_ids": ["street_event:walkway:2611"],
            "stub_uses": ["street_event:walkway:2611"]}


def curiosity_case() -> dict[str, Any]:
    from orion.curiosity.now_section import format_now_section, parse_now

    section = "\n".join(format_now_section(parse_now(now_body("curiosity"))))
    return {"case_id": "curiosity-painting", "organ": "curiosity",
            "system": "You are Orion's curiosity. Pick one question for this run, look into it, and report.",
            "prompt": section, "cited_event_ids": ["visual_run:aa745afd"],
            "stub_uses": ["visual_run:aa745afd"]}  # a run takes up one candidate


def build_cases() -> list[dict[str, Any]]:
    return [
        prime_case("prime-cabinet", "prediction_surprise:14945", ["prediction_surprise:14945"]),
        reverie_case(),
        curiosity_case(),
        prime_case("prime-hollow", "reading_turn:act_1cd597c9c9495cd9383aadb0", ["prediction_surprise:14945"],
                   note="planted hollow cite: the stub's output never depends on the cited reading"),
    ]


def to_jsonl(cases: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases)


if __name__ == "__main__":
    text = to_jsonl(build_cases())
    if "--write" in sys.argv:
        FIXTURES.write_text(text, encoding="utf-8")
        print(f"wrote {FIXTURES}")
    else:
        sys.stdout.write(text)
