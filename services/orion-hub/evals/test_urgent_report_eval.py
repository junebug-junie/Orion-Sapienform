"""Pytest lane for `run_urgent_report_eval.py`: every urgent outcome ends in a
critical Hub + email notice that leads with the flag or verdict."""

from __future__ import annotations

import pytest

import run_urgent_report_eval as ev


@pytest.mark.parametrize("name,kind,title,problems", ev.run_cases(), ids=[case[0] for case in ev.CASES])
def test_urgent_report_case(name, kind, title, problems) -> None:
    assert problems == [], f"{name} ({kind}, {title!r})"


def test_checker_catches_prose_before_the_verdict() -> None:
    _, request = ev.CASES[0][1]()
    parts = request.body_text.split("\n\n")
    prose_first = [p for p in parts if p.startswith(ev.PROSE_MARK)] + [
        p for p in parts if not p.startswith(ev.PROSE_MARK)
    ]
    doctored = request.model_copy(update={"body_text": "\n\n".join(prose_first)})
    problems = ev.check_case(doctored, kind="final", want_kind="final", flag=None, bundle=False)
    assert "prose comes before the verdict" in problems
