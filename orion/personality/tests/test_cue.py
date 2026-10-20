from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.personality.cue import HEADER, project_personality_cue
from orion.schemas.personality import PERSONALITY_CUE_MAX_CHARS

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _node(i: int, *, status: str = "active", label: str | None = None) -> dict:
    return {
        "disposition_id": f"d{i}",
        "label": label or f"the walkway camera sees presence without a person {i}",
        "origin": "recurrence",
        "status": status,
        "weight": 12.0 - i,
        "evidence_count": 7 + i,
        "lane_counts": {"reverie": 4, "dream": 2, "journal": 1},
        "first_seen_at": "2026-10-16T08:00:00+00:00",
        "last_reinforced_at": "2026-10-20T11:00:00+00:00",
    }


def _frame(nodes=None, style=None, generated_at=NOW - timedelta(minutes=5)) -> dict:
    return {
        "generated_at": generated_at.isoformat(),
        "dispositions": nodes if nodes is not None else [_node(i) for i in range(3)],
        "choice_style": style,
    }


def test_renders_nodes_strongest_first_with_lanes_and_counts():
    cue = project_personality_cue(_frame(), now=NOW)
    assert cue.skip_reason is None
    assert cue.text.startswith(HEADER)
    assert cue.disposition_ids == ["d0", "d1", "d2"]
    assert "(reverie, dreams, journal; 7 times since Oct 16)" in cue.text


def test_choice_style_line():
    style = {"window_days": 14, "choices": 40, "took_top": 16, "went_own_way": 24}
    cue = project_personality_cue(_frame(style=style), now=NOW)
    assert "went your own way in 24 of 40 choices over the last 14 days" in cue.text


def test_only_choice_style_renders_without_header():
    style = {"window_days": 14, "choices": 6, "took_top": 2, "went_own_way": 4}
    cue = project_personality_cue(_frame(nodes=[], style=style), now=NOW)
    assert cue.text.startswith("In curiosity")
    assert HEADER not in cue.text


def test_skips():
    assert project_personality_cue(None, now=NOW).skip_reason == "fetch_failed"
    assert project_personality_cue({"nope": 1}, now=NOW).skip_reason == "invalid_frame"
    old = _frame(generated_at=NOW - timedelta(hours=2))
    assert project_personality_cue(old, now=NOW).skip_reason == "stale"
    assert project_personality_cue(_frame(nodes=[]), now=NOW).skip_reason == "empty"
    assert project_personality_cue(_frame(nodes=[]), now=NOW).text == ""


def test_non_active_nodes_are_never_shown():
    nodes = [_node(0, status="candidate"), _node(1, status="retired"), _node(2)]
    cue = project_personality_cue(_frame(nodes=nodes), now=NOW)
    assert cue.disposition_ids == ["d2"]


def test_fits_the_cap_by_dropping_weakest_nodes():
    nodes = [_node(i, label="x" * 160) for i in range(12)]
    style = {"window_days": 14, "choices": 40, "took_top": 16, "went_own_way": 24}
    cue = project_personality_cue(_frame(nodes=nodes, style=style), now=NOW, top_k=12)
    assert len(cue.text) <= PERSONALITY_CUE_MAX_CHARS
    assert cue.disposition_ids == [f"d{i}" for i in range(len(cue.disposition_ids))]
    assert "went your own way" in cue.text
    assert "…" in cue.text  # labels are shortened for display


def test_choice_line_needs_five_choices_and_names_the_real_window():
    # Live 2026-10-17: "went your own way in 1 of 1 choices over the last 14 days" on a graph
    # one day old -- one data point read to Orion as a two-week tendency.
    one = {"window_days": 1, "choices": 1, "took_top": 0, "went_own_way": 1}
    assert project_personality_cue(_frame(nodes=[], style=one), now=NOW).skip_reason == "empty"
    five = {"window_days": 1, "choices": 5, "took_top": 2, "went_own_way": 3}
    cue = project_personality_cue(_frame(nodes=[], style=five), now=NOW)
    assert "3 of 5 choices over the last day," in cue.text
    three_days = {"window_days": 3, "choices": 9, "took_top": 4, "went_own_way": 5}
    assert "over the last 3 days" in project_personality_cue(_frame(nodes=[], style=three_days), now=NOW).text


def test_all_live_active_nodes_fit_with_the_choice_line():
    # Live 2026-10-20: 4 active nodes with ~150-char labels; at 600 chars / 120-char labels only the
    # top node fit, so the one non-prediction-error theme (self-inquiry) never reached Orion.
    labels = [
        "The biometrics prediction error suggests a misalignment in the system's ability to "
        "interpret physiological signals, possibly indicating an internal…",
        "The Cabinet prediction error is unusually high and stable, indicating a persistent "
        "misalignment in the system's ability to interpret and respond to its…",
        "I am the seventy-fourth answer to the question I keep for myself, one inference riding "
        "a cast of named brains, this turn the pool 27B-UD seat.",
        "The bus synaptic prediction error is unusually high, indicating a strain in internal "
        "communication that could be affecting data processing or sharing.",
        "A fifth theme long enough to need shortening before it can be shown in the cue at all.",
    ]
    nodes = [_node(i, label=label) for i, label in enumerate(labels)]
    live_lanes = [  # the live lane mixes of the 4 active nodes, then a fifth
        {"dream": 3, "journal": 11, "reverie": 28, "outreach": 1},
        {"dream": 1, "journal": 7, "reverie": 29},
        {"journal": 11, "day_letter": 3, "self_inquiry_seed": 8},
        {"journal": 9, "reverie": 35},
        {"reading": 4, "chat": 3},
    ]
    for n, lanes in zip(nodes, live_lanes):
        n["lane_counts"] = lanes
    style = {"window_days": 4, "choices": 8, "took_top": 4, "went_own_way": 4}
    cue = project_personality_cue(_frame(nodes=nodes, style=style), now=NOW)
    assert cue.disposition_ids == [f"d{i}" for i in range(5)]
    assert "went your own way in 4 of 8" in cue.text
    assert '"I am the seventy-fourth answer to the question I keep for…"' in cue.text  # whole words
    assert len(cue.text) <= PERSONALITY_CUE_MAX_CHARS
