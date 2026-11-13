from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.personality.cue import (
    HEADER,
    LABEL_SHOWN_CHARS,
    NAMED_LABEL_MIN_CHARS,
    _label,
    project_personality_cue,
)
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


def test_label_shortening_edges():
    assert _label("short label") == "short label"
    assert _label("x" * 200) == "x" * (LABEL_SHOWN_CHARS - 1) + "…"  # no space: hard cut
    assert _label("a " + "b" * 100) == "a " + "b" * (LABEL_SHOWN_CHARS - 3) + "…"  # keeps the theme
    long = "word… " * 20
    assert not _label(long).endswith("……")
    assert not _label("Hello world. " * 10).endswith(".…")


def _named(i: int, name: str | None, **kw) -> dict:
    n = _node(i, **kw)
    n["name"] = name
    if name:
        n["name_sentence"] = f"I named it {name} because it keeps returning."
    return n


def test_named_node_keeps_the_excerpt_beside_the_name():
    cue = project_personality_cue(_frame(nodes=[_named(0, "the watched gap")]), now=NOW)
    assert ('- you call this "the watched gap": "the walkway camera sees presence without a person 0" '
            "(reverie, dreams, journal; 7 times since Oct 16)") in cue.text
    assert cue.text.startswith(HEADER)
    assert "I named it" not in cue.text  # the sentence is for the panel, not the cue


def test_unnamed_node_renders_exactly_as_before():
    plain = project_personality_cue(_frame(nodes=[_node(0)]), now=NOW).text
    null_name = project_personality_cue(_frame(nodes=[_named(0, None)]), now=NOW).text
    blank = project_personality_cue(_frame(nodes=[{**_node(0), "name": "   "}]), now=NOW).text
    assert plain == null_name == blank
    assert '\n- "the walkway camera sees presence without a person 0" (reverie' in plain
    assert "you call this" not in plain


def test_mixed_named_and_unnamed_keep_order():
    nodes = [_named(0, "first"), _node(1), _named(2, "third")]
    lines = project_personality_cue(_frame(nodes=nodes), now=NOW).text.split("\n")[1:]
    assert lines[0].startswith('- you call this "first": "the walkway')
    assert lines[1].startswith('- "the walkway')
    assert lines[2].startswith('- you call this "third": "the walkway')


def test_name_whitespace_is_collapsed_to_one_line():
    cue = project_personality_cue(_frame(nodes=[_named(0, "two\nlines  here")]), now=NOW)
    assert 'you call this "two lines here":' in cue.text
    assert len(cue.text.split("\n")) == 2


def test_five_named_nodes_with_max_length_names_fit_or_drop_weakest():
    # NAMING_NAME_MAX_CHARS is 40, adding ~58 chars per line. The cap holds; named excerpts
    # shrink first (never vanish), then nodes drop weakest-first exactly as unnamed nodes do.
    labels = [("word " * 30)[:148] + str(i) for i in range(5)]
    nodes = [_named(i, ("n%d " % i + "y" * 40)[:40], label=labels[i]) for i in range(5)]
    style = {"window_days": 4, "choices": 8, "took_top": 4, "went_own_way": 4}
    cue = project_personality_cue(_frame(nodes=nodes, style=style), now=NOW)
    assert cue.skip_reason is None
    assert len(cue.text) <= PERSONALITY_CUE_MAX_CHARS
    assert cue.disposition_ids == [f"d{i}" for i in range(len(cue.disposition_ids))]
    assert len(cue.disposition_ids) == 4  # 5 unnamed fit; 40-char names cost the weakest node
    assert "went your own way in 4 of 8" in cue.text
    for line in cue.text.split("\n")[1:-1]:
        assert line.startswith('- you call this "n') and '": "word word' in line and "…" in line
        excerpt = line.split('": "', 1)[1].split('"', 1)[0]
        assert len(excerpt) <= NAMED_LABEL_MIN_CHARS
    unnamed = project_personality_cue(
        _frame(nodes=[_node(i, label=labels[i]) for i in range(5)], style=style), now=NOW)
    assert len(unnamed.disposition_ids) == 5


def test_live_four_nodes_still_all_reach_orion_with_max_length_names():
    labels = [
        "The biometrics prediction error suggests a misalignment in the system's ability to "
        "interpret physiological signals, possibly indicating an internal…",
        "The Cabinet prediction error is unusually high and stable, indicating a persistent "
        "misalignment in the system's ability to interpret and respond to its…",
        "I am the seventy-fourth answer to the question I keep for myself, one inference riding "
        "a cast of named brains, this turn the pool 27B-UD seat.",
        "The bus synaptic prediction error is unusually high, indicating a strain in internal "
        "communication that could be affecting data processing or sharing.",
    ]
    style = {"window_days": 4, "choices": 8, "took_top": 4, "went_own_way": 4}
    nodes = [_named(i, "z" * 40, label=label) for i, label in enumerate(labels)]
    cue = project_personality_cue(_frame(nodes=nodes, style=style), now=NOW)
    assert cue.disposition_ids == ["d0", "d1", "d2", "d3"]
    assert '"I am the seventy-fourth answer to…"' in cue.text
    assert len(cue.text) <= PERSONALITY_CUE_MAX_CHARS
    # Short names do not shrink anything: the full 64-char excerpt stays.
    short = [_named(i, "gap", label=label) for i, label in enumerate(labels)]
    text = project_personality_cue(_frame(nodes=short, style=style), now=NOW).text
    assert '"I am the seventy-fourth answer to the question I keep for…"' in text


def test_unnamed_nodes_never_get_the_shorter_named_excerpt():
    labels = [("word " * 30)[:148] + str(i) for i in range(5)]
    nodes = [_named(0, "y" * 40, label=labels[0])] + [_node(i, label=labels[i]) for i in range(1, 5)]
    cue = project_personality_cue(_frame(nodes=nodes), now=NOW, top_k=5)
    for line in cue.text.split("\n")[1:]:
        if not line.startswith("- you call this"):
            assert line.startswith('- "' + _label(labels[0])[:40])


def test_frame_from_older_durable_runs_without_name_fields_still_renders():
    cue = project_personality_cue(_frame(), now=NOW)  # _node carries no name keys at all
    assert cue.skip_reason is None and "you call this" not in cue.text


def test_double_quote_in_a_name_does_not_break_the_quoted_framing():
    cue = project_personality_cue(_frame(nodes=[_named(0, 'the "gap"')]), now=NOW)
    assert "you call this \"the 'gap'\": \"the walkway" in cue.text
