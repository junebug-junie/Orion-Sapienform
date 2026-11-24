"""#3375: a self-inquiry run's test of a self-prior it already holds must count as tested.

Live 2026-11-12..24: 45 self-inquiry runs, 1 with n_tested > 0. Diffing each run's start snapshot
against the next run's showed self-inquiry runs DID test existing self-priors (times_tested went up,
confidence moved) but left `last_run_id` unset, so `diff_snapshots` filed every test as
unattributed. The self-inquiry prompt listed the self-priors but never showed the one testing
command that stamps `last_run_id`; the investigation kickoff did.
"""
from datetime import datetime, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt, prior_testing_lines
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.value import KIND_TESTED, PriorState, diff_snapshots
from orion.curiosity.worldview import WorldviewSnapshot

RUN = "abcd12ef3456"
STAMP = f'p.last_run_id = "{RUN}"'
COUNT = (
    "p.times_tested = coalesce(p.times_tested, 0) + CASE WHEN "
    f'coalesce(p.last_run_id, "") <> "{RUN}" THEN 1 ELSE 0 END,'
)


def _self_inquiry(**over) -> str:
    kwargs = dict(view=WorldviewSnapshot(), run_id=RUN, graph_enabled=True, repo_root="/repo")
    kwargs.update(over)
    return build_self_inquiry_prompt(**kwargs)


def test_writable_self_inquiry_prompt_carries_the_stamped_testing_command() -> None:
    text = _self_inquiry()
    assert "TESTING ONE you already hold" in text
    assert COUNT in text
    assert STAMP in text
    assert f'MERGE (v:PriorRevision {{prior_id: "...", run_id: "{RUN}",' in text
    assert "Inconclusive is a real answer: run the counting command" in text
    assert "<RUN_ID>" not in text


def test_testing_command_follows_the_self_prior_shape() -> None:
    text = _self_inquiry()
    assert text.index("A SELF-PRIOR -- a claim about yourself") < text.index("TESTING ONE you already hold")


def test_read_only_self_inquiry_prompt_does_not_offer_a_write() -> None:
    assert "TESTING ONE you already hold" not in _self_inquiry(graph_enabled=False)
    assert "TESTING ONE you already hold" not in _self_inquiry(run_id="")


def test_kickoff_and_self_inquiry_show_the_same_testing_command() -> None:
    kickoff = build_kickoff_prompt(
        StudyMaterial(generated_at=datetime(2026, 11, 24, tzinfo=timezone.utc)), run_id=RUN, view=WorldviewSnapshot()
    )
    for line in prior_testing_lines():
        rendered = line.replace("<RUN_ID>", RUN)
        assert rendered in kickoff
        assert rendered in _self_inquiry()


def test_a_test_written_with_the_command_is_counted_and_one_without_the_stamp_is_not() -> None:
    """Pins the scorer behaviour the prompt fix relies on (unchanged by #3375): only a stamped
    test counts. The prompt tests above are the regression guard."""
    before = {
        "self:a": PriorState("self:a", 0.6, 1, "open", last_run_id="older", run_id="older"),
        "self:b": PriorState("self:b", 0.6, 0, "open", run_id="older"),
    }
    after = {
        # What the prompt's command writes: count + confidence + last_run_id in one SET.
        "self:a": PriorState("self:a", 0.75, 2, "revised", last_run_id=RUN, run_id="older"),
        # What self-inquiry runs wrote before #3375: count and confidence, no stamp.
        "self:b": PriorState("self:b", 0.75, 1, "revised", run_id="older"),
    }
    out = diff_snapshots(before, after, run_id=RUN)
    assert out.n_tested == 1
    assert out.n_unattributed == 1
    assert [p.kind for p in out.per_prior] == [KIND_TESTED]
    assert out.realized_nats is not None and out.realized_nats > 0
