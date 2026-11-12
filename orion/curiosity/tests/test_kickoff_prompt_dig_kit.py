"""The investigation prompt points at the dig kit: recent PRs first, then `dig help`."""
import re

from orion.curiosity.kickoff_prompt import DIG_HELP_FIRST, DIG_RECENT_WORK_LINES, _access_section
from orion.dig.__main__ import HELP


def _section(**over) -> str:
    kwargs = dict(own_graph="orion_worldview", atlas_graph="orion_substrate", hub_url="http://hub")
    kwargs.update(over)
    return "\n".join(_access_section(**kwargs))


def test_access_section_carries_the_dig_kit_lines_by_default() -> None:
    text = _section()
    for line in DIG_RECENT_WORK_LINES:
        assert line in text
    assert "python -P -m orion.dig help" in DIG_HELP_FIRST
    assert "python -m orion.dig" not in text, "always -P: plain -m runs the sandbox copy"


def test_dig_kit_false_leaves_it_out() -> None:
    text = _section(dig_kit=False)
    assert "orion.dig" not in text


def test_every_named_dig_command_exists_in_dig_help() -> None:
    named = set(re.findall(r"orion\.dig ([a-z][a-z-]*)", "\n".join(DIG_RECENT_WORK_LINES)))
    assert named == {"field-notes", "history", "help"}
    for cmd in named - {"help"}:
        assert re.search(rf"^  {re.escape(cmd)} ", HELP, re.M), cmd
