from __future__ import annotations

import unittest
from pathlib import Path


class TestJournalComposePrompt(unittest.TestCase):
    def test_prompt_is_narrow_and_structured(self):
        prompt = (Path(__file__).resolve().parents[1] / "orion" / "cognition" / "prompts" / "journal_compose_prompt.j2").read_text()
        verb = (Path(__file__).resolve().parents[1] / "orion" / "cognition" / "verbs" / "journal.compose.yaml").read_text()

        self.assertIn("Return STRICT JSON only", prompt)
        self.assertIn("Do not choose tools.", prompt)
        self.assertIn("Do not plan actions.", prompt)
        self.assertIn("given this already-normalized journal trigger", prompt)
        self.assertIn("Scheduler daily entry:", prompt)
        self.assertIn("Novelty vs repetition:", prompt)
        self.assertIn("Automated digest-shaped journal:", prompt)
        self.assertIn("Treat operational side-effect claims", prompt)
        self.assertIn("Do not imply hidden automation unless the normalized trigger payload explicitly contains that signal.", prompt)
        self.assertIn("I marked this as something to revisit in my next reflection, but no automatic alert is implied.", prompt)
        self.assertIn("no tool selection or agentic planning", verb)


if __name__ == "__main__":
    unittest.main()


def test_walkway_rules_render_only_for_walkway_triggers():
    import jinja2

    src = (Path(__file__).resolve().parents[1] / "orion" / "cognition" / "prompts" / "journal_compose_prompt.j2").read_text()
    tmpl = jinja2.Template(src)

    def render(kind):
        return tmpl.render(
            metadata={"journal_trigger": {"trigger_kind": kind, "source_kind": "scheduler", "prompt_seed": "{}"},
                      "journal_mode": "digest"},
            raw_user_text="s", memory_digest="",
        )

    for kind in ("walkway_forecast", "walkway_grade"):
        text = render(kind)
        assert "Walkway journal:" in text
        assert "confirmed, disconfirmed, unscored" in text
        assert "enough_days: false" in text
    assert "Walkway journal:" not in render("metacog_digest")
