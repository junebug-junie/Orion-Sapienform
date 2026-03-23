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
        self.assertIn("no tool selection or agentic planning", verb)


if __name__ == "__main__":
    unittest.main()
