import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.executor import _clean_raw_llm_content

class TestExecutorCleaning(unittest.TestCase):
    def test_clean_raw_json(self):
        text = '{"foo": "bar"}'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_clean_markdown_json_block(self):
        text = '```json\n{"foo": "bar"}\n```'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_clean_markdown_plain_block(self):
        text = '```\n{"foo": "bar"}\n```'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_clean_preamble(self):
        text = 'Here is the JSON object:\n{"foo": "bar"}'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_clean_preamble_and_markdown(self):
        text = 'Here is the JSON:\n```json\n{"foo": "bar"}\n```'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_clean_preamble_case_insensitive(self):
        text = 'here IS the json:\n{"foo": "bar"}'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_surrounding_whitespace(self):
        text = '  {"foo": "bar"}  '
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_partial_preamble(self):
        # Should not strip if it doesn't match the pattern fully
        text = 'Here is something else:\n{}'
        self.assertEqual(_clean_raw_llm_content(text), 'Here is something else:\n{}')

    def test_clean_single_line_block(self):
        text = '```json {"foo": "bar"}```'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

    def test_clean_single_line_no_lang(self):
        text = '```{"foo": "bar"}```'
        self.assertEqual(_clean_raw_llm_content(text), '{"foo": "bar"}')

if __name__ == "__main__":
    unittest.main()
