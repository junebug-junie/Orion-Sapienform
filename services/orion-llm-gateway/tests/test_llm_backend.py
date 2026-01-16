import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.llm_backend import (  # noqa: E402
    _build_ollama_payload,
    _extract_text_from_ollama_response,
    _extract_vector_from_openai_response,
    _serialize_messages,
)
from app.models import ChatBody, ChatMessage  # noqa: E402


class TestLLMBackendHelpers(unittest.TestCase):
    def test_extract_vector_from_action_indices(self) -> None:
        data = {"action_indices": [[1, 2, 3]]}
        self.assertEqual(_extract_vector_from_openai_response(data), [1.0, 2.0, 3.0])

    def test_extract_text_from_ollama_message(self) -> None:
        data = {"message": {"role": "assistant", "content": "hello"}}
        self.assertEqual(_extract_text_from_ollama_response(data), "hello")

    def test_extract_text_from_ollama_generate(self) -> None:
        data = {"response": "hi"}
        self.assertEqual(_extract_text_from_ollama_response(data), "hi")

    def test_build_ollama_payload_maps_options(self) -> None:
        body = ChatBody(
            messages=[ChatMessage(role="user", content="ping")],
            options={"temperature": 0.2, "max_tokens": 128, "backend": "ollama"},
        )
        payload = _build_ollama_payload(body, model="llama3")
        self.assertEqual(payload["model"], "llama3")
        self.assertEqual(payload["messages"][0]["content"], "ping")
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertEqual(payload["options"]["num_predict"], 128)

    def test_serialize_messages_accepts_dicts(self) -> None:
        serialized = _serialize_messages([{"role": "user", "content": "hello"}])
        self.assertEqual(serialized, [{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
