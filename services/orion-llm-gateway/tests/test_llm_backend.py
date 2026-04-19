import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.llm_backend import (  # noqa: E402
    _build_ollama_payload,
    _extract_reasoning_from_openai_response,
    _extract_text_from_ollama_response,
    _extract_vector_from_openai_response,
    _split_think_blocks,
    _serialize_messages,
    _execute_openai_chat,
    _load_route_targets,
    run_llm_chat,
)
from app.models import ChatBody, ChatMessage  # noqa: E402
from app.settings import settings  # noqa: E402


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

    def test_split_think_blocks_strips_closed_blocks_and_keeps_visible_text(self) -> None:
        visible, reasoning = _split_think_blocks("<think>draft reasoning</think>Hi")
        self.assertEqual(visible, "Hi")
        self.assertEqual(reasoning, "draft reasoning")

    def test_split_think_blocks_strips_unclosed_block_without_leaking_reasoning(self) -> None:
        visible, reasoning = _split_think_blocks("Hi\n<think>long reasoning without close")
        self.assertEqual(visible, "Hi")
        self.assertEqual(reasoning, "long reasoning without close")

    def test_extract_reasoning_supports_reasoning_aliases(self) -> None:
        data = {"choices": [{"message": {"content": "answer", "reasoning_text": "step trace"}}]}
        self.assertEqual(_extract_reasoning_from_openai_response(data), "step trace")

    def test_extract_reasoning_supports_content_reasoning_parts(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "output_text", "text": "answer"},
                            {"type": "reasoning", "text": "hidden rationale"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(_extract_reasoning_from_openai_response(data), "hidden rationale")


class TestLLMBackendExecution(unittest.TestCase):
    def tearDown(self) -> None:
        _load_route_targets.cache_clear()

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_passes_response_format_for_llamacpp(self, mock_client_factory):
        # Setup mock
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }

        # Setup input
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"response_format": {"type": "json_object"}}
        )

        # Execute
        _execute_openai_chat(
            body=body,
            model="test-model",
            base_url="http://localhost",
            backend_name="llamacpp"
        )

        # Verify
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        payload = kwargs["json"]

        self.assertIn("response_format", payload, "response_format should be present for llamacpp")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_passes_response_format_for_llama_cola(self, mock_client_factory):
        # Setup mock
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }

        # Setup input
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"response_format": {"type": "json_object"}}
        )

        # Execute
        _execute_openai_chat(
            body=body,
            model="test-model",
            base_url="http://localhost",
            backend_name="llama-cola"
        )

        # Verify
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        payload = kwargs["json"]

        self.assertIn("response_format", payload, "response_format should be present for llama-cola")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_passes_response_format_for_vllm(self, mock_client_factory):
        # Setup mock
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }

        # Setup input
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"response_format": {"type": "json_object"}}
        )

        # Execute
        _execute_openai_chat(
            body=body,
            model="test-model",
            base_url="http://localhost",
            backend_name="vllm"
        )

        # Verify
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        payload = kwargs["json"]

        self.assertIn("response_format", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_route_table_accepts_agent_route_in_merged_mode(self):
        original = settings.llm_route_table_json
        try:
            settings.llm_route_table_json = (
                '{"chat":{"url":"http://atlas:8011","served_by":"atlas-worker-1","backend":"llamacpp"},'
                '"agent":{"url":"http://atlas:8011","served_by":"atlas-worker-1","backend":"llamacpp"},'
                '"metacog":{"url":"http://atlas:8012","served_by":"atlas-worker-2","backend":"llamacpp"},'
                '"quick":{"url":"http://atlas:8013","served_by":"atlas-worker-fast-1","backend":"llamacpp"}}'
            )
            _load_route_targets.cache_clear()
            targets = _load_route_targets()
            self.assertEqual(targets["chat"].served_by, "atlas-worker-1")
            self.assertIn("agent", targets)
            self.assertEqual(targets["agent"].served_by, "atlas-worker-1")
            self.assertEqual(targets["metacog"].served_by, "atlas-worker-2")
            self.assertEqual(targets["quick"].served_by, "atlas-worker-fast-1")
        finally:
            settings.llm_route_table_json = original
            _load_route_targets.cache_clear()

    def test_route_table_accepts_agent_route_in_split_mode(self):
        original = settings.llm_route_table_json
        try:
            settings.llm_route_table_json = (
                '{"chat":{"url":"http://atlas:8011","served_by":"atlas-worker-1","backend":"llamacpp"},'
                '"agent":{"url":"http://atlas:8014","served_by":"atlas-worker-agent-1","backend":"llamacpp"},'
                '"metacog":{"url":"http://atlas:8012","served_by":"atlas-worker-2","backend":"llamacpp"},'
                '"quick":{"url":"http://atlas:8013","served_by":"atlas-worker-fast-1","backend":"llamacpp"}}'
            )
            _load_route_targets.cache_clear()
            targets = _load_route_targets()
            self.assertIn("agent", targets)
            self.assertEqual(targets["agent"].served_by, "atlas-worker-agent-1")
            self.assertEqual(targets["quick"].served_by, "atlas-worker-fast-1")
        finally:
            settings.llm_route_table_json = original
            _load_route_targets.cache_clear()

    def test_route_table_metacog_served_by_defaults_from_atlas_service_name(self):
        original_table = settings.llm_route_table_json
        original_service = settings.atlas_metacog_service_name
        try:
            settings.atlas_metacog_service_name = "atlas-worker-2"
            settings.llm_route_table_json = (
                '{"metacog":{"url":"http://atlas:8012","backend":"llamacpp"}}'
            )
            _load_route_targets.cache_clear()
            targets = _load_route_targets()
            self.assertEqual(targets["metacog"].served_by, "atlas-worker-2")
        finally:
            settings.llm_route_table_json = original_table
            settings.atlas_metacog_service_name = original_service
            _load_route_targets.cache_clear()

    @patch("app.llm_backend._select_profile")
    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_injects_atlas_metacog_profile_when_missing(
        self,
        mock_execute,
        mock_select_profile,
    ):
        original_table = settings.llm_route_table_json
        original_profile = settings.atlas_metacog_profile_name
        try:
            settings.atlas_metacog_profile_name = "llama3-8b-instruct-q4km-atlas-metacog"
            settings.llm_route_table_json = (
                '{"metacog":{"url":"http://atlas:8012","served_by":"atlas-worker-2","backend":"llamacpp"}}'
            )
            mock_select_profile.return_value = None
            mock_execute.return_value = {"text": "OK", "raw": {}}
            _load_route_targets.cache_clear()
            run_llm_chat(
                ChatBody(
                    route="metacog",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )
            mock_select_profile.assert_called_once_with("llama3-8b-instruct-q4km-atlas-metacog")
        finally:
            settings.llm_route_table_json = original_table
            settings.atlas_metacog_profile_name = original_profile
            _load_route_targets.cache_clear()

    @patch("app.llm_backend._select_profile")
    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_keeps_explicit_profile_on_metacog_route(
        self,
        mock_execute,
        mock_select_profile,
    ):
        original_table = settings.llm_route_table_json
        original_profile = settings.atlas_metacog_profile_name
        try:
            settings.atlas_metacog_profile_name = "llama3-8b-instruct-q4km-atlas-metacog"
            settings.llm_route_table_json = (
                '{"metacog":{"url":"http://atlas:8012","served_by":"atlas-worker-2","backend":"llamacpp"}}'
            )
            mock_select_profile.return_value = None
            mock_execute.return_value = {"text": "OK", "raw": {}}
            _load_route_targets.cache_clear()
            run_llm_chat(
                ChatBody(
                    route="metacog",
                    profile_name="custom-profile",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )
            mock_select_profile.assert_called_once_with("custom-profile")
        finally:
            settings.llm_route_table_json = original_table
            settings.atlas_metacog_profile_name = original_profile
            _load_route_targets.cache_clear()

    def test_missing_route_fails_closed_when_route_table_active(self):
        original = settings.llm_route_table_json
        try:
            settings.llm_route_table_json = '{"chat":{"url":"http://atlas:8011","served_by":"atlas-worker-1","backend":"llamacpp"}}'
            _load_route_targets.cache_clear()
            result = run_llm_chat(
                ChatBody(
                    route="agent",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )
            self.assertEqual(result["raw"]["error"], "route_not_configured")
            self.assertEqual(result["route"], "agent")
        finally:
            settings.llm_route_table_json = original
            _load_route_targets.cache_clear()

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_does_not_promote_inline_think_to_structured_reasoning(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "<think>scratchpad</think>Visible answer"}}]
        }
        body = ChatBody(messages=[ChatMessage(role="user", content="hi")], trace_id="corr-inline-think-001")
        result = _execute_openai_chat(
            body=body,
            model="test-model",
            base_url="http://localhost",
            backend_name="llamacpp",
        )
        self.assertEqual(result.get("text"), "Visible answer")
        self.assertIsNone(result.get("reasoning_content"))
        self.assertEqual(result.get("inline_think_content"), "scratchpad")

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_extracts_close_tag_only_inline_think(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "scratchpad only</think>Visible answer"}}]
        }
        body = ChatBody(messages=[ChatMessage(role="user", content="hi")], trace_id="corr-close-tag-001")
        result = _execute_openai_chat(
            body=body,
            model="test-model",
            base_url="http://localhost",
            backend_name="llamacpp",
        )
        self.assertEqual(result.get("text"), "Visible answer")
        self.assertIsNone(result.get("reasoning_content"))
        self.assertEqual(result.get("inline_think_content"), "scratchpad only")

    @patch("app.llm_backend._execute_openai_chat")
    def test_no_route_still_uses_default_chat_when_route_table_active(self, mock_execute):
        original = settings.llm_route_table_json
        original_default = settings.llm_route_default
        try:
            settings.llm_route_default = "chat"
            settings.llm_route_table_json = (
                '{"chat":{"url":"http://atlas:8011","served_by":"atlas-worker-1","backend":"llamacpp"}}'
            )
            mock_execute.return_value = {"text": "OK", "raw": {}}
            _load_route_targets.cache_clear()
            result = run_llm_chat(
                ChatBody(
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )
            self.assertEqual(result["route"], "chat")
            self.assertEqual(result["served_by"], "atlas-worker-1")
        finally:
            settings.llm_route_table_json = original
            settings.llm_route_default = original_default
            _load_route_targets.cache_clear()


if __name__ == "__main__":
    unittest.main()
