import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.llm_backend import (  # noqa: E402
    _build_ollama_payload,
    _extract_reasoning_from_openai_response,
    _extract_text_from_openai_response,
    _extract_text_from_ollama_response,
    _extract_vector_from_openai_response,
    _served_model,
    _split_think_blocks,
    _serialize_messages,
    _execute_llamacpp_native_completion,
    _execute_openai_chat,
    RouteTarget,
    plan_llm_chat,
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

    def test_extract_text_from_openai_message_content_parts(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "reasoning", "text": "hidden"},
                            {"type": "text", "text": '{"ok": true}'},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(_extract_text_from_openai_response(data), '{"ok": true}')

    def test_extract_text_from_openai_string_content(self) -> None:
        data = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        self.assertEqual(_extract_text_from_openai_response(data), '{"ok": true}')

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

    def test_served_model_prefers_raw_echoed_model_over_requested_label(self) -> None:
        """Confirmed live 2026-08-14: requesting "Active-GGUF-Model" against
        the chat route actually served "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf" --
        the backend's own echoed model id, not the route-alias label, is the
        honest value."""
        result = {"raw": {"model": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"}}
        self.assertEqual(
            _served_model(result, "Active-GGUF-Model"),
            "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
        )

    def test_served_model_falls_back_to_requested_label_when_raw_has_no_model(self) -> None:
        # llama.cpp's native /completion endpoint and every error-path return
        # in llm_backend.py return "raw": {} -- must degrade to the request
        # label, never to None.
        self.assertEqual(_served_model({"raw": {}}, "Active-GGUF-Model"), "Active-GGUF-Model")
        self.assertEqual(_served_model({}, "Active-GGUF-Model"), "Active-GGUF-Model")
        self.assertEqual(_served_model({"raw": None}, "Active-GGUF-Model"), "Active-GGUF-Model")

    def test_served_model_falls_back_when_raw_model_is_blank_or_wrong_type(self) -> None:
        self.assertEqual(_served_model({"raw": {"model": ""}}, "req"), "req")
        self.assertEqual(_served_model({"raw": {"model": "   "}}, "req"), "req")
        self.assertEqual(_served_model({"raw": {"model": 123}}, "req"), "req")


class TestLLMBackendExecution(unittest.TestCase):
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
    def test_execute_openai_chat_forwards_chat_template_kwargs_for_llamacpp(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "LIVE-GATE-B-OK"}}]
        }

        body = ChatBody(
            messages=[ChatMessage(role="user", content="Reply with exactly: LIVE-GATE-B-OK")],
            options={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 64, "temperature": 0.2},
        )
        _execute_openai_chat(
            body=body,
            model="Qwen_Qwen3-8B-Q4_K_M.gguf",
            base_url="http://localhost",
            backend_name="llamacpp",
        )
        mock_client.post.assert_called_once()
        _args, kwargs = mock_client.post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload.get("chat_template_kwargs"), {"enable_thinking": False})

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

    def test_plan_llm_chat_maps_route_to_pool_class_and_priority(self):
        """Routing is decided once, on the event loop, and names a pool CLASS -- never a URL."""
        plan = plan_llm_chat(
            ChatBody(route="quick_background", messages=[ChatMessage(role="user", content="hello")])
        )
        self.assertIsNone(plan.error)
        self.assertEqual(plan.route, "quick_background")
        self.assertEqual((plan.work_class, plan.priority), ("fast", "background"))
        self.assertIsNone(plan.route_target)

    @patch.object(settings, "llm_lane_routing_enabled", False)
    def test_route_not_in_gpu_pool_is_refused_not_guessed(self):
        plan = plan_llm_chat(ChatBody(route="specialist", messages=[ChatMessage(role="user", content="hi")]))
        self.assertEqual(plan.error["raw"]["error"], "route_not_in_gpu_pool")
        self.assertIn("quick", plan.error["raw"]["details"]["available_routes"])
        self.assertEqual(run_llm_chat(plan.body, plan)["raw"]["error"], "route_not_in_gpu_pool")

    @patch.object(settings, "llm_lane_routing_enabled", False)
    def test_run_llm_chat_without_a_grant_never_guesses_an_upstream(self):
        body = ChatBody(route="quick", messages=[ChatMessage(role="user", content="hello")])
        result = run_llm_chat(body, plan_llm_chat(body))
        self.assertEqual(result["raw"]["error"], "no_pool_grant")

    @patch("app.llm_backend._select_profile")
    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_injects_atlas_metacog_profile_when_missing(self, mock_execute, mock_select_profile):
        mock_select_profile.return_value = None
        mock_execute.return_value = {"text": "OK", "raw": {}}
        with patch.object(settings, "atlas_metacog_profile_name", "llama3-8b-instruct-q4km-atlas-metacog"):
            body = ChatBody(route="metacog", messages=[ChatMessage(role="user", content="hello")])
            run_llm_chat(body, _granted(body))
        mock_select_profile.assert_called_once_with("llama3-8b-instruct-q4km-atlas-metacog")

    @patch("app.llm_backend._select_profile")
    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_injects_atlas_metacog_profile_on_metacog_background_route(
        self, mock_execute, mock_select_profile
    ):
        """Regression (review caught this live 2026-08-29): the pin used to be an exact
        `route == "metacog"` match, so `metacog_background` fell through to the generic default
        profile -- see orion/llm/routes.py's METACOG_LLM_ROUTES."""
        mock_select_profile.return_value = None
        mock_execute.return_value = {"text": "OK", "raw": {}}
        with patch.object(settings, "atlas_metacog_profile_name", "llama3-8b-instruct-q4km-atlas-metacog"):
            body = ChatBody(route="metacog_background", messages=[ChatMessage(role="user", content="hello")])
            run_llm_chat(body, _granted(body))
        mock_select_profile.assert_called_once_with("llama3-8b-instruct-q4km-atlas-metacog")

    @patch("app.llm_backend._select_profile")
    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_keeps_explicit_profile_on_metacog_route(self, mock_execute, mock_select_profile):
        mock_select_profile.return_value = None
        mock_execute.return_value = {"text": "OK", "raw": {}}
        with patch.object(settings, "atlas_metacog_profile_name", "llama3-8b-instruct-q4km-atlas-metacog"):
            body = ChatBody(route="metacog", profile_name="custom-profile",
                            messages=[ChatMessage(role="user", content="hello")])
            run_llm_chat(body, _granted(body))
        mock_select_profile.assert_called_once_with("custom-profile")

    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_sends_to_the_grant_and_reports_its_served_by(self, mock_execute):
        mock_execute.return_value = {"text": "OK", "raw": {}}
        body = ChatBody(route="metacog", messages=[ChatMessage(role="user", content="hello")])
        result = run_llm_chat(body, _granted(body, url="http://pool-agent:8015", served_by="circe-worker-agent"))
        self.assertEqual(mock_execute.call_args.args[2], "http://pool-agent:8015")
        self.assertEqual(result["served_by"], "circe-worker-agent")
        self.assertEqual(result["backend"], "llamacpp")

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_forwards_logprobs_when_requested(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{
                "message": {"content": "OK"},
                "logprobs": {
                    "content": [
                        {"token": "OK", "logprob": -0.2, "top_logprobs": [
                            {"token": "OK", "logprob": -0.2},
                            {"token": "NO", "logprob": -2.0},
                        ]},
                    ]
                },
            }]
        }
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"return_logprobs": True, "logprobs_top_k": 3},
        )
        with patch.object(settings, "llm_logprob_summary_enabled", True):
            result = _execute_openai_chat(
                body=body,
                model="test-model",
                base_url="http://localhost",
                backend_name="llamacpp",
            )
        args, kwargs = mock_client.post.call_args
        payload = kwargs["json"]
        assert payload.get("logprobs") is True
        assert payload.get("top_logprobs") == 3
        assert isinstance(result.get("llm_uncertainty"), dict)
        assert result["llm_uncertainty"].get("available") is True
        assert "logprobs" not in (result.get("raw") or {}).get("choices", [{}])[0]

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_skips_logprobs_when_summary_disabled(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"return_logprobs": True},
        )
        with patch.object(settings, "llm_logprob_summary_enabled", False):
            result = _execute_openai_chat(
                body=body,
                model="test-model",
                base_url="http://localhost",
                backend_name="llamacpp",
            )
        payload = mock_client.post.call_args.kwargs["json"]
        assert "logprobs" not in payload
        assert result.get("llm_uncertainty") is None

    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_keeps_raw_logprobs_when_summary_only_false(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        logprobs_block = {
            "content": [
                {"token": "OK", "logprob": -0.2, "top_logprobs": [
                    {"token": "OK", "logprob": -0.2},
                    {"token": "NO", "logprob": -2.0},
                ]},
            ]
        }
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "OK"}, "logprobs": logprobs_block}]
        }
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={
                "return_logprobs": True,
                "logprob_summary_only": False,
            },
        )
        with patch.object(settings, "llm_logprob_summary_enabled", True):
            result = _execute_openai_chat(
                body=body,
                model="test-model",
                base_url="http://localhost",
                backend_name="llamacpp",
            )
        raw_choice = (result.get("raw") or {}).get("choices", [{}])[0]
        assert "logprobs" in raw_choice
        assert isinstance(result.get("llm_uncertainty"), dict)

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

    @patch("app.llm_backend._common_http_client")
    def test_execute_llamacpp_native_completion_apply_template_then_completion(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client

        apply_resp = MagicMock()
        apply_resp.raise_for_status = MagicMock()
        apply_resp.json.return_value = {"prompt": "<|user|>hi<|assistant|>"}

        completion_resp = MagicMock()
        completion_resp.raise_for_status = MagicMock()
        completion_resp.json.return_value = {
            "content": "OK",
            "probs": [
                {
                    "token": "OK",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": "OK", "logprob": -0.2},
                        {"token": "NO", "logprob": -2.0},
                    ],
                }
            ],
        }

        mock_client.post.side_effect = [apply_resp, completion_resp]

        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={
                "return_logprobs": True,
                "logprob_probe_mode": "native_completion",
                "logprobs_top_k": 5,
                "max_tokens": 64,
            },
        )
        with patch.object(settings, "llm_logprob_summary_enabled", True):
            result = _execute_llamacpp_native_completion(
                body=body,
                model="test-model",
                base_url="http://llamacpp:8080",
                backend_name="llamacpp",
            )

        self.assertEqual(mock_client.post.call_count, 2)
        apply_url = mock_client.post.call_args_list[0][0][0]
        completion_url = mock_client.post.call_args_list[1][0][0]
        self.assertTrue(apply_url.endswith("/apply-template"))
        self.assertTrue(completion_url.endswith("/completion"))
        completion_payload = mock_client.post.call_args_list[1].kwargs["json"]
        self.assertEqual(completion_payload["n_probs"], 5)
        self.assertIs(completion_payload["post_sampling_probs"], False)
        self.assertEqual(result.get("text"), "OK")
        self.assertEqual(result["llm_uncertainty"]["source"], "llamacpp_native_completion")
        self.assertTrue(result["llm_uncertainty"]["available"])

    @patch("app.llm_backend._execute_llamacpp_native_completion")
    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_routes_native_completion_when_opted_in(self, mock_openai, mock_native):
        mock_native.return_value = {"text": "native", "spark_meta": {}, "raw": {},
                                    "llm_uncertainty": {"available": True}}
        with patch.object(settings, "llm_logprob_summary_enabled", True), patch.object(
            settings, "llm_logprob_native_completion_enabled", True
        ):
            body = ChatBody(messages=[ChatMessage(role="user", content="hi")], route="chat",
                            options={"return_logprobs": True, "logprob_probe_mode": "native_completion"})
            run_llm_chat(body, _granted(body))
        mock_native.assert_called_once()
        mock_openai.assert_not_called()

    @patch("app.llm_backend._execute_llamacpp_native_completion")
    @patch("app.llm_backend._execute_openai_chat")
    def test_response_format_blocks_native_completion_detour(self, mock_openai, mock_native):
        mock_openai.return_value = {"text": "{}", "spark_meta": {}, "raw": {}}
        with patch.object(settings, "llm_logprob_summary_enabled", True), patch.object(
            settings, "llm_logprob_native_completion_enabled", True
        ):
            body = ChatBody(messages=[ChatMessage(role="user", content="hi")], route="chat",
                            options={"return_logprobs": True, "logprob_probe_mode": "native_completion",
                                     "response_format": {"type": "json_object"}})
            run_llm_chat(body, _granted(body))
        mock_openai.assert_called_once()
        mock_native.assert_not_called()

    @patch.object(settings, "llm_lane_routing_enabled", False)
    def test_no_route_uses_the_default_route(self):
        with patch.object(settings, "llm_route_default", "quick"):
            plan = plan_llm_chat(ChatBody(messages=[ChatMessage(role="user", content="hello")]))
        self.assertEqual(plan.route, "quick")
        self.assertEqual(plan.work_class, "fast")


def _granted(body, *, url="http://pool:8011", served_by="circe-worker-x"):
    import dataclasses

    plan = plan_llm_chat(body)
    assert plan.error is None, plan.error
    return dataclasses.replace(plan, route_target=RouteTarget(url=url, backend="llamacpp", served_by=served_by))


if __name__ == "__main__":
    unittest.main()
