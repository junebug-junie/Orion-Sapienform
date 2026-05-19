"""Tests for structured_output helpers and gateway wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.structured_output import (  # noqa: E402
    METHOD_JSON_OBJECT_ONLY,
    METHOD_JSON_OBJECT_SCHEMA,
    METHOD_JSON_SCHEMA_SCHEMA,
    METHOD_NONE,
    build_response_format,
    resolve_structured_output_method,
    response_format_shape_label,
)
from app.llm_backend import _execute_openai_chat  # noqa: E402
from app.models import ChatBody, ChatMessage  # noqa: E402


class TestStructuredOutputHelpers(unittest.TestCase):
    def test_json_object_only_shape(self) -> None:
        rf = build_response_format(METHOD_JSON_OBJECT_ONLY, {"type": "object"})
        self.assertEqual(rf, {"type": "json_object"})

    def test_json_object_schema_shape(self) -> None:
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        rf = build_response_format(METHOD_JSON_OBJECT_SCHEMA, schema)
        self.assertEqual(rf, {"type": "json_object", "schema": schema})

    def test_json_schema_schema_shape(self) -> None:
        schema = {"type": "object"}
        rf = build_response_format(METHOD_JSON_SCHEMA_SCHEMA, schema)
        self.assertEqual(rf, {"type": "json_schema", "schema": schema})

    def test_unknown_method_returns_none(self) -> None:
        self.assertIsNone(build_response_format("not_a_method", {}))

    def test_resolve_auto_falls_back_to_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                resolve_structured_output_method({"structured_output_method": "auto"}),
                METHOD_NONE,
            )

    def test_structured_schema_wins_over_raw_response_format(self) -> None:
        from app.structured_output import apply_structured_output_to_payload

        opts = {
            "structured_output_schema": {"type": "object"},
            "structured_output_method": METHOD_JSON_OBJECT_SCHEMA,
            "response_format": {"type": "json_object"},
        }
        apply_structured_output_to_payload(opts, backend_name="llamacpp")
        self.assertEqual(opts["response_format"]["schema"], {"type": "object"})

    def test_disabled_for_artifact_injects_thinking_off(self) -> None:
        from app.structured_output import apply_structured_output_to_payload

        opts = {
            "structured_output_schema": {"type": "object"},
            "structured_output_method": METHOD_JSON_OBJECT_SCHEMA,
            "structured_output_thinking_policy": "disabled_for_artifact",
        }
        _, diag = apply_structured_output_to_payload(opts, backend_name="llamacpp")
        self.assertEqual(opts["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(diag.get("thinking_policy"), "disabled_for_artifact")

    def test_shape_labels(self) -> None:
        self.assertEqual(
            response_format_shape_label({"type": "json_object", "schema": {}}),
            "json_object_schema",
        )
        self.assertEqual(
            response_format_shape_label(
                {
                    "type": "json_schema",
                    "json_schema": {"name": "X", "strict": True, "schema": {}},
                }
            ),
            "openai_json_schema_wrapper",
        )


class TestStructuredOutputGateway(unittest.TestCase):
    @patch("app.llm_backend._common_http_client")
    def test_llamacpp_json_object_schema_in_payload(self, mock_client_factory: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "{}"}}]
        }
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={
                "structured_output_schema": schema,
                "structured_output_method": "json_object_schema",
                "structured_output_schema_name": "SuggestDraftV1",
            },
        )
        result = _execute_openai_chat(
            body=body,
            model="m",
            base_url="http://localhost",
            backend_name="llamacpp",
        )
        payload = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"]["type"], "json_object")
        self.assertEqual(payload["response_format"]["schema"], schema)
        diag = result.get("structured_output_diagnostics") or {}
        self.assertTrue(diag.get("structured_output_requested"))

    @patch("app.llm_backend._common_http_client")
    def test_llamacpp_json_schema_schema_in_payload(self, mock_client_factory: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "{}"}}]
        }
        schema = {"type": "object"}
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={
                "structured_output_schema": schema,
                "structured_output_method": "json_schema_schema",
            },
        )
        _execute_openai_chat(
            body=body,
            model="m",
            base_url="http://localhost",
            backend_name="llamacpp",
        )
        payload = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["schema"], schema)

    @patch("app.llm_backend._common_http_client")
    def test_no_structured_schema_unchanged(self, mock_client_factory: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"temperature": 0.5},
        )
        _execute_openai_chat(
            body=body,
            model="m",
            base_url="http://localhost",
            backend_name="llamacpp",
        )
        payload = mock_client.post.call_args.kwargs["json"]
        self.assertNotIn("response_format", payload)
