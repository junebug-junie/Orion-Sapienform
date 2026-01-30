import pytest

from app.topic_rail.doc_builders.chat_history import ChatHistoryDocBuilder


def test_doc_builder_prompt_mode():
    builder = ChatHistoryDocBuilder(doc_mode="prompt", max_chars=50)
    row = {"prompt": "Hello", "response": "World"}
    assert builder.build(row) == "Hello"


def test_doc_builder_response_mode():
    builder = ChatHistoryDocBuilder(doc_mode="response", max_chars=50)
    row = {"prompt": "Hello", "response": "World"}
    assert builder.build(row) == "World"


def test_doc_builder_prompt_response_mode():
    builder = ChatHistoryDocBuilder(doc_mode="prompt+response", max_chars=50)
    row = {"prompt": "Hello", "response": "World"}
    assert builder.build(row) == "User: Hello\nAssistant: World"


def test_doc_builder_truncation():
    builder = ChatHistoryDocBuilder(doc_mode="prompt", max_chars=5)
    row = {"prompt": "Hello world", "response": ""}
    assert builder.build(row) == "Hello"
