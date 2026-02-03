from unittest.mock import MagicMock, patch
import pytest

from app.main import TopicRailService
from app.settings import settings

# We mock the heavy dependencies to avoid import errors or runtime overhead
@patch("app.main.TopicRailReader")
@patch("app.main.TopicRailWriter")
@patch("app.main.VectorHostEmbeddingProvider")
@patch("app.main.ModelStore")
@patch("app.main.TopicRailBusPublisher")
@patch("app.main.asyncio.run") # Mock asyncio.run to avoid event loop issues and verify calls
def test_service_publishes_on_assign(
    MockAsyncioRun, MockPublisher, MockModelStore, MockEmbedder, MockWriter, MockReader
):
    # Setup - temporarily override settings
    original_enabled = settings.topic_rail_bus_publish_enabled
    settings.topic_rail_bus_publish_enabled = True
    settings.topic_rail_bus_topic_assigned_channel = "test-assigned"

    try:
        # Mock publisher instance
        mock_publisher_instance = MagicMock()
        MockPublisher.return_value = mock_publisher_instance

        # Setup mocks to simulate success
        service = TopicRailService()

        service.model_store.exists.return_value = True
        service.reader.fetch_unassigned_rows.return_value = [{"id": 1, "prompt": "hello", "response": "world"}]
        service.embedder.embed_texts.return_value = [[0.1] * 768]
        service.model_store.load.return_value = (MagicMock(), MagicMock(), {})

        mock_topic_model = service.model_store.load.return_value[0]
        # Transform returns (topics, probs)
        mock_topic_model.transform.return_value = ([1], [[0.9]])

        service.writer.upsert_assignments.return_value = 1

        # Act
        service._assign_only()

        # Assert
        # Check that publisher was instantiated
        MockPublisher.assert_called()

        # Check that asyncio.run was called with the publisher coroutine
        # We can't easily check the coroutine argument value in a sync mock of asyncio.run,
        # but we can check that publish_assignment_batch was called to GENERATE the coroutine.
        mock_publisher_instance.publish_assignment_batch.assert_called_once()

        args = mock_publisher_instance.publish_assignment_batch.call_args
        channel, payload = args[0]
        assert channel == "test-assigned"
        assert payload["doc_count"] == 1
        assert payload["top_topic_ids"] == [1]

    finally:
        settings.topic_rail_bus_publish_enabled = original_enabled
