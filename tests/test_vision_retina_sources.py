from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"))

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2

from app.sources import FolderFrameSource, MockFrameSource, create_frame_source


@pytest.mark.asyncio
async def test_folder_source_reads_jpg(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((10, 10, 3), dtype=np.uint8))
    src = FolderFrameSource(str(tmp_path))
    await src.start()
    result = await src.read()
    await src.stop()
    assert result is not None
    assert result.width == 10
    assert result.height == 10


@pytest.mark.asyncio
async def test_folder_source_empty_returns_none(tmp_path: Path) -> None:
    src = FolderFrameSource(str(tmp_path))
    await src.start()
    assert await src.read() is None
    await src.stop()


@pytest.mark.asyncio
async def test_mock_source_reads_from_folder(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "m.png"), np.ones((8, 8, 3), dtype=np.uint8) * 127)
    src = MockFrameSource(str(tmp_path))
    await src.start()
    result = await src.read()
    await src.stop()
    assert result is not None


def test_create_frame_source_factory() -> None:
    assert isinstance(create_frame_source("folder", "/x"), FolderFrameSource)
    assert isinstance(create_frame_source("mock", "/x"), MockFrameSource)


@pytest.mark.asyncio
async def test_folder_source_refreshes_after_new_file(tmp_path: Path) -> None:
    src = FolderFrameSource(str(tmp_path))
    await src.start()
    assert await src.read() is None

    cv2.imwrite(str(tmp_path / "late.jpg"), np.zeros((6, 6, 3), dtype=np.uint8))
    result = await src.read()
    await src.stop()
    assert result is not None
    assert result.width == 6
